/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include "kernel.h"
#include "support/common.h"
#include "support/partitioner.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

/*extern "C" {
void m5_work_begin(int workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}*/

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_gpu_threads;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    const char *comparison_file;
    int         display = 0;

    Params(int argc, char **argv) {
        device          = 0;
        n_gpu_threads   = 16;
        n_threads       = 4;
				n_warmup        = 1;
				n_reps          = 10;
        alpha           = 0.2;
        file_name       = "input/peppa/";
        comparison_file = "output/peppa/";
        int opt;
        while((opt = getopt(argc, argv, "hd:i:t:w:r:a:f:c")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_gpu_threads   = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'a': alpha           = atof(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
            assert((n_gpu_threads > 0 || n_threads > 0) && "Invalid # of host + device workers!");
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./cedd [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=16)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=10)"
                "\n    -r <R>    # of timed repetition iterations (default=100)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    folder containing input video files (default=input/peppa/)"
                "\n    -c <C>    folder containing comparison files (default=output/peppa/)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char** all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {

    for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

        char FileName[100];
        sprintf(FileName, "%s%d.txt", p.file_name, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

        fscanf(fp, "%d\n", &rowsc);
        fscanf(fp, "%d\n", &colsc);

        in_size = rowsc * colsc * sizeof(unsigned char);
        all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
        }
        fclose(fp);
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    Params      p(argc, argv);
    hipError_t  cudaStatus;

    // Initialize (part 1)
    const int n_frames = p.n_warmup + p.n_reps;
    unsigned char **all_gray_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);

    // Allocate buffers
    const int CPU_PROXY = 0;
    const int GPU_PROXY = 1;
    unsigned char *    h_in_out[2];
    h_in_out[CPU_PROXY] = (unsigned char *)malloc(in_size);
    h_in_out[GPU_PROXY] = (unsigned char *)malloc(in_size);
    unsigned char *d_in_out = h_in_out[GPU_PROXY];
    unsigned char *h_interm_cpu_proxy = (unsigned char *)malloc(in_size);
    unsigned char *h_theta_cpu_proxy  = (unsigned char *)malloc(in_size);
    unsigned char *d_interm_gpu_proxy;
    cudaStatus = hipMalloc((void**)&d_interm_gpu_proxy, in_size);
    unsigned char *d_theta_gpu_proxy;
    cudaStatus = hipMalloc((void**)&d_theta_gpu_proxy, in_size);
    std::atomic<int> next_frame;
    hipDeviceSynchronize();
    ALLOC_ERR(h_in_out[CPU_PROXY], h_interm_cpu_proxy, h_theta_cpu_proxy);
    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };

    // Initialize (part 2)
    unsigned char **all_out_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
    for(int i = 0; i < n_frames; i++) {
        all_out_frames[i] = (unsigned char *)malloc(in_size);
    }
    std::atomic_int *worklist    = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    ALLOC_ERR(worklist);
    if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
        worklist[0].store(0);
    }
    next_frame.store(0);

    CoarseGrainPartitioner partitioner = partitioner_create(n_frames, p.alpha, worklist);
    std::vector<std::thread> proxy_threads;
    fprintf(stderr, "AM: Entering loop\n"); 
    for(int proxy_tid = 0; proxy_tid < 2; proxy_tid++) {
        proxy_threads.push_back(std::thread([&, proxy_tid]() {

            if(proxy_tid == GPU_PROXY) {

                for(int task_id = gpu_first(&partitioner); gpu_more(&partitioner); task_id = gpu_next(&partitioner)) {

                    // Next frame
                    memcpy(h_in_out[proxy_tid], all_gray_frames[task_id], in_size);

                    //m5_work_begin(0, 0);

                    // GAUSSIAN KERNEL
                    // Kernel launch
                    fprintf(stderr, "AM: Launching GPU kernels\n"); 
                    cudaStatus = call_gaussian_kernel(p.n_gpu_threads, d_in_out, d_interm_gpu_proxy,
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));
                    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };

                    // SOBEL KERNEL
                    // Kernel launch
                    fprintf(stderr, "AM: Launching GPU kernels\n"); 
                    cudaStatus = call_sobel_kernel(p.n_gpu_threads, d_interm_gpu_proxy, d_in_out, d_theta_gpu_proxy, 
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));
                    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };

                    // NON-MAXIMUM SUPPRESSION KERNEL
                    // Kernel launch
                    fprintf(stderr, "AM: Launching GPU kernels\n"); 
                    cudaStatus = call_non_max_supp_kernel(p.n_gpu_threads, d_in_out, d_interm_gpu_proxy, d_theta_gpu_proxy, 
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));
                    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };

                    // HYSTERESIS KERNEL
                    // Kernel launch
                    fprintf(stderr, "AM: Launching GPU kernels\n"); 
                    cudaStatus = call_hyst_kernel(p.n_gpu_threads, d_interm_gpu_proxy, d_in_out, 
                        rowsc, colsc);
                    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };

                    fprintf(stderr, "AM: Synchronising\n"); 
                    hipDeviceSynchronize();
                    //m5_work_end(0, 0);

                    memcpy(all_out_frames[task_id], h_in_out[proxy_tid], in_size);
                    
                }

            } else if(proxy_tid == CPU_PROXY) {

                for(int task_id = cpu_first(&partitioner); cpu_more(&partitioner); task_id = cpu_next(&partitioner)) {

                    // Next frame
                    fprintf(stderr, "AM: A strnage memcpy\n"); 
                    memcpy(h_in_out[proxy_tid], all_gray_frames[task_id], in_size);

                    // Launch CPU threads
                    fprintf(stderr, "AM: Launching CPU threads\n");
                    std::thread main_thread(run_cpu_threads, h_in_out[proxy_tid], h_interm_cpu_proxy, h_theta_cpu_proxy,
                        rowsc, colsc, p.n_threads, task_id);
                    main_thread.join();

                    memcpy(all_out_frames[task_id], h_in_out[proxy_tid], in_size);

                }

            }

        }));
    }
    std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });

    fprintf(stderr, "AM: System level barrier\n");
    hipDeviceSynchronize();

    // Verify answer
    verify(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);

    // Release buffers
    free(h_in_out[GPU_PROXY]);
    free(h_in_out[CPU_PROXY]);
    free(h_interm_cpu_proxy);
    free(h_theta_cpu_proxy);
    for(int i = 0; i < n_frames; i++) {
        free(all_gray_frames[i]);
    }
    free(all_gray_frames);
    for(int i = 0; i < n_frames; i++) {
        free(all_out_frames[i]);
    }
    free(all_out_frames);
    cudaStatus = hipFree(d_interm_gpu_proxy);
    cudaStatus = hipFree(d_theta_gpu_proxy);
    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };
    free(worklist);

    printf("Test Passed\n");
    return 0;
}
