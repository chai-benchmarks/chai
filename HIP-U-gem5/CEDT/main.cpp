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
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <atomic>
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
    const char *file_name;
    const char *comparison_file;
    int         display = 0;

    Params(int argc, char **argv) {
        device          = 0;
        n_gpu_threads   = 16;
        n_threads       = 4;
				n_warmup        = 1;
				n_reps          = 10;
        file_name       = "input/peppa/";
        comparison_file = "output/peppa/";
        int opt;
        while((opt = getopt(argc, argv, "hd:i:t:w:r:f:c")) >= 0) {
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
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./cedt [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=16)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=10)"
                "\n    -r <R>    # of timed repetition iterations (default=100)"
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

    // Initialize (part 1)
    unsigned char **all_gray_frames = (unsigned char **)malloc((p.n_warmup + p.n_reps) * sizeof(unsigned char *));
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);

    // Allocate buffers
    const int CPU_PROXY = 0;
    const int GPU_PROXY = 1;
    unsigned char **   h_in_out;
    h_in_out = (unsigned char **)malloc((p.n_warmup + p.n_reps)*sizeof(unsigned char *));
    ALLOC_ERR(h_in_out);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        h_in_out[i] = (unsigned char *)malloc(in_size);
        ALLOC_ERR(h_in_out[i]);
    }
    unsigned char * h_interm = (unsigned char *)malloc(in_size);
    ALLOC_ERR(h_interm);
    unsigned char * d_interm;
    hipError_t cudaStatus = hipMalloc((void**)&d_interm, in_size);
    unsigned char **   h_theta;
    h_theta = (unsigned char **)malloc((p.n_warmup + p.n_reps)*sizeof(unsigned char *));
    ALLOC_ERR(h_theta);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        h_theta[i] = (unsigned char *)malloc(in_size);
        ALLOC_ERR(h_theta[i]);
    }
    std::atomic<int> sobel_ready[p.n_warmup + p.n_reps];
    hipDeviceSynchronize();
    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };;

    // Initialize (part 2)
    unsigned char **all_out_frames = (unsigned char **)malloc((p.n_warmup + p.n_reps) * sizeof(unsigned char *));
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        all_out_frames[i] = (unsigned char *)malloc(in_size);
    }
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        sobel_ready[i].store(0);
    }

    std::vector<std::thread> proxy_threads;
    for(int proxy_tid = 0; proxy_tid < 2; proxy_tid++) {
        proxy_threads.push_back(std::thread([&, proxy_tid]() {

            for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

                if(proxy_tid == GPU_PROXY) {

                    // Next frame
                    if(all_gray_frames[rep] == NULL) {
                        (&sobel_ready[rep])->store(-1);
                        continue;
                    }
                    memcpy(h_in_out[rep], all_gray_frames[rep], in_size);

                    //m5_work_begin(0, 0);

                    // GAUSSIAN KERNEL
                    // Kernel launch
                    fprintf(stderr, "AM: Launching GPU kernels\n"); 
                    hipError_t cudaStatus = call_gaussian_kernel(p.n_gpu_threads, h_in_out[rep], d_interm,
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));
                    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };;

                    // SOBEL KERNEL
                    // Kernel launch
                    fprintf(stderr, "AM: Launching GPU kernels\n");
                    cudaStatus = call_sobel_kernel(p.n_gpu_threads, d_interm, h_in_out[rep], h_theta[rep], 
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));

                    fprintf(stderr, "AM: Syncing\n");
                    hipDeviceSynchronize();
                    fprintf(stderr, "AM: Done syncing\n");
                    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };;

                    //m5_work_end(0, 0);

                    // Release CPU proxy
                    (&sobel_ready[rep])->store(1);

                } else if(proxy_tid == CPU_PROXY) {

                    // Wait for GPU proxy
                    while((&sobel_ready[rep])->load() == 0) {
                    }
                    if((&sobel_ready[rep])->load() == -1)
                        continue;
                    fprintf(stderr, "AM: Launching CPU threads\n");
                    std::thread main_thread(
                        run_cpu_threads, h_in_out[rep], h_interm, h_theta[rep], rowsc, colsc, p.n_threads, rep);

                    fprintf(stderr, "AM: CPU thread joining\n");
                    main_thread.join();
                    fprintf(stderr, "AM: CPU thread joined\n");

                    memcpy(all_out_frames[rep], h_in_out[rep], in_size);
                    fprintf(stderr, "AM: memcpying\n");
                }
            }
        }));
    }
    fprintf(stderr, "AM: tying threads\n");
    std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });

    fprintf(stderr, "AM: System level barrier\n");
    hipDeviceSynchronize();
    fprintf(stderr, "AM: System level barrier crossed!\n");

    // Verify answer
    verify(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);
    fprintf(stderr, "AM: Verified\n");

    // Release buffers
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        hipError_t cudaStatus = hipFree(h_in_out[i]);
    }
    cudaStatus = hipFree(h_in_out);
    free(h_interm);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        hipError_t cudaStatus = hipFree(h_theta[i]);
    }
    cudaStatus = hipFree(h_theta);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        free(all_gray_frames[i]);
    }
    free(all_gray_frames);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        free(all_out_frames[i]);
    }
    free(all_out_frames);
    cudaStatus = hipFree(d_interm);
    if(cudaStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(cudaStatus), __FILE__, __LINE__); exit(-1); };;

    printf("Test Passed\n");
    return 0;
}
