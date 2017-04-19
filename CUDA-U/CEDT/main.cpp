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

#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <atomic>
#include <assert.h>

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
        n_warmup        = 10;
        n_reps          = 100;
        file_name       = "input/peppa/";
        comparison_file = "output/peppa/";
        int opt;
        while((opt = getopt(argc, argv, "hd:i:t:w:r:f:c:x")) >= 0) {
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
            case 'x': display         = 1; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_threads > 0 && "Invalid # of host threads!");
#ifndef CHAI_OPENCV
        assert(display != 1 && "Compile with CHAI_OPENCV");
#endif
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
                "\n    -x        display output video (with CHAI_OPENCV)"
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
    CUDASetup    setcuda(p.device);
    Timer        timer;
    cudaError_t  cudaStatus;

    // Initialize (part 1)
    timer.start("Initialization");
    const int max_gpu_threads = setcuda.max_gpu_threads();
    unsigned char **all_gray_frames = (unsigned char **)malloc((p.n_warmup + p.n_reps) * sizeof(unsigned char *));
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);
    timer.stop("Initialization");

    // Allocate buffers
    timer.start("Allocation");
    const int CPU_PROXY = 0;
    const int GPU_PROXY = 1;
    unsigned char **   h_in_out;
    cudaStatus = cudaMallocManaged(&h_in_out, (p.n_warmup + p.n_reps)*sizeof(unsigned char *));
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        cudaStatus = cudaMallocManaged(&h_in_out[i], in_size);
    }
    unsigned char * h_interm = (unsigned char *)malloc(in_size);
    ALLOC_ERR(h_interm);
    unsigned char * d_interm;
    cudaStatus = cudaMalloc((void**)&d_interm, in_size);
    unsigned char **   h_theta;
    cudaStatus = cudaMallocManaged(&h_theta, (p.n_warmup + p.n_reps)*sizeof(unsigned char *));
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        cudaStatus = cudaMallocManaged(&h_theta[i], in_size);
    }
    std::atomic<int> sobel_ready[p.n_warmup + p.n_reps];
    cudaDeviceSynchronize();
    CUDA_ERR();
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize (part 2)
    timer.start("Initialization");
    unsigned char **all_out_frames = (unsigned char **)malloc((p.n_warmup + p.n_reps) * sizeof(unsigned char *));
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        all_out_frames[i] = (unsigned char *)malloc(in_size);
    }
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        sobel_ready[i].store(0);
    }
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    timer.start("Total Proxies");
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

                    timer.start("GPU Proxy: Kernel");

                    // GAUSSIAN KERNEL
                    assert(p.n_gpu_threads * p.n_gpu_threads <= max_gpu_threads && 
                        "The thread block size is greater than the maximum thread block size that can be used to execute gaussian kernel");
                    // Kernel launch
                    cudaStatus = call_gaussian_kernel(p.n_gpu_threads, h_in_out[rep], d_interm,
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));
                    CUDA_ERR();

                    // SOBEL KERNEL
                    assert(p.n_gpu_threads * p.n_gpu_threads <= max_gpu_threads && 
                        "The thread block size is greater than the maximum thread block size that can be used to execute sobel kernel");
                    // Kernel launch
                    cudaStatus = call_sobel_kernel(p.n_gpu_threads, d_interm, h_in_out[rep], h_theta[rep], 
                        rowsc, colsc, (p.n_gpu_threads + 2) * (p.n_gpu_threads + 2) * sizeof(int));
                    cudaDeviceSynchronize();
                    CUDA_ERR();
                    timer.stop("GPU Proxy: Kernel");

                    // Release CPU proxy
                    (&sobel_ready[rep])->store(1);

                } else if(proxy_tid == CPU_PROXY) {

                    // Wait for GPU proxy
                    while((&sobel_ready[rep])->load() == 0) {
                    }
                    if((&sobel_ready[rep])->load() == -1)
                        continue;

                    timer.start("CPU Proxy: Kernel");
                    std::thread main_thread(
                        run_cpu_threads, h_in_out[rep], h_interm, h_theta[rep], rowsc, colsc, p.n_threads, rep);
                    main_thread.join();
                    timer.stop("CPU Proxy: Kernel");

                    memcpy(all_out_frames[rep], h_in_out[rep], in_size);
                }
            }
        }));
    }
    std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });
    cudaDeviceSynchronize();
    timer.stop("Total Proxies");
    timer.print("Total Proxies", 1);
    timer.print("CPU Proxy: Kernel", 1);
    timer.print("GPU Proxy: Kernel", 1);

#ifdef CHAI_OPENCV
    // Display the result
    if(p.display){
        for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
            cv::Mat out_frame = cv::Mat(rowsc, colsc, CV_8UC1);
            memcpy(out_frame.data, all_out_frames[rep], in_size);
            if(!out_frame.empty())
                imshow("canny", out_frame);
            if(cv::waitKey(30) >= 0)
                break;
        }
    }
#endif

    // Verify answer
    verify(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);

    // Release buffers
    timer.start("Deallocation");
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        cudaStatus = cudaFree(h_in_out[i]);
    }
    cudaStatus = cudaFree(h_in_out);
    free(h_interm);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        cudaStatus = cudaFree(h_theta[i]);
    }
    cudaStatus = cudaFree(h_theta);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        free(all_gray_frames[i]);
    }
    free(all_gray_frames);
    for(int i = 0; i < p.n_warmup + p.n_reps; i++) {
        free(all_out_frames[i]);
    }
    free(all_out_frames);
    cudaStatus = cudaFree(d_interm);
    CUDA_ERR();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
