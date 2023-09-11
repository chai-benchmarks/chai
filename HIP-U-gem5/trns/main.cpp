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
#include <string.h>
#include <assert.h>

/*extern "C" {
void m5_work_begin(int workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}*/

// Params ---------------------------------------------------------------------
struct Params {

    int device;
    int n_gpu_threads;
    int n_gpu_blocks;
    int n_threads;
    int n_warmup;
    int n_reps;
    int m;
    int n;
    int s;

    Params(int argc, char **argv) {
        device        = 0;
        n_gpu_threads = 64;
        n_gpu_blocks  = 16;
#ifdef CUDA_8_0
        n_threads     = 4;
#else
        n_threads     = 0;
#endif
				n_warmup      = 0;
				n_reps        = 1;
        m             = 49;
        n             = 45;
        s             = 32;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:m:n:s:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device        = atoi(optarg); break;
            case 'i': n_gpu_threads = atoi(optarg); break;
            case 'g': n_gpu_blocks  = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'm': m             = atoi(optarg); break;
            case 'n': n             = atoi(optarg); break;
            case 's': s             = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
#ifdef CUDA_8_0
        assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of CPU + GPU workers!");
#else
        assert(((n_gpu_threads > 0 && n_gpu_blocks > 0) ^ (n_threads > 0))
            && "TRNS only runs on CPU-only or GPU-only: './trns -g 0' or './trns -t 0'");
#endif
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./trns [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    HIP device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=64)"
                "\n    -g <G>    # of device blocks (default=16)"
#ifdef CUDA_8_0
                "\n    -t <T>    # of host threads (default=4)"
#else
                "\n    -t <T>    # of host threads (default=0)"
#endif
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
#ifdef CUDA_8_0
                "\n    TRNS only supports dynamic partitioning"
#else
                "\n    TRNS only supports CPU-only or GPU-only execution"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -m <M>    matrix height (default=197)"
                "\n    -n <N>    matrix width (default=35588)"
                "\n    -s <M>    super-element size (default=32)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(T *x_vector, const Params &p) {
    int tiled_n = divceil(p.n, p.s);
    int in_size = p.m * tiled_n * p.s;
    srand(5432);
    for(int i = 0; i < in_size; i++) {
        x_vector[i] = ((T)(rand() % 100) / 100);
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    hipError_t  hipStatus;

    // Allocate
    int tiled_n       = divceil(p.n, p.s);
    int in_size       = p.m * tiled_n * p.s;
    int finished_size = p.m * tiled_n;
#ifdef CUDA_8_0
    T * h_in_out = (T *)malloc(in_size * sizeof(T));
    T * d_in_out = h_in_out;
    std::atomic_int * h_finished = (std::atomic_int *)malloc(sizeof(std::atomic_int) * finished_size);
    std::atomic_int *d_finished = h_finished;
    std::atomic_int * h_head = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    std::atomic_int *d_head = h_head;
#else
    T *              h_in_out = (T *)malloc(in_size * sizeof(T));
    std::atomic_int *h_finished =
        (std::atomic_int *)malloc(sizeof(std::atomic_int) * finished_size);
    std::atomic_int *h_head = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    T * d_in_out;
    int * d_finished;
    int * d_head;
    if(p.n_gpu_blocks != 0) {
        hipStatus = hipMalloc((void**)&d_in_out, in_size * sizeof(T));
        hipStatus = hipMalloc((void**)&d_finished, (p.n_gpu_blocks != 0) ? sizeof(int) * finished_size : 0);
        hipStatus = hipMalloc((void**)&d_head, (p.n_gpu_blocks != 0) ? sizeof(int) : 0);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    }
#endif
    T *h_in_backup = (T *)malloc(in_size * sizeof(T));
    ALLOC_ERR(h_in_out, h_finished, h_head);
    ALLOC_ERR(h_in_backup);
    hipDeviceSynchronize();

    // Initialize
    read_input(h_in_out, p);
    memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
    h_head[0].store(0);
    memcpy(h_in_backup, h_in_out, in_size * sizeof(T)); // Backup for reuse across iterations

#ifndef CUDA_8_0
    // Copy to device
    if(p.n_gpu_blocks != 0) {
        hipStatus = hipMemcpy(d_in_out, h_in_backup, in_size * sizeof(T), hipMemcpyHostToDevice);
        hipStatus = hipMemcpy(d_finished, h_finished, sizeof(int) * finished_size, hipMemcpyHostToDevice);
        hipStatus = hipMemcpy(d_head, h_head, sizeof(int), hipMemcpyHostToDevice);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    }
    hipDeviceSynchronize();
#endif

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memcpy(h_in_out, h_in_backup, in_size * sizeof(T));
        memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
        h_head[0].store(0);
#ifndef CUDA_8_0
        if(p.n_gpu_blocks != 0) {
            hipStatus = hipMemcpy(d_in_out, h_in_backup, in_size * sizeof(T), hipMemcpyHostToDevice);
            hipStatus = hipMemcpy(d_finished, h_finished, sizeof(int) * finished_size, hipMemcpyHostToDevice);
            hipStatus = hipMemcpy(d_head, h_head, sizeof(int), hipMemcpyHostToDevice);
            if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        }
#endif
        hipDeviceSynchronize();

        //m5_work_begin(0, 0);

        // Launch GPU threads
        if(p.n_gpu_blocks > 0) {
            // Kernel launch
            hipStatus = call_PTTWAC_soa_asta(p.n_gpu_blocks, p.n_gpu_threads, p.m, tiled_n, p.s,
                d_in_out, (int*)d_finished, (int*)d_head, sizeof(int) + sizeof(int));
            if(hipStatus != hipSuccess) { fprintf(stderr, "HIP error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_finished, h_head, p.m, tiled_n, p.s, p.n_threads);

        hipDeviceSynchronize();
        main_thread.join();

        //m5_work_end(0, 0);
    }

#ifndef CUDA_8_0
    // Copy back
    if(p.n_gpu_blocks != 0) {
        hipStatus = hipMemcpy(h_in_out, d_in_out, in_size * sizeof(T), hipMemcpyDeviceToHost);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        hipDeviceSynchronize();
    }
#endif

    // Verify answer
    verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);

    // Free memory
#ifdef CUDA_8_0
    free(h_in_out);
    free(h_finished);
    free(h_head);
#else
    free(h_in_out);
    free(h_finished);
    free(h_head);
    if(p.n_gpu_blocks != 0) {
        hipStatus = hipFree(d_in_out);
        hipStatus = hipFree(d_finished);
        hipStatus = hipFree(d_head);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    }
#endif
    free(h_in_backup);
    hipDeviceSynchronize();

    printf("Test Passed\n");
    return 0;
}
