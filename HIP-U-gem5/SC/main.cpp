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

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

/*extern "C" {
void m5_work_begin(int workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}*/

// Params ---------------------------------------------------------------------
struct Params {

    int   device;
    int   n_gpu_threads;
    int   n_gpu_blocks;
    int   n_threads;
    int   n_warmup;
    int   n_reps;
    float alpha;
    int   in_size;
    int   compaction_factor;
    int   remove_value;

    Params(int argc, char **argv) {
        device            = 0;
        n_gpu_threads     = 256;
        n_gpu_blocks      = 8;
        n_threads         = 4;
				n_warmup          = 0;
				n_reps            = 1;
        alpha             = 0.1;
        in_size           = 1048576;
        compaction_factor = 50;
        remove_value      = 0;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:a:n:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device            = atoi(optarg); break;
            case 'i': n_gpu_threads      = atoi(optarg); break;
            case 'g': n_gpu_blocks     = atoi(optarg); break;
            case 't': n_threads         = atoi(optarg); break;
            case 'w': n_warmup          = atoi(optarg); break;
            case 'r': n_reps            = atoi(optarg); break;
            case 'a': alpha             = atof(optarg); break;
            case 'n': in_size           = atoi(optarg); break;
            case 'c': compaction_factor = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
#ifdef CUDA_8_0
            assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of host + device workers!");
#else
            assert(0 && "Illegal value for -a");
#endif
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./sc [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    HIP device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=8)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.1)"
#ifdef CUDA_8_0
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -n <N>    input size (default=1048576)"
                "\n    -c <C>    compaction factor (default=50)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(T *input, const Params &p) {

    // Initialize the host input vectors
    srand(time(NULL));
    for(int i = 0; i < p.in_size; i++) {
        input[i] = (T)p.remove_value;
    }
    int M = (p.in_size * p.compaction_factor) / 100;
    int m = M;
    while(m > 0) {
        int x = (int)(p.in_size * (((float)rand() / (float)RAND_MAX)));
        if(x < p.in_size)
            if(input[x] == p.remove_value) {
                input[x] = (T)(x + 2);
                m--;
            }
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    hipError_t hipStatus;

    // Allocate buffers
    const int n_tasks     = divceil(p.in_size, p.n_gpu_threads * REGS);
    const int n_tasks_cpu = n_tasks * p.alpha;
    const int n_tasks_gpu = n_tasks - n_tasks_cpu;
    const int n_flags     = n_tasks + 1;
#ifdef CUDA_8_0
    T * h_in_out = (T *)malloc(p.in_size * sizeof(T));
    T * d_in_out = h_in_out;
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));
    std::atomic_int *d_flags  = h_flags;
    std::atomic_int * worklist = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    ALLOC_ERR(h_in_out, h_flags, worklist);
#else
    T *    h_in_out = (T *)malloc(n_tasks * p.n_gpu_threads * REGS * sizeof(T));
    T *    d_in_out;
    hipStatus = hipMalloc((void**)&d_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T));
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));
    int* d_flags;
    hipStatus = hipMalloc((void**)&d_flags, n_flags * sizeof(int));
    if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    ALLOC_ERR(h_in_out, h_flags);
#endif
    T *h_in_backup = (T *)malloc(p.in_size * sizeof(T));
    ALLOC_ERR(h_in_backup);
    hipDeviceSynchronize();

    // Initialize
    read_input(h_in_out, p);
#ifdef CUDA_8_0
    h_flags[0].store(1);
#else
    h_flags[0]           = 1;
    h_flags[n_tasks_cpu] = 1;
#endif
    memcpy(h_in_backup, h_in_out, p.in_size * sizeof(T)); // Backup for reuse across iterations

#ifndef CUDA_8_0
    // Copy to device
    hipStatus = hipMemcpy(d_in_out, h_in_out + n_tasks_cpu * p.n_gpu_threads * REGS, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T), hipMemcpyHostToDevice);
    hipStatus = hipMemcpy(d_flags, h_flags, n_flags * sizeof(int), hipMemcpyHostToDevice);
    if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    hipDeviceSynchronize();
#endif

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memcpy(h_in_out, h_in_backup, p.in_size * sizeof(T));
        memset(h_flags, 0, n_flags * sizeof(std::atomic_int));
#ifdef CUDA_8_0
        h_flags[0].store(1);
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#else
        h_flags[0]           = 1;
        h_flags[n_tasks_cpu] = 1;
        hipStatus = hipMemcpy(d_in_out, h_in_out + n_tasks_cpu * p.n_gpu_threads * REGS, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T), hipMemcpyHostToDevice);
        hipStatus = hipMemcpy(d_flags, h_flags, n_flags * sizeof(int), hipMemcpyHostToDevice);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        hipDeviceSynchronize();
#endif

        //m5_work_begin(0, 0);

        // Kernel launch
        if(p.n_gpu_blocks > 0) {
            hipStatus = call_StreamCompaction_kernel(p.n_gpu_blocks, p.n_gpu_threads, p.in_size, p.remove_value, n_tasks, p.alpha,
                d_in_out, d_in_out, (int*)d_flags,
                p.n_gpu_threads * sizeof(int) + sizeof(int)
#ifdef CUDA_8_0
                + sizeof(int), (int*)worklist
#endif
                );
            if(hipStatus != hipSuccess) { fprintf(stderr, "HIP error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_in_out, h_flags, p.in_size, p.remove_value, p.n_threads,
            p.n_gpu_threads, n_tasks, p.alpha
#ifdef CUDA_8_0
            ,
            worklist
#endif
            );

        hipDeviceSynchronize();
        main_thread.join();

        //m5_work_end(0, 0);
    }

#ifndef CUDA_8_0
    // Copy back
    if(p.alpha < 1.0) {
        int offset = n_tasks_cpu == 0 ? 1 : 2;
        hipStatus = hipMemcpy(h_in_out + h_flags[n_tasks_cpu] - offset, d_in_out, n_tasks_gpu * p.n_gpu_threads * REGS * sizeof(T), hipMemcpyDeviceToHost);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    }
    hipDeviceSynchronize();
#endif

    // Verify answer
    verify(h_in_out, h_in_backup, p.in_size, p.remove_value, (p.in_size * p.compaction_factor) / 100);

    // Free memory
#ifdef CUDA_8_0
    free(h_in_out);
    free(h_flags);
    free(worklist);
#else
    free(h_in_out);
    free(h_flags);
    hipStatus = hipFree(d_in_out);
    hipStatus = hipFree(d_flags);
#endif
    if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    free(h_in_backup);
    hipDeviceSynchronize();

    printf("Test Passed\n");
    return 0;
}
