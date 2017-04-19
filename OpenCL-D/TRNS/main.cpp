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
#include "support/ocl.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <string.h>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int platform;
    int device;
    int n_work_items;
    int n_work_groups;
    int n_threads;
    int n_warmup;
    int n_reps;
    int m;
    int n;
    int s;

    Params(int argc, char **argv) {
        platform      = 0;
        device        = 0;
        n_work_items  = 64;
        n_work_groups = 16;
#ifdef OCL_2_0
        n_threads     = 4;
#else
        n_threads     = 0;
#endif
        n_warmup      = 5;
        n_reps        = 50;
        m             = 197;
        n             = 35588;
        s             = 32;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:m:n:s:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform      = atoi(optarg); break;
            case 'd': device        = atoi(optarg); break;
            case 'i': n_work_items  = atoi(optarg); break;
            case 'g': n_work_groups = atoi(optarg); break;
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
#ifdef OCL_2_0
        assert((n_work_items > 0 && n_work_groups > 0 || n_threads > 0) && "Invalid # of CPU + GPU workers!");
#else
        assert(((n_work_items > 0 && n_work_groups > 0) ^ (n_threads > 0))
            && "TRNS only runs on CPU-only or GPU-only: './trns -g 0' or './trns -t 0'");
#endif
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./trns [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=64)"
                "\n    -g <G>    # of device work-groups (default=16)"
#ifdef OCL_2_0
                "\n    -t <T>    # of host threads (default=4)"
#else
                "\n    -t <T>    # of host threads (default=0)"
#endif
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
#ifdef OCL_2_0
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
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;

    // Allocate
    timer.start("Allocation");
    int tiled_n       = divceil(p.n, p.s);
    int in_size       = p.m * tiled_n * p.s;
    int finished_size = p.m * tiled_n;
#ifdef OCL_2_0
    T *              h_in_out   = (T *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, in_size * sizeof(T), 0);
    T *              d_in_out   = h_in_out;
    std::atomic_int *h_finished = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * finished_size, 0);
    std::atomic_int *d_finished = h_finished;
    std::atomic_int *h_head     = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    std::atomic_int *d_head = h_head;
#else
    T *              h_in_out = (T *)malloc(in_size * sizeof(T));
    std::atomic_int *h_finished =
        (std::atomic_int *)malloc(sizeof(std::atomic_int) * finished_size);
    std::atomic_int *h_head = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    cl_mem           d_in_out;
    cl_mem           d_finished;
    cl_mem           d_head;
    if(p.n_work_groups != 0) {
        d_in_out   = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size * sizeof(T), NULL, &clStatus);
        d_finished = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE,
            (p.n_work_groups != 0) ? sizeof(int) * finished_size : 0, NULL, &clStatus);
        d_head =
            clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, (p.n_work_groups != 0) ? sizeof(int) : 0, NULL, &clStatus);
        CL_ERR();
    }
#endif
    T *h_in_backup = (T *)malloc(in_size * sizeof(T));
    ALLOC_ERR(h_in_out, h_finished, h_head, h_in_backup);
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(h_in_out, p);
    memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
    h_head[0].store(0);
    timer.stop("Initialization");
    timer.print("Initialization", 1);
    memcpy(h_in_backup, h_in_out, in_size * sizeof(T)); // Backup for reuse across iterations

#ifndef OCL_2_0
    // Copy to device
    timer.start("Copy To Device");
    if(p.n_work_groups != 0) {
        clStatus = clEnqueueWriteBuffer(
            ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size * sizeof(T), h_in_backup, 0, NULL, NULL);
        clStatus = clEnqueueWriteBuffer(
            ocl.clCommandQueue, d_finished, CL_TRUE, 0, sizeof(int) * finished_size, h_finished, 0, NULL, NULL);
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_head, CL_TRUE, 0, sizeof(int), h_head, 0, NULL, NULL);
        CL_ERR();
    }
    clFinish(ocl.clCommandQueue);
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);
#endif

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memcpy(h_in_out, h_in_backup, in_size * sizeof(T));
        memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
        h_head[0].store(0);
#ifndef OCL_2_0
        if(p.n_work_groups != 0) {
            clStatus = clEnqueueWriteBuffer(
                ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size * sizeof(T), h_in_backup, 0, NULL, NULL);
            clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_head, CL_TRUE, 0, sizeof(int), h_head, 0, NULL, NULL);
            clStatus = clEnqueueWriteBuffer(
                ocl.clCommandQueue, d_finished, CL_TRUE, 0, sizeof(int) * finished_size, h_finished, 0, NULL, NULL);
            CL_ERR();
        }
#endif
        clFinish(ocl.clCommandQueue);

        // start timer
        if(rep >= p.n_warmup)
            timer.start("Kernel");

// Launch GPU threads
        if(p.n_work_groups > 0) {
            clSetKernelArg(ocl.clKernel, 0, sizeof(int), &p.m);
            clSetKernelArg(ocl.clKernel, 1, sizeof(int), &tiled_n);
            clSetKernelArg(ocl.clKernel, 2, sizeof(int), &p.s);
            clSetKernelArg(ocl.clKernel, 3, sizeof(int), NULL);
            clSetKernelArg(ocl.clKernel, 4, sizeof(int), NULL);
#ifdef OCL_2_0
            clSetKernelArgSVMPointer(ocl.clKernel, 5, d_in_out);
            clSetKernelArgSVMPointer(ocl.clKernel, 6, d_finished);
            clSetKernelArgSVMPointer(ocl.clKernel, 7, d_head);
#else
            clSetKernelArg(ocl.clKernel, 5, sizeof(cl_mem), &d_in_out);
            clSetKernelArg(ocl.clKernel, 6, sizeof(cl_mem), &d_finished);
            clSetKernelArg(ocl.clKernel, 7, sizeof(cl_mem), &d_head);
#endif

            // Kernel launch
            size_t ls[1] = {(size_t)p.n_work_items};
            size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
            clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_finished, h_head, p.m, tiled_n, p.s, p.n_threads);

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        // end timer
        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

#ifndef OCL_2_0
    // Copy back
    timer.start("Copy Back and Merge");
    if(p.n_work_groups != 0) {
        clStatus = clEnqueueReadBuffer(
            ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size * sizeof(T), h_in_out, 0, NULL, NULL);
        CL_ERR();
        clFinish(ocl.clCommandQueue);
    }
    timer.stop("Copy Back and Merge");
    timer.print("Copy Back and Merge", 1);
#endif

    // Verify answer
    verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);

    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in_out);
    clSVMFree(ocl.clContext, h_finished);
    clSVMFree(ocl.clContext, h_head);
#else
    free(h_in_out);
    free(h_finished);
    free(h_head);
    if(p.n_work_groups != 0) {
        clStatus = clReleaseMemObject(d_in_out);
        clStatus = clReleaseMemObject(d_finished);
        clStatus = clReleaseMemObject(d_head);
        CL_ERR();
    }
#endif
    free(h_in_backup);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
