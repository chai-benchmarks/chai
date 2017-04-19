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
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int   platform;
    int   device;
    int   n_work_items;
    int   n_work_groups;
    int   n_threads;
    int   n_warmup;
    int   n_reps;
    float alpha;
    int   in_size;
    int   n_bins;

    Params(int argc, char **argv) {
        platform      = 0;
        device        = 0;
        n_work_items  = 256;
        n_work_groups = 16;
        n_threads     = 4;
        n_warmup      = 5;
        n_reps        = 50;
        alpha         = 0.25;
        in_size       = 1536 * 1024;
        n_bins        = 256;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:n:b:")) >= 0) {
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
            case 'a': alpha         = atof(optarg); break;
            case 'n': in_size       = atoi(optarg); break;
            case 'b': n_bins        = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_work_items > 0 && "Invalid # of device work-items!");
            assert(n_work_groups > 0 && "Invalid # of device work-groups!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_work_items > 0 && "Invalid # of device work-items!");
            assert(n_work_groups > 0 && "Invalid # of device work-groups!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
            assert(0 && "Illegal value for -a");
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./hsto [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=256)"
                "\n    -g <G>    # of device work-groups (default=16)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of output bins to process on host (default=0.25)"
                "\n              NOTE: <A> must be between 0.0 and 1.0"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -n <N>    input size (default=1572864, i.e., 1536x1024)"
                "\n    -b <B>    # of bins in histogram (default=256)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned int *input, const Params &p) {

    char  dctFileName[100];
    FILE *File = NULL;

    // Open input file
    unsigned short temp;
    sprintf(dctFileName, "./input/image_VanHateren.iml");
    if((File = fopen(dctFileName, "rb")) != NULL) {
        for(int y = 0; y < p.in_size; y++) {
            int fr   = fread(&temp, sizeof(unsigned short), 1, File);
            input[y] = (unsigned int)ByteSwap16(temp);
            if(input[y] >= 4096)
                input[y] = 4095;
        }
        fclose(File);
    } else {
        printf("%s does not exist\n", dctFileName);
        exit(1);
    }
}

// Main -----------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;

    // Allocate buffers
    timer.start("Allocation");
#ifdef OCL_2_0
    unsigned int *h_in =
        (unsigned int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.in_size * sizeof(unsigned int), 0);
    std::atomic_uint *h_histo = (std::atomic_uint *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, p.n_bins * sizeof(std::atomic_uint), 0);
    unsigned int *    d_in    = h_in;
    std::atomic_uint *d_histo = h_histo;
    ALLOC_ERR(h_in, h_histo);
#else
    unsigned int *    h_in          = (unsigned int *)malloc(p.in_size * sizeof(unsigned int));
    std::atomic_uint *h_histo       = (std::atomic_uint *)malloc(p.n_bins * sizeof(std::atomic_uint));
    unsigned int *    h_histo_merge = (unsigned int *)malloc(p.n_bins * sizeof(unsigned int));
    cl_mem            d_in          = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.in_size * sizeof(unsigned int), NULL, &clStatus);
    cl_mem d_histo = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.n_bins * sizeof(unsigned int), NULL, &clStatus);
    CL_ERR();
    ALLOC_ERR(h_in, h_histo, h_histo_merge);
#endif
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(h_in, p);
#ifdef OCL_2_0
    for(int i = 0; i < p.n_bins; i++) {
        h_histo[i].store(0);
    }
#else
    memset(h_histo, 0, p.n_bins * sizeof(unsigned int));
#endif
    int n_cpu_bins = p.n_bins * p.alpha;
    timer.stop("Initialization");
    timer.print("Initialization", 1);

#ifndef OCL_2_0
    // Copy to device
    timer.start("Copy To Device");
    clStatus = clEnqueueWriteBuffer(
        ocl.clCommandQueue, d_in, CL_TRUE, 0, p.in_size * sizeof(unsigned int), h_in, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(
        ocl.clCommandQueue, d_histo, CL_TRUE, 0, p.n_bins * sizeof(unsigned int), h_histo, 0, NULL, NULL);
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);
#endif

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
#ifdef OCL_2_0
        for(int i = 0; i < p.n_bins; i++) {
            h_histo[i].store(0);
        }
#else
        memset(h_histo, 0, p.n_bins * sizeof(unsigned int));
        clStatus = clEnqueueWriteBuffer(
            ocl.clCommandQueue, d_histo, CL_TRUE, 0, p.n_bins * sizeof(unsigned int), h_histo, 0, NULL, NULL);
        clFinish(ocl.clCommandQueue);
        CL_ERR();
#endif

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Launch GPU threads
        clSetKernelArg(ocl.clKernel, 0, sizeof(int), &p.in_size);
        clSetKernelArg(ocl.clKernel, 1, sizeof(int), &p.n_bins);
        clSetKernelArg(ocl.clKernel, 2, sizeof(int), &n_cpu_bins);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 3, d_in);
        clSetKernelArgSVMPointer(ocl.clKernel, 4, d_histo);
#else
        clSetKernelArg(ocl.clKernel, 3, sizeof(cl_mem), &d_in);
        clSetKernelArg(ocl.clKernel, 4, sizeof(cl_mem), &d_histo);
#endif
        clSetKernelArg(ocl.clKernel, 5, p.n_bins * sizeof(std::atomic_int), NULL);
        // Kernel launch
        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
        if(p.n_work_groups > 0) {
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
            clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, (unsigned int *)h_histo, h_in, p.in_size, p.n_bins, p.n_threads,
            p.n_work_items, n_cpu_bins);

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

#ifndef OCL_2_0
    // Copy back
    timer.start("Copy Back and Merge");
    clStatus = clEnqueueReadBuffer(
        ocl.clCommandQueue, d_histo, CL_TRUE, 0, p.n_bins * sizeof(unsigned int), h_histo_merge, 0, NULL, NULL);
    CL_ERR();
    clFinish(ocl.clCommandQueue);
    for(unsigned int i = 0; i < p.n_bins; ++i) {
        h_histo_merge[i] += (unsigned int)h_histo[i];
    }
    timer.stop("Copy Back and Merge");
    timer.print("Copy Back and Merge", 1);
#endif

    // Verify answer
#ifdef OCL_2_0
    verify((unsigned int *)h_histo, h_in, p.in_size, p.n_bins);
#else
    verify((unsigned int *)h_histo_merge, h_in, p.in_size, p.n_bins);
#endif

    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in);
    clSVMFree(ocl.clContext, h_histo);
#else
    clStatus = clReleaseMemObject(d_in);
    clStatus = clReleaseMemObject(d_histo);
    CL_ERR();
    free(h_in);
    free(h_histo);
    free(h_histo_merge);
#endif
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
