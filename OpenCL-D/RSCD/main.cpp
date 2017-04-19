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

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_work_groups;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;

    Params(int argc, char **argv) {
        platform              = 0;
        device                = 0;
        n_work_items          = 256;
        n_work_groups         = 8;
        n_threads             = 4;
        n_warmup              = 5;
        n_reps                = 50;
        alpha                 = 0.2;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:e:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform              = atoi(optarg); break;
            case 'd': device                = atoi(optarg); break;
            case 'i': n_work_items          = atoi(optarg); break;
            case 'g': n_work_groups         = atoi(optarg); break;
            case 't': n_threads             = atoi(optarg); break;
            case 'w': n_warmup              = atoi(optarg); break;
            case 'r': n_reps                = atoi(optarg); break;
            case 'a': alpha                 = atof(optarg); break;
            case 'f': file_name             = optarg; break;
            case 'm': max_iter              = atoi(optarg); break;
            case 'e': error_threshold       = atoi(optarg); break;
            case 'c': convergence_threshold = atof(optarg); break;
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
#ifdef OCL_2_0
            assert((n_work_items > 0 && n_work_groups > 0 || n_threads > 0) && "Invalid # of host + device workers!");
#else
            assert(0 && "Illegal value for -a");
#endif
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./rscd [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=256)"
                "\n    -g <G>    # of device work-groups (default=8)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
#ifdef OCL_2_0
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name (default=input/vectors.csv)"
                "\n    -m <M>    maximum # of iterations (default=2000)"
                "\n    -e <E>    error threshold (default=3)"
                "\n    -c <C>    convergence threshold (default=0.75)"
                "\n");
    }
};

// Input ----------------------------------------------------------------------
int read_input_size(const Params &p) {
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error al abrir el fichero");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);

    return n;
}

void read_input(flowvector *v, int *r, const Params &p) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(time(NULL));
    for(int i = 0; i < 2 * p.max_iter; i++) {
        r[i] = ((int)rand()) % n;
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
    int n_flow_vectors = read_input_size(p);
    int best_model     = -1;
    int best_outliers  = n_flow_vectors;
#ifdef OCL_2_0
    flowvector *h_flow_vector_array =
        (flowvector *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, n_flow_vectors * sizeof(flowvector), 0);
    int *h_random_numbers =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, 2 * p.max_iter * sizeof(int), 0);
    int *h_model_candidate =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.max_iter * sizeof(int), 0);
    int *h_outliers_candidate =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.max_iter * sizeof(int), 0);
    float *h_model_param_local =
        (float *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, 4 * p.max_iter * sizeof(float), 0);
    std::atomic_int *h_g_out_id = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    flowvector *     d_flow_vector_array  = h_flow_vector_array;
    int *            d_random_numbers     = h_random_numbers;
    int *            d_model_candidate    = h_model_candidate;
    int *            d_outliers_candidate = h_outliers_candidate;
    float *          d_model_param_local  = h_model_param_local;
    std::atomic_int *d_g_out_id           = h_g_out_id;
    std::atomic_int *worklist             = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    ALLOC_ERR(worklist);
#else
    flowvector *     h_flow_vector_array  = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *            h_random_numbers     = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *            h_model_candidate    = (int *)malloc(p.max_iter * sizeof(int));
    int *            h_outliers_candidate = (int *)malloc(p.max_iter * sizeof(int));
    float *          h_model_param_local  = (float *)malloc(4 * p.max_iter * sizeof(float));
    std::atomic_int *h_g_out_id           = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    cl_mem           d_flow_vector_array  = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n_flow_vectors * sizeof(flowvector), NULL, &clStatus);
    cl_mem d_random_numbers = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 2 * p.max_iter * sizeof(int), NULL, &clStatus);
    cl_mem d_model_candidate = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.max_iter * sizeof(int), NULL, &clStatus);
    cl_mem d_outliers_candidate = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.max_iter * sizeof(int), NULL, &clStatus);
    cl_mem d_model_param_local = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 4 * p.max_iter * sizeof(float), NULL, &clStatus);
    cl_mem d_g_out_id =
        clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int), NULL, &clStatus);
    CL_ERR();
#endif
    ALLOC_ERR(h_flow_vector_array, h_random_numbers, h_model_candidate, h_outliers_candidate, h_model_param_local, h_g_out_id);
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(h_flow_vector_array, h_random_numbers, p);
    clFinish(ocl.clCommandQueue);
    timer.stop("Initialization");
    timer.print("Initialization", 1);

#ifndef OCL_2_0
    // Copy to device
    timer.start("Copy To Device");
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_flow_vector_array, CL_TRUE, 0,
        n_flow_vectors * sizeof(flowvector), h_flow_vector_array, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_random_numbers, CL_TRUE, 0, 2 * p.max_iter * sizeof(int),
        h_random_numbers, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(
        ocl.clCommandQueue, d_model_candidate, CL_TRUE, 0, p.max_iter * sizeof(int), h_model_candidate, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_outliers_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
        h_outliers_candidate, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_model_param_local, CL_TRUE, 0, 4 * p.max_iter * sizeof(float),
        h_model_param_local, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_g_out_id, CL_TRUE, 0, sizeof(int), h_g_out_id, 0, NULL, NULL);
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);
#endif

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));
#ifdef OCL_2_0
        h_g_out_id[0].store(0);
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#else
        h_g_out_id[0] = 0;
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_model_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
            h_model_candidate, 0, NULL, NULL);
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_outliers_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
            h_outliers_candidate, 0, NULL, NULL);
        clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_model_param_local, CL_TRUE, 0,
            4 * p.max_iter * sizeof(float), h_model_param_local, 0, NULL, NULL);
        clStatus =
            clEnqueueWriteBuffer(ocl.clCommandQueue, d_g_out_id, CL_TRUE, 0, sizeof(int), h_g_out_id, 0, NULL, NULL);
        CL_ERR();
#endif
        clFinish(ocl.clCommandQueue);

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Launch GPU threads
        clSetKernelArg(ocl.clKernel, 0, sizeof(int), &n_flow_vectors);
        clSetKernelArg(ocl.clKernel, 1, sizeof(int), &p.max_iter);
        clSetKernelArg(ocl.clKernel, 2, sizeof(int), &p.error_threshold);
        clSetKernelArg(ocl.clKernel, 3, sizeof(float), &p.convergence_threshold);
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &p.max_iter);
        clSetKernelArg(ocl.clKernel, 5, sizeof(float), &p.alpha);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 6, d_model_param_local);
        clSetKernelArgSVMPointer(ocl.clKernel, 7, d_flow_vector_array);
        clSetKernelArgSVMPointer(ocl.clKernel, 8, d_random_numbers);
        clSetKernelArgSVMPointer(ocl.clKernel, 9, d_model_candidate);
        clSetKernelArgSVMPointer(ocl.clKernel, 10, d_outliers_candidate);
        clSetKernelArgSVMPointer(ocl.clKernel, 11, d_g_out_id);
        clSetKernelArg(ocl.clKernel, 12, sizeof(std::atomic_int), NULL);
        clSetKernelArgSVMPointer(ocl.clKernel, 13, worklist);
        clSetKernelArg(ocl.clKernel, 14, sizeof(int), NULL);
#else
        clSetKernelArg(ocl.clKernel, 6, sizeof(cl_mem), &d_model_param_local);
        clSetKernelArg(ocl.clKernel, 7, sizeof(cl_mem), &d_flow_vector_array);
        clSetKernelArg(ocl.clKernel, 8, sizeof(cl_mem), &d_random_numbers);
        clSetKernelArg(ocl.clKernel, 9, sizeof(cl_mem), &d_model_candidate);
        clSetKernelArg(ocl.clKernel, 10, sizeof(cl_mem), &d_outliers_candidate);
        clSetKernelArg(ocl.clKernel, 11, sizeof(cl_mem), &d_g_out_id);
        clSetKernelArg(ocl.clKernel, 12, sizeof(int), NULL);
#endif

        // Kernel launch
        if(p.n_work_groups > 0) {
            size_t ls[1] = {(size_t)p.n_work_items};
            size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
            clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_model_candidate, h_outliers_candidate, h_model_param_local,
            h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
            p.convergence_threshold, h_g_out_id, p.n_threads, p.max_iter, p.alpha
#ifdef OCL_2_0
            ,
            worklist);
#else
            );
#endif

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");

#ifndef OCL_2_0
        // Copy back
        if(rep >= p.n_warmup)
            timer.start("Copy Back and Merge");
        int d_candidates = 0;
        if(p.alpha < 1.0) {
            clStatus = clEnqueueReadBuffer(
                ocl.clCommandQueue, d_g_out_id, CL_TRUE, 0, sizeof(int), &d_candidates, 0, NULL, NULL);
            clStatus = clEnqueueReadBuffer(ocl.clCommandQueue, d_model_candidate, CL_TRUE, 0,
                d_candidates * sizeof(int), &h_model_candidate[h_g_out_id[0]], 0, NULL, NULL);
            clStatus = clEnqueueReadBuffer(ocl.clCommandQueue, d_outliers_candidate, CL_TRUE, 0,
                d_candidates * sizeof(int), &h_outliers_candidate[h_g_out_id[0]], 0, NULL, NULL);
            CL_ERR();
        }
        h_g_out_id[0] += d_candidates;
        clFinish(ocl.clCommandQueue);
        if(rep >= p.n_warmup)
            timer.stop("Copy Back and Merge");
#endif

        // Post-processing (chooses the best model among the candidates)
        if(rep >= p.n_warmup)
            timer.start("Kernel");
        for(int i = 0; i < h_g_out_id[0]; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);
    timer.print("Copy Back and Merge", p.n_reps);

    // Verify answer
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
        p.convergence_threshold, h_g_out_id[0], best_outliers);

    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_model_candidate);
    clSVMFree(ocl.clContext, h_outliers_candidate);
    clSVMFree(ocl.clContext, h_model_param_local);
    clSVMFree(ocl.clContext, h_g_out_id);
    clSVMFree(ocl.clContext, h_flow_vector_array);
    clSVMFree(ocl.clContext, h_random_numbers);
    clSVMFree(ocl.clContext, worklist);
#else
    free(h_model_candidate);
    free(h_outliers_candidate);
    free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);
    clStatus = clReleaseMemObject(d_model_candidate);
    clStatus = clReleaseMemObject(d_outliers_candidate);
    clStatus = clReleaseMemObject(d_model_param_local);
    clStatus = clReleaseMemObject(d_g_out_id);
    clStatus = clReleaseMemObject(d_flow_vector_array);
    clStatus = clReleaseMemObject(d_random_numbers);
    CL_ERR();
#endif
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
