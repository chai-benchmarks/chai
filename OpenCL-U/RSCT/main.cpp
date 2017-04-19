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
    const char *file_name;
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;

    Params(int argc, char **argv) {
        platform              = 0;
        device                = 0;
        n_work_items          = 256;
        n_work_groups         = 64;
        n_threads             = 1;
        n_warmup              = 5;
        n_reps                = 50;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:f:m:e:c:")) >= 0) {
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
        assert(n_work_items > 0 && "Invalid # of device work-items!");
        assert(n_work_groups > 0 && "Invalid # of device work-groups!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./rsct [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=256)"
                "\n    -g <G>    # of device work-groups (default=64)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name (default=input/vectors.csv)"
                "\n    -m <M>    maximum # of iterations (default=2000)"
                "\n    -e <E>    error threshold (default=3)"
                "\n    -c <C>    convergence threshold (default=0.75)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
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
    int         n_flow_vectors = read_input_size(p);
    int         best_model     = -1;
    int         best_outliers  = n_flow_vectors;
    flowvector *flow_vector_array =
        (flowvector *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, n_flow_vectors * sizeof(flowvector), 0);
    int *random_numbers =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, 2 * p.max_iter * sizeof(int), 0);
    int *model_candidate = (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.max_iter * sizeof(int), 0);
    int *outliers_candidate =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.max_iter * sizeof(int), 0);
    float *model_param_local =
        (float *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, 4 * p.max_iter * sizeof(float), 0);
    std::atomic_int *g_out_id = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    std::atomic_int *launch_gpu =
        (std::atomic_int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
            (p.max_iter + p.n_work_groups) * sizeof(std::atomic_int) * 4, 0);
    clFinish(ocl.clCommandQueue);
    ALLOC_ERR(flow_vector_array, random_numbers, model_candidate, outliers_candidate, model_param_local, g_out_id, 
        launch_gpu);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(flow_vector_array, random_numbers, p);
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memset((void *)model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)model_param_local, 0, 4 * p.max_iter * sizeof(float));
        g_out_id[0].store(0);
        for(int i = 0; i < p.max_iter + p.n_work_groups; i++) {
            launch_gpu[i].store(0);
        }
        clFinish(ocl.clCommandQueue);

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Launch GPU threads
        clSetKernelArgSVMPointer(ocl.clKernel, 0, model_param_local);
        clSetKernelArgSVMPointer(ocl.clKernel, 1, flow_vector_array);
        clSetKernelArg(ocl.clKernel, 2, sizeof(int), &n_flow_vectors);
        clSetKernelArgSVMPointer(ocl.clKernel, 3, random_numbers);
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &p.max_iter);
        clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.error_threshold);
        clSetKernelArg(ocl.clKernel, 6, sizeof(float), &p.convergence_threshold);
        clSetKernelArgSVMPointer(ocl.clKernel, 7, g_out_id);
        clSetKernelArg(ocl.clKernel, 8, sizeof(std::atomic_int), NULL);
        clSetKernelArgSVMPointer(ocl.clKernel, 9, model_candidate);
        clSetKernelArgSVMPointer(ocl.clKernel, 10, outliers_candidate);
        clSetKernelArgSVMPointer(ocl.clKernel, 11, launch_gpu);

        // Kernel launch
        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
        assert(ls[0] <= max_wi && 
            "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
        clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
        CL_ERR();

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, model_param_local, flow_vector_array, n_flow_vectors, random_numbers,
            p.max_iter, p.error_threshold, p.convergence_threshold, g_out_id, p.n_threads, launch_gpu);

        clFinish(ocl.clCommandQueue);

        main_thread.join();

        // Post-processing (chooses the best model among the candidates)
        for(int i = 0; i < g_out_id[0]; i++) {
            if(outliers_candidate[i] < best_outliers) {
                best_outliers = outliers_candidate[i];
                best_model    = model_candidate[i];
            }
        }

        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

    // Verify answer
    verify(flow_vector_array, n_flow_vectors, random_numbers, p.max_iter, p.error_threshold, p.convergence_threshold,
        g_out_id[0], best_outliers);

    // Free memory
    timer.start("Deallocation");
    clSVMFree(ocl.clContext, model_candidate);
    clSVMFree(ocl.clContext, outliers_candidate);
    clSVMFree(ocl.clContext, model_param_local);
    clSVMFree(ocl.clContext, g_out_id);
    clSVMFree(ocl.clContext, flow_vector_array);
    clSVMFree(ocl.clContext, random_numbers);
    clSVMFree(ocl.clContext, launch_gpu);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
