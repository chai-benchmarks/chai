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
    int         pool_size;
    int         queue_size;
    int         m;
    int         n;
    int         n_bins;

    Params(int argc, char **argv) {
        platform      = 0;
        device        = 0;
        n_work_items  = 64;
        n_work_groups = 160;
        n_threads     = 1;
        n_warmup      = 1;
        n_reps        = 10;
        file_name     = "input/basket/basket";
        pool_size     = 1600;
        queue_size    = 320;
        m             = 288;
        n             = 352;
        n_bins        = 256;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:f:s:q:m:n:b:")) >= 0) {
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
            case 'f': file_name     = optarg; break;
            case 's': pool_size     = atoi(optarg); break;
            case 'q': queue_size    = atoi(optarg); break;
            case 'm': m             = atoi(optarg); break;
            case 'n': n             = atoi(optarg); break;
            case 'b': n_bins        = atoi(optarg); break;
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
                "\nUsage:  ./tqh [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=64)"
                "\n    -g <G>    # of device work-groups (default=160)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=1)"
                "\n    -r <R>    # of timed repetition iterations (default=10)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input video file name (default=input/basket/basket)"
                "\n    -s <S>    task pool size, i.e., # of videos frames (default=1600)"
                "\n    -q <Q>    task queue size (default=320)"
                "\n    -m <M>    video height (default=288)"
                "\n    -n <N>    video width (default=352)"
                "\n    -b <B>    # of histogram bins (default=256)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(int *data, task_t *task_pool, const Params &p) {
    // Read frames from files
    int   fr = 0;
    char  dctFileName[100];
    FILE *File;
    float v;
    int   frame_size = p.n * p.m;
    for(int i = 0; i < p.pool_size; i++) {
        sprintf(dctFileName, "%s%d.float", p.file_name, (i % 2));
        if((File = fopen(dctFileName, "rt")) != NULL) {
            for(int y = 0; y < p.m; ++y) {
                for(int x = 0; x < p.n; ++x) {
                    fscanf(File, "%f ", &v);
                    *(data + i * frame_size + y * p.n + x) = (int)v;
                }
            }
            fclose(File);
        } else {
            printf("Unable to open file %s\n", dctFileName);
            exit(-1);
        }
        fr++;
    }

    for(int i = 0; i < p.pool_size; i++) {
        task_pool[i].id = i;
        task_pool[i].op = SIGNAL_WORK_KERNEL;
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
    int     frame_size    = p.n * p.m;
    task_t *h_task_pool   = (task_t *)malloc(p.pool_size * sizeof(task_t));
    task_t *h_task_queues = (task_t *)malloc(p.queue_size * sizeof(task_t));
    cl_mem  d_task_queues = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, p.queue_size * sizeof(task_t), NULL, &clStatus);
    int *   h_data_pool   = (int *)malloc(p.pool_size * frame_size * sizeof(int));
    int *   h_data_queues = (int *)malloc(p.queue_size * frame_size * sizeof(int));
    cl_mem  d_data_queues =
        clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, p.queue_size * frame_size * sizeof(int), NULL, &clStatus);
    int *  h_histo        = (int *)malloc(p.pool_size * p.n_bins * sizeof(int));
    int *  h_histo_queues = (int *)malloc(p.queue_size * p.n_bins * sizeof(int));
    cl_mem d_histo_queues =
        clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, p.queue_size * p.n_bins * sizeof(int), NULL, &clStatus);
    int *  h_consumed = (int *)malloc(sizeof(int));
    cl_mem d_consumed = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    ALLOC_ERR(h_task_pool, h_task_queues, h_data_pool, h_data_queues, h_histo, h_histo_queues, h_consumed);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(h_data_pool, h_task_pool, p);
    memset((void *)h_histo_queues, 0, p.queue_size * p.n_bins * sizeof(int));
    memset((void *)h_consumed, 0, sizeof(int));
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        int n_written_tasks = 0;

        for(int n_consumed_tasks = 0; n_consumed_tasks < p.pool_size; n_consumed_tasks += p.queue_size) {

            if(rep >= p.n_warmup)
                timer.start("Kernel");
            host_insert_tasks(h_task_queues, h_data_queues, h_task_pool, h_data_pool, &n_written_tasks, p.queue_size,
                n_consumed_tasks, frame_size);
            if(rep >= p.n_warmup)
                timer.stop("Kernel");

            if(rep >= p.n_warmup)
                timer.start("Copy To Device");
            clStatus = clEnqueueWriteBuffer(
                ocl.clCommandQueue, d_task_queues, CL_TRUE, 0, p.queue_size * sizeof(task_t), h_task_queues, 0, NULL, 
                NULL);
            clStatus = clEnqueueWriteBuffer(
                ocl.clCommandQueue, d_data_queues, CL_TRUE, 0, p.queue_size * frame_size * sizeof(int), h_data_queues, 
                0, NULL, NULL);
            clStatus = clEnqueueWriteBuffer(
                ocl.clCommandQueue, d_histo_queues, CL_TRUE, 0, p.queue_size * p.n_bins * sizeof(int), h_histo_queues, 
                0, NULL, NULL);
            clStatus = clEnqueueWriteBuffer(
                ocl.clCommandQueue, d_consumed, CL_TRUE, 0, sizeof(int), h_consumed, 0, NULL, NULL);
            CL_ERR();
            if(rep >= p.n_warmup)
                timer.stop("Copy To Device");

            if(rep >= p.n_warmup)
                timer.start("Kernel");
            // Setting kernel arguments
            clSetKernelArg(ocl.clKernel, 0, sizeof(task_t *), &d_task_queues);
            clSetKernelArg(ocl.clKernel, 1, sizeof(int *), &d_data_queues);
            clSetKernelArg(ocl.clKernel, 2, sizeof(int *), &d_histo_queues);
            clSetKernelArg(ocl.clKernel, 3, sizeof(int), &n_consumed_tasks);
            clSetKernelArg(ocl.clKernel, 4, sizeof(task_t), NULL);
            clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.queue_size);
            clSetKernelArg(ocl.clKernel, 6, sizeof(cl_mem), &d_consumed);
            clSetKernelArg(ocl.clKernel, 7, sizeof(int), NULL);
            clSetKernelArg(ocl.clKernel, 8, p.n_bins * sizeof(int), NULL);
            clSetKernelArg(ocl.clKernel, 9, sizeof(int), &frame_size);
            clSetKernelArg(ocl.clKernel, 10, sizeof(int), &p.n_bins);
            // Kernel launch
            size_t ls[1] = {(size_t)p.n_work_items};
            size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
            clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
            clFinish(ocl.clCommandQueue);
            if(rep >= p.n_warmup)
                timer.stop("Kernel");

            if(rep >= p.n_warmup)
                timer.start("Copy Back and Merge");
            clStatus = clEnqueueReadBuffer(
                ocl.clCommandQueue, d_histo_queues, CL_TRUE, 0, p.queue_size * p.n_bins * sizeof(int),
                &h_histo[n_consumed_tasks * p.n_bins], 0, NULL, NULL);
            CL_ERR();
            if(rep >= p.n_warmup)
                timer.stop("Copy Back and Merge");
        }
    }
    timer.print("Copy To Device", p.n_reps);
    timer.print("Kernel", p.n_reps);
    timer.print("Copy Back and Merge", p.n_reps);

    // Verify answer
    verify(h_histo, h_data_pool, p.pool_size, frame_size, p.n_bins);

    // Free memory
    timer.start("Deallocation");
    clStatus = clReleaseMemObject(d_task_queues);
    clStatus = clReleaseMemObject(d_data_queues);
    clStatus = clReleaseMemObject(d_histo_queues);
    clStatus = clReleaseMemObject(d_consumed);
    CL_ERR();
    free(h_consumed);
    free(h_task_queues);
    free(h_data_queues);
    free(h_histo_queues);
    free(h_task_pool);
    free(h_data_pool);
    free(h_histo);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
