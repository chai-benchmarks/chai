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
    int     frame_size = p.n * p.m;
    task_t *task_pool =
        (task_t *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.pool_size * sizeof(task_t), 0);
    task_t *task_queues = (task_t *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        NUM_TASK_QUEUES * p.queue_size * sizeof(task_t), 0);
    int *data_pool =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.pool_size * frame_size * sizeof(int), 0);
    std::atomic_int *histo = (std::atomic_int *)clSVMAlloc(ocl.clContext,
        CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * p.pool_size * p.n_bins, 0);
    std::atomic_int *n_task_in_queue = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * NUM_TASK_QUEUES, 0);
    std::atomic_int *n_written_tasks = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * NUM_TASK_QUEUES, 0);
    std::atomic_int *n_consumed_tasks = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * NUM_TASK_QUEUES, 0);
    task_t *task_pool_backup = (task_t *)malloc(p.pool_size * sizeof(task_t));
    clFinish(ocl.clCommandQueue);
    ALLOC_ERR(task_pool, task_queues, data_pool, histo);
    ALLOC_ERR(n_task_in_queue, n_written_tasks, n_consumed_tasks, task_pool_backup);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(data_pool, task_pool, p);
    for(int i = 0; i < p.pool_size * p.n_bins; i++) {
        histo[i].store(0);
    }
    for(int i = 0; i < NUM_TASK_QUEUES; i++) {
        n_task_in_queue[i].store(0);
    }
    for(int i = 0; i < NUM_TASK_QUEUES; i++) {
        n_written_tasks[i].store(0);
    }
    for(int i = 0; i < NUM_TASK_QUEUES; i++) {
        n_consumed_tasks[i].store(0);
    }
    timer.stop("Initialization");
    timer.print("Initialization", 1);
		memcpy(task_pool_backup, task_pool, p.pool_size * sizeof(task_t));

    for(int y = 0; y < p.n_reps + p.n_warmup; y++) {

        // Reset
        memcpy(task_pool, task_pool_backup, p.pool_size * sizeof(task_t));
        for(int i = 0; i < p.pool_size * p.n_bins; i++) {
            histo[i].store(0);
        }
        for(int i = 0; i < NUM_TASK_QUEUES; i++) {
            n_task_in_queue[i].store(0);
        }
        for(int i = 0; i < NUM_TASK_QUEUES; i++) {
            n_written_tasks[i].store(0);
        }
        for(int i = 0; i < NUM_TASK_QUEUES; i++) {
            n_consumed_tasks[i].store(0);
        }
        int num_tasks  = p.queue_size;
        int last_queue = 0;
        int offset     = 0;

        if(y >= p.n_warmup)
            timer.start("Kernel");

        std::thread main_thread(run_cpu_threads, p.n_threads, task_queues, n_task_in_queue, n_written_tasks,
            n_consumed_tasks, task_pool, data_pool, p.queue_size, &offset, &last_queue, &num_tasks, p.queue_size, p.pool_size,
            p.n_work_groups);

        clSetKernelArgSVMPointer(ocl.clKernel, 0, task_queues);
        clSetKernelArgSVMPointer(ocl.clKernel, 1, n_task_in_queue);
        clSetKernelArgSVMPointer(ocl.clKernel, 2, n_written_tasks);
        clSetKernelArgSVMPointer(ocl.clKernel, 3, n_consumed_tasks);
        clSetKernelArgSVMPointer(ocl.clKernel, 4, histo);
        clSetKernelArgSVMPointer(ocl.clKernel, 5, data_pool);
        clSetKernelArg(ocl.clKernel, 6, sizeof(int), &p.queue_size);
        clSetKernelArg(ocl.clKernel, 7, sizeof(task_t), NULL);
        clSetKernelArg(ocl.clKernel, 8, sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 9, p.n_bins * sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 10, sizeof(int), &frame_size);
        clSetKernelArg(ocl.clKernel, 11, sizeof(int), &p.n_bins);

        // Kernel launch
        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
        assert(ls[0] <= max_wi && 
            "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
        clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
        CL_ERR();

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(y >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

    // Verify answer
    verify(histo, data_pool, p.pool_size, frame_size, p.n_bins);

    // Free memory
    timer.start("Deallocation");
    clSVMFree(ocl.clContext, task_queues);
    clSVMFree(ocl.clContext, n_task_in_queue);
    clSVMFree(ocl.clContext, n_written_tasks);
    clSVMFree(ocl.clContext, n_consumed_tasks);
    clSVMFree(ocl.clContext, task_pool);
    clSVMFree(ocl.clContext, data_pool);
    clSVMFree(ocl.clContext, histo);
    free(task_pool_backup);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
