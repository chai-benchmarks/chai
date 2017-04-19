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
    int         pattern;
    int         pool_size;
    int         queue_size;
    int         iterations;

    Params(int argc, char **argv) {
        platform      = 0;
        device        = 0;
        n_work_items  = 64;
        n_work_groups = 320;
        n_threads     = 1;
        n_warmup      = 2;
        n_reps        = 10;
        file_name     = "input/patternsNP100NB512FB25.txt";
        pattern       = 1;
        pool_size     = 3200;
        queue_size    = 320;
        iterations    = 50;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:f:k:s:q:n:")) >= 0) {
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
            case 'k': pattern       = atoi(optarg); break;
            case 's': pool_size     = atoi(optarg); break;
            case 'q': queue_size    = atoi(optarg); break;
            case 'n': iterations    = atoi(optarg); break;
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
                "\nUsage:  ./tq [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=64)"
                "\n    -g <G>    # of device work-groups (default=320)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    patterns file name (default=input/patternsNP100NB512FB25.txt)"
                "\n    -k <K>    pattern in file (default=1)"
                "\n    -s <S>    task pool size (default=3200)"
                "\n    -q <Q>    task queue size (default=320)"
                "\n    -n <N>    # of iterations in heavy task (default=50)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(int *pattern, task_t *task_pool, const Params &p) {

    // Patterns file name
    char filePatterns[100];

    sprintf(filePatterns, "%s", p.file_name);

    // Read line from patterns file
    FILE *File;
    int r;
    if((File = fopen(filePatterns, "rt")) != NULL) {
        for(int y = 0; y <= p.pattern; y++) {
            for(int x = 0; x < 512; x++) {
                fscanf(File, "%d ", &r);
                pattern[x] = r;
            }
        }
        fclose(File);
    } else {
        printf("Unable to open file %s\n", filePatterns);
        exit(-1);
    }

    for(int i = 0; i < p.pool_size; i++) {
        //Setting tasks in the tasks pool
        task_pool[i].id = i;
        task_pool[i].op = SIGNAL_NOTWORK_KERNEL;
    }

    //Read the pattern
    for(int i = 0; i < p.pool_size; i++) {
        pattern[i] = pattern[i%512];
        if(pattern[i] == 1) {
            task_pool[i].op = SIGNAL_WORK_KERNEL;
        }
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
    int *   pattern = (int *)malloc(p.pool_size * sizeof(int));
    task_t *task_pool =
        (task_t *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.pool_size * sizeof(task_t), 0);
    int *data =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.pool_size * p.n_work_items * sizeof(int), 0);
    task_t *task_queues = (task_t *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS,
        NUM_TASK_QUEUES * p.queue_size * sizeof(task_t), 0);
    std::atomic_int *n_tasks_in_queue = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * NUM_TASK_QUEUES, 0);
    std::atomic_int *n_written_tasks = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * NUM_TASK_QUEUES, 0);
    std::atomic_int *n_consumed_tasks = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * NUM_TASK_QUEUES, 0);
    task_t *task_pool_backup = (task_t *)malloc(p.pool_size * sizeof(task_t));
    clFinish(ocl.clCommandQueue);
    ALLOC_ERR(pattern, task_pool, data, task_queues);
    ALLOC_ERR(n_tasks_in_queue, n_written_tasks, n_consumed_tasks, task_pool_backup);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    read_input(pattern, task_pool, p);
    memset((void *)data, 0, p.pool_size * p.n_work_items * sizeof(int));
    for(int i = 0; i < NUM_TASK_QUEUES; i++) {
        n_tasks_in_queue[i].store(0);
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

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        memcpy(task_pool, task_pool_backup, p.pool_size * sizeof(task_t));
        memset((void *)data, 0, p.pool_size * p.n_work_items * sizeof(int));
        for(int i = 0; i < NUM_TASK_QUEUES; i++) {
            n_tasks_in_queue[i].store(0);
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

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        std::thread main_thread(run_cpu_threads, p.n_threads, task_queues, n_tasks_in_queue, n_written_tasks,
            n_consumed_tasks, task_pool, data, p.queue_size, &offset, &last_queue, &num_tasks, p.queue_size,
            p.pool_size, p.n_work_groups);

        // Setting kernel arguments
        clSetKernelArgSVMPointer(ocl.clKernel, 0, task_queues);
        clSetKernelArgSVMPointer(ocl.clKernel, 1, n_tasks_in_queue);
        clSetKernelArgSVMPointer(ocl.clKernel, 2, n_written_tasks);
        clSetKernelArgSVMPointer(ocl.clKernel, 3, n_consumed_tasks);
        clSetKernelArgSVMPointer(ocl.clKernel, 4, data);
        clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.queue_size);
        clSetKernelArg(ocl.clKernel, 6, sizeof(int), &p.iterations);
        clSetKernelArg(ocl.clKernel, 7, sizeof(task_t), NULL);
        clSetKernelArg(ocl.clKernel, 8, sizeof(int), NULL);

        // Kernel launch
        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_groups * p.n_work_items};
        assert(ls[0] <= max_wi && 
            "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
        clStatus     = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
        CL_ERR();

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

    // Verify answer
    verify(data, pattern, p.pool_size, p.iterations, p.n_work_items);

    // Free memory
    timer.start("Deallocation");
    clSVMFree(ocl.clContext, task_queues);
    clSVMFree(ocl.clContext, n_tasks_in_queue);
    clSVMFree(ocl.clContext, n_written_tasks);
    clSVMFree(ocl.clContext, n_consumed_tasks);
    clSVMFree(ocl.clContext, task_pool);
    clSVMFree(ocl.clContext, data);
    free(pattern);
    free(task_pool_backup);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
