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

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_gpu_threads;
    int         n_gpu_blocks;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    int         pattern;
    int         pool_size;
    int         queue_size;
    int         iterations;

    Params(int argc, char **argv) {
        device        = 0;
        n_gpu_threads  = 64;
        n_gpu_blocks = 320;
        n_threads     = 1;
        n_warmup      = 2;
        n_reps        = 10;
        file_name     = "input/patternsNP100NB512FB25.txt";
        pattern       = 1;
        pool_size     = 3200;
        queue_size    = 320;
        iterations    = 50;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:k:s:q:n:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device        = atoi(optarg); break;
            case 'i': n_gpu_threads  = atoi(optarg); break;
            case 'g': n_gpu_blocks = atoi(optarg); break;
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
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./tq [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=64)"
                "\n    -g <G>    # of device blocks (default=320)"
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
    CUDASetup    setcuda(p.device);
    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    int *   pattern = (int *)malloc(p.pool_size * sizeof(int));
    task_t *task_pool;
    cudaStatus = cudaMallocManaged(&task_pool, p.pool_size * sizeof(task_t));
    int *data;
    cudaStatus = cudaMallocManaged(&data, p.pool_size * p.n_gpu_threads * sizeof(int));
    task_t *task_queues;
    cudaStatus = cudaMallocManaged(&task_queues, NUM_TASK_QUEUES * p.queue_size * sizeof(task_t));
    std::atomic_int *n_tasks_in_queue;
    cudaStatus = cudaMallocManaged(&n_tasks_in_queue, sizeof(std::atomic_int) * NUM_TASK_QUEUES);
    std::atomic_int *n_written_tasks;
    cudaStatus = cudaMallocManaged(&n_written_tasks, sizeof(std::atomic_int) * NUM_TASK_QUEUES);
    std::atomic_int *n_consumed_tasks;
    cudaStatus = cudaMallocManaged(&n_consumed_tasks, sizeof(std::atomic_int) * NUM_TASK_QUEUES);
    task_t *task_pool_backup = (task_t *)malloc(p.pool_size * sizeof(task_t));
    cudaDeviceSynchronize();
    CUDA_ERR();
    ALLOC_ERR(pattern, task_pool_backup);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_gpu_threads = setcuda.max_gpu_threads();
    read_input(pattern, task_pool, p);
    memset((void *)data, 0, p.pool_size * p.n_gpu_threads * sizeof(int));
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
        memset((void *)data, 0, p.pool_size * p.n_gpu_threads * sizeof(int));
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
            p.pool_size, p.n_gpu_blocks);

        // Kernel launch
        assert(p.n_gpu_threads <= max_gpu_threads && 
            "The thread block size is greater than the maximum thread block size that can be used on this device");
        cudaStatus = call_TaskQueue_gpu(p.n_gpu_blocks, p.n_gpu_threads, task_queues, (int *)n_tasks_in_queue, 
            (int *)n_written_tasks, (int *)n_consumed_tasks, data, p.queue_size, p.iterations, sizeof(int) + sizeof(task_t));
        CUDA_ERR();

        cudaDeviceSynchronize();
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

    // Verify answer
    verify(data, pattern, p.pool_size, p.iterations, p.n_gpu_threads);

    // Free memory
    timer.start("Deallocation");
    cudaStatus = cudaFree(task_queues);
    cudaStatus = cudaFree(n_tasks_in_queue);
    cudaStatus = cudaFree(n_written_tasks);
    cudaStatus = cudaFree(n_consumed_tasks);
    cudaStatus = cudaFree(task_pool);
    cudaStatus = cudaFree(data);
    free(pattern);
    free(task_pool_backup);
    CUDA_ERR();
    cudaDeviceSynchronize();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    // Release timers
    timer.release("Allocation");
    timer.release("Initialization");
    timer.release("Kernel");
    timer.release("Deallocation");

    printf("Test Passed\n");
    return 0;
}
