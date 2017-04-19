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
    int         pool_size;
    int         queue_size;
    int         m;
    int         n;
    int         n_bins;

    Params(int argc, char **argv) {
        device        = 0;
        n_gpu_threads  = 64;
        n_gpu_blocks = 320;
        n_threads     = 1;
        n_warmup      = 1;
        n_reps        = 10;
        file_name     = "input/basket/basket";
        pool_size     = 3200;
        queue_size    = 320;
        m             = 288;
        n             = 352;
        n_bins        = 256;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:s:q:m:n:b:")) >= 0) {
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
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./tqh [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=64)"
                "\n    -g <G>    # of device blocks (default=320)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=1)"
                "\n    -r <R>    # of timed repetition iterations (default=10)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input video file name (default=input/basket/basket)"
                "\n    -s <S>    task pool size, i.e., # of videos frames (default=3200)"
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
    CUDASetup    setcuda(p.device);
    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    int     frame_size = p.n * p.m;
    task_t *task_pool;
    cudaStatus = cudaMallocManaged(&task_pool, p.pool_size * sizeof(task_t));
    task_t *task_queues;
    cudaStatus = cudaMallocManaged(&task_queues, NUM_TASK_QUEUES * p.queue_size * sizeof(task_t));
    int *data_pool;
    cudaStatus = cudaMallocManaged(&data_pool, p.pool_size * frame_size * sizeof(int));
    std::atomic_int *histo;
    cudaStatus = cudaMallocManaged(&histo, sizeof(std::atomic_int) * p.pool_size * p.n_bins);
    std::atomic_int *n_task_in_queue;
    cudaStatus = cudaMallocManaged(&n_task_in_queue, sizeof(std::atomic_int) * NUM_TASK_QUEUES);
    std::atomic_int *n_written_tasks;
    cudaStatus = cudaMallocManaged(&n_written_tasks, sizeof(std::atomic_int) * NUM_TASK_QUEUES);
    std::atomic_int *n_consumed_tasks;
    cudaStatus = cudaMallocManaged(&n_consumed_tasks, sizeof(std::atomic_int) * NUM_TASK_QUEUES);
    task_t *task_pool_backup = (task_t *)malloc(p.pool_size * sizeof(task_t));
    cudaDeviceSynchronize();
    CUDA_ERR();
    ALLOC_ERR(task_pool_backup);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_gpu_threads = setcuda.max_gpu_threads();
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
            p.n_gpu_blocks);

        // Kernel launch
        assert(p.n_gpu_threads <= max_gpu_threads && 
            "The thread block size is greater than the maximum thread block size that can be used on this device");
        cudaStatus = call_TQHistogram_gpu(p.n_gpu_blocks, p.n_gpu_threads, task_queues, (int*)n_task_in_queue, (int*)n_written_tasks, (int*)n_consumed_tasks,
            (int*)histo, data_pool, p.queue_size, frame_size, p.n_bins, 
            sizeof(int) + sizeof(task_t) + p.n_bins * sizeof(int));
        CUDA_ERR();

        cudaDeviceSynchronize();
        main_thread.join();

        if(y >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

    // Verify answer
    verify(histo, data_pool, p.pool_size, frame_size, p.n_bins);

    // Free memory
    timer.start("Deallocation");
    cudaStatus = cudaFree(task_queues);
    cudaStatus = cudaFree(n_task_in_queue);
    cudaStatus = cudaFree(n_written_tasks);
    cudaStatus = cudaFree(n_consumed_tasks);
    cudaStatus = cudaFree(task_pool);
    cudaStatus = cudaFree(data_pool);
    cudaStatus = cudaFree(histo);
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
