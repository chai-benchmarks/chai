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
        n_gpu_threads = 64;
        n_gpu_blocks  = 320;
        n_threads     = 1;
				n_warmup      = 0;
				n_reps        = 1;
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
    cudaError_t  cudaStatus;

    // Allocate
    int     frame_size    = p.n * p.m;
    task_t *h_task_pool   = (task_t *)malloc(p.pool_size * sizeof(task_t));
    task_t *h_task_queues = (task_t *)malloc(p.queue_size * sizeof(task_t));
    task_t * d_task_queues;
    cudaStatus = cudaMalloc((void**)&d_task_queues, p.queue_size * sizeof(task_t));
    int *   h_data_pool   = (int *)malloc(p.pool_size * frame_size * sizeof(int));
    int *   h_data_queues = (int *)malloc(p.queue_size * frame_size * sizeof(int));
    int * d_data_queues;
    cudaStatus = cudaMalloc((void**)&d_data_queues, p.queue_size * frame_size * sizeof(int));
    int *  h_histo        = (int *)malloc(p.pool_size * p.n_bins * sizeof(int));
    int *  h_histo_queues = (int *)malloc(p.queue_size * p.n_bins * sizeof(int));
    int * d_histo_queues;
    cudaStatus = cudaMalloc((void**)&d_histo_queues, p.queue_size * p.n_bins * sizeof(int));
    int *  h_consumed = (int *)malloc(sizeof(int));
    int * d_consumed;
    cudaStatus = cudaMalloc((void**)&d_consumed, sizeof(int));
    cudaThreadSynchronize();
    CUDA_ERR();
    ALLOC_ERR(h_task_pool, h_task_queues, h_data_pool, h_data_queues, h_histo, h_histo_queues, h_consumed);

    // Initialize
    read_input(h_data_pool, h_task_pool, p);
    memset((void *)h_histo_queues, 0, p.queue_size * p.n_bins * sizeof(int));
    memset((void *)h_consumed, 0, sizeof(int));

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        int n_written_tasks = 0;

        //m5_work_begin(0, 0);

        for(int n_consumed_tasks = 0; n_consumed_tasks < p.pool_size; n_consumed_tasks += p.queue_size) {

            host_insert_tasks(h_task_queues, h_data_queues, h_task_pool, h_data_pool, &n_written_tasks, p.queue_size,
                n_consumed_tasks, frame_size);

            cudaStatus = cudaMemcpy(d_task_queues, h_task_queues, p.queue_size * sizeof(task_t), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(d_data_queues, h_data_queues, p.queue_size * frame_size * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(d_histo_queues, h_histo_queues, p.queue_size * p.n_bins * sizeof(int), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(d_consumed, h_consumed, sizeof(int), cudaMemcpyHostToDevice);
            CUDA_ERR();

            // Kernel launch
            cudaStatus = call_TQHistogram_gpu(p.n_gpu_blocks, p.n_gpu_threads, d_task_queues, d_data_queues, d_histo_queues,
                n_consumed_tasks, p.queue_size, d_consumed, frame_size, p.n_bins, 
                sizeof(int) + sizeof(task_t) + p.n_bins * sizeof(int));
            CUDA_ERR();
            cudaThreadSynchronize();

            cudaStatus = cudaMemcpy(&h_histo[n_consumed_tasks * p.n_bins], d_histo_queues, p.queue_size * p.n_bins * sizeof(int), cudaMemcpyDeviceToHost);
            CUDA_ERR();
        }

        //m5_work_end(0, 0);
    }

    // Verify answer
    verify(h_histo, h_data_pool, p.pool_size, frame_size, p.n_bins);

    // Free memory
    cudaStatus = cudaFree(d_task_queues);
    cudaStatus = cudaFree(d_data_queues);
    cudaStatus = cudaFree(d_histo_queues);
    cudaStatus = cudaFree(d_consumed);
    CUDA_ERR();
    free(h_consumed);
    free(h_task_queues);
    free(h_data_queues);
    free(h_histo_queues);
    free(h_task_pool);
    free(h_data_pool);
    free(h_histo);
    cudaThreadSynchronize();

    printf("Test Passed\n");
    return 0;
}
