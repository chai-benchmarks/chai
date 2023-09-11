#include "hip/hip_runtime.h"
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

#define _CUDA_COMPILER_

#include "support/common.h"

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void TQHistogram_gpu(task_t *queues, int *n_task_in_queue,
    int *n_written_tasks, int *n_consumed_tasks,
    int *histo, int *data, int gpuQueueSize, int frame_size, int n_bins) {

    HIP_DYNAMIC_SHARED( int, l_mem)
    int* last_queue = l_mem;
    task_t* t = (task_t*)&last_queue[1];
    int* l_histo = (int*)&t[1];
    
    const int tid       = threadIdx.x;
    const int tile_size = blockDim.x;

    while(true) {
        // Fetch task
        if(tid == 0) {
            int  idx_queue = *last_queue;
            int  j, jj;
            bool not_done = true;

            do {
                if(atomicAdd(n_consumed_tasks + idx_queue, 0) == atomicAdd(n_written_tasks + idx_queue, 0)) { //if(atomicAdd_system(n_consumed_tasks + idx_queue, 0) == atomicAdd_system(n_written_tasks + idx_queue, 0)) {
                    idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                } else {
                    if(atomicAdd(n_task_in_queue + idx_queue, 0) > 0) { //atomicAdd_system(n_task_in_queue + idx_queue, 0)
                        j = atomicAdd(n_task_in_queue + idx_queue, -1) - 1; //atomicAdd_system(n_task_in_queue + idx_queue, -1)
                        if(j >= 0) {
                            t->id    = (queues + idx_queue * gpuQueueSize + j)->id;
                            t->op    = (queues + idx_queue * gpuQueueSize + j)->op;
                            jj       = atomicAdd(n_consumed_tasks + idx_queue, 1) + 1; //atomicAdd_system(n_consumed_tasks + idx_queue, 1)
                            not_done = false;
                            if(jj == atomicAdd(n_written_tasks + idx_queue, 0)) { //atomicAdd_system(n_written_tasks + idx_queue, 0)
                                idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                            }
                            *last_queue = idx_queue;
                        } else {
                            idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                        }
                    } else {
                        idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                    }
                }
            } while(not_done);
        }
        __syncthreads(); // It can be removed if work-group = wavefront

        // Compute task
        if(t->op == SIGNAL_STOP_KERNEL) {
            break;
        } else {
            if(t->op == SIGNAL_WORK_KERNEL) {
                // Reset local histogram
                for(int i = tid; i < n_bins; i += tile_size) {
                    l_histo[i] = 0;
                }
                __syncthreads();

                for(int i = tid; i < frame_size; i += tile_size) {
                    int value = (data[t->id * frame_size + i] * n_bins) >> 8;

                    atomicAdd(l_histo + value, 1);
                }

                __syncthreads();
                // Store in global memory
                for(int i = tid; i < n_bins; i += tile_size) {
                    histo[t->id * n_bins + i] = l_histo[i];
                }
            }
        }
    }
}

hipError_t call_TQHistogram_gpu(int blocks, int threads, task_t *queues, int *n_task_in_queue,
    int *n_written_tasks, int *n_consumed_tasks, int *histo, int *data, int gpuQueueSize, 
    int frame_size, int n_bins, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    hipLaunchKernelGGL(TQHistogram_gpu, dim3(dimGrid), dim3(dimBlock), l_mem_size, 0, queues, n_task_in_queue,
        n_written_tasks, n_consumed_tasks, histo, data, gpuQueueSize,
        frame_size, n_bins);
    
    hipError_t err = hipGetLastError();
    return err;
}
