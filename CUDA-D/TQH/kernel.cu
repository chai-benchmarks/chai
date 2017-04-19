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
__global__ void TQHistogram_gpu(task_t *queue, int *data, int *histo, int offset,
    int gpuQueueSize, int *consumed, int frame_size, int n_bins) {

    extern __shared__ int l_mem[];
    int* next = l_mem;
    task_t* t = (task_t*)&next[1];
    int* l_histo = (int*)&t[1];

    const int tid       = threadIdx.x;
    const int tileid    = blockIdx.x;
    const int tile_size = blockDim.x;

    // Fetch task
    if(tid == 0) {
        *next = atomicAdd(consumed, 1);
        t->id = queue[*next].id;
        t->op = queue[*next].op;
    }
    __syncthreads();

    while(*next < gpuQueueSize) {
        // Compute task
        if(t->op == SIGNAL_WORK_KERNEL) {
            // Reset local histogram
            for(int i = tid; i < n_bins; i += tile_size) {
                l_histo[i] = 0;
            }
            __syncthreads();

            for(int i = tid; i < frame_size; i += tile_size) {
                int value = (data[(t->id - offset) * frame_size + i] * n_bins) >> 8;

                atomicAdd(&l_histo[value], 1);
            }

            __syncthreads();
            // Store in global memory
            for(int i = tid; i < n_bins; i += tile_size) {
                histo[(t->id - offset) * n_bins + i] = l_histo[i];
            }
        }

        if(tid == 0) {
            *next = atomicAdd(consumed, 1);
            // Fetch task
            t->id = queue[*next].id;
            t->op = queue[*next].op;
        }
        __syncthreads();
    }
}

cudaError_t call_TQHistogram_gpu(int blocks, int threads, task_t *queue, int *data, int *histo, 
    int offset, int gpuQueueSize, int *consumed, int frame_size, int n_bins, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    TQHistogram_gpu<<<dimGrid, dimBlock, l_mem_size>>>(queue, data, histo, 
        offset, gpuQueueSize, consumed, frame_size, n_bins);
    
    cudaError_t err = cudaGetLastError();
    return err;
}
