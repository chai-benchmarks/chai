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
#include "support/partitioner.h"

// Device auxiliary functions
__device__ void reduce(int *l_count, int local_cnt, int *l_data) {
    const int tid       = threadIdx.x;
    const int localSize = blockDim.x;
    // load shared mem
    l_data[tid] = local_cnt;
    __syncthreads();
    // do reduction in shared mem
    for(int s = localSize >> 1; s > 0; s >>= 1) {
        if(tid < s) {
            l_data[tid] += l_data[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if(tid == 0)
        *l_count = l_data[0];
}

__device__ int block_binary_prefix_sums(int *l_count, int x, int *l_data) {
    l_data[threadIdx.x] = x;
    const int length     = blockDim.x;
    // Build up tree
    int offset = 1;
    for(int l = length >> 1; l > 0; l >>= 1) {
        __syncthreads();
        if(threadIdx.x < l) {
            int ai = offset * (2 * threadIdx.x + 1) - 1;
            int bi = offset * (2 * threadIdx.x + 2) - 1;
            l_data[bi] += l_data[ai];
        }
        offset <<= 1;
    }
    if(offset < length) {
        offset <<= 1;
    }
    // Build down tree
    int maxThread = offset >> 1;
    for(int d = 0; d < maxThread; d <<= 1) {
        d += 1;
        offset >>= 1;
        __syncthreads();
        if(threadIdx.x < d) {
            int ai = offset * (threadIdx.x + 1) - 1;
            int bi = ai + (offset >> 1);
            l_data[bi] += l_data[ai];
        }
    }
    __syncthreads();
    int output = l_data[threadIdx.x] + *l_count - x;
    __syncthreads();
    if(threadIdx.x == blockDim.x - 1)
        *l_count += l_data[threadIdx.x];

    return output;
}

// HIP kernel ------------------------------------------------------------------------------------------
__global__ void StreamCompaction_kernel(int size, T value, int n_tasks, float alpha, T *output, T *input,
    int *flags
#ifdef CUDA_8_0
    , int *worklist
#endif
    ) {

    HIP_DYNAMIC_SHARED( int, l_mem)
    int* l_data = l_mem;
    int* l_count = &l_data[blockDim.x];
#ifdef CUDA_8_0
    int* l_tmp = &l_count[1];
#endif

#ifdef CUDA_8_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif

    for(int my_s = gpu_first(&p); gpu_more(&p); my_s = gpu_next(&p)) {

        if(threadIdx.x == 0) {
            l_count[0] = 0;
        }
        __syncthreads();

        int local_cnt = 0;
        // Declare on-chip memory
        T reg[REGS];
#ifdef CUDA_8_0
        int pos = my_s * REGS * blockDim.x + threadIdx.x;
#else
        int pos = (my_s - p.cut) * REGS * blockDim.x + threadIdx.x;
#endif
// Load in on-chip memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            if(pos < size) {
                reg[j] = input[pos];
                if(reg[j] != value)
                    local_cnt++;
            } else
                reg[j] = value;
            pos += blockDim.x;
        }
        reduce(&l_count[0], local_cnt, &l_data[0]);

        // Set global synch
        if(threadIdx.x == 0) {
            int p_count;
#ifdef CUDA_8_0
            while((p_count = atomicAdd(&flags[my_s], 0)) == 0) { //atomicAdd_system(&flags[my_s], 0))
            }
            atomicAdd(&flags[my_s + 1], p_count + l_count[0]); //atomicAdd_system(&flags[my_s + 1], p_count + l_count[0]);
#else
            while((p_count = atomicAdd(&flags[my_s], 0)) == 0) {
            }
            atomicAdd(&flags[my_s + 1], p_count + l_count[0]);
#endif
            l_count[0] = p_count - 1;
        }
        __syncthreads();

// Store to global memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            pos = block_binary_prefix_sums(&l_count[0], (int)((reg[j] != value) ? 1 : 0), &l_data[0]);
            if(reg[j] != value) {
                output[pos] = reg[j];
            }
        }
    }
}

hipError_t call_StreamCompaction_kernel(int blocks, int threads, int size, T value, int n_tasks, float alpha, 
    T *output, T *input, int *flags, int l_mem_size
#ifdef CUDA_8_0
    , int *worklist
#endif
    ){
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    hipLaunchKernelGGL(StreamCompaction_kernel, dim3(dimGrid), dim3(dimBlock), l_mem_size, 0, size, value, n_tasks, alpha,
        output, input, flags
#ifdef CUDA_8_0
        , worklist
#endif
        );
    hipError_t err = hipGetLastError();
    return err;
}
