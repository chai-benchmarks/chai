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

// HIP kernel ------------------------------------------------------------------------------------------
__global__ void Padding_kernel(int n, int m, int pad, int n_tasks, float alpha, T *matrix_out, T *matrix,
    int *flags
#ifdef CUDA_8_0
    , int *worklist
#endif
    ) {

#ifdef CUDA_8_0
    HIP_DYNAMIC_SHARED( int, l_mem)
    int* l_tmp = l_mem;
#endif

#ifdef CUDA_8_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif

    const int matrix_size = m * (n + pad);
    const int matrix_size_align =
        (matrix_size + blockDim.x * REGS - 1) / (blockDim.x * REGS) * (blockDim.x * REGS);

    for(int my_s = gpu_first(&p); gpu_more(&p); my_s = gpu_next(&p)) {

        // Declare on-chip memory
        T   reg[REGS];
        int pos      = matrix_size_align - 1 - (my_s * REGS * blockDim.x + threadIdx.x);
        int my_s_row = pos / (n + pad);
        int my_x     = pos % (n + pad);
        int pos2     = my_s_row * n + my_x;
// Load in on-chip memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            if(pos2 >= 0 && my_x < n && pos2 < matrix_size)
                reg[j] = matrix[pos2];
            else
                reg[j] = 0;
            pos -= blockDim.x;
            my_s_row = pos / (n + pad);
            my_x     = pos % (n + pad);
            pos2     = my_s_row * n + my_x;
        }

        __syncthreads();

        // Set global synch
        if(threadIdx.x == 0) {
#ifdef CUDA_8_0
            while(atomicAdd(&flags[my_s], 0) == 0) { //atomicAdd_system(&flags[my_s], 0)
            }
            atomicAdd(&flags[my_s + 1], 1); //atomicAdd_system(&flags[my_s + 1], 1);
#else
            while(atomicAdd(&flags[my_s], 0) == 0) {
            }
            atomicAdd(&flags[my_s + 1], 1);
#endif
        }
        __syncthreads();

        pos = matrix_size_align - 1 - (my_s * REGS * blockDim.x + threadIdx.x);
// Store to global memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            if(pos >= 0 && pos < matrix_size)
                matrix_out[pos] = reg[j];
            pos -= blockDim.x;
        }
    }
}

hipError_t call_Padding_kernel(int blocks, int threads, int n, int m, int pad, int n_tasks, float alpha, 
    T *matrix_out, T *matrix, int *flags
#ifdef CUDA_8_0
    , int l_mem_size, int *worklist
#endif
    ){
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    Padding_kernel<<<dimGrid, dimBlock
#ifdef CUDA_8_0
        , l_mem_size
#endif
        >>>(n, m, pad, n_tasks, alpha, 
        matrix_out, matrix, flags
#ifdef CUDA_8_0
        , worklist
#endif
        );
    hipError_t err = hipGetLastError();
    return err;
}
