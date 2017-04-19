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

#define _OPENCL_COMPILER_

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#include "support/common.h"
#include "support/partitioner.h"

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void Padding_kernel(int n, int m, int pad, int n_tasks, float alpha, __global T *matrix_out, __global T *matrix,
#ifdef OCL_2_0
    __global atomic_int *flags, __global atomic_int *worklist, __local int *l_tmp
#else
    __global int *flags
#endif
    ) {

#ifdef OCL_2_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif

    const int matrix_size = m * (n + pad);
    const int matrix_size_align =
        (matrix_size + get_local_size(0) * REGS - 1) / (get_local_size(0) * REGS) * (get_local_size(0) * REGS);

    for(int my_s = gpu_first(&p); gpu_more(&p); my_s = gpu_next(&p)) {

        // Declare on-chip memory
        T   reg[REGS];
        int pos      = matrix_size_align - 1 - (my_s * REGS * get_local_size(0) + get_local_id(0));
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
            pos -= get_local_size(0);
            my_s_row = pos / (n + pad);
            my_x     = pos % (n + pad);
            pos2     = my_s_row * n + my_x;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Set global synch
        if(get_local_id(0) == 0) {
#ifdef OCL_2_0
            while(atomic_load(&flags[my_s]) == 0) {
            }
            atomic_fetch_add(&flags[my_s + 1], 1);
#else
            while(atomic_add(&flags[my_s], 0) == 0) {
            }
            atomic_add(&flags[my_s + 1], 1);
#endif
        }
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

        pos = matrix_size_align - 1 - (my_s * REGS * get_local_size(0) + get_local_id(0));
// Store to global memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            if(pos >= 0 && pos < matrix_size)
                matrix_out[pos] = reg[j];
            pos -= get_local_size(0);
        }
    }
}
