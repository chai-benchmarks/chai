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

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#include "support/common.h"
#include "support/partitioner.h"

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void Histogram_kernel(int size, int bins, int n_tasks, float alpha, __global unsigned int *data,
#ifdef OCL_2_0
    __global atomic_uint *histo, __local atomic_uint *l_histo, __global atomic_int *worklist, __local int *l_tmp
#else
    __global unsigned int *histo, __local unsigned int *l_histo
#endif
    ) {
    
#ifdef OCL_2_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif
    
    // Block and thread index
    const int bx = get_group_id(0);
    const int tx = get_local_id(0);
    const int bD = get_local_size(0);
    const int gD = get_num_groups(0);

    // Sub-histograms initialization
    for(int pos = tx; pos < bins; pos += bD) {
#ifdef OCL_2_0
        atomic_store(&l_histo[pos], 0);
#else
        l_histo[pos] = 0;
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE); // Intra-block synchronization

    // Main loop
    for(int i = gpu_first(&p); gpu_more(&p); i = gpu_next(&p)) {
    
        // Global memory read
        unsigned int d = data[i * bD + tx];

// Atomic vote in shared memory
#ifdef OCL_2_0
        atomic_fetch_add(&l_histo[((d * bins) >> 12)], 1);
#else
        atomic_add(&l_histo[((d * bins) >> 12)], 1);
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE); // Intra-block synchronization

    // Merge per-block histograms and write to global memory
    for(int pos = tx; pos < bins; pos += bD) {
// Atomic addition in global memory
#ifdef OCL_2_0
        atomic_fetch_add(histo + pos, (unsigned int)atomic_load(&l_histo[pos]));
#else
        atomic_add(histo + pos, l_histo[pos]);
#endif
    }
}
