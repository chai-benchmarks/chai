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

// BezierBlend (http://paulbourke.net/geometry/bezier/)
T BezierBlendGPU(int k, T mu, int n) {
    int nn, kn, nkn;
    T   blend = 1;
    nn        = n;
    kn        = k;
    nkn       = n - k;
    while(nn >= 1) {
        blend *= nn;
        nn--;
        if(kn > 1) {
            blend /= (T)kn;
            kn--;
        }
        if(nkn > 1) {
            blend /= (T)nkn;
            nkn--;
        }
    }
    if(k > 0)
        blend *= pow(mu, (T)k);
    if(n - k > 0)
        blend *= pow(1 - mu, (T)(n - k));
    return (blend);
}

// OpenCL kernel --------------------------------------------------------------
__kernel void Bezier_surface(int n_tasks, float alpha, int in_size_i, int in_size_j, int out_size_i,
    int out_size_j, __local XYZ *l_in, __global XYZ *in, __global XYZ *outp
#ifdef OCL_2_0
    , __global atomic_int *worklist, __local int *l_tmp
#endif
    ) {

#ifdef OCL_2_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif
    
    const int wg_in_J = divceil(out_size_j, get_local_size(0));
    const int wg_in_I = divceil(out_size_i, get_local_size(1));

    for(int i = get_local_id(1) * get_local_size(0) + get_local_id(0); i < (in_size_i + 1) * (in_size_j + 1);
        i += get_local_size(0) * get_local_size(1))
        l_in[i] = in[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int t = gpu_first(&p); gpu_more(&p); t = gpu_next(&p)) {
        const int my_s1 = t / wg_in_J;
        const int my_s0 = t % wg_in_J;

        int Row = my_s1 * get_local_size(1) + get_local_id(1);
        int Col = my_s0 * get_local_size(0) + get_local_id(0);
        T   bi;
        T   bj;
        T   mui = Row / (T)(out_size_i - 1);
        T   muj = Col / (T)(out_size_j - 1);

        if(Row < out_size_i && Col < out_size_j) {
            XYZ out = {0, 0, 0};
#pragma unroll
            for(int ki = 0; ki <= in_size_i; ki++) {
                bi = BezierBlendGPU(ki, mui, in_size_i);
#pragma unroll
                for(int kj = 0; kj <= in_size_j; kj++) {
                    bj = BezierBlendGPU(kj, muj, in_size_j);
                    out.x += (l_in[ki * (in_size_j + 1) + kj].x * bi * bj);
                    out.y += (l_in[ki * (in_size_j + 1) + kj].y * bi * bj);
                    out.z += (l_in[ki * (in_size_j + 1) + kj].z * bi * bj);
                }
            }
            outp[Row * out_size_j + Col] = out;
        }
    }
}
