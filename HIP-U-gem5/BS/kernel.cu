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

// BezierBlend (http://paulbourke.net/geometry/bezier/)
__device__ T BezierBlendGPU(int k, T mu, int n) {
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

// CUDA kernel --------------------------------------------------------------
__global__ void Bezier_surface(int n_tasks, float alpha, int in_size_i, int in_size_j, int out_size_i,
    int out_size_j, XYZ *in, XYZ *outp
#ifdef CUDA_8_0
    , int *worklist
#endif
    ) {
      
    HIP_DYNAMIC_SHARED( XYZ, l_mem)
    XYZ* l_in = l_mem;
    int* l_tmp = (int*)&l_in[(in_size_i+1)*(in_size_j+1)];

#ifdef CUDA_8_0
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif

    const int wg_in_J = divceil(out_size_j, blockDim.x);
    const int wg_in_I = divceil(out_size_i, blockDim.y);

    for(int i = threadIdx.y * blockDim.x + threadIdx.x; i < (in_size_i + 1) * (in_size_j + 1);
        i += blockDim.x * blockDim.y){
        l_in[i] = in[i];
    }
    __syncthreads();

    for(int t = gpu_first(&p); gpu_more(&p); t = gpu_next(&p)) {
        const int my_s1 = t / wg_in_J;
        const int my_s0 = t % wg_in_J;

        int Row = my_s1 * blockDim.y + threadIdx.y;
        int Col = my_s0 * blockDim.x + threadIdx.x;
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


hipError_t call_Bezier_surface(int blocks, int threads, int n_tasks, float alpha,
    int in_size_i, int in_size_j, int out_size_i, int out_size_j,
    int l_mem_size, XYZ* d_in, XYZ* d_out
#ifdef CUDA_8_0
    , int* worklist
#endif
    ){
    dim3 dimGrid(blocks, 1);
    dim3 dimBlock(threads, threads);
    hipLaunchKernelGGL(Bezier_surface, dim3(dimGrid), dim3(dimBlock), l_mem_size, 0, n_tasks, alpha, in_size_i, in_size_j, out_size_i, out_size_j,
        d_in, d_out
#ifdef CUDA_8_0
        , worklist
#endif
        );
    hipError_t err = hipGetLastError();
    return err;
}
