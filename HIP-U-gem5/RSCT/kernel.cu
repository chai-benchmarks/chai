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

// CUDA heterogeneous kernel ------------------------------------------------------------------------------------------
__global__ void RANSAC_kernel_block(float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int max_iter, int error_threshold, float convergence_threshold,
    int *g_out_id, int *model_candidate, int *outliers_candidate, int *launch_gpu) {

    extern __shared__ int l_mem[];
    int* outlier_block_count = l_mem;

    const int tx         = threadIdx.x;
    const int bx         = blockIdx.x;
    const int num_blocks = gridDim.x;

    float vx_error, vy_error;
    int   outlier_local_count = 0;

    // Each block performs one iteration
    for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

        float *model_param =
            &model_param_local
                [4 *
                    loop_count]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
        // Wait until CPU computes F-o-F model
        if(tx == 0) {
            outlier_block_count[0] = 0;
            while(atomicAdd(&launch_gpu[loop_count], 0) == 0) { //atomicAdd_system(&launch_gpu[loop_count], 0)
            }
        }
        __syncthreads();

        if(model_param[0] == -2011)
            continue;

        // Reset local outlier counter
        outlier_local_count = 0;

        // Compute number of outliers
        for(int i = tx; i < flowvector_count; i += blockDim.x) {
            flowvector fvreg = flowvectors[i]; // x, y, vx, vy
            vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
                                     (int)((fvreg.y - model_param[1]) * model_param[3])) -
                       fvreg.vx;
            vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
                                     (int)((fvreg.x - model_param[0]) * model_param[3])) -
                       fvreg.vy;
            if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
                outlier_local_count++;
            }
        }

        atomicAdd(&outlier_block_count[0], outlier_local_count);

        __syncthreads();
        if(tx == 0) {
            // Compare to threshold
            if(outlier_block_count[0] < flowvector_count * convergence_threshold) {
                int index                 = atomicAdd(g_out_id, 1);
                model_candidate[index]    = loop_count;
                outliers_candidate[index] = outlier_block_count[0];
            }
        }
    }
}

cudaError_t call_RANSAC_kernel_block(int blocks, int threads, float *model_param_local, flowvector *flowvectors, 
    int flowvector_count, int *random_numbers, int max_iter, int error_threshold, float convergence_threshold,
    int *g_out_id, int *model_candidate, int *outliers_candidate, int *launch_gpu, int l_mem_size){
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    RANSAC_kernel_block<<<dimGrid, dimBlock, l_mem_size>>>(model_param_local, flowvectors, 
    flowvector_count, random_numbers, max_iter, error_threshold, convergence_threshold,
    g_out_id, model_candidate, outliers_candidate, launch_gpu);
    cudaError_t err = cudaGetLastError();
    return err;
}
