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

// CUDA baseline kernel ------------------------------------------------------------------------------------------
// Generate model on GPU side
__device__ int gen_model_paramGPU(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param) {
    float temp;
    // xc -> model_param[0], yc -> model_param[1], D -> model_param[2], R -> model_param[3]
    temp = (float)((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[0] = (((vx1 * ((-vx2 * x1) + (vx1 * x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                          (vy1 * ((-vy2 * x1) + (vy1 * x2) - (vy2 * x2) - (vx2 * y1) + (vx2 * y2))) +
                          (x1 * ((vy2 * vy2) + (vx2 * vx2)))) /
                      temp);
    model_param[1] = (((vx2 * ((vy1 * x1) - (vy1 * x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                          (vy2 * ((-vx1 * x1) + (vx1 * x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                          (y2 * ((vx1 * vx1) + (vy1 * vy1)))) /
                      temp);

    temp = (float)((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[2] = ((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp);
    model_param[3] = ((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp);
    return (1);
}

__global__ void RANSAC_kernel_block(int flowvector_count, int max_iter, int error_threshold, float convergence_threshold,
    int n_tasks, float alpha, float *model_param_local, flowvector *flowvectors,
    int *random_numbers, int *model_candidate, int *outliers_candidate, 
    int *g_out_id
    , int *worklist
    ) {

    HIP_DYNAMIC_SHARED( int, l_mem)
    int* outlier_block_count = l_mem;
    int* l_tmp = &outlier_block_count[1];
    
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
    
    const int tx         = threadIdx.x;
    const int bx         = blockIdx.x;
    const int num_blocks = gridDim.x;

    float vx_error, vy_error;
    int   outlier_local_count = 0;

    // Each block performs one iteration
    for(int iter = gpu_first(&p); gpu_more(&p); iter = gpu_next(&p)) {

        float *model_param =
            &model_param_local
                [4 * iter]; // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
        // Thread 0 computes F-o-F model (SISD phase)
        if(tx == 0) {
            outlier_block_count[0] = 0;
            // Select two random flow vectors
            int        rand_num = random_numbers[iter * 2 + 0];
            flowvector fv[2];
            fv[0]    = flowvectors[rand_num];
            rand_num = random_numbers[iter * 2 + 1];
            fv[1]    = flowvectors[rand_num];

            int ret = 0;
            int vx1 = fv[0].vx - fv[0].x;
            int vy1 = fv[0].vy - fv[0].y;
            int vx2 = fv[1].vx - fv[1].x;
            int vy2 = fv[1].vy - fv[1].y;

            // Function to generate model parameters according to F-o-F (xc, yc, D and R)
            ret = gen_model_paramGPU(fv[0].x, fv[0].y, vx1, vy1, fv[1].x, fv[1].y, vx2, vy2, model_param);
            if(ret == 0)
                model_param[0] = -2011;
        }
        __syncthreads();

        if(model_param[0] == -2011)
            continue;

        // SIMD phase
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
        atomicAdd(&outlier_block_count[0], outlier_local_count); //atomicAdd_system(&outlier_block_count[0], outlier_local_count);

        __syncthreads();
        if(tx == 0) {
            // Compare to threshold
            if(atomicAdd(&outlier_block_count[0], 0) < flowvector_count * convergence_threshold) { //atomicAdd_system(&outlier_block_count[0], 0)
                int index                 = atomicAdd(g_out_id, 1); //atomicAdd_system(g_out_id, 1);
                outliers_candidate[index] = atomicAdd(&outlier_block_count[0], 0); //atomicAdd_system(&outlier_block_count[0], 0);
                model_candidate[index] = iter;
            }
        }
    }
}

hipError_t call_RANSAC_kernel_block(int blocks, int threads, int flowvector_count, int max_iter, int error_threshold, 
    float convergence_threshold, int n_tasks, float alpha, float *model_param_local, flowvector *flowvectors,
    int *random_numbers, int *model_candidate, int *outliers_candidate, 
    int *g_out_id, int l_mem_size
    , int *worklist
    ){
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    hipLaunchKernelGGL(RANSAC_kernel_block, dim3(dimGrid), dim3(dimBlock), l_mem_size, 0, flowvector_count, max_iter, error_threshold, 
        convergence_threshold, n_tasks, alpha, model_param_local, flowvectors,
        random_numbers, model_candidate, outliers_candidate, 
        g_out_id
        , worklist
        );
    hipError_t err = hipGetLastError();
    return err;
}
