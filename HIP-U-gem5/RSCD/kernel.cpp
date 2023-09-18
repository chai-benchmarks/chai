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

#include "kernel.h"
#include "support/partitioner.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

// Function to generate model parameters for first order flow (xc, yc, D and R)
int gen_model_param(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param) {
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

// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(int *model_candidate, int *outliers_candidate, float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int max_iter, int error_threshold, float convergence_threshold,
    std::atomic_int *g_out_id, int n_threads, int n_tasks, float alpha
#ifdef CUDA_8_0
    , std::atomic_int *worklist
#endif
    ) {

    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {

#ifdef CUDA_8_0
            Partitioner p = partitioner_create(n_tasks, alpha, k, n_threads, worklist);
#else
            Partitioner p = partitioner_create(n_tasks, alpha, k, n_threads);
#endif

            flowvector fv[2];
            float      vx_error, vy_error;
            int        outlier_local_count = 0;

            // Each thread performs one iteration
            for(int iter = cpu_first(&p); cpu_more(&p); iter = cpu_next(&p)) {

                // Obtain model parameters for First Order Flow - gen_firstOrderFlow_model
                float *model_param =
                    &model_param_local
                        [4 * iter]; // xc=model_param[0], yc=model_param[1], D=model_param[2], R=model_param[3]
                // Select two random flow vectors
                int rand_num = random_numbers[iter * 2 + 0];
                fv[0]        = flowvectors[rand_num];
                rand_num     = random_numbers[iter * 2 + 1];
                fv[1]        = flowvectors[rand_num];

                int ret = 0;
                int vx1 = fv[0].vx - fv[0].x;
                int vy1 = fv[0].vy - fv[0].y;
                int vx2 = fv[1].vx - fv[1].x;
                int vy2 = fv[1].vy - fv[1].y;

                // Function to generate model parameters according to F-o-F (xc, yc, D and R)
                ret = gen_model_param(fv[0].x, fv[0].y, vx1, vy1, fv[1].x, fv[1].y, vx2, vy2, model_param);
                if(ret == 0)
                    model_param[0] = -2011;

                if(model_param[0] == -2011)
                    continue;

                // SIMD phase
                // Reset local outlier counter
                outlier_local_count = 0;

                // Compute number of outliers
                for(int i = 0; i < flowvector_count; i++) {
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

                // Compare to threshold
                if(outlier_local_count < flowvector_count * convergence_threshold) {
                    int index                 = g_out_id->fetch_add(1);
                    model_candidate[index]    = iter;
                    outliers_candidate[index] = outlier_local_count;
                }
            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
