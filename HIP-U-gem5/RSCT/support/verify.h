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

#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

inline int compare_output(int count1, int count2, int outliers1, int outliers2) {
    if(count1 != count2) {
        printf("Test failed, count1 = %d | count2 = %d\n", count1, count2);
        exit(EXIT_FAILURE);
    }
    if(outliers1 != outliers2) {
        printf("Test failed, ol1=%d, ol2=%d\n", outliers1, outliers2);
        exit(EXIT_FAILURE);
    }
    return 0;
}

// Sequential implementation for comparison purposes
// Function to compute new set of motion vectors based on the first order flow model
inline void gen_firstOrderFlow_vectors(
    float *model_param, int flow_vector_count, flowvector *flow_vector_array, flowvector *ego_vector_array) {
    float temp_x, temp_y;
    // Compute new set of motion vectors for each point specified in the flow_vector_array
    for(int i = 0; i < flow_vector_count; i++) {
        temp_x                 = ((float)(flow_vector_array[i].x)) - model_param[0];
        temp_y                 = ((float)(flow_vector_array[i].y)) - model_param[1];
        ego_vector_array[i].x  = flow_vector_array[i].x;
        ego_vector_array[i].y  = flow_vector_array[i].y;
        ego_vector_array[i].vx = flow_vector_array[i].x + ((temp_x * model_param[2]) - (temp_y * model_param[3]));
        ego_vector_array[i].vy = flow_vector_array[i].y + ((temp_y * model_param[2]) + (temp_x * model_param[3]));
    }
}
// Compare to better model
inline void choose_better_model(flowvector *flow_vector_array, flowvector *ego_vector_array, int flow_vector_count,
    float *model_param, int *model_candidate, int *outliers_candidate, int *count_candidates, int error_threshold,
    float convergence_threshold, int iter) {
    int   outlier_count = 0;
    int   vx1, vy1;
    float vx_error, vy_error;

    // This loop calculates the no of outliers
    for(int i = 0; i < flow_vector_count; i++) {
        vx1      = flow_vector_array[i].vx - flow_vector_array[i].x;
        vy1      = flow_vector_array[i].vy - flow_vector_array[i].y;
        vx_error = flow_vector_array[i].x + ((int)((flow_vector_array[i].x - model_param[0]) * model_param[2]) -
                                                (int)((flow_vector_array[i].y - model_param[1]) * model_param[3])) -
                   flow_vector_array[i].vx;
        vy_error = flow_vector_array[i].y + ((int)((flow_vector_array[i].y - model_param[1]) * model_param[2]) +
                                                (int)((flow_vector_array[i].x - model_param[0]) * model_param[3])) -
                   flow_vector_array[i].vy;

        if((fabs(vx_error) < error_threshold) && (fabs(vy_error) < error_threshold)) {
            ego_vector_array[i].x  = 0;
            ego_vector_array[i].y  = 0;
            ego_vector_array[i].vx = 0;
            ego_vector_array[i].vy = 0;
        } else {
            outlier_count++;
            ego_vector_array[i].vx = ego_vector_array[i].x + vx_error;
            ego_vector_array[i].vy = ego_vector_array[i].y + vy_error;
        }
    }

    // Compare to threshold
    if(outlier_count < flow_vector_count * convergence_threshold) {
        int ind                 = count_candidates[0]++;
        model_candidate[ind]    = iter;
        outliers_candidate[ind] = outlier_count;
    }
}
// Function to generate model parameters for first order flow (xc, yc, D and R)
inline int gen_model_param(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param) {
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
// Generate F-o-F model
inline int gen_firstOrderFlow_model(
    int flow_vector_count, flowvector *flow_vector_array, float *model_param, int *random_numbers, int iter) {
    int   rand_num;
    int   x1, x2, y1, y2, vx1, vx2, vy1, vy2;
    int   ret;
    float highest_val = 0;

    // Select two motion vectors at random
    rand_num = random_numbers[iter * 2];
    x1       = flow_vector_array[rand_num].x;
    y1       = flow_vector_array[rand_num].y;
    vx1      = flow_vector_array[rand_num].vx - flow_vector_array[rand_num].x;
    vy1      = flow_vector_array[rand_num].vy - flow_vector_array[rand_num].y;

    rand_num = random_numbers[iter * 2 + 1];
    x2       = flow_vector_array[rand_num].x;
    y2       = flow_vector_array[rand_num].y;
    vx2      = flow_vector_array[rand_num].vx - flow_vector_array[rand_num].x;
    vy2      = flow_vector_array[rand_num].vy - flow_vector_array[rand_num].y;

    // Function to generate model parameters according to first order flow (xc, yc, D and R)
    ret = gen_model_param(x1, y1, vx1, vy1, x2, y2, vx2, vy2, model_param);
    return (ret);
}
// Estimate egomotion using RANSAC
inline int estimate_ego_motion_first_order_flow(flowvector *flow_vector_array, int size_flow_vector_array,
    int *model_candidate, int *outliers_candidate, int *count_candidates, int *random_numbers, int max_iter,
    int error_threshold, float convergence_threshold) {
    int iter = 0;
    int ret;
    if(size_flow_vector_array == 0) {
        return (0);
    }

    // Allocate memory to store newly generated vectors
    flowvector ego_vector_array[size_flow_vector_array];
    *count_candidates = 0;

    for(iter = 0; iter < max_iter; iter++) {
        float model_param_seq[4];
        // Obtain model parameters for First Order Flow
        ret =
            gen_firstOrderFlow_model(size_flow_vector_array, flow_vector_array, model_param_seq, random_numbers, iter);
        if(ret == 0) {
            continue;
        }
        // Compute motion vectors at every point of optical flow using First Order Flow equations
        gen_firstOrderFlow_vectors(model_param_seq, size_flow_vector_array, flow_vector_array, ego_vector_array);
        // Decide if the new model is better than the previous model
        choose_better_model(flow_vector_array, ego_vector_array, size_flow_vector_array, model_param_seq,
            model_candidate, outliers_candidate, count_candidates, error_threshold, convergence_threshold, iter);
    }

    return (1);
}

inline void verify(flowvector *flow_vector_array, int size_flow_vector_array, int *random_numbers, int max_iter,
    int error_threshold, float convergence_threshold, int candidates, int b_outliers) {

    int *model_candidate    = (int *)malloc(max_iter * sizeof(int));
    int *outliers_candidate = (int *)malloc(max_iter * sizeof(int));
    int  count_candidates   = 0;
    estimate_ego_motion_first_order_flow(flow_vector_array, size_flow_vector_array, model_candidate, outliers_candidate,
        &count_candidates, random_numbers, max_iter, error_threshold, convergence_threshold);
    // Post-processing (chooses the best model among the candidates)
    int best_model    = -1;
    int best_outliers = size_flow_vector_array;
    for(int i = 0; i < count_candidates; i++) {
        if(outliers_candidate[i] < best_outliers) {
            best_outliers = outliers_candidate[i];
            best_model    = model_candidate[i];
        }
    }
    compare_output(candidates, count_candidates, best_outliers, b_outliers);
    free(model_candidate);
    free(outliers_candidate);
}
