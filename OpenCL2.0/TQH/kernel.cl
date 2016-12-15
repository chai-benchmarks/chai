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

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void TQHistogram_gpu(__global task_t *queues, __global atomic_int *n_task_in_queue,
    __global atomic_int *n_written_tasks, __global atomic_int *n_consumed_tasks,
    __global atomic_int *histo, __global int *data, int gpuQueueSize, __local task_t *t,
    __local int *last_queue, __local int *l_histo, int frame_size, int n_bins) {

    const int tid       = get_local_id(0);
    const int tile_size = get_local_size(0);

    while(true) {
        // Fetch task
        if(tid == 0) {
            int  idx_queue = *last_queue;
            int  j, jj;
            bool not_done = true;

            do {
                if(atomic_load(n_consumed_tasks + idx_queue) == atomic_load(n_written_tasks + idx_queue)) {
                    idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                } else {
                    if(atomic_load(n_task_in_queue + idx_queue) > 0) {
                        j = atomic_fetch_sub(n_task_in_queue + idx_queue, 1) - 1;
                        if(j >= 0) {
                            t->id    = (queues + idx_queue * gpuQueueSize + j)->id;
                            t->op    = (queues + idx_queue * gpuQueueSize + j)->op;
                            jj       = atomic_fetch_add(n_consumed_tasks + idx_queue, 1) + 1;
                            not_done = false;
                            if(jj == atomic_load(n_written_tasks + idx_queue)) {
                                idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                            }
                            *last_queue = idx_queue;
                        } else {
                            idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                        }
                    } else {
                        idx_queue = (idx_queue + 1) % NUM_TASK_QUEUES;
                    }
                }
            } while(not_done);
        }
        barrier(CLK_LOCAL_MEM_FENCE); // It can be removed if work-group = wavefront

        // Compute task
        if(t->op == SIGNAL_STOP_KERNEL) {
            break;
        } else {
            if(t->op == SIGNAL_WORK_KERNEL) {
                // Reset local histogram
                for(int i = tid; i < n_bins; i += tile_size) {
                    l_histo[i] = 0;
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                for(int i = tid; i < frame_size; i += tile_size) {
                    int value = (data[t->id * frame_size + i] * n_bins) >> 8;

                    atomic_add(l_histo + value, 1);
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                // Store in global memory
                for(int i = tid; i < n_bins; i += tile_size) {
                    histo[t->id * n_bins + i] = l_histo[i];
                }
            }
        }
    }
}
