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

#include "hip/hip_runtime.h"
#include <atomic>
#include <string>
#include "support/common.h"
using namespace std;

void host_insert_tasks(task_t *queues, task_t *task_pool, std::atomic_int *n_consumed_tasks,
    std::atomic_int *n_written_tasks, std::atomic_int *n_task_in_queue, int *last_queue, int *n_tasks,
    int gpuQueueSize, int *offset);
void run_cpu_threads(int n_threads, task_t *queues, std::atomic_int *n_task_in_queue,
    std::atomic_int *n_written_tasks, std::atomic_int *n_consumed_tasks, task_t *task_pool,
    int *data, int gpuQueueSize, int *offset, int *last_queue, int *n_tasks, int tpi, int poolSize,
    int n_work_groups);

hipError_t call_TQHistogram_gpu(int blocks, int threads, task_t *queues, int *n_task_in_queue,
    int *n_written_tasks, int *n_consumed_tasks, int *histo, int *data, int gpuQueueSize, 
    int frame_size, int n_bins, int l_mem_size);
