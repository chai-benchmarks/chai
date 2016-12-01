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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

//----------------------------------------------------------------------------
// CPU: Host enqueue task
// 1: repeat
// 2: l ← (end − start + size) mod size
// 3: until l<(size−1)
// 4: queue[end] ← task
// 5: end ← (end + 1) mod size
//----------------------------------------------------------------------------
void host_insert_tasks(task_t *queues, task_t *task_pool, std::atomic_int *num_consumed_tasks,
    std::atomic_int *num_written_tasks, std::atomic_int *num_task_in_queue, int *last_queue, int *num_tasks,
    int gpuQueueSize, int *offset) {
    int i                       = (*last_queue + 1) % NUM_TASK_QUEUES;
    int total_num_tasks         = *num_tasks;
    int remaining_num_tasks     = *num_tasks;
    int num_tasks_to_write_next = (remaining_num_tasks > gpuQueueSize) ? gpuQueueSize : remaining_num_tasks;
    *num_tasks                  = num_tasks_to_write_next;
#if PRINT
    printf("Inserting Tasks...\t");
#endif
    do {
        if(num_consumed_tasks[i].load() == num_written_tasks[i].load()) {
#if PRINT
            printf("Inserting Tasks... %d (%d) in queue %d\n", remaining_num_tasks, *num_tasks, i);
#endif
            // Insert tasks in queue i
            memcpy(&queues[i * gpuQueueSize], &task_pool[(*offset) + total_num_tasks - remaining_num_tasks],
                (*num_tasks) * sizeof(task_t));
            // Update number of tasks in queue i
            num_task_in_queue[i].store(*num_tasks);
            // Total number of tasks written in queue i
            num_written_tasks[i].fetch_add(*num_tasks);
            // Next queue
            i = (i + 1) % NUM_TASK_QUEUES;
            // Remaining tasks
            remaining_num_tasks -= num_tasks_to_write_next;
            num_tasks_to_write_next = (remaining_num_tasks > gpuQueueSize) ? gpuQueueSize : remaining_num_tasks;
            *num_tasks              = num_tasks_to_write_next;
        } else {
            i = (i + 1) % NUM_TASK_QUEUES;
        }
    } while(num_tasks_to_write_next > 0);
    *last_queue = i;
}

void run_cpu_threads(int n_threads, task_t *ptr_queues, std::atomic_int *ptr_num_task_in_queue,
    std::atomic_int *ptr_num_written_tasks, std::atomic_int *ptr_num_consumed_tasks, task_t *ptr_task_pool,
    int *ptr_data, int gpuQueueSize, int *ptr_offset, int *ptr_last_queue, int *ptr_num_tasks, int tpi, int poolSize,
    int n_work_groups) {
///////////////// Run CPU worker threads /////////////////////////////////
#if PRINT
    printf("Starting 1 CPU thread\n");
#endif

    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {

        cpu_threads.push_back(std::thread([=]() {

            int maxConcurrentBlocks = n_work_groups;

            // Insert tasks in queue
            host_insert_tasks(ptr_queues, ptr_task_pool, ptr_num_consumed_tasks, ptr_num_written_tasks,
                ptr_num_task_in_queue, ptr_last_queue, ptr_num_tasks, gpuQueueSize, ptr_offset);
            *ptr_offset += tpi;
#if PRINT
            for(int i = 0; i < NUM_TASK_QUEUES; i++) {
                int task_in_queue = (ptr_num_task_in_queue + i)->load();
                int written       = (ptr_num_written_tasks + i)->load();
                int consumed      = (ptr_num_consumed_tasks + i)->load();
                printf("Queue = %i, written = %i, task_in_queue = %i, consumed = %i\n", i, written, task_in_queue,
                    consumed);
            }
#endif

            while(poolSize > *ptr_offset) {
                *ptr_num_tasks = tpi;
                // Insert tasks in queue
                host_insert_tasks(ptr_queues, ptr_task_pool, ptr_num_consumed_tasks, ptr_num_written_tasks,
                    ptr_num_task_in_queue, ptr_last_queue, ptr_num_tasks, gpuQueueSize, ptr_offset);
                *ptr_offset += tpi;
#if PRINT
                for(int i = 0; i < NUM_TASK_QUEUES; i++) {
                    int task_in_queue = (ptr_num_task_in_queue + i)->load();
                    int written       = (ptr_num_written_tasks + i)->load();
                    int consumed      = (ptr_num_consumed_tasks + i)->load();
                    printf("Queue = %i, written = %i, task_in_queue = %i, consumed = %i, ptr_offset = %i\n", i, written,
                        task_in_queue, consumed, *ptr_offset);
                }
#endif
            }
            // Create stop tasks
            for(int i = 0; i < maxConcurrentBlocks; i++) {
                (ptr_task_pool + i)->id = -1;
                (ptr_task_pool + i)->op = SIGNAL_STOP_KERNEL;
            }
            *ptr_num_tasks = maxConcurrentBlocks;
            *ptr_offset    = 0;
            // Insert stop tasks in queue
            host_insert_tasks(ptr_queues, ptr_task_pool, ptr_num_consumed_tasks, ptr_num_written_tasks,
                ptr_num_task_in_queue, ptr_last_queue, ptr_num_tasks, gpuQueueSize, ptr_offset);
#if PRINT
            for(int i = 0; i < NUM_TASK_QUEUES; i++) {
                int task_in_queue = (ptr_num_task_in_queue + i)->load();
                int written       = (ptr_num_written_tasks + i)->load();
                int consumed      = (ptr_num_consumed_tasks + i)->load();
                printf("Queue = %i, written = %i, task_in_queue = %i, consumed = %i, ptr_offset = %i\n", i, written,
                    task_in_queue, consumed, *ptr_offset);
            }
#endif

        }));
    }

    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
