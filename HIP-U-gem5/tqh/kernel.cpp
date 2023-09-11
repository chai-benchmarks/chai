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
//----------------------------------------------------------------------------
void host_insert_tasks(task_t *queues, task_t *task_pool, std::atomic_int *n_consumed_tasks,
    std::atomic_int *n_written_tasks, std::atomic_int *n_task_in_queue, int *last_queue, int *n_tasks,
    int gpuQueueSize, int *offset) {
    int i                     = (*last_queue + 1) % NUM_TASK_QUEUES;
    int n_total_tasks         = *n_tasks;
    int n_remaining_tasks     = *n_tasks;
    int n_tasks_to_write_next = (n_remaining_tasks > gpuQueueSize) ? gpuQueueSize : n_remaining_tasks;
    *n_tasks                  = n_tasks_to_write_next;
    printf("Inserting Tasks...\t");
    do {
        if(n_consumed_tasks[i].load() == n_written_tasks[i].load()) {
            printf("Inserting Tasks... %d (%d) in queue %d\n", n_remaining_tasks, *n_tasks, i);
            // Insert tasks in queue i
            memcpy(&queues[i * gpuQueueSize], &task_pool[(*offset) + n_total_tasks - n_remaining_tasks],
                (*n_tasks) * sizeof(task_t));
            // Update number of tasks in queue i
            n_task_in_queue[i].store(*n_tasks);
            // Total number of tasks written in queue i
            n_written_tasks[i].fetch_add(*n_tasks);
            // Next queue
            i = (i + 1) % NUM_TASK_QUEUES;
            // Remaining tasks
            n_remaining_tasks -= n_tasks_to_write_next;
            n_tasks_to_write_next = (n_remaining_tasks > gpuQueueSize) ? gpuQueueSize : n_remaining_tasks;
            *n_tasks              = n_tasks_to_write_next;
        } else {
            i = (i + 1) % NUM_TASK_QUEUES;
        }
    } while(n_tasks_to_write_next > 0);
    *last_queue = i;
}

void run_cpu_threads(int n_threads, task_t *queues, std::atomic_int *n_task_in_queue,
    std::atomic_int *n_written_tasks, std::atomic_int *n_consumed_tasks, task_t *task_pool,
    int *data, int gpuQueueSize, int *offset, int *last_queue, int *n_tasks, int tpi, int poolSize,
    int n_work_groups) {
///////////////// Run CPU worker threads /////////////////////////////////
    printf("Starting 1 CPU thread\n");

    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {

        cpu_threads.push_back(std::thread([=]() {

            int maxConcurrentBlocks = n_work_groups;

            // Insert tasks in queue
            host_insert_tasks(queues, task_pool, n_consumed_tasks, n_written_tasks,
                n_task_in_queue, last_queue, n_tasks, gpuQueueSize, offset);
            *offset += tpi;
            for(int i = 0; i < NUM_TASK_QUEUES; i++) {
                int task_in_queue = (n_task_in_queue + i)->load();
                int written       = (n_written_tasks + i)->load();
                int consumed      = (n_consumed_tasks + i)->load();
                printf("Queue = %i, written = %i, task_in_queue = %i, consumed = %i\n", i, written, task_in_queue,
                    consumed);
            }

            while(poolSize > *offset) {
                *n_tasks = tpi;
                // Insert tasks in queue
                host_insert_tasks(queues, task_pool, n_consumed_tasks, n_written_tasks,
                    n_task_in_queue, last_queue, n_tasks, gpuQueueSize, offset);
                *offset += tpi;
                for(int i = 0; i < NUM_TASK_QUEUES; i++) {
                    int task_in_queue = (n_task_in_queue + i)->load();
                    int written       = (n_written_tasks + i)->load();
                    int consumed      = (n_consumed_tasks + i)->load();
                    printf("Queue = %i, written = %i, task_in_queue = %i, consumed = %i, offset = %i\n", i, written,
                        task_in_queue, consumed, *offset);
                }
            }
            // Create stop tasks
            for(int i = 0; i < maxConcurrentBlocks; i++) {
                (task_pool + i)->id = -1;
                (task_pool + i)->op = SIGNAL_STOP_KERNEL;
            }
            *n_tasks = maxConcurrentBlocks;
            *offset    = 0;
            // Insert stop tasks in queue
            host_insert_tasks(queues, task_pool, n_consumed_tasks, n_written_tasks,
                n_task_in_queue, last_queue, n_tasks, gpuQueueSize, offset);
            for(int i = 0; i < NUM_TASK_QUEUES; i++) {
                int task_in_queue = (n_task_in_queue + i)->load();
                int written       = (n_written_tasks + i)->load();
                int consumed      = (n_consumed_tasks + i)->load();
                printf("Queue = %i, written = %i, task_in_queue = %i, consumed = %i, offset = %i\n", i, written,
                    task_in_queue, consumed, *offset);
            }

        }));
    }

    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
