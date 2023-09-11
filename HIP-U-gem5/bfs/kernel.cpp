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
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdio>

int atomic_maximum(std::atomic_int *maximum_value, int value) {
    int prev_value = (maximum_value)->load();
    while(prev_value < value && !(maximum_value)->compare_exchange_strong(prev_value, value))
        ;
    return prev_value;
}

void run_cpu_threads(Node *h_graph_nodes, Edge *h_graph_edges, std::atomic_int *cost, std::atomic_int *color,
    int *q1, int *q2, int *n_t, std::atomic_int *head, std::atomic_int *tail,
    std::atomic_int *threads_end, std::atomic_int *threads_run, int n_threads, int max_wg, int wg_size,
    int LIMIT, const int GPU) {
///////////////// Run CPU worker threads /////////////////////////////////
    //printf("Starting %d CPU threads\n", n_threads);
    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {
        //printf("CPU thread_id %d\n", k);
        cpu_threads.push_back(std::thread([=]() {

            int *qin, *qout;

            int iter = 1;

            while(*n_t != 0) {
                //printf("CPU n_t = %d\n", *n_t);

                // Swap queues
                if(iter % 2 == 0) {
                    qin  = q1;
                    qout = q2;
                } else {
                    qin  = q2;
                    qout = q1;
                }

                if(*n_t < LIMIT || GPU == 0) {
                    //printf("CPU under limit\n");

                    int base = (head)->fetch_add(1); // Fetch new node from input queue
                    while(base < *n_t) {
                        //printf("CPU limit n_t = %d\n", base);
                        int pid = qin[base];
                        cost[pid].store(iter); // Node visited
                        // For each outgoing edge
                        for(int i = h_graph_nodes[pid].x; i < (h_graph_nodes[pid].y + h_graph_nodes[pid].x); i++) {
                            //printf("CPU limit iter = %d | max: %d\n", i, h_graph_nodes[pid].y + h_graph_nodes[pid].x);
                            int id        = h_graph_edges[i].x;
                            int old_color = atomic_maximum(&color[id], BLACK);
                            if(old_color < BLACK) {
                                // Push to the queue
                                int index_o       = (tail)->fetch_add(1);
                                qout[index_o] = id;
                            }
                        }
                        base = (head)->fetch_add(1); // Fetch new node from input queue
                    }
                }
                //printf("CPU threads syncing\n");
                // Synchronization
                iter++;
                (threads_end)->fetch_add(1);
                if(k == 0) {
                    //printf("CPU k is 0\n");
                    while((threads_end)->load() != GPU * max_wg * wg_size + n_threads) {
                    }
                    //printf("CPU k=0 ended\n");
                    *n_t = (tail)->load();
                    (tail)->store(0);
                    (head)->store(0);
                    (threads_end)->store(0);
                    (threads_run)->fetch_add(1);
                } else {
                    //printf("CPU k = %d\n", k);
                    while((threads_run)->load() < iter) {
                    }
                    //printf("CPU k!=0 ended\n");
                }
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
