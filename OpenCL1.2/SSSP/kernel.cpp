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

int atomic_maximum(std::atomic_int *maximum_value, int value) {
    int prev_value = (maximum_value)->load();
    while(prev_value < value && !(maximum_value)->compare_exchange_strong(prev_value, value))
        ;
    return prev_value;
}

// CPU threads-----------------------------------------------------------------
void run_cpu_threads(Node *h_graph_nodes, Edge *h_graph_edges, std::atomic_int *ptr_cost, std::atomic_int *ptr_color,
    int *ptr_q1, int *ptr_q2, int *ptr_num_t, std::atomic_int *ptr_head, std::atomic_int *ptr_tail,
    std::atomic_int *ptr_threads_end, std::atomic_int *ptr_threads_run, std::atomic_int *ptr_gray_shade,
    std::atomic_int *ptr_iter, int num_threads, int LIMIT, const int GPU) {
///////////////// Run CPU worker threads /////////////////////////////////
#if PRINT
    printf("Starting %d CPU threads\n", num_threads * CPU);
#endif
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < num_threads; i++) {
        cpu_threads.push_back(std::thread([=]() {

            int iter = (ptr_iter)->load(); // Current iteration/level

            int gray_shade = (ptr_gray_shade)->load();
            int base       = (ptr_head)->fetch_add(1); // Fetch new node from input queue
            while(base < *ptr_num_t) {
                int pid = ptr_q1[base];
                ptr_color[pid].store(BLACK); // Node visited
                int cur_cost = ptr_cost[pid].load();
                //For each outgoing edge
                for(int i = h_graph_nodes[pid].x; i < (h_graph_nodes[pid].y + h_graph_nodes[pid].x); i++) {
                    int id   = h_graph_edges[i].x;
                    int cost = h_graph_edges[i].y;
                    cost += cur_cost;
                    int orig_cost = atomic_maximum(&ptr_cost[id], cost);
                    if(orig_cost < cost) {
                        int old_color = atomic_maximum(&ptr_color[id], gray_shade);
                        if(old_color != gray_shade) {
                            //push to the queue
                            int index_o     = (ptr_tail)->fetch_add(1);
                            ptr_q2[index_o] = id;
                        }
                    }
                }
                base = (ptr_head)->fetch_add(1); // Fetch new node from input queue
            }

            if(i == 0) {
                (ptr_iter)->fetch_add(1);
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
