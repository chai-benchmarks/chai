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
__kernel void SSSP_gpu(__global Node *graph_nodes_av, __global Edge *graph_edges_av, __global atomic_int *cost,
    __global atomic_int *color, __global int *q1, __global int *q2, __global int *n_t,
    __global atomic_int *head, __global atomic_int *tail, __global atomic_int *threads_end,
    __global atomic_int *threads_run, __global int *overflow, __global atomic_int *gray_shade,
    __local int *tail_bin, __local int *l_qout, __local int *shift, __local int *base, int LIMIT, int CPU) {

    const int tid     = get_local_id(0);
    const int gtid    = get_global_id(0);
    const int MAXWG   = get_num_groups(0);
    const int WG_SIZE = get_local_size(0);

    __global int *qin, *qout;

    int iter = 1;

    while(*n_t != 0) {

        // Swap queues
        if(iter % 2 == 0) {
            qin  = q1;
            qout = q2;
        } else {
            qin  = q2;
            qout = q1;
        }

        if((*n_t >= LIMIT) | (CPU == 0)) {

            int gray_shade_local = atomic_load(&gray_shade[0]);

            if(tid == 0) {
                // Reset queue
                *tail_bin = 0;
            }

            // Fetch frontier elements from the queue
            if(tid == 0)
                *base = atomic_fetch_add(&head[0], WG_SIZE);
            barrier(CLK_LOCAL_MEM_FENCE);

            int my_base = *base;
            while(my_base < *n_t) {

                // If local queue might overflow
                if(*tail_bin >= W_QUEUE_SIZE / 2) {
                    if(tid == 0) {
                        // Add local tail_bin to tail
                        *shift = atomic_fetch_add(&tail[0], *tail_bin);
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    int local_shift = tid;
                    while(local_shift < *tail_bin) {
                        qout[*shift + local_shift] = l_qout[local_shift];
                        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                        local_shift += WG_SIZE;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                    if(tid == 0) {
                        // Reset local queue
                        *tail_bin = 0;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                if(my_base + tid < *n_t && *overflow == 0) {
                    // Visit a node from the current frontier
                    int pid = qin[my_base + tid];
                    //////////////// Visit node ///////////////////////////
                    atomic_store(&color[pid], BLACK); // Node visited
                    int  cur_cost = atomic_load(&cost[pid]); // Look up shortest-path distance to this node
                    Node cur_node;
                    cur_node.x = graph_nodes_av[pid].x;
                    cur_node.y = graph_nodes_av[pid].y;
                    Edge cur_edge;
                    // For each outgoing edge
                    for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                        cur_edge.x = graph_edges_av[i].x;
                        cur_edge.y = graph_edges_av[i].y;
                        int id     = cur_edge.x;
                        int cost_local   = cur_edge.y;
                        cost_local += cur_cost;
                        int orig_cost = atomic_fetch_max(&cost[id], cost_local);
                        if(orig_cost < cost_local) {
                            int old_color = atomic_fetch_max(&color[id], gray_shade_local);
                            if(old_color != gray_shade_local) {
                                // Push to the queue
                                int tail_index = atomic_add(tail_bin, 1);
                                if(tail_index >= W_QUEUE_SIZE) {
                                    *overflow = 1;
                                    break;
                                } else
                                    l_qout[tail_index] = id;
                            }
                        }
                    }
                }
                if(tid == 0)
                    *base = atomic_fetch_add(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
                barrier(CLK_LOCAL_MEM_FENCE);
                my_base = *base;
            }
            /////////////////////////////////////////////////////////
            // Compute size of the output and allocate space in the global queue
            if(tid == 0) {
                *shift = atomic_fetch_add(&tail[0], *tail_bin);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            ///////////////////// CONCATENATE INTO HOST COHERENT MEMORY /////////////////////
            int local_shift = tid;
            while(local_shift < *tail_bin) {
                qout[*shift + local_shift] = l_qout[local_shift];
                // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                local_shift += WG_SIZE;
            }
            //////////////////////////////////////////////////////////////////////////
        }

        // Synchronization
        if(*overflow == 1) {
            break;
        }

        if(CPU) { // if CPU is available
            iter++;
            if(tid == 0) {
                atomic_fetch_add(&threads_end[0], WG_SIZE);

                while(atomic_load(&threads_run[0]) < iter) {
                }
            }
        } else { // if GPU only
            iter++;
            if(tid == 0)
                atomic_fetch_add(&threads_end[0], WG_SIZE);
            if(gtid == 0) {
                while(atomic_load(&threads_end[0]) != MAXWG * WG_SIZE) {
                }
                *n_t = atomic_load(&tail[0]);
                atomic_store(&tail[0], 0);
                atomic_store(&head[0], 0);
                atomic_store(&threads_end[0], 0);
                if(iter % 2 == 0)
                    atomic_store(&gray_shade[0], GRAY0);
                else
                    atomic_store(&gray_shade[0], GRAY1);
                atomic_fetch_add(&threads_run[0], 1);
            }
            if(tid == 0 && gtid != 0) {
                while(atomic_load(&threads_run[0]) < iter) {
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
