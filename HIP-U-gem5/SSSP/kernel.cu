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

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void SSSP_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end,
    int *threads_run, int *overflow, int *gray_shade,
    int LIMIT, int CPU) {

    HIP_DYNAMIC_SHARED( int, l_mem)
    int* tail_bin = l_mem;
    int* l_qout = (int*)&tail_bin[1];
    int* shift = (int*)&l_qout[W_QUEUE_SIZE];
    int* base = (int*)&shift[1];

    const int tid     = threadIdx.x;
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAXWG   = gridDim.x;
    const int WG_SIZE = blockDim.x;

    int *qin, *qout;

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

        if((*n_t >= LIMIT) || (CPU == 0)) {

            int gray_shade_local = atomicAdd(&gray_shade[0], 0); //atomicAdd_system(&gray_shade[0], 0)

            if(tid == 0) {
                // Reset queue
                *tail_bin = 0;
            }

            // Fetch frontier elements from the queue
            if(tid == 0)
                *base = atomicAdd(&head[0], WG_SIZE); //atomicAdd_system(&head[0], WG_SIZE)
            __syncthreads();

            int my_base = *base;
            while(my_base < *n_t) {

                // If local queue might overflow
                if(*tail_bin >= W_QUEUE_SIZE / 2) {
                    if(tid == 0) {
                        // Add local tail_bin to tail
                        *shift = atomicAdd(&tail[0], *tail_bin); //atomicAdd_system(&tail[0], *tail_bin)
                    }
                    __syncthreads();
                    int local_shift = tid;
                    while(local_shift < *tail_bin) {
                        qout[*shift + local_shift] = l_qout[local_shift];
                        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                        local_shift += WG_SIZE;
                    }
                    __syncthreads();
                    if(tid == 0) {
                        // Reset local queue
                        *tail_bin = 0;
                    }
                    __syncthreads();
                }

                if(my_base + tid < *n_t && *overflow == 0) {
                    // Visit a node from the current frontier
                    int pid = qin[my_base + tid];
                    //////////////// Visit node ///////////////////////////
                    atomicExch(&color[pid], BLACK); // Node visited //atomicExch_system(&color[pid], BLACK);
                    int  cur_cost = atomicAdd(&cost[pid], 0); // Look up shortest-path distance to this node //atomicAdd_system(&cost[pid], 0)
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
                        int orig_cost = atomicMax(&cost[id], cost_local); //atomicMax_system(&cost[id], cost_local)
                        if(orig_cost < cost_local) {
                            int old_color = atomicMax(&color[id], gray_shade_local); //atomicMax_system(&color[id], gray_shade_local)
                            if(old_color != gray_shade_local) {
                                // Push to the queue
                                int tail_index = atomicAdd(tail_bin, 1);
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
                    *base = atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue //atomicAdd_system(&head[0], WG_SIZE)
                __syncthreads();
                my_base = *base;
            }
            /////////////////////////////////////////////////////////
            // Compute size of the output and allocate space in the global queue
            if(tid == 0) {
                *shift = atomicAdd(&tail[0], *tail_bin); //atomicAdd_system(&tail[0], *tail_bin)
            }
            __syncthreads();
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
                atomicAdd(&threads_end[0], WG_SIZE); //atomicAdd_system(&threads_end[0], WG_SIZE);

                while(atomicAdd(&threads_run[0], 0) < iter) { //atomicAdd_system(&threads_run[0], 0)
                }
            }
        } else { // if GPU only
            iter++;
            if(tid == 0)
                atomicAdd(&threads_end[0], WG_SIZE); //atomicAdd_system(&threads_end[0], WG_SIZE);
            if(gtid == 0) {
                while(atomicAdd(&threads_end[0], 0) != MAXWG * WG_SIZE) { //atomicAdd_system(&threads_end[0], 0)
                }
                *n_t = atomicAdd(&tail[0], 0); //atomicAdd_system(&tail[0], 0);
                atomicExch(&tail[0], 0); //atomicExch_system(&tail[0], 0);
                atomicExch(&head[0], 0); //atomicExch_system(&head[0], 0);
                atomicExch(&threads_end[0], 0); //atomicExch_system(&threads_end[0], 0);
                if(iter % 2 == 0)
                    atomicExch(&gray_shade[0], GRAY0); //atomicExch_system(&gray_shade[0], GRAY0);
                else
                    atomicExch(&gray_shade[0], GRAY1); //atomicExch_system(&gray_shade[0], GRAY1);
                atomicAdd(&threads_run[0], 1); //atomicAdd_system(&threads_run[0], 1);
            }
            if(tid == 0 && gtid != 0) {
                while(atomicAdd(&threads_run[0], 0) < iter) { //atomicAdd_system(&threads_run[0], 0)
                }
            }
        }
        __syncthreads();
    }
}

hipError_t call_SSSP_gpu(int blocks, int threads, Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *gray_shade, int LIMIT, const int CPU, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    hipLaunchKernelGGL(SSSP_gpu, dim3(dimGrid), dim3(dimBlock), l_mem_size, 0, graph_nodes_av, graph_edges_av, cost,
        color, q1, q2, n_t,
        head, tail, threads_end, threads_run,
        overflow, gray_shade, LIMIT, CPU);
    
    hipError_t err = hipGetLastError();
    return err;
}
