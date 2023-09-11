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
#include "hip/hip_runtime.h"

// CUDA kernel ------------------------------------------------------------------------------------------
__global__ void BFS_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end,
    int *threads_run, int *overflow, int LIMIT, const int CPU) {

    //printf("GPU kernel printing...\n");
    HIP_DYNAMIC_SHARED( int, l_mem)
    //printf("GPU dynamic shared memory inited\n");
    int* tail_bin = l_mem;
    int* l_q2 = (int*)&tail_bin[1];
    int* shift = (int*)&l_q2[W_QUEUE_SIZE];
    int* base = (int*)&shift[1];
    
    const int tid     = threadIdx.x;
    const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int MAXWG   = gridDim.x;
    const int WG_SIZE = blockDim.x;

    //printf("GPU variables inited: %d %d %d %d\n", tid, gtid, MAXWG, WG_SIZE);
    int *qin, *qout;

    int iter = 1;

    while(*n_t != 0) {
        //printf("GPU n_t != 0\n");

        // Swap queues
        if(iter % 2 == 0) {
            qin  = q1;
            qout = q2;
        } else {
            qin  = q2;
            qout = q1;
        }

        if((*n_t >= LIMIT) | (CPU == 0)) {
            //printf("GPU over limit; n_t = %d\n", *n_t);

            if(tid == 0) {
                // Reset queue
                *tail_bin = 0;
            }
            //printf("GPU Q reset\n");

            // Fetch frontier elements from the queue
            if(tid == 0)
                *base = atomicAdd(&head[0], WG_SIZE); //*base = atomicAdd_system(&head[0], WG_SIZE);
            //printf("GPU fetched\n");
            __syncthreads();
            //printf("GPU synced\n");

            int my_base = *base;
            while(my_base < *n_t) {
                //printf("GPU my_base(%d) < n_t\n", my_base);
                if(my_base + tid < *n_t && *overflow == 0) {
                    // Visit a node from the current frontier
                    int pid = qin[my_base + tid];
                    //////////////// Visit node ///////////////////////////
                    //printf("GPU visiting node\n");
                    atomicExch(&cost[pid], iter); //atomicExch_system(&cost[pid], iter); // Node visited
                    Node cur_node;
                    cur_node.x = graph_nodes_av[pid].x;
                    cur_node.y = graph_nodes_av[pid].y;
                    // For each outgoing edge
                    //printf("GPU for outgoing nodes\n");
                    for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                        int id        = graph_edges_av[i].x;
                        int old_color = atomicMax(&color[id], BLACK); //int old_color = atomicMax_system(&color[id], BLACK);
                        if(old_color < BLACK) {
                            // Push to the queue
                            int tail_index = atomicAdd(tail_bin, 1);
                            if(tail_index >= W_QUEUE_SIZE) {
                                *overflow = 1;
                                break;
                            } else
                                l_q2[tail_index] = id;
                        }
                    }
                    //printf("GPU for done\n"); 
                }
                if(tid == 0) {
                    //printf("GPU tid = 0\n");
                    *base = atomicAdd(&head[0], WG_SIZE); //*base = atomicAdd_system(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
                }
                __syncthreads();
                my_base = *base;
                //printf("GPU synced\n");
            }
            //printf("GPU while done\n");
            /////////////////////////////////////////////////////////
            // Compute size of the output and allocate space in the global queue
            if(tid == 0) {
                //printf("GPU again, TID=0\n");
                *shift = atomicAdd(&tail[0], *tail_bin); //*shift = atomicAdd_system(&tail[0], *tail_bin);
            }
            __syncthreads();
            ///////////////////// CONCATENATE INTO HOST COHERENT MEMORY /////////////////////
            int local_shift = tid;
            //printf("GPU concatenating to host CM\n");
            while(local_shift < *tail_bin) {
                qout[*shift + local_shift] = l_q2[local_shift];
                // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                local_shift += WG_SIZE;
            }
            //printf("GPU while done (2)\n");
            //////////////////////////////////////////////////////////////////////////
        }

        //printf("GPU syncing\n");
        // Synchronization
        if(*overflow == 1) {
            break;
        }

        //printf("GPU -- CPU: %d\n", CPU);
        if(CPU) { // if CPU is available
            iter++;
            if(tid == 0) {
                //printf("GPU atomically adding\n");
                atomicAdd(&threads_end[0], WG_SIZE); //atomicAdd_system(&threads_end[0], WG_SIZE);

                //printf("GPU whiling...\n");
                while(atomicAdd(&threads_run[0], 0) < iter) { //atomicAdd_system(&threads_run[0], 0)
                }
                //printf("GPU done whiling...\n");
            }
        } else { // if GPU only
            //printf("GPU will never enter here\n");
            iter++;
            if(tid == 0)
                atomicAdd(&threads_end[0], WG_SIZE); //atomicAdd_system(&threads_end[0], WG_SIZE);
            if(gtid == 0) {
                while(atomicAdd(&threads_end[0], 0) != MAXWG * WG_SIZE) { //atomicAdd_system(&threads_end[0], 0)
                }
                *n_t = atomicAdd(&tail[0], 0); //*n_t = atomicAdd_system(&tail[0], 0);
                atomicExch(&tail[0], 0); //atomicExch_system(&tail[0], 0);
                atomicExch(&head[0], 0); //atomicExch_system(&head[0], 0);
                atomicExch(&threads_end[0], 0); //atomicExch_system(&threads_end[0], 0);
                atomicAdd(&threads_run[0], 1); //atomicAdd_system(&threads_run[0], 1);
            }
            if(tid == 0 && gtid != 0) {
                while(atomicAdd(&threads_run[0], 0) < iter) { //atomicAdd_system(&threads_run[0], 0)
                }
            }
        }
        //printf("GPU syncing (3)\n");
        __syncthreads();
    }
    //printf("GPU n_t != 0 whiling done!\n");
}

hipError_t call_BFS_gpu(int blocks, int threads, Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int LIMIT, const int CPU, int l_mem_size){

    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);

    //printf("GPU launching...");
    hipLaunchKernelGGL(BFS_gpu, dim3(dimGrid), dim3(dimBlock), l_mem_size, 0, graph_nodes_av, graph_edges_av, cost,
        color, q1, q2, n_t,
        head, tail, threads_end, threads_run,
        overflow, LIMIT, CPU);
    
    hipError_t err = hipGetLastError();
    //printf("GPU launched threads without errors");
    return err;
}
