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
#include "support/common.h"
#include "support/ocl.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_work_groups;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    const char *comparison_file;
    int         switching_limit;

    Params(int argc, char **argv) {
        platform        = 0;
        device          = 0;
        n_work_items    = 256;
        n_work_groups   = 8;
        n_threads       = 2;
        n_warmup        = 1;
        n_reps          = 1;
        file_name       = "input/NYR_input.dat";
        comparison_file = "output/NYR_bfs_BFS.out";
        switching_limit = 128;
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:f:c:l:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform        = atoi(optarg); break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_work_items    = atoi(optarg); break;
            case 'g': n_work_groups   = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            case 'l': switching_limit = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(n_work_items > 0 && "Invalid # of device work-items!");
        assert(n_work_groups > 0 && "Invalid # of device work-groups!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./bfs [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=256)"
                "\n    -g <G>    # of device work-groups (default=8)"
                "\n    -t <T>    # of host threads (default=2)"
                "\n    -w <W>    # of untimed warmup iterations (default=1)"
                "\n    -r <R>    # of timed repetition iterations (default=1)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    name of input file with control points (default=input/NYR_input.dat)"
                "\n    -c <C>    comparison file (default=output/NYR_bfs_BFS.out)"
                "\n    -l <L>    switching limit (default=128)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input_size(int &n_nodes, int &n_edges, const Params &p) {
    FILE *fp = fopen(p.file_name, "r");
    fscanf(fp, "%d", &n_nodes);
    fscanf(fp, "%d", &n_edges);
    if(fp)
        fclose(fp);
}

void read_input(int &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

    int   start, edgeno;
    int   n_nodes, n_edges;
    int   id, cost;
    FILE *fp = fopen(p.file_name, "r");

    fscanf(fp, "%d", &n_nodes);
    fscanf(fp, "%d", &n_edges);
    fscanf(fp, "%d", &source);
    printf("Number of nodes = %d\t", n_nodes);
    printf("Number of edges = %d\t", n_edges);

    // initalize the memory: Nodes
    for(int i = 0; i < n_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_nodes[i].x = start;
        h_nodes[i].y = edgeno;
    }
#if PRINT_ALL
    for(int i = 0; i < n_nodes; i++) {
        printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y);
    }
#endif

    // initalize the memory: Edges
    for(int i = 0; i < n_edges; i++) {
        fscanf(fp, "%d", &id);
        fscanf(fp, "%d", &cost);
        h_edges[i].x = id;
        h_edges[i].y = -cost;
    }
    if(fp)
        fclose(fp);
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;

    // Allocate
    int n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);
    timer.start("Allocation");
    Node * h_nodes = (Node *)malloc(sizeof(Node) * n_nodes);
    cl_mem d_nodes = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(Node) * n_nodes, NULL, &clStatus);
    Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);
    cl_mem d_edges = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(Edge) * n_edges, NULL, &clStatus);
    std::atomic_int *h_color = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    cl_mem           d_color = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int) * n_nodes, NULL, &clStatus);
    std::atomic_int *h_cost  = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    cl_mem           d_cost  = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int) * n_nodes, NULL, &clStatus);
    int *            h_q1    = (int *)malloc(n_nodes * sizeof(int));
    cl_mem           d_q1    = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int) * n_nodes, NULL, &clStatus);
    int *            h_q2    = (int *)malloc(n_nodes * sizeof(int));
    cl_mem           d_q2    = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int) * n_nodes, NULL, &clStatus);
    std::atomic_int  h_head[1];
    cl_mem           d_head = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    std::atomic_int  h_tail[1];
    cl_mem           d_tail = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    std::atomic_int  h_threads_end[1];
    cl_mem           d_threads_end = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    std::atomic_int  h_threads_run[1];
    cl_mem           d_threads_run = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    int              h_num_t[1];
    cl_mem           d_num_t = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    int              h_overflow[1];
    cl_mem           d_overflow = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    std::atomic_int  h_iter[1];
    cl_mem           d_iter = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &clStatus);
    clFinish(ocl.clCommandQueue);
    ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);
    CL_ERR();
    timer.stop("Allocation");

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    int source;
    read_input(source, h_nodes, h_edges, p);
    for(int i = 0; i < n_nodes; i++) {
        h_cost[i].store(INF);
    }
    h_cost[source].store(0);
    for(int i = 0; i < n_nodes; i++) {
        h_color[i].store(WHITE);
    }
    h_tail[0].store(0);
    h_head[0].store(0);
    h_threads_end[0].store(0);
    h_threads_run[0].store(0);
    h_q1[0] = source;
    h_iter[0].store(0);
    h_overflow[0] = 0;
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    // Copy to device
    timer.start("Copy To Device");
    clStatus =
        clEnqueueWriteBuffer(ocl.clCommandQueue, d_nodes, CL_TRUE, 0, sizeof(Node) * n_nodes, h_nodes, 0, NULL, NULL);
    clStatus =
        clEnqueueWriteBuffer(ocl.clCommandQueue, d_edges, CL_TRUE, 0, sizeof(Edge) * n_edges, h_edges, 0, NULL, NULL);
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Copy To Device");

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        for(int i = 0; i < n_nodes; i++) {
            h_cost[i].store(INF);
        }
        h_cost[source].store(0);
        for(int i = 0; i < n_nodes; i++) {
            h_color[i].store(WHITE);
        }
        h_tail[0].store(0);
        h_head[0].store(0);
        h_threads_end[0].store(0);
        h_threads_run[0].store(0);
        h_q1[0] = source;
        h_iter[0].store(0);
        h_overflow[0] = 0;

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Run first iteration in master CPU thread
        h_num_t[0] = 1;
        int pid;
        int index_i, index_o;
        for(index_i = 0; index_i < h_num_t[0]; index_i++) {
            pid = h_q1[index_i];
            h_color[pid].store(BLACK);
            for(int i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x); i++) {
                int id = h_edges[i].x;
                h_color[id].store(BLACK);
                index_o       = h_tail[0].fetch_add(1);
                h_q2[index_o] = id;
            }
        }
        h_num_t[0] = h_tail[0].load();
        h_tail[0].store(0);
        h_threads_run[0].fetch_add(1);
        h_iter[0].fetch_add(1);
        if(rep >= p.n_warmup)
            timer.stop("Kernel");

        // Pointers to input and output queues
        int *  h_qin  = h_q2;
        int *  h_qout = h_q1;
        cl_mem d_qin  = d_q2;
        cl_mem d_qout = d_q1;

        const int CPU_EXEC = (p.n_threads > 0) ? 1 : 0;
        const int GPU_EXEC = (p.n_work_groups > 0 && p.n_work_items > 0) ? 1 : 0;

        // Run subsequent iterations on CPU or GPU until number of input queue elements is 0
        while(*h_num_t != 0) {

            if((*h_num_t < p.switching_limit || GPU_EXEC == 0) &&
                CPU_EXEC == 1) { // If the number of input queue elements is lower than switching_limit

                if(rep >= p.n_warmup)
                    timer.start("Kernel");

                // Continue until switching_limit condition is not satisfied
                while((*h_num_t != 0) && (*h_num_t < p.switching_limit || GPU_EXEC == 0) && CPU_EXEC == 1) {

                    // Swap queues
                    if(h_iter[0] % 2 == 0) {
                        h_qin  = h_q1;
                        h_qout = h_q2;
                    } else {
                        h_qin  = h_q2;
                        h_qout = h_q1;
                    }

                    std::thread main_thread(run_cpu_threads, h_nodes, h_edges, h_cost, h_color, h_qin, h_qout, h_num_t,
                        h_head, h_tail, h_threads_end, h_threads_run, h_iter, p.n_threads, p.switching_limit, GPU_EXEC);
                    main_thread.join();

                    h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
                    h_tail[0].store(0);
                    h_head[0].store(0);
                }

                if(rep >= p.n_warmup)
                    timer.stop("Kernel");

            } else if((*h_num_t >= p.switching_limit || CPU_EXEC == 0) &&
                      GPU_EXEC ==
                          1) { // If the number of input queue elements is higher than or equal to switching_limit

                if(rep >= p.n_warmup)
                    timer.start("Copy To Device");
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_cost, CL_TRUE, 0, sizeof(int) * n_nodes, h_cost, 0, NULL, NULL);
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_color, CL_TRUE, 0, sizeof(int) * n_nodes, h_color, 0, NULL, NULL);
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_threads_run, CL_TRUE, 0, sizeof(int), h_threads_run, 0, NULL, NULL);
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_threads_end, CL_TRUE, 0, sizeof(int), h_threads_end, 0, NULL, NULL);
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_overflow, CL_TRUE, 0, sizeof(int), h_overflow, 0, NULL, NULL);
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_q1, CL_TRUE, 0, sizeof(int) * n_nodes, h_q1, 0, NULL, NULL);
                clStatus = clEnqueueWriteBuffer(
                    ocl.clCommandQueue, d_q2, CL_TRUE, 0, sizeof(int) * n_nodes, h_q2, 0, NULL, NULL);
                clStatus =
                    clEnqueueWriteBuffer(ocl.clCommandQueue, d_iter, CL_TRUE, 0, sizeof(int), h_iter, 0, NULL, NULL);
                clFinish(ocl.clCommandQueue);
                CL_ERR();
                if(rep >= p.n_warmup)
                    timer.stop("Copy To Device");

                if(rep >= p.n_warmup)
                    timer.start("Kernel");
                // Setting kernel arguments
                clSetKernelArg(ocl.clKernel, 0, sizeof(cl_mem), &d_nodes);
                clSetKernelArg(ocl.clKernel, 1, sizeof(cl_mem), &d_edges);
                clSetKernelArg(ocl.clKernel, 2, sizeof(cl_mem), &d_cost);
                clSetKernelArg(ocl.clKernel, 3, sizeof(cl_mem), &d_color);
                clSetKernelArg(ocl.clKernel, 6, sizeof(cl_mem), &d_num_t);
                clSetKernelArg(ocl.clKernel, 7, sizeof(cl_mem), &d_head);
                clSetKernelArg(ocl.clKernel, 8, sizeof(cl_mem), &d_tail);
                clSetKernelArg(ocl.clKernel, 9, sizeof(cl_mem), &d_threads_end);
                clSetKernelArg(ocl.clKernel, 10, sizeof(cl_mem), &d_threads_run);
                clSetKernelArg(ocl.clKernel, 11, sizeof(cl_mem), &d_overflow);
                clSetKernelArg(ocl.clKernel, 12, sizeof(cl_mem), &d_iter);
                clSetKernelArg(ocl.clKernel, 13, sizeof(int), NULL);
                clSetKernelArg(ocl.clKernel, 14, sizeof(int) * W_QUEUE_SIZE, NULL);
                clSetKernelArg(ocl.clKernel, 15, sizeof(int), NULL);
                clSetKernelArg(ocl.clKernel, 16, sizeof(int), NULL);
                clSetKernelArg(ocl.clKernel, 17, sizeof(int), &p.switching_limit);
                clSetKernelArg(ocl.clKernel, 18, sizeof(int), &CPU_EXEC);
                size_t ls[1] = {(size_t)p.n_work_items};
                size_t gs[1] = {(size_t)p.n_work_items * p.n_work_groups};
                clFinish(ocl.clCommandQueue);
                if(rep >= p.n_warmup)
                    timer.stop("Kernel");

                // Continue until switching_limit condition is not satisfied
                while((*h_num_t != 0) && (*h_num_t >= p.switching_limit || CPU_EXEC == 0) && GPU_EXEC == 1) {

                    // Swap queues
                    if(h_iter[0] % 2 == 0) {
                        d_qin  = d_q1;
                        d_qout = d_q2;
                    } else {
                        d_qin  = d_q2;
                        d_qout = d_q1;
                    }

                    if(rep >= p.n_warmup)
                        timer.start("Copy To Device");
                    clStatus = clEnqueueWriteBuffer(
                        ocl.clCommandQueue, d_num_t, CL_TRUE, 0, sizeof(int), h_num_t, 0, NULL, NULL);
                    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_tail, CL_TRUE, 0, sizeof(int), h_tail, 0,
                        NULL, NULL); // Number of elements in output queue
                    clStatus = clEnqueueWriteBuffer(
                        ocl.clCommandQueue, d_head, CL_TRUE, 0, sizeof(int), h_head, 0, NULL, NULL);
                    clFinish(ocl.clCommandQueue);
                    CL_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Copy To Device");

                    if(rep >= p.n_warmup)
                        timer.start("Kernel");
                    clSetKernelArg(ocl.clKernel, 4, sizeof(cl_mem), &d_qin); // Input and output queues
                    clSetKernelArg(ocl.clKernel, 5, sizeof(cl_mem), &d_qout);
                    assert(ls[0] <= max_wi && 
                        "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
                    clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
                    clFinish(ocl.clCommandQueue);
                    CL_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Kernel");

                    if(rep >= p.n_warmup)
                        timer.start("Copy Back and Merge");
                    clStatus =
                        clEnqueueReadBuffer(ocl.clCommandQueue, d_tail, CL_TRUE, 0, sizeof(int), h_tail, 0, NULL, NULL);
                    clStatus =
                        clEnqueueReadBuffer(ocl.clCommandQueue, d_iter, CL_TRUE, 0, sizeof(int), h_iter, 0, NULL, NULL);
                    clFinish(ocl.clCommandQueue);
                    CL_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Copy Back and Merge");

                    h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
                    h_tail[0].store(0);
                    h_head[0].store(0);
                }

                if(rep >= p.n_warmup)
                    timer.start("Copy Back and Merge");
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_cost, CL_TRUE, 0, sizeof(int) * n_nodes, h_cost, 0, NULL, NULL);
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_color, CL_TRUE, 0, sizeof(int) * n_nodes, h_color, 0, NULL, NULL);
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_threads_run, CL_TRUE, 0, sizeof(int), h_threads_run, 0, NULL, NULL);
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_threads_end, CL_TRUE, 0, sizeof(int), h_threads_end, 0, NULL, NULL);
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_overflow, CL_TRUE, 0, sizeof(int), h_overflow, 0, NULL, NULL);
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_q1, CL_TRUE, 0, sizeof(int) * n_nodes, h_q1, 0, NULL, NULL);
                clStatus = clEnqueueReadBuffer(
                    ocl.clCommandQueue, d_q2, CL_TRUE, 0, sizeof(int) * n_nodes, h_q2, 0, NULL, NULL);
                clFinish(ocl.clCommandQueue);
                CL_ERR();
                if(rep >= p.n_warmup)
                    timer.stop("Copy Back and Merge");
            }
        }

    } // end of iteration
    timer.print("Allocation", 1);
    timer.print("Copy To Device", p.n_reps);
    timer.print("Kernel", p.n_reps);
    timer.print("Copy Back and Merge", p.n_reps);

    // Verify answer
    verify(h_cost, n_nodes, p.comparison_file);

    // Free memory
    timer.start("Deallocation");
    free(h_nodes);
    free(h_edges);
    free(h_color);
    free(h_cost);
    free(h_q1);
    free(h_q2);
    clStatus = clReleaseMemObject(d_nodes);
    clStatus = clReleaseMemObject(d_edges);
    clStatus = clReleaseMemObject(d_cost);
    clStatus = clReleaseMemObject(d_color);
    clStatus = clReleaseMemObject(d_q1);
    clStatus = clReleaseMemObject(d_q2);
    clStatus = clReleaseMemObject(d_num_t);
    clStatus = clReleaseMemObject(d_head);
    clStatus = clReleaseMemObject(d_tail);
    clStatus = clReleaseMemObject(d_threads_end);
    clStatus = clReleaseMemObject(d_threads_run);
    clStatus = clReleaseMemObject(d_overflow);
    clStatus = clReleaseMemObject(d_iter);
    CL_ERR();
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
