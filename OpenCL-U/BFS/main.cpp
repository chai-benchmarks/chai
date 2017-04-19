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

#include <string.h>
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
                "\n              WARNING: This benchmark uses persistent threads. Setting -g too large may deadlock."
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
    Node *           nodes = (Node *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(Node) * n_nodes, 0);
    Edge *           edges = (Edge *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(Edge) * n_edges, 0);
    std::atomic_int *color = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * n_nodes, 0);
    std::atomic_int *cost = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int) * n_nodes, 0);
    int *q1 =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(int) * n_nodes, 0);
    int *q2 =
        (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(int) * n_nodes, 0);
    std::atomic_int *head = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    std::atomic_int *tail = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    std::atomic_int *threads_end = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    std::atomic_int *threads_run = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    int *num_t    = (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(int), 0);
    int *overflow = (int *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeof(int), 0);
    clFinish(ocl.clCommandQueue);
    ALLOC_ERR(nodes, edges, color, cost, q1, q2);
    ALLOC_ERR(head, tail, threads_end, threads_run, num_t, overflow);
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    int source;
    read_input(source, nodes, edges, p);
    for(int i = 0; i < n_nodes; i++) {
        cost[i].store(INF);
    }
    cost[source].store(0);
    for(int i = 0; i < n_nodes; i++) {
        color[i].store(WHITE);
    }
    tail[0].store(0);
    head[0].store(0);
    threads_end[0].store(0);
    threads_run[0].store(0);
    q1[0]       = source;
    overflow[0] = 0;
    timer.stop("Initialization");
    timer.print("Initialization", 1);

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        for(int i = 0; i < n_nodes; i++) {
            cost[i].store(INF);
        }
        cost[source].store(0);
        for(int i = 0; i < n_nodes; i++) {
            color[i].store(WHITE);
        }
        // Initialize
        tail[0].store(0);
        head[0].store(0);
        threads_end[0].store(0);
        threads_run[0].store(0);
        q1[0]       = source;
        overflow[0] = 0;

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Run first iteration in master CPU thread
        num_t[0] = 1;
        int pid;
        int index_i, index_o;
        for(index_i = 0; index_i < num_t[0]; index_i++) {
            pid = q1[index_i];
            color[pid].store(BLACK);
            for(int i = nodes[pid].x; i < (nodes[pid].y + nodes[pid].x); i++) {
                int id = edges[i].x;
                color[id].store(BLACK);
                index_o     = tail[0].fetch_add(1);
                q2[index_o] = id;
            }
        }
        num_t[0] = tail[0].load();
        tail[0].store(0);
        threads_run[0].fetch_add(1);

        const int CPU_EXEC = (p.n_threads > 0) ? 1 : 0;
        const int GPU_EXEC = (p.n_work_groups > 0 && p.n_work_items > 0) ? 1 : 0;

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, nodes, edges, cost, color, q1, q2, num_t, head, tail, threads_end,
            threads_run, p.n_threads, p.n_work_groups, p.n_work_items, p.switching_limit, GPU_EXEC);

        // Setting kernel arguments
        clSetKernelArgSVMPointer(ocl.clKernel, 0, nodes);
        clSetKernelArgSVMPointer(ocl.clKernel, 1, edges);
        clSetKernelArgSVMPointer(ocl.clKernel, 2, cost);
        clSetKernelArgSVMPointer(ocl.clKernel, 3, color);
        clSetKernelArgSVMPointer(ocl.clKernel, 4, q1);
        clSetKernelArgSVMPointer(ocl.clKernel, 5, q2);
        clSetKernelArgSVMPointer(ocl.clKernel, 6, num_t);
        clSetKernelArgSVMPointer(ocl.clKernel, 7, head);
        clSetKernelArgSVMPointer(ocl.clKernel, 8, tail);
        clSetKernelArgSVMPointer(ocl.clKernel, 9, threads_end);
        clSetKernelArgSVMPointer(ocl.clKernel, 10, threads_run);
        clSetKernelArgSVMPointer(ocl.clKernel, 11, overflow);
        clSetKernelArg(ocl.clKernel, 12, sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 13, sizeof(int) * W_QUEUE_SIZE, NULL);
        clSetKernelArg(ocl.clKernel, 14, sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 15, sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 16, sizeof(int), &p.switching_limit);
        clSetKernelArg(ocl.clKernel, 17, sizeof(int), &CPU_EXEC);
        // Kernel launch
        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_items * p.n_work_groups};
        if(GPU_EXEC == 1) {
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
            clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");

    } // end of iteration
    timer.print("Kernel", p.n_reps);

    // Verify answer
    verify(cost, n_nodes, p.comparison_file);

    // Free memory
    timer.start("Deallocation");
    clSVMFree(ocl.clContext, nodes);
    clSVMFree(ocl.clContext, edges);
    clSVMFree(ocl.clContext, cost);
    clSVMFree(ocl.clContext, color);
    clSVMFree(ocl.clContext, q1);
    clSVMFree(ocl.clContext, q2);
    clSVMFree(ocl.clContext, num_t);
    clSVMFree(ocl.clContext, head);
    clSVMFree(ocl.clContext, tail);
    clSVMFree(ocl.clContext, threads_end);
    clSVMFree(ocl.clContext, threads_run);
    clSVMFree(ocl.clContext, overflow);
    ocl.release();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    printf("Test Passed\n");
    return 0;
}
