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
#include "support/verify.h"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

/*extern "C" {
void m5_work_begin(int workid, uint64_t threadid);
void m5_work_end(uint64_t workid, uint64_t threadid);
}*/

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_gpu_threads;
    int         n_gpu_blocks;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    const char *comparison_file;
    int         switching_limit;

    Params(int argc, char **argv) {
        device          = 0;
        n_gpu_threads   = 256;
        n_gpu_blocks    = 8;
        n_threads       = 2;
				n_warmup        = 0;
        n_reps          = 1;
        file_name       = "input/NYR_input.dat";
        comparison_file = "output/NYR_bfs.out";
        switching_limit = 128;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_gpu_threads    = atoi(optarg); break;
            case 'g': n_gpu_blocks   = atoi(optarg); break;
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
        assert(n_gpu_threads > 0 && "Invalid # of device threads!");
        assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        assert(n_threads > 0 && "Invalid # of host threads!");
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./sssp [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    HIP device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=8)"
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
    hipError_t  hipStatus;

    // Allocate
    int n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);
    Node * nodes = (Node *)malloc(sizeof(Node) * n_nodes);
    Edge * edges = (Edge *)malloc(sizeof(Edge) * n_edges);
    std::atomic_int * color = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    std::atomic_int * cost = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    int * q1 = (int *)malloc(sizeof(int) * n_nodes);
    int * q2 = (int *)malloc(sizeof(int) * n_nodes);
    std::atomic_int * head = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    std::atomic_int * tail = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    std::atomic_int * threads_end = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    std::atomic_int * threads_run = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    int * num_t = (int *)malloc(sizeof(int));
    int * overflow = (int *)malloc(sizeof(int));
    std::atomic_int * gray_shade = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    ALLOC_ERR(nodes, edges, color, cost, q1, q2);
    ALLOC_ERR(head, tail, threads_end, threads_run, num_t, overflow, gray_shade);
    hipDeviceSynchronize();

    // Initialize
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
        gray_shade[0].store(GRAY0);

        //m5_work_begin(0, 0);

        // Run first iteration in master CPU thread
        num_t[0] = 1;
        int pid;
        int index_i, index_o;
        for(index_i = 0; index_i < num_t[0]; index_i++) {
            pid = q1[index_i];
            color[pid].store(BLACK);
            int cur_cost = cost[pid].load();
            for(int i = nodes[pid].x; i < (nodes[pid].y + nodes[pid].x); i++) {
                int id = edges[i].x;
                int c  = edges[i].y;
                c += cur_cost;
                cost[id].store(c);
                color[id].store(GRAY0);
                index_o     = tail[0].fetch_add(1);
                q2[index_o] = id;
            }
        }
        num_t[0] = tail[0].load();
        tail[0].store(0);
        threads_run[0].fetch_add(1);
        gray_shade[0].store(GRAY1);

        const int CPU_EXEC = (p.n_threads > 0) ? 1 : 0;
        const int GPU_EXEC = (p.n_gpu_blocks > 0 && p.n_gpu_threads > 0) ? 1 : 0;

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, nodes, edges, cost, color, q1, q2, num_t, head, tail, threads_end,
            threads_run, gray_shade, p.n_threads, p.n_gpu_blocks, p.n_gpu_threads, p.switching_limit, GPU_EXEC);

        // Kernel launch
        if(GPU_EXEC == 1) {
            hipStatus = call_SSSP_gpu(p.n_gpu_blocks, p.n_gpu_threads, nodes, edges, (int*)cost,
                (int*)color, q1, q2, num_t,
                (int*)head, (int*)tail, (int*)threads_end, (int*)threads_run,
                overflow, (int*)gray_shade, p.switching_limit, CPU_EXEC, sizeof(int) * (W_QUEUE_SIZE + 3));
            if(hipStatus != hipSuccess) { fprintf(stderr, "HIP error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        }

        hipDeviceSynchronize();
        main_thread.join();

        //m5_work_end(0, 0);

    } // end of iteration

    // Verify answer
    verify(cost, n_nodes, p.comparison_file);

    // Free memory
    free(nodes);
    free(edges);
    free(cost);
    free(color);
    free(q1);
    free(q2);
    free(num_t);
    free(head);
    free(tail);
    free(threads_end);
    free(threads_run);
    free(overflow);
    free(gray_shade);

    printf("Test Passed\n");
    return 0;
}
