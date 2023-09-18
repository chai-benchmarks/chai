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
    float       alpha;
    const char *file_name;
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;

    Params(int argc, char **argv) {
        device                = 0;
        n_gpu_threads         = 256;
        n_gpu_blocks          = 8;
        n_threads             = 4;
				n_warmup              = 0;
				n_reps                = 1;
        alpha                 = 0.2;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:a:f:m:e:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device                = atoi(optarg); break;
            case 'i': n_gpu_threads          = atoi(optarg); break;
            case 'g': n_gpu_blocks         = atoi(optarg); break;
            case 't': n_threads             = atoi(optarg); break;
            case 'w': n_warmup              = atoi(optarg); break;
            case 'r': n_reps                = atoi(optarg); break;
            case 'a': alpha                 = atof(optarg); break;
            case 'f': file_name             = optarg; break;
            case 'm': max_iter              = atoi(optarg); break;
            case 'e': error_threshold       = atoi(optarg); break;
            case 'c': convergence_threshold = atof(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
#ifdef CUDA_8_0
            assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of host + device workers!");
#else
            assert(0 && "Illegal value for -a");
#endif
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./rscd [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -d <D>    HIP device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=256)"
                "\n    -g <G>    # of device blocks (default=8)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
#ifdef CUDA_8_0
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name (default=input/vectors.csv)"
                "\n    -m <M>    maximum # of iterations (default=2000)"
                "\n    -e <E>    error threshold (default=3)"
                "\n    -c <C>    convergence threshold (default=0.75)"
                "\n");
    }
};

// Input ----------------------------------------------------------------------
int read_input_size(const Params &p) {
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error al abrir el fichero");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);

    return n;
}

void read_input(flowvector *v, int *r, const Params &p) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(time(NULL));
    for(int i = 0; i < 2 * p.max_iter; i++) {
        r[i] = ((int)rand()) % n;
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    hipError_t  hipStatus;

    // Allocate
    int n_flow_vectors = read_input_size(p);
    int best_model     = -1;
    int best_outliers  = n_flow_vectors;
#ifdef CUDA_8_0
    flowvector *h_flow_vector_array = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *h_random_numbers = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *h_model_candidate = (int *)malloc(p.max_iter * sizeof(int));
    int *h_outliers_candidate = (int *)malloc(p.max_iter * sizeof(int));
    float *h_model_param_local = (float *)malloc(4 * p.max_iter * sizeof(float));
    std::atomic_int *h_g_out_id = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    flowvector *     d_flow_vector_array  = h_flow_vector_array;
    int *            d_random_numbers     = h_random_numbers;
    int *            d_model_candidate    = h_model_candidate;
    int *            d_outliers_candidate = h_outliers_candidate;
    float *          d_model_param_local  = h_model_param_local;
    std::atomic_int *d_g_out_id           = h_g_out_id;
    std::atomic_int * worklist = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    ALLOC_ERR(worklist);
#else
    flowvector *     h_flow_vector_array  = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *            h_random_numbers     = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *            h_model_candidate    = (int *)malloc(p.max_iter * sizeof(int));
    int *            h_outliers_candidate = (int *)malloc(p.max_iter * sizeof(int));
    float *          h_model_param_local  = (float *)malloc(4 * p.max_iter * sizeof(float));
    std::atomic_int *h_g_out_id           = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    flowvector *     d_flow_vector_array;
    hipStatus = hipMalloc((void**)&d_flow_vector_array, n_flow_vectors * sizeof(flowvector));
    int *            d_random_numbers;
    hipStatus = hipMalloc((void**)&d_random_numbers, 2 * p.max_iter * sizeof(int));
    int *            d_model_candidate;
    hipStatus = hipMalloc((void**)&d_model_candidate, p.max_iter * sizeof(int));
    int *            d_outliers_candidate;
    hipStatus = hipMalloc((void**)&d_outliers_candidate, p.max_iter * sizeof(int));
    float *          d_model_param_local;
    hipStatus = hipMalloc((void**)&d_model_param_local, 4 * p.max_iter * sizeof(float));
    int *d_g_out_id;
    hipStatus = hipMalloc((void**)&d_g_out_id, sizeof(int));
    if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
#endif
    ALLOC_ERR(h_flow_vector_array, h_random_numbers, h_model_candidate, h_outliers_candidate, h_model_param_local, h_g_out_id);
    hipDeviceSynchronize();

    // Initialize
    read_input(h_flow_vector_array, h_random_numbers, p);
    hipDeviceSynchronize();

#ifndef CUDA_8_0
    // Copy to device
    hipStatus = hipMemcpy(d_flow_vector_array, h_flow_vector_array, n_flow_vectors * sizeof(flowvector), hipMemcpyHostToDevice);
    hipStatus = hipMemcpy(d_random_numbers, h_random_numbers, 2 * p.max_iter * sizeof(int), hipMemcpyHostToDevice);
    hipStatus = hipMemcpy(d_model_candidate, h_model_candidate, p.max_iter * sizeof(int), hipMemcpyHostToDevice);
    hipStatus = hipMemcpy(d_outliers_candidate, h_outliers_candidate, p.max_iter * sizeof(int), hipMemcpyHostToDevice);
    hipStatus = hipMemcpy(d_model_param_local, h_model_param_local, 4 * p.max_iter * sizeof(float), hipMemcpyHostToDevice);
    hipStatus = hipMemcpy(d_g_out_id, h_g_out_id, sizeof(int), hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
#endif

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));
#ifdef CUDA_8_0
        h_g_out_id[0].store(0);
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#else
        h_g_out_id[0] = 0;
        hipStatus = hipMemcpy(d_model_candidate, h_model_candidate, p.max_iter * sizeof(int), hipMemcpyHostToDevice);
        hipStatus = hipMemcpy(d_outliers_candidate, h_outliers_candidate, p.max_iter * sizeof(int), hipMemcpyHostToDevice);
        hipStatus = hipMemcpy(d_model_param_local, h_model_param_local, 4 * p.max_iter * sizeof(float), hipMemcpyHostToDevice);
        hipStatus = hipMemcpy(d_g_out_id, h_g_out_id, sizeof(int), hipMemcpyHostToDevice);
        if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
#endif
        hipDeviceSynchronize();

        //m5_work_begin(0, 0);

        // Launch GPU threads
        // Kernel launch
        if(p.n_gpu_blocks > 0) {
            hipStatus = call_RANSAC_kernel_block(p.n_gpu_blocks, p.n_gpu_threads, n_flow_vectors, p.max_iter, 
                p.error_threshold, p.convergence_threshold, p.max_iter, p.alpha, d_model_param_local, 
                d_flow_vector_array, d_random_numbers, d_model_candidate, d_outliers_candidate, (int*)d_g_out_id, 
                sizeof(int)
#ifdef CUDA_8_0
                + sizeof(int), (int*)worklist
#endif
                );
            if(hipStatus != hipSuccess) { fprintf(stderr, "HIP error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        }
        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_model_candidate, h_outliers_candidate, h_model_param_local,
            h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
            p.convergence_threshold, h_g_out_id, p.n_threads, p.max_iter, p.alpha
#ifdef CUDA_8_0
            ,
            worklist);
#else
            );
#endif

        hipDeviceSynchronize();
        main_thread.join();

        //m5_work_end(0, 0);

#ifndef CUDA_8_0
        // Copy back
        int d_candidates = 0;
        if(p.alpha < 1.0) {
            hipStatus = hipMemcpy(&d_candidates, d_g_out_id, sizeof(int), hipMemcpyDeviceToHost);
            hipStatus = hipMemcpy(&h_model_candidate[h_g_out_id[0]], d_model_candidate, d_candidates * sizeof(int), hipMemcpyDeviceToHost);
            hipStatus = hipMemcpy(&h_outliers_candidate[h_g_out_id[0]], d_outliers_candidate, d_candidates * sizeof(int), hipMemcpyDeviceToHost);
            if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
        }
        h_g_out_id[0] += d_candidates;
        hipDeviceSynchronize();
#endif

        // Post-processing (chooses the best model among the candidates)
        for(int i = 0; i < h_g_out_id[0]; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
    }

    // Verify answer
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
        p.convergence_threshold, h_g_out_id[0], best_outliers);

    // Free memory
#ifdef CUDA_8_0
    free(h_model_candidate);
    free(h_outliers_candidate);
    free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);
    free(worklist);
#else
    free(h_model_candidate);
    free(h_outliers_candidate);
    free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);
    hipStatus = hipFree(d_model_candidate);
    hipStatus = hipFree(d_outliers_candidate);
    hipStatus = hipFree(d_model_param_local);
    hipStatus = hipFree(d_g_out_id);
    hipStatus = hipFree(d_flow_vector_array);
    hipStatus = hipFree(d_random_numbers);
#endif
    if(hipStatus != hipSuccess) { fprintf(stderr, "CUDA error: %s\n at %s, %d\n", hipGetErrorString(hipStatus), __FILE__, __LINE__); exit(-1); };;
    hipDeviceSynchronize();

    printf("Test Passed\n");
    return 0;
}
