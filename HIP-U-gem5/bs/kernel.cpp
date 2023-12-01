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
#include "support/partitioner.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

// BezierBlend (http://paulbourke.net/geometry/bezier/)
T BezierBlend(int k, T mu, int n) {
    int nn, kn, nkn;
    T   blend = 1;
    nn        = n;
    kn        = k;
    nkn       = n - k;
    while(nn >= 1) {
        blend *= nn;
        nn--;
        if(kn > 1) {
            blend /= (T)kn;
            kn--;
        }
        if(nkn > 1) {
            blend /= (T)nkn;
            nkn--;
        }
    }
    if(k > 0)
        blend *= pow(mu, (T)k);
    if(n - k > 0)
        blend *= pow(1 - mu, (T)(n - k));
    return (blend);
}

// CPU threads-----------------------------------------------------------------
void run_cpu_threads(XYZ *in, XYZ *outp, int n_tasks, float alpha, int n_threads, int n_gpu_threads, int in_size_i, int in_size_j,
    int out_size_i, int out_size_j
#ifdef CUDA_8_0
    , std::atomic_int *worklist
#endif
    ) {

    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {

#ifdef CUDA_8_0
            Partitioner p = partitioner_create(n_tasks, alpha, k, n_threads, worklist);
#else
            Partitioner p = partitioner_create(n_tasks, alpha, k, n_threads);
#endif

            const int wg_in_J = divceil(out_size_j, n_gpu_threads);
            const int wg_in_I = divceil(out_size_i, n_gpu_threads);

            for(int t = cpu_first(&p); cpu_more(&p); t = cpu_next(&p)) {
                const int my_s1 = t / wg_in_J;
                const int my_s0 = t % wg_in_J;

                int Row = my_s1 * n_gpu_threads;
                int Col = my_s0 * n_gpu_threads;
                T   bi;
                T   bj;
                T   mui, muj;

                for(int i = Row; i < Row + n_gpu_threads; i++) {
                    mui = i / (T)(out_size_i - 1);
                    for(int j = Col; j < Col + n_gpu_threads; j++) {
                        muj = j / (T)(out_size_j - 1);
                        if(i < out_size_i && j < out_size_j) {
                            XYZ out = {0, 0, 0};
#pragma unroll
                            for(int ki = 0; ki <= in_size_i; ki++) {
                                bi = BezierBlend(ki, mui, in_size_i);
#pragma unroll
                                for(int kj = 0; kj <= in_size_j; kj++) {
                                    bj = BezierBlend(kj, muj, in_size_j);
                                    out.x += (in[ki * (in_size_j + 1) + kj].x * bi * bj);
                                    out.y += (in[ki * (in_size_j + 1) + kj].y * bi * bj);
                                    out.z += (in[ki * (in_size_j + 1) + kj].z * bi * bj);
                                }
                            }
                            outp[i * out_size_j + j] = out;
                        }
                    }
                }
            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
