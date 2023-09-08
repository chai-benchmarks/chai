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

// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(T *matrix_out, T *matrix, std::atomic_int *flags, int n, int m, int pad, int n_threads, int ldim, int n_tasks, float alpha
    , std::atomic_int *worklist
    ) {

    const int                REGS_CPU = REGS * ldim;
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {
    
        cpu_threads.push_back(std::thread([=]() {

            Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads, worklist);

            const int matrix_size       = m * (n + pad);
            const int matrix_size_align = (matrix_size + ldim * REGS - 1) / (ldim * REGS) * (ldim * REGS);

            for(int my_s = cpu_first(&p); cpu_more(&p); my_s = cpu_next(&p)) {

                // Declare on-chip memory
                T   reg[REGS_CPU];
                int pos      = matrix_size_align - 1 - (my_s * REGS_CPU);
                int my_s_row = pos / (n + pad);
                int my_x     = pos % (n + pad);
                int pos2     = my_s_row * n + my_x;
// Load in on-chip memory
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(pos2 >= 0 && my_x < n && pos2 < matrix_size)
                        reg[j] = matrix[pos2];
                    else
                        reg[j] = 0;
                    pos--;
                    my_s_row = pos / (n + pad);
                    my_x     = pos % (n + pad);
                    pos2     = my_s_row * n + my_x;
                }

                // Set global synch
                while((&flags[my_s])->load() == 0) {
                }
                (&flags[my_s + 1])->fetch_add(1);

                // Store to global memory
                pos = matrix_size_align - 1 - (my_s * REGS_CPU);
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(pos >= 0 && pos < matrix_size)
                        matrix_out[pos] = reg[j];
                    pos--;
                }
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
