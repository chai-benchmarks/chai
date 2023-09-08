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
void run_cpu_threads(T *output, T *input, std::atomic_int *flags, int size, int value, int n_threads, int ldim,
    int n_tasks, float alpha
    , std::atomic_int *worklist
    ) {

    const int                REGS_CPU = REGS * ldim;
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {
        cpu_threads.push_back(std::thread([=]() {

            Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads, worklist);

            for(int my_s = cpu_first(&p); cpu_more(&p); my_s = cpu_next(&p)) {

                int l_count = 0;
                // Declare on-chip memory
                T   reg[REGS_CPU];
                int pos = my_s * REGS_CPU;
// Load in on-chip memory
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(pos < size) {
                        reg[j] = input[pos];
                        if(reg[j] != value)
                            l_count++;
                    } else
                        reg[j] = value;
                    pos++;
                }

                // Set global synch
                int p_count;
                while((p_count = (&flags[my_s])->load()) == 0) {
                }
                (&flags[my_s + 1])->fetch_add(p_count + l_count);
                l_count = p_count - 1;

                // Store to global memory
                pos = l_count;
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(reg[j] != value) {
                        output[pos] = reg[j];
                        pos++;
                    }
                }
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
