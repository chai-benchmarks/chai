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
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(
    unsigned int *histo, unsigned int *data, int size, int bins, int num_threads, int chunk, int cpu_bins) {
    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < num_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {

            // Output partition
            int bins_per_thread = cpu_bins / num_threads;
            int my_bins_start   = k * bins_per_thread;
            int my_bins_end     = my_bins_start + bins_per_thread;

            unsigned int Hs[bins_per_thread];
            // Local histogram initialization
            for(int i = 0; i < bins_per_thread; i++) {
                Hs[i] = 0;
            }

            const int begin = 0;
            const int end   = size;

            for(int i = begin; i < end; i++) {
                // Read pixel
                unsigned int d = ((data[i] * bins) >> 12);

                if(d >= my_bins_start && d < my_bins_end) {
                    // Vote in histogram
                    Hs[d - my_bins_start]++;
                }
            }

            // Merge to global histogram
            for(int i = 0; i < bins_per_thread; i++) {
                histo[i + my_bins_start] += Hs[i];
            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
