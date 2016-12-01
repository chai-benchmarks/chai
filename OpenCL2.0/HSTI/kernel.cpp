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
void run_cpu_threads(std::atomic_uint *histo, unsigned int *data, int size, int bins, int num_threads, int chunk,
    Partitioner p
#ifdef OCL_2_0
    ,
    std::atomic_int *wl) {
#else
    ) {
#endif
    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < num_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {

            unsigned int Hs[bins];
            // Local histogram initialization
            for(int i = 0; i < bins; i++) {
                Hs[i] = 0;
            }

#ifdef OCL_2_0
            for(int i = cpu_first(&p, k, wl); cpu_more(&p, i); i = cpu_next(&p, i, num_threads, wl)) {
#else
            for(int i = cpu_first(&p, k); cpu_more(&p, i); i = cpu_next(&p, i, num_threads)) {
#endif
                for(int j = 0; j < chunk; j++) {
                    // Read pixel
                    unsigned int d = ((data[i * chunk + j] * bins) >> 12);

                    // Vote in histogram
                    Hs[d]++;
                }
            }

            // Merge to global histogram
            for(int i = 0; i < bins; i++) {
                (&histo[i])->fetch_add(Hs[i]);
            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
