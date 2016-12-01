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

float gaus[3][3] = {{0.0625, 0.125, 0.0625}, {0.1250, 0.250, 0.1250}, {0.0625, 0.125, 0.0625}};
// Some of the available convolution kernels
int sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
int soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(unsigned char *buffer0, unsigned char *buffer1, unsigned char *theta, int rows, int cols,
    int num_threads, int t_index) {

    // NON-MAXIMUM SUPPRESSION KERNEL
    std::vector<std::thread> cpu_threads_nonmax;
    for(int k = 0; k < num_threads; k++) {
        cpu_threads_nonmax.push_back(std::thread([=]() {

            unsigned char *data = buffer0;
            unsigned char *out  = buffer1;
            for(int row = k + 1; row < rows - 1; row += num_threads) {
                for(int col = 1; col < cols - 1; col++) {
                    // These variables are offset by one to avoid seg. fault errors
                    // As such, this kernel ignores the outside ring of pixels

                    // The following variables are used to address the matrices more easily
                    const size_t POS = row * cols + col;
                    const size_t N   = (row - 1) * cols + col;
                    const size_t NE  = (row - 1) * cols + (col + 1);
                    const size_t E   = row * cols + (col + 1);
                    const size_t SE  = (row + 1) * cols + (col + 1);
                    const size_t S   = (row + 1) * cols + col;
                    const size_t SW  = (row + 1) * cols + (col - 1);
                    const size_t W   = row * cols + (col - 1);
                    const size_t NW  = (row - 1) * cols + (col - 1);

                    switch(theta[POS]) {
                    // A gradient angle of 0 degrees = an edge that is North/South
                    // Check neighbors to the East and West
                    case 0:
                        // supress me if my neighbor has larger magnitude
                        if(data[POS] <= data[E] || data[POS] <= data[W]) {
                            out[POS] = 0;
                        }
                        // otherwise, copy my value to the output buffer
                        else {
                            out[POS] = data[POS];
                        }
                        break;

                    // A gradient angle of 45 degrees = an edge that is NW/SE
                    // Check neighbors to the NE and SW
                    case 45:
                        // supress me if my neighbor has larger magnitude
                        if(data[POS] <= data[NE] || data[POS] <= data[SW]) {
                            out[POS] = 0;
                        }
                        // otherwise, copy my value to the output buffer
                        else {
                            out[POS] = data[POS];
                        }
                        break;

                    // A gradient angle of 90 degrees = an edge that is E/W
                    // Check neighbors to the North and South
                    case 90:
                        // supress me if my neighbor has larger magnitude
                        if(data[POS] <= data[N] || data[POS] <= data[S]) {
                            out[POS] = 0;
                        }
                        // otherwise, copy my value to the output buffer
                        else {
                            out[POS] = data[POS];
                        }
                        break;

                    // A gradient angle of 135 degrees = an edge that is NE/SW
                    // Check neighbors to the NW and SE
                    case 135:
                        // supress me if my neighbor has larger magnitude
                        if(data[POS] <= data[NW] || data[POS] <= data[SE]) {
                            out[POS] = 0;
                        }
                        // otherwise, copy my value to the output buffer
                        else {
                            out[POS] = data[POS];
                        }
                        break;

                    default: out[POS] = data[POS]; break;
                    }
                }
            }

        }));
    }
    std::for_each(cpu_threads_nonmax.begin(), cpu_threads_nonmax.end(), [](std::thread &t) { t.join(); });

    // HYSTERESIS KERNEL
    std::vector<std::thread> cpu_threads_hyst;
    for(int k = 0; k < num_threads; k++) {
        cpu_threads_hyst.push_back(std::thread([=]() {

            unsigned char *data = buffer1;
            unsigned char *out  = buffer0;
            // Establish our high and low thresholds as floats
            float        lowThresh  = 10;
            float        highThresh = 70;
            const size_t EDGE       = 255;
            for(int row = k + 1; row < rows - 1; row += num_threads) {
                for(int col = 1; col < cols - 1; col++) {
                    size_t pos = row * cols + col;

                    if(data[pos] >= highThresh)
                        out[pos] = EDGE;
                    else if(data[pos] <= lowThresh)
                        out[pos] = 0;
                    else {
                        float med = (highThresh + lowThresh) / 2;

                        if(data[pos] >= med)
                            out[pos] = EDGE;
                        else
                            out[pos] = 0;
                    }
                }
            }

        }));
    }
    std::for_each(cpu_threads_hyst.begin(), cpu_threads_hyst.end(), [](std::thread &t) { t.join(); });
}
