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

#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

inline int compare_output(T *output, T *ref, int dim) {
    int i;
    for(i = 0; i < dim; i++) {
        T diff = fabs(ref[i] - output[i]);
        if((diff - 0.0f) > 0.00001f && diff > 0.01 * fabs(ref[i])) {
            printf("line: %d ref: %f actual: %f diff: %f\n", i, ref[i], output[i], diff);
            exit(EXIT_FAILURE);
        }
    }
    return 0;
}

// Sequential transposition for comparison purposes
//[w][h/t][t] to [h/t][w][t]
inline void cpu_soa_asta(T *src, T *dst, int height, int width, int tile_size) {
    // We only support height == multiple of tile size
    if((height / tile_size) * tile_size == height)
        for(int k = 0; k < width; k++) {
            for(int i = 0; i < height / tile_size; i++) { //For all tiles
                for(int j = 0; j < tile_size; j++) {
                    //from src[k][i][j] to dst[i][k][j]
                    dst[i * width * tile_size + k * tile_size + j] = src[k * height + i * tile_size + j];
                }
            }
        }
}

inline void verify(T *input2, T *input, int height, int width, int tile_size) {
    T *output = (T *)malloc(width * height * sizeof(T));
    cpu_soa_asta(input, output, height, width, tile_size);
    compare_output(input2, output, height * width);
    free(output);
}
