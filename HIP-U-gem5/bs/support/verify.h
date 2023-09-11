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

inline int compare_output(XYZ *outp, XYZ *outpCPU, int NI, int NJ, int RESOLUTIONI, int RESOLUTIONJ) {
    double sum_delta2, sum_ref2, L1norm2;
    sum_delta2 = 0;
    sum_ref2   = 0;
    L1norm2    = 0;
    for(int i = 0; i < RESOLUTIONI; i++) {
        for(int j = 0; j < RESOLUTIONJ; j++) {
            sum_delta2 += fabs(outp[i * RESOLUTIONJ + j].x - outpCPU[i * RESOLUTIONJ + j].x);
            sum_ref2 += fabs(outpCPU[i * RESOLUTIONJ + j].x);
            sum_delta2 += fabs(outp[i * RESOLUTIONJ + j].y - outpCPU[i * RESOLUTIONJ + j].y);
            sum_ref2 += fabs(outpCPU[i * RESOLUTIONJ + j].y);
            sum_delta2 += fabs(outp[i * RESOLUTIONJ + j].z - outpCPU[i * RESOLUTIONJ + j].z);
            sum_ref2 += fabs(outpCPU[i * RESOLUTIONJ + j].z);
        }
    }
    L1norm2 = (double)(sum_delta2 / sum_ref2);
    if(L1norm2 >= 1e-6){
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}

// BezierBlend (http://paulbourke.net/geometry/bezier/)
inline T BezierBlend(int k, T mu, int n) {
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

// Sequential implementation for comparison purposes
inline void BezierCPU(XYZ *inp, XYZ *outp, int NI, int NJ, int RESOLUTIONI, int RESOLUTIONJ) {
    int i, j, ki, kj;
    T   mui, muj, bi, bj;
    for(i = 0; i < RESOLUTIONI; i++) {
        mui = i / (T)(RESOLUTIONI - 1);
        for(j = 0; j < RESOLUTIONJ; j++) {
            muj     = j / (T)(RESOLUTIONJ - 1);
            XYZ out = {0, 0, 0};
            for(ki = 0; ki <= NI; ki++) {
                bi = BezierBlend(ki, mui, NI);
                for(kj = 0; kj <= NJ; kj++) {
                    bj = BezierBlend(kj, muj, NJ);
                    out.x += (inp[ki * (NJ + 1) + kj].x * bi * bj);
                    out.y += (inp[ki * (NJ + 1) + kj].y * bi * bj);
                    out.z += (inp[ki * (NJ + 1) + kj].z * bi * bj);
                }
            }
            outp[i * RESOLUTIONJ + j] = out;
        }
    }
}

inline void verify(XYZ *in, XYZ *out, int in_size_i, int in_size_j, int out_size_i, int out_size_j) {
    XYZ *gold = (XYZ *)malloc(out_size_i * out_size_j * sizeof(XYZ));
    BezierCPU(in, gold, in_size_i, in_size_j, out_size_i, out_size_j);
    compare_output(out, gold, in_size_i, in_size_j, out_size_i, out_size_j);
    free(gold);
}
