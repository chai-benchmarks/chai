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

inline int verify(std::atomic_int *h_cost, int num_of_nodes, const char *file_name) {
// Compare to output file
#if PRINT
    printf("Comparing outputs...\n");
#endif
    FILE *fpo = fopen(file_name, "r");
    if(!fpo) {
        printf("Error Reading output file\n");
        exit(EXIT_FAILURE);
    }
#if PRINT
    printf("Reading Output: %s\n", file_name);
#endif

    // the number of nodes in the output
    int num_of_nodes_o = 0;
    fscanf(fpo, "%d", &num_of_nodes_o);
    if(num_of_nodes != num_of_nodes_o) {
        printf("FAIL: Number of nodes does not match the expected value\n");
        exit(EXIT_FAILURE);
    }

    // cost of nodes in the output
    for(int i = 0; i < num_of_nodes_o; i++) {
        int j, cost;
        fscanf(fpo, "%d %d", &j, &cost);
        if(i != j || h_cost[i].load() * -1 != cost) {
            printf("FAIL: Computed node %d cost (%d != %d) does not match the expected value\n", i, h_cost[i].load(), 
                cost);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fpo);
    return 0;
}
