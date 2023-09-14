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

inline int compare_output(unsigned char **all_out_frames, int image_size, const char *file_name, int num_frames, int rowsc, int colsc, int rowsc_, int colsc_) {

    int count_error = 0;
    for(int i = 0; i < num_frames; i++) {

        // Compare to output file
        char FileName[100];
        sprintf(FileName, "%s%d.txt", file_name, i);
        FILE *out_file = fopen(FileName, "r");
        if(!out_file) {
            printf("Error reading output file\n");
            return 1;
        }
#if PRINT
        printf("Reading Output: %s\n", file_name);
#endif

        for(int r = 0; r < rowsc; r++) {
            for(int c = 0; c < colsc; c++) {
                int pix;
                fscanf(out_file, "%d ", &pix);
                if((int)all_out_frames[i][r*colsc+c] != pix) {
                    if(r > 3 && r < rowsc-32 && c > 3 && c < colsc-32){
                        count_error++;
                    }
                }
            }
            // Scan until end of row
            if(colsc<colsc_) fscanf(out_file, "%*[^\n]\n");
        }
        // Scan until end of frame
        for(int rr=rowsc;rr<rowsc_;rr++) fscanf(out_file, "%*[^\n]\n");

        fclose(out_file);
    }

    if((float)count_error / (float)(image_size * num_frames) >= 1e-6){
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}

inline void verify(unsigned char **all_out_frames, int image_size, const char *file_name, int num_frames, int rowsc, int colsc, int rowsc_, int colsc_) {
    compare_output(all_out_frames, image_size, file_name, num_frames, rowsc, colsc, rowsc_, colsc_);
}
