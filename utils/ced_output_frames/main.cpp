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
#include "support/common.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         n_threads;
    int         n_frames;
    const char *file_name;
    const char *output_file;
    int         display = 0;

    Params(int argc, char **argv) {
        n_threads   = 4;
        n_frames    = 110;
        file_name   = "input/peppa/";
        output_file = "./";
        int opt;
        while((opt = getopt(argc, argv, "ht:n:f:o:x")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 't': n_threads       = atoi(optarg); break;
            case 'n': n_frames        = atoi(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'o': output_file     = optarg; break;
            case 'x': display         = 1; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        assert(n_threads > 0 && "Invalid # of host workers!");
#ifndef CHAI_OPENCV
        assert(display != 1 && "Compile with CHAI_OPENCV");
#endif
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./output_frames [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -n <N>    # of frames (default=110)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input video file name (default=input/peppa/?.txt)"
                "\n    -o <O>    output folder (default=./)"
                "\n    -x        display output video (with CHAI_OPENCV)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char** all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {

    for(int task_id = 0; task_id < p.n_frames; task_id++) {

        char FileName[100];
        sprintf(FileName, "%s%d.txt", p.file_name, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

        fscanf(fp, "%d\n", &rowsc);
        fscanf(fp, "%d\n", &colsc);

        in_size = rowsc * colsc * sizeof(unsigned char);
        all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
        }
        fclose(fp);
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    Params p(argc, argv);

    // Initialize (part 1)
    unsigned char **all_gray_frames = (unsigned char **)malloc(p.n_frames * sizeof(unsigned char *));
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);

    // Allocate buffers
    unsigned char *h_in_out = (unsigned char *)malloc(in_size);
    unsigned char *h_interm_cpu_proxy = (unsigned char *)malloc(in_size);
    unsigned char *h_theta_cpu_proxy  = (unsigned char *)malloc(in_size);
    ALLOC_ERR(h_in_out, h_interm_cpu_proxy, h_theta_cpu_proxy);

    // Initialize (part 2)
    unsigned char **all_out_frames = (unsigned char **)malloc(p.n_frames * sizeof(unsigned char *));
    for(int i = 0; i < p.n_frames; i++) {
        all_out_frames[i] = (unsigned char *)malloc(in_size);
    }

    for(int task_id = 0; task_id < p.n_frames; task_id++) {

        // Next frame
        memcpy(h_in_out, all_gray_frames[task_id], in_size);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_interm_cpu_proxy, h_theta_cpu_proxy,
            rowsc, colsc, p.n_threads, task_id);
        main_thread.join();

        memcpy(all_out_frames[task_id], h_in_out, in_size);

        // Save to output file
        char FileName[100];
        sprintf(FileName, "%s%d.txt", p.output_file, task_id);
        FILE *fp = fopen(FileName,"w");
        for(int i=0; i<rowsc; i++){
            for(int j=0; j<colsc; j++){
                fprintf(fp, "%d ", all_out_frames[task_id][i*colsc+j]);
            }
        }
        fprintf(fp, "\n");
        fclose(fp);

    }

#ifdef CHAI_OPENCV
    // Display the result
    if(p.display){
        for(int rep = 0; rep < p.n_frames; rep++) {
            cv::Mat out_frame = cv::Mat(rowsc, colsc, CV_8UC1);
            memcpy(out_frame.data, all_out_frames[rep], in_size);
            if(!out_frame.empty())
                imshow("canny", out_frame);
            if(cv::waitKey(30) >= 0)
                break;
        }
    }
#endif

    // Release buffers
    free(h_in_out);
    free(h_interm_cpu_proxy);
    free(h_theta_cpu_proxy);
    for(int i = 0; i < p.n_frames; i++) {
        free(all_gray_frames[i]);
    }
    free(all_gray_frames);
    for(int i = 0; i < p.n_frames; i++) {
        free(all_out_frames[i]);
    }
    free(all_out_frames);

    printf("Done\n");
    return 0;
}
