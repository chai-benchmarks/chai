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

#include <stdio.h>
#include <unistd.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>

// Params ---------------------------------------------------------------------
struct Params {

    int         n_work_items;
    const char *file_name;
    const char *output_file;
    int         n_frames;
    int         display;

    Params(int argc, char **argv) {
        n_frames     = 110;
        n_work_items = 16;
        file_name    = "input/PeppaPigandSuzieSheepWhistle.mov";
        output_file  = "./";
        int opt;
        while((opt = getopt(argc, argv, "hi:f:n:o:x")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_work_items    = atoi(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'n': n_frames        = atoi(optarg); break;
            case 'o': output_file     = optarg; break;
            case 'x': display         = 1; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./convert [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -i <I>    tile dim (default=32)"
                "\n    -n <N>    # frames to convert (default=110)"
                "\n    -f <F>    input video file name (default=input/PeppaPigandSuzieSheepWhistle.mov)"
                "\n    -o <O>    output folder (default=./)"
                "\n    -x        display output video"
                "\n");
    }
};

// Main ---------------------------------------------------------------------
int main(int argc, char **argv) {

    Params p(argc, argv);

    cv::VideoCapture cap(p.file_name);
    if(!cap.isOpened()) { // if not success, exit program
        printf("Cannot open the video file\n");
        exit(EXIT_FAILURE);
    }

    for(int task_id = 0; task_id < p.n_frames; task_id++) {

        cv::Mat in_frame, gray_frame;

        // Read frames
        cap.read(in_frame);

        // Convert to grayscale
        cv::cvtColor(in_frame, gray_frame, cv::COLOR_BGR2GRAY);
        int rowsc_ = gray_frame.rows;
        int colsc_ = gray_frame.cols;

        // Image dimensions
        int rowsc = ((gray_frame.rows - 2) / p.n_work_items) * p.n_work_items + 2;
        int colsc = ((gray_frame.cols - 2) / p.n_work_items) * p.n_work_items + 2;

        // Use these row/cols to create a rectangle which will serve as our crop
        cv::Rect croppedArea(0, 0, colsc, rowsc);
				
        // Crop the image and clone it. If it is not cloned, the layout does not change
        gray_frame = gray_frame(croppedArea).clone();

        // Write into file
        char FileName[100];
        sprintf(FileName, "%s%d.txt", p.output_file, task_id);
        FILE *fp = fopen(FileName,"w");
        fprintf(fp, "%d\n", gray_frame.rows);
        fprintf(fp, "%d\n", gray_frame.cols);
        for(int i=0; i<gray_frame.rows; i++){
            for(int j=0; j<gray_frame.cols; j++){
                fprintf(fp, "%d ", gray_frame.data[i*gray_frame.cols+j]);
            }
        }
        fprintf(fp, "\n");
        fclose(fp);

        // Display the video
        if(p.display){
            cv::Mat out_frame = gray_frame; 
            if(!out_frame.empty())
                imshow("canny", out_frame);
            if(cv::waitKey(30) >= 0)
                break;
        }
    }

    printf("Done\n");
    return 0;
}
