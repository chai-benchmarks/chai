# 
#  Copyright (c) 2016 University of Cordoba and University of Illinois
#  All rights reserved.
# 
#  Developed by:    IMPACT Research Group
#                   University of Cordoba and University of Illinois
#                   http://impact.crhc.illinois.edu/
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  with the Software without restriction, including without limitation the 
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#       > Redistributions of source code must retain the above copyright notice,
#         this list of conditions and the following disclaimers.
#       > Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimers in the
#         documentation and/or other materials provided with the distribution.
#       > Neither the names of IMPACT Research Group, University of Cordoba, 
#         University of Illinois nor the names of its contributors may be used 
#         to endorse or promote products derived from this Software without 
#         specific prior written permission.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#  CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
#  THE SOFTWARE.
# 

NVCC=nvcc
CPP = /usr/bin/g++-4.8
CC = /usr/bin/gcc-4.8

COMPUTETARGET ?= sm_20
CFLAGS=-Wl,--no-as-needed
CUFLAGS=-ccbin=/usr/bin/g++-4.6


# Set path to gem5-gpu
GEM5GPU=

INC=-I$(CUDAHOME)/include/ -I$(NVIDIA_CUDA_SDK_LOCATION)/common/inc/ -I$(GEM5GPU)/gem5/util/m5/ -I$(GEM5GPU)/benchmarks/libcuda/ -I$(NVIDIA_CUDA_SDK_LOCATION)/C/common/inc/
LIB=-L$(NVIDIA_CUDA_SDK_LOCATION)/lib -L/usr/lib64 -lcutil_x86_64 -lm5op_x86

INTERMED_FILES := *.cpp*.i *.cpp*.ii *.cu.c *.cudafe*.* *.fatbin.c *.cu.cpp *.linkinfo *.cpp_o core

CCFILES=kernel.cpp main.cpp
CUFILES=kernel.cu

EXE=bs

SIM_OBJDIR :=
SIM_OBJS +=  $(patsubst %.cpp,$(SIM_OBJDIR)%.cpp_o,$(CCFILES))
SIM_OBJS +=  $(patsubst %.cu,$(SIM_OBJDIR)%.cu_o,$(CUFILES))


.SUFFIXES:

$(EXE): $(SIM_OBJS)
	$(CPP) $(CFLAGS) -DGEM5_FUSION $(notdir $(SIM_OBJS)) -L$(GEM5GPU)/benchmarks/libcuda -lcuda \
                $(LIB) -O3 -lz -static -static-libgcc -o $(EXE) -lm -lc $(EXTRA_OBJS) -Wl,--whole-archive -lpthread -Wl,--no-whole-archive

%.cpp_o: %.cpp
	$(CPP) -Wl,--no-as-needed $(CCFLAGS) $(INCFLAGEXTRA) -Wall -O3 $(INC) $(LIB) -c $< -o $(notdir $@) -std=c++0x -pthread

%.cu_o: %.cu
	nvcc $(CUFLAGS) -Xcompiler -Wall,-O3 -c -arch $(COMPUTETARGET) --keep --compiler-options "-fno-strict-aliasing" \
            -DGEM5_FUSION $(INC)  $(INCFLAGEXTRA) $(LIB) -lcutil -DUNIX -O3 $< -o $(EXE)
	python $(GEM5GPU)/benchmarks/common/sizeHack.py -f $<.cpp -t $(COMPUTETARGET)

	$(CPP) $(CFLAGS) -Wall -O3 -c $(notdir $<.cpp) -o $(notdir $@)
	echo $(SIM_OBJS)

%.cu: %.cu.c


all: $(EXE)

clean_all:
	rm -f $(INTERMED_FILES) *.cubin *.o *_o *.hash *.ptx *.ptxinfo cubin.bin $(EXE)

clean:
	rm -f $(INTERMED_FILES) *.cubin *.o *_o *.hash *.ptx *.ptxinfo cubin.bin

