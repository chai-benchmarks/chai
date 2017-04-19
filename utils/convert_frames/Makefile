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

ifndef CXX
    CXX=g++
else
    ifeq ($(CXX),c++)
    CXX=g++
    endif
endif

LIB=-L/usr/lib/ -lm -lopencv_core -lopencv_highgui -lopencv_imgproc 

INC=-I.

SRC=main.cpp
OBJECTS=$(SRC:.cpp=.o)
DEPS=$(OBJECTS:.o=.d)
EXE=convert

.PHONY: all
all: $(EXE)

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INC) -c -MMD -o $@ $<

%.d: ;
.PRECIOUS: %.d

$(EXE): $(OBJECTS)
	$(CXX) $(CXX_FLAGS) $(OBJECTS) $(LIB) -o $(EXE) 

clean:
	rm -f $(EXE) *.o *.d

-include $(DEPS)
