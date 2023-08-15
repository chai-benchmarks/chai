CPPSRC = main.cpp kernel.cu kernel.cpp
EXECUTABLE = bfs.gem5
# BC has a compilation error in host code with ROCm 4.0, so O2 and O3 do not
# pass even on real GPUs
OPTS = -O1

HIP_PATH ?= /opt/rocm/hip
HIPCC = $(HIP_PATH)/bin/hipcc

# these are needed for m5ops
GEM5_PATH ?= $(shell pwd)/../gem5
CFLAGS += -I$(GEM5_PATH)/include  -DDEBUG -g -O0 -lz -lm -lc 
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5 -lpthread -pthread

BIN_DIR ?= ./bin

all: $(BIN_DIR)/$(EXECUTABLE)

$(BIN_DIR)/$(EXECUTABLE): $(CPPSRC) $(BIN_DIR)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801,gfx803 $(CXXFLAGS) $(CPPSRC) -DGEM5_FUSION -o $(BIN_DIR)/$(EXECUTABLE) $(CFLAGS) $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: bc clean
