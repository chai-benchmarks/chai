CPPSRC = main.cpp kernel.cu kernel.cpp
EXECUTABLE = bfs.gem5
# BC has a compilation error in host code with ROCm 4.0, so O2 and O3 do not
# pass even on real GPUs
OPTS =

HIP_PATH ?= /opt/rocm/hip
HIPCC = $(HIP_PATH)/bin/hipcc

# these are needed for m5ops
GEM5_PATH ?= $(shell pwd)/../.gem5
CFLAGS += -I$(GEM5_PATH)/include -lz -lm -lc -lpthread -O1
CFLAGS += -DENABLE_PARSEC_HOOKS -I$(shell pwd)/../.roi/include -L$(shell pwd)/../.roi/lib/ -Wl,-rpath,/home/anoop/new/gem5-resources/src/gpu/chai/HIP-U-gem5/BFS/bin -l:libhooks.so
LDFLAGS += -L$(GEM5_PATH)/util/m5/build/x86/out -lm5

BIN_DIR ?= ./bin

all: $(BIN_DIR)/$(EXECUTABLE)

$(BIN_DIR)/$(EXECUTABLE): $(CPPSRC) $(BIN_DIR)
	$(HIPCC) $(OPTS) --amdgpu-target=gfx801,gfx803 $(CXXFLAGS) $(CPPSRC) -o $(BIN_DIR)/$(EXECUTABLE) $(CFLAGS) $(LDFLAGS)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: clean
