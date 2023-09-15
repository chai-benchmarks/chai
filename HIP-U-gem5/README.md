## Instructions to compile for gem5
1. Download (or, in the Makefiles, update the path to) gem5:
`git clone https://github.com/gem5/gem5`
2. Employ the published docker image to compile libm5.a:
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 scons -C gem5/util/m5  x86.CROSS_COMPILE=x86_64-linux-gnu- build/x86/out/libm5.a`
PS: if the gem5 directory is not a strict-subdirectory of $(pwd), docker might not find it; the solution is to launch docker from at least the same directory as gem5, or upwards.
3. Compile the benchmark of choice (for example, BFS):
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 make -B -C BFS`

## Instructions to run on gem5
PS: The tests are run against gem5's GCN3 model using apu\_se.py config.
Run from the directory containing gem5 and gem5-resources:
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt -d chai_cedd/ gem5/configs/example/apu_se.py --cpu-type=DerivO3CPU --num-cpus=8 --mem-size=4GB -c gem5-resources/src/gpu/chai/HIP-U-gem5/CEDD/bin/cedd.gem5 --options="-f gem5-resources/src/gpu/chai/HIP-U-gem5/BS/input/peppa/ <any other arguments as required by the program>" 1>chai_cedd_simout 2>chai_cedd_simerr`

## Errata
```
 1. BFS:                         gem5.opt: src/mem/ruby/system/GPUCoalescer.cc:607: void gem5::ruby::GPUCoalescer::hitCallback(gem5::ruby::CoalescedRequest*, gem5::ruby::MachineType, gem5::ruby::DataBlock&, bool, gem5::Cycles, gem5::Cycles, gem5::Cycles, bool): Assertion `data.numAtomicLogEntries() == 0' failed.

 2. BS:     Works (try CUDA_8)

 3. CEDD:   Works

    CEDT:                        Stuck (no GPU progress, unterminated)

 4. HSTI:   Works

    HSTO:                        Stuck (no GPU progress, unterminated)

 5. PAD:    Works (try CUDA_8)

 6. RSCD:                        

    RSCT:   Test failed

 7. SC:                          
                                 
 8. SSSP:                        Stuck (no GPU progress, unterminated)

 9. TQ:     Works

    TQH:                         Stuck (no GPU progress, unterminated)

10. TRNS:                        Stuck (no GPU progress, unterminated)
```
