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
 1. BFS:                         src/gpu-compute/compute\_unit.cc:565: panic: panic condition (numWfs * vregDemandPerWI) > (numVectorALUs * numVecRegsPerSimd) occurred: WG with 1 WFs and 29285 VGPRs per WI can not be allocated to CU that has 8192 VGPRs

                                 panic condition (numWfs * sregDemandPerWI) > numScalarRegsPerSimd occurred: WG with 1 WFs and 26656 SGPRs per WI can not be scheduled to CU with 2048 SGPRs

                                 std::bad_alloc -- memory exhaustion mid-program run (different from docker memory exhaustion)

 2. BS:     Works (try CUDA_8)

 3. CEDD:   Works                >24 hrs; no new misses from CPU/GPU; both CPU and GPU threads have been launched; verified!

    CEDT:                        >24 hrs; failed due to server mem exhaustion?; Retry!

 4. HSTI:   Works

    HSTO:                        >12 hrs; unterminated

 5. PAD:    Works (try CUDA_8)   cannot allocate enough memory (unconvinced)A

 6. RSCD:                        

    RSCT:

 7. SC:                          Unrolling fails; cannot allocate memory (possibly due to server memory exhaustion?)
                                 
 8. SSSP:                        Stuck (no GPU progress, unterminated)

 9. TQ:     Works

    TQH:                         >12 hrs; unterminated

10. TRNS:                        std::bad_alloc -- src/central_freelist.cc:333] tcmalloc: allocation failed 16384 -- server memory issue. Stuck (no GPU progress, unterminated)
``
