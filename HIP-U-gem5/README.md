## Instructions to compile for gem5
1. Download (or in the Makefiles, update the path to) gem5:
`git clone https://github.com/gem5/gem5`
2. Employ the published docker image to compile libm5.a:
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 scons -C gem5/util/m5  x86.CROSS_COMPILE=x86_64-linux-gnu- build/x86/out/libm5.a`
3. Compile the benchmark of choice (for example, BFS):
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 make -C BFS -B`

## Instructions to run on gem5
PS: The tests are run against gem5's GCN3 model using apu\_se.py config.
Run from the directory containing gem5 and gem5-resources:
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt -d chai_cedd/ gem5/configs/example/apu_se.py --cpu-type=DerivO3CPU --num-cpus=8 --mem-size=4GB --ruby --mem-type=SimpleMemory --benchmark-root=gem5-resources/src/gpu/chai/HIP-U-gem5 -c CEDD/bin/cedd.gem5 --options="-f gem5-resources/src/gpu/chai/HIP-U-gem5/BS/input/peppa/" 1>chai_cedd_simout 2>chai_cedd_simerr`
(PS: `--options="<fill accordingly>"`)
## Errata
Does not work:
 1. BFS: src/gpu-compute/compute\_unit.cc:565: panic: panic condition (numWfs * vregDemandPerWI) > (numVectorALUs * numVecRegsPerSimd) occurred: WG with 1 WFs and 29285 VGPRs per WI can not be allocated to CU that has 8192 VGPRs
 2. BS: Works
 3. CEDD: >24 hrs; no new misses from CPU/GPU; both CPU and GPU threads have been launched; unterminated
 4. CEDT:
 5. HSTI: Failed: ds_add_u32
    HSTO:
 6. PAD:  Unrolling fails; Running ...
 7. RSCD: Failed: ds_add_u32
    RSCT:
 8. SC:   Unrolling fails;   WAIT
 9. SSSP:                    WAIT
10. TQ:                      WAIT
11. TRNS:                    WAIT
