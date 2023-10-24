## Instructions to compile for gem5
1. Download (or, in the Makefiles, update the path to) gem5:
`git clone https://github.com/gem5/gem5`
2. Employ the published docker image to compile libm5.a:
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 scons -C gem5/util/m5  x86.CROSS_COMPILE=x86_64-linux-gnu- build/x86/out/libm5.a`
PS: if the gem5 directory is not a strict-subdirectory of $(pwd), docker might not find it; the solution is to launch docker from at least the same directory as gem5, or upwards.
3. Compile the benchmark of choice (for example, BFS):
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 make -B -C <program> GEM5_PATH=<path to gem5>`
PS: the compiled binary will be in `<program>/bin/<program>`

## Instructions to run on gem5
PS: The tests are run against gem5's GCN3 model using apu\_se.py config.
Run from the directory containing gem5 and gem5-resources:
`docker run -u $UID:$GID --volume $(pwd):$(pwd) -w $(pwd) gcr.io/gem5-test/gcn-gpu:v22-1 gem5/build/GCN3_X86/gem5.opt -d chai_cedd/ gem5/configs/example/apu_se.py --cpu-type=DerivO3CPU --num-cpus=8 --mem-size=4GB -c gem5-resources/src/gpu/chai/HIP-U-gem5/CEDD/bin/cedd.gem5 --options="-f gem5-resources/src/gpu/chai/HIP-U-gem5/BS/input/peppa/ <any other arguments as required by the program>" 1>chai_cedd_simout 2>chai_cedd_simerr`

## Errata
Data Paritioning:
```
Both  BS     Works
Main  CEDD   Works                (with develop) Doesn't terminate
Main  HSTI   Works                (with develop) Doesn't terminate
Dev   HSTO   Works                (with main)    Doesn't terminate
Both  PAD    Works
Dev   SC     Works                (with gfx801)
Dev   RSCD   Verification failed  (same as RSCT)
      TRNS                        src/mem/port.cc:209: panic: panic condition !ext occurred: There is no TracingExtension in the packet.
                                  Random matrix size flexibility
```

Fine-grained Task Partitioning:
```
Both  RSCT   Verification failed 
                                  The best fitting model computed by the verification code does not match the model identified by GPU+CPU
                                    will need to contact CHAI folks for algorithm insight (or read the reference paper)
Both  TQ     Works 
Main  TQH    Completes with reduced data, fails verification
                                  (with develop) (same as SC)
```

Coarse-grained Task Partitioning:
```
      BFS                         https://github.com/farkhor/PaRMAT -- figure out how to format correctly
Dev   CEDT   Works                (with gfx902) unimplemented instruction -- v_add_u32_e32
      SSSP                        Futex syscall -- returns 0 and waits perpetually (Debug how?)
                                  (same as BFS)
```
