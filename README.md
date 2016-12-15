# Chai
This is the Chai benchmark suite, a suite of Collaborative Heterogeneous Applications for Integrated-architectures. This repository currently includes OpenCL 1.2 and OpenCL 2.0 implementations of each benchmark.

The current version of Chai is 1.0-alpha.

#

If you think this work is useful, please cite us at https://github.com/chai-benchmarks/chai for now, until we provide another reference.

## Running OpenCL 1.2 benchmarks on NVIDIA with Docker

Install `docker` for your system.

Install `nvidia-docker`.

To build the docker image, use

    nvidia-docker build . -t chai

To run a benchmark (for example, BS), do

    nvidia-docker run -it chai bash -c "cd chai/OpenCL1.2/BS/ && ./bs"

## Docker: Running OpenCL 1.2 and 2.0 Benchmarks with the Intel OpenCL CPU Stacks

Install `docker` for your system.

To build the docker image for OpenCL 1.2 and 2.0 respectively, use

    docker build . -f Dockerfile.intel_ocl1.2_cpu -t chai-intel-1.2
    docker build . -f Dockerfile.intel_ocl2.0_cpu -t chai-intel-2.0

To run a benchmark (for example, BS) from the two images, use

    docker run -it chai-intel-1.2 bash -c "cd chai/OpenCL1.2/BS/ && ./bs"
    docker run -it chai-intel-2.0 bash -c "cd chai/OpenCL2.0/BS/ && ./bs"