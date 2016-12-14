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
