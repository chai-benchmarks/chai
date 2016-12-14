FROM nvidia/cuda:8.0-devel

ENV DEBIAN_FRONTEND noninteractive

# Install OpenCV
RUN apt-get update && apt-get install --no-install-recommends -y libopencv-dev

# Set up NVIDIA OpenCL environment
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    #nvidia-modprobe \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    #clinfo \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Make the OpenCL1.2 files visible to the container
ADD ./OpenCL1.2 chai/OpenCL1.2

# Build  OpenCL 1.2 version of all benchmarks
RUN export CHAI_OCL_LIB=/usr/local/cuda/lib64 export CHAI_OCL_INC=/usr/local/cuda/include && \
    cd chai/OpenCL1.2 && \
    for bench in *; do \
      echo $bench; \
      cd $bench; \
      make -j; \
      cd ..; \
    done
