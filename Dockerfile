#FROM nvidia/cuda:8.0-devel-ubuntu16.04
FROM nvidia/cuda:8.0-devel

#RUN nvcc --version
#RUN ls /usr/local/cuda/include
#RUN ls /usr/local/cuda/lib64

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y libopencv-dev

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
#    nvidia-opencl-icd-367 \
    nvidia-modprobe \
    ocl-icd-libopencl1 \
    ocl-icd-opencl-dev \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
#RUN ls /etc/OpenCL/vendors
#RUN cat /etc/OpenCL/vendors/nvidia.icd

RUN cat /etc/ld.so.conf.d/nvidia.conf

ADD ./OpenCL1.2 chai/OpenCL1.2

# Build  OpenCL 1.2
RUN export CHAI_OCL_LIB=/usr/local/cuda/lib64 export CHAI_OCL_INC=/usr/local/cuda/include && \
    cd chai/OpenCL1.2 && \
    for bench in *; do \
      echo $bench; \
      cd $bench; \
      make -j; \
      cd ..; \
    done
