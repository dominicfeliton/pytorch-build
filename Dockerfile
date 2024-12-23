# *****************************************************************************
# Dockerfile: Build PyTorch v2.3.1 with CUDA 12.1 + cuDNN 8.9.7.29
#             PLUS TorchVision & TorchAudio wheels
# *****************************************************************************
# Usage:
#   1) docker build -t my-pytorch-builder:latest .
#   2) docker run --rm -it -v $(pwd)/output:/wheelhouse my-pytorch-builder:latest
#      --> This puts the final .whl files in ./output on your host.
# *****************************************************************************

# 1) Start from the official CUDA 12.1-devel image on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2) Set some environment/container variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ===== Please update magma-cuda + the base image if you need to bump CUDA! =====
ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.3.1
ARG TORCH_AUDIO_VERSION=0.18.1
ARG CUDNN_VERSION=8.9.7.29
ARG CUDNN_NAME="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive"
ARG CUDNN_FILE="${CUDNN_NAME}.tar.xz"
ARG CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_FILE}"

# For building PyTorch (with CUDA/CUDNN:)
ENV USE_CUDA=1                 
ENV USE_CUDNN=1                  
ENV USE_MKLDNN=1
ENV USE_MAGMA=ON 
ENV USE_DISTRIBUTED=ON 
ENV USE_MPI=ON 
ENV USE_SYSTEM_NCCL=ON 
ENV _GLIBCXX_USE_CXX11_ABI=0
# Set architecture list (modify as needed)
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
ENV TORCH_CUDA_ARCH_LIST="6.1 7.5 8.0 8.6 9.0"
ENV GPU_TARGET="sm_61 sm_75 sm_80 sm_86 sm_90"

# For building TorchVision with GPU support:
ENV FORCE_CUDA=1

# 3) Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        git \
        wget \
        ca-certificates \
        libssl-dev \
        curl \
        # The following are optional but often needed by TorchVision / TorchAudio
        # especially for reading various image/audio formats:
        libpng-dev \
        libjpeg-dev \
        sox \
        ffmpeg \
        libpthread-stubs0-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# 4) Download and install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# 5) Download cuDNN 8.9.7.29 (CUDA 12) from NVIDIA archive
RUN echo "Downloading cuDNN from ${CUDNN_URL}" && \
    wget -q "${CUDNN_URL}" -O "/tmp/${CUDNN_FILE}" && \
    tar -xf "/tmp/${CUDNN_FILE}" -C /tmp && \
    mkdir -p /usr/local/cuda/include/ && \
    mkdir -p /usr/local/cuda/lib/ && \
    cp -P /tmp/${CUDNN_NAME}/include/* /usr/local/cuda/include/ && \
    cp -P /tmp/${CUDNN_NAME}/lib/* /usr/local/cuda/lib/ && \
    rm -rf /tmp/${CUDNN_NAME} /tmp/${CUDNN_FILE}

# 6) Ensure dynamic linker can find these libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:${LD_LIBRARY_PATH}

# 7) Install conda packages needed for building
RUN conda install -c conda-forge -y \
       python="${PYTHON_VERSION}" \
       numpy \
       ninja \
       pyyaml \
       setuptools \
       cmake \
       cffi \
       typing_extensions \
       future \
       six \
       requests \
       dataclasses \
       libstdcxx-ng \
       gcc=11.4.0 \
       gxx_linux-64 \
       glib \
       pthread-stubs \
       gfortran \
       && \
    conda clean -ya

# Fix build errors
RUN mkdir -p /lib64 && \
    for file in /lib/x86_64-linux-gnu/*; do \
        ln -s "$file" "/lib64/$(basename "$file")"; \
    done
# Create symlinks for all files from /usr/lib/x86_64-linux-gnu to /usr/lib64
RUN mkdir -p /usr/lib64 && \
    for file in /usr/lib/x86_64-linux-gnu/*; do \
        ln -s "$file" "/usr/lib64/$(basename "$file")"; \
    done
RUN ln -s /usr/lib/x86_64-linux-gnu/libpthread.a /usr/lib64/libpthread_nonshared.a

# Build MAGMA from source
RUN git clone --depth 1 https://github.com/icl-utk-edu/magma.git /opt/magma && \
    cd /opt/magma && \
    echo "GPU_TARGET = ${GPU_TARGET}" > make.inc && \
    echo "BACKEND = cuda" >> make.inc && \
    echo "FORT = false" >> make.inc && \
    make generate && \
    cmake -DGPU_TARGET="${GPU_TARGET}" -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" -DCMAKE_INSTALL_PREFIX=build/target . -Bbuild && \
    cmake --build build -j $(nproc) --target install && \
    cp build/target/include/* ${CONDA_PREFIX}/include/ && \
    cp build/target/lib/*.so ${CONDA_PREFIX}/lib/ && \
    cp build/target/lib/pkgconfig/*.pc ${CONDA_PREFIX}/lib/pkgconfig/

# 8) Clone PyTorch and checkout
RUN git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch && \
    cd /opt/pytorch && \
    git checkout v${TORCH_VERSION} && \
    git submodule sync && \
    git submodule update --init --recursive

# Replace the build_bundled.py script with the fixed version
COPY build_bundled_fixed.py /opt/pytorch/third_party/build_bundled.py

# 9) Install additional dependencies (like magma for GPU ops) & PyTorch’s Python deps
RUN pip install -r /opt/pytorch/requirements.txt \
    && pip install mkl-static mkl-include

# Also help PyTorch’s CMake find the Conda environment:
ENV CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'/opt/conda'}:${CMAKE_PREFIX_PATH}"

# 11) Prepare + Build PyTorch from source
ENV MAX_JOBS=10
WORKDIR /opt/pytorch

RUN python setup.py clean
RUN python setup.py bdist_wheel

# 12) Copy the PyTorch wheel into /wheelhouse
RUN mkdir -p /wheelhouse && cp dist/*.whl /wheelhouse

# Install torch
RUN pip install /wheelhouse/*.whl

# 13) Clone TorchVision from GitHub and build from source

RUN git clone https://github.com/pytorch/vision /opt/vision && \
    cd /opt/vision && \
    git checkout v${TORCH_AUDIO_VERSION} && \
    git submodule update --init --recursive
WORKDIR /opt/vision
RUN python setup.py clean
RUN python setup.py bdist_wheel
RUN cp dist/*.whl /wheelhouse

# 14) Clone TorchAudio from GitHub and build from source

RUN git clone https://github.com/pytorch/audio /opt/audio && \
    cd /opt/audio && \
    git checkout v${TORCH_VERSION} && \
    git submodule update --init --recursive
WORKDIR /opt/audio
RUN python setup.py clean
RUN python setup.py bdist_wheel
RUN cp dist/*.whl /wheelhouse

# 15) Final command
CMD echo "Build complete. The wheels are in /wheelhouse." && ls -l /wheelhouse
