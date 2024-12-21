# *****************************************************************************
# Dockerfile: Build PyTorch v2.3.1 with CUDA 12.1 + cuDNN 8.9.7.29 (Archive)
# *****************************************************************************
# Usage:
#   1) docker build -t my-pytorch-builder:latest .
#   2) docker run --rm -it -v $(pwd)/output:/wheelhouse my-pytorch-builder:latest
#      --> This puts the final .whl in ./output on your host.
#
# NOTE: You must have the NVIDIA driver installed on the host for GPU access,
# but for building the wheel, you just need the user-level CUDA dev libs.
# *****************************************************************************

# 1) Start from the official CUDA 12.1-devel image on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2) Set some environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# 3) Install system packages
#    - build-essential, ninja-build, git, wget, etc.
#    - libssl-dev (for conda install) is often helpful as well
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        git \
        wget \
        ca-certificates \
        libssl-dev \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# 4) Download and install Miniconda (if you prefer system Python, skip this)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# 5) Download cuDNN 8.9.7.29 (for CUDA 12) from the NVIDIA archive
#    Link reference:
#    https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/
#    cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
#
#    If the URL or filename changes, update it below.
ARG CUDNN_VERSION=8.9.7.29
ARG CUDNN_NAME="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive"
ARG CUDNN_FILE="${CUDNN_NAME}.tar.xz"
ARG CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_FILE}"

RUN echo "Downloading cuDNN from ${CUDNN_URL}" && \
    wget -q "${CUDNN_URL}" -O "/tmp/${CUDNN_FILE}" && \
    tar -xf "/tmp/${CUDNN_FILE}" -C /tmp && \
    mkdir -p /usr/local/cuda/include/ && \
    mkdir -p /usr/local/cuda/lib/ && \
    cp -P /tmp/${CUDNN_NAME}/include/* /usr/local/cuda/include/ && \
    cp -P /tmp/${CUDNN_NAME}/lib/* /usr/local/cuda/lib/ && \
    rm -rf /tmp/${CUDNN_NAME} /tmp/${CUDNN_FILE}

# 6) Ensure dynamic linker can find libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:${LD_LIBRARY_PATH}

# 7) Install PyTorch build dependencies with conda
#    (You can also do 'pip install' if you prefer.)
RUN conda install -y \
       python=3.11 \
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
       && \
    conda clean -ya

# 8) Clone PyTorch and checkout v2.3.1
#    (If you want a different version, set a build arg or manually change below.)
RUN git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch && \
    cd /opt/pytorch && \
    git checkout v2.3.1 && \
    git submodule sync && \
    git submodule update --init --recursive

# 9) Build PyTorch from source inside the container
#    We set environment variables to ensure it detects and uses CUDA + cuDNN.
WORKDIR /opt/pytorch
ENV USE_CUDA=1
ENV USE_CUDNN=1
# Not setting CAFFE2_STATIC_LINK_CUDA=1, so cuDNN is dynamically linked.

# Clean any prior build artifacts, then build the wheel
RUN python setup.py clean && \
    python setup.py bdist_wheel

# 10) Our final .whl is in dist/*.whl. We'll copy it somewhere for convenience.
#     We'll create /wheelhouse and copy the wheel there. Then, when you run
#     "docker run -v $(pwd)/output:/wheelhouse ..." you can retrieve it.
RUN mkdir -p /wheelhouse && cp dist/*.whl /wheelhouse

# 11) Provide a default command or simply exit
CMD echo "Build complete. The wheel is in /wheelhouse." && ls -l /wheelhouse
