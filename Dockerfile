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
RUN conda install -y \
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
       && \
    conda clean -ya

# 8) Clone PyTorch and checkout
RUN git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch && \
    cd /opt/pytorch && \
    git checkout v"${TORCH_VERSION}" && \
    git submodule sync && \
    git submodule update --init --recursive

# 9) Install additional dependencies (like magma for GPU ops) & PyTorch’s Python deps
RUN conda install -y -c pytorch magma-cuda121  \
    && pip install -r /opt/pytorch/requirements.txt \
    && pip install mkl-static mkl-include

# 10) Optional: If you want to use new C++ ABI
ENV _GLIBCXX_USE_CXX11_ABI=0

# Also help PyTorch’s CMake find the Conda environment:
ENV CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'/opt/conda'}:${CMAKE_PREFIX_PATH}"

# Set architecture list (modify as needed)
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

# 11) Build PyTorch from source
ENV MAX_JOBS=10
WORKDIR /opt/pytorch
RUN mkdir -p third_party/opentelemetry-cpp/tools/vcpkg/ports/gettimeofday && \
    echo "Provided under BSD-3-Clause (placeholder)" > third_party/opentelemetry-cpp/tools/vcpkg/ports/gettimeofday/LICENSE
RUN python setup.py clean
RUN python setup.py bdist_wheel

# 12) Copy the PyTorch wheel into /wheelhouse
RUN mkdir -p /wheelhouse && cp dist/*.whl /wheelhouse

# 13) Clone TorchVision from GitHub and build from source
#     Adjust the version tag/branch as desired
RUN git clone https://github.com/pytorch/vision /opt/vision && \
    cd /opt/vision && \
    git checkout v"${TORCH_AUDIO_VERSION}" && \
    git submodule update --init --recursive
WORKDIR /opt/vision
RUN python setup.py clean
RUN python setup.py bdist_wheel
RUN cp dist/*.whl /wheelhouse

# 14) Clone TorchAudio from GitHub and build from source
#     Adjust the version tag/branch as desired
RUN git clone https://github.com/pytorch/audio /opt/audio && \
    cd /opt/audio && \
    git checkout v"${TORCH_VERSION}" && \
    git submodule update --init --recursive
WORKDIR /opt/audio
RUN python setup.py clean
RUN python setup.py bdist_wheel
RUN cp dist/*.whl /wheelhouse

# 15) Final command
CMD echo "Build complete. The wheels are in /wheelhouse." && ls -l /wheelhouse
