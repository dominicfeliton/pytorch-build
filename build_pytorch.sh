#!/usr/bin/env bash
#
# build_pytorch.sh
#
# Example usage:
#   chmod +x build_pytorch.sh
#   ./build_pytorch.sh --cuda-version "12.1" --cudnn-version "8.9.7.29" --pytorch-version "v2.3.1"
#
# This script:
#   1. Removes conflicting NVIDIA packages on Ubuntu 22.04.
#   2. Installs the specified CUDA version (12.1) via .deb local installer.
#   3. Installs cuDNN from a tar.xz archive for CUDA 12 (version 8.9.7.29).
#   4. Clones PyTorch, checks out the specified tag/branch (v2.3.1).
#   5. Builds a PyTorch wheel that DOES NOT statically bundle cuDNN.

set -euxo pipefail

###############################################################################
# 1. Parse Arguments
###############################################################################
CUDA_VERSION="12.1"
CUDNN_VERSION="8.9.7.29"
PYTORCH_VERSION="v2.3.1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-version)
      CUDA_VERSION="$2"
      shift 2
      ;;
    --cudnn-version)
      CUDNN_VERSION="$2"
      shift 2
      ;;
    --pytorch-version)
      PYTORCH_VERSION="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "CUDNN_VERSION: ${CUDNN_VERSION}"
echo "PYTORCH_VERSION: ${PYTORCH_VERSION}"

###############################################################################
# 2. Remove Any Existing NVIDIA or CUDA Packages
###############################################################################
sudo apt-get update
sudo apt-get remove -y --purge "^nvidia-.*" "cuda-*" "libcudnn*"

###############################################################################
# 3. Install CUDA 12.1 on Ubuntu 22.04
#
# Based on:
#   https://developer.nvidia.com/cuda-12-1-0-download-archive
#
# You can find the .deb local installers by selecting:
#   OS: Linux -> x86_64 -> Ubuntu -> 22.04 -> .deb (local)
###############################################################################
sudo apt-get install -y wget gnupg

# 3a. Set up pinning so the CUDA repo has priority
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 3b. Download & install the local CUDA repo .deb for 12.1.0
#    (Adjust the exact filename if needed; at time of writing it is 530.30.02)
CUDA_DEB="cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb"
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/${CUDA_DEB}
sudo dpkg -i ${CUDA_DEB}

# 3c. Import the keyring
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# 3d. Install the CUDA toolkit
sudo apt-get -y install cuda

# At this point, the CUDA toolkit is typically installed at /usr/local/cuda-12.1
# We'll symlink /usr/local/cuda -> /usr/local/cuda-12.1 for convenience (if not done).
if [ ! -d "/usr/local/cuda-12.1" ]; then
  echo "Warning: /usr/local/cuda-12.1 not found. Check your CUDA install path."
else
  sudo ln -s /usr/local/cuda-12.1 /usr/local/cuda || true
fi

###############################################################################
# 4. Download & Install cuDNN 8.9.7.29 (CUDA 12) from Archive
#
# Reference:
#   https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/
# (We use the "cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz" file.)
###############################################################################
CUDNN_FILE="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_FILE}"

wget -q "${CUDNN_URL}" -O "${CUDNN_FILE}"
tar -xf "${CUDNN_FILE}"

# The extracted directory usually contains 'cuda/include' and 'cuda/lib64' subfolders.
# We'll copy them into /usr/local/cuda-12.1/ (or /usr/local/cuda).
sudo cp -P cuda/include/* /usr/local/cuda/include/
sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/

# Optionally, set file permissions:
sudo chmod a+r /usr/local/cuda/include/*
sudo chmod a+r /usr/local/cuda/lib64/*

###############################################################################
# 5. Configure Environment Variables
###############################################################################
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Ensure dynamic linker picks up new libs
echo "/usr/local/cuda/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig

###############################################################################
# 6. Clone PyTorch and Checkout the Specified Version (v2.3.1)
###############################################################################
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout "${PYTORCH_VERSION}"
git submodule sync
git submodule update --init --recursive

###############################################################################
# 7. Install Build Dependencies
#    You can use conda or pip+apt approach. Below uses conda:
###############################################################################
if ! command -v conda &> /dev/null; then
  echo "Conda not found. Install it or adjust to use pip+apt."
  exit 1
fi

conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi \
               typing_extensions future six requests dataclasses

###############################################################################
# 8. Environment Flags to Enable CUDA/cuDNN (Dynamically)
###############################################################################
export USE_CUDA=1
export USE_CUDNN=1
# We do NOT set CAFFE2_STATIC_LINK_CUDA=1, so cuDNN remains dynamically linked.

###############################################################################
# 9. Build PyTorch: produce a .whl
###############################################################################
python setup.py clean
python setup.py bdist_wheel

# The wheel will appear in dist/*.whl
mkdir -p ../dist
cp dist/*.whl ../dist

echo "-----------------------------------------------------------"
echo "PyTorch build complete! Wheel is in dist/ (outside pytorch/)."
echo "CUDA Version: ${CUDA_VERSION}"
echo "cuDNN Version: ${CUDNN_VERSION}"
echo "PyTorch Version: ${PYTORCH_VERSION}"
echo "-----------------------------------------------------------"
