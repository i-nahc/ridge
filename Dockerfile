# Ridge dev image.
#
# CUDA "devel" base gives nvcc and cuBLAS for Phases 2 to 5. It also builds the
# pure-C++ Phase 1 with the same gcc toolchain, so one image covers everything.
# A GPU is only needed at RUNTIME (docker run --gpus all), not to build the image
# or to build and test Phase 1. So you can build this image and run all of Phase 1
# on a machine with no GPU.
#
# A100 is compute capability sm_80, supported by CUDA 12.x. Phase 2 kernels compile
# with nvcc -arch=sm_80.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      ninja-build \
      git \
      python3 \
      python3-pip \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python tooling for the calibration and validation scripts (Phases 3 to 5).
RUN pip3 install --no-cache-dir numpy pandas matplotlib

# Nsight Compute (ncu) for Phase 4 bottleneck cross-checks is NOT installed here,
# because the right package depends on the host driver and CUDA image. Install it
# on demand inside the gpu container, see docs/DOCKER.md, or profile from the host.

WORKDIR /workspace

CMD ["/bin/bash"]
