#!/usr/bin/env bash
set -euo pipefail

### Swap – keep, but reset each run
sudo swapoff -a && sudo swapon -a

# ### Clean
# rm -rf build/ dist/ *.egg-info flash_attn_2_cuda*/*.so flash_attn*/_C*.so \
#        .setuptools-cmake-build/ _skbuild/

### CUDA / arch flags
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST="12.0"
export CMAKE_CUDA_ARCHITECTURES="120"

### Conservative parallelism
export MAX_JOBS=8           # start safe; bump to 2 if RAM stays <40 GiB
export NVCC_THREADS=2        # our own flag, see below`

### Tiny patch: inject --threads ${NVCC_THREADS} into every nvcc call
export CFLAGS_NVCC="--threads ${NVCC_THREADS}"

echo "→ Building with ninja -j${MAX_JOBS} (nvcc threads ${NVCC_THREADS})"
python setup.py build_ext --inplace -j${MAX_JOBS}

echo "→ Installing editable…"
python setup.py develop -q

echo "✅ Done!  Keep an eye on RAM with:  watch -n1 free -h"
