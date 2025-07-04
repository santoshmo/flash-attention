# examples/multi_gpu_flash/Makefile

PYTHON       := python3
CXX          := g++
NVCC         := nvcc
PROJECT_ROOT := $(abspath ../..)
TORCH_PKG    := $(shell env PYTHONPATH= $(PYTHON) -c "import os, torch; print(os.path.dirname(torch.__file__))")
FLASH_ROOT   := $(PROJECT_ROOT)/csrc/flash_attn

# cutlass lives next to flash_attn
CUTLASS_INC  := $(PROJECT_ROOT)/csrc/cutlass/include

# 1) Python headers
PY_INC       := -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")

# 2) PyTorch headers
TORCH_INC    := -I$(TORCH_PKG)/include \
                -I$(TORCH_PKG)/include/torch/csrc/api/include

# 3) ATen CUDA-detail (philox_unpack.cuh)
ATEN_DETAIL  := -I$(TORCH_PKG)/include/ATen/cuda/detail

# 4) CUDA / NCCL
CUDA_INC     := -I/usr/local/cuda/include
NCCL_INC     := -I/usr/local/nccl/include

# 5) Flash-Attention itself
FLASH_INC    := -I$(FLASH_ROOT) \
                -I$(FLASH_ROOT)/src \
                -I$(CUTLASS_INC)

# Define the module name expected by Python
EXT_NAME      := pipelined_fa

# compile flags
CXXFLAGS     := -O2 -std=c++17 -fPIC \
                $(PY_INC) $(TORCH_INC) $(ATEN_DETAIL) \
                $(CUDA_INC) $(NCCL_INC) $(FLASH_INC)

# CUDA compile flags (separate from host CXXFLAGS)
NVCCFLAGS    := -O2 -std=c++17 -Xcompiler -fPIC -rdc=true \
                $(PY_INC) $(TORCH_INC) $(ATEN_DETAIL) \
                $(CUDA_INC) $(NCCL_INC) $(FLASH_INC)

FLASH_LIB := $(PROJECT_ROOT)/build/lib.linux-x86_64-cpython-312/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so

# link libraries (no nvcc-unsupported flags here)
LDLIBS       := \
                -L$(TORCH_PKG)/lib -ltorch -lc10 \
                -L/usr/local/cuda/lib64 -lcudart \
                -L/usr/local/nccl/lib -lnccl \
                $(FLASH_LIB)

# your two .cpp files
SRCS         := pipelined_fa.cpp renorm_kernel.cu

# Object files (handle both .cu and .cpp)
OBJS         := $(SRCS:.cu=.o)
OBJS         := $(OBJS:.cpp=.o)

TARGET       := pipelined_fa.so

all: $(TARGET)

unit_renorm.so: unit_test_renorm.cu renorm_kernel.cu
	$(NVCC) -shared -x cu -std=c++17 -Xcompiler -fPIC \
	        $(PY_INC) $(TORCH_INC) $(ATEN_DETAIL) $(CUDA_INC) \
	        unit_test_renorm.cu renorm_kernel.cu -o $@ \
	        -L$(TORCH_PKG)/lib -ltorch_python -ltorch -lc10 -lcudart \
	        -Xlinker -rpath -Xlinker $(TORCH_PKG)/lib

# Pattern rules to build object files with nvcc (treat .cpp as CUDA for kernel launches)
%.o: %.cpp
	$(NVCC) -x cu $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link with nvcc to ensure proper device linking
$(TARGET): $(OBJS) $(FLASH_ROOT)/flash_api.h
	$(NVCC) -shared -Xlinker --no-as-needed -o $@ $(OBJS) $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJS)