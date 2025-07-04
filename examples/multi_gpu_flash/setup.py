# examples/multi_gpu_flash/setup.py
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

THIS_DIR = os.path.dirname(__file__)
FLASH_API = os.path.abspath(os.path.join(THIS_DIR, "../../csrc/flash_attn/flash_api.cpp"))

setup(
    name="run_flash_chain",
    ext_modules=[
        CUDAExtension(
            name="run_flash_chain_cuda",
            sources=[
                "run_flash_attn_chain.cu",
                FLASH_API,              # absolute path here
            ],
            extra_compile_args={"cxx": ["-std=c++17"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)