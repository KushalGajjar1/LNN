from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='liquid_cuda',
    ext_modules=[
        CUDAExtension('liquid_cuda', [
            'liquid_pybind.cpp', # <-- RENAMED
            'liquid_cuda.cu',
        ],
        libraries=['cublas']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

