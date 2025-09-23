from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_ode_step',
    ext_modules=[
        CUDAExtension('cuda_ode_step', [
            'ode_step.cpp',
            'ode_step_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

