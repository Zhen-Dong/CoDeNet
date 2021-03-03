from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from subprocess import call

setup(
    name='dcn_deform_conv',
    ext_modules=[
        CUDAExtension('dcn_deform_conv_cuda', [
            'src/dcn_deform_conv_cuda.cpp',
            'src/dcn_deform_conv_cuda_kernel.cu',
        ]),
        CUDAExtension(
            'dcn_deform_pool_cuda',
            ['src/dcn_deform_pool_cuda.cpp', 'src/dcn_deform_pool_cuda_kernel.cu']),
    ],
    cmdclass={'build_ext': BuildExtension})

call('mv dcn_deform_*.so _ext/dcn/', shell=True)