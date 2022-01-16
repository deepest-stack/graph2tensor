# coding=utf-8
import numpy as np

from setuptools import find_packages, setup, Extension


sampler = Extension('graph2tensor.client.sampler',
                    include_dirs=[np.get_include(), ],
                    extra_compile_args=['-std=c++11', '-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    sources=['graph2tensor/client/sampler.cpp'])

setup(
    name='graph2tensor',
    version='0.2.0',
    description='Graph learning framework based on neighbour sampling',
    packages=find_packages(exclude=("test",)),
    ext_modules=[sampler, ]
)

