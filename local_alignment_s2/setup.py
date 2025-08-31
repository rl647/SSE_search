from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

# Get Pybind11 include path
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("Please install pybind11 first: pip install pybind11")
    sys.exit(-1)
extra_compile_args = ['-fopenmp', '-std=c++17']  # Added OpenMP flag here
extra_link_args = ['-fopenmp']
ext_modules = [
    Extension(
        "local_alignment_v2",
        ["local_alignment.cpp"],
        include_dirs=[pybind11_include],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(
    name="local_alignment_v2",
    version="0.1",
    author="Runfeng",
    description="Local alignment callable for SSE",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)


#python setup.py build_ext --inplace
