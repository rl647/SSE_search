from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# Get Pybind11 include path
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("Please install pybind11 first: pip install pybind11")
    sys.exit(-1)

ext_modules = [
    Extension(
        "local_alignment",  # The module name that will be created
        ["local_alignment.cpp"],  # The C++ source file
        include_dirs=[pybind11_include],  # Include directory for Pybind11
        language="c++"
    ),
]

setup(
    name="local_alignment",
    version="0.1",
    author="Runfeng",
    description="Local alignment callable for SSE",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

#python setup.py build_ext --inplace
