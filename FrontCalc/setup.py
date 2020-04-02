from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension("_csupport", ["_csupport.c", "csupport.c"])],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)

