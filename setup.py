"""Cython setup.

Only use this if you want Deriche filter gradient magnitude.

Make sure that you have cython. In the terminal, cd to this file's folder then
run:

    python setup.py build_ext --inplace
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("deriche_3D.pyx")
)
