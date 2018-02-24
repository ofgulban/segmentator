"""Segmentator setup.

To install for development, using the commandline do:
    pip install -e /path/to/segmentator

"""

from setuptools import setup
from setuptools.extension import Extension
import numpy

ext_modules = [Extension(
    "segmentator.deriche_3D", ['segmentator/deriche/deriche_3D.c'],
    include_dirs=[numpy.get_include()])
    ]

setup(name='segmentator',
      version='1.5.0',
      description=('Multi-dimensional data exploration and segmentation for 3D \
                   images.'),
      url='https://github.com/ofgulban/segmentator',
      author='Omer Faruk Gulban',
      author_email='faruk.gulban@maastrichtuniversity.nl',
      license='GNU General Public License Version 3',
      packages=['segmentator'],
      install_requires=['numpy', 'matplotlib', 'scipy'],
      keywords=['mri', 'segmentation', 'image', 'voxel'],
      zip_safe=True,
      entry_points={
          'console_scripts': [
              'segmentator = segmentator.__main__:main',
              ]},
      ext_modules=ext_modules,
      )
