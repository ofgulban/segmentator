"""Segmentator setup."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from Cython.Build import cythonize  # required for deriche filter

setup(name='segmentator',
      version='1.2.0',
      description=('Multi-dimensional data exploration and segmentation for 3D \
                   images.'),
      url='https://github.com/ofgulban/segmentator',
      download_url='',
      author='Omer Faruk Gulban',
      author_email='faruk.gulban@maastrichtuniversity.nl',
      license='GNU General Public License Version 3',
      packages=['segmentator'],
      install_requires=['numpy', 'matplotlib', 'cython'],
      keywords=['mri', 'segmentation'],
      zip_safe=True,
      entry_points={
          'console_scripts': [
              'segmentator = segmentator.__main__:main',
              ]},
      ext_modules=cythonize("segmentator/deriche_3D.pyx")
      )
