[![DOI](https://zenodo.org/badge/59303623.svg)](https://zenodo.org/badge/latestdoi/59303623)

# Segmentator

<img src="visuals/logo.png" width=420 align="right" />

Segmentator is a free and open-source package for multi-dimensional data exploration and segmentation for 3D images. This application is mainly developed and tested using ultra-high field magnetic resonance imaging (MRI) brain data.


The goal is to provide a complementary tool to the already available brain tissue segmentation methods (to the best of our knowledge) in other software packages ([FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/), [CBS-Tools](https://www.cbs.mpg.de/institute/software/cbs-tools), [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php), [Freesurfer](https://surfer.nmr.mgh.harvard.edu/), [SPM](http://www.fil.ion.ucl.ac.uk/spm/software/spm12/), [Brainvoyager](http://www.brainvoyager.com/), etc.).

### Citation:
- Our paper can be accessed from __[this link.](https://doi.org/10.1371/journal.pone.0198335)__
- Released versions of this package can be cited by using our __[Zenodo DOI](https://zenodo.org/badge/latestdoi/59303623).__

<img src="visuals/animation_01.gif" width=840 align="center" />

## Core dependencies
**[Python 3.6](https://www.python.org/downloads/release/python-363/)** or **[Python 2.7](https://www.python.org/download/releases/2.7/)** (compatible with both).

| Package                                        | Tested version |
|------------------------------------------------|----------------|
| [matplotlib](http://matplotlib.org/)           | 3.1.1          |
| [NumPy](http://www.numpy.org/)                 | 1.22.0         |
| [NiBabel](http://nipy.org/nibabel/)            | 2.5.1          |
| [SciPy](http://scipy.org/)                     | 1.3.1          |
| [Compoda](https://github.com/ofgulban/compoda) | 0.3.5          |

## Installation & Quick Start
- Download [the latest release](https://github.com/ofgulban/segmentator/releases) and unzip it.
- Change directory in your command line:
```
cd /path/to/segmentator
```
- Install the requirements by running the following command:
```
pip install -r requirements.txt
```
- Install Segmentator:
```
python setup.py install
```
- Simply call segmentator with a nifti file:
```
segmentator /path/to/file.nii.gz
```
- Or see the help for available options:
```
segmentator --help
```

Check out __[our wiki](https://github.com/ofgulban/segmentator/wiki)__ for further details such as [GUI controls](https://github.com/ofgulban/segmentator/wiki/Controls), [alternative installation methods](https://github.com/ofgulban/segmentator/wiki/Installation) and more...

## Support
Please use [GitHub issues](https://github.com/ofgulban/segmentator/issues) for questions, bug reports or feature requests.

## License
Copyright © 2019, [Omer Faruk Gulban](https://github.com/ofgulban) and [Marian Schneider](https://github.com/MSchnei).
This project is licensed under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause).

## References
This application is mainly based on the following work:

* Kniss, J., Kindlmann, G., & Hansen, C. D. (2005). Multidimensional transfer functions for volume rendering. Visualization Handbook, 189–209. <http://doi.org/10.1016/B978-012387582-2/50011-3>

## Acknowledgements
Since early 2020, development and maintenance of this project is being actively supported by [Brain Innovation](https://www.brainvoyager.com/) as the main developer ([Omer Faruk Gulban](https://github.com/ofgulban)) works there.
