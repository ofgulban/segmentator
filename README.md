[![DOI](https://zenodo.org/badge/59303623.svg)](https://zenodo.org/badge/latestdoi/59303623) [![Build Status](https://travis-ci.org/ofgulban/segmentator.svg?branch=master)](https://travis-ci.org/ofgulban/segmentator) [![Build status](https://ci.appveyor.com/api/projects/status/lkxp4y5ahssqv6ng?svg=true)](https://ci.appveyor.com/project/ofgulban/segmentator) [![codecov](https://codecov.io/gh/ofgulban/segmentator/branch/master/graph/badge.svg)](https://codecov.io/gh/ofgulban/segmentator) [![Code Health](https://landscape.io/github/ofgulban/segmentator/master/landscape.svg?style=flat)](https://landscape.io/github/ofgulban/segmentator/master) [![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/segmentator/Lobby)

# Segmentator

<img src="logo/logo.png" width=420 align="right" />

Segmentator is a free and open-source package for multi-dimensional data exploration and segmentation for 3D images. This application is mainly developed and tested using ultra-high field magnetic resonance imaging (MRI) brain data.


The goal is to provide a complementary tool to the already available brain tissue segmentation methods (to the best of our knowledge) in other software packages (FSL, Freesurfer, SPM, Brainvoyager, itk-SNAP, etc.).

## Core dependencies
[**Python 2.7**](https://www.python.org/download/releases/2.7/)

| Package                              | Tested version |
|--------------------------------------|----------------|
| [matplotlib](http://matplotlib.org/) | **1.5.3**      |
| [NumPy](http://www.numpy.org/)       | 1.13.1         |
| [NiBabel](http://nipy.org/nibabel/)  | 2.1.0          |
| [SciPy](http://scipy.org/)           | 0.19.1         |

## Installation & Quick Start
Please visit [our wiki](https://github.com/ofgulban/segmentator/wiki/Installation) to see how to install and use Segmentator.

## Support
Please use [GitHub issues](https://github.com/ofgulban/segmentator/issues) for questions, bug reports or feature requests.

## License
Copyright © 2016, [Omer Faruk Gulban](https://github.com/ofgulban) and [Marian Schneider](https://github.com/MSchnei).
Released under [GNU General Public License Version 3](http://www.gnu.org/licenses/gpl.html).

## References
This application is based on the following work:

* Kindlmann, G., & Durkin, J. W. (1998). Semi-automatic generation of transfer functions for direct volume rendering. In Proceedings of the 1998 IEEE symposium on Volume visualization - VVS ’98 (pp. 79–86). New York, New York, USA: ACM Press. http://doi.org/10.1145/288126.288167
* Kniss, J., Kindlmann, G., & Hansen, C. (2001). Interactive volume rendering using multi-dimensional transfer functions and direct manipulation widgets. In Proceedings Visualization, 2001. VIS ’01. (pp. 255–562). IEEE. http://doi.org/10.1109/VISUAL.2001.964519
* Kniss, J., Kindlmann, G., & Hansen, C. (2002). Multidimensional transfer functions for interactive volume rendering. IEEE Transactions on Visualization and Computer Graphics, 8(3), 270–285. http://doi.org/10.1109/TVCG.2002.1021579
* Kniss, J., Kindlmann, G., & Hansen, C. D. (2005). Multidimensional transfer functions for volume rendering. Visualization Handbook, 189–209. http://doi.org/10.1016/B978-012387582-2/50011-3
* Jianbo Shi, & Malik, J. (2000). Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888–905. http://doi.org/10.1109/34.868688
* Ip, C. Y., Varshney, A., & Jaja, J. (2012). Hierarchical exploration of volumes using multilevel segmentation of the intensity-gradient histograms. IEEE Transactions on Visualization and Computer Graphics, 18(12), 2355–2363. http://doi.org/10.1109/TVCG.2012.231
