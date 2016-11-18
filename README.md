# Segmentator (work in progress)
[![DOI](https://zenodo.org/badge/59303623.svg)](https://zenodo.org/badge/latestdoi/59303623)

Segmentator is a free and open-source package for multi-dimensional data exploration and segmentation. This application is mainly developed for ultra-high field magnetic resonance imaging (MRI) brain data.

The goal is to provide a complementary tool to the already available brain tissue segmentation methods (to the best of our knowledge) in other software packages (FSL, Freesurfer, SPM, Brainvoyager, ITK-Snap, MIPAV).

## Dependencies

| [Python 2.7](https://www.python.org/download/releases/2.7/)| |
|---|--- |
| [NumPy](http://www.numpy.org/) | tested on  *1.11.1* |
| [matplotlib](http://matplotlib.org/) | tested on *1.5.3* |
| [NiBabel](http://nipy.org/nibabel/) | tested on *2.1.0* |

##### Optional
- [scikit-image](http://scikit-image.org/), for normalized graph cuts. However you should install it from [this fork](https://github.com/ofgulban/scikit-image) for the required functionality. If you are scared of messing up your current python modules consider using [miniconda](http://conda.pydata.org/miniconda.html) with [virtual python environments](http://conda.pydata.org/docs/using/envs.html).

- [cython](http://cython.org/) (tested on 0.24.1), for Deriche filter gradient magnitude calculation.

## How to start

Open a terminal, navigate to Segmentator's folder (for instance: `cd /home/john/segmentator/` ) and type ```python segmentator.py --help``` to see the usage and all the available options.

To load data simply run:
```bash
python segmentator.py /path/to/your/file.nii.gz
```

You should see a window appearing soon after. Try dragging the red circle around. You can even draw directly on the histogram after turning the lasso tool on:

![demo](images/animated.gif)

To use advanced features (normalized graph cuts, deriche filter) please visit our [github wiki](https://github.com/ofgulban/segmentator/wiki).

## Support

Please use [GitHub issues](https://github.com/ofgulban/segmentator/issues) for questions, bug reports or feature requests.


## License

The project is licensed under [GNU Geneal Public License Version 3](http://www.gnu.org/licenses/gpl.html).

## References

This application is based on the following work:

* Kindlmann, G., & Durkin, J. W. (1998). Semi-automatic generation of transfer functions for direct volume rendering. In Proceedings of the 1998 IEEE symposium on Volume visualization - VVS ’98 (pp. 79–86). New York, New York, USA: ACM Press. http://doi.org/10.1145/288126.288167

* Kniss, J., Kindlmann, G., & Hansen, C. (2001). Interactive volume rendering using multi-dimensional transfer functions and direct manipulation widgets. In Proceedings Visualization, 2001. VIS ’01. (pp. 255–562). IEEE. http://doi.org/10.1109/VISUAL.2001.964519

* Kniss, J., Kindlmann, G., & Hansen, C. (2002). Multidimensional transfer functions for interactive volume rendering. IEEE Transactions on Visualization and Computer Graphics, 8(3), 270–285. http://doi.org/10.1109/TVCG.2002.1021579

* Kniss, J., Kindlmann, G., & Hansen, C. D. (2005). Multidimensional transfer functions for volume rendering. Visualization Handbook, 189–209. http://doi.org/10.1016/B978-012387582-2/50011-3

* Jianbo Shi, & Malik, J. (2000). Normalized cuts and image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(8), 888–905. http://doi.org/10.1109/34.868688

* Ip, C. Y., Varshney, A., & Jaja, J. (2012). Hierarchical exploration of volumes using multilevel segmentation of the intensity-gradient histograms. IEEE Transactions on Visualization and Computer Graphics, 18(12), 2355–2363. http://doi.org/10.1109/TVCG.2012.231

* Monga, O., Deriche, R., & Rocchisani, J.-M. (1991). 3D edge detection using recursive filtering: Application to scanner images. CVGIP: Image Understanding, 53(1), 76–87. http://doi.org/10.1016/1049-9660(91)90006-B
