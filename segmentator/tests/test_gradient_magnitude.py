"""Experiment with difference grdient magnitude calculations."""

import os
import numpy as np
from nibabel import load, Nifti1Image, save
from scipy.ndimage import generic_gradient_magnitude, sobel, prewitt, laplace

# load
nii = load('/home/faruk/gdrive/Segmentator/data/faruk/arcweld/mprage_t1w_restore.nii.gz')
ima = nii.get_data()
basename = nii.get_filename().split(os.extsep, 1)[0]

# calculate numpy gradient magnitude
gra = np.gradient(ima)
gra = np.sqrt(np.power(gra[0], 2) + np.power(gra[1], 2) + np.power(gra[2], 2))
out = Nifti1Image(gra, affine=nii.affine)
save(out, basename + '_numpy_gradient.nii.gz')

# calculate sobel
gra = generic_gradient_magnitude(ima, sobel)/32.
out = Nifti1Image(gra, affine=nii.affine)
save(out, basename + '_scipy_sobel.nii.gz')

# calculate prewitt
gra = generic_gradient_magnitude(ima, prewitt)/32.
out = Nifti1Image(gra, affine=nii.affine)
save(out, basename + '_scipy_prewitt.nii.gz')
