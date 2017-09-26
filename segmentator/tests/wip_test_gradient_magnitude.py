"""Experiment with different gradient magnitude calculations.

TODO: turn this into unit tests.

"""

import os
from nibabel import load, Nifti1Image, save
from segmentator.utils import compute_gradient_magnitude

# load
nii = load('/home/faruk/gdrive/Segmentator/data/faruk/gramag_test/mprage_S02_restore.nii.gz')
ima = nii.get_data()
basename = nii.get_filename().split(os.extsep, 1)[0]

# 3D Scharr gradient magnitude
gra_mag = compute_gradient_magnitude(ima, method='scharr')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_scharr.nii.gz')

# 3D Sobel gradient magnitude
gra_mag = compute_gradient_magnitude(ima, method='sobel')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_sobel.nii.gz')

# 3D Prewitt gradient magnitude
gra_mag = compute_gradient_magnitude(ima, method='prewitt')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_prewitt.nii.gz')

# numpy gradient magnitude
gra_mag = compute_gradient_magnitude(ima, method='numpy')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_numpy_gradient.nii.gz')
