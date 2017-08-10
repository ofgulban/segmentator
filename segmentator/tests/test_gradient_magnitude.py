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

# calculate 3D Sobel
gra_mag = compute_gradient_magnitude(ima, method='3D_sobel')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_3D_sobel.nii.gz')

# calculate 3D Prewitt
gra_mag = compute_gradient_magnitude(ima, method='3D_prewitt')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_3D_prewitt.nii.gz')

# calculate numpy gradient magnitude
gra_mag = compute_gradient_magnitude(ima, method='numpy')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_numpy_gradient.nii.gz')

# calculate scipy sobel
gra_mag = compute_gradient_magnitude(ima, method='scipy_sobel')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_scipy_sobel.nii.gz')

# calculate scipy prewitt
gra_mag = compute_gradient_magnitude(ima, method='scipy_prewitt')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_scipy_prewitt.nii.gz')

# calculate 3D Prewitt
gra_mag = compute_gradient_magnitude(ima, method='3D_scharr')
out = Nifti1Image(gra_mag, affine=nii.affine)
save(out, basename + '_3D_scharr.nii.gz')
