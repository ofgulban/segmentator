"""Full automation experiments for T1w-like data (MPRAGE & MP2RAGE).

TODO:
    - Put anisotropic diffusion based smoothing to segmentator utilities.
    - MP2RAGE, 2 type of gray matter giving issues, barycentric weights might
    be useful to deal with this issue.

"""

import os
import peakutils
import numpy as np
import matplotlib.pyplot as plt
from nibabel import load, Nifti1Image, save
from scipy.ndimage.filters import gaussian_filter1d
from segmentator.utils import compute_gradient_magnitude, aniso_diff_3D

# load
nii = load('/home/faruk/gdrive/Segmentator/data/faruk/arcweld/mp2rage_S001_restore.nii.gz')
ima = nii.get_data()
basename = nii.get_filename().split(os.extsep, 1)[0]

# non-zero mask
msk = (ima != 0)  # TODO: Parametrize
#
ima[msk] = ima[msk] + np.min(ima)

# aniso. diff. filter
ima = aniso_diff_3D(ima, niter=2, kappa=500, gamma=0.1, option=1)

# calculate gradient magnitude
gra = compute_gradient_magnitude(ima, method='3D_sobel')

# save for debugging
# out = Nifti1Image(gra.reshape(nii.shape), affine=nii.affine)
# save(out, basename + '_gra' + '.nii.gz')

ima_max = np.percentile(ima, 100)

# reshape for histograms
ima, gra, msk = ima.flatten(), gra.flatten(), msk.flatten()
gra_thr = np.percentile(gra[msk], 20)

# select low gradient magnitude regime (lr)
msk_lr = (gra < gra_thr) & (msk)
n, bins, _ = plt.hist(ima[msk_lr], 200, range=(0, ima_max))

# smooth histogram (good for peak detection)
n = gaussian_filter1d(n, 1)

# detect 'pure tissue' peaks (TODO: Checks for always finding 3 peaks)
peaks = peakutils.indexes(n, thres=0.01/max(n), min_dist=40)
# peaks = peaks[0:-1]
tissues = []
for p in peaks:
    tissues.append(bins[p])
tissues = np.array(tissues)
print peaks
print tissues

# insert zero-max arc
zmax_max = ima_max
zmax_center = (0 + zmax_max) / 2.
zmax_radius = zmax_max - zmax_center
tissues = np.append(tissues, zmax_center)

# create smooth maps (distance to pure tissue)
voxels = np.vstack([ima, gra])
soft = []  # to hold soft tissue membership maps
for i, t in enumerate(tissues):
    tissue = np.array([t, 0])
    # euclidean distance
    edist = np.sqrt(np.sum((voxels - tissue[:, None])**2., axis=0))
    soft.append(edist)
    # save intermediate maps
    # out = Nifti1Image(edist.reshape(nii.shape), affine=nii.affine)
    # save(out, basename + '_t' + str(i) + '.nii.gz')
soft = np.array(soft)

# interface translation (shift zero circle to another radius)
soft[-1, :] = soft[-1, :] - zmax_radius
# zmax_neg = soft[-1, :] < 0  # voxels fall inside zero-max arc

# weight zero-max arc to not coincide with pure class regions
# zmax_weight = (gra[zmax_neg] / zmax_radius)**-1
zmax_weight = (gra / zmax_radius)**-1
# soft[-1, :][zmax_neg] = zmax_weight*np.abs(soft[-1, :][zmax_neg])
soft[-1, :] = zmax_weight*np.abs(soft[-1, :])
# save for debugging
# out = Nifti1Image(soft[-1, :].reshape(nii.shape), affine=nii.affine)
# save(out, basename + '_zmaxarc' + '.nii.gz')

# arbitrary weighting (TODO: Can be turned into config file of some sort)
# save these values for MP2RAGE UNI
# soft[0, :] = soft[0, :] * 1  # csf
# soft[-1, :] = soft[-1, :] * 0.5  # zero-max arc

# hard tissue membership maps
hard = np.argmin(soft, axis=0)

# append masked out areas
hard = hard + 1
hard[~msk] = 0

# save hard classification
out = Nifti1Image(hard.reshape(nii.shape), affine=nii.affine)
save(out, basename + '_arcweld' + '.nii.gz')

# save segmentator polish mask (not GM and WM)
labels = np.unique(hard)[[0, 1, -1]]
polish = np.ones(hard.shape)
polish[hard == labels[0]] = 0
polish[hard == labels[1]] = 0
polish[hard == labels[2]] = 0
out = Nifti1Image(polish.reshape(nii.shape), affine=nii.affine)
save(out, basename + '_arcweld_mask' + '.nii.gz')
print 'Finished.'

# plt.show()
