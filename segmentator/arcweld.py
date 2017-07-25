"""Full automation experiments for T1w-like data (MPRAGE & MP2RAGE)."""

import os
import peakutils
import numpy as np
import matplotlib.pyplot as plt
from nibabel import load, Nifti1Image, save

# load
nii = load('/home/faruk/gdrive/Segmentator/data/faruk/mp2rage_roy/S001_UNI_restore_aniso.nii.gz')
ima = nii.get_data()
basename = nii.get_filename().split(os.extsep, 1)[0]

# calculate gradient magnitude
gra = np.gradient(ima)
gra = np.sqrt(np.power(gra[0], 2) + np.power(gra[1], 2) + np.power(gra[2], 2))

ima_max = np.percentile(ima, 99)
gra_max = np.percentile(gra, 99)

# reshape for histograms
ima, gra = ima.flatten(), gra.flatten()
msk = ima > 0

# select low gradient magnitude regime (lr)
msk_lr = (gra < np.percentile(gra[msk], 25)) & (msk)
n, bins, _ = plt.hist(ima[msk_lr], 100, range=(0, ima_max))

# detect 'pure tissue' peaks (TODO: Checks for always finding 3 peaks)
peaks = peakutils.indexes(n, thres=0.25/max(n), min_dist=10)
# peaks = peaks[0:-1]
tissues = []
for p in peaks:
    tissues.append(bins[p])
tissues = np.array(tissues)

# insert extreme brightness (eg. vessel) anchor
# tissues = np.append(tissues, tissues[-1] + (tissues[-1] - tissues[-2]) / 2)

# insert zero-max arc
zmax_max = tissues[-1] + tissues[-1]-tissues[-2]
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
    out = Nifti1Image(edist.reshape(nii.shape), affine=nii.affine)
    save(out, basename + '_t' + str(i) + '.nii.gz')
soft = np.array(soft)

# interface translation (shift zero circle to another radius)
soft[-1, :] = soft[-1, :] - zmax_radius
# zmax_neg = soft[-1, :] < 0  # voxels fall inside zero-max arc

# weight zero-max arc to not coincide with pure class regions
# zmax_weight = (gra[zmax_neg] / zmax_radius)**-1
zmax_weight = (gra / zmax_radius)**-1
# soft[-1, :][zmax_neg] = zmax_weight*np.abs(soft[-1, :][zmax_neg])
soft[-1, :] = zmax_weight*np.abs(soft[-1, :])
out = Nifti1Image(soft[-1, :].reshape(nii.shape), affine=nii.affine)
save(out, basename + '_zmaxarc' + '.nii.gz')

# arbitrary weighting (TODO: Can be turned into config file of some sort)
# save these values for MPRAGE
# soft[0, :] = soft[0, :] * 0.66  # csf
# soft[1, :] = soft[1, :] * 1.  # gm
# soft[2, :] = soft[2, :] * 1.25  # wm
# soft[3, :] = soft[3, :] * 0.5  # zero-max arc

# hard tissue membership maps
hard = np.argmin(soft, axis=0)

# save intermediate maps
out = Nifti1Image(hard.reshape(nii.shape), affine=nii.affine)
save(out, basename + '_hard' + '.nii.gz')
print 'Finished.'
