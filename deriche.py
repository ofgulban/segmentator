"""Calculate gradient magnitude with 3D Deriche filter.

This is not very well integrated into segmentator for now. Plase use --graMag
flag with segmentator.py to load the output of this script (suffix: '_graMag').
"""

import deriche_prepare
from deriche_3D import deriche_3D
import os
import numpy as np
from nibabel import load, save, Nifti1Image
import time

start = time.time()
print('computing gradients...')

nii = load(deriche_prepare.args.filename)
basename = nii.get_filename().split(os.extsep, 1)[0]
data = nii.get_data()
data = np.ascontiguousarray(data, dtype=np.float32)

# calculate gradients
print(".")
alpha = 2
gra_x = deriche_3D(data, alpha=alpha)

print(".")
data = np.transpose(data, (2, 0, 1))
data_t1 = np.ascontiguousarray(data, dtype=np.float32)
gra_y = deriche_3D(data_t1, alpha=alpha)
gra_y = np.transpose(gra_y, (1, 2, 0))

print(".")
data = np.transpose(data, (2, 0, 1))
data_t2 = np.ascontiguousarray(data, dtype=np.float32)
gra_z = deriche_3D(data_t2, alpha=alpha)
gra_z = np.transpose(gra_z, (2, 0, 1))

end = time.time()
print 'Gradients are computed in:', (end - start), 'seconds'

print 'Saving the gradient magnitude image.'
# put the data in 4D format and save
temp = np.array([gra_x, gra_y, gra_z])
temp = np.transpose(temp, (1, 2, 3, 0))
# out = Nifti1Image(temp, affine=nii.get_affine())
# save(out, basename+'_deriche_a' + str(alpha) + '.nii.gz')

# Deriche gradient magnitude
graMag = np.sqrt(np.power(temp[:, :, :, 0], 2) +
                 np.power(temp[:, :, :, 1], 2) +
                 np.power(temp[:, :, :, 2], 2))
out = Nifti1Image(graMag, affine=nii.get_affine())
outName = basename+'_deriche_a' + str(alpha) + '_graMag.nii.gz'
save(out, outName)
print 'Saved as:', outName
