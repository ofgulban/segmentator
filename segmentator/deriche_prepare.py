"""Calculate gradient magnitude with 3D Deriche filter.

You can use --graMag flag to pass resulting nifti files from this script.
"""

# Part of the Segmentator library
# Copyright (C) 2016  Omer Faruk Gulban and Marian Schneider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import segmentator.config as cfg
from segmentator.deriche_3D import deriche_3D
import os
import numpy as np
from nibabel import load, save, Nifti1Image
from time import time


def Deriche_Gradient_Magnitude(image, alpha, return_gradients=False):
    """Compute Deriche gradient magnitude of a volumetric image."""
    start = time()
    print('  Computing gradients with alpha: ' + str(alpha))
    # calculate gradients
    print("  .")
    gra_x = deriche_3D(image, alpha=alpha)

    print("  .")
    image = np.transpose(image, (2, 0, 1))
    data_t1 = np.ascontiguousarray(image, dtype=np.float32)
    gra_y = deriche_3D(data_t1, alpha=alpha)
    gra_y = np.transpose(gra_y, (1, 2, 0))

    print("  .")
    image = np.transpose(image, (2, 0, 1))
    data_t2 = np.ascontiguousarray(image, dtype=np.float32)
    gra_z = deriche_3D(data_t2, alpha=alpha)
    gra_z = np.transpose(gra_z, (2, 0, 1))

    end = time()
    print("  Gradients are computed in: " + str(int(end-start)) + " seconds.")
    print("  Saving the gradient magnitude image...")

    # Put the image gradients in 4D format
    gradients = np.array([gra_x, gra_y, gra_z])
    gradients = np.transpose(gradients, (1, 2, 3, 0))

    # Deriche gradient magnitude
    gra_mag = np.sqrt(np.power(gradients[:, :, :, 0], 2.0) +
                      np.power(gradients[:, :, :, 1], 2.0) +
                      np.power(gradients[:, :, :, 2], 2.0))
    return gra_mag


print('-------------------------')
print('Deriche filter initiated.')
for alpha in cfg.deriche_alpha:
    # Load nifti
    nii = load(cfg.filename)
    basename = nii.get_filename().split(os.extsep, 1)[0]
    image = nii.get_data()
    image = np.ascontiguousarray(image, dtype=np.float32)

    # Compute gradient magnitude image with Deriche filter
    gra_mag = Deriche_Gradient_Magnitude(image, alpha=alpha)

    out = Nifti1Image(gra_mag, affine=nii.get_affine())
    outName = basename+'_deriche_a' + str(alpha) + '_graMag'
    outName = outName.replace('.', 'pt')
    save(out, outName + '.nii.gz')
    print('  Saved as: ' + outName)
