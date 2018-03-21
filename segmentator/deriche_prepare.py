"""Calculate gradient magnitude with 3D Deriche filter.

You can use --graMag flag to pass resulting nifti files from this script.
"""

# Part of the Segmentator library
# Copyright (C) 2018  Omer Faruk Gulban and Marian Schneider
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

from segmentator.deriche_3D import deriche_3D
import numpy as np


def Deriche_Gradient_Magnitude(image, alpha, normalize=False,
                               return_gradients=False):
    """Compute Deriche gradient magnitude of a volumetric image."""
    # calculate gradients
    image = np.ascontiguousarray(image, dtype=np.float32)
    gra_x = deriche_3D(image, alpha=alpha)
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image, dtype=np.float32)
    gra_y = deriche_3D(image, alpha=alpha)
    gra_y = np.transpose(gra_y, (1, 2, 0))
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image, dtype=np.float32)
    gra_z = deriche_3D(image, alpha=alpha)
    gra_z = np.transpose(gra_z, (2, 0, 1))

    # Put the image gradients in 4D format
    gradients = np.array([gra_x, gra_y, gra_z])
    gradients = np.transpose(gradients, (1, 2, 3, 0))

    if return_gradients:
        return gradients

    else:  # Deriche gradient magnitude
        gra_mag = np.sqrt(np.power(gradients[:, :, :, 0], 2.0) +
                          np.power(gradients[:, :, :, 1], 2.0) +
                          np.power(gradients[:, :, :, 2], 2.0))
        if normalize:
            min_ima, max_ima = np.percentile(image, [0, 100])
            min_der, max_der = np.percentile(gra_mag, [0, 100])
            range_ima, range_der = max_ima - min_ima, max_der - min_der

            gra_mag = gra_mag * (range_ima / range_der)
        return gra_mag
