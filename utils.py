"""Some utility functions."""

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

import numpy as np


def sub2ind(array_shape, rows, cols):
    """Pixel to voxel mapping. Similar to matlabs function."""
    # return (rows*array_shape + cols)
    return (cols*array_shape + rows)


def Ima2VolHistMapping(xinput, yinput, binsArray):
    """Image to volume histogram mapping. Kind of inverse histogram."""
    dgtzData = np.digitize(xinput, binsArray)-1
    dgtzGra = np.digitize(yinput, binsArray)-1
    nrBins = len(binsArray)-1  # subtract 1 (more borders than containers)
    vox2pixMap = sub2ind(nrBins, dgtzData, dgtzGra)  # 1D
    return vox2pixMap


def VolHist2ImaMapping(imaSlc2volHistMap, volHistMask):
    """Volume histogram to image mapping."""
    imaSlcMask = np.zeros(imaSlc2volHistMap.flatten().shape)
    idxUnique = np.unique(volHistMask)
    for idx in idxUnique:
        linIndices = np.where(volHistMask.flatten() == idx)[0]
        # return logical array with length equal to nr of voxels
        voxMask = np.in1d(imaSlc2volHistMap.flatten(), linIndices)
        # reset mask and apply logical indexing
        imaSlcMask[voxMask] = idx
    imaSlcMask = imaSlcMask.reshape(imaSlc2volHistMap.shape)
    return imaSlcMask
