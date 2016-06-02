#!/usr/bin/env python

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


# Pixel to voxel mapping
def sub2ind(array_shape, rows, cols):
    # return (rows*array_shape + cols)
    return (cols*array_shape + rows)


# Kind of inverse histogram
def Ima2VolHistMapping(xinput, yinput, binsArray):
    dgtzData = np.digitize(xinput, binsArray)-1
    dgtzGra = np.digitize(yinput, binsArray)-1
    vox2pixMap = sub2ind(binsArray.shape, dgtzData, dgtzGra)  # 1D
    return vox2pixMap


def VolHist2ImaMapping(data2D, volHistMask):

    linIndices = np.arange(0, volHistMask.size)
    idxMask = linIndices[volHistMask.flatten()]

    # return logical array with length equal to nr of voxels
    voxMask = np.in1d(data2D.flatten(), idxMask)

    # reset mask and apply logical indexing
    mask2D = np.zeros(data2D.flatten().shape)
    mask2D[voxMask] = 1
    mask2D = mask2D.reshape(data2D.shape)
    return mask2D
