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


def VolHist2ImaMapping4Vols(vox2pixMap, nrBins):
    """Volume histogram to image mapping for volumes. Uses logical indexing"""
    # get bincount (to know how many voxels are in pixel 0,1,etc.. of volHist)
    counts = np.bincount(vox2pixMap, minlength=nrBins**2)
    # get sorting indices (used to sort list acc. to pixels for vox2pixMap)
    sortIdx = np.argsort(vox2pixMap)
    # transform array to list, so we can hold a varying number of elem
    pix2VoxMap = sortIdx.tolist()
    # get position of zero elements
    posZeros = np.where(counts == 0)[0]
    # prepare to split the list into list of lists
    upperBound = np.delete(counts, posZeros)
    upperBound = np.cumsum(upperBound).astype(int)
    lowerBound = np.hstack(([0], upperBound[0:-1])).astype(int)
    # split the list into list of lists
    pix2VoxMap = [pix2VoxMap[i:j] for i, j in zip(lowerBound, upperBound)]
    # insert empty lists at zero positions
    for idx in posZeros:
        pix2VoxMap.insert(idx, [])
    # convert back to numpy array
    pix2VoxMap = np.array(pix2VoxMap)
    return pix2VoxMap
