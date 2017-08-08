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

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, prewitt, laplace, generic_gradient_magnitude

def sub2ind(array_shape, rows, cols):
    """Pixel to voxel mapping (similar to matlab's function).

    Parameters
    ----------
    array_shape : TODO
    rows : TODO
    cols : TODO

    Returns
    -------
    TODO

    """
    # return (rows*array_shape + cols)
    return (cols*array_shape + rows)


def Ima2VolHistMapping(xinput, yinput, binsArray):
    """Image to volume histogram mapping (kind of inverse histogram).

    Parameters
    ----------
    xinput : TODO
        First image, which is often the intensity image (eg. T1w).
    yinput : TODO
        Second image, which is often the gradient magnitude image
        derived from the first image.
    binsArray : TODO

    Returns
    -------
    vox2pixMap : TODO
        Voxel to pixel mapping.

    """
    dgtzData = np.digitize(xinput, binsArray)-1
    dgtzGra = np.digitize(yinput, binsArray)-1
    nrBins = len(binsArray)-1  # subtract 1 (more borders than containers)
    vox2pixMap = sub2ind(nrBins, dgtzData, dgtzGra)  # 1D
    return vox2pixMap


def VolHist2ImaMapping(imaSlc2volHistMap, volHistMask):
    """Volume histogram to image mapping for slices (uses np.ind1).

    Parameters
    ----------
    imaSlc2volHistMap : TODO
    volHistMask : TODO

    Returns
    -------
    imaSlcMask :  TODO

    """
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


def TruncateRange(data, percMin=0.25, percMax=99.75):
    """Truncate too low and too high values.

    Parameters
    ----------
    data : np.ndarray
        Image to be truncated.
    percMin : float
        Percentile minimum.
    percMax : float
        Percentile maximum.

    Returns
    -------
    data : np.ndarray

    """
    percDataMin, percDataMax = np.percentile(data, [percMin, percMax])
    data[data < percDataMin] = percDataMin  # adjust minimum
    data[data > percDataMax] = percDataMax  # adjust maximum
    return data


def ScaleRange(data, scaleFactor=500, delta=0):
    """Scale values as a preprocessing step.

    Parameters
    ----------
    data : np.ndarray
        Image to be scaled.
    scaleFactor : float
        Lower scaleFactors provides faster interface due to loweing the
        resolution of 2D histogram ( 500 seems fast enough).
    delta : float
        Delta ensures that the max data points fall inside the last bin
        when this function is used with histograms.

    Returns
    -------
    data: np.ndarray
        Scaled image.

    """
    scaleFactor = scaleFactor - delta
    data = data - data.min()
    return scaleFactor / data.max() * data


def Hist2D(ima, gra):
    """Prepare 2D histogram related variables.

    Parameters
    ----------
    ima : np.ndarray
        First image, which is often the intensity image (eg. T1w).
    gra : np.ndarray
        Second image, which is often the gradient magnitude image
        derived from the first image.

    Returns
    -------
    counts : integer
    volHistH : TODO
    dataMin : float
        Minimum of the first image.
    dataMax : float
        Maximum of the first image.
    nrBins : integer
        Number of one dimensional bins (not the pixels).
    binEdges : TODO

    Notes
    -----
    This function is modularized to be called from the terminal.

    """
    dataMin = np.round(ima.min())
    dataMax = np.round(ima.max())
    nrBins = int(dataMax - dataMin)
    binEdges = np.arange(dataMin, dataMax+1)
    counts, _, _, volHistH = plt.hist2d(ima, gra, bins=binEdges, cmap='Greys')
    return counts, volHistH, dataMin, dataMax, nrBins, binEdges


def compute_gradient_magnitude(ima, method='sobel'):
    """Compute gradient magnitude of images.

    Parameters
    ----------
    ima : np.ndarray
        First image, which is often the intensity image (eg. T1w).
    method : string
        Gradient computation method. Available options are 'sobel', 'prewitt',
        'numpy'.
    Returns
    -------
    gra_mag : np.ndarray
        Second image, which is often the gradient magnitude image
        derived from the first image
    """
    if method == 'sobel':
        return generic_gradient_magnitude(ima, sobel)/32.
    elif method == 'prewitt':
        return generic_gradient_magnitude(ima, prewitt)/18.
    elif method == 'numpy':
        gra = np.gradient(ima)
        gra_mag = np.zeros(ima.shape)
        for d in range(len(gra)):
            gra_mag = np.sum(gra_mag, np.power(gra[d], 2.))
        return np.sqrt(gra_mag)
    else:
        print 'Gradient magnitude method is invalid!'
