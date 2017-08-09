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
import config as cfg
from nibabel import load
from scipy.ndimage import sobel, prewitt, convolve, generic_gradient_magnitude


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


def map_ima_to_2D_hist(xinput, yinput, bins_arr):
    """Image to volume histogram mapping (kind of inverse histogram).

    Parameters
    ----------
    xinput : TODO
        First image, which is often the intensity image (eg. T1w).
    yinput : TODO
        Second image, which is often the gradient magnitude image
        derived from the first image.
    bins_arr : TODO
        Array of bins.

    Returns
    -------
    vox2pixMap : TODO
        Voxel to pixel mapping.

    """
    dgtzData = np.digitize(xinput, bins_arr)-1
    dgtzGra = np.digitize(yinput, bins_arr)-1
    nrBins = len(bins_arr)-1  # subtract 1 (more borders than containers)
    vox2pixMap = sub2ind(nrBins, dgtzData, dgtzGra)  # 1D
    return vox2pixMap


def map_2D_hist_to_ima(imaSlc2volHistMap, volHistMask):
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


def truncate_range(data, percMin=0.25, percMax=99.75, discard_zeros=True):
    """Truncate too low and too high values.

    Parameters
    ----------
    data : np.ndarray
        Image to be truncated.
    percMin : float
        Percentile minimum.
    percMax : float
        Percentile maximum.
    discard_zeros : bool
        Discard voxels with value 0 from truncation.

    Returns
    -------
    data : np.ndarray

    """
    if discard_zeros:
        msk = data != 0
    else:
        msk = np.ones(data.shape)
    percDataMin, percDataMax = np.percentile(data[msk], [percMin, percMax])
    data[data < percDataMin] = percDataMin  # adjust minimum
    data[data > percDataMax] = percDataMax  # adjust maximum
    data[~msk] = 0  # put back masked out voxels
    return data


def scale_range(data, scale_factor=500, delta=0, discard_zeros=True):
    """Scale values as a preprocessing step.

    Parameters
    ----------
    data : np.ndarray
        Image to be scaled.
    scale_factor : float
        Lower scaleFactors provides faster interface due to loweing the
        resolution of 2D histogram ( 500 seems fast enough).
    delta : float
        Delta ensures that the max data points fall inside the last bin
        when this function is used with histograms.
    discard_zeros : bool
        Discard voxels with value 0 from truncation.

    Returns
    -------
    data: np.ndarray
        Scaled image.

    """
    if discard_zeros:
        msk = data != 0
    else:
        msk = np.ones(data.shape)
    scale_factor = scale_factor - delta
    data[msk] = data[msk] - data[msk].min()
    data[msk] = scale_factor / data[msk].max() * data[msk]
    return data


def prep_2D_hist(ima, gra):
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


def create_3D_kernel(operator='sobel'):
    """Create various 3D kernels.

    Parameters
    ----------
    operator : np.ndarray, shape=(n, n, 3)
        Input can be 'sobel', 'prewitt' or any 3D numpy array.

    Returns
    -------
    kernel : np.ndarray, shape(6, n, n, 3)

    """
    if operator == 'sobel':
        operator = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]],
                            dtype='float')
    elif operator == 'prewitt':
        operator = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]],
                            dtype='float')
    # create permutations operator that will be used in gradient computation
    kernel = np.zeros([6, 3, 3, 3])
    kernel[0, ...] = operator
    kernel[1, ...] = kernel[0, ::-1, ::-1, ::-1]
    kernel[2, ...] = np.transpose(kernel[0, ...], [1, 2, 0])
    kernel[3, ...] = kernel[2, ::-1, ::-1, ::-1]
    kernel[4, ...] = np.transpose(kernel[0, ...], [2, 0, 1])
    kernel[5, ...] = kernel[4, ::-1, ::-1, ::-1]
    return kernel


def compute_gradient_magnitude(ima, method='sobel'):
    """Compute gradient magnitude of images.

    Parameters
    ----------
    ima : np.ndarray
        First image, which is often the intensity image (eg. T1w).
    method : string
        Gradient computation method. Available options are '3D_sobel',
        'scipy_sobel', 'scipy_prewitt', 'numpy'.
    Returns
    -------
    gra_mag : np.ndarray
        Second image, which is often the gradient magnitude image
        derived from the first image
    """
    if method == '3D_sobel':  # magnitude scale is similar to numpy method
        kernel = create_3D_kernel(operator='sobel')
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1))/32.
        return gra_mag
    elif method == '3D_prewitt':
        kernel = create_3D_kernel(operator='prewitt')
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1))/18.
        return gra_mag
    elif method == 'scipy_sobel':
        return generic_gradient_magnitude(ima, sobel)/32.
    elif method == 'scipy_prewitt':
        return generic_gradient_magnitude(ima, prewitt)/18.
    elif method == 'numpy':
        gra = np.asarray(np.gradient(ima))
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=0))
        return gra_mag
    else:
        print 'Gradient magnitude method is invalid!'


def set_gradient_magnitude(image, gramag_option):
    """Set gradient magnitude based on the command line flag.

    Parameters
    ----------
    image : np.ndarray
        First image, which is often the intensity image (eg. T1w).
    gramag_option : string
        A keyword string or a path to a nigti file.

    Returns
    -------
    gra_mag : np.ndarray
        Gradient magnitude image, which has the same shape as image.

    """
    if gramag_option not in cfg.gramag_options:
        gra_mag_nii = load(gramag_option)
        gra_mag = np.squeeze(gra_mag_nii.get_data())
        gra_mag = truncate_range(gra_mag, percMin=cfg.perc_min,
                                 percMax=cfg.perc_max)
        gra_mag = scale_range(gra_mag, scale_factor=cfg.scale, delta=0.0001)

    else:
        gra_mag = compute_gradient_magnitude(image, method=gramag_option)
    return gra_mag
