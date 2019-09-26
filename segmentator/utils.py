#!/usr/bin/env python
"""Some utility functions."""

from __future__ import division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import segmentator.config as cfg
from nibabel import load, Nifti1Image, save
from scipy.ndimage import convolve
from time import time


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
    nr_bins = len(bins_arr)-1  # subtract 1 (more borders than containers)
    vox2pixMap = sub2ind(nr_bins, dgtzData, dgtzGra)  # 1D
    return vox2pixMap


def map_2D_hist_to_ima(imaSlc2volHistMap, volHistMask):
    """Volume histogram to image mapping for slices (uses np.ind1).

    Parameters
    ----------
    imaSlc2volHistMap : 1D numpy array
        Flattened image slice.
    volHistMask : 1D numpy array
        Flattened volume histogram mask.

    Returns
    -------
    imaSlcMask : 1D numpy array
        Flat image slice mask based on labeled pixels in volume histogram.

    """
    imaSlcMask = np.zeros(imaSlc2volHistMap.shape)
    idxUnique = np.unique(volHistMask)
    for idx in idxUnique:
        linIndices = np.where(volHistMask.flatten() == idx)[0]
        # return logical array with length equal to nr of voxels
        voxMask = np.in1d(imaSlc2volHistMap, linIndices)
        # reset mask and apply logical indexing
        imaSlcMask[voxMask] = idx
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
        Truncated data.
    pMin : float
        Minimum truncation threshold which is used.
    pMax : float
        Maximum truncation threshold which is used.

    """
    if discard_zeros:
        msk = ~np.isclose(data, 0.)
        pMin, pMax = np.nanpercentile(data[msk], [percMin, percMax])
    else:
        pMin, pMax = np.nanpercentile(data, [percMin, percMax])
    temp = data[~np.isnan(data)]
    temp[temp < pMin], temp[temp > pMax] = pMin, pMax  # truncate min and max
    data[~np.isnan(data)] = temp
    if discard_zeros:
        data[~msk] = 0  # put back masked out voxels
    return data, pMin, pMax


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
        msk = ~np.isclose(data, 0)
    else:
        msk = np.ones(data.shape, dtype=bool)
    scale_factor = scale_factor - delta
    data[msk] = data[msk] - np.nanmin(data[msk])
    data[msk] = scale_factor / np.nanmax(data[msk]) * data[msk]
    if discard_zeros:
        data[~msk] = 0  # put back masked out voxels
    return data


def check_data(data, force_original_precision=True):
    """Do type casting here."""
    data = np.squeeze(data)  # to prevent singular dimension error
    dims = data.shape
    print('Input image data type is {}.'.format(data.dtype.name))
    if force_original_precision:
        pass
    elif data.dtype != 'float32':
        data = data.astype('float32')
        print('  Data type is casted to {}.'.format(data.dtype.name))
    return data, dims


def prep_2D_hist(ima, gra, discard_zeros=True):
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
    d_min : float
        Minimum of the first image.
    d_max : float
        Maximum of the first image.
    nr_bins : integer
        Number of one dimensional bins (not the pixels).
    bin_edges : TODO

    Notes
    -----
    This function is modularized to be called from the terminal.

    """
    if discard_zeros:
        gra = gra[~np.isclose(ima, 0)]
        ima = ima[~np.isclose(ima, 0)]
    d_min, d_max = np.round(np.nanpercentile(ima, [0, 100]))
    nr_bins = int(d_max - d_min)
    bin_edges = np.arange(d_min, d_max+1)
    counts, _, _, volHistH = plt.hist2d(ima, gra, bins=bin_edges, cmap='Greys')
    return counts, volHistH, d_min, d_max, nr_bins, bin_edges


def create_3D_kernel(operator='scharr'):
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
                            dtype='float32')
    elif operator == 'prewitt':
        operator = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]],
                            dtype='float32')
    elif operator == 'scharr':
        operator = np.array([[[9, 30, 9], [30, 100, 30], [9, 30, 9]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]]],
                            dtype='float32')
    scale_normalization_factor = np.sum(np.abs(operator))
    operator = np.divide(operator, scale_normalization_factor)

    # create permutations operator that will be used in gradient computation
    kernel = np.zeros([3, 3, 3, 3])
    kernel[0, ...] = operator
    kernel[1, ...] = np.transpose(kernel[0, ...], [2, 0, 1])
    kernel[2, ...] = np.transpose(kernel[0, ...], [1, 2, 0])
    return kernel


def compute_gradient_magnitude(ima, method='scharr'):
    """Compute gradient magnitude of images.

    Parameters
    ----------
    ima : np.ndarray
        First image, which is often the intensity image (eg. T1w).
    method : string
        Gradient computation method. Available options are 'scharr',
        'sobel', 'prewitt', 'numpy'.
    Returns
    -------
    gra_mag : np.ndarray
        Second image, which is often the gradient magnitude image
        derived from the first image

    """
    start = time()
    print('  Computing gradients...')
    if method.lower() == 'sobel':  # magnitude scale is similar to numpy method
        kernel = create_3D_kernel(operator=method)
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1) * 2.)
    elif method.lower() == 'prewitt':
        kernel = create_3D_kernel(operator=method)
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1) * 2.)
    elif method.lower() == 'scharr':
        kernel = create_3D_kernel(operator=method)
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1) * 2.)
    elif method.lower() == 'numpy':
        gra = np.asarray(np.gradient(ima))
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=0))
    elif method.lower() == 'deriche':
        from segmentator.deriche_prepare import Deriche_Gradient_Magnitude
        alpha = cfg.deriche_alpha
        print('    Selected alpha: {}'.format(alpha))
        ima = np.ascontiguousarray(ima, dtype=np.float32)
        gra_mag = Deriche_Gradient_Magnitude(ima, alpha, normalize=True)
    else:
        print('  Gradient magnitude method is invalid!')
    end = time()
    print("  Gradient magnitude computed in: " + str(int(end-start))
          + " seconds.")
    return gra_mag


def set_gradient_magnitude(image, gramag_option):
    """Set gradient magnitude based on the command line flag.

    Parameters
    ----------
    image : np.ndarray
        First image, which is often the intensity image (eg. T1w).
    gramag_option : string
        A keyword string or a path to a nifti file.

    Returns
    -------
    gra_mag : np.ndarray
        Gradient magnitude image, which has the same shape as image.

    """
    if gramag_option not in cfg.gramag_options:
        print("Selected gradient magnitude method is not available,"
              + " interpreting as a file path...")
        gra_mag_nii = load(gramag_option)
        gra_mag = np.squeeze(gra_mag_nii.get_data())
        gra_mag, _ = check_data(gra_mag, cfg.force_original_precision)
        gra_mag, _, _ = truncate_range(gra_mag, percMin=cfg.perc_min,
                                       percMax=cfg.perc_max)
        gra_mag = scale_range(gra_mag, scale_factor=cfg.scale, delta=0.0001)

    else:
        print('{} gradient method is selected.'.format(gramag_option.title()))
        gra_mag = compute_gradient_magnitude(image, method=gramag_option)
    return gra_mag


def export_gradient_magnitude_image(img, filename, filtername, affine):
    """Export computed gradient magnitude image as a nifti file."""
    basename = filename.split(os.extsep, 1)[0]
    out_img = Nifti1Image(img, affine=affine)
    if filtername == 'deriche':  # add alpha as suffix for extra information
        filtername = '{}_alpha{}'.format(filtername.title(),
                                         cfg.deriche_alpha)
        filtername = filtername.replace('.', 'pt')
    else:
        filtername = filtername.title()
    out_path = '{}_GraMag{}.nii.gz'.format(basename, filtername)
    print("Exporting gradient magnitude image...")
    save(out_img, out_path)
    print('  Gradient magnitude image exported in this path:\n  ' + out_path)
