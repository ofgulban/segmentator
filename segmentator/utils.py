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
import warnings
import matplotlib.pyplot as plt
import config as cfg
from nibabel import load
from scipy.ndimage import convolve


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
        msk = ~np.isclose(data, 0)
        pMin, pMax = np.nanpercentile(data[msk], [percMin, percMax])
    else:
        pMin, pMax = np.nanpercentile(data, [percMin, percMax])
    temp = data[~np.isnan(data)]
    temp[temp < pMin], temp[temp > pMax] = pMin, pMax  # truncate min and max
    data[~np.isnan(data)] = temp
    if discard_zeros:
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
        msk = ~np.isclose(data, 0)
    else:
        msk = np.ones(data.shape, dtype=bool)
    scale_factor = scale_factor - delta
    data[msk] = data[msk] - np.nanmin(data[msk])
    data[msk] = scale_factor / np.nanmax(data[msk]) * data[msk]
    if discard_zeros:
        data[~msk] = 0  # put back masked out voxels
    return data


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


def create_3D_kernel(operator='3D_sobel'):
    """Create various 3D kernels.

    Parameters
    ----------
    operator : np.ndarray, shape=(n, n, 3)
        Input can be 'sobel', 'prewitt' or any 3D numpy array.

    Returns
    -------
    kernel : np.ndarray, shape(6, n, n, 3)

    """
    if operator == '3D_sobel':
        operator = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]],
                            dtype='float')
    elif operator == '3D_prewitt':
        operator = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]],
                            dtype='float')
    elif operator == '3D_scharr':
        operator = np.array([[[9, 30, 9], [30, 100, 30], [9, 30, 9]],
                             [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             [[-9, -30, -9], [-30, -100, -30], [-9, -30, -9]]],
                            dtype='float')
    scale_normalization_factor = np.sum(np.abs(operator))
    operator = np.divide(operator, scale_normalization_factor)

    # create permutations operator that will be used in gradient computation
    kernel = np.zeros([6, 3, 3, 3])
    kernel[0, ...] = operator
    kernel[1, ...] = kernel[0, ::-1, ::-1, ::-1]
    kernel[2, ...] = np.transpose(kernel[0, ...], [1, 2, 0])
    kernel[3, ...] = kernel[2, ::-1, ::-1, ::-1]
    kernel[4, ...] = np.transpose(kernel[0, ...], [2, 0, 1])
    kernel[5, ...] = kernel[4, ::-1, ::-1, ::-1]
    return kernel


def compute_gradient_magnitude(ima, method='3D_scharr'):
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
        kernel = create_3D_kernel(operator=method)
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1))
        return gra_mag
    elif method == '3D_prewitt':
        kernel = create_3D_kernel(operator=method)
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1))
        return gra_mag
    elif method == '3D_scharr':
        kernel = create_3D_kernel(operator=method)
        gra = np.zeros(ima.shape + (kernel.shape[0],))
        for d in range(kernel.shape[0]):
            gra[..., d] = convolve(ima, kernel[d, ...])
        # compute generic gradient magnitude with normalization
        gra_mag = np.sqrt(np.sum(np.power(gra, 2.), axis=-1))
        return gra_mag
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


def aniso_diff_3D(stack, niter=1, kappa=50, gamma=0.1, step=(1., 1., 1.),
                  option=1, ploton=False):
    """3D anisotropic diffusion based smoothing.

    Disclosure
    ----------
    This script is adapted from a stackoverflow  post by user ali_m:
    [1] http://stackoverflow.com/questions/10802611/anisotropic-diffusion-2d-images  #noqa
    [2] http://pastebin.com/sBsPX4Y7

    Parameters
    ----------
    stack : 3d numpy array
        Input stack/image/volume/data.
    niter : int
        Number of iterations
    kappa : float
        Conduction coefficient (20-100?). Controls conduction as a function of
        gradient.  If kappa is low small intensity gradients are able to block
        conduction and hence diffusion across step edges. A large value reduces
        the influence of intensity gradients on conduction.
    gamma : float
        Controls speed of diffusion (you usually want it at a maximum of 0.25
        for stability).
    step : tuple
        The distance between adjacent pixels in (z,y,x). Step is used to scale
        the gradients in case the spacing between adjacent pixels differs in
        the x,y and/or z axes.
    option : int, 1 or 2
        1 favours high contrast edges over low contrast ones (Perona & Malik
        [1] diffusion equation No 1).
        2 favours wide regions over smaller ones (Perona & Malik [1] diffusion
        equation No 2).
    ploton : bool
        If True, the middle z-plane will be plotted on every iteration.

    Returns
    -------
    stackout : 3d numpy array
        Diffused stack/image/volume/data.

    Reference
    ---------
    [1] P. Perona and J. Malik.
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine
        Intelligence, 12(7):629-639, July 1990.

    Notes
    -----
    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering,
    The University of Western Australia
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology, University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion equation No 2.
    July 2012 translated to Python
    January 2017 docstring reorganization.

    """
    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
            warnings.warn("Only grayscale stacks allowed, converting to 3D \
                          matrix")
            stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl

        showplane = stack.shape[0]//2

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(stack[showplane, ...].squeeze(),
                   interpolation='nearest')
        ih = ax2.imshow(stackout[showplane, ...].squeeze(),
                        interpolation='nearest', animated=True)
        ax1.set_title("Original stack (Z = %i)" % showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in xrange(niter):

            # calculate the diffs
            deltaD[:-1, :, :] = np.diff(stackout, axis=0)
            deltaS[:, :-1, :] = np.diff(stackout, axis=1)
            deltaE[:, :, :-1] = np.diff(stackout, axis=2)

            # conduction gradients (only need to compute one per dim!)
            if option == 1:
                    gD = np.exp(-(deltaD/kappa)**2.)/step[0]
                    gS = np.exp(-(deltaS/kappa)**2.)/step[1]
                    gE = np.exp(-(deltaE/kappa)**2.)/step[2]
            elif option == 2:
                    gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
                    gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
                    gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

            # update matrices
            D = gD*deltaD
            E = gE*deltaE
            S = gS*deltaS

            # subtract a copy that has been shifted 'Up/North/West' by one
            # pixel. don't as questions. just do it. trust me.
            UD[:] = D
            NS[:] = S
            EW[:] = E
            UD[1:, :, :] -= D[:-1, :, :]
            NS[:, 1:, :] -= S[:, :-1, :]
            EW[:, :, 1:] -= E[:, :, :-1]

            # update the image
            stackout += gamma*(UD+NS+EW)

            if ploton:
                    iterstring = "Iteration %i" % (ii+1)
                    ih.set_data(stackout[showplane, ...].squeeze())
                    ax2.set_title(iterstring)
                    fig.canvas.draw()
                    # sleep(0.01)

    return stackout
