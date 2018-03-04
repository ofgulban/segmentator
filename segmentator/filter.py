#!/usr/bin/env python
"""Diffusion based image smoothing."""

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

from __future__ import division
import os
import numpy as np
import config_filters as cfg
from nibabel import load, Nifti1Image, save
from numpy.linalg import eigh
from scipy.ndimage import gaussian_filter
from time import time
from segmentator.filters_utils import (
    self_outer_product, dot_product_matrix_vector, divergence,
    compute_diffusion_weights, construct_diffusion_tensors,
    smooth_matrix_image)


def QC_export(image, basename, identifier):
    """Quality control exports."""
    out = Nifti1Image(image, affine=nii.affine)
    save(out, '{}_{}.nii.gz'.format(basename, identifier))


# Input
file_name = cfg.filename

# Primary parameters
MODE = cfg.smoothing
NR_ITER = cfg.nr_iterations
SAVE_EVERY = cfg.save_every
RHO = cfg.noise_scale  # feature scale
SIGMA = cfg.feature_scale  # noise scale
LAMBDA = cfg.edge_threshold

# Secondary parameters
GAMMA = cfg.diffusion_speed
ALPHA = 0.001
M = 4

# Export parameters
identifier = MODE

# Load data
basename = file_name.split(os.extsep, 1)[0]
nii = load(file_name)
ima = (nii.get_data()).astype('float32')
idx_msk_flat = ima.flatten() != 0
dims = ima.shape

# The main loop
start = time()
for t in range(NR_ITER):
    iteration = str(t+1).zfill(len(str(NR_ITER)))
    print("Iteration: " + iteration)
    # Update export parameters
    params = '{}_n{}_l{}_r{}_s{}_g{}'.format(
        identifier, iteration, LAMBDA, RHO, SIGMA, GAMMA)

    # Smoothing
    if SIGMA == 0:
        ima_temp = np.copy(ima)
    else:
        ima_temp = gaussian_filter(ima, sigma=SIGMA, mode="nearest")

    # Compute gradient
    gra = np.transpose(np.gradient(ima_temp), [1, 2, 3, 0])
    ima_temp = None

    print('  Constructing structure tensors...')
    struct = self_outer_product(gra)

    # Gaussian smoothing on tensor components
    struct = smooth_matrix_image(struct, RHO=RHO)

    # # Unit trace rescaling
    # FIXME: I am not sure I understand this right, or the expected effect
    # rescale = struct[:, 0, 0] + struct[:, 1, 1] + struct[:, 2, 2]
    # struct = struct / np.max(rescale)
    # print('    Trace rescaling factor: {}'.format(np.max(rescale)))

    print('  Running eigen decomposition...')
    struct = struct.reshape([np.prod(dims), 3, 3])
    struct = struct[idx_msk_flat, :, :]
    eigvals, eigvecs = eigh(struct)
    struct = None

    print('  Constructing diffusion tensors...')
    mu = compute_diffusion_weights(eigvals, mode=MODE, LAMBDA=LAMBDA,
                                   ALPHA=ALPHA, M=M)
    difft = construct_diffusion_tensors(eigvecs, weights=mu)
    eigvecs, eigvals, mu = None, None, None

    # Reshape processed voxels (not masked) back to image space
    temp = np.zeros([np.prod(dims), 3, 3])
    temp[idx_msk_flat, :, :] = difft
    difft = temp.reshape(dims + (3, 3))
    temp = None

    # Weickert, 1998, eq. 1.1 (Fick's law).
    negative_flux = dot_product_matrix_vector(difft, gra)
    difft, gra = None, None
    # Weickert, 1998, eq. 1.2 (continuity equation)
    diffusion_difference = divergence(negative_flux)
    negative_flux = None

    # Update image (diffuse image using the difference)
    ima += GAMMA*diffusion_difference
    diffusion_difference = None

    # Convenient exports for intermediate outputs
    if (t+1) % SAVE_EVERY == 0 and (t+1) != NR_ITER:
        QC_export(ima, basename, params)
        duration = time() - start
        mins, secs = int(duration / 60), int(duration % 60)
        print('  Image saved (took {} min {} sec)'.format(mins, secs))

print('Saving final image...')
QC_export(ima, basename, params)

duration = time() - start
mins, secs = int(duration / 60), int(duration % 60)
print('  Finished (Took: {} min {} sec).'.format(mins, secs))
