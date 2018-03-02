#!/usr/bin/env python

"""Common functions used in filters."""
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
from scipy.ndimage.filters import gaussian_filter


def self_outer_product(vector_field):
    """Vectorized computation of outer product.

    Parameters
    ----------
    vector_field: np.ndarray, shape(..., 3)

    Returns
    -------
    outer: np.ndarray, shape(..., 3, 3)
    """
    dims = vector_field.shape
    outer = np.repeat(vector_field, dims[-1], axis=-1)
    outer = outer * outer[..., [0, 3, 6, 1, 4, 7, 2, 5, 8]]
    outer = outer.reshape((np.prod(dims[:-1]), dims[-1], dims[-1]))
    return outer


def dot_product_matrix_vector(matrix_field, vector_field):
    """Vectorized computation of dot product."""
    dims = vector_field.shape
    dotp = np.repeat(vector_field, dims[-1], axis=-1)
    dotp = dotp.reshape(dims[:-1] + (dims[-1], dims[-1]))
    dims = dotp.shape  # update dimensions
    idx_dims = tuple(range(dotp.ndim))
    dotp = dotp.transpose(idx_dims[:-2] + (idx_dims[-1], idx_dims[-2]))
    dotp = np.multiply(matrix_field, dotp)
    dotp = np.sum(dotp, axis=-1)
    return dotp


def divergence(vector_field):
    """Vectorized computation of divergence, also called Laplacian."""
    gra_1, _, _ = np.gradient(vector_field[..., 0])
    _, gra_2, _ = np.gradient(vector_field[..., 1])
    _, _, gra_3 = np.gradient(vector_field[..., 2])
    return gra_1 + gra_2 + gra_3


def compute_diffusion_weights(eigvals, mode, LAMBDA=0.001, ALPHA=0.001, M=4):
    """Vectorized computation diffusion weights."""
    mu = np.ones(eigvals.shape)
    idx_pos = eigvals[:, 0] > 0  # positive indices

    if mode is 'EED':  # edge enhancing diffusion
        term = LAMBDA / (eigvals[idx_pos, 1:] - eigvals[idx_pos, 0, None])
        mu[idx_pos, 1:] = 1. - (1. - ALPHA) * np.exp(-term**M)
    elif mode is 'cEED':  # conservative edge enhancing diffusion
        term = LAMBDA / eigvals[idx_pos, :]
        mu[idx_pos, :] = 1. - (1. - ALPHA) * np.exp(-term**M)

    # weights for the non-positive eigen values
    mu[eigvals[:, 1] <= 0, 2] = ALPHA  # surely surfels
    mu[eigvals[:, 0] <= 0, 1:] = ALPHA  # surely curvels
    return mu


def construct_diffusion_tensors(eigvecs, weights):
    """Vectorized consruction of diffusion tensors."""
    dims = eigvecs.shape
    D = np.zeros(dims[:-2] + (dims[-1], dims[-1]))
    for i in range(3):  # weight vectors
        temp = weights[:, i, None, None] * self_outer_product(eigvecs[..., i])
        D = D + temp
    return D


def smooth_matrix_image(matrix_image, RHO=0):
    """Gaussian smoothing applied to matrix image."""
    if RHO == 0:
        return matrix_image
    else:
        dims = matrix_image.shape
        matrix_image = np.zeros(dims)
        for x in range(dims[-2]):
            for y in range(dims[-1]):
                matrix_image[..., x, y] = gaussian_filter(
                    matrix_image[..., x, y], sigma=RHO, mode='nearest')
        return matrix_image
