"""3D Deriche filter implementation."""

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

import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def deriche_3D(np.ndarray[float, mode="c", ndim=3] inputData, float alpha=1):
    """Reference: Monga et al. 1991."""
    # c definitions
    cdef float s, a_0, a_1, a_2, a_3 , b_1, b_2
    cdef np.ndarray[double, mode="c", ndim=3] S_p, S_n, R_p, R_n, T_p, T_n
    cdef np.ndarray[double, mode="c", ndim=3] S, R, P
    cdef int imax = inputData.shape[0]
    cdef int jmax = inputData.shape[1]
    cdef int kmax = inputData.shape[2]
    cdef int i, j, k

    # constants
    s = np.power((1 - np.exp(-alpha)), 2) / \
        (1 + 2 * alpha * np.exp(-alpha) - np.exp(-2*alpha))
    a_0 = s
    a_1 = s * (alpha - 1) * np.exp(-alpha)
    b_1 = -2 * np.exp(-alpha)
    b_2 = np.exp(-2 * alpha)
    a_2 = a_1 - s * b_1
    a_3 = -s * b_2

    # Recursive filter implementation
    S_p = np.zeros((imax, jmax, kmax))
    S_n = np.zeros((imax, jmax, kmax))
    for k in range(kmax):
        for j in range(jmax):
            for i in range(0, imax):
                if i > 2:
                    S_p[i, j, k] = inputData[i-1, j, k] \
                                   - b_1*S_p[i-1, j, k] \
                                   - b_2*S_p[i-2, j, k]
                i = i*-1 % inputData.shape[0]  # inverse index
                if i < inputData.shape[0]-2:
                    S_n[i, j, k] = inputData[i+1, j, k] \
                                   - b_1*S_n[i+1, j, k] \
                                   - b_2*S_n[i+2, j, k]

    S = alpha*(S_p - S_n)

    R_p = np.zeros((imax, jmax, kmax))
    R_n = np.zeros((imax, jmax, kmax))
    for k in range(kmax):
        for i in range(imax):
            for j in range(0, jmax):
                if j > 2:
                    R_p[i, j, k] = a_0*S[i, j, k] + a_1*S[i, j-1, k] \
                                   - b_1*R_p[i, j-1, k] - b_2*R_p[i, j-2, k]
                j = j*-1 % inputData.shape[1]  # inverse index
                if j < inputData.shape[1]-2:
                    R_n[i, j, k] = a_2*S[i, j+1, k] + a_3*S[i, j+2, k] \
                                   - b_1*R_n[i, j+1, k] - b_2*R_n[i, j+2, k]

    R = R_n + R_p

    T_p = np.zeros((imax, jmax, kmax))
    T_n = np.zeros((imax, jmax, kmax))
    for i in range(imax):
        for j in range(jmax):
            for k in range(0, kmax):
                if k > 2:
                    T_p[i, j, k] = a_0*R[i, j, k] + a_1*R[i, j, k-1] \
                                 - b_1*T_p[i, j, k-1] - b_2*T_p[i, j, k-2]
                k = k*-1 % inputData.shape[2]  # inverse index
                if k < inputData.shape[2]-2:
                    T_n[i, j, k] = a_2*R[i, j, k+1] + a_3*R[i, j, k+2] \
                                   - b_1*T_n[i, j, k+1] - b_2*T_n[i, j, k+2]

    T = T_n + T_p
    return T
