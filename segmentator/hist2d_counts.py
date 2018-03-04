#!/usr/bin/env python
"""Save 2D histogram image without displaying GUI."""

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

from __future__ import print_function
import os
import numpy as np
import segmentator.config as cfg
from segmentator.utils import truncate_range, scale_range, check_data
from segmentator.utils import set_gradient_magnitude, prep_2D_hist
from nibabel import load

# load data
nii = load(cfg.filename)
basename = nii.get_filename().split(os.extsep, 1)[0]

# data processing
orig, _ = check_data(nii.get_data(), cfg.force_original_precision)
orig, _, _ = truncate_range(orig, percMin=cfg.perc_min, percMax=cfg.perc_max)
orig = scale_range(orig, scale_factor=cfg.scale, delta=0.0001)
gra = set_gradient_magnitude(orig, cfg.gramag)

# reshape ima (a bit more intuitive for voxel-wise operations)
ima = np.ndarray.flatten(orig)
gra = np.ndarray.flatten(gra)

counts, _, _, _, _, _ = prep_2D_hist(ima, gra, discard_zeros=cfg.discard_zeros)
outName = '{}_volHist_pcMax{}_pcMin{}_sc{}'.format(
    basename, cfg.perc_max, cfg.perc_min, int(cfg.scale))
outName = outName.replace('.', 'pt')
np.save(outName, counts)
print('  Image saved as:\n {}'.format(outName))
