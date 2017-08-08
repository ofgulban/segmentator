"""Save 2D histogram image without displaying GUI."""

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

import os
import numpy as np
import config as cfg
from utils import TruncateRange, ScaleRange, Hist2D, compute_gradient_magnitude
from nibabel import load

# load data
nii = load(cfg.filename)
basename = nii.get_filename().split(os.extsep, 1)[0]

# data processing
orig = np.squeeze(nii.get_data())
orig = TruncateRange(orig, percMin=cfg.perc_min, percMax=cfg.perc_max)
orig = ScaleRange(orig, scaleFactor=cfg.scale, delta=0.0001)

# copy intensity data so we can flatten the copy and leave original intact
ima = orig.copy()
if cfg.gramag not in ['sobel', 'prewitt', 'numpy']:
    nii2 = load(cfg.gramag)
    gra = np.squeeze(nii2.get_data())
    gra = TruncateRange(gra, percMin=cfg.perc_min, percMax=cfg.perc_max)
    gra = ScaleRange(gra, scaleFactor=cfg.scale, delta=0.0001)

else:
    gra = compute_gradient_magnitude(ima, method=cfg.gramag)

# reshape ima (a bit more intuitive for voxel-wise operations)
ima = np.ndarray.flatten(ima)
gra = np.ndarray.flatten(gra)

counts, _, _, _, _, _ = Hist2D(ima, gra)
outName = (basename + '_volHist'
           + '_pMax' + str(cfg.perc_max) + '_pMin' + str(cfg.perc_min)
           + '_sc' + str(int(cfg.scale))
           )
outName = outName.replace('.', 'pt')
np.save(outName, counts)
print '----Image saved as:\n ' + outName
