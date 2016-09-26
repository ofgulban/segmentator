"""Command line agument parsing for segmentator."""

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
import argparse
import numpy as np
from utils import TruncateRange, ScaleRange, Hist2D
from nibabel import load

parser = argparse.ArgumentParser()

parser.add_argument('filename',  metavar='path',
                    help="Path to nii file with image data")

parser.add_argument("--gramag", metavar='path',
                    required=False,
                    help="Path to gradient magnitude (useful for deriche)")

parser.add_argument("--scale", metavar='500',
                    required=False, default=500, type=float,
                    help="Data is scaled from 0 to this number.")

parser.add_argument("--percmin", metavar='0.25',
                    required=False, default=0.25, type=float,
                    help="Minimum percentile used in truncation.")

parser.add_argument("--percmax",  metavar='99.75',
                    required=False, default=99.75, type=float,
                    help="Maximum percentile used in truncation.")

args = parser.parse_args()

#
"""Load Data"""
nii = load(args.filename)
basename = nii.get_filename().split(os.extsep, 1)[0]

#
"""Data Processing"""
orig = np.squeeze(nii.get_data())

percMin, percMax = args.percmin, args.percmax
orig = TruncateRange(orig, percMin=percMin, percMax=percMax)
orig = ScaleRange(orig, scaleFactor=args.scale, delta=0.0001)

# copy intensity data so we can flatten the copy and leave original intact
ima = orig.copy()
if args.gramag:
    nii2 = load(args.gramag)
    gra = np.squeeze(nii2.get_data())
    gra = TruncateRange(gra, percMin=percMin, percMax=percMax)
    gra = ScaleRange(gra, scaleFactor=args.scale, delta=0.0001)

else:
    # calculate gradient magnitude (using L2 norm of the vector)
    gra = np.gradient(ima)
    gra = np.sqrt(np.power(gra[0], 2) + np.power(gra[1], 2) +
                  np.power(gra[2], 2))

# reshape ima (more intuitive for voxel-wise operations)
ima = np.ndarray.flatten(ima)
gra = np.ndarray.flatten(gra)

counts, _, _, _, _, _ = Hist2D(ima, gra)
outName = basename + '_volHist'
np.save(outName, counts)
print 'Counts saved as: ' + outName
