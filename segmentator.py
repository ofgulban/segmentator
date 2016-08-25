#!/usr/bin/env python

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    help="path to nii file with image data",)
parser.add_argument("--ncut", '-l',
                    required=False,
                    help="path to nyp file with ncut labels",)
args = parser.parse_args()


if args.ncut:
    import segmentator_ncut
else:
    import segmentator_main

print('initializing...')
