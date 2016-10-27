"""Command line agument parsing for normalized graph cuts."""

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

parser.add_argument('filename',  metavar='path',
                    help="Path to npy file with volume histogram counts data")

parser.add_argument("--maxRec", metavar='6',
                    required=False, default=6, type=int,
                    help="Maximum number of recursions.")

parser.add_argument("--nrSupPix", "-sp", metavar='2500',
                    required=False, default=2500, type=int,
                    help="Number of regions/superpixels.")

parser.add_argument("--compactness", "-c", metavar='2',
                    required=False, default=2, type=float,
                    help="Compactness balances intensity proximity and space \
                    proximity of the superpixels. \
                    Higher values give more weight to space proximity, making \
                    superpixel shapes more square/cubic. This parameter \
                    depends strongly on image contrast and on the shapes of \
                    objects in the image.")

args = parser.parse_args()

print('initializing...')
