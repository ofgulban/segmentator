"""Command line agument parsing for deriche filter."""

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('filename',  metavar='path',
                    help="Path to a 3D nifti file.")

parser.add_argument("--alpha", "-a", metavar='2',
                    required=False, default=2, type=float,
                    help="Alpha controls smoothing, lower -> smoother")

args = parser.parse_args()

import deriche
