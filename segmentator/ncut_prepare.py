"""Command line agument parsing for normalized graph cuts."""

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

import norm_graph_cut
