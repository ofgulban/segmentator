#!/usr/bin/env python
"""Entry point.

Use config_filters.py to hold parameters.
"""

from __future__ import print_function
import argparse
import segmentator.config_filters as cfg
from segmentator import __version__


def main():
    """Command line call argument parsing."""
    # Instantiate argument parser object:
    parser = argparse.ArgumentParser()

    # Add arguments to namespace:
    parser.add_argument(
        'filename', metavar='path',
        help="Path to input. A nifti file with image data."
        )
    parser.add_argument(
        "--smoothing", metavar=str(cfg.smoothing), required=False,
        default=cfg.smoothing,
        help="Variants are CURED and STEDI. Use CURED for removing tubular,\
        honeycomb-like structures as well as smoothing isotropic areas.\
        STEDI is the more conservative version that retains more edges."
        )
    # parser.add_argument(
    #     "--edge_thr", metavar=str(cfg.edge_thr), required=False,
    #     type=float, default=cfg.edge_thr,
    #     help="Lambda, edge threshold, lower values preserves more edges. Not\
    #     used in CURED and STEDI."
    #     )
    parser.add_argument(
        "--noise_scale", metavar=str(cfg.noise_scale), required=False,
        type=float, default=cfg.noise_scale,
        help="Sigma, determines the spatial scale of the noise that will be \
        corrected. Recommended lower bound is 0.5. Adjusted for each axis \
        to account for non-isotropic voxels. For example, if the selected \
        value is 1 for an image with [0.7, 0.7, 1.4] mm voxels, sigma is \
        adjusted to be [1, 1, 0.5] in the corresponding axes."
        )
    parser.add_argument(
        "--feature_scale", metavar=str(cfg.feature_scale), required=False,
        type=float, default=cfg.noise_scale,
        help="Rho, determines the spatial scale of the features that will be \
        enhanced. Recommended lower bound is 0.5. Adjusted for each axis \
        to account for non-isotropic voxels same way as in --noise_scale."
        )
    parser.add_argument(
        "--gamma", metavar=str(cfg.gamma), required=False,
        type=float, default=cfg.gamma,
        help="Strength of the updates in every iteration. Recommended range is\
        0.5 to 2."
        )
    parser.add_argument(
        "--nr_iterations", metavar=str(cfg.nr_iterations), required=False,
        type=int, default=cfg.nr_iterations,
        help="Number of maximum iterations. More iterations will produce \
        smoother images."
        )
    parser.add_argument(
        "--save_every", metavar=str(cfg.save_every), required=False,
        type=int, default=cfg.save_every,
        help="Save every Nth iterations. Useful to track the effect of \
        smoothing as it evolves."
        )
    parser.add_argument(
        "--downsampling", metavar=str(cfg.downsampling), required=False,
        type=int, default=cfg.downsampling,
        help="(!WIP!) Downsampling factor, use integers > 1. E.g. factor of 2 \
        reduces the amount of voxels 8 times."
        )

    # set cfg file variables to be accessed from other scripts
    args = parser.parse_args()
    cfg.filename = args.filename
    cfg.smoothing = args.smoothing
    # cfg.edge_thr = args.edge_thr  # lambda
    cfg.noise_scale = args.noise_scale  # sigma
    cfg.feature_scale = args.feature_scale  # rho
    cfg.gamma = args.gamma
    cfg.nr_iterations = args.nr_iterations
    cfg.save_every = args.save_every
    cfg.downsampling = args.downsampling

    welcome_str = 'Segmentator {}'.format(__version__)
    welcome_decor = '=' * len(welcome_str)
    print('{}\n{}\n{}'.format(welcome_decor, welcome_str, welcome_decor))
    print('Filters initiated...')

    import segmentator.filter


if __name__ == "__main__":
    main()
