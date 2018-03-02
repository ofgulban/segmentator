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
        help="Path to input. Mostly a nifti file with image data."
        )
    parser.add_argument(
        "--smoothing", metavar=str(cfg.gramag), required=True,
        default='EED',
        help="Edge engancing diffusion (EED) or coherence enhencing diffusion (CED)."
        )
    parser.add_argument(
        "--noise_scale", metavar=str(cfg.noise_scale), required=False, type=float,
        default=cfg.noise_scale,
        help="Lorem ipsum."
        )
    parser.add_argument(
        "--feature_scale", metavar=str(cfg.feature_scale), required=False, type=float,
        default=cfg.noise_scale,
        help="Lorem ipsum."
        )
    parser.add_argument(
        "--iterations",  metavar=str(cfg.cbar_max), required=False,  type=int,
        default=cfg.nr_iter,
        help="Lorem ipsum."
        )
    parser.add_argument(
        "--save_every",  metavar=str(cfg.cbar_max), required=False,  type=int,
        default=cfg.save_every,
        help="Lorem ipsum."
        )
    parser.add_argument(
        "--debug", action='store_true',
        help="Extra outputs. Useful in development."
        )

    # set cfg file variables to be accessed from other scripts
    args = parser.parse_args()
    # used in all
    cfg.filename = args.filename
    # used in segmentator GUI (main and ncut)
    cfg.smoothing = args.gramag
    cfg.noise_scale = args.scale
    cfg.feature_scale = args.percmin
    cfg.iterations = args.percmax
    cfg.save_every = args.cbar_max

    welcome_str = 'Segmentator {}'.format(__version__)
    welcome_decor = '=' * len(welcome_str)
    print('{}\n{}\n{}'.format(welcome_decor, welcome_str, welcome_decor))
    print('Filters...')

if __name__ == "__main__":
    main()
