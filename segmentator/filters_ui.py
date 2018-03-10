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
        default='EED',
        help="EED, cEED, CED, cCED."
        )
    parser.add_argument(
        "--edge_threshold", metavar=str(cfg.edge_threshold), required=False,
        type=float, default=cfg.edge_threshold,
        help="Also known as lambda."
        )
    parser.add_argument(
        "--noise_scale", metavar=str(cfg.noise_scale), required=False,
        type=float, default=cfg.noise_scale,
        help="lower bound 0.5 for now"
        )
    parser.add_argument(
        "--feature_scale", metavar=str(cfg.feature_scale), required=False,
        type=float, default=cfg.noise_scale,
        help="lower bound is 0.5 for now"
        )
    parser.add_argument(
        "--nr_iterations", metavar=str(cfg.nr_iterations), required=False,
        type=int, default=cfg.nr_iterations,
        help="Number of maximum iterations."
        )
    parser.add_argument(
        "--save_every", metavar=str(cfg.save_every), required=False,
        type=int, default=cfg.save_every,
        help="Save every Nth iterations."
        )

    # set cfg file variables to be accessed from other scripts
    args = parser.parse_args()
    cfg.filename = args.filename
    cfg.smoothing = args.smoothing
    cfg.edge_threshold = args.edge_threshold  # lambda
    cfg.noise_scale = args.noise_scale
    cfg.feature_scale = args.feature_scale
    cfg.nr_iterations = args.nr_iterations
    cfg.save_every = args.save_every

    welcome_str = 'Segmentator {}'.format(__version__)
    welcome_decor = '=' * len(welcome_str)
    print('{}\n{}\n{}'.format(welcome_decor, welcome_str, welcome_decor))
    print('Filters initiated...')
    print('  !!!WARNING | Highly experimental feature!!!')

    import segmentator.filter

if __name__ == "__main__":
    main()
