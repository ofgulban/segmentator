"""Entry point.

Mostly following this example:
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Use config.py to hold arguments to be accessed by imported scripts.

TODO: Argument parsing can be better structured, maybe by using parents. help
looks a bit messy as is.
"""

import argparse
import config as cfg
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
        "--gramag", metavar=str(cfg.gramag), required=False,
        default=cfg.gramag,
        help="'scharr', 'sobel', 'prewitt', 'numpy' \
        or path to a gradient magnitude nifti."
        )
    parser.add_argument(
        "--ncut",  metavar='path', required=False,
        help="Path to nyp file with ncut labels"
        )
    parser.add_argument(
        "--scale", metavar=str(cfg.scale), required=False, type=float,
        default=cfg.scale,
        help="Data is scaled from 0 to this number."
        )
    parser.add_argument(
        "--percmin", metavar=str(cfg.perc_min), required=False, type=float,
        default=cfg.perc_min,
        help="Minimum percentile used in truncation."
        )
    parser.add_argument(
        "--percmax",  metavar=str(cfg.perc_max), required=False,  type=float,
        default=cfg.perc_max,
        help="Maximum percentile used in truncation."
        )
    parser.add_argument(
        "--cbar_max",  metavar=str(cfg.cbar_max), required=False,  type=float,
        default=cfg.cbar_max,
        help="Maximum value (power of 10) of the colorbar slider."
        )
    parser.add_argument(
        "--cbar_init",  metavar=str(cfg.cbar_init), required=False,
        type=float, default=cfg.cbar_init,
        help="Initial value (power of 10) of the colorbar slider. \
              Also used with --ncut_prepare flag."
        )
    parser.add_argument(
        "--nogui", action='store_true',
        help="Only save 2D histogram image without showing GUI."
        )
    parser.add_argument(
        "--include_zeros", action='store_true',
        help="Include image zeros in histograms. Not used by default."
        )

    # used in ncut preparation  (TODO: not yet tested after restructuring.)
    parser.add_argument(
        "--ncut_prepare", action='store_true',
        help=("------------------(utility feature)------------------ \
              Use this flag with the following arguments:")
        )
    parser.add_argument(
        "--ncut_figs", action='store_true',
        help="Figures are presented (useful for debugging)."
        )
    parser.add_argument(
        "--ncut_maxRec", required=False, type=int,
        default=cfg.max_rec, metavar=cfg.max_rec,
        help="Maximum number of recursions."
        )
    parser.add_argument(
        "--ncut_nrSupPix", required=False, type=int,
        default=cfg.nr_sup_pix, metavar=cfg.nr_sup_pix,
        help="Number of regions/superpixels."
        )
    parser.add_argument(
        "--ncut_compactness", required=False, type=float,
        default=cfg.compactness, metavar=cfg.compactness,
        help="Compactness balances intensity proximity and space \
        proximity of the superpixels. \
        Higher values give more weight to space proximity, making \
        superpixel shapes more square/cubic. This parameter \
        depends strongly on image contrast and on the shapes of \
        objects in the image."
        )

    # set cfg file variables to be accessed from other scripts
    args = parser.parse_args()
    # used in all
    cfg.filename = args.filename
    # used in segmentator GUI (main and ncut)
    cfg.gramag = args.gramag
    cfg.scale = args.scale
    cfg.perc_min = args.percmin
    cfg.perc_max = args.percmax
    cfg.cbar_max = args.cbar_max
    cfg.cbar_init = args.cbar_init
    if args.include_zeros:
        cfg.discard_zeros = False
    # used in ncut preparation
    cfg.ncut_figs = args.ncut_figs
    cfg.max_rec = args.ncut_maxRec
    cfg.nr_sup_pix = args.ncut_nrSupPix
    cfg.compactness = args.ncut_compactness
    # used in ncut
    cfg.ncut = args.ncut

    welcome_str = 'Segmentator ' + __version__
    welcome_decoration = '=' * len(welcome_str)
    print(welcome_decoration + '\n' + welcome_str + '\n' + welcome_decoration)

    # Call other scripts with import method (couldn't find a better way).
    if args.nogui:
        print '--No GUI option is selected. Saving 2D histogram image...'
        import hist2d_counts
    elif args.ncut_prepare:
        print '--Preparing N-cut related files...'
        import ncut_prepare
    elif args.ncut:
        print '--Experimental N-cut feature is selected.'
        import segmentator_ncut
    else:
        import segmentator_main


if __name__ == "__main__":
    main()
