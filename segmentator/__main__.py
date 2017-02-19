"""Entry point.

Mostly following this example:
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

"""

import sys
import argparse


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    print("---")

    # Do argument parsing here (eg. with argparse).
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',  metavar='path',
                        help="Path to nii file with image data")

    parser.add_argument("--gramag", metavar='path',
                        required=False,
                        help="Path to gradient magnitude (useful for deriche)")

    parser.add_argument("--ncut",  metavar='path',
                        required=False,
                        help="Path to nyp file with ncut labels")

    parser.add_argument("--scale", metavar='500',
                        required=False, default=500, type=float,
                        help="Data is scaled from 0 to this number.")

    parser.add_argument("--percmin", metavar='0.25',
                        required=False, default=0.25, type=float,
                        help="Minimum percentile used in truncation.")

    parser.add_argument("--percmax",  metavar='99.75',
                        required=False, default=99.75, type=float,
                        help="Maximum percentile used in truncation.")

    # TODO: could not figure out how to call other scripts properly from here
    # parser.add_argument("--nogui", dest='gui', action='store_false',
    #                     required=False,
    #                     help=("Use this to bypass GUI and only create a 2D \
    #                           histogram file."))
    # parser.set_defaults(gui=True)

    args = parser.parse_args()
    if args.ncut:
        import segmentator_ncut
    else:
        import segmentator_main


if __name__ == "__main__":
    main()
