"""Make the version number available."""

import pkg_resources  # part of setuptools

__version__ = pkg_resources.require("segmentator")[0].version
