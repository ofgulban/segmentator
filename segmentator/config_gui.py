"""This file contains GUI related variables."""

import matplotlib.pyplot as plt

# Colormap for mask overlays
palette = plt.cm.Reds
palette.set_over('r', 1.0)
palette.set_under('w', 0)
palette.set_bad('m', 1.0)

# Slider colors
axcolor = '0.875'
hovcolor = '0.975'
