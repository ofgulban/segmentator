"""This file contains variables that are shared by several modules.

Also useful to hold command line arguments.

"""

# define variables used to initialise the sector mask
init_centre = (0, 0)
init_radius = 100
init_theta = (0, 360)

# segmentator main command line variables
filename = 'sample_filename_here'
gramag = 'gradient_magnitude_sample_here'
perc_min = 0.25
perc_max = 99.75
scale = 400

# used in Deriche filter
deriche_alpha = 2

# used in segmentator ncut
ncut = False
max_rec = 6
nr_sup_pix = 2500
compactness = 2
