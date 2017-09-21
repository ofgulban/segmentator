"""This file contains variables that are shared by several modules.

Also useful to hold command line arguments.

"""

# define variables used to initialise the sector mask
init_centre = (0, 0)
init_radius = 100
init_theta = (0, 360)

# segmentator main command line variables
filename = 'sample_filename_here'
gramag = '3D_scharr'
perc_min = 2.5
perc_max = 97.5
scale = 400

# possible gradient magnitude computation keyword options
gramag_options = ['3D_scharr', '3D_sobel', '3D_prewitt', 'numpy']

# used in segmentator ncut
ncut = False
max_rec = 8
nr_sup_pix = 2500
compactness = 2
