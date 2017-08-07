"""Test and demonstsrate arcweld classification.

TODO: Turn this into unit tests for arcweld.

"""

import numpy as np
import matplotlib.pyplot as plt

# create toy data
res = 500  # resolution
data = np.mgrid[0:res, 0:res].astype('float')
ima, gra = data[0, :, :].flatten(), data[1, :, :].flatten()
dims = data.shape

# have 3 arbitrary classes (standing for classes csf, gm, wm)
classes = np.array([res/7, res/7*4, res/7*6])

# find arc anchor
arc_center = (data.max() + data.min()) / 2.
arc_radius = (data.max() - data.min()) / 2.
arc_weight = (gra / arc_radius)**-1
classes = np.hstack([classes, arc_center])

# find euclidean distances to classes
soft = []
data = data.reshape(dims[0], dims[1]*dims[2])
for i, a in enumerate(classes):
    c = np.array([a, 0])
    # euclidean distance
    edist = np.sqrt(np.sum((data - c[:, None])**2., axis=0))
    soft.append(edist)
soft = np.asarray(soft)

# arc translation
soft[-1, :] = soft[-1, :] - arc_radius
soft[-1, :] = arc_weight * np.abs(soft[-1, :])

# arbitrary weights
soft[0, :] = soft[0, :] * 0.66  # csf
soft[-1, :] = soft[-1, :] * 0.5  # arc

# hard class membership maps
hard = np.argmin(soft, axis=0)
hard = hard.reshape(dims[1], dims[2])

plt.imshow(hard.T, origin="lower")
plt.show()
