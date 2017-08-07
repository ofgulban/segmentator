"""Test and demonstsrate arcweld classification.

TODO: Turn this into unit tests for arcweld.

"""

import numpy as np
import matplotlib.pyplot as plt

# create toy data
res = 100  # resolution
data = np.mgrid[0:res, 0:res].astype('float')
ima, gra = data[0, :, :].flatten(), data[1, :, :].flatten()
dims = data.shape

# have 3 arbitrary classes (standing for classes csf, gm, wm)
classes = np.array([res/5, res/5*3, res/5*4])

# find arc anchor
arc_center = (data.max() + data.min()) / 2.
arc_radius = (data.max() - data.min()) / 2.
arc_weight = (gra / arc_radius)**-1
classes = np.hstack([classes, arc_center])

# find euclidean distances to classes
soft = []
data = data.reshape(dims[0], dims[1]*dims[2])
for i, a in enumerate(classes):
    tissue = np.array([a, 0])
    # euclidean distance
    edist = np.sqrt(np.sum((data - tissue[:, None])**2., axis=0))
    soft.append(edist)
soft = np.asarray(soft)

# arc translation
soft[-1, :] = soft[-1, :] - arc_radius
soft[-1, :] = arc_weight * np.abs(soft[-1, :])

# hard tissue membership maps
hard = np.argmin(soft, axis=0)
hard = hard.reshape(dims[1], dims[2])

plt.imshow(hard.T, origin="lower")
plt.show()
