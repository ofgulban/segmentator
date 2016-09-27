"""Normalized graph cuts for segmentator (experimental)."""

import ncut_prepare
import os
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.future import graph
from skimage.morphology import square, closing
from skimage.segmentation import slic


def norm_grap_cut(image, closing_size=5, max_edge=10000000, max_rec=3,
                  nrReg=2500):
    """
    Normalized graph cut wrapper for 2D numpy arrays.

    Arguments:
    -----------
        image: np.ndarray (2D)
            Volume histogram.
        closing_size: int, positive
            determines dilation size closing operation.
        max_edge: float, optional
            The maximum possible value of an edge in the RAG. This corresponds
            to an edge between identical regions. This is used to put self
            edges in the RAG.
        nrReg: int, positive
            The number or nodes in the region adjacency graph. I think these
            merged pixels can also be called superpixels.

    Returns:
    -----------
        sector_mask: np.ndarray (2D)
            Segmented volume histogram mask image. Each label has a unique
            identifier.
    """
    # truncate very high values to gain precision later in uint8 conversion
    perc = np.percentile(image, 99.75)
    image[image > perc] = perc

    # scale for uint8 conversion
    image = np.round(255 / image.max() * image)
    image = image.astype('uint8')

    # dilate and erode (closing) to fill in white dots in grays (arbitrary)
    image = closing(image, square(closing_size))

    # a bit of blurring to help getting smoother edges (arbitrary)
    image = gaussian(image, 2)

    # scikit implementation expects rgb format (shape: NxMx3)
    image = np.tile(image, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0))

    labels1 = slic(image, compactness=2, n_segments=nrReg)
    # region adjacency graph (rag)
    g = graph.rag_mean_color(img, labels1, mode='similarity_and_proximity')
    labels2 = graph.cut_normalized(labels1, g, max_edge=max_edge,
                                   num_cuts=5000, max_rec=max_rec)
    return labels2

path = ncut_prepare.args.filename
basename = path.split(os.extsep, 1)[0]

img = np.load(path)
img = np.log10(img+1)

max_recursion = ncut_prepare.args.maxRec
nr_regions = ncut_prepare.args.nrReg
ncut = np.zeros((img.shape[0], img.shape[1], max_recursion + 1))
for i in range(0, max_recursion + 1):
    msk = norm_grap_cut(img, max_rec=i, nrReg=nr_regions)
    ncut[:, :, i] = msk


# plots
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(img.T, origin="lower", cmap=plt.cm.inferno)
ax2.imshow(msk.T, origin="lower", cmap=plt.cm.nipy_spectral)

ax1.set_title('Source')
ax2.set_title('Ncut')

plt.show()

fig = plt.figure()
unq = np.unique(msk)
idx = -1

im = plt.imshow(msk.T, origin="lower", cmap=plt.cm.flag,
                animated=True)


def updatefig(*args):
    """Animate the plot."""
    global unq, msk, idx, tmp
    idx += 1
    idx = idx % ncut.shape[2]
    tmp = np.copy(ncut[:, :, idx])
    im.set_array(tmp.T)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=750, blit=True)
plt.show()

np.save(basename+'_ncut', ncut)
