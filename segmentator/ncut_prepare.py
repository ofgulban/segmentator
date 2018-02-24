#!/usr/bin/env python
"""Normalized graph cuts for segmentator (experimental).

TODO: Replacing the functionality using scikit-learn?
"""

import os
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from skimage.future import graph
from skimage.segmentation import slic
import segmentator.config as cfg


def norm_grap_cut(image, max_edge=10000000, max_rec=4, compactness=2,
                  nrSupPix=2000):
    """Normalized graph cut wrapper for 2D numpy arrays.

    Parameters
    ----------
        image: np.ndarray (2D)
            Volume histogram.
        max_edge: float
            The maximum possible value of an edge in the RAG. This corresponds
            to an edge between identical regions. This is used to put self
            edges in the RAG.
        compactness: float
            From skimage slic_superpixels.py slic function:
            Balances color proximity and space proximity. Higher values give
            more weight to space proximity, making superpixel shapes more
            square/cubic. This parameter depends strongly on image contrast and
            on the shapes of objects in the image.
        nrSupPix: int, positive
            The (approximate) number of superpixels in the region adjacency
            graph.

    Returns
    -------
        labels2, labels1: np.ndarray (2D)
            Segmented volume histogram mask image. Each label has a unique
            identifier.

    """
    # scale for uint8 conversion
    image = np.round(255 / image.max() * image)
    image = image.astype('uint8')

    # scikit implementation expects rgb format (shape: NxMx3)
    image = np.tile(image, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0))

    labels1 = slic(image, compactness=compactness, n_segments=nrSupPix,
                   sigma=2)
    # region adjacency graph (rag)
    g = graph.rag_mean_color(img, labels1, mode='similarity_and_proximity')
    labels2 = graph.cut_normalized(labels1, g, max_edge=max_edge,
                                   num_cuts=1000, max_rec=max_rec)
    return labels2, labels1


path = cfg.filename
basename = path.split(os.extsep, 1)[0]

# load data
img = np.load(path)
# take logarithm of every count to make it similar to what is seen in gui
img = np.log10(img+1.)

# truncate very high values
img_max = cfg.cbar_init
img[img > img_max] = img_max

max_recursion = cfg.max_rec
ncut = np.zeros((img.shape[0], img.shape[1], max_recursion + 1))
for i in range(0, max_recursion + 1):
    msk, regions = norm_grap_cut(img, max_rec=i,
                                 nrSupPix=cfg.nr_sup_pix,
                                 compactness=cfg.compactness)
    ncut[:, :, i] = msk

# plots
if cfg.ncut_figs:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ax1.imshow(img.T, origin="lower", cmap=plt.cm.inferno)
    ax1.imshow(regions.T, origin="lower", cmap=plt.cm.inferno)
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

# save output
outName = '{}_ncut_sp{}_c{}'.format(basename, cfg.nr_sup_pix, cfg.compactness)
outName = outName.replace('.', 'pt')
np.save(outName, ncut)
print("    Saved as: {}{}".format(outName, '.npy'))
