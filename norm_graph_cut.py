"""Normalized graph cuts for segmentator (experimental)."""

import os
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from skimage.future import graph
from skimage.morphology import square, closing
from skimage.segmentation import slic
from nibabel import save, Nifti1Image


def norm_grap_cut(image, closing_size=10, max_edge=100000000, max_rec=3):
    """
    Normalized graph cut wrapper for 2D numpy arrays.

    Arguments:
    -----------
        image: np.ndarray (2D)
            Volume histogram.
        closing_size: positive integer
            determines dilation size closing operation.
        max_edge: float, optional
            The maximum possible value of an edge in the RAG. This corresponds
            to an edge between identical regions. This is used to put self
            edges in the RAG.

    Returns:
    -----------
        sector_mask: np.ndarray (2D)
            Segmented volume histogram mask image. Each label has a unique
            identifier.
    """
    # truncate very high values to gain precision later in uint8 conversion
    perc = np.percentile(image, 99.9)
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

    # parameters might be optimized for ~500x500 volume histograms
    labels1 = slic(image, compactness=2, n_segments=2500)
    # region adjacency graph (rag)
    g = graph.rag_mean_color(img, labels1, mode='similarity_and_proximity')
    labels2 = graph.cut_normalized(labels1, g, max_edge=max_edge, num_cuts=500,
                                   max_rec=max_rec)
    return labels2

path = '/home/faruk/Data/T1_volHist.npy'
basename = path.split(os.extsep, 1)[0]

img = np.load(path)
img = np.log10(img+1)

max_recursion = 7
marian = np.zeros((img.shape[0], img.shape[1], max_recursion + 1))
for i in range(0, max_recursion + 1):
    msk = norm_grap_cut(img, max_rec=i)
    marian[:, :, i] = msk


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
    idx = idx % marian.shape[2]
    tmp = np.copy(marian[:, :, idx])
    im.set_array(tmp.T)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=750, blit=True)
plt.show()

# new_image = Nifti1Image(marian[:, :, :], affine=np.eye(4, 4))
# save(new_image, basename+'_ncut')

np.save(basename+'_ncut', marian)
