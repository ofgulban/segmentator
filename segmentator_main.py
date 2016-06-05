#!/usr/bin/env python

"""Segmentator main."""

# Part of the Segmentator library
# Copyright (C) 2016  Omer Faruk Gulban and Marian Schneider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from nibabel import load, save, Nifti1Image
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button, LassoSelector
from matplotlib import path
from sector_mask import sector_mask
from utils import Ima2VolHistMapping, VolHist2ImaMapping

"""Load Data"""
#
nii = load('/home/faruk/Data/T1.nii.gz')

#
"""Data Processing"""
orig = np.squeeze(nii.get_data())

# auto-scaling for faster interface (0-500 or 600 seems fine)
percDataMin = np.percentile(orig, 0.01)
orig[np.where(orig < percDataMin)] = percDataMin
orig = orig - orig.min()
dataMin = orig.min()
percDataMax = np.percentile(orig, 99.9)
orig[np.where(orig > percDataMax)] = percDataMax
orig = 500./orig.max() * orig
percDataMax = orig.max()

# gradient magnitude (using L2 norm of the vector)
ima = orig.copy()
gra = np.gradient(ima)
gra = np.sqrt(np.power(gra[0], 2) + np.power(gra[1], 2) + np.power(gra[2], 2))

# reshape ima (more intuitive for voxel-wise operations)
ima = np.ndarray.flatten(ima)
gra = np.ndarray.flatten(gra)

#
"""Plots"""
# Set up a colormap:
palette = plt.cm.Reds
palette.set_over('r', 1.0)
palette.set_under('w', 0)
palette.set_bad('m', 1.0)

# Plot 2D histogram
fig = plt.figure()
ax = fig.add_subplot(121)
nrBins = int(percDataMax - dataMin + 1)
binVals = np.arange(dataMin, percDataMax)
_, xedges, yedges, _ = plt.hist2d(ima, gra,
                                  bins=binVals,
                                  norm=LogNorm(vmax=10000),
                                  cmap='Greys'
                                  )
ax.set_xlim(dataMin, percDataMax)
ax.set_ylim(0, percDataMax)
plt.subplots_adjust(bottom=0.40)
plt.colorbar()
plt.xlabel("Intensity f(x)")
plt.ylabel("Gradient Magnitude f'(x)")
plt.title("2D Histogram")

# plot 3D ima by default
ax2 = fig.add_subplot(122)
slc = ax2.imshow(orig[:, :, int(orig.shape[2]/2)],
                 cmap=plt.cm.gray, vmin=ima.min(), vmax=ima.max(),
                 interpolation='none'
                 )
imaMask = np.ones(orig.shape[0:2])  # TODO: Magic numbers
ovl = ax2.imshow(imaMask,
                 cmap=palette, vmin=0.1,
                 interpolation='none',
                 alpha=0.5
                 )
# plt.subplots_adjust(left=0.25, bottom=0.25)
plt.axis('off')

#
"""Update Functions"""
# Default circle parameters
volHistMask = sector_mask((nrBins, nrBins), (0, 0), 300, (0, 360))

circ = ax.imshow(volHistMask, cmap=palette, alpha=0.2, vmin=0.1,
                 interpolation='nearest',
                 origin='lower',
                 extent=[dataMin, percDataMax, gra.min(), percDataMax])

# Histogram to volume, volume to histogram mappings
ima2volHistMap = Ima2VolHistMapping(xinput=ima, yinput=gra, binsArray=binVals)
invHistVolume = np.reshape(ima2volHistMap, orig.shape)


def update(val):
    global imaMask, volHistMask
    sliceNr = int(sSliceNr.val*orig.shape[2])

    # Scale slider value for the log colorbar
    histVMax = np.power(10, sHistC.val)
    plt.clim(vmax=histVMax)

    # 2D mask is for fast visualization
    imaMask = VolHist2ImaMapping(invHistVolume[:, :, sliceNr], volHistMask)
    ovl.set_data(imaMask)
    fig.canvas.draw_idle()  # TODO:How to do properly? (massive speed up)


def updateSectorMask(val):
    global volHistMask
    # Set circle parameters
    volHistMask = sector_mask((nrBins, nrBins),
                              (sCircI.val, sCircJ.val), sCircR.val,
                              (sThetaMin.val, sThetaMax.val)
                              )
    circ.set_data(volHistMask)


def updateDataBrowser(val):
    global imaMask
    # Scale slider value [0,1) to dimension index to allow variation in shape
    sliceNr = int(sSliceNr.val*orig.shape[2])
    slc.set_data(orig[:, :, sliceNr])
    slc.set_extent((0, orig.shape[1]-1, orig.shape[0]-1, 0))
    # 2D mask is for fast visualization
    imaMask = VolHist2ImaMapping(invHistVolume[:, :, sliceNr], volHistMask)
    ovl.set_data(imaMask)
    ovl.set_extent((0, imaMask.shape[1]-1, imaMask.shape[0]-1, 0))
    fig.canvas.draw_idle()

#
"""Sliders"""
# colorbar slider
axcolor = 'lightgoldenrodyellow'
axHistC = plt.axes([0.15, 0.25, 0.25, 0.025], axisbg=axcolor)
sHistC = Slider(axHistC, 'Colorbar', 1, 5, valinit=3, valfmt='%0.1f')

# circle sliders
axCircI = plt.axes([0.15, 0.20, 0.25, 0.025], axisbg=axcolor)
sCircI = Slider(axCircI, 'Ci', -nrBins/5, nrBins, valinit=0, valfmt='%0.2f')
axCircJ = plt.axes([0.15, 0.15, 0.25, 0.025], axisbg=axcolor)
sCircJ = Slider(axCircJ, 'Cj', -nrBins/5, nrBins, valinit=0, valfmt='%0.2f')
axCircR = plt.axes([0.15, 0.10, 0.25, 0.025], axisbg=axcolor)
sCircR = Slider(axCircR, 'Cr', 0, nrBins, valinit=100, valfmt='%0.2f')

aThetaMin = plt.axes([0.15, 0.06, 0.25, 0.02], axisbg=axcolor)
sThetaMin = Slider(aThetaMin, 'Theta Min', 0, 360, valinit=0, valfmt='%0.1f')
aThetaMax = plt.axes([0.15, 0.02, 0.25, 0.02], axisbg=axcolor)
sThetaMax = Slider(aThetaMax, 'Theta Max', 0, 360, valinit=360, valfmt='%0.1f')

# ima browser slider
axSliceNr = plt.axes([0.6, 0.25, 0.25, 0.025], axisbg=axcolor)
sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999, valinit=0.5, valfmt='%0.3f')

#
"""Buttons"""
# cycle button
cycleax = plt.axes([0.6, 0.125, 0.075, 0.075])
bCycle = Button(cycleax, 'Cycle\nView',
                color=axcolor, hovercolor='0.975')
cycleCount = 0


def cycleView(event):
    global orig, imaMask, cycleCount, invHistVolume
    cycleCount = (cycleCount+1) % 3
    orig = np.transpose(orig, (2, 0, 1))
    invHistVolume = np.transpose(invHistVolume, (2, 0, 1))

# export button
aExport = plt.axes([0.75, 0.025, 0.075, 0.075])
bExport = Button(aExport, 'Export\nNifti', color=axcolor, hovercolor='0.975')


def exportNifti(event):
    global volHistMask, nrBins, orig, invHistVolume, cycleCount

    # put the permuted indices back to their original format
    cycBackPerm = (cycleCount, (cycleCount+1) % 3, (cycleCount+2) % 3)
    orig = np.transpose(orig, cycBackPerm)
    invHistVolume = np.transpose(invHistVolume, cycBackPerm)
    cycleCount = 0

    linIdx = np.arange(0, nrBins*nrBins)
    idxMask = linIdx[volHistMask.flatten()]

    # return logical array with length equal to nr of voxels
    voxMask = np.in1d(invHistVolume.flatten(), idxMask)
    # reset imaMask and apply logical indexing
    mask3D = np.zeros(invHistVolume.flatten().shape)
    mask3D[voxMask] = 1
    mask3D = mask3D.reshape(invHistVolume.shape)

    new_image = Nifti1Image(mask3D, header=nii.get_header(),
                            affine=nii.get_affine()
                            )
    save(new_image, nii.get_filename()[:-4]+'_OUT.nii.gz')

# reset button
aReset = plt.axes([0.6, 0.025, 0.075, 0.075])
bReset = Button(aReset, 'Reset', color=axcolor, hovercolor='0.975')


def resetMask(event):
    sCircI.reset()
    sCircJ.reset()
    sCircR.reset()
    sThetaMin.reset()
    sThetaMax.reset()

#
"""Updates"""
sSliceNr.on_changed(updateDataBrowser)
sHistC.on_changed(update)
sCircI.on_changed(updateSectorMask), sCircI.on_changed(update)
sCircJ.on_changed(updateSectorMask), sCircJ.on_changed(update)
sCircR.on_changed(updateSectorMask), sCircR.on_changed(update)
sThetaMin.on_changed(updateSectorMask), sThetaMin.on_changed(update)
sThetaMax.on_changed(updateSectorMask), sThetaMax.on_changed(update)
bCycle.on_clicked(cycleView)
bCycle.on_clicked(update)
bCycle.on_clicked(updateDataBrowser)
bExport.on_clicked(exportNifti)
bReset.on_clicked(resetMask)

#
"""New stuff: Lasso (Experimental)"""
# Pixel coordinates
pixCoords = np.arange(nrBins)
xv, yv = np.meshgrid(pixCoords, pixCoords)
pixCoords = np.vstack((xv.flatten(), yv.flatten())).T


def updateArray(array, indices):
    lin = np.arange(array.size)
    newArray = array.flatten()
    newArray[lin[indices]] = 1
    return newArray.reshape(array.shape)


def onselect(verts):
    global volHistMask, pixCoords
    p = path.Path(verts)
    ind = p.contains_points(pixCoords, radius=1.5)
    volHistMask = updateArray(volHistMask, ind)
    circ.set_array(volHistMask)

    # update 3D volume
    sliceNr = int(sSliceNr.val*orig.shape[2])
    imaMask = VolHist2ImaMapping(invHistVolume[:, :, sliceNr], volHistMask)
    ovl.set_data(imaMask)

    fig.canvas.draw_idle()

lasso = LassoSelector(ax, onselect)

plt.show()
plt.close()
