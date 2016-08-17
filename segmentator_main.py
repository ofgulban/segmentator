#!/usr/bin/env python

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
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button, LassoSelector
from matplotlib import path
from nibabel import load
from segmentator_functions import MakeFigureInteractive
from sector_mask import sector_mask
from utils import Ima2VolHistMapping, VolHist2ImaMapping
import config as cfg

#
"""Load Data"""
import openSegmentator


#

nii = load(openSegmentator.test.path)

#
"""Data Processing"""
orig = np.squeeze(nii.get_data())
# truncate too low and too high values
percDataMin = np.percentile(orig, 0.01)
orig[orig < percDataMin] = percDataMin
percDataMax = np.percentile(orig, 99.9)
orig[orig > percDataMax] = percDataMax
# auto-scaling for faster interface (0-500 or 600 seems fine)
scaleFactor = 500
orig = orig - orig.min()
orig = scaleFactor/orig.max() * orig
# define dataMin and dataMax for later use
dataMin = np.round(orig.min())
dataMax = np.round(orig.max())

# copy intensity data so we can flatten the copy and leave original intact
ima = orig.copy()
# calculate gradient magnitude (using L2 norm of the vector)
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
# Plot 2D histogram
fig = plt.figure()
ax = fig.add_subplot(121)
binEdges = np.arange(dataMin, dataMax+1)
nrBins = len(binEdges)-1
counts, xedges, yedges, volHistH = plt.hist2d(ima, gra,
                                              bins=binEdges,
                                              cmap='Greys'
                                              )

ax.set_xlim(dataMin, dataMax)
ax.set_ylim(0, dataMax)
ax.set_xlabel("Intensity f(x)")
ax.set_ylabel("Gradient Magnitude f'(x)")
ax.set_title("2D Histogram")

# plot colorbar for 2d hist
volHistH.set_norm(LogNorm(vmax=1000))
plt.colorbar(volHistH)

# plot 3D ima by default
ax2 = fig.add_subplot(122)
slc = ax2.imshow(orig[:, :, int(orig.shape[2]/2)],
                 cmap=plt.cm.gray, vmin=ima.min(), vmax=ima.max(),
                 interpolation='none', extent=[0, orig.shape[1],
                                               orig.shape[0], 0]
                 )
imaMask = np.ones(orig.shape[0:2])  # TODO: Magic numbers
imaMaskHandle = ax2.imshow(imaMask, cmap=palette, vmin=0.1,
                           interpolation='none', alpha=0.5,
                           extent=[0, orig.shape[1], orig.shape[0], 0])
# adjust subplots on figure
bottom = 0.30
fig.subplots_adjust(bottom=bottom)
plt.axis('off')


#
"""Initialisation"""
# create first instance of sector mask
sectorObj = sector_mask((nrBins, nrBins), cfg.init_centre,
                        cfg.init_radius, cfg.init_theta)

# draw sector mask for the first time
volHistMaskHandle, volHistMask = sectorObj.draw(
    ax, cmap='Reds', alpha=0.2, vmin=0.1, interpolation='nearest',
    origin='lower', extent=[dataMin, percDataMax, gra.min(), percDataMax])


# initiate a flexible figure object, pass to it usefull properties
flexFig = MakeFigureInteractive(figure=ax.figure, axes=ax.axes, axes2=ax2.axes,
                                orig=orig,
                                nii=nii,
                                sectorObj=sectorObj,
                                nrOfBins=nrBins,
                                sliceNr=int(0.5*orig.shape[2]),
                                slc=slc,
                                imaMask=imaMask,
                                imaMaskHandle=imaMaskHandle,
                                volHistMask=volHistMask,
                                volHistMaskHandle=volHistMaskHandle,
                                contains=volHistMaskHandle.contains)

# make the figure responsive to clicks
flexFig.connect()
ima2volHistMap = Ima2VolHistMapping(xinput=ima, yinput=gra, binsArray=binEdges)
flexFig.invHistVolume = np.reshape(ima2volHistMap, orig.shape)

#
"""Sliders and Buttons"""
# colorbar slider
axcolor = 'lightgoldenrodyellow'
axHistC = plt.axes([0.15, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
flexFig.sHistC = Slider(axHistC, 'Colorbar', 1, 5, valinit=3, valfmt='%0.1f')

# ima browser slider
axSliceNr = plt.axes([0.6, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
flexFig.sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999, valinit=0.5,
                          valfmt='%0.3f')

# theta slider
aTheta = plt.axes([0.15, bottom-0.10, 0.25, 0.025], axisbg=axcolor)
flexFig.thetaInit = 0
flexFig.sTheta = Slider(aTheta, 'Theta', 0, 359.99,
                        valinit=flexFig.thetaInit, valfmt='%0.1f')

# cycle button
cycleax = plt.axes([0.6, bottom-0.275, 0.075, 0.075])
flexFig.bCycle = Button(cycleax, 'Cycle\nView',
                        color=axcolor, hovercolor='0.975')
flexFig.cycleCount = 0

# export button
exportax = plt.axes([0.8, bottom-0.275, 0.075, 0.075])
flexFig.bExport = Button(exportax, 'Export\nNifti',
                         color=axcolor, hovercolor='0.975')

# reset button
resetax = plt.axes([0.7, bottom-0.275, 0.075, 0.075])
flexFig.bReset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

#
"""Updates"""
flexFig.sHistC.on_changed(flexFig.updateColorBar)
flexFig.sSliceNr.on_changed(flexFig.updateImaBrowser)
flexFig.sTheta.on_changed(flexFig.updateTheta)
flexFig.bCycle.on_clicked(flexFig.cycleView)
flexFig.bExport.on_clicked(flexFig.exportNifti)
flexFig.bReset.on_clicked(flexFig.resetGlobal)

#
"""New stuff: Lasso (Experimental)"""
# Lasso button
lassoax = plt.axes([0.15, bottom-0.275, 0.075, 0.075])
bLasso = Button(lassoax, 'Lasso\nON OFF', color=axcolor, hovercolor='0.975')


# define switch for Lasso option
switchCounter = 1
def lassoSwitch(event):
    global lasso, switchCounter, OnSelectCounter
    lasso = []
    switchCounter += 1
    switchStatus = switchCounter % 2
    if switchStatus == 0:
        # disable drag function of sector mask
        flexFig.disconnect()
        # enable lasso
        lasso = LassoSelector(ax, onselect)
    elif switchStatus == 1:
        OnSelectCounter = 0
        lasso = []
        # enable drag function of sector mask
        flexFig.connect()

# Pixel coordinates
pix = np.arange(nrBins)
xv, yv = np.meshgrid(pix, pix)
pix = np.vstack((xv.flatten(), yv.flatten())).T


def updateArray(array, indices):
    lin = np.arange(array.size)
    newArray = array.flatten()
    newArray[lin[indices]] = 1
    return newArray.reshape(array.shape)


OnSelectCounter = 0
def onselect(verts):
    global pix, OnSelectCounter
    p = path.Path(verts)
    ind = p.contains_points(pix, radius=1.5)
    # update volHistMask
    if OnSelectCounter == 0:
        flexFig.volHistMask = flexFig.sectorObj.binaryMask()
    OnSelectCounter += 1
    flexFig.volHistMask = updateArray(flexFig.volHistMask, ind)
    flexFig.volHistMaskHandle.set_data(flexFig.volHistMask)
    # update imaMask
    flexFig.imaMask = VolHist2ImaMapping(
        flexFig.invHistVolume[:, :, flexFig.sliceNr],
        flexFig.volHistMask)
    flexFig.imaMaskHandle.set_data(flexFig.imaMask)
    fig.canvas.draw_idle()

bLasso.on_clicked(lassoSwitch)


plt.show()
