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
from nibabel import load, save, Nifti1Image
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button, LassoSelector
from matplotlib import path
from sector_mask import sector_mask
from utils import Ima2VolHistMapping, VolHist2ImaMapping
from draggable import DraggableSector

#%%
"""Load Data"""
#
nii = load('/media/sf_D_DRIVE/Segmentator/ExpNii/P06_T1w_divPD_IIHC_v16.nii')

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

#%%
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
nrBins = int(percDataMax - dataMin + 2)  # TODO: variable name fix
binVals = np.arange(dataMin, percDataMax)
_, xedges, yedges, _ = plt.hist2d(ima, gra,
                                  bins=binVals,
                                  norm=LogNorm(vmax=10000),
                                  cmap='Greys'
                                  )
ax.set_xlim(dataMin, percDataMax)
ax.set_ylim(0, percDataMax)
bottom = 0.30
plt.subplots_adjust(bottom=bottom)
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
imaMaskFigHand = ax2.imshow(imaMask,
                            cmap=palette, vmin=0.1,
                            interpolation='none',
                            alpha=0.5
                            )
#ax2.set_xlim(0, orig.shape[0])
#ax2.set_ylim(0, orig.shape[1])
#ax2.xticks(x
#ax2.yticks(x
plt.axis('off')

#%%
#
"""Initialisation"""
# create first instance of sector mask
shape = (nrBins, nrBins)
centre = (0, 0)
radius = 200
theta = (0, 360)
sectorObj = sector_mask(shape, centre, radius, theta)

# draw sector mask for the first time
volHistMaskFigHand, volHistMask = sectorObj.draw(
    ax, cmap='Reds', alpha=0.2, vmin=0.1, interpolation='nearest',
    origin='lower', extent=[percDataMin, percDataMax, gra.min(), percDataMax])

# pass on some properties to sector object
sectorObj.figure = ax.figure
sectorObj.axes = ax.axes
sectorObj.axes2 = ax2.axes
sectorObj.nrOfBins = len(binVals)
sectorObj.sliceNr = int(0.5*orig.shape[2])
sectorObj.imaMaskFigHand = imaMaskFigHand
sectorObj.volHistMaskFigHand = volHistMaskFigHand

# make sector draggable object, pass on properties
drSectorObj = DraggableSector(sectorObj)
drSectorObj.connect()
drSectorObj.volHistMask = volHistMask
drSectorObj.imaMask = imaMask
ima2volHistMap = Ima2VolHistMapping(xinput=ima, yinput=gra, binsArray=binVals)
drSectorObj.invHistVolume = np.reshape(ima2volHistMap, orig.shape)


#%%
"""Sliders"""
# colorbar slider
axcolor = 'lightgoldenrodyellow'
axHistC = plt.axes([0.15, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
sHistC = Slider(axHistC, 'Colorbar', 1, 5, valinit=3, valfmt='%0.1f')


def updateColorBar(val):
    # update slider for scaling log colorbar in 2D hist
    histVMax = np.power(10, sHistC.val)
    plt.clim(vmax=histVMax)

# ima browser slider
axSliceNr = plt.axes([0.6, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999, valinit=0.5, valfmt='%0.3f')


def updateImaBrowser(val):
    # Scale slider value [0,1) to dimension index to allow variation in shape
    drSectorObj.sector.sliceNr = int(sSliceNr.val*orig.shape[2])
    slc.set_data(orig[:, :, drSectorObj.sector.sliceNr])
    slc.set_extent((0, orig.shape[1]-1, orig.shape[0]-1, 0))

    # update imaMask
    drSectorObj.imaMask = VolHist2ImaMapping(
        drSectorObj.invHistVolume[:, :, drSectorObj.sector.sliceNr],
        drSectorObj.volHistMask)
    drSectorObj.sector.imaMaskFigHand.set_data(drSectorObj.imaMask)
    drSectorObj.sector.figure.canvas.draw()

# theta slider
aTheta = plt.axes([0.15, bottom-0.10, 0.25, 0.025], axisbg=axcolor)
thetaInit = 0
drSectorObj.thetaInit = thetaInit
sTheta = Slider(aTheta, 'Theta', 0, 359.99, valinit=thetaInit, valfmt='%0.1f')


def updateTheta(val):
    # get current theta value from slider
    thetaVal = sTheta.val
    # update mouth of sector mask
    diff = thetaVal-drSectorObj.thetaInit
    drSectorObj.sector.mouthChange(diff)
    # adjust thetaInit
    drSectorObj.thetaInit = thetaVal
    # update volHistMask
    drSectorObj.volHistMask = drSectorObj.sector.binaryMask()
    drSectorObj.sector.volHistMaskFigHand.set_data(drSectorObj.volHistMask)
    # update imaMask
    drSectorObj.imaMask = VolHist2ImaMapping(
        drSectorObj.invHistVolume[:, :, drSectorObj.sector.sliceNr],
        drSectorObj.volHistMask)
    drSectorObj.sector.imaMaskFigHand.set_data(drSectorObj.imaMask)
    # draw to canvas
    drSectorObj.sector.figure.canvas.draw()

#%%
"""Buttons"""
# cycle button
cycleax = plt.axes([0.6, bottom-0.275, 0.075, 0.075])
bCycle = Button(cycleax, 'Cycle\nView',
                color=axcolor, hovercolor='0.975')
cycleCount = 0

  
def cycleView(event):
    global orig, cycleCount
    cycleCount = (cycleCount+1) % 3
    # transpose data
    orig = np.transpose(orig, (2, 0, 1))
    # transpose ima2volHistMap
    drSectorObj.invHistVolume = np.transpose(
        drSectorObj.invHistVolume, (2, 0, 1))
    # plot new data
    slc.set_data(orig[:, :, drSectorObj.sector.sliceNr])
    slc.set_extent((0, orig.shape[1]-1, orig.shape[0]-1, 0))
    # update imaMask
    drSectorObj.imaMask = VolHist2ImaMapping(
        drSectorObj.invHistVolume[:, :, drSectorObj.sector.sliceNr],
        drSectorObj.volHistMask)
    # plot new imaMask
    drSectorObj.sector.imaMaskFigHand.set_data(drSectorObj.imaMask)
    drSectorObj.sector.imaMaskFigHand.set_extent(
        (0, drSectorObj.imaMask.shape[1]-1,
         drSectorObj.imaMask.shape[0]-1, 0))
    # draw to canvas
    drSectorObj.sector.figure.canvas.draw()


# export button
exportax = plt.axes([0.8, bottom-0.275, 0.075, 0.075])
bExport = Button(exportax, 'Export\nNifti',
                 color=axcolor, hovercolor='0.975')


def exportNifti(event):
    global cycleCount, orig
    # put the permuted indices back to their original format
    cycBackPerm = (cycleCount, (cycleCount+1) % 3, (cycleCount+2) % 3)
    orig = np.transpose(orig, cycBackPerm)
    drSectorObj.invHistVolume = np.transpose(drSectorObj.invHistVolume,
                                             cycBackPerm)
    cycleCount = 0
    # get linear indices
    linIndices = np.arange(0, nrBins*nrBins)
    idxMask = linIndices[drSectorObj.volHistMask.flatten()]
    # return logical array with length equal to nr of voxels
    voxMask = np.in1d(drSectorObj.invHistVolume.flatten(), idxMask)
    # reset mask and apply logical indexing
    mask3D = np.zeros(drSectorObj.invHistVolume.flatten().shape)
    mask3D[voxMask] = 1
    mask3D = mask3D.reshape(drSectorObj.invHistVolume.shape)
    # save image, check whether nii or nii.gz
    new_image = Nifti1Image(
        mask3D, header=nii.get_header(), affine=nii.get_affine())
    if nii.get_filename()[-4:] == '.nii':
        save(new_image, nii.get_filename()[:-4]+'_OUT.nii.gz')
    elif nii.get_filename()[-7:] == '.nii.gz':
        save(new_image, nii.get_filename()[:-7]+'_OUT.nii.gz')

# reset button
resetax = plt.axes([0.7, bottom-0.275, 0.075, 0.075])
bReset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def resetGlobal(event):
    # reset color bar
    sHistC.reset()
    # reset ima browser slider
    sSliceNr.reset()
    # reset slice number
    drSectorObj.sector.sliceNr = int(sSliceNr.val*orig.shape[2])
    slc.set_data(orig[:, :, drSectorObj.sector.sliceNr])
    slc.set_extent((0, orig.shape[1]-1, orig.shape[0]-1, 0))
    # reset theta
    drSectorObj.thetaInit = thetaInit
    sTheta.reset()
    # reset values for mask
    drSectorObj.sector.set_x(centre[0])
    drSectorObj.sector.set_y(centre[1])
    drSectorObj.sector.set_r(radius)
    drSectorObj.sector.tmin, drSectorObj.sector.tmax = np.deg2rad(theta)
    # update volHistMask
    drSectorObj.volHistMask = drSectorObj.sector.binaryMask()
    drSectorObj.sector.volHistMaskFigHand.set_data(drSectorObj.volHistMask)
    # update imaMask
    drSectorObj.imaMask = VolHist2ImaMapping(
        drSectorObj.invHistVolume[:, :, drSectorObj.sector.sliceNr],
        drSectorObj.volHistMask)
    drSectorObj.sector.imaMaskFigHand.set_data(drSectorObj.imaMask)


#%%
"""Updates"""
sHistC.on_changed(updateColorBar)
sSliceNr.on_changed(updateImaBrowser)
sTheta.on_changed(updateTheta)
bCycle.on_clicked(cycleView)
bExport.on_clicked(exportNifti)
bReset.on_clicked(resetGlobal)

#%%
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
        drSectorObj.disconnect()
        # enable lasso
        lasso = LassoSelector(ax, onselect)
    elif switchStatus == 1:
        OnSelectCounter = 0
        lasso = []
        # enable drag function of sector mask
        drSectorObj.connect()

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
        drSectorObj.volHistMask = drSectorObj.sector.binaryMask()
    OnSelectCounter += 1
    drSectorObj.volHistMask = updateArray(drSectorObj.volHistMask, ind)
    drSectorObj.sector.volHistMaskFigHand.set_data(drSectorObj.volHistMask)
    # update imaMask
    drSectorObj.imaMask = VolHist2ImaMapping(
        drSectorObj.invHistVolume[:, :, drSectorObj.sector.sliceNr],
        drSectorObj.volHistMask)
    drSectorObj.sector.imaMaskFigHand.set_data(drSectorObj.imaMask)
    fig.canvas.draw_idle()

bLasso.on_clicked(lassoSwitch)


plt.show()
