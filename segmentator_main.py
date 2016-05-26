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


import matplotlib
matplotlib.use('TKAgg')
from nibabel import load, save, Nifti1Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button, LassoSelector
from matplotlib import path
from sector_mask import sector_mask
from utils import Vox2PixMapping, pix2vox_hist2D
from draggable import DraggableSector

## different masks - inconsistent naming: change mask to sliceMask?
# mask := mask the voxels on a slice of brain (right side, 3D data)
# pixMask := mask the pixels in the 2D histogram (left side)


#%%
"""Load Data"""
#/media/marian/DATADRIVE1/MotionQuartet/Analysis/P3/Anatomy_BBR/P03_T1_divPD_IIHC_v16_s1p2.nii.gz
img = load('/media/sf_D_DRIVE/MotionQuartet/Analysis/P7/Anatomy_BBR/P07_T1w_divPD_IIHC.nii')

#%%
"""Data Processing"""
data = np.squeeze( img.get_data() )

# gradient magnitude (using L2 norm of the vector)
gra = np.gradient(data)
gra = np.sqrt( np.power(gra[0],2) + np.power(gra[1],2) + np.power(gra[2],2) )

# reshape data (more intuitive for voxel-wise operations)
data = np.ndarray.flatten(data)
gra = np.ndarray.flatten(gra)

# to crop rare values (TODO: Parametrize this part)
percDataMax = np.percentile(data, 99.9)
percDataMin = 0
data[np.where(data>percDataMax)] = percDataMax
data[np.where(data<0)] = 0
gra[np.where(gra>percDataMax)] = percDataMax  # TODO: Can be bound to gra itself

#%%
"""Plots"""
# plot volume histogram (left side)
fig = plt.figure()
ax = fig.add_subplot(121)
nrBins = int(percDataMax - percDataMin + 2 )  #TODO: variable name fix
binVals = np.arange(percDataMin, percDataMax)
_, xedges, yedges, _ = plt.hist2d(data, gra,
                                  bins=binVals,
                                  norm=LogNorm(vmax=10000),
                                  cmap='Greys'
                                  )
# label and set axes                                      
ax.set_xlim(percDataMin, percDataMax)
ax.set_ylim(0, percDataMax)
bottom = 0.30
plt.subplots_adjust(bottom=bottom)
plt.colorbar()
plt.xlabel("Intensity f(x)")
plt.ylabel("Gradient Magnitude f'(x)")
plt.title("2D Histogram")


# plot image browser (right side)
ax2 = fig.add_subplot(122)
# plot brain data
orig = img.get_data()
slc = ax2.imshow(orig[:,:,int(orig.shape[2]/2)],
                 cmap=plt.cm.gray, vmin=0, vmax=500,
                 interpolation='none'
                 )
# plot brain mask on top
mask = np.ones(img.shape[0:2]) # TODO: Magic numbers
ovl = ax2.imshow(mask,
                 cmap=plt.cm.Reds, vmin=0.1,
                 interpolation='none',
                 alpha = 0.5
                 )
# plt.subplots_adjust(left=0.25, bottom=0.25)
plt.axis('off')

#%%
"""Functions and Init"""
# define a volume (vox) to histogram (pix) map
vox2PixMap = Vox2PixMapping(xinput=data, yinput=gra, binsArray=binVals)
invHistVolume = np.reshape(vox2PixMap, img.shape)

# initialise scliceNr
sliceNr = int(0.5*orig.shape[2])

# create first instance of sector mask
shape = (nrBins,nrBins)
centre = (0,0)
radius = 300
theta = (0,360)
sectorObj = sector_mask(shape, centre, radius, theta)

# draw sector mask for the first time
sectorFig, pixMask = sectorObj.draw(ax, cmap='Reds', alpha=0.2, vmin=0.1,
                 interpolation='nearest',
                 origin='lower',
                 extent=[percDataMin, percDataMax, gra.min(), percDataMax])

# pass on some necessary evil
sectorObj.figure = ax.figure
sectorObj.axes = ax.axes
sectorObj.axes2 = ax2.axes
sectorObj.invHistVolume = invHistVolume
sectorObj.brainMaskFigHandle = ovl
sectorObj.sliceNr = sliceNr
sectorObj.nrOfBins = len(binVals)

# make sector draggable                 
drSectorObj = DraggableSector(sectorObj)
drSectorObj.pixMask = pixMask
drSectorObj.connect() 

# define what should happen if update is called
def updateHistBrowser(val): 
    # update slider for scaling log colorbar in 2D hist
    histVMax = np.power(10, sHistC.val)
    plt.clim(vmax=histVMax)

def updateBrainBrowser(val):
    global sliceNr
    # Scale slider value [0,1) to dimension index to allow variation in shape
    sliceNr = int(sSliceNr.val*orig.shape[2])
    slc.set_data(orig[:,:,sliceNr])
    # update slice number for draggable sector mask
    drSectorObj.sector.sliceNr = sliceNr
    # get current pixMask
    pixMask = drSectorObj.pixMask
    # update the mask (2D mask is for fast visualization)
    mask = pix2vox_hist2D(invHistVolume[:,:,sliceNr],pixMask)
    ovl.set_data(mask)
    fig.canvas.draw_idle() # TODO:How to do properly? (massive speed up)

#%%
"""Sliders"""
# colorbar slider
axcolor = 'lightgoldenrodyellow'
axHistC = plt.axes([0.15, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
sHistC = Slider(axHistC, 'Colorbar', 1, 5, valinit=3, valfmt='%0.1f')

# brain browser slider
axSliceNr = plt.axes([0.6, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
sSliceNr  = Slider(axSliceNr, 'Slice', 0, 0.999, valinit=0.5, valfmt='%0.3f')

#%%
"""Buttons"""
#%% cycle button
cycleax = plt.axes([0.6, bottom-0.275, 0.075, 0.075])
bCycle  = Button(cycleax, 'Cycle\nView', color=axcolor, hovercolor='0.975')

# change view (TODO: Not working properly for now)
def cycleView(event):
    global orig, mask
    orig = np.transpose(orig, (2,0,1))
    slc.set_extent((0, orig.shape[1]-1, 0, orig.shape[0]-1))
    mask = np.transpose(mask, (2,0,1))
    ovl.set_extent((0, mask.shape[1]-1, 0, mask.shape[0]-1))

#%% export button
exportax = plt.axes([0.8, bottom-0.275, 0.075, 0.075])
bExport  = Button(exportax, 'Export\nNifti', color=axcolor, hovercolor='0.975')

def exportNifti(event):
    linIndices = np.arange(0, nrBins*nrBins)
    # get current pixMask
    pixMask = drSectorObj.pixMask
    # get linear indices
    idxMask = linIndices[pixMask.flatten()]
    # return logical array with length equal to nr of voxels
    voxMask = np.in1d(drSectorObj.sector.invHistVolume.flatten(), idxMask)
    # reset mask and apply logical indexing
    mask3D = np.zeros(drSectorObj.sector.invHistVolume.flatten().shape)
    mask3D[voxMask] = 1
    mask3D = mask3D.reshape(drSectorObj.sector.invHistVolume.shape)
    # save image, check whether nii or nii.gz
    new_image = Nifti1Image(mask3D, header=img.get_header() ,affine=img.get_affine())
    if img.get_filename()[-4:] == '.nii':
        save(new_image, img.get_filename()[:-4]+'_OUT.nii.gz')
    elif img.get_filename()[-7:] == '.nii.gz':
        save(new_image, img.get_filename()[:-7]+'_OUT.nii.gz')
        
        
#%% reset button
resetax = plt.axes([0.7, bottom-0.275, 0.075, 0.075])
bReset  = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def resetMask(event):
    global sliceNr
    # reset brain browser slider
    sSliceNr.reset()
    # Scale slider value [0,1) to dimension index to allow variation in shape
    sliceNr = int(sSliceNr.val*orig.shape[2])
    slc.set_data(orig[:,:,sliceNr])
    # update slice number for draggable sector mask
    drSectorObj.sector.sliceNr = sliceNr
    # revert to initial sector mask paramters 
    drSectorObj.sector.update(shape, centre, radius, theta)
    # update pix mask (histogram)  
    pixMask = drSectorObj.sector.binaryMask()
    drSectorObj.sector.FigObj.set_data(pixMask)
    # update brain mask
    mask = pix2vox_hist2D(
        drSectorObj.sector.invHistVolume[:,:,sliceNr],
        pixMask)
    drSectorObj.sector.brainMaskFigHandle.set_data(mask)   
             

#%%
"""Updates"""
sSliceNr.on_changed(updateBrainBrowser)
sHistC.on_changed(updateHistBrowser)
bCycle.on_clicked(cycleView)  #TODO: must be reworked to fit with 2Dmask
bExport.on_clicked(exportNifti)
bReset.on_clicked(resetMask)

#%%
"""New stuff: Lasso (Experimental)"""
# Lasso button
lassoax = plt.axes([0.15, bottom-0.275, 0.1, 0.075])
bLasso  = Button(lassoax, 'Lasso\nON OFF', color=axcolor, hovercolor='0.975')

# define switch of Lasso option
switchCounter = 1 # TODO: use modulus 
def lassoSwitch(event):
    global lasso, switchCounter, OnSelectCounter
    lasso = []
    switchCounter += 1
    switchStatus = switchCounter%2
    print switchStatus
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
xv, yv = np.meshgrid(pix,pix)  
pix = np.vstack( (xv.flatten(), yv.flatten()) ).T

def updateArray(array, indices):
    lin = np.arange(array.size)
    newArray = array.flatten()
    newArray[lin[indices]] = 1
    return newArray.reshape(array.shape)

OnSelectCounter = 0
def onselect(verts):
    global pixMask, pix, OnSelectCounter
    p = path.Path(verts)
    ind = p.contains_points(pix, radius=1.5)
    # update pix mask (histogram)
    # PROBLEM: it gets pix mask from dr every time (lasso from previous time gets lost )
    if OnSelectCounter == 0:
        pixMask = drSectorObj.sector.binaryMask()
    OnSelectCounter +=1
    pixMask = updateArray(pixMask, ind)
    drSectorObj.sector.FigObj.set_data(pixMask)    
    # update brain mask
    sliceNr = drSectorObj.sector.sliceNr
    mask = pix2vox_hist2D(
        drSectorObj.sector.invHistVolume[:,:,sliceNr],
        pixMask)
    drSectorObj.sector.brainMaskFigHandle.set_data(mask) 

    fig.canvas.draw_idle()

bLasso.on_clicked(lassoSwitch)

## does not work
#LassoSelector(ax, onselect)
## works
#lasso = LassoSelector(ax, onselect)

plt.show()
#plt.close()
