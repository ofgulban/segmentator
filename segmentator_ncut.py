"""Processing input and plotting, for experimental ncut feature.

Lots of code repetition, will be integrated better in the future.
"""

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
from nibabel import load
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider, Button, RadioButtons
from utils import Ima2VolHistMapping, TruncateRange, ScaleRange, Hist2D
from segmentator_functions import responsiveObj
from VolHist2ImaMapping4Vols import VolHist2ImaOffline
import segmentator

# %%
"""Load Data"""
#
nii = load(segmentator.args.filename)
ncut_labels = np.load(segmentator.args.ncut)

# transpose the labels
ncut_labels = np.transpose(ncut_labels, (1, 0, 2))
nrTotal_labels = sum([2**x for x in range(ncut_labels.shape[2])])
total_labels = np.arange(nrTotal_labels)
total_labels[1::2] = total_labels[-2:0:-2]

# relabel the labels from ncut, assign ascending integers starting with counter
counter = 0
for ind in np.arange(ncut_labels.shape[2]):
    tmp = ncut_labels[:, :, ind]
    uniqueVals = np.unique(tmp)
    nrUniqueVals = len(uniqueVals)
    newInd = np.arange(nrUniqueVals) + counter
    newVals = total_labels[newInd]
    tmp2 = np.zeros((tmp.shape))
    for ind2, val in enumerate(uniqueVals):
        tmp2[tmp == val] = newVals[ind2]
    counter = counter + nrUniqueVals
    ncut_labels[:, :, ind] = tmp2
lMax = np.max(ncut_labels)

orig_ncut_labels = ncut_labels.copy()
ima_ncut_labels = ncut_labels.copy()


# %%
"""Data Pre-Processing"""
orig = np.squeeze(nii.get_data())
percMin, percMax = segmentator.args.percmin, segmentator.args.percmax
orig = TruncateRange(orig, percMin=percMin, percMax=percMax)
orig = ScaleRange(orig, scaleFactor=segmentator.args.scale, delta=0.0001)
# define dataMin and dataMax for later use
dataMin = np.round(orig.min())
dataMax = np.round(orig.max())

# copy intensity data so we can flatten the copy and leave original intact
ima = orig.copy()
if segmentator.args.gramag:
    nii2 = load(segmentator.args.gramag)
    gra = np.squeeze(nii2.get_data())
    gra = TruncateRange(gra, percMin=percMin, percMax=percMax)
    gra = ScaleRange(gra, scaleFactor=segmentator.args.scale, delta=0.0001)
else:
    # calculate gradient magnitude (using L2 norm of the vector)
    gra = np.gradient(ima)
    gra = np.sqrt(np.power(gra[0], 2) + np.power(gra[1], 2) +
                  np.power(gra[2], 2))

# reshape ima (more intuitive for voxel-wise operations)
ima = np.ndarray.flatten(ima)
gra = np.ndarray.flatten(gra)

# %%
"""Plots"""
# Plot 2D histogram
fig = plt.figure()
ax = fig.add_subplot(121)

counts, volHistH, dataMin, dataMax, nrBins, binEdges = Hist2D(ima, gra)

ax.set_xlim(dataMin, dataMax)
ax.set_ylim(dataMin, dataMax)
ax.set_xlabel("Intensity f(x)")
ax.set_ylabel("Gradient Magnitude f'(x)")
ax.set_title("2D Histogram")

# plot map for poltical borders
pltMap = np.zeros((nrBins, nrBins, 1)).repeat(4, 2)
cmapPltMap = ListedColormap(['w', 'black', 'red', 'blue'])
boundsPltMap = [0, 1, 2, 3, 4]
normPltMap = BoundaryNorm(boundsPltMap, cmapPltMap.N)
pltMapH = ax.imshow(pltMap, alpha=1, cmap=cmapPltMap, norm=normPltMap,
                    extent=[0, nrBins, nrBins, 0])

# plot colorbar for 2d hist
volHistH.set_norm(LogNorm(vmax=1000))
plt.colorbar(volHistH)

# Set up a colormap for ncut labels
ncut_palette = plt.cm.gist_rainbow
ncut_palette.set_under('w', 0)

# fill ncut_labels up with zeros if needed
if ncut_labels.shape[0] < nrBins:
    dif1 = nrBins - ncut_labels.shape[0]
    ncut_labels = np.append(ncut_labels,
                            np.zeros((dif1,
                                      ncut_labels.shape[1],
                                      ncut_labels.shape[2])), axis=0)

if ncut_labels.shape[1] < nrBins:
    dif2 = nrBins - ncut_labels.shape[1]
    ncut_labels = np.append(ncut_labels,
                            np.zeros((ncut_labels.shape[0],
                                      dif2,
                                      ncut_labels.shape[2])), axis=1)

# plot hist mask (with ncut labels)
volHistMask = np.squeeze(ncut_labels[:, :, 0])
volHistMaskH = ax.imshow(volHistMask, interpolation='none',
                         alpha=0.2, cmap=ncut_palette,
                         vmin=np.min(ncut_labels)+1,  # to make 0 transparent
                         vmax=lMax,
                         extent=[0, nrBins, nrBins, 0])

# plot 3D ima by default
ax2 = fig.add_subplot(122)
slcH = ax2.imshow(orig[:, :, int(orig.shape[2]/2)], cmap=plt.cm.gray,
                  vmin=ima.min(), vmax=ima.max(), interpolation='none',
                  extent=[0, orig.shape[1], orig.shape[0], 0])
imaMask = np.zeros(orig.shape[0:2])*total_labels[1]
imaMaskH = ax2.imshow(imaMask, interpolation='none', alpha=0.5,
                      cmap=ncut_palette, vmin=np.min(ncut_labels)+1,
                      vmax=lMax,
                      extent=[0, orig.shape[1], orig.shape[0], 0])

# adjust subplots on figure
bottom = 0.30
fig.subplots_adjust(bottom=bottom)
plt.axis('off')


# %%
"""Initialisation"""
segmType = 'ncut'
# initiate a flexible figure object, pass to it usefull properties
flexFig = responsiveObj(figure=ax.figure,
                        axes=ax.axes,
                        axes2=ax2.axes,
                        segmType=segmType,
                        orig=orig,
                        nii=nii,
                        ima=ima,
                        nrBins=nrBins,
                        sliceNr=int(0.5*orig.shape[2]),
                        slcH=slcH,
                        imaMask=imaMask,
                        imaMaskH=imaMaskH,
                        volHistMask=volHistMask,
                        volHistMaskH=volHistMaskH,
                        pltMap=pltMap,
                        pltMapH=pltMapH,
                        counterField=np.zeros((nrBins, nrBins)),
                        orig_ncut_labels=orig_ncut_labels,
                        ima_ncut_labels=ima_ncut_labels,
                        lMax=lMax
                        )

# make the figure responsive to clicks
flexFig.connect()
# get mapping from image slice to volume histogram
ima2volHistMap = Ima2VolHistMapping(xinput=ima, yinput=gra, binsArray=binEdges)
flexFig.invHistVolume = np.reshape(ima2volHistMap, orig.shape)
# get mapping from volume histogram to all volxes in data (slow!!!)
#print "start VolHist to Ima calculation"
#flexFig.volHist2ImaMap = VolHist2ImaOffline(ima2volHistMap, nrBins)
#print "calculation done"

# %%
"""Sliders and Buttons"""
axcolor = 'lightgoldenrodyellow'

# radio buttons (ugly but good enough for now)
rax = plt.axes([0.91, 0.35, 0.08, 0.5], axisbg=axcolor)  # x, y, xsize, ysize
flexFig.radio = RadioButtons(rax, [str(i) for i in range(7)])

# colorbar slider
axHistC = plt.axes([0.15, bottom-0.230, 0.25, 0.025], axisbg=axcolor)
flexFig.sHistC = Slider(axHistC, 'Colorbar', 1, 5, valinit=3, valfmt='%0.1f')

# label slider
axLabels = plt.axes([0.15, bottom-0.270, 0.25, 0.025], axisbg=axcolor)
flexFig.sLabelNr = Slider(axLabels, 'Labels', 0, lMax,
                          valinit=lMax, valfmt='%i')

# ima browser slider
axSliceNr = plt.axes([0.6, bottom-0.15, 0.25, 0.025], axisbg=axcolor)
flexFig.sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999,
                          valinit=0.5, valfmt='%0.3f')

# cycle button
cycleax = plt.axes([0.55, bottom-0.285, 0.075, 0.075])
flexFig.bCycle = Button(cycleax, 'Cycle\nView',
                        color=axcolor, hovercolor='0.975')
flexFig.cycleCount = 0

# export nii button
exportax = plt.axes([0.75, bottom-0.285, 0.075, 0.075])
flexFig.bExport = Button(exportax, 'Export\nNifti',
                         color=axcolor, hovercolor='0.975')

# export nyp button
exportax = plt.axes([0.85, bottom-0.285, 0.075, 0.075])
flexFig.bExportNyp = Button(exportax, 'Export\nLabels',
                            color=axcolor, hovercolor='0.975')

# reset button
resetax = plt.axes([0.65, bottom-0.285, 0.075, 0.075])
flexFig.bReset = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

# imaMask button
imaMaskax = plt.axes([0.9125, 0.25, 0.075, 0.075])
flexFig.bImaMask = Button(imaMaskax, 'Transp\nMask',
                          color=axcolor, hovercolor='0.975')


# %%
"""Updates"""
flexFig.sHistC.on_changed(flexFig.updateColorBar)
flexFig.sSliceNr.on_changed(flexFig.updateImaBrowser)
flexFig.sLabelNr.on_changed(flexFig.updateLabels)
flexFig.bCycle.on_clicked(flexFig.cycleView)
flexFig.bExport.on_clicked(flexFig.exportNifti)
flexFig.bExportNyp.on_clicked(flexFig.exportNyp)
flexFig.bReset.on_clicked(flexFig.resetGlobal)
flexFig.bImaMask.on_clicked(flexFig.imaMaskTrans)
flexFig.radio.on_clicked(flexFig.updateLabelsRadio)

plt.show()
