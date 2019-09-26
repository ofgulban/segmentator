#!/usr/bin/env python
"""Processing input and plotting, for experimental ncut feature.

TODO: Lots of code repetition, will be integrated better in the future.
"""

from __future__ import division, print_function
import numpy as np
import segmentator.config as cfg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider, Button, RadioButtons
from nibabel import load
from segmentator.utils import map_ima_to_2D_hist, prep_2D_hist
from segmentator.utils import truncate_range, scale_range, check_data
from segmentator.utils import set_gradient_magnitude
from segmentator.utils import export_gradient_magnitude_image
from segmentator.gui_utils import responsiveObj
from segmentator.config_gui import palette, axcolor, hovcolor

#
"""Load Data"""
nii = load(cfg.filename)
ncut_labels = np.load(cfg.ncut)

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


#
"""Data Processing"""
orig, dims = check_data(nii.get_data(), cfg.force_original_precision)
# Save min and max truncation thresholds to be used in axis labels
orig, pMin, pMax = truncate_range(orig, percMin=cfg.perc_min,
                                  percMax=cfg.perc_max)
# Continue with scaling the original truncated image and recomputing gradient
orig = scale_range(orig, scale_factor=cfg.scale, delta=0.0001)
gra = set_gradient_magnitude(orig, cfg.gramag)
if cfg.export_gramag:
    export_gradient_magnitude_image(gra, nii.get_filename(), nii.affine)

# Reshape ima (more intuitive for voxel-wise operations)
ima = np.ndarray.flatten(orig)
gra = np.ndarray.flatten(gra)

#
"""Plots"""
print("Preparing GUI...")
# Plot 2D histogram
fig = plt.figure(facecolor='0.775')
ax = fig.add_subplot(121)

counts, volHistH, d_min, d_max, nr_bins, bin_edges \
    = prep_2D_hist(ima, gra, discard_zeros=cfg.discard_zeros)

ax.set_xlim(d_min, d_max)
ax.set_ylim(d_min, d_max)
ax.set_xlabel("Intensity f(x)")
ax.set_ylabel("Gradient Magnitude f'(x)")
ax.set_title("2D Histogram")

# Plot map for poltical borders
pltMap = np.zeros((nr_bins, nr_bins, 1)).repeat(4, 2)
cmapPltMap = ListedColormap([[1, 1, 1, 0],  # transparent zeros
                             [0, 0, 0, 0.75],  # political borders
                             [1, 0, 0, 0.5],  # other colors for future use
                             [0, 0, 1, 0.5]])
boundsPltMap = [0, 1, 2, 3, 4]
normPltMap = BoundaryNorm(boundsPltMap, cmapPltMap.N)
pltMapH = ax.imshow(pltMap, cmap=cmapPltMap, norm=normPltMap,
                    vmin=boundsPltMap[1], vmax=boundsPltMap[-1],
                    extent=[0, nr_bins, nr_bins, 0], interpolation='none')

# Plot colorbar for 2d hist
volHistH.set_norm(LogNorm(vmax=np.power(10, cfg.cbar_init)))
fig.colorbar(volHistH, fraction=0.046, pad=0.04)  # magical perfect scaling

# Set up a colormap for ncut labels
ncut_palette = plt.cm.gist_rainbow
ncut_palette.set_under('w', 0)

# Plot hist mask (with ncut labels)
volHistMask = np.squeeze(ncut_labels[:, :, 0])
volHistMaskH = ax.imshow(volHistMask, interpolation='none',
                         alpha=0.2, cmap=ncut_palette,
                         vmin=np.min(ncut_labels)+1,  # to make 0 transparent
                         vmax=lMax,
                         extent=[0, nr_bins, nr_bins, 0])

# Plot 3D ima by default
ax2 = fig.add_subplot(122)
sliceNr = int(0.5*dims[2])
imaSlcH = ax2.imshow(orig[:, :, sliceNr], cmap=plt.cm.gray,
                     vmin=ima.min(), vmax=ima.max(), interpolation='none',
                     extent=[0, dims[1], dims[0], 0])
imaSlcMsk = np.zeros(dims[0:2])*total_labels[1]
imaSlcMskH = ax2.imshow(imaSlcMsk, interpolation='none', alpha=0.5,
                        cmap=ncut_palette, vmin=np.min(ncut_labels)+1,
                        vmax=lMax,
                        extent=[0, dims[1], dims[0], 0])

# Adjust subplots on figure
bottom = 0.30
fig.subplots_adjust(bottom=bottom)
fig.canvas.set_window_title(nii.get_filename())
plt.axis('off')


# %%
"""Initialisation"""
# Initiate a flexible figure object, pass to it usefull properties
flexFig = responsiveObj(figure=ax.figure, axes=ax.axes, axes2=ax2.axes,
                        segmType='ncut', orig=orig, nii=nii, ima=ima,
                        nrBins=nr_bins,
                        sliceNr=sliceNr,
                        imaSlcH=imaSlcH,
                        imaSlcMsk=imaSlcMsk, imaSlcMskH=imaSlcMskH,
                        volHistMask=volHistMask,
                        volHistMaskH=volHistMaskH,
                        pltMap=pltMap, pltMapH=pltMapH,
                        counterField=np.zeros((nr_bins, nr_bins)),
                        orig_ncut_labels=orig_ncut_labels,
                        ima_ncut_labels=ima_ncut_labels,
                        lMax=lMax)

# Make the figure responsive to clicks
flexFig.connect()
# Get mapping from image slice to volume histogram
ima2volHistMap = map_ima_to_2D_hist(xinput=ima, yinput=gra, bins_arr=bin_edges)
flexFig.invHistVolume = np.reshape(ima2volHistMap, dims)

# %%
"""Sliders and Buttons"""
axcolor, hovcolor = '0.875', '0.975'

# Radio buttons (ugly but good enough for now)
rax = plt.axes([0.91, 0.35, 0.08, 0.5], facecolor=(0.75, 0.75, 0.75))
flexFig.radio = RadioButtons(rax, [str(i) for i in range(7)],
                             activecolor=(0.25, 0.25, 0.25))

# Colorbar slider
axHistC = plt.axes([0.15, bottom-0.230, 0.25, 0.025], facecolor=axcolor)
flexFig.sHistC = Slider(axHistC, 'Colorbar', 1, cfg.cbar_max,
                        valinit=cfg.cbar_init, valfmt='%0.1f')

# Label slider
axLabels = plt.axes([0.15, bottom-0.270, 0.25, 0.025], facecolor=axcolor)
flexFig.sLabelNr = Slider(axLabels, 'Labels', 0, lMax,
                          valinit=lMax, valfmt='%i')

# Image browser slider
axSliceNr = plt.axes([0.6, bottom-0.15, 0.25, 0.025], facecolor=axcolor)
flexFig.sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999,
                          valinit=0.5, valfmt='%0.3f')

# Cycle button
cycleax = plt.axes([0.55, bottom-0.285, 0.075, 0.075])
flexFig.bCycle = Button(cycleax, 'Cycle\nView',
                        color=axcolor, hovercolor=hovcolor)
flexFig.cycleCount = 0

# Rotate button
rotateax = plt.axes([0.55, bottom-0.285, 0.075, 0.0375])
flexFig.bRotate = Button(rotateax, 'Rotate',
                         color=axcolor, hovercolor=hovcolor)
flexFig.rotationCount = 0

# Export nii button
exportax = plt.axes([0.75, bottom-0.285, 0.075, 0.075])
flexFig.bExport = Button(exportax, 'Export\nNifti',
                         color=axcolor, hovercolor=hovcolor)

# Export nyp button
exportax = plt.axes([0.85, bottom-0.285, 0.075, 0.075])
flexFig.bExportNyp = Button(exportax, 'Export\nHist',
                            color=axcolor, hovercolor=hovcolor)

# Reset button
resetax = plt.axes([0.65, bottom-0.285, 0.075, 0.075])
flexFig.bReset = Button(resetax, 'Reset', color=axcolor, hovercolor=hovcolor)


# %%
"""Updates"""
flexFig.sHistC.on_changed(flexFig.updateColorBar)
flexFig.sSliceNr.on_changed(flexFig.updateImaBrowser)
flexFig.sLabelNr.on_changed(flexFig.updateLabels)
flexFig.bCycle.on_clicked(flexFig.cycleView)
flexFig.bRotate.on_clicked(flexFig.changeRotation)
flexFig.bExport.on_clicked(flexFig.exportNifti)
flexFig.bExportNyp.on_clicked(flexFig.exportNyp)
flexFig.bReset.on_clicked(flexFig.resetGlobal)
flexFig.radio.on_clicked(flexFig.updateLabelsRadio)


# TODO: Temporary solution for displaying original x-y axis labels
def update_axis_labels(event):
    """Swap histogram bin indices with original values."""
    xlabels = [item.get_text() for item in ax.get_xticklabels()]
    orig_range_labels = np.linspace(pMin, pMax, len(xlabels))

    # Adjust displayed decimals based on data range
    data_range = pMax - pMin
    if data_range > 200:  # arbitrary value
        xlabels = [('%i' % i) for i in orig_range_labels]
    elif data_range > 20:
        xlabels = [('%.1f' % i) for i in orig_range_labels]
    elif data_range > 2:
        xlabels = [('%.2f' % i) for i in orig_range_labels]
    else:
        xlabels = [('%.3f' % i) for i in orig_range_labels]

    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(xlabels)  # limits of y axis assumed to be the same as x


fig.canvas.mpl_connect('resize_event', update_axis_labels)

plt.show()
