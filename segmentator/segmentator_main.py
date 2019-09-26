#!/usr/bin/env python
"""Processing input and plotting."""

from __future__ import division, print_function
import numpy as np
import segmentator.config as cfg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button, LassoSelector
from matplotlib import path
from nibabel import load
from segmentator.utils import map_ima_to_2D_hist, prep_2D_hist
from segmentator.utils import truncate_range, scale_range, check_data
from segmentator.utils import set_gradient_magnitude
from segmentator.utils import export_gradient_magnitude_image
from segmentator.gui_utils import sector_mask, responsiveObj
from segmentator.config_gui import palette, axcolor, hovcolor

#
"""Data Processing"""
nii = load(cfg.filename)
orig, dims = check_data(nii.get_data(), cfg.force_original_precision)
# Save min and max truncation thresholds to be used in axis labels
if np.isnan(cfg.valmin) or np.isnan(cfg.valmax):
    orig, pMin, pMax = truncate_range(orig, percMin=cfg.perc_min,
                                      percMax=cfg.perc_max)
else:  # TODO: integrate this into truncate range function
    orig[orig < cfg.valmin] = cfg.valmin
    orig[orig > cfg.valmax] = cfg.valmax
    pMin, pMax = cfg.valmin, cfg.valmax

# Continue with scaling the original truncated image and recomputing gradient
orig = scale_range(orig, scale_factor=cfg.scale, delta=0.0001)
gra = set_gradient_magnitude(orig, cfg.gramag)
if cfg.export_gramag:
    export_gradient_magnitude_image(gra, nii.get_filename(), cfg.gramag,
                                    nii.affine)
# Reshape for voxel-wise operations
ima = np.copy(orig.flatten())
gra = gra.flatten()

#
"""Plots"""
print("Preparing GUI...")
# Plot 2D histogram
fig = plt.figure(facecolor='0.775')
ax = fig.add_subplot(121)

counts, volHistH, d_min, d_max, nr_bins, bin_edges \
    = prep_2D_hist(ima, gra, discard_zeros=cfg.discard_zeros)

# Set x-y axis range to the same (x-axis range)
ax.set_xlim(d_min, d_max)
ax.set_ylim(d_min, d_max)
ax.set_xlabel("Intensity f(x)")
ax.set_ylabel("Gradient Magnitude f'(x)")
ax.set_title("2D Histogram")

# Plot colorbar for 2D hist
volHistH.set_norm(LogNorm(vmax=np.power(10, cfg.cbar_init)))
fig.colorbar(volHistH, fraction=0.046, pad=0.04)  # magical scaling

# Plot 3D ima by default
ax2 = fig.add_subplot(122)
sliceNr = int(0.5*dims[2])
imaSlcH = ax2.imshow(orig[:, :, sliceNr], cmap=plt.cm.gray, vmin=ima.min(),
                     vmax=ima.max(), interpolation='none',
                     extent=[0, dims[1], dims[0], 0], zorder=0)

imaSlcMsk = np.ones(dims[0:2])
imaSlcMskH = ax2.imshow(imaSlcMsk, cmap=palette, vmin=0.1,
                        interpolation='none', alpha=0.5,
                        extent=[0, dims[1], dims[0], 0], zorder=1)

# Adjust subplots on figure
bottom = 0.30
fig.subplots_adjust(bottom=bottom)
fig.canvas.set_window_title(nii.get_filename())
plt.axis('off')

#
"""Initialisation"""
# Create first instance of sector mask
sectorObj = sector_mask((nr_bins, nr_bins), cfg.init_centre, cfg.init_radius,
                        cfg.init_theta)

# Draw sector mask for the first time
volHistMaskH, volHistMask = sectorObj.draw(ax, cmap=palette, alpha=0.2,
                                           vmin=0.1, interpolation='nearest',
                                           origin='lower', zorder=1,
                                           extent=[0, nr_bins, 0, nr_bins])

# Initiate a flexible figure object, pass to it useful properties
idxLasso = np.zeros(nr_bins*nr_bins, dtype=bool)
lassoSwitchCount = 0
lassoErase = 1  # 1 for drawing, 0 for erasing
flexFig = responsiveObj(figure=ax.figure, axes=ax.axes, axes2=ax2.axes,
                        segmType='main', orig=orig, nii=nii,
                        sectorObj=sectorObj,
                        nrBins=nr_bins,
                        sliceNr=sliceNr,
                        imaSlcH=imaSlcH,
                        imaSlcMsk=imaSlcMsk, imaSlcMskH=imaSlcMskH,
                        volHistMask=volHistMask, volHistMaskH=volHistMaskH,
                        contains=volHistMaskH.contains,
                        counts=counts,
                        idxLasso=idxLasso,
                        lassoSwitchCount=lassoSwitchCount,
                        lassoErase=lassoErase)

# Make the figure responsive to clicks
flexFig.connect()
ima2volHistMap = map_ima_to_2D_hist(xinput=ima, yinput=gra, bins_arr=bin_edges)
flexFig.invHistVolume = np.reshape(ima2volHistMap, dims)
ima, gra = None, None

#
"""Sliders and Buttons"""
# Colorbar slider
axHistC = plt.axes([0.15, bottom-0.20, 0.25, 0.025], facecolor=axcolor)
flexFig.sHistC = Slider(axHistC, 'Colorbar', 1, cfg.cbar_max,
                        valinit=cfg.cbar_init, valfmt='%0.1f')

# Image browser slider
axSliceNr = plt.axes([0.6, bottom-0.15, 0.25, 0.025], facecolor=axcolor)
flexFig.sSliceNr = Slider(axSliceNr, 'Slice', 0, 0.999, valinit=0.5,
                          valfmt='%0.2f')

# Theta sliders
aThetaMin = plt.axes([0.15, bottom-0.10, 0.25, 0.025], facecolor=axcolor)
flexFig.sThetaMin = Slider(aThetaMin, 'ThetaMin', 0, 359.9,
                           valinit=cfg.init_theta[0], valfmt='%0.1f')
aThetaMax = plt.axes([0.15, bottom-0.15, 0.25, 0.025], facecolor=axcolor)
flexFig.sThetaMax = Slider(aThetaMax, 'ThetaMax', 0, 359.9,
                           valinit=cfg.init_theta[1]-0.1, valfmt='%0.1f')

# Cycle button
cycleax = plt.axes([0.55, bottom-0.2475, 0.075, 0.0375])
flexFig.bCycle = Button(cycleax, 'Cycle',
                        color=axcolor, hovercolor=hovcolor)

# Rotate button
rotateax = plt.axes([0.55, bottom-0.285, 0.075, 0.0375])
flexFig.bRotate = Button(rotateax, 'Rotate',
                         color=axcolor, hovercolor=hovcolor)

# Reset button
resetax = plt.axes([0.65, bottom-0.285, 0.075, 0.075])
flexFig.bReset = Button(resetax, 'Reset', color=axcolor, hovercolor=hovcolor)

# Export nii button
exportax = plt.axes([0.75, bottom-0.285, 0.075, 0.075])
flexFig.bExport = Button(exportax, 'Export\nNifti',
                         color=axcolor, hovercolor=hovcolor)

# Export nyp button
exportax = plt.axes([0.85, bottom-0.285, 0.075, 0.075])
flexFig.bExportNyp = Button(exportax, 'Export\nHist',
                            color=axcolor, hovercolor=hovcolor)

#
"""Updates"""
flexFig.sHistC.on_changed(flexFig.updateColorBar)
flexFig.sSliceNr.on_changed(flexFig.updateImaBrowser)
flexFig.sThetaMin.on_changed(flexFig.updateThetaMin)
flexFig.sThetaMax.on_changed(flexFig.updateThetaMax)
flexFig.bCycle.on_clicked(flexFig.cycleView)
flexFig.bRotate.on_clicked(flexFig.changeRotation)
flexFig.bExport.on_clicked(flexFig.exportNifti)
flexFig.bExportNyp.on_clicked(flexFig.exportNyp)
flexFig.bReset.on_clicked(flexFig.resetGlobal)


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

#
"""Lasso selection"""
# Lasso button
lassoax = plt.axes([0.15, bottom-0.285, 0.075, 0.075])
bLasso = Button(lassoax, 'Lasso\nOff', color=axcolor, hovercolor=hovcolor)

# Lasso draw/erase
lassoEraseAx = plt.axes([0.25, bottom-0.285, 0.075, 0.075])
bLassoErase = Button(lassoEraseAx, 'Erase\nOff', color=axcolor,
                     hovercolor=hovcolor)
bLassoErase.ax.patch.set_visible(False)
bLassoErase.label.set_visible(False)
bLassoErase.ax.axis('off')


def lassoSwitch(event):
    """Enable disable lasso tool."""
    global lasso
    lasso = []
    flexFig.lassoSwitchCount = (flexFig.lassoSwitchCount+1) % 2
    if flexFig.lassoSwitchCount == 1:  # enable lasso
        flexFig.disconnect()  # disable drag function of sector mask
        lasso = LassoSelector(ax, onselect)
        bLasso.label.set_text("Lasso\nOn")
        # Make erase button appear on in lasso mode
        bLassoErase.ax.patch.set_visible(True)
        bLassoErase.label.set_visible(True)
        bLassoErase.ax.axis('on')

    else:  # disable lasso
        flexFig.connect()  # enable drag function of sector mask
        bLasso.label.set_text("Lasso\nOff")
        # Make erase button disappear
        bLassoErase.ax.patch.set_visible(False)
        bLassoErase.label.set_visible(False)
        bLassoErase.ax.axis('off')

# Pixel coordinates
pix = np.arange(nr_bins)
xv, yv = np.meshgrid(pix, pix)
pix = np.vstack((xv.flatten(), yv.flatten())).T


def onselect(verts):
    """Lasso related."""
    global pix
    p = path.Path(verts)
    newLasIdx = p.contains_points(pix, radius=1.5)  # New lasso indices
    flexFig.idxLasso[newLasIdx] = flexFig.lassoErase  # Update lasso indices
    flexFig.remapMsks()  # Update volume histogram mask
    flexFig.updatePanels(update_slice=False, update_rotation=True,
                         update_extent=True)


def lassoEraseSwitch(event):
    """Enable disable lasso erase function."""
    flexFig.lassoErase = (flexFig.lassoErase + 1) % 2
    if flexFig.lassoErase is 1:
        bLassoErase.label.set_text("Erase\nOff")
    elif flexFig.lassoErase is 0:
        bLassoErase.label.set_text("Erase\nOn")


bLasso.on_clicked(lassoSwitch)  # lasso on/off
bLassoErase.on_clicked(lassoEraseSwitch)  # lasso erase on/off
flexFig.remapMsks()
flexFig.updatePanels(update_slice=True, update_rotation=False,
                     update_extent=False)

print("GUI is ready.")
plt.show()
