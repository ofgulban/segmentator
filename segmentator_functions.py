"""Functions covering the user interaction with the GUI."""

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
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import VolHist2ImaMapping
from nibabel import save, Nifti1Image
import config as cfg


class responsiveObj:
    """Stuff to interact in the user interface."""

    def __init__(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                setattr(self, key, value)
        self.basename = self.nii.get_filename().split(os.extsep, 1)[0]
        self.press = None
        self.ctrlHeld = False
        self.labelNr = 0
        self.imaMaskSwitchCount = 0
        self.nrExports = 0
        self.entropWin = 0

    def updateMsks(self):
        """Update volume histogram mask."""
        if self.segmType == 'main':
            self.volHistMask = self.sectorObj.binaryMask()
            self.volHistMask = self.lassoArr(self.volHistMask,
                                             self.idxLasso)
            self.volHistMaskH.set_data(self.volHistMask)
        elif self.segmType == 'ncut':
            self.labelContours()
            self.volHistMaskH.set_data(self.volHistMask)
            self.volHistMaskH.set_extent((0, self.nrBins, self.nrBins, 0))
        # update imaMask
        self.imaMask = VolHist2ImaMapping(
            self.invHistVolume[:, :, self.sliceNr], self.volHistMask)
        self.imaMaskH.set_data(self.imaMask)
        self.imaMaskH.set_extent((0, self.imaMask.shape[1],
                                  self.imaMask.shape[0], 0))
        # draw to canvas
        self.figure.canvas.draw()

    def updateSlc(self):
        """Update image browser slices."""
        self.slcH.set_data(self.orig[:, :, self.sliceNr])
        self.slcH.set_extent((0, self.orig.shape[1], self.orig.shape[0], 0))

    def connect(self):
        """Make the object responsive."""
        self.cidpress = self.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.cidkeypress = self.figure.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        self.cidkeyrelease = self.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release)

    def on_key_press(self, event):
        """Determine what happens if key is pressed."""
        if event.key == 'control':
            self.ctrlHeld = True
        elif event.key == 'q':
            self.imaMaskIncr(-0.1)
        elif event.key == 'w':
            self.imaMaskTransSwitch()
        elif event.key == 'e':
            self.imaMaskIncr(0.1)
        if self.segmType == 'main':
            if event.key == 'up':
                self.sectorObj.scale_r(1.05)
                self.updateMsks()
            elif event.key == 'down':
                self.sectorObj.scale_r(0.95)
                self.updateMsks()
            elif event.key == 'right':
                self.sectorObj.rotate(-10.0)
                self.updateMsks()
            elif event.key == 'left':
                self.sectorObj.rotate(10.0)
                self.updateMsks()
            else:
                return

    def on_key_release(self, event):
        """Determine what happens if key is released."""
        if event.key == 'control':
            self.ctrlHeld = False

    def findVoxInHist(self, event):
        """Find voxel's location in histogram."""
        self.press = event.xdata, event.ydata
        xvoxel = int(np.floor(event.xdata))
        yvoxel = int(np.floor(event.ydata))
        # SWITCH x and y voxel to get linear index
        # since NOT Cartes.!!!
        pixelLin = self.invHistVolume[
            yvoxel, xvoxel, self.sliceNr]
        # ind2sub
        xpix = (pixelLin / self.nrBins)
        ypix = (pixelLin % self.nrBins)
        # SWITCH x and y for circle centre
        # since back TO Cartesian!!!
        self.circle1 = plt.Circle(
            (ypix, xpix), radius=5, color='b')
        self.axes.add_artist(self.circle1)
        self.figure.canvas.draw()

    def on_press(self, event):
        """Determine what happens if mouse button is clicked."""
        if self.segmType == 'main':
            if event.button == 1:  # left button
                if event.inaxes == self.axes:  # cursor in left plot (hist)
                    if self.ctrlHeld is False:  # ctrl no
                        contains = self.contains(event)
                        if not contains:
                            print 'cursor outside circle mask'
                        if not contains:
                            return
                        # get sector centre x and y positions
                        x0 = self.sectorObj.cx
                        y0 = self.sectorObj.cy
                        # also get cursor x and y position and safe to press
                        self.press = x0, y0, event.xdata, event.ydata
                elif event.inaxes == self.axes2:  # cursor in right plot (brow)
                    self.findVoxInHist(event)
                else:
                    return
            elif event.button == 2:  # scroll button
                if event.inaxes != self.axes:  # outside axes
                    return
                # increase/decrease radius of the sector mask
                if self.ctrlHeld is False:  # ctrl no
                    self.sectorObj.scale_r(1.05)
                    self.updateMsks()
                elif self.ctrlHeld is True:  # ctrl yes
                    self.sectorObj.rotate(10.0)
                    self.updateMsks()
                else:
                    return
            elif event.button == 3:  # right button
                if event.inaxes != self.axes:
                    return
                # rotate the sector mask
                if self.ctrlHeld is False:  # ctrl no
                    self.sectorObj.scale_r(0.95)
                    self.updateMsks()
                elif self.ctrlHeld is True:  # ctrl yes
                    self.sectorObj.rotate(-10.0)
                    self.updateMsks()
                else:
                    return
        elif self.segmType == 'ncut':
            if event.button == 1:  # left button
                if event.inaxes == self.axes:  # cursor in left plot (hist)
                    xbin = int(np.floor(event.xdata))
                    ybin = int(np.floor(event.ydata))
                    val = self.volHistMask[ybin][xbin]
                    # increment counterField for values in clicked subfield, at
                    # the first click the entire field constitutes the subfield
                    counter = int(self.counterField[ybin][xbin])
                    if counter+1 >= self.ima_ncut_labels.shape[2]:
                        print "already at maximum ncut dimension"
                        return
                    self.counterField[(
                        self.ima_ncut_labels[:, :, counter] ==
                        self.ima_ncut_labels[[ybin], [xbin], counter])] += 1
                    print "counter:" + str(counter+1)
                    # define arrays with old and new labels for later indexing
                    oLabels = self.ima_ncut_labels[:, :, counter]
                    nLabels = self.ima_ncut_labels[:, :, counter+1]
                    # replace old values with new values (in clicked subfield)
                    self.volHistMask[oLabels == val] = np.copy(
                        nLabels[oLabels == val])
                    self.updateMsks()

                elif event.inaxes == self.axes2:  # cursor in right plot (brow)
                    self.findVoxInHist(event)
                else:
                    return
            elif event.button == 3:  # right button
                if event.inaxes == self.axes:  # cursor in left plot (hist)
                    xbin = np.floor(event.xdata)
                    ybin = np.floor(event.ydata)
                    val = self.volHistMask[ybin][xbin]
                    # fetch the slider value to get label nr
                    self.volHistMask[self.volHistMask == val] = \
                        np.copy(self.labelNr)
                    self.updateMsks()

    def on_motion(self, event):
        """Determine what happens if mouse button moves."""
        if self.segmType == 'main':
            # ... button is pressed
            if self.press is None:
                return
            # ... cursor is in left plot
            if event.inaxes != self.axes:
                return
            # get former sector centre x and y positions,
            # cursor x and y positions
            x0, y0, xpress, ypress = self.press
            # calculate difference betw cursor pos on click
            # and new pos dur motion
            # switch x0 & y0 cause volHistMask not Cart
            dy = event.xdata - xpress
            dx = event.ydata - ypress
            # update x and y position of sector,
            # based on past motion of cursor
            self.sectorObj.set_x(x0+dx)
            self.sectorObj.set_y(y0+dy)
            # update masks
            self.updateMsks()
        else:
            return

    def on_release(self, event):
        """Determine what happens if mouse button is released."""
        self.press = None
        # try to remove the blue circle
        try:
            self.circle1.remove()
        except:
            return
        self.figure.canvas.draw()

    def disconnect(self):
        """Make the object unresponsive."""
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)

    def updateColorBar(self, val):
        """Update slider for scaling log colorbar in 2D hist."""
        histVMax = np.power(10, self.sHistC.val)
        plt.clim(vmax=histVMax)

    def updateImaBrowser(self, val):
        """Update image browse."""
        # scale slider value [0,1) to dimension index
        self.sliceNr = int(self.sSliceNr.val*self.orig.shape[2])
        self.updateSlc()  # update brain slice
        self.updateMsks()

    def cycleView(self, event):
        """Cycle through views."""
        self.cycleCount = (self.cycleCount+1) % 3
        # transpose data
        self.orig = np.transpose(self.orig, (2, 0, 1))
        # transpose ima2volHistMap
        self.invHistVolume = np.transpose(self.invHistVolume, (2, 0, 1))
        # update slice number
        self.sliceNr = int(self.sSliceNr.val*self.orig.shape[2])
        # update brain slice
        self.updateSlc()
        self.updateMsks()

    def exportNifti(self, event):
        """Export labels in the image browser as a nifti file."""
        print "start exporting labels..."
        # put the permuted indices back to their original format
        cycBackPerm = (self.cycleCount, (self.cycleCount+1) % 3,
                       (self.cycleCount+2) % 3)
        # assing unique integers (for ncut labels)
        out_volHistMask = np.copy(self.volHistMask)
        labels = np.unique(self.volHistMask)
        intLabels = [i for i in range(labels.size)]
        for label, newLabel in zip(labels, intLabels):
            out_volHistMask[out_volHistMask == label] = intLabels[newLabel]
        # get 3D brain mask
        temp = np.transpose(self.invHistVolume, cycBackPerm)
        outNii = VolHist2ImaMapping(temp, out_volHistMask)
        outNii = outNii.reshape(temp.shape)
        # save mask image as nii
        new_image = Nifti1Image(outNii, header=self.nii.get_header(),
                                affine=self.nii.get_affine())
        # get new flex file name and check for overwriting
        self.nrExports = 0
        self.flexfilename = '_labels_' + str(self.nrExports) + '.nii.gz'
        while os.path.isfile(self.basename + self.flexfilename):
            self.nrExports += 1
            self.flexfilename = '_labels_' + str(self.nrExports) + '.nii.gz'
        save(new_image, self.basename + self.flexfilename)
        print "successfully exported image labels as: \n" + \
            self.basename + self.flexfilename

    def resetGlobal(self, event):
        """Reset stuff."""
        # reset color bar
        self.sHistC.reset()
        # reset ima browser slider
        self.sSliceNr.reset()
        # reset transparency ima mask slider
        self.sImaMaskTrans.reset()
        # reset slice number
        self.sliceNr = int(self.sSliceNr.val*self.orig.shape[2])
        # update brain slice
        self.updateSlc()
        if self.segmType == 'main':
            if self.lassoSwitchCount == 1:  # reset only lasso drawing
                self.idxLasso = np.zeros(self.nrBins*self.nrBins, dtype=bool)
            else:
                # reset theta sliders
                self.sThetaMin.reset()
                self.sThetaMax.reset()
                # reset values for mask
                self.sectorObj.set_x(cfg.init_centre[0])
                self.sectorObj.set_y(cfg.init_centre[1])
                self.sectorObj.set_r(cfg.init_radius)
                self.sectorObj.tmin, self.sectorObj.tmax = np.deg2rad(
                    cfg.init_theta)

        elif self.segmType == 'ncut':
            self.sLabelNr.reset()
            # reset ncut labels
            self.ima_ncut_labels = np.copy(self.orig_ncut_labels)
            # reset values for volHistMask
            self.volHistMask = self.ima_ncut_labels[:, :, 0].reshape(
                (self.nrBins, self.nrBins))
            # reset counter field
            self.counterField = np.zeros((self.nrBins, self.nrBins))
            # reset political borders
            self.pltMap = np.zeros((self.nrBins, self.nrBins))
            self.pltMapH.set_data(self.pltMap)
        self.updateMsks()

    def updateThetaMin(self, val):
        """Update volume histogram mask's minimum theta."""
        if self.segmType == 'main':
            thetaVal = self.sThetaMin.val  # get theta value from slider
            self.sectorObj.updateThetaMin(thetaVal)
            self.updateMsks()
        else:
            return

    def updateThetaMax(self, val):
        """Update volume histogram mask's maximum theta."""
        if self.segmType == 'main':
            thetaVal = self.sThetaMax.val  # get theta value from slider
            self.sectorObj.updateThetaMax(thetaVal)
            self.updateMsks()
        else:
            return

    def exportNyp(self, event):
        """Export histogram counts as a numpy array."""
        if self.segmType == 'ncut':
            filename = '_volHistLabels' + '_pcMax' + \
                str(int(self.initTpl[0])) + '_pcMin' + \
                str(int(self.initTpl[1])) + '_sc' + \
                str(int(self.initTpl[2]))
            np.save(self.basename + filename, self.volHistMask)
            print "successfully exported histogram colors as: \n" + \
                self.basename + filename
        elif self.segmType == 'main':
            filename = '_volHist' + '_pcMax' + str(int(self.initTpl[0])) + \
                '_pcMin' + str(int(self.initTpl[1])) + '_sc' + \
                str(int(self.initTpl[2]))
            np.save(self.basename + filename, self.counts)
            print "successfully exported histogram counts as: \n" + \
                self.basename + filename
        else:
            return

    def updateLabels(self, val):
        """Update labels in volume histogram with slider."""
        if self.segmType == 'ncut':
            self.labelNr = self.sLabelNr.val
        else:
            return

    def imaMaskIncr(self, incr):
        """Update transparency of image mask by increment."""
        self.TranspVal = self.sImaMaskTrans.val
        if (self.TranspVal + incr >= 0) & (self.TranspVal + incr <= 1):
            self.TranspVal += incr
        self.imaMaskH.set_alpha(self.TranspVal)
        self.sImaMaskTrans.set_val(self.TranspVal)
        self.updateMsks()

    def imaMaskTransS(self, val):
        """Update transparency of image mask with slider."""
        self.TranspVal = self.sImaMaskTrans.val
        self.imaMaskH.set_alpha(self.TranspVal)
        # set imaMaskSwitchCount to 0 for smooth integration of slider + button
        self.imaMaskSwitchCount = 0
        self.updateMsks()

    def imaMaskTransSwitch(self):
        """Update transparency of image mask to toggle transparency of it."""
        self.imaMaskSwitchCount = (self.imaMaskSwitchCount+1) % 2
        if self.imaMaskSwitchCount == 1:  # set imaMask transp
            self.TranspVal = 0
        else:  # set imaMask opaque
            self.TranspVal = self.sImaMaskTrans.val
        self.imaMaskH.set_alpha(self.TranspVal)
        self.updateMsks()

    def updateLabelsRadio(self, val):
        """Update labels with radio buttons."""
        labelScale = self.lMax / 6.  # nr of non-zero radio buttons
        self.labelNr = int(float(val) * labelScale)

    def labelContours(self):
        """
        Calculate and plot political borders.

        Used in ncut version.
        """
        grad = np.gradient(self.volHistMask)
        self.pltMap = np.greater(np.sqrt(np.power(grad[0], 2) +
                                         np.power(grad[1], 2)), 0)
        self.pltMapH.set_data(self.pltMap)
        self.pltMapH.set_extent((0, self.nrBins, self.nrBins, 0))

    def lassoArr(self, array, indices):
        """Lasso related."""
        lin = np.arange(array.size)
        newArray = array.flatten()
        newArray[lin[indices]] = 1
        return newArray.reshape(array.shape)
