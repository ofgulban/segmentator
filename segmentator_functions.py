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


import numpy as np
import matplotlib.pyplot as plt
from utils import VolHist2ImaMapping
from nibabel import save, Nifti1Image
import config as cfg


class MakeFigureInteractive:
    def __init__(self, **kwargs):
        self.press = None
        self.ctrlHeld = False
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                setattr(self, key, value)

    def update(self):  # determine what should happen during an update
        # update volHistMask
        self.volHistMask = self.sectorObj.binaryMask()
        self.volHistMaskHandle.set_data(self.volHistMask)
        # update imaMask
        self.imaMask = VolHist2ImaMapping(
            self.invHistVolume[:, :, self.sliceNr],
            self.volHistMask)
        self.imaMaskHandle.set_data(self.imaMask)

    def connect(self):  # this will make the object responsive
        'connect to all the events we need'
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
        if event.key == 'control':
            self.ctrlHeld = True

    def on_key_release(self, event):
        if event.key == 'control':
            self.ctrlHeld = False

    def on_press(self, event):
        if event.button == 1:  # left button
            'on left button press, check if mouse is in fig and on Sector'
            if event.inaxes == self.axes:
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
            elif event.inaxes == self.axes2:
                print "Subplot 2: x and y pos"
                print event.xdata, event.ydata
                self.press = event.xdata, event.ydata
                xvoxel = np.floor(event.xdata)
                yvoxel = np.floor(event.ydata)
                # SWITCH x and y voxel to get linear index since NOT Cartes.!!!
                pixelLin = self.invHistVolume[
                    yvoxel, xvoxel, self.sliceNr]
                # ind2sub
                xpix = (pixelLin / self.nrOfBins)
                ypix = (pixelLin % self.nrOfBins)
                # SWITCH x and y for circle centre since back TO Cartesian!!!
                self.circle1 = plt.Circle(
                    (ypix, xpix), radius=5, color='b')
                self.axes.add_artist(self.circle1)
                self.figure.canvas.draw()
            else:
                return

        elif event.button == 2:  # scroll button
            'on scroll button press, check if mouse is in fig'
            if event.inaxes != self.axes:
                return
            if self.ctrlHeld is False:  # ctrl no
                self.sectorObj.scale_r(1.05)
                # update
                self.update()
                # draw to canvas
                self.figure.canvas.draw()
            elif self.ctrlHeld is True:  # ctrl yes
                self.sectorObj.rotate(10.0)
                # update
                self.update()
                # draw to canvas
                self.figure.canvas.draw()

        elif event.button == 3:  # right button
            'on right button press, check if mouse is in fig'
            if event.inaxes != self.axes:
                return
            if self.ctrlHeld is False:  # ctrl no
                self.sectorObj.scale_r(0.95)
                # update
                self.update()
                # draw to canvas
                self.figure.canvas.draw()
            elif self.ctrlHeld is True:  # ctrl yes
                self.sectorObj.rotate(-10.0)
                # update
                self.update()
                # draw to canvas
                self.figure.canvas.draw()

    def on_motion(self, event):
        'on motion, check if...'
        # ... button is pressed
        if self.press is None:
            return
        # ... cursor is in figure
        if event.inaxes != self.axes:
            return
        # get former sector centre x and y positions, cursor x and y positions
        x0, y0, xpress, ypress = self.press
        # calculate difference betw cursor pos on click and new pos dur motion
        dy = event.xdata - xpress  # switch x0 & y0 cause volHistMask not Cart
        dx = event.ydata - ypress  # switch x0 & y0 cause volHistMask not Cart

        # update x and y position of sector, based on past motion of cursor
        self.sectorObj.set_x(x0+dx)
        self.sectorObj.set_y(y0+dy)

        # update
        self.update()
        # draw to canvas
        self.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        try:
            self.circle1.remove()
        except:
            return
        self.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.figure.canvas.mpl_disconnect(self.cidpress)
        self.figure.canvas.mpl_disconnect(self.cidrelease)
        self.figure.canvas.mpl_disconnect(self.cidmotion)

    def updateColorBar(self, val):
        # update slider for scaling log colorbar in 2D hist
        histVMax = np.power(10, self.sHistC.val)
        plt.clim(vmax=histVMax)

    def updateImaBrowser(self, val):
        # Scale slider value [0,1) to dimension index to allow var in shape
        self.sliceNr = int(self.sSliceNr.val*self.orig.shape[2])
        self.slc.set_data(self.orig[:, :, self.sliceNr])
        self.slc.set_extent((0, self.orig.shape[1], self.orig.shape[0], 0))
        # update
        self.update()
        # set extent
        self.imaMaskHandle.set_extent(
            (0, self.imaMask.shape[1],
             self.imaMask.shape[0], 0))
        # draw to canvas
        self.figure.canvas.draw()

    def updateTheta(self, val):
        # get current theta value from slider
        thetaVal = self.sTheta.val
        # update mouth of sector mask by difference
        diff = thetaVal-self.thetaInit
        self.sectorObj.mouthChange(diff)
        # adjust thetaInit
        self.thetaInit = thetaVal
        # update
        self.update()
        # draw to canvas
        self.figure.canvas.draw()

    def cycleView(self, event):
        self.cycleCount = (self.cycleCount+1) % 3
        # transpose data
        self.orig = np.transpose(self.orig, (2, 0, 1))
        # transpose ima2volHistMap
        self.invHistVolume = np.transpose(
            self.invHistVolume, (2, 0, 1))
        # update slice number
        self.sliceNr = int(self.sSliceNr.val*self.orig.shape[2])
        # plot new data
        self.slc.set_data(self.orig[:, :, self.sliceNr])
        self.slc.set_extent((0, self.orig.shape[1], self.orig.shape[0], 0))
        # update
        self.update()
        # set extent
        self.imaMaskHandle.set_extent(
            (0, self.imaMask.shape[1], self.imaMask.shape[0], 0))
        # draw to canvas
        self.figure.canvas.draw()

    def exportNifti(self, event):
        # put the permuted indices back to their original format
        cycBackPerm = (self.cycleCount, (self.cycleCount+1) % 3,
                       (self.cycleCount+2) % 3)
        self.orig = np.transpose(self.orig, cycBackPerm)
        self.invHistVolume = np.transpose(self.invHistVolume, cycBackPerm)
        self.cycleCount = 0
        # get linear indices
        linIndices = np.arange(0, self.nrOfBins*self.nrOfBins)
        idxMask = linIndices[self.volHistMask.flatten()]
        # return logical array with length equal to nr of voxels
        voxMask = np.in1d(self.invHistVolume.flatten(), idxMask)
        # reset mask and apply logical indexing
        mask3D = np.zeros(self.invHistVolume.flatten().shape)
        mask3D[voxMask] = 1
        mask3D = mask3D.reshape(self.invHistVolume.shape)
        # save image, check whether nii or nii.gz
        new_image = Nifti1Image(
            mask3D, header=self.nii.get_header(), affine=self.nii.get_affine())
        if self.nii.get_filename()[-4:] == '.nii':
            save(new_image, self.nii.get_filename()[:-4]+'_OUT.nii.gz')
        elif self.nii.get_filename()[-7:] == '.nii.gz':
            save(new_image, self.nii.get_filename()[:-7]+'_OUT.nii.gz')

    def resetGlobal(self, event):
        # reset color bar
        self.sHistC.reset()
        # reset ima browser slider
        self.sSliceNr.reset()
        # reset slice number
        self.sliceNr = int(self.sSliceNr.val*self.orig.shape[2])
        self.slc.set_data(self.orig[:, :, self.sliceNr])
        self.slc.set_extent((0, self.orig.shape[1], self.orig.shape[0], 0))
        # reset theta
        self.sTheta.reset()
        # reset values for mask
        self.sectorObj.set_x(cfg.init_centre[0])
        self.sectorObj.set_y(cfg.init_centre[1])
        self.sectorObj.set_r(cfg.init_radius)
        self.sectorObj.tmin, self.sectorObj.tmax = np.deg2rad(
            cfg.init_theta)
        # update
        self.update()
        self.imaMaskHandle.set_extent((0, self.imaMask.shape[1],
                                       self.imaMask.shape[0], 0))
        # draw to canvas
        self.figure.canvas.draw()
