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


class DraggableSector:
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