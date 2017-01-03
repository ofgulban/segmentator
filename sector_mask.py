"""Sector mask to mask stuff in volume histogram."""

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
#
# This code is taken from user 'ali_m' from StackOverflow:
# <http://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array>

import numpy as np


class sector_mask:
    """A shape with useful parameters to detect some tissues quickly."""

    def __init__(self, shape, centre, radius, angle_range):
        self.radius = radius
        self.shape = shape
        self.x, self.y = np.ogrid[:shape[0], :shape[1]]
        self.cx, self.cy = centre
        self.tmin, self.tmax = np.deg2rad(angle_range)
        # ensure stop angle > start angle
        if self.tmax < self.tmin:
            self.tmax += 2*np.pi
        # convert cartesian --> polar coordinates
        self.r2 = (self.x-self.cx)*(self.x-self.cx) + (
            self.y-self.cy)*(self.y-self.cy)
        self.theta = np.arctan2(self.x-self.cx, self.y-self.cy) - self.tmin
        # wrap angles between 0 and 2*pi
        self.theta %= (2*np.pi)

    def set_polCrd(self):
        """Convert cartesian to polar coordinates."""
        self.r2 = (self.x-self.cx)*(self.x-self.cx) + (
            self.y-self.cy)*(self.y-self.cy)
        self.theta = np.arctan2(self.x-self.cx, self.y-self.cy) - self.tmin
        # wrap angles between 0 and 2*pi
        self.theta %= (2*np.pi)

    def set_x(self, x):
        """Set x axis value."""
        self.cx = x
        self.set_polCrd()  # update polar coordinates

    def set_y(self, y):
        """Set y axis value."""
        self.cy = y
        self.set_polCrd()  # update polar coordinates

    def set_r(self, radius):
        """Set radius of the circle."""
        self.radius = radius

    def scale_r(self, scale):
        """Scale (multiply) the radius."""
        self.radius = self.radius * scale

    def rotate(self, degree):
        """Rotate shape."""
        rad = np.deg2rad(degree)
        self.tmin += rad
        self.tmax += rad
        self.set_polCrd()  # update polar coordinates

    def updateThetaMin(self, degree):
        """There is another updateThetaMin in segmentator_functions.py.

        Why?
        """
        rad = np.deg2rad(degree)
        self.tmin = rad
        # ensure stop angle > start angle
        if self.tmax <= self.tmin:
            self.tmax += 2*np.pi
        # ensure stop angle- 2*np.pi NOT > start angle
        if self.tmax - 2*np.pi >= self.tmin:
            self.tmax -= 2*np.pi
        # update polar coordinates
        self.set_polCrd()

    def updateThetaMax(self, degree):
        """There is another updateThetaMax in segmentator_functions.py.

        Why?
        """
        rad = np.deg2rad(degree)
        self.tmax = rad
        # ensure stop angle > start angle
        if self.tmax <= self.tmin:
            self.tmax += 2*np.pi
        # ensure stop angle- 2*np.pi NOT > start angle
        if self.tmax - 2*np.pi >= self.tmin:
            self.tmax -= 2*np.pi
        # update polar coordinates
        self.set_polCrd()

    def binaryMask(self):
        """Return a boolean mask for a circular sector."""
        # circular mask
        self.circmask = self.r2 <= self.radius*self.radius
        # angular mask
        self.anglemask = self.theta <= (self.tmax-self.tmin)
        # return binary mask
        return self.circmask*self.anglemask

    def contains(self, event):
        """Check if a cursor pointer is inside the sector mask."""
        xbin = np.floor(event.xdata)
        ybin = np.floor(event.ydata)
        Mask = self.binaryMask()
        # the next line doesn't follow pep 8 (otherwise it fails)
        if Mask[ybin][xbin] is True:  # switch x and ybin, volHistMask not Cart
            return True
        else:
            return False

    def draw(self, ax, cmap='Reds', alpha=0.2, vmin=0.1,
             interpolation='nearest', origin='lower', extent=[0, 100, 0, 100]):
        """Draw stuff."""
        BinMask = self.binaryMask()
        FigObj = ax.imshow(
            BinMask,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            interpolation=interpolation,
            origin=origin,
            extent=extent)
        return (FigObj, BinMask)
