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

# This code is taken from user 'ali_m' from StackOverflow:
# <http://stackoverflow.com/questions/18352973/mask-a-circular-sector-in-a-numpy-array>

import numpy as np


class sector_mask:
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
        # convert cartesian --> polar coordinates
        self.r2 = (self.x-self.cx)*(self.x-self.cx) + (
            self.y-self.cy)*(self.y-self.cy)
        self.theta = np.arctan2(self.x-self.cx, self.y-self.cy) - self.tmin
        # wrap angles between 0 and 2*pi
        self.theta %= (2*np.pi)

    def set_x(self, x):
        self.cx = x
        # update polar coordinates
        self.set_polCrd()

    def set_y(self, y):
        self.cy = y
        # update polar coordinates
        self.set_polCrd()

    def set_r(self, radius):
        self.radius = radius

    def scale_r(self, scale):
        self.radius = self.radius * scale
        self.radius = self.radius * scale

    def rotate(self, degree):
        rad = np.deg2rad(degree)
        self.tmin += rad
        self.tmax += rad
        self.set_polCrd()

    def mouthChange(self, degree):
        rad = np.deg2rad(degree)
        self.tmin += rad
        self.tmax -= rad
        # ensure stop angle > start angle
        if self.tmax <= self.tmin:
            self.tmax += 2*np.pi
        # ensure stop angle- 2*np.pi NOT > start angle
        if self.tmax - 2*np.pi >= self.tmin:
            self.tmax -= 2*np.pi
        # update polar coordinates
        self.set_polCrd()

    """
    Define function that returns a boolean mask for a circular sector.
    """
    def binaryMask(self):
        # circular mask
        self.circmask = self.r2 <= self.radius*self.radius
        # angular mask
        self.anglemask = self.theta <= (self.tmax-self.tmin)
        # return binary mask
        return self.circmask*self.anglemask

    """
    Develop logical test to check if a cursor pointer is inside the sector mask
    """
    def contains(self, event):
        xbin = np.floor(event.xdata)
        ybin = np.floor(event.ydata)
        Mask = self.binaryMask()
        # the next line doesn't follow pep 8 (otherwise it fails)
        if Mask[ybin][xbin] == True:  # switch x and ybin, volHistMask not Cart
            return True
        else:
            return False

    def draw(self, ax,
             cmap='Reds', alpha=0.2, vmin=0.1, interpolation='nearest',
             origin='lower', extent=[0, 100, 0, 100]):
        BinMask = self.binaryMask()
        FigObj = ax.imshow(BinMask, cmap=cmap, alpha=alpha, vmin=vmin,
                           interpolation=interpolation, origin=origin,
                           extent=extent)
        return (FigObj, BinMask)
