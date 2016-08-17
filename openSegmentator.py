# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:20:00 2016

@author: marian
"""

import sys

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)

path = str(sys.argv[-1])

class openSegmentator:
    def __init__(self, path, **kwargs):
        self.path = path
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                setattr(self, key, value)
        import segmentator_main

test = openSegmentator(path)