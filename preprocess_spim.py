#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:06:08 2018

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, filters

#dirname = '/data/Movies/Liberali/07-26-2018 FUCCI lentiviral'
#channels = ['Channel 0','Channel 1']
#basename = 'c0_t58_stack.tif'
#
#Z = 76
#T = 189

# Load test file
filename = '/data/Movies/Liberali/07-26-2018 FUCCI lentiviral/c0_t58_stack.tif'
im = io.imread(filename)
#truncate for test image
im = im[10:50,...]
[Z,X,Y] = im.shape

# Compute global Otsu threshold on the maximum intensity projection
#total_int = dapi.sum( axis = 0 )
global_thresh = filters.threshold_otsu(im)
mask3D = im > global_thresh

mask3D_adaptive = np.zeros(im.shape)
for z in xrange(Z):
    mask3D_adaptive[z,...] = filters.threshold_local(im[z,...], 21)




# Iterate through 