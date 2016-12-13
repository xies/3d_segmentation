# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:59:44 2016

@todo: level set preprocessing, watershed
@author: xies@stanford.edu
"""

import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage import io,filters,morphology
import matplotlib.pyplot as plt

"""
PARAMETERS
"""
filename = '/Users/mimi/Desktop/test.tif'
min_obj_size_2D = 20; # min px size for objects in 2D
min_obj_size_3D = 1000;


"""
Image I/O
"""
im_stack = io.imread(filename)
[Z,C,Y,X] = im_stack.shape
rb = im_stack[:,0,:,:]
dapi = im_stack[:,1,:,:]


"""
Preprocessing
"""

global_thresh = filters.threshold_otsu(dapi)
mask3D = dapi > global_thresh



mask3D = close_and_remove_small_obj(mask3D, min_obj_size_2D)
labels = morphology.label(mask3D)
labels = remove_small_3d(labels,min_obj_size_3D)

"""
Preview
"""
obj2display = 4
edgelist = get_object_edgelist(labels,obj2display, display = True)
plot_object_surface(edgelist)


