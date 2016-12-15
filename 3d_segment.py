# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:59:44 2016

@todo: level set preprocessing, watershed
@author: xies@stanford.edu
"""

import skimage
from skimage import io, filters, morphology
import matplotlib

"""
PARAMETERS

"""
filename = '/Users/mimi/Desktop/test.tif'
min_obj_size_2D = 20; # min px size for objects in 2D
min_obj_size_3D = 500;


"""
Image I/O

"""

im_stack = io.imread(filename).astype(np.float)
im_stack = skimage.util.img_as_float(im_stack)
[Z,C,Y,X] = im_stack.shape
rb = im_stack[:,0,:,:]
dapi = im_stack[:,1,:,:]

"""
Preprocessing

"""

global_thresh = filters.threshold_otsu(dapi)
mask3D = dapi > global_thresh

mask_clean = mask_cleanup(mask3D, min_obj_size_2D) # clean up small obj and fill holes
[D, sure_fg,sure_bg,unknown] = find_fg_bg(mask_clean) # get bg/fg
markers = morphology.label(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown > 0] = 0
#markers = remove_small_3d(markers,min_obj_size_3D)

watershedImg = -filters.gaussian_filter(mask_clean,1)

labels = morphology.watershed( watershedImg, markers)

"""
Preview

"""

print "Total # of final labels: %d " % (np.unique(labels).size - 1)

obj2display = 3

edgelist = get_object_edgelist(labels,obj2display, display = True)
plot_object_surface(edgelist)

io.imsave('labels.tif',np.stack((dapi,markers,labels),axis = 1).transpose((0,2,3,1)).astype(np.int16))

