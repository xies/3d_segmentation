# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:59:44 2016

# OUTPUT: labels - ndarray of the labelled nuclei

@todo: level set preprocessing, 2D stats -> merge/split

@author: xies@stanford.edu
"""

from skimage import io, filters, morphology, measure, util
from scipy.ndimage import distance_transform_edt
import pandas as pd

"""
File
"""
filename = '/Users/mimi/Desktop/int_org/organoid_1.tif'
um_per_px = 0.1317882
um_per_z = 0.5

"""
PARAMETERS
"""

smooth_size = 25 # pixels
min_radius = 20
max_radius = 100

min_obj_size_2D = 500; # min px size for objects in 2D
min_obj_size_3D = 1000;

"""
Image I/O
"""

im_stack = io.imread(filename)

im_stack = util.img_as_float(im_stack)
[Z,C,Y,X] = im_stack.shape
rb = im_stack[:,0,:,:]
dapi = im_stack[:,1,:,:]
axial_anisotropy = um_per_px/um_per_z

"""
Preprocessing to generate clean 3D mask

"""

# Compute global Otsu threshold on the maximum intensity projection
max_int_proj = dapi.max( axis = 0 )
global_thresh = filters.threshold_otsu(dapi)
mask3D = dapi > global_thresh

# Do preliminary topological cleaning
mask_clean = mask_cleanup(mask3D, min_obj_size_2D) # clean up small obj and fill holes
[D, sure_fg,sure_bg,unknown] = find_fg_bg(mask_clean) # get bg/fg

"""
Watershed
1) Get distance transform (Euclidean)
2) Generate foreground markers (object markers) from the local maxima of dist
transform; markers no closer than specified threshold (typically 10 px). Dilate
markers for easy visualization
4) Mark background as 0
5) Perform watershed

"""

# get Distance transform of cleaned up mask (euclidean)
distTransform = distance_transform_edt(mask_clean)
watershedImg = -distTransform / distTransform.max() + 1

## Get local maxima from bwdist image and filter 
#I = filters.gaussian(distTransform,4)
#I = feature.peak_local_max(I, min_distance=20,indices=False)
#for i in range(5):
#    I = morphology.dilation(I)

markers = measure.label(sure_fg)
#markers += 1
#markers[sure_bg] = 0

# Perform watershed
labels = morphology.watershed( watershedImg, markers)
objectIDs = np.setdiff1d( np.unique(labels), [0,1] )

"""
Get statistics in 3D
"""

properties = measure.regionprops(labels, dapi) # 3D regionprops (some properties are not supported yet)
columns = ('x','y','z','I','w')



for z, frame in enumerate(labels):
    f_prop = measure.regionprops(frame.astype(np.int),
                intensity_image=dapi[z,:,:])
    for d in f_prop:
        radius = (d.area / np.pi)**0.5
        if (min_radius < radius < max_radius):
            properties.append([d.weighted_centroid[0],
                              d.weighted_centroid[1],
                              z, d.mean_intensity * d.area,
                              radius])
            indices.append(d.label)

if not len(indices):
    all_props = pd.DataFrame([],index=[])
indices = pd.Index(indices, name='label')
properties = pd.DataFrame(properties, index=indices, columns=columns)
properties['I'] /= properties['I'].max()

"""
Orthogonal projection
"""
plot_stack_projections(labels)

"""
Preview
u
"""

print "Total # of final labels: %d " % (objectIDs.size)

io.imsave('labels.tif',
          np.stack((util.img_as_uint(dapi),markers,labeled),
                   axis = 1).transpose((0,2,3,1)).astype(np.int16))

