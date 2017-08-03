# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:59:44 2016

# OUTPUT: labels - ndarray of the labelled nuclei

@todo: 3D stats -> merge/split (e.g. look at convex image)

@author: xies@stanford.edu
"""

import numpy as np
from skimage import io, filters, morphology, measure, util
from scipy.ndimage import distance_transform_edt
import pandas as pd

"""
File
"""



#filename = '/Users/mimi/Desktop/mIOs/400nM Palblociclib 04-11-2017/400nM Palblociclib/DAPI pRB EdU/3/crop.tif'
filename = "/Users/mimi/Desktop/mIOs DMSO Pablociclib 10 100 400nM 05-06-2017/TIFF/400nM/mIOs_Pabl_400nM_46hr_10.tif"
um_per_px = 0.1944334
um_per_z = 1

"""
PARAMETERS
"""

smooth_size = 20 # pixels
min_radius = 10
max_radius = 80

min_obj_size_2D = 200; # min px size for objects in 2D
min_obj_size_3D = 500;

"""
Image I/O
"""

im_stack = io.imread(filename)

#im_stack = util.img_as_float(im_stack)
[Z,Y,X,C] = im_stack.shape
rb = im_stack[:,:,:,1]
dapi = im_stack[:,:,:,0]
edu = im_stack[:,:,:,2]
axial_anisotropy = um_per_px/um_per_z

"""
Preprocessing to generate clean 3D mask

"""

# Gaussian filter
im_stack = filters.gaussian(im_stack,sigma=1)

# Compute global Otsu threshold on the maximum intensity projection
#total_int = dapi.sum( axis = 0 )
global_thresh = filters.threshold_otsu(dapi)
mask3D = dapi > global_thresh
#mask3D = filters.threshold_adaptive(dapi, 21)

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

labeled = labels
labeled[~mask_clean.astype(bool)] = 0

"""
Write to TIF file
"""

print "Total # of final labels: %d " % (objectIDs.size)

io.imsave('raw_labels.tif',
          np.stack((util.img_as_uint(dapi),markers,labels),
                   axis = 1).transpose((0,2,3,1)).astype(np.int16))

"""
Get statistics in 3D
"""

dapiProps = measure.regionprops(labels, dapi) # 3D regionprops (some properties are not supported yet)
#rbProps = measure.regionprops(labels,rb)
columns = ('volume','radius','mean_dapi')

object_properties = []
indices = []

for z, d in enumerate(dapiProps):
    volume = d.area * (um_per_px)**2 * um_per_z # in um^3
    radius = 3 * (float(volume))**(1.0/3) / np.pi / 4
    object_properties.append([volume, radius,
                       d.mean_intensity,
                       ])
    indices.append(d.label)

if not len(indices):
    all_props = pd.DataFrame([],index=[])
    
indices = pd.Index(indices, name='objectID')
object_properties = pd.DataFrame(object_properties, index=indices, columns=columns)

"""
Write properties to Excel file

"""

writer = pd.ExcelWriter('properties.xlsx')
object_properties.to_excel(writer,'Sheet1')
writer.save()

"""
Preview items
"""
#plot_stack_projections(labels)
visualize_objectID(labels,2)

"""
Read in the annotated Excel file
"""

nuclei = pd.ExcelFile("int_org/organoid_1.xlsx")
nuclei = nuclei.parse("Sheet1")

good_nuclei = nuclei[nuclei['segmentation_status'] == 'good']
bad_objectIDs = setdiff1d(unique(labels),good_nuclei['objectID'])

# Generate a mask of the "good" segmentation objects for use in visualization
good_mask = labels
for ID in bad_objectIDs:
    good_mask[good_mask == ID] = 0

io.imsave('good_labels.tif',
      np.stack((util.img_as_uint(dapi),markers,good_mask),
               axis = 1).transpose((0,2,3,1)).astype(np.int16))
