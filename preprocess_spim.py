#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:06:08 2018

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, filters, morphology, segmentation, feature, measure
from scipy import ndimage as ndi
from os import path

# Load test file
dirname = '/data/Movies/Liberali/07-26-2018 FUCCI lentiviral/test/'
fname = 't58.tif'
im = io.imread(path.join(dirname,fname))
#Crop test image
im = im[11:43,:,380:380+320,400:400+320]
[Z,C,X,Y] = im.shape
# Save test image
io.imsave(path.join(dirname,'t58_test.tif'),im.astype(np.int8))
io.imsave(path.join(dirname,'t58_inv.tif'),-im[:,1,:,:].astype(np.int8))

# Load manual markers
markers_csv = pd.read_csv(path.join(dirname,'markers.csv'))
markers = np.zeros([Z,X,Y])
for i in range(markers_csv.shape[0]):
    markers[ markers_csv['Slice'][i],markers_csv['Y'][i],markers_csv['X'][i] ] = 255
io.imsave(path.join(dirname,'markers.tif'),markers.astype(np.int8))
markers = morphology.label(markers)

# Load Ilastik probability
fname = 'nuc_label.tiff'
prob = io.imread(path.join(dirname,fname))

# Load Ilastik segmentation mask
fname = 't58_test_Simple Segmentation.tiff'
mask = io.imread(path.join(dirname,fname))
mask = mask > 1

mask_clean = mask_cleanup(mask, 50) # clean up small obj and fill holes
io.imsave(path.join(dirname,'mask_clean.tif'),mask_clean.astype(np.int8))

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
distTransform = ndi.morphology.distance_transform_edt(mask_clean)

labels_randwalk = segmentation.random_walker(prob,markers)
labels_water = segmentation.watershed(-prob,markers)
labels_water = labels_water * mask_clean

io.imsave('/data/Movies/Liberali/07-26-2018 FUCCI lentiviral/c0_t58_fg.tiff',foreground.astype(np.int8))
io.imsave('/data/Movies/Liberali/07-26-2018 FUCCI lentiviral/c0_t58_labels_water.tiff',labels_water.astype(np.int8))
io.imsave('/data/Movies/Liberali/07-26-2018 FUCCI lentiviral/c0_t58_labels_randwalk.tiff',labels_randwalk)

#============================================================================
# Iterate through 2D methods

im = io.imread(path.join(dirname,'t58_c1_gauss.tif'))
labels = np.zeros((Z,Y,X))
max_count = 0
for z in range(Z):
    zim = im[z,...]
    threshold = filters.threshold_otsu(zim)
    mask = zim > threshold
    mask = mask_cleanup(mask,50)
    
    # Find distance transform of 2D mask and use local maxima finding for auto-seeding watershed
    D = ndi.morphology.distance_transform_bf(mask)
    fgD = morphology.label(feature.peak_local_max(D,min_distance=10,indices=False))
    lD = segmentation.watershed(-D,fgD)
    
    labels[z,...] = ((lD + max_count) * mask).astype(np.int)
    
    # Keep running tally of the labels already used
    max_count = max_count + lD.max()
    
#    plt.subplot(121)
#    io.imshow(zim)
#    plt.subplot(122)
#    io.imshow(lD * mask)

    io.imsave(path.join(dirname,''.join(('2d/label_z',str(z),'.tif'))), (lD * mask).astype(np.int8))

# Go through each slice and make

# Use measure.regionprops to get statistics on all labelled objects
labels = labels.astype(np.int)    
columns = ['Area','Cx','Cy','Z','label','Mean intensity','Euler number','eccentricity','stitched_label']

object_properties = []
for z in range(Z):
    propsz = measure.regionprops(labels[z,...],im[z,...])
    for p in propsz:
        object_properties.append([p.area,p.centroid[0],p.centroid[1], z, p.label,
                                  p.mean_intensity,p.euler_number, p.eccentricity,p.label] )
# Convert to dataframes
props = pd.DataFrame(data=object_properties,columns = columns)

# Go through each slice and group all the objects within 10px of each other's centroid
props['stiched_label'] = props['label']
new_labels = labels.copy()
for z in range(Z-1):
    # Doing this as nested loops for clarity    
    this_props = props[props['Z'] == z]
    next_props = props[props['Z'] == z+1]
    for i in this_props.index:
        thisp = this_props[this_props.index == i]
        this_mask = labels[z,...] == i+1
        
        for j in next_props.index:
            nextp = next_props[next_props.index == j]
            next_mask = labels[z+1,...] == j+1
            
            # Look for overlap
            if ( (next_mask & this_mask).sum().astype(np.float) / this_mask.sum().astype(np.float) > .5 ) or \
                ( (next_mask & this_mask).sum().astype(np.float) / next_mask.sum().astype(np.float) > .5):
                
                dd = np.sqrt((thisp['Cx'].tolist()[0] - nextp['Cx'].tolist()[0])**2
                             + (thisp['Cy'].tolist()[0] - nextp['Cy'].tolist()[0])**2)
                if dd < 50: # within centroid threshold
                    II = np.unique(new_labels[z,this_mask])[0]
                    print j, ' >>> ', II
                    new_labels[z+1,next_mask] = II
                    
#                    nl = props['stitched_label']
#                    nl[props.index == j] = II
#                    props['stitched_label'] = nl
       
io.imsave(path.join(dirname,'2d/stitched_labels.tif'),new_labels.astype(np.int8))

# Detect objects that are 




