# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:59:44 2016

# OUTPUT: labels - ndarray of the labelled nuclei

@todo: 3D stats -> merge/split (e.g. look at convex image)

@author: xies@stanford.edu
"""

import os
import numpy as np
from skimage import io, filters, morphology, measure, util
from scipy.ndimage import distance_transform_edt
import pandas as pd

"""
File input
"""

#filename = "/Users/mimi/Box Sync/mIOs/Confocal/12-06-2017 mIO IC+CV Day4/DMSO/DAPI Ki67488 EdU594/1/crop.tif"
filename = "/Users/mimi/Box Sync/mIOs/Confocal/SP8/02-07-19 Lgr5GFP mCherryNLS Clone 1/cherry_t1.tif"
um_per_px = 0.0901876
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
[Z,Y,X] = im_stack.shape
nuclei = im_stack
axial_anisotropy = um_per_px/um_per_z

"""
Preprocessing to generate clean 3D mask

"""

# Gaussian filter
#im_stack = filters.gaussian(im_stack,sigma=1)

# Compute global Otsu threshold on the maximum intensity projection
#total_int = dapi.sum( axis = 0 )
global_thresh = filters.threshold_otsu(nuclei) * 5
mask_nuclei = nuclei > global_thresh
#mask_nuclei = filters.threshold_local(nuclei, 21)

mask_roi = np.zeros(nuclei.shape, dtype = bool)
#mask_roi[nuclei>filters.threshold_otsu(nuclei)] = True
mask_roi[nuclei>1000] = True
roi_ilot = morphology.remove_small_objects(mask_roi, 500000)
bbox = ndi.find_objects(roi_ilot)

dapi_roi = dapi[bbox[0]];
#rb_roi = rb[bbox[0]]

# Denoise using chambolle
dapi_float = util.img_as_float(dapi_roi)
dapi_float_tvc = restoration.denoise_tv_chambolle(dapi_float, weight = 0.05)

# Find z-stack w/highest variance
variance = [np.var(image) for _, image in enumerate(mask_dapi)]
idx = np.argmax(variance)
dapi_roi_z = dapi_float_tvc[idx-3:idx+3]
distance_a = ndi.distance_transform_edt(mask_dapi[3])

#Local peak markers
smooth_distance = filters.gaussian(distance_a, sigma=4)
local_maxi = feature.peak_local_max(smooth_distance, indices=True, 
                                    exclude_border=False,footprint=np.ones((15, 15)))
markers_lm = np.zeros(distance_a.shape, dtype=np.int)
markers_lm[local_maxi[:,0].astype(np.int), local_maxi[:,1].astype(np.int)] = np.arange(len(local_maxi[:,0])) + 1
markers_lm = morphology.dilation(markers_lm, morphology.disk(5))
markers_smooth = np.zeros(dapi_roi_z.shape, dtype = np.int)
markers_smooth[3] = markers_lm
markers_smooth = ndi.label(markers_smooth)[0]

#Watershed
dapi_seg = segmentation.watershed(sobel, markers_3D, compactness = 0.001)

# Do preliminary topological cleaning
mask_clean = mask_cleanup(mask3D, min_obj_size_2D) # clean up small obj and fill holes
[D, sure_fg,sure_bg,unknown] = find_fg_bg(mask_clean) # get bg/fg


nucleus_mask = np.copy(mask_dapi)
nucleus_mask[nucleus_mask>0] = 1
nucleus_mask = nucleus_mask.astype('bool')
labeled = measure.label(dapi_seg)
labeled_ext = np.copy(labeled)
labeled[~nucleus_mask] = 0
nucleus = measure.label(labeled)

plt.imshow(label2rgb(nucleus[3], bg_label=0))


"""
Load manual seeds

"""

filename = "/Users/mimi/Box Sync/mIOs/Confocal/09-13-2017 mIO IC CHIR VA Day 4 DAPI phosphoRB EdU/500nM_PD/500nM_PD_3/seeds.txt"
seeds = pd.read_csv(filename,delimiter = '\t')

x_coord = np.round(seeds['X'])
y_coord = np.round(seeds['Y'])
z_coord = np.round(seeds['Slice'])
seed_im = np.zeros((39,209,209))
for i,x in enumerate(x_coord):
    y = y_coord[i]
    z = z_coord[i]
    seed_im[z,y,x] = 255
    
io.imsave( ''.join( (os.path.dirname(filename), '/seeds.tif') ), seed_im.astype(np.int8))


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

io.imsave(''.join( (os.path.dirname(filename), '/'
          'raw_labels.tif') ),
          np.stack((util.img_as_uint(dapi),mask_clean,labels),
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
