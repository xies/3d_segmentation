# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:59:44 2016

# OUTPUT: labels - ndarray of the labelled nuclei

@todo: level set preprocessing, 2D stats -> merge/split

@author: xies@stanford.edu
"""

from skimage import io, filters, morphology, measure
import SimpleITK
import matplotlib
import pandas as pd

"""
File
"""
filename = '/Users/mimi/Desktop/test.tif'
um_per_px = 0.1317882
um_per_z = 0.5

"""
PARAMETERS

"""

smooth_size = 25 # pixels
min_radius = 20
max_radius = 100

min_obj_size_2D = 20; # min px size for objects in 2D
min_obj_size_3D = 500;

"""
Image I/O

"""

im_stack = io.imread(filename)
# im_stack = skimage.util.img_as_float(im_stack)
[Z,C,Y,X] = im_stack.shape
rb = im_stack[:,0,:,:]
dapi = im_stack[:,1,:,:]

"""
Preprocessing
`
"""

# Compute global Otsu threshold on the maximum intensity projection
max_int_proj = dapi.max( axis = 0 )
global_thresh = filters.threshold_otsu(dapi)
mask3D = dapi > global_thresh

# Do preliminary topological cleaning
mask_clean = mask_cleanup(mask3D, min_obj_size_2D) # clean up small obj and fill holes
[D, sure_fg,sure_bg,unknown] = find_fg_bg(mask_clean) # get bg/fg
markers = morphology.label(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[unknown > 0] = 0
#markers = remove_small_3d(markers,min_obj_size_3D)

# Initialize
watershedImg = -filters.gaussian(mask_clean,1)
labels = morphology.watershed( watershedImg, markers)

objectIDs = np.setdiff1d( np.unique(labels), [0,1] )

"""
Get statistics
"""

properties = []
columns = ('x','y','z','I','w')
indices = []

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
Plot stats on image
"""

fig, axes = plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows))
ind = 0

for z in range(Z):
    
    plane_props = properties[properties['z'] == z]
    if not plane_props.shape[0]:
        continue
    
    ind += 1
    i = ind // ncols
    j = ind % nrows
    axes[i,j].imshow(labels[z,...], interpolation='nearest',cmap='Dark2')
    axes[i,j].set_xticks([])
    axes[i,j].set_yticks([])
    xlim = axes[i,j].get_xlim()
    ylim = axes[i,j].get_ylim()
    
    axes[i,j].scatter(plane_props['y'],plane_props['x'])
    axes[i,j].set_xlim(xlim)
    axes[i,j].set_ylim(ylim)
    
for ax in axes.ravel():
    if not(len(ax.images)):
        fig.delaxes(ax)

"""
Orthogonal projection
"""

fig = plt.figure(figsize=(12,12))
colors = plt.cm.jet(properties.index.astype(np.int32))

# xy projection
ax_xy = fig.add_subplot(111)
ax_xy.imshow(dapi.max(axis=0) , cmap='gray')
ax_xy.scatter(properties['y'],properties['x'], c=colors, alpha=1)

# xz projection
divider = make_axes_locatable(ax_xy)
ax_zx = divider.append_axes('top',2, pad=0.2, sharex=ax_xy)
ax_zx.imshow(dapi.max(axis=1), aspect=um_per_z/um_per_px, cmap='gray')
ax_zx.scatter(properties['y'],properties['z'], c=colors, alpha=1)

# yz projection
ax_yz = divider.append_axes('right',2, pad=0.2, sharex=ax_xy)
ax_yz.imshow(dapi.max(axis=2).T, aspect=um_per_px/um_per_z, cmap='gray')
ax_yz.scatter(properties['z'],properties['x'], c=colors, alpha=1)

plt.draw()


"""
Preview

"""

print "Total # of final labels: %d " % (objectIDs.size)

io.imsave('labels.tif',
          np.stack((dapi,markers,labels),axis = 1).transpose((0,2,3,1)).astype(np.int16))

