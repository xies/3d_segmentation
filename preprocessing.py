#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:02:21 2016

@author: mimi
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from skimage import feature, morphology, filters
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier

def blob_log_3d(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,
             overlap=.5, log_scale=False):
    """Finds blobs in the given 3D grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    """
    
    sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    # s**2 provides scale invariance
    gl_images = [-sp.ndimage.gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
    image_cube = np.stack(gl_images,axis=3)

    local_maxima = feature.peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3, 3, 3, 3)),
                                  threshold_rel=0.0,
                                  exclude_border=False)

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    
    # Convert the last index to its corresponding scale value
    lm[:, 3] = sigma_list[local_maxima[:, 3]]
    local_maxima = lm
    return local_maxima
#    return feature.blob._prune_blobs(local_maxima, overlap)


def max_scale_lap_of_gauss(im,smin,smax,axial_anisotropy=1):
    
    Imax = np.zeros(im.shape).astype(np.float64)
    scales = np.zeros(im.shape).astype(np.float64)
    im2 = im.copy() / im.max()
    
    smin = int(smin); smax = int(smax)
    for s in range(smin,smax+1):
        I = -sp.ndimage.gaussian_laplace(im2,
                                         np.multiply([axial_anisotropy,1,1],s))
        ind = np.argmax(np.stack((Imax,I)),axis=0)
        Imax = np.max(np.stack((Imax,I)),axis=0)
        
        scales[ind == 0] = s
    
    return (I,scales)

def mask_cleanup(mask_stack,min_obj_size):
    """
    mask_cleanup(im_stack,min_obj_size)
    
    Uses the following (in order) 2D morphological operations on a stack of
    binary images to generate a 3D stack of cleaned masks:
        1) Close holes
        2) Fill in holes
    
    PARAMETERS
    ----------
    mask_stack : ndarray
        stack of masks arranged T,C,Y,X as output by skimage.io.imread
    min_obj_size : int
        threshold size below which all 'objects' will be removed
        
    RETURNS
    -------
    mask3D : ndarray
        stack of boolian cleaned up masks
    """
    
    [Z,Y,X] = mask_stack.shape;
    mask3D = np.ndarray([Z,Y,X]) # preallocate
    
    for z in range(0,Z):
        
        mask = mask_stack[z,:,:] 
        
        # Close all 'pepper' objects and objects connected with single bridge
        mask_closed = mask
        for i in range(0,3):
            mask_closed = morphology.binary_closing(mask_closed)
        # Fill holes contained topologically inside object
        mask_closed = sp.ndimage.binary_fill_holes(mask_closed)

        # Remove objects that are too small
        mask3D[z,:,:] = morphology.remove_small_objects(mask_closed,min_obj_size)
                
    return mask3D
    
def find_fg_bg(mask, fg_thresh = .7):
    """
    """
    
    fg = np.copy(mask)
    bg = np.copy(fg)
    unknown = np.copy(fg)
    
    # Open small objects
    opening = morphology.binary_opening(mask)

    # Erode 5 times
    for i in range(2):
        opening = morphology.binary_erosion(opening)
        
    bg = opening
    # Dilate 6 times to get BG
    for i in range(3):
        bg = morphology.binary_dilation(bg)
        
    # BW distance (3D) to get sure foregrounds
    D = sp.ndimage.distance_transform_edt(opening)
    
    ker = np.ones((5,15,15),dtype=bool)
    D_blur = filters.gaussian(D,8)
    fg = feature.peak_local_max(D_blur,footprint = ker,indices=False)
#    fg = D > fg_thresh * D.max()
#    ker = np.ones((5,25,25),bool)
#    fg = filters.rank.otsu(mask, ker3D)
    
    for i in range(1):
        fg = morphology.binary_dilation(fg)

    unknown = bg.astype(np.int) - fg.astype(np.int)
    bg = np.invert(bg)
    
    return [D,fg,bg,unknown]
    
def region_statistics(labels):
    """
    region_statistics(labels)
    
    Returns the number of pixels, number of z-slices, and label # of all
    objects found in input labelled 3D image
    
    """
    all_labels = np.unique(labels);
    all_labels = np.setdiff1d(all_labels,[0,1])
    
    # Use NDarray to contain a defined tuple
    stats = np.zeros(len(all_labels) , dtype = [('num_pixel',int),
                                                ('num_z_slice',int),
                                                ('label',int)])
    
    idx = 0;
    for l in all_labels:
        # Find statistics
        num_pixel = labels[labels == l].size
        num_z_slice = np.sum(np.any(labels ==l,(1,2)))
        
        stats[idx] = (num_pixel,num_z_slice,l)
        idx += 1 # Advance counter for preallocation
        
    return stats

def remove_small_3d(labels,min_size = 500):
    """
    remove_small_3d
    
    Remove objects from labels list if the total pixel size is smaller than
    threshold
    
    INPUT
    ---
    labels : ndarray
        labeled matrix of objects in 3D
    min_size : int
        default = 500
        
    OUTPUT
    ---
    labels - filtered by size
    """
    # Generate statistics on detected objects and filter them
    stats = region_statistics(labels)
    labels2remove = stats[stats['num_pixel'] < min_size]['label']
    
    for l in labels2remove.tolist():
        labels[ labels == l]  = 0
    
    return labels

def get_object_edgelist(labels, l, display=False):
    """
    get_object_edgelist
    
    Returns the list of 'edge pixel' locations as Nx3 array, where indices
    are X,Y,Z.
    
    INPUT
    ---
    labels : ndarray
        labeled matrix of objects in 3D
    l : int
        the one object label you're interested in
    display : bool
        Defaut = False, Turn on to return plot
        
    
    """
    mask3D = labels == l;
    edges = np.copy(mask3D)
    Z = mask3D.shape[2]
    
    for z in range(0,Z):
        dilated = morphology.dilation(mask3D[:,:,z])
        edges[:,:,z] = np.logical_xor(dilated,mask3D[:,:,z])

    XX = np.ndarray(0,float);
    YY = np.copy(XX); ZZ = np.copy(XX)
    for z in range(Z):
        [X,Y] = np.where(edges[:,:,z])
        if X.size > 0:
            XX = np.append(XX,X.T)
            YY = np.append(YY,Y.T)
            ZZ = np.append(ZZ,np.ones(X.size) * z)
    
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(XX,YY,ZZ)
    
    plt.show()
    
    print XX.shape
    return np.column_stack((XX,YY,ZZ))
    
    
def plot_object_surface(X):
    
    """
    plot_object_surface
    
    Use delaunay triangulation to plot the surface represented by a coordinate
    list of 3D points
    
    """
        
    if X.size > 0:
        #Center about origin
        X -= X.mean(axis=0)
        
        rad = np.linalg.norm(X, axis=1)
        zen = np.arccos(X[:,-1] / rad)
        azi = np.arctan2(X[:,1], X[:,0])
        
        rad = np.linalg.norm(X, axis=1)
        zen = np.arccos(X[:,-1] / rad)
        azi = np.arctan2(X[:,1], X[:,0])
        
        tris = mtri.Triangulation(zen, azi)
        
        fig = plt.gcf()
        ax  = fig.add_subplot(122, projection='3d')
        ax.plot_trisurf(X[:,0], X[:,1], X[:,2], triangles=tris.triangles, cmap=plt.cm.bone)
        plt.show()

        
    