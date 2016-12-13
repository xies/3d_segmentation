#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:02:21 2016

@author: mimi
"""

import numpy as np
import scipy as sp
import scipy.ndimage
from skimage import io,filters,morphology
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

def mask_cleanup(mask_stack,min_obj_size):
    """
    mask_cleanup(im_stack,min_obj_size)
    
    Uses the following (in order) morphological operations on a stack of binary
    images to generate a 3D stack of masks:
        1) Open
        2) close
    
    INPUT
    ---
    mask_stack : ndarray
        stack of masks arranged T,C,Y,X as output by skimage.io.imread
    min_obj_size : int
        threshold size below which all 'objects' will be removed
        
    OUTPUT
    ---
    mask3D : ndarray
        stack of boolian cleaned up masks
    """
    
    [Z,Y,X] = mask_stack.shape;
    mask3D = np.ndarray([Y,X,Z]) # preallocate
    
    for z in range(0,Z):
        
        mask = mask_stack[z,:,:] 
        # Close all 'open' objects
        mask_closed = morphology.binary_closing(mask)
        # Fill holes contained topologically inside object
        mask_closed = scipy.ndimage.binary_fill_holes(mask_closed)

        # Remove objects that are too small
        mask3D[:,:,z] = morphology.remove_small_objects(mask_closed,min_obj_size)
                
    return mask3D
        
def region_statistics(labels):
    """
    region_statistics(labels)
    
    Returns the number of pixels, number of z-slices, and label # of all
    objects found in input labelled 3D image
    
    """
    all_labels = np.unique(labels);
    # Get rid of 0 as label
    all_labels = all_labels[all_labels != 0]
    
    # Use NDarray to contain a defined tuple
    stats = np.zeros(len(all_labels) , dtype = [('num_pixel',int),
                                                ('num_z_slice',int),
                                                ('label',int)])
    
    idx = 0;
    for l in all_labels:
        # Find statistics
        num_pixel = np.sum(labels == l)
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
    labels2remove = stats[stats['num_pixel'] < min_obj_size_3D]['label']
    
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
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XX,YY,ZZ)
    
    plt.show()
    
    print XX.shape
    return np.column_stack((XX,YY,ZZ))
    
    
def plot_object_surface(X):
        
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
        
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(X[:,0], X[:,1], X[:,2], triangles=tris.triangles, cmap=plt.cm.bone)
        plt.show()
            
    