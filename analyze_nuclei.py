#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:16:45 2016

@author: mimi
"""

dapi_stats = [ sum( dapi[labels == i] ) for i in objectIDs ]

rb_stats = [ sum( rb[labels == i] ) for i in objectIDs ]

fig = plt.figure()
ax1 = fig.add_subplot(221)
plt.hist(dapi_stats)
ax.set_xlabel('Total DAPI fluorescence')
ax.set_ylabel('Count')

ax2 = fig.add_subplot(222)
plt.hist(rb_stats)
ax.set_xlabel('Total RB fluorescence')
ax.set_ylabel('Count')

ax3 = fig.add_subplot(223)
plt.scatter( dapi_stats,rb_stats )
ax.set_xlabel('Total DAPI fluorescence')
ax.set_ylabel('Total RB fluorescence')

ax4 = fig.add_subplot(224)
plt.scatter( stats['num_pixel'],rb_stats )
ax.set_xlabel('# of pixels')
ax.set_ylabel('Total RB fluorescence')

# Remember that 1 is background
obj2display = 21

fig = plt.figure()
#ax = fig.

#edgelist = get_object_edgelist(labels,obj2display, display = True, axis=ax)
# plot_object_surface(edgelist)

