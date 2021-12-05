#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:26:14 2021

@author: dan
"""

def fastCoordInBounds(coord,bounds):
    #seems slower by an order of magnitude
    outVal=False
    if coord[0]>bounds[0,0]:
        if coord[0]<bounds[0,1]:
            if coord[1]>bounds[1,0]:
                if coord[1]<bounds[1,1]:
                    if coord[2]>bounds[2,0]:
                        if coord[2]<bounds[2,1]:
                            outVal=True
    
    return outVal

def fastCoordInBounds(coord, bounds):
    yield bounds[0, 0] < coord[0] < bounds[0, 1]
    yield bounds[1, 0] < coord[1] < bounds[1, 1]
    yield bounds[2, 0] < coord[2] < bounds[2, 1]
    

def complexStreamlinesIntersect(streamlines, maskNifti):
        import dipy.tracking.utils as ut
        import copy
        import numpy as np
        voxelInds=np.asarray(np.where(maskNifti.get_fdata())).T
        doubleResAffine=np.copy(maskNifti.affine)
        doubleResAffine[0:3,0:3]=doubleResAffine[0:3,0:3]*.5
        lin_T, offset =ut._mapping_to_voxel(doubleResAffine)
        inds = ut._to_voxel_coordinates(streamlines._data, lin_T, offset)
        streamlineCoords=np.floor((inds*.5))