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
        
        def extendROIinDirection(roiNifti,direction,iterations):
            """
            Extends a binarized, volumetric nifti roi in the specified directions.
            
            Potential midline issues
            
            ASSUMES ACPC ALIGNMENT
            
            NOTE: THIS CURRENTLY DOESN'T WORK RIGHT NOW., IT WOULD RESULT IN A WEIRD,
            BOX-CONSTRAINED INFLATION, DESIRED FUNCTIONALITY WOULD BE MORE LIKE
            ORTHOGONAL SPATIAL TRANSLATION

            Parameters
            ----------
            roiNifti : TYPE
               a binarized, volumetric nifti roi that is to be extended.
            direction : TYPE
                The direction(s) to be extended in.
            iterations : TYPE
                The number of iterations to perform the inflation.  Proceeds voxelwise.
                
            Returns
            -------
            extendedROI : TYPE
                The ROI nifti, after the extension has been performed
                
            """
            from scipy import ndimage
            import nibabel as nib
            
            #first, get the planar bounding box of the input ROI
            boundaryDictionary=boundaryROIPlanesFromMask(roiNifti)
            
            #establish the appropriate pairings.  Kind of assumes the order from
            #boundaryROIPlanesFromMask wherein complimentary borders are right next to one another
            boundaryLabels=list(boundaryDictionary.keys())

            
            #if a singleton direction, and thus a string, has been entered,
            #convert it to a one item list for iteration purposes
            if isinstance(direction,str):
                direction=list(direction)
                
            #go ahead and do the inflation
            concatData=ndimage.binary_dilation(roiNifti.get_fdata(), iterations=iterations)
            
            #convert that data array to a nifti
            extendedROI=nib.Nifti1Image(concatData, roiNifti.affine, roiNifti.header)
            
            #now remove all of the undesired proportions
            for iDirections in direction:
                #if the current direction isn't one of the desired expansion directions
                #go ahead and cut anything off that was expanded
                if not any([x in direction for x in boundaryLabels]):
                    extendedROI=sliceROIwithPlane(extendedROI,boundaryDictionary[iDirections],iDirections)
            return extendedROI