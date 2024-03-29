# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:01 2021

@author: Daniel
"""
def makePlanarROI(reference, mmPlane, dimension):
    """
    this plane will be oblique to the subject's *actual anatomy* if they aren't
    oriented orthogonally. As such, do this only AFTER acpc-alignment
    
    adapted from https://github.com/DanNBullock/wma_tools/blob/master/ROI_Tools/bsc_makePlanarROI_v3.m
    
    Parameters
    ----------
    reference : TYPE
        The nifti that the ROI will be applied to, also functions as 
        the source of affine transform.
    mmPlane : TYPE
        the ACPC (i.e. post affine application) mm plane that you would like 
        to generate a planar ROI at.  i.e. mmPlane=0 and dimension= x would 
        be a planar roi situated along the midsaggital plane.
    dimension : TYPE
        either 'x', 'y', or 'z', to indicate the plane that you would like the
        roi generated along

    Raises
    ------
    ValueError
        Raises error if requested coordinate is outside of reference nifti

    Returns
    -------
    returnedNifti : TYPE
        the nifti roi structure of the planar ROI

    """
    
    # #consider replacing some of this with this:
    # refDimBounds=np.asarray(mask.bounding_box(region_of_interest))
    
    # #use itertools and cartesian product to generate vertex img space coordinates
    # outCoordnates=np.asarray(list(itertools.product(refDimBounds[:,0], refDimBounds[:,1], refDimBounds[:,2])))
     
    # #perform affine
    # convertedBoundCoords=nib.affines.apply_affine(affine,outCoordnates)
    
    
    
    import nibabel as nib
    import numpy as np
    import dipy.tracking.utils as ut
    
    fullMask = nib.nifti1.Nifti1Image(np.ones(reference.get_data().shape), reference.affine, reference.header)
    #pass full mask to subject space boundary function
    convertedBoundCoords=subjectSpaceMaskBoundaryCoords(fullMask)
    
    #create a dict to interpret input
    dimDict={'x':0,'y':1,'z':2}
    selectedDim=dimDict[dimension]
    
    #if the requested planar coordinate is outside of the image, throw a full on error
    if convertedBoundCoords[0,selectedDim]>mmPlane or convertedBoundCoords[1,selectedDim]<mmPlane: 
        raise ValueError('Requested planar coordinate outside of reference image \n coordinate ' + str(mmPlane) + ' requested for bounds ' + str(convertedBoundCoords[:,selectedDim]))   
    
    #not always 0,0,0
    subjectCenterCoord=np.mean(convertedBoundCoords,axis=0)
    #copy it over to be the plane center coord and replace with requested value
    planeCenterCoord=subjectCenterCoord.copy()
    planeCenterCoord[selectedDim]=mmPlane
    
    planeCoords=[]
    #i guess the only way
    for iDims in list(range(len(dimDict))):
        if iDims==selectedDim:
            #set the value to the plane coord in the relevant dimension
            planeCoords.append(mmPlane)
        else:
            #set step size at half the voxel length in this dimension
            stepSize=fullMask.header.get_zooms()[iDims]*.5
            #create a vector with the coords for this dimension
            dimCoords=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],stepSize)
            #append it to the planeCoords list object
            planeCoords.append(list(dimCoords))
    x, y, z = np.meshgrid(planeCoords[0], planeCoords[1], planeCoords[2],indexing='ij')
    #squeeze the output (maybe not necessary)          
    planeCloud= np.squeeze([x, y, z])
    #convert to coordinate vector
    testSplit=np.vstack(planeCloud).reshape(3,-1).T
    #use dipy functions to treat point cloud like one big streamline, and move it back to image space
    #alternatively just use inds=nib.affines.apply_affine(np.linalg.inv(fullMask.affine),testSplit)
    lin_T, offset =ut._mapping_to_voxel(fullMask.affine)
    inds = ut._to_voxel_coordinates(testSplit, lin_T, offset)
    
    #create a blank array for the output
    outData=np.zeros(reference.shape).astype(bool)
    #set the relevant indexes to true
    #-1 because of zero indexing
    outData[inds[:,0]-1,inds[:,1]-1,inds[:,2]-1]=True
    #format output
    returnedNifti=nib.nifti1.Nifti1Image(outData, reference.affine, header=reference.header)
    return returnedNifti

# def roiFromAtlas(atlas,roiNum):
#     """roiFromAtlas(atlas,roiNum)
#     #creates a nifti structure mask for the input atlas image of the specified label
#     #
#     #  DEPRICATED BY  multiROIrequestToMask
#     #
#     # INPUTS:
#     # -atlas:  an atlas nifti
#     #
#     # -roiNum: an int input indicating the SINGLE label that is to be extracted.  Will throw warning if not present
#     #
#     # OUTPUTS:
#     # -outImg:  a mask with int(1) in those voxels where the associated label was found.  If the label wasn't found, an empty nifti structure is output.
#     """

#     import numpy as np
#     import nibabel as nib
#     outHeader = atlas.header.copy()
#     #numpy and other stuff has been acting weird with ints lately
#     atlasData = np.round(atlas.get_data()).astype(int)
#     outData = np.zeros((atlasData.shape)).astype(int)
#     #check to make sure it is in the atlas
#     #not entirely sure how boolean array behavior works here
#     if  np.isin(roiNum,atlasData):
#             outData[atlasData==roiNum]=int(1)
#     else:
#         import warnings
#         warnings.warn("WMA.roiFromAtlas WARNING: ROI label " + str(roiNum) + " not found in input Nifti structure.")
                
#     outImg = nib.nifti1.Nifti1Image(outData, atlas.affine, outHeader)
#     return outImg

def planeAtMaskBorder(inputMask,relativePosition):
    """
    creates a planar roi at the specified border of the specified ROI.

    Parameters
    ----------
    inputMask : TYPE
        a nifti with ONLY 1 and 0 (int) as the content, a boolean mask, in essence
        The planes will be generated at the edge of the non-zero values
    relativePosition : TYPE
        String input indicating which border to obtain planar roi at
        Valid inputs: 'superior','inferior','medial','lateral','anterior',
        'posterior','rostral','caudal','left', or 'right'
        
    Raises
    ------
    Exception
        Raises exeption of input term not recognized

    Returns
    -------
    outPlaneNifti : TYPE
        planar ROI as Nifti at specified border

    """
 
    import numpy as np
    
    #establish valid positional terms
    validPositionTerms=['superior','inferior','medial','lateral','anterior','posterior','rostral','caudal','left','right']
    #cased relativePosition check
    #again, don't know how arrays work with booleans
    if ~np.isin(relativePosition.lower(),validPositionTerms):
         raise Exception("planeAtROIborder Error: input relative position " + relativePosition + " not valid.")
    
    #convert the boundary coords of the mask to subject space in order to interpret positional terms
    convertedBoundCoords=subjectSpaceMaskBoundaryCoords(inputMask)
    
    #kind of assumes at least moderately ACPC aligned data, at least insofar
    #as relative anatomical position terms are concerned
    
    positionTermsDict={'superior': np.max(convertedBoundCoords[:,2]),
                      'inferior': np.min(convertedBoundCoords[:,2]),
                      'medial':   np.min(convertedBoundCoords[np.min(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
                      'lateral': np.max(convertedBoundCoords[np.max(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
                      'anterior': np.max(convertedBoundCoords[:,1]),
                      'posterior': np.min(convertedBoundCoords[:,1]),
                      'rostral': np.max(convertedBoundCoords[:,1]),
                      'caudal': np.min(convertedBoundCoords[:,1]),
                      'left': np.min(convertedBoundCoords[:,0]),
                      'right': np.max(convertedBoundCoords[:,0])}    

    #similar to the positonal term Dict but for interpreting pertinant dimension
    dimensionDict={'superior': 'z',
                   'inferior': 'z',
                   'medial':   'x',
                   'lateral': 'x',
                   'anterior': 'y',
                   'posterior': 'y',
                   'rostral': 'y',
                   'caudal': 'y',
                   'left': 'x',
                   'right': 'x'}

    outPlaneNifti=makePlanarROI(inputMask,positionTermsDict[relativePosition] , dimensionDict[relativePosition])
    
    #return the planar roi you have created
    return outPlaneNifti

def returnNiftiBounds(inputNifti):
    """
    Returns the subject space boundary coordinates of the input nifti.
    

    Parameters
    ----------
    inputNifti : nifti1.Nifti1Image
        A nifti image from which you would like to obtain the subject-space
        boundary coordinates

    Returns
    -------
    boundingBox : np.array, 2x3
        The first row (row 0) corresponds to the max dimension values,
        while the second row (row 1) corresponds to the min dimension values.

    """
  
    import itertools
    import numpy as np
    import nibabel as nib
  
    #obtain the nifti shape
    niftiShape=inputNifti.shape
    
    #stack these alongside 0, the min bound in image space
    imgBounds=np.vstack([np.asarray(niftiShape),[0,0,0]])
    
    #iterate to obtain all possible combinations
    verticies=np.asarray(list(itertools.product(imgBounds[:,0], imgBounds[:,1], imgBounds[:,2])))
   
    #apply affine to obtain subject space coordinates
    outCoordnates=nib.affines.apply_affine(inputNifti.affine,verticies)
    
    #find the min and max in each dimension, and stack acordingly
    boundingBox=np.vstack([np.max(outCoordnates,axis=0),np.min(outCoordnates,axis=0)])
    
    return boundingBox

def createSphere(r, p, reference, supress=True):
    """
    create a sphere of given radius at some point p in the brain mask
    
    modified version of nltools sphere function which outputs the sphere ROI
    in the coordinate space of the input reference
    
    Parameters
    ----------
    r : TYPE
        radius of the sphere IN MM SPACE/UNITS
        (probably need to test to infer rounding behavior resulting from 
         subjectSpaceMaskBoundaryCoords / dipy coord converion)
    p : TYPE
        point (in subject coordinates of the brain mask--i.e. NOT the image 
        space--of the center of the sphere)
    reference : TYPE
        The reference nifti whose space the sphere will be in
    supress : TYPE, optional
        Bool indicating whether to supress the output print behavior.  Useful
        if function is being used repeatedly.  At the same time, the method
        used in streamlinesInWindowFromMapping may actually be more efficient
        The default is True.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    import nibabel as nib
    import numpy as np
    
    if not supress:
        print('Creating '+ str(r) +' mm radius spherical roi at '+str(p))
    
    fullMask = nib.nifti1.Nifti1Image(np.ones(reference.get_data().shape), reference.affine, reference.header)
    #obtain boundary coords in subject space in order set max min values for interactive visualization
    convertedBoundCoords=subjectSpaceMaskBoundaryCoords(fullMask)
    
    #if the sphere centroid is outside of the image, throw a full on error
    if np.any(    [np.min(convertedBoundCoords[:,dims])>p[dims] or  
                   np.max(convertedBoundCoords[:,dims])<p[dims] 
                   for dims in list(range(len(reference.shape))) ]):
        raise ValueError('Requested sphere centroid outside of reference image')
    
    if np.any(    [np.min(convertedBoundCoords[:,dims])-r>p[dims] or  
                   np.max(convertedBoundCoords[:,dims])+r<p[dims] 
                   for dims in list(range(len(reference.shape))) ]):
        import warnings
        warnings.warn('Requested sphere partially outside of reference image')
    
    #get the dimensions of the source image
    dims = reference.shape
    
    imgCoord=np.floor(nib.affines.apply_affine(np.linalg.inv(reference.affine),p))
    
    #previous version of this misunderstood this process and included header.zooms
    #radius is interpreted in mm, but image indexing is interpreted in voxels
    #as such, you have to normalize the later distance mask computation (mask_r)
    #with that information
    
    #for each dimension, compute the orthogonal distance of the relevant centroid
    #coordinate component from each other point in the mask
    #NO NEED FOR AFFINE USAGE, BECAUSE THIS IS ALL IN IMAGE SPACE
    dimVoxelDistVals=[np.abs((np.arange(0, dims[i]) )-imgCoord[i])
                      for i in list(range(len(dims)))]
    #ogrid doesnt work?  meshgrid seems to work fine
    #not sure why previous version was forcing to type int
    x, y, z = np.meshgrid(dimVoxelDistVals[0], dimVoxelDistVals[1], dimVoxelDistVals[2],indexing='ij')          
    
    #clever element-wise computation and summation of 3-dimensional Pythagorean
    #components, followed by masking via radius value
    #NOTE THAT THE SUBSEQUENT FORMULATION HAS IMAGESPACE UNITS ON THE RIGHT
    #AND MM SPACE ON THE LEFT.  AS SUCH WE MUST MODIFY THE DISTANCE COMPUTATION
    #mask_r = x*x + y*y + z*z <= r*r
    voxelDims=reference.header.get_zooms()
    mask_r = x*x*voxelDims[0] + y*y*voxelDims[1] + z*z*voxelDims[2] <= r*r

    outSphereROI = np.zeros(dims, dtype=bool)
    outSphereROI[mask_r] = True
    #not sure of robustness to strange input affines, but seems to work
    return nib.Nifti1Image(outSphereROI, affine=reference.affine, header=reference.header)

def multiROIrequestToMask(atlas,roiNums,inflateIter=0):
    """
    Creates a nifti structure ROI mask for the input atlas image of the 
    specified labels.  Optionally inflates
    
    
    NOTE" REPLACE WITH nil.masking.intersect_masks when you get a chance

    Parameters
    ----------
    atlas : nifti1.Nifti1Image
        an atlas nifti from which the selected labels will be extracted and
        turned into a nifti ROI mask.
    roiNums : TYPE
        a 1d int array input indicating the labels that are to be extracted.
        Singleton request (single int) will work fine.  Will throw warning 
        if not present in input atlas
    inflateIter : TYPE, optional
        The number of inflate iterations you would like to perform. If no value
        is passed, no inflation is performedThe default is 0.

    Returns
    -------
    concatOutNifti : TYPE
        a mask with int(1) in those voxels where the associated labels were
        found.  If the label wasn't found, an empty nifti structure is output.
        int is used because nibabel throws a fit with binary ROI files
   
    """
    
    import numpy as np
    import nibabel as nib
    from scipy import ndimage
    
    #force input roiNums to array, don't want to deal with lists and dicts
    roiNumsInArray=np.asarray(roiNums)
    
    if  np.logical_not(np.all(np.isin(roiNumsInArray,np.unique(np.round(atlas.get_data()).astype(int))))):
        import warnings
        warnings.warn("WMA.multiROIrequestToMask WARNING: ROI label " + str(list(roiNumsInArray[np.logical_not(np.isin(roiNumsInArray,np.unique(np.round(atlas.get_data()).astype(int))))])) + " not found in input Nifti structure.")
        
    #obtain coordiantes of all relevant label 
    #ATLASES ARE ACTING WEIRD THESE DAYS, gotta do round then int, not other way
    #if you don't do it this way, somehow you get a reduced number of labels
    labelCoords=np.where(np.isin(np.round(atlas.get_data()).astype(int),roiNumsInArray))

    #create blank data structure
    concatData=np.zeros(atlas.shape)
    #set all appropriate values to true
    concatData[labelCoords]=True
    
    #if the inflation parameter has been set
    if inflateIter!=0:
        concatData=ndimage.binary_dilation(concatData, iterations=inflateIter)
   
    #create the output nifti
    concatOutNifti=nib.nifti1.Nifti1Image(concatData, affine=atlas.affine, header=atlas.header)
    
    return concatOutNifti

# def multiROIrequestToMask_Inflate(atlas,roiNums,):
#     """
    
#     maybe clean this up and merge with multiROIrequestToMask at some point
    

#     Parameters
#     ----------
#     atlas : TYPE
        
#     roiNums : TYPE
#         DESCRIPTION.
#     inflateIter : TYPE, optional
#         DESCRIPTION. The default is 0.

#     Returns
#     -------
#     inflatedOutNifti : TYPE
#         DESCRIPTION.

#     """
#     """multiROIrequestToMask(atlas,roiNums):
#     #creates a nifti structure mask for the input atlas image of the specified labels
#     #
#     # INPUTS:
#     # -atlas: 
#     #
#     # -roiNums: an 1d int array input indicating the labels that are to be extracted.  Singleton request (single int) will work fine.  Will throw warning if not present
#     #
#     # -inflateIter: the number of inflate iterations you would like to perform
#     #
#     # OUTPUTS:
#     # -outImg:  a mask with int(1) in those voxels where the associated labels were found.  If the label wasn't found, an empty nifti structure is output.
    
#     ##NOTE REPLACE WITH nil.masking.intersect_masks when you get a chance
#     """

#     import nibabel as nib
#     from scipy import ndimage
    
#     #yes, this function is a glorified wrapper, why do you ask?
#     selectROINifti=multiROIrequestToMask(atlas,roiNums)
    
  
    
#     #set all appropriate values to true
#     inflatedOutNifti=nib.nifti1.Nifti1Image(inflatedArray, affine=atlas.affine, header=atlas.header)
    
#     return inflatedOutNifti

def planarROIFromAtlasLabelBorder(inputAtlas,roiNums, relativePosition):
    """#planarROIFromAtlasLabelBorder(referenceNifti, mmPlane, dimension):
    #generates a planar ROI at the specified label border from the input atlas
    #
    # INPUTS:
    # -inputAtlas:  the atlas that the numeric labels specified in roiNums will be extracted from
    #
    # -roiNums: either a specification of a single or multiple ROIs which the planar ROI border will be assesed for.
    # a submission of multiple ROIs will be assessed as the amalgamation (i.e. merging) of those labels.
    #
    # -relativePosition: string input indicating which planar border to generate
    # Valid inputs: 'superior','inferior','medial','lateral','anterior','posterior','rostral','caudal','left', or 'right'
    #
    # OUTPUTS:
    # -planarROI: the roi structure of the planar ROI
    #
    #  Daniel Bullock 2020 Bloomington
    """
    
    #this plane will be oblique to the subject's *actual anatomy* if they aren't
    #oriented orthogonally. As such, do this only after acpc-alignment

    #merge the inputs if necessary
    mergedRequest=multiROIrequestToMask(inputAtlas,roiNums)
    
    #use that mask to generate a planar border
    planeOut=planeAtMaskBorder(mergedRequest,relativePosition)
    
    return(planeOut)
    
def sliceROIwithPlane(inputROINifti,inputPlanarROI,relativePosition):
    """
    Slices input ROI Nifti using input planarROI and returns portion specified
    by relativePosition.  Useful for modifying anatomical ROI masks and other
    derived ROIS
    
    Can probably be refactored to take advantage of dipy coord conversion 
    mechanics
    
    PRESUMES ACPC ALIGNMENT
    
    Parameters
    ----------
    inputROINifti : TYPE
        a (presumed ROI) nifti with ONLY 1 and 0 (int) 
        as the content, a boolean mask, in essence
    inputPlanarROI : TYPE
        a planar roi (nifti) that is to be used to perform the slicing
        operation on the inputROINifti.  If this doesn't intersect the input
        ROI, an exception will be thrown
    relativePosition : TYPE
    which portion of the sliced ROI to return.
    Valid inputs: 'superior','inferior','medial','lateral','anterior',
    'posterior','rostral','caudal','left', or 'right'
  
    Raises
    ------
    ValueError
        Will throw exceptions if input ROIs do not intersect, if input
        slicer roi isn't planar, or if return portion value is not understood

    Returns
    -------
    remainingROI : TYPE
        A nifti wherein the voxels not meeting the criteria have been dropped/
        set to zero

    """
    import nibabel as nib
    import numpy as np
    
    #get the data
    inputROINiftiData=inputROINifti.get_data()
    inputPlanarROIData=inputPlanarROI.get_data()
    
    #boolean to check if intersection
    intersectBool=np.any(np.logical_and(inputROINiftiData!=0,inputPlanarROIData!=0))
    if ~intersectBool:
        import warnings
        warnings.warn("sliceROIwithPlane WARNING: input planar ROI does not intersect with input ROI.")

    #implement test to determine if input planar roi is indeed planar
    #get coordinates of mask voxels in image space
    planeVoxCoords=np.where(inputPlanarROI.get_data())
    #find the unique values of img space coordinates for each dimension
    uniqueCoordCounts=[len(np.unique(iCoords)) for iCoords in planeVoxCoords]
    #one of them should be singular in the case of a planar roi, throw an error if not
    if ~np.any(np.isin(uniqueCoordCounts,1)):
        raise ValueError('input cut ROI not planar (i.e. single voxel thick for True values)')
    
    fullMask = nib.nifti1.Nifti1Image(np.ones(inputROINifti.get_data().shape), inputROINifti.affine, inputROINifti.header)
    #pass full mask to subject space boundary function
    fullVolumeBoundCoords=subjectSpaceMaskBoundaryCoords(fullMask)
    #get boundary mask coords for mask
    maskVolumeBoundCoords=subjectSpaceMaskBoundaryCoords(inputPlanarROI)
    #find the subject space plane that the dim is in
    subjSpacePlaneDimIndex=np.where(~np.all(np.equal(fullVolumeBoundCoords,maskVolumeBoundCoords),axis=0))[0][0]
    
    #set up the dictionary for boundaries
    positionTermsDict={'superior': np.max(fullVolumeBoundCoords[:,2]),
                      'inferior': np.min(fullVolumeBoundCoords[:,2]),
                      'medial':   0,
                      'lateral': [np.min(fullVolumeBoundCoords[:,0]),np.max(fullVolumeBoundCoords[:,0])],
                      'anterior': np.max(fullVolumeBoundCoords[:,1]),
                      'posterior': np.min(fullVolumeBoundCoords[:,1]),
                      'rostral': np.max(fullVolumeBoundCoords[:,1]),
                      'caudal': np.min(fullVolumeBoundCoords[:,1]),
                      'left': np.min(fullVolumeBoundCoords[:,0]),
                      'right': np.max(fullVolumeBoundCoords[:,0])}
    
    #set up the dictionary for dimensions
    dimensionDict={'superior': 2,
                   'inferior': 2,
                   'medial':   0,
                   'lateral': 0,
                   'anterior': 1,
                   'posterior': 1,
                   'rostral': 1,
                   'caudal': 1,
                   'left': 0,
                   'right': 0}    


    #silly solution to lateral case
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    planeCoords=[]
    #i guess the only way
    for iDims in list(range(len(inputROINifti.shape))):
        #set step size at half the voxel length in this dimension
        stepSize=fullMask.header.get_zooms()[iDims]*.5
        if iDims==dimensionDict[relativePosition]:
            #kind of wishy,washy, but because we halved the step size in the previous step
            #by taking the average of the coord bounds (in subject space) we should actually be fine
            #we'll use this as one of our two bounds
            #just do 5 cases
            if relativePosition=='medial':
               #i guess it doesn't matter what sign is, one bound is always 0
               thisDimBounds=np.sort([np.mean(maskVolumeBoundCoords[:,subjSpacePlaneDimIndex]),0])
            elif relativePosition=='lateral':
               #find the nearest x bound, since you're not on the midline, it will work.
               thisDimBounds=np.sort([np.mean(maskVolumeBoundCoords[:,subjSpacePlaneDimIndex]),find_nearest([positionTermsDict['left'],positionTermsDict['right']], np.mean(maskVolumeBoundCoords[:,subjSpacePlaneDimIndex]))])
       
            else:
               thisDimBounds=np.sort([np.mean(maskVolumeBoundCoords[:,subjSpacePlaneDimIndex]),positionTermsDict[relativePosition]])
           
        else:
            thisDimBounds=np.sort([fullVolumeBoundCoords[0,iDims],fullVolumeBoundCoords[1,iDims]])
           
        #create a vector with the coords for this dimension
        dimCoords=np.arange(thisDimBounds[0],thisDimBounds[1],stepSize)
        #append it to the planeCoords list object
        planeCoords.append(list(dimCoords))
            
    x, y, z = np.meshgrid(planeCoords[0], planeCoords[1], planeCoords[2],indexing='ij')
    #squeeze the output (maybe not necessary)          
    planeCloud= np.squeeze([x, y, z])
    #convert to coordinate vector
    testSplit=np.vstack(planeCloud).reshape(3,-1).T
    #use dipy functions to treat point cloud like one big streamline, and move it back to image space
    import dipy.tracking.utils as ut
    lin_T, offset =ut._mapping_to_voxel(inputROINifti.affine)
    inds = ut._to_voxel_coordinates(testSplit, lin_T, offset)
    
    #create a blank array for the keep area mask
    keepArea=np.zeros(inputPlanarROI.shape).astype(bool)
    #set the relevant indexes to true
    #-1 because of zero indexing
    #could be an issue here if mismatch between input nifti and planar roi
    keepArea[inds[:,0]-1,inds[:,1]-1,inds[:,2]-1]=True 
    
    #create a nifti structure for this object
    sliceKeepNifti=nib.nifti1.Nifti1Image(keepArea, inputPlanarROI.affine, header=inputPlanarROI.header)
    
    #intersect the ROIs to return the remaining portion
    #will cause a problem if Niftis have different affines.
    from nilearn import masking 
    remainingROI=masking.intersect_masks([sliceKeepNifti,inputROINifti], threshold=1, connected=False)
    #consider throwing an error here if the output Nifti is empty
    
    return remainingROI

def pointCoordsToNIFTI(coords,refNifti,radius=None):
    
    from dipy.core.geometry import dist_to_corner
    import nibabel as nib
    import numpy as np
    #get a blank nifti for these coords
    blankNiftiData=np.zeros(refNifti.shape, dtype=int)
    
    # import dipy.tracking.utils as ut
    # lin_T, offset =ut._mapping_to_voxel(refNifti.affine)
    # inds = ut._to_voxel_coordinates(coords, lin_T, offset)
    
    # #iterate across the coords and add +1 to the nifti for each coord.
    # for iCoords in inds:
    #     #do i have to index like this?  I can't remember
    #     blankNiftiData[iCoords[0],iCoords[1],iCoords[2]]=blankNiftiData[iCoords[0],iCoords[1],iCoords[2]]+1
    
    #or
    
    #set the radius to the min val 
    if radius==None:
        #radius=dist_to_corner(refNifti.affine)
        radius=0
    
    print ('creating amalgamated ROI for ' + str (len(coords)) + ' coordinates using a ' +str(radius) + ' mm radius')
    
    for iCoords in coords:
        currentSphere=createSphere(radius, iCoords, refNifti, supress=True)
        blankNiftiData=np.add(blankNiftiData,currentSphere.get_data())
        
    #create the output nifti
    outNifti=nib.nifti1.Nifti1Image(blankNiftiData, refNifti.affine, header=refNifti.header)
    
    return outNifti
    

def alignROItoReference(inputROI,reference):
    """
    extracts the coordinates of an ROI and reinstantites them as an ROI
    in the refernce space of the reference input.
    Helps avoid affine weirdness.  Unclear if actually needed or used.
    Likely redundant with functions in other packages
    
    replace with either dipy affinemap or nilearn resample
    

    Parameters
    ----------
    inputROI : TYPE
        an input ROI in nifti format, to be converted
    reference : TYPE
        the reference nifti that you would like the ROI moved into the space of.

    Returns
    -------
    outROI : TYPE
        Output nifti ROI in the reference space of the input reference nifti

    """
    import numpy as np
    import nibabel as nib
    from dipy.tracking.utils import seeds_from_mask
    
    #infer the sampling density
    densityKernel=np.asarray(reference.header.get_zooms())
    
    #get the coordinates themselves
    roiCoords=seeds_from_mask(inputROI.get_data(), inputROI.affine, density=densityKernel)
    
    #use dipy functions to treat point cloud like one big streamline, and move it back to image space
    import dipy.tracking.utils as ut
    lin_T, offset =ut._mapping_to_voxel(reference.affine)
    inds = ut._to_voxel_coordinates(roiCoords, lin_T, offset)
    #create a blank array for the keep area mask
    outData=np.zeros(reference.shape).astype(bool)
    #set the relevant indexes to true
    #-1 because of zero indexing
    #TEST THIS
    outData[inds[:,0]-1,inds[:,1]-1,inds[:,2]-1]=True 
    
    #create a nifti structure for this object
    outROI=nib.nifti1.Nifti1Image(outData, reference.affine, header=reference.header)

    return outROI

def alignNiftis(nifti1,nifti2):
    """
    What is this?

    Parameters
    ----------
    nifti1 : TYPE
        DESCRIPTION.
    nifti2 : TYPE
        DESCRIPTION.

    Returns
    -------
    nifti1 : TYPE
        DESCRIPTION.
    nifti2 : TYPE
        DESCRIPTION.

    """
    
    from nilearn.image import crop_img, resample_img 
    from dipy.core.geometry import dist_to_corner
    from dipy.align.imaffine import AffineMap
    import numpy as np
    import nibabel as nib
    
    if   dist_to_corner(nifti1.affine)>dist_to_corner(nifti2.affine):
        print('resampling nifti1 to nifti2')
        nifti1=resample_img(nifti1,target_affine=nifti2.affine[0:3,0:3])
    elif dist_to_corner(nifti2.affine)>dist_to_corner(nifti1.affine):
        print('resampling nifti2 to nifti1')
        nifti2=resample_img(nifti2,target_affine=nifti1.affine[0:3,0:3])
    #if they are the same resolution you don't do anything
    
    #roundabout hack: spoof a full mask to get the subject space boundary coords
    fullMask1 = nib.nifti1.Nifti1Image(np.ones(nifti1.get_data().shape), nifti1.affine, nifti1.header)
    fullMask2 = nib.nifti1.Nifti1Image(np.ones(nifti2.get_data().shape), nifti2.affine, nifti2.header)
    #pass full mask to subject space boundary function
    convertedBoundCoords1=subjectSpaceMaskBoundaryCoords(fullMask1)
    convertedBoundCoords2=subjectSpaceMaskBoundaryCoords(fullMask2)
    
    #get the subject space min and max values for these boundaries
    minVals=np.min(np.asarray([convertedBoundCoords1[0,:],convertedBoundCoords2[0,:]]),axis=0)
    maxVals=np.max(np.asarray([convertedBoundCoords1[1,:],convertedBoundCoords2[1,:]]),axis=0)
    #so now we have the dimension spans for each axis
    #do i need to pad somewhere arround here?
    spans=maxVals-minVals
    dummyDataBlock=np.zeros(np.ceil(spans).astype(int))
    #we can create a dummy nifti that encompases this entire volume?
    #we can spoof a blank nifti here, and then resample it to one of the above as needed.
    #maybe don't throw it a header and see what happens
    dummyAffine=np.eye(4)
    dummyAffine[0,0:3]=maxVals
    dummyNifti=nib.nifti1.Nifti1Image(dummyDataBlock, dummyAffine)
    
    #now resample it to a nifti from above
    #note that when we arbitrarily created it above, we assumed it was orthogonal
    #this _probably_ takes care of that assumption
    dummyNifti=resample_img(dummyNifti,target_affine=nifti1.affine[0:3,0:3])
    
    #now use dipy affineMAp to ensure the volume overlap
    affine_map1 = AffineMap(np.eye(4),
                       dummyNifti.shape, dummyNifti.affine,
                       nifti1.shape, nifti1.affine)
    resampled1 = affine_map1.transform(nifti1)
    
    affine_map2 = AffineMap(np.eye(4),
                       dummyNifti.shape, dummyNifti.affine,
                       nifti2.shape, nifti2.affine)
    resampled2 = affine_map2.transform(nifti2)
    
    #now create out niftis
    nifti1=nib.nifti1.Nifti1Image(resampled1, dummyNifti.affine, nifti1.header)
    nifti2=nib.nifti1.Nifti1Image(resampled2, dummyNifti.affine, nifti2.header)
    
    return nifti1, nifti2

def subjectSpaceMaskBoundaryCoords(maskNifti):
    """
    convert the boundary voxel indexes of non-zero entries into subject space
    coordinates
    
    NOTE: first row is "lowest" (i.e. smallest value), second row is "highest"
    as such, need to be careful about double - side coordinates
    
    Parameters
    ----------
    maskNifti : TYPE
        a binarized mask nifti

    Returns
    -------
    subjectSpaceBounds : TYPE
        a 2 x 3 dimensional array indicating the minimum and maximum bounds
        for each dimension

    """
    
    import dipy.segment.mask as mask
    import numpy as np
    
    #get the bounding box in image space
    refDimBounds=np.asarray(mask.bounding_box(maskNifti.get_data()))
    
    #use itertools and cartesian product to generate vertex img space coordinates
    import itertools
    outCoordnates=np.asarray(list(itertools.product(refDimBounds[:,0], refDimBounds[:,1], refDimBounds[:,2])))
     
    #perform affine
    import nibabel as nib
    convertedBoundCoords=nib.affines.apply_affine(maskNifti.affine,outCoordnates)
    
    #create holder for output
    import numpy as np
    subjectSpaceBounds=np.zeros([2,convertedBoundCoords.shape[1]])
    
    #list comprehension to iterate across dimensions
    #picking out min and max vals
    #min
    subjectSpaceBounds[0,:]=[np.min(convertedBoundCoords[:,iDems]) for iDems in list(range(convertedBoundCoords.shape[1]))]
    #max
    subjectSpaceBounds[1,:]=[np.max(convertedBoundCoords[:,iDems]) for iDems in list(range(convertedBoundCoords.shape[1]))]
    
    return subjectSpaceBounds

# def dualCropNifti(nifti1,nifti2):
#     """dualCropNifti(nifti1,nifti2):
#     This function crops two niftis to the same size, using the largest of the
#     two post cropped niftis to establish the consensus dimensions of the output
#     nifti data blocks.
    
#     Note:  this won't do much if the background values of your images haven't
#     been masked / set to zero.
  
#     INPUTS
#     nifti1 / nifti2:  The niftis that you would like cropped to the same size
    
#     OUTPUTS
#     nifti1 / nifti2:  The cropped niftis

#     """
#     import nilearn
#     from nilearn.image import crop_img, resample_to_img 
#     import numpy as np
#     import nibabel as nib
    
#     inShape1=nifti1.shape
#     inShape2=nifti2.shape

#     #get 
#     #nilearn doesn't handle NAN gracefully, so we have to be inelegant
#     nifti1=nib.nifti1.Nifti1Image(np.nan_to_num(nifti1.get_data()), nifti1.affine, nifti1.header)
#     nifti2=nib.nifti1.Nifti1Image(np.nan_to_num(nifti2.get_data()), nifti2.affine, nifti2.header)
    
#     cropped1=crop_img(nifti1)
#     cropped2=crop_img(nifti2)
    
#     # find max values in each dimension and create a dummy
#     maxDimShape=np.max(np.asarray([cropped1.shape,cropped2.shape]),axis=0)
#     dummyArray=np.zeros(maxDimShape)
#     #arbitrarily selecting the first nifit should be fine, they should be aligned
#     dummyNifti= nib.nifti1.Nifti1Image(dummyArray, nifti1.affine, nifti1.header)
    
#     outNifti1=resample_to_img(cropped1,dummyNifti)
#     outNifti2=resample_to_img(cropped2,dummyNifti)
    
#     return outNifti1, outNifti2

def removeIslandsFromAtlas_viaLabelCount(atlasNifti,ignoreLabelsRequest=[]):
    """
    Removes island label values from atlas iteratively across values present.
    Important for when extracted atlas labels are being used as anatomical 
    markers.  Instead of using erosion, we'll use scipy.ndimage.measurements.label
    
    Potentially faster ways of doing this in other packages.
    
    Parameters
    ----------
    atlasNifti : string path to file loadable via nib.load or nifti-like object
        A multi label atlas/parc, with integer based labels.  Will load if string
        is passed
    ignoreLabelsRequest :  list of int, optional
        The number of inflate iterations you would like to perform. If no value
        is passed, no inflation is performedThe default is 0.

    Returns
    -------
    atlasNiftiOut : nibabel.nifti1.Nifti1Image
        The input atlas nifti, but with islands removed
    removalReport : pandas.DataFrame
        A pandas based table indicating the number of islands found, their total
        volume, and the pre and post voxel count for each label

    """

    #scipy.ndimage.measurements.labeled_comprehension?        

    import nibabel as nib
    from scipy.ndimage.measurements import label
    import numpy as np
    import copy
    import pandas as pd
    #because it unnecessarily warns about things
    pd.options.mode.chained_assignment = None
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti
    
    #by default we ignore the background
    ignoreLabelsRequest.append(0)
    #just in case they included 0 or repeats
    ignoreLabels=np.unique(ignoreLabelsRequest)

    #do something to ensure that it is int based
    atlasData=np.round( np.asanyarray(atlas.dataobj)).astype(int)
    
    #create a seprate object to begin de-islanding
    noIslandData=copy.deepcopy(atlasData)
    
    #get counts of the unique lables
    [labels,counts]=np.unique(atlasData, return_counts=True)
    
    #create a report of the island removal process
    removalReport=pd.DataFrame(columns=['labelVal','startVoxCount','islandCount','totalIslandVox','endVoxCount'])
    removalReport['labelVal']=labels
    removalReport['startVoxCount']=counts
    
    
    for labelIterator,iLabels in enumerate(labels):
        #we don't want to do 0
        

        if not iLabels in ignoreLabels:
            #label the distinct elements of the parcellation matching the current label
            #startLabel = time.time()
            #~.5 of .9 seconds per loop here
            labeled_array, num_features = label(atlasData==iLabels)
            #endLabel = time.time()
            #print(endLabel - startLabel)
            #get the counts for these elements
            [curr_labels,curr_counts]=np.unique(labeled_array, return_counts=True)

            #always ignore 0, add 1 because you're ignoring it
            maxSizeIndex=np.where(curr_counts[1:]==np.max(curr_counts[1:]))[0][0]+1
            #create a mask of the locations to be zeroed out
            currIslandMask=np.logical_and(labeled_array>0,labeled_array!=maxSizeIndex)
            
            #compute and populate the data for the report
            #-2 because we don't include 0 or the largest island
            removalReport['islandCount'].loc[removalReport['labelVal']==iLabels]=len(curr_counts)-2
            removalReport['totalIslandVox'].loc[removalReport['labelVal']==iLabels]=np.sum(currIslandMask)
            
            #use the mask to set the relevant voxels to 0
            noIslandData[currIslandMask]=0
            
            #count the new total voxel count for this label
            removalReport['endVoxCount'].loc[removalReport['labelVal']==iLabels]=np.sum(noIslandData==iLabels)
            #print(removalReport.loc[removalReport['labelVal']==iLabels])

    
    #now populate data for the lables we didn't modify
    for iLabels in ignoreLabels:
            labeled_array, num_features = label(atlasData==iLabels)
            #get the counts for these elements
            [curr_labels,curr_counts]=np.unique(labeled_array, return_counts=True)

            #always ignore 0, add 1 because you're ignoring it
            maxSizeIndex=np.where(curr_counts[1:]==np.max(curr_counts[1:]))[0][0]+1
            #create a mask of the locations to be zeroed out
            currIslandMask=np.logical_and(labeled_array>0,labeled_array!=maxSizeIndex)
            
            #compute and populate the data for the report
            #-2 because we don't include 0 or the largest island
            removalReport['islandCount'].loc[removalReport['labelVal']==iLabels]=len(curr_counts)-2
            removalReport['totalIslandVox'].loc[removalReport['labelVal']==iLabels]=np.sum(currIslandMask)
            
            #use the mask to set the relevant voxels to 0
            #except we arent changing these
            #noIslandData[currIslandMask]=0
            
            #count the new total voxel count for this label
            removalReport['endVoxCount'].loc[removalReport['labelVal']==iLabels]=np.sum(noIslandData==iLabels)
            
    #now prepare the nifti output
    atlasNiftiOut=nib.Nifti1Image(noIslandData, atlas.affine, atlas.header)
    
    return atlasNiftiOut,removalReport
  
def preProcParc(inParc,deIslandBool=True,inflateIter=0,retainOrigBorders=False,maintainIslandsLabels=None,erodeLabels=None):
    """
    

    Parameters
    ----------
    inParc : String path to file loadable via nib.load or NIFTI-like object
           A multi label atlas/parc, with integer based labels.  Will load if string
           is passed
    deIslandBool : Bool, optional
        Whether to perform a de-islanding operation. The default is True.
    inflateIter : int, optional
        The number of intlation iterations to perform. The default is 0, thus 
        no inflation.
    retainOrigBorders : Bool, optional
        Retains the borders of the origional input parcellation.
        The default is False, and so inflation, if performed, will extend outside
        the origional mask.
    maintainIslandsLabels : list or array of int, optional
        During the de-island operation (if performed) do not remove islands from
        these labels.  NOTE:  Potentially useful for cross-hemispheric labels.
        The default is None.
    erodeLabels : list or array of int, optional
        During the inflation operation (if perfrormed), allow these labels
        to be eroded into. The default is None.

    Returns
    -------
    atlasNiftiOut : nibabel.nifti1.Nifti1Image
        The nifti image, after requested preprocessing has been performed.
    deIslandReport : pandas.DataFrame
        A pandas based table indicating the number of islands found, their total
        volume, and the pre and post voxel count for each label
    inflationReport : pandas.DataFrame
        A pandas based table indicating which labels had voxels removed, and
        how many

    """

    import copy
    import numpy as np
    import nibabel as nib
    
    
    outAtlas=copy.deepcopy(inParc)
    
    #setup the ignore labels request vec
    if maintainIslandsLabels==None:
        ignoreLabelsRequest=[]
    else:
        ignoreLabelsRequest=maintainIslandsLabels
    
    #perform deislanding, if requested
    if deIslandBool:
        [outAtlas, deIslandReport]=removeIslandsFromAtlas_viaLabelCount(outAtlas,ignoreLabelsRequest)
    else:
        deIslandReport=None
    #hold this to hold the effect of the deislanding, if it occured
    deIslandHold=copy.deepcopy(outAtlas)
        
    #if inflation has been requested perform inflation
    if inflateIter>0:
        #but first, set up the erode atlas labels
        if erodeLabels==None:
            erodeLabels=[]
            
        for iToErode in erodeLabels:
            #changeMultpleLabelValues(inputParc,targetLabelNums,newLabelNum) ?
            #this sets these values to 0, allowing them to be inflated into
            outAtlas=changeLabelValue(outAtlas,iToErode,0)
        
        [outAtlas, inflationReport]=inflateAtlas(outAtlas,inflateIter)
        #but now we have to reset the values were eroded into
        #get the data blocks now, outside of the loop
        deIslandHoldData=np.round( np.asanyarray(deIslandHold.dataobj)).astype(int)
        outAtlasData=np.round( np.asanyarray(outAtlas.dataobj)).astype(int)
        
        
        for iToErode in erodeLabels:
            #find the indexes == to the erode value in the deIslandHold and == to 0 in outAtlas
            #set them back to the current erode value
            outAtlasData[np.logical_and(deIslandHoldData==iToErode,outAtlasData==0)]=iToErode
    else:
        #get the out atlas data just in case
        outAtlasData=np.round( np.asanyarray(outAtlas.dataobj)).astype(int)
    
    if retainOrigBorders:
        brainMask=np.asanyarray(inParc.dataobj)>0
        #negate it and set the corresponding data to 0
        outAtlasData[np.logical_not(brainMask)]=0
    
    #now set the outAtlasData s a new nifti
    
    #should be done, although the final report won't reflect that toErode or maintain borders operation
    #TODO
    #edit output report to reflect the recent change
    
    #make a nifti of it
    atlasNiftiOut=nib.Nifti1Image(outAtlasData, inParc.affine, inParc.header)

    return atlasNiftiOut,deIslandReport,inflationReport
    

def removeIslandsFromAtlas(atlasNifti):
    """
    Removes island label values from atlas iteratively across values present.
    Important for when extracted atlas labels are being used as anatomical 
    markers.
    
    Potentially faster ways of doing this in other packages.
    
    Parameters
    ----------
    atlasNifti : String path to file loadable via nib.load or NIFTI-like object
        A multi label atlas, with integer based labels.  Will load if string
        is passed

    Returns
    -------
    atlasNiftiOut : nibabel.nifti1.Nifti1Image
        The input atlas nifti, but with islands removed
    removalReport : pandas.DataFrame
        A pandas based table indicating which labels had voxels removed, and
        how many

    """
    
    import nibabel as nib
    import scipy
    import numpy as np
    import pandas as pd
    from scipy.ndimage import binary_erosion
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti


    # atlasData=np.round(atlas.get_data()).astype(int)
    # #COSIDER REPLACING ALL OR MOST OF THIS WITH
    #Doesn't seem to work?
    # relabelConnectedData=scipy.ndimage.measurements.label(atlasData.astype(bool))

    #count the label instances, and make an inference as to the background value
    #it's almost certianly the most common value, eg. 0
    (labels,counts)=np.unique(atlas.get_data(), return_counts=True)
    detectBackground=int(labels[np.where(np.max(counts)==counts)[0]])
    
    #get the list of labels without the background
    atlasLabels=labels.astype(int)[labels.astype(int)!=detectBackground]
    
    atlasData=atlas.get_data()
    removalReport=pd.DataFrame(columns=['label','max_erosion','islands_removed'])
    #remove the background values
    for iLabels in atlasLabels:
        #create a blank vector to add to the report
        reportRow=[]
        #add the current label
        reportRow.append(iLabels)
        erosionIterations=0
        #the default max iterations, will be reduced in cases where implementation
        #results in complete loss
        forcedMaxIterations=3
        currentErosion=atlas.get_data()==iLabels
        #I guess we can use this to set something like minimum island size?
        #a 3x3 block (last before singleton point) would have 27 voxels in it
        #using strcit less than allows for escape in cases where forcedMaxIterations
        #gets down to zero
        while np.logical_and(erosionIterations<forcedMaxIterations,np.sum(currentErosion)>28):
            currentErosion=binary_erosion(currentErosion)
            erosionIterations=erosionIterations+1
            #reset the thing if the number of voxels hits zero
            if np.sum(currentErosion) ==0:
                forcedMaxIterations=forcedMaxIterations-1
                erosionIterations=0
                currentErosion=atlas.get_data()==iLabels
        #print how much was eroded
        print(str(np.abs(np.sum(currentErosion) - (np.sum(atlas.get_data()==iLabels))))+ ' of ' + str(np.sum(atlas.get_data()==iLabels)) + ' voxels eroded label ' + str(iLabels) )
        #add that value to the report list for this label
        reportRow.append(np.abs(np.sum(currentErosion) - (np.sum(atlas.get_data()==iLabels))))
        #perform the propigation function, wherein the eroded label is inflated back
        #islandRemoved=scipy.ndimage.binary_propagation(currentErosion, structure=np.ones([3,3,3]).astype(int), mask=atlas.get_data()==iLabels)
        islandRemoved=scipy.ndimage.binary_propagation(currentErosion, mask=atlas.get_data()==iLabels)
        #print the resulting difference between the intitial label and this eroded+inflated one
        print(str(np.abs(np.sum(islandRemoved) - (np.sum(atlas.get_data()==iLabels))))+ ' of ' + str(np.sum(atlas.get_data()==iLabels)) + ' voxels removed for label ' + str(iLabels) )
        #add that value to the report list for this label
        reportRow.append(np.abs(np.sum(islandRemoved) - (np.sum(atlas.get_data()==iLabels))))
        #add the row to the output report
        removalReport.loc[len(removalReport)]=reportRow

        #now actually modify the data
        #what you want are the areas that are true in the origional atlas but false in the eroded+inflated version
        #set those to background
        atlasData[np.logical_and(atlas.get_data()==iLabels,np.logical_not(islandRemoved))]=detectBackground
    
        #make a nifti of it
        atlasNiftiOut=nib.Nifti1Image(atlasData, atlas.affine, atlas.header)
        
    return atlasNiftiOut, removalReport

def inflateAtlas(atlasNifti,iterations):
    """
    Inflates label values of input atlas background (i.e. 0) labels.  
    Will perform island removal first.  More self contained rewrite of
    inflateAtlasIntoWMandBG
       
    Potentially faster methods in other packages

    Parameters
    ----------
    atlasNifti : String path to file loadable via nib.load or NIFTI-like object
        A multi label atlas, with integer based labels.  Will load if string
        is passed
    iterations : int
        The number of iterations to perform the inflation.  Proceeds voxelwise.

    Returns
    -------
    atlasNiftiOut : nibabel.nifti1.Nifti1Image
        The input atlas nifti, but with islands removed and then inflated into
    removalReport : pandas DataFrame
        a dataframe detailing the alterations performed upon the input atlas

    """
    
    
    import numpy as np
    import scipy
    import nibabel as nib
    import pandas as pd
    
    #conditional import of tqdm
    try: 
        import tqdm
        tqdmFlag=True
    except:
        tqdmFlag=False
        
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti
 
    #do something to ensure that it is int based
    atlasData=np.round( np.asanyarray(atlas.dataobj)).astype(int)
    
    #get counts of the unique lables
    [labels,counts]=np.unique(atlasData, return_counts=True)
    
    #create a report of the island removal process
    removalReport=pd.DataFrame(columns=['labelVal','startVoxCount','endVoxCount','changeCount'])
    removalReport['labelVal']=labels
    removalReport['startVoxCount']=counts
    
    #get the image space bounds
    imgSpaceBounds=np.asarray([[0,0,0],list(atlasData.shape)])
    
    #within bounds check function
    def coordsInBounds(coords,bounds):
        
        boolHold=np.zeros(coords.shape,dtype=bool)
        for iDims in range(bounds.shape[1]):
            
            boolHold[:,iDims]=np.less_equal.outer(bounds[0,iDims] ,coords[:,iDims]) & np.greater.outer(bounds[1,iDims] ,coords[:,iDims])
        
        return coords[np.all(boolHold,axis=1)]
        
    
    #precompute some search windows using meshgrid
    #couldn't ever go above 8 right?
    searchKernels=[[] for ival in range(1,9)]
    for iterator,iVals in enumerate(range(1,9)):
        #because ranges don't include the end?
        curSpan=np.arange(-iVals,iVals+1)
        centerCoord=np.where(curSpan==0)[0][0]
        
        x, y, z = np.meshgrid(curSpan, curSpan, curSpan,indexing='ij')          
   
        mask_r = x*x + y*y + z*z <= iVals*iVals
        
        windowCoords=np.asarray(np.where(mask_r))
        
        centeredWindowCoords=windowCoords-centerCoord
        
        searchKernels[iterator]=centeredWindowCoords
        
        
    for iInflations in range(iterations):
        print('inflation iteration ' + str(iInflations+1))
        #create a blank holder for the new label identities
        newIDentityHolder=np.zeros(atlasData.shape)
        #create a mask of the non target data

        inflatedLabelsMask=scipy.ndimage.binary_dilation(atlasData>0)
        #find the intersection of the inflated data mask and the inflation targets
        infationArrayTargets=np.logical_and(inflatedLabelsMask,atlasData==0)
        #find the coordinates for these targets
        inflationTargetCoords=np.asarray(np.where(infationArrayTargets)).T
    
        #this manuver is going to cost us .gif
        #NOTE, sequencing of coordinates may influence votes
        #create a dummy array for each inflation iteration to avoid this
        if  tqdmFlag:
            for iInflationTargetCoords in tqdm.tqdm(inflationTargetCoords):
                #because I dont trust iteration in 
                #print(str(iInflationTargetCoords))
                #set some while loop iteration values
                #this is 1 less than the voxel-wise radius of the expanse you're considering
                window=0
                
                voteWinner=[0,0]
                while len(voteWinner)>1:
                    initialTargetCords=np.asarray([np.add(iInflationTargetCoords,iCoords) for iCoords in searchKernels[window].T])
                    #now check to see that they are within bounds
                    validCoords=coordsInBounds(initialTargetCords,imgSpaceBounds)
                    
                    #this seems dumb, but ok
                    curLabelBlock=np.array([atlasData[iValidCoords[0],iValidCoords[1],iValidCoords[2]] for iValidCoords in validCoords])
  
                    (labels,counts)=np.unique(curLabelBlock, return_counts=True)
                    voteWinner=labels[np.where(np.max(counts)==counts)[0]]
                    #just in case
                    window=window+1

                newIDentityHolder[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner)
                del voteWinner
        else: 
            for iInflationTargetCoords in inflationTargetCoords:
                #because I dont trust iteration in 
                #print(str(iInflationTargetCoords))
                #set some while loop iteration values
                #this is 1 less than the voxel-wise radius of the expanse you're considering
                window=0
                
                voteWinner=[0,0]
                while len(voteWinner)>1:
                    initialTargetCords=np.asarray([np.add(iInflationTargetCoords,iCoords) for iCoords in searchKernels[window].T])
                    #now check to see that they are within bounds
                    validCoords=coordsInBounds(initialTargetCords,imgSpaceBounds)
                    
                    #this seems dumb, but ok
                    curLabelBlock=np.array([atlasData[iValidCoords[0],iValidCoords[1],iValidCoords[2]] for iValidCoords in validCoords])
  
                    (labels,counts)=np.unique(curLabelBlock, return_counts=True)
                    voteWinner=labels[np.where(np.max(counts)==counts)[0]]
                    #just in case
                    window=window+1

                newIDentityHolder[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner)
                del voteWinner
            
        #once the inflations are done for this round, set the values in the atlas data
        #to their new identities
        #2/25/2022 note:  previously we had been directly setting the identity
        #within the iInflationTargetCoords loop.  This was probably leading
        #to over-percolation of low number identities
        atlasData[infationArrayTargets]=newIDentityHolder[infationArrayTargets]
        
    
    #create end report
    #get counts of the unique lables
    [labels,counts]=np.unique(atlasData, return_counts=True)
    
    #create a report of the island removal process
    for iterator,iLabels in enumerate(labels):
        removalReport['endVoxCount'].loc[removalReport['labelVal']==iLabels]=counts[iterator]
        removalReport['changeCount'].loc[removalReport['labelVal']==iLabels]=removalReport['startVoxCount'].loc[removalReport['labelVal']==iLabels]-counts[iterator]
    
    inflatedAtlas=nib.Nifti1Image(atlasData, atlas.affine, atlas.header)
    
    return inflatedAtlas,removalReport
    

def inflateAtlasIntoWMandBG(atlasNifti,iterations,inferWM=False):
    """
    Inflates label values of input atlas into (INFERRED, SEE IN CODE NOTES)
    white matter labels, and background 0 labels.  Will perform island removal
    first.
    
    NOTE:  Ordering effects likely in inflation outcomes (i.e. lower labels
    have higher priority)
    
    Potentially faster methods in other packages

    Parameters
    ----------
    atlasNifti : TYPE
        An integer based, multi label nifti atlas.
    iterations : TYPE
        The number of iterations to perform the inflation.  Proceeds voxelwise.

    Returns
    -------
    atlasNiftiOut : TYPE
        The input atlas nifti, but with islands removed and then inflated into

    """
    import nibabel as nib
    import numpy as np
    #from scipy.interpolate import NearestNDInterpolator
    import scipy
    #conditional import of tqdm
    try: 
        import tqdm
        tqdmFlag=True
    except:
        tqdmFlag=False
        
    from nilearn.image import crop_img, resample_img
    from scipy import ndimage
    #import dipy.tracking.utils as ut
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti
    
    #first perform the island removal operation
    [noIslandAtlas,report]=removeIslandsFromAtlas(atlasNifti)
    #just save it
    report.to_csv('de-island_report.csv')
    
    
    #now we need to detect the wm and bg labels
    #lots of ways we could do this, I guess
    #bg we can detect in the same way that we detected in the deIsland code
        
    #count the label instances, and make an inference as to the background value
    #it's almost certianly the most common value, eg. 0
    (labels,counts)=np.unique(atlas.get_data(), return_counts=True)
    detectBackground=int(labels[np.where(np.max(counts)==counts)[0]])
    
    #now comes a tougher part, how do we detect WM labels, if any (because they might not be there)
    #one heuristic would be that it would be the label that would be inflated into most 
    #(and would inflated into the most other things) if you did some sort of iterative
    #inflation assesment across labels
    #seems time consuming
    #another alternative would be to use the relative total labeled volume proportion
    #occupied by any given label, and infer which of these (if any) might plausibly
    #be the wm
    
    #we can remove the bg from contention just like we did last time
    nonBGLabels=labels.astype(int)[labels.astype(int)!=detectBackground]
    #but we should also remove it from the counts
    nonBGCounts=counts[labels.astype(int)!=detectBackground]
    
    #now we can get the total labeled volume of the brain, in mm, using the size of a single voxel
    #the result of this is the cubic volume of a single voxel in this atlas
    voxVolume=np.prod(atlas.header.get_zooms())
    #we can then use this to multiply the total number of labeled voxels
    #I guess we can use the initial atlas rather than the modified atlas
    #shouldn't matter in either case
    #should be around 1 to 1.5 liters, with 1 liter = 1000000 cubic mm
    totalVolume=np.sum(nonBGCounts)*voxVolume
    #normalize the volumes of each label to this sum
    volumeProportions=(nonBGCounts*voxVolume)/totalVolume
    # generally speaking the brain is between 40 to 50 % wm by volume, with 
    #that tisue being split between both hemisphers.  However, in order 
    #to account for ventricles and other such things we can reduce this to a
    #lower threshold of approximately 12 percent.  This can be played with
    #in the future if it proves to be aproblem
    thresholdValue=.12
    #now find the indexes of the labels that meet this criterion
    presumedWMLabels=nonBGLabels[volumeProportions>thresholdValue]
    #its possible that the atlas didn't have WM to begin with, or that the WM
    #has been subdivided, in which case this heuristic would work, but probably
    #neither would the iterative inflation assesment heuristic either, because
    #we would have no apriori intutions about which labels were the WM.
    #in the event that no labels meet the threshold, the presumedWMLabels should
    #be empty, and a simple cat of it and the BG label shouldn't result in a problem
    
    #actually, it seems like this might be a problem sometimes, so lets set this as an option
    if inferWM:
        inflationLabelTargets=list(presumedWMLabels)+list(np.atleast_1d(detectBackground))
        #also get remaining labels
        remaining=nonBGLabels[np.isin(nonBGLabels,inflationLabelTargets,invert=True)]
    else:
        inflationLabelTargets=list(np.atleast_1d(detectBackground))
        #also get remaining labels
        remaining=nonBGLabels[np.isin(nonBGLabels,inflationLabelTargets,invert=True)]
        
    #now iteratively perform the inflation
    
    #so long as nothing is on the edge this will probably work, due to coord weirdness
    #This was forshadowing, code updated on 1/12/22 to take care of this
    #get the image data
    noIslandAtlasData=noIslandAtlas.get_data()
    noIslandAtlasData=np.round(noIslandAtlasData).astype(int)
    
    atlasBounds=subjectSpaceMaskBoundaryCoords(atlas)  
    
    for iInflations in range(iterations):
        print('inflation iteration ' + str(iInflations+1))
        #pad the data to avoid indexing issues
        noIslandAtlasData=np.pad(noIslandAtlasData,[1,1])
        #create a blank holder for the new label identities
        newIDentityHolder=np.zeros(noIslandAtlasData.shape)
        #create a mask of the non target data
        dataMask=np.isin(noIslandAtlasData,remaining)
        #get essentially the opposite of this, in the form of the mask of the inflation targets
        inflationTargetsMask=np.isin(noIslandAtlasData,inflationLabelTargets)
        #inflate the datamask
        inflatedLabelsMask=scipy.ndimage.binary_dilation(dataMask)
        #find the intersection of the inflated data mask and the inflation targets
        infationArrayTargets=np.logical_and(inflatedLabelsMask,inflationTargetsMask)
        #find the coordinates for these targets
        inflationTargetCoords=np.asarray(np.where(infationArrayTargets)).T
    
        #this manuver is going to cost us .gif
        #NOTE, sequencing of coordinates may influence votes
        #create a dummy array for each inflation iteration to avoid this
        if  tqdmFlag:
            for iInflationTargetCoords in tqdm.tqdm(inflationTargetCoords):
                #because I dont trust iteration in 
                #print(str(iInflationTargetCoords))
                #set some while loop iteration values
                #this is the voxel-wise radius of the expans you're considering
                window=1
                #if you set the vector to 2 or longer by default, it will run until fixed
                voteWinner=[0,0]
                while len(voteWinner)>1:
                #+2 because python is weird and doesn't include the top index
                    #we need to consider those cases in which the voxels are adjacent to the edge of the volume
                    xRange=np.asarray(range((iInflationTargetCoords[0]-window),(iInflationTargetCoords[0]+1+window)))
                    yRange=np.asarray(range((iInflationTargetCoords[1]-window),(iInflationTargetCoords[1]+1+window)))
                    zRange=np.asarray(range((iInflationTargetCoords[2]-window),(iInflationTargetCoords[2]+1+window)))
                    #remove invalid indexes
                    #-1, right?
                    xRange=xRange[np.all([xRange<noIslandAtlasData.shape[0]-1,  xRange>=0],axis=0)]
                    yRange=yRange[np.all([yRange<noIslandAtlasData.shape[1]-1,  yRange>=0],axis=0)]
                    zRange=zRange[np.all([zRange<noIslandAtlasData.shape[2]-1,  zRange>=0],axis=0)]
                    
                    x, y, z = np.meshgrid(xRange, yRange, zRange,indexing='ij')
                    #squeeze the output (maybe not necessary)          
                    windowBlock= np.squeeze([x, y, z])
                    #convert to array of coordinates
                    windowIndexes=np.vstack(windowBlock).reshape(3,-1).T
     
                    #probably dumb and inefficient, but here we are
                    voxelVotes=np.asarray([noIslandAtlasData[iVoxel[0],iVoxel[1],iVoxel[2]] for iVoxel in list(windowIndexes)])
                    #get the vote counts
                    (labels,counts)=np.unique(voxelVotes[np.isin(voxelVotes,inflationLabelTargets,invert=True)], return_counts=True)
                    voteWinner=labels[np.where(np.max(counts)==counts)[0]]
                    #just in case
                    del voxelVotes
                    window=window+1
                #print(str(noIslandAtlasData[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]))
                #print(str(voteWinner))
                newIDentityHolder[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner)
                del voteWinner
        else: 
            #kind of ugly here in that there's duplicated code
            for iInflationTargetCoords in inflationTargetCoords:
                #because I dont trust iteration in 
                #print(str(iInflationTargetCoords))
                #set some while loop iteration values
                #this is the voxel-wise radius of the expans you're considering
                window=1
                #if you set the vector to 2 or longer by default, it will run until fixed
                voteWinner=[0,0]
                while len(voteWinner)>1:
                #+2 because python is weird and doesn't include the top index
                    #we need to consider those cases in which the voxels are adjacent to the edge of the volume
                    xRange=np.asarray(range((iInflationTargetCoords[0]-window),(iInflationTargetCoords[0]+1+window)))
                    yRange=np.asarray(range((iInflationTargetCoords[1]-window),(iInflationTargetCoords[1]+1+window)))
                    zRange=np.asarray(range((iInflationTargetCoords[2]-window),(iInflationTargetCoords[2]+1+window)))
                    #remove invalid indexes
                    #-1, right?
                    xRange=xRange[np.all([xRange<noIslandAtlasData.shape[0]-1,  xRange>=0],axis=0)]
                    yRange=yRange[np.all([yRange<noIslandAtlasData.shape[1]-1,  yRange>=0],axis=0)]
                    zRange=zRange[np.all([zRange<noIslandAtlasData.shape[2]-1,  zRange>=0],axis=0)]
                    
                    x, y, z = np.meshgrid(xRange, yRange, zRange,indexing='ij')
                    #squeeze the output (maybe not necessary)          
                    windowBlock= np.squeeze([x, y, z])
                    #convert to array of coordinates
                    windowIndexes=np.vstack(windowBlock).reshape(3,-1).T
     
                    #probably dumb and inefficient, but here we are
                    voxelVotes=np.asarray([noIslandAtlasData[iVoxel[0],iVoxel[1],iVoxel[2]] for iVoxel in list(windowIndexes)])
                    #get the vote counts
                    (labels,counts)=np.unique(voxelVotes[np.isin(voxelVotes,inflationLabelTargets,invert=True)], return_counts=True)
                    voteWinner=labels[np.where(np.max(counts)==counts)[0]]
                    #just in case
                    del voxelVotes
                    window=window+1
                #print(str(noIslandAtlasData[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]))
                #print(str(voteWinner))
                newIDentityHolder[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner)
                del voteWinner
            
        #once the inflations are done for this round, set the values in the atlas data
        #to their new identities
        #2/25/2022 note:  previously we had been directly setting the identity
        #within the iInflationTargetCoords loop.  This was probably leading
        #to over-percolation of low number identities
        noIslandAtlasData[infationArrayTargets]=newIDentityHolder[infationArrayTargets]
        
        
        #return the data object to its regular size
        noIslandAtlasData=noIslandAtlasData[1:-1,1:-1,1:-1]
    
    inflatedAtlas=nib.Nifti1Image(noIslandAtlasData, atlas.affine, atlas.header)
    return inflatedAtlas

def extendROIinDirection(roiNifti,direction,iterations):
    """
    Extends a binarized, volumetric nifti roi in the specified directions.
    
    Potential midline issues
    
    ASSUMES ACPC ALIGNMENT
    
    NOTE: THIS CURRENTLY DOESN'T WORK RIGHT NOW., IT WOULD RESULT IN A WEIRD,
    BOX-CONSTRAINED INFLATION, DESIRED FUNCTIONALITY WOULD BE MORE LIKE
    ORTHOGONAL TRANSLATION

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
    concatData=ndimage.binary_dilation(roiNifti.get_data(), iterations=iterations)
    
    #convert that data array to a nifti
    extendedROI=nib.Nifti1Image(concatData, roiNifti.affine, roiNifti.header)
    
    #now remove all of the undesired proportions
    for iDirections in direction:
        #if the current direction isn't one of the desired expansion directions
        #go ahead and cut anything off that was expanded
        if not any([x in direction for x in boundaryLabels]):
            extendedROI=sliceROIwithPlane(extendedROI,boundaryDictionary[iDirections],iDirections)
    return extendedROI

def boundaryROIPlanesFromMask(inputMask):
    """
    creates a 6 item output dictionary featuring the planar borders which
    encompass the input mask

    Parameters
    ----------
    inputMask : nifti
        a 3D, nifti mask.  Presumably of a tract.  Can be binarized, 
        probabalistic (0-1 bound), or density based.

    Returns
    -------
    borderDict: dictionary
        A dictionary with keys ['medial','lateral','anterior','posterior','superior','inferior']
        corresponding to each of the planar borders of the input mask.

    """
    import nibabel as nib
    
    if isinstance(inputMask,str):
        inputMask=nib.load(inputMask)
        
    #create a binarized copy of the mask
    binarizedMask=nib.Nifti1Image(inputMask.get_data()>0, inputMask.affine, inputMask.header)
    
    #initialize the output directory
    borderDict={}
    
    #create list of borders x, y, z
    bordersList=['medial','lateral','anterior','posterior','superior','inferior']
    
    #iterate across dimensions
    
    for iBorders in range(len(bordersList)):
        #generate plane for current border of interest 
        currentBorderROI=planeAtMaskBorder(binarizedMask,bordersList[iBorders])
        borderDict[bordersList[iBorders]]=currentBorderROI
    
    #having generated those borders, return the dictionary
    return borderDict

def findROISintersection(ROIs,inflateiter=0):
    """
    Finds the intersection of input rois (if any), also permits inflation 
    should it be desired.  Potentially useful for finding the intersection/border
    between WM and GM, for example.
    
    Largely, a glorified wrapper around nilearn.masking.intersect_masks

    Parameters
    ----------
    ROIs : List of nifti objects
        DESCRIPTION.
    inflateiter : TYPE
        DESCRIPTION.

    Returns
    -------
    intersectionNifti: an output nifti, with the binarized intersection mask
    of the provided niftis

    """
    from nilearn import masking
    import numpy as np
    
    if len(ROIs)==1:
        raise Exception("Only one nifti provided; intersection of 1 item not sensible")
   
    #perform the inflation, if necessary
    for iROIs in range(len(ROIs)):
        #is this a recursive, convoluted nightmare, yes
        #is it efficient, also yes
        #a mask is _kind_ of like a atlas, just with 1 label, right?
        ROIs[iROIs]=multiROIrequestToMask(ROIs[iROIs],[1],inflateIter=inflateiter)   
    
    #perform the intersection operation
    intersectionNifti=masking.intersect_masks(ROIs, threshold=1)
    
    #if it's empty throw a warning
    if not np.any(intersectionNifti.get_data()):
        import warnings
        warnings.warn("Empty mask for intersection returned; likely no mutual intersection between input " + str(len(ROIs)) + "ROIs")
                      
    return intersectionNifti

def changeLabelValue(inputParc,targetLabelNum,newLabelNum):
    import numpy as np
    import nibabel as nib
    parcData=inputParc.get_data().astype(int)
    locations=np.where(parcData==targetLabelNum)
    #this takes a bewilderingly long time
    #I must be using it wrong
    #parcData[np.ix_(locations[0],locations[1],locations[2])]=newLabelNum
    for iToChange in range(len(locations[0])):
        parcData[locations[0][iToChange],locations[1][iToChange],locations[2][iToChange]]=newLabelNum
    #for 
    print(str(len(locations[0])) + ' labels changed from ' + str(targetLabelNum) +' to ' + str(newLabelNum))
    #create the output
    inputParc=nib.Nifti1Image(parcData, inputParc.affine, inputParc.header)
    return inputParc
    
def changeMultpleLabelValues(inputParc,targetLabelNums,newLabelNum):
    
    for iLabels in targetLabelNums:
        inputParc=changeLabelValue(inputParc,iLabels,newLabelNum)
        
    return inputParc



def mergeVolParcellations(parc1, parc2, parc1LUT, parc2LUT,parc1_override_labels=None,parc2_override_labels=None):
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import sys
    import wmaPyTools.analysisTools
    
    print('USAGE NOTE/RECOMMENDATION')
    print('inflation (via inflateAtlasIntoWMandBG) and label removal (via changeLabelValue) ')
    print('recommended PRIOR to mergeVolParcellations usage')
    from nilearn.image import crop_img, resample_img
    print('loading parcellations')
    #load the nifti files if necesary
    if isinstance(parc1,str):
        parc1=nib.load(parc1)
        
    if isinstance(parc2,str):
        parc2=nib.load(parc2)
        
    #load the lookup tables
    if isinstance(parc1LUT,str):
        parc1LUT=pd.read_csv(parc1LUT)
    if isinstance(parc2LUT,str):
        parc2LUT=pd.read_csv(parc2LUT)
    
   
    #need to do this to identify the relevant labels
    parc1LUT_reduced=wmaPyTools.analysisTools.reduceLUTtoAvail(parc1,parc1LUT,removeAbsentLabels=True,reduceRenameColumns=True)
    if not parc1_override_labels==None and np.any(parc1_override_labels):
        print('Reducing parc1 LUT and identifying overrides')
        #create an override name holder
        override1Names=[[] for iOverride in parc1_override_labels ]
        #collect the names
        for iterator,iOverride in enumerate(parc1_override_labels):
            #if it's actually in the atlas...
            if iOverride in parc1LUT_reduced.iloc[:,0].values:
                override1Names[iterator]=parc1LUT_reduced[parc1LUT_reduced.columns[1]].loc[parc1LUT_reduced.iloc[:,0]==iOverride].values[0]
            else:
                print (' Override request '+ str(iOverride) + ' not present in parc1' )
                
        print('Labels : ' + ' '.join(override1Names) + ' will serve as overrides for parc1')
    #honestly we don't need to do this for atlas 1 due to the default behavior... but so long as we allow parc1_override_labels inputs...
    
    #do the same for atlas 2
    parc2LUT_reduced=wmaPyTools.analysisTools.reduceLUTtoAvail(parc2,parc2LUT,removeAbsentLabels=True,reduceRenameColumns=True)
    if not parc2_override_labels==None and np.any(parc2_override_labels):
        print('Reducing parc2 LUT and identifying overrides')
        #create an override name holder
        override2Names=[[] for iOverride in parc2_override_labels ]
        #collect the names
        for iterator,iOverride in enumerate(parc2_override_labels):
            #if it's actually in the atlas...
            if iOverride in parc2LUT_reduced.iloc[:,0].values:
                override2Names[iterator]=parc2LUT_reduced[parc2LUT_reduced.columns[1]].loc[parc2LUT_reduced.iloc[:,0]==iOverride].values[0]
            else:
                print (' Override request '+ str(iOverride) + ' not present in parc2' )
        print('Labels : ' + ' '.join(override2Names) + ' will serve as overrides for parc2')
    #coordinate the labeling schemes
    print('coordinating atlas lookuptables')
    combinedLut, parc1, parc2=wmaPyTools.analysisTools.coordinateLUTsAndAtlases(parc1LUT,parc2LUT,parc1,parc2)
    
    
    #invert the transform you just did to get the new label values
    if not parc1_override_labels==None and np.any(parc1_override_labels):
        print('back converting rquested overrides for parc1 into new (coordinated) label values')
        #create an override name holder
        newLabels1=[[] for iOverride in parc1_override_labels ]
        #collect the names
        for iterator,iOverride in enumerate(override1Names):
            if iOverride in combinedLut.iloc[:,1].values:
                newLabels1[iterator]=combinedLut[combinedLut.columns[0]].loc[combinedLut.iloc[:,1]==iOverride].values[0]
        #reset the input override labels
        parc1_override_labels=newLabels1
        print('New label values associated with parc1:' + str(newLabels1))
        
    #do the same for the other parcellation
    if not parc2_override_labels==None and np.any(parc2_override_labels):
        print('back converting rquested overrides for parc2 into new (coordinated) label values')
        #create an override name holder
        newLabels2=[[] for iOverride in parc2_override_labels ]
        #collect the names
        for iterator,iOverride in enumerate(override2Names):
            if iOverride in combinedLut.iloc[:,1].values:
                newLabels2[iterator]=combinedLut[combinedLut.columns[0]].loc[combinedLut.iloc[:,1]==iOverride].values[0]
        #reset the input override labels
        parc2_override_labels=newLabels2
        print('New label values associated with parc2:' + str(parc2_override_labels))
    
    
    #detect conflicts
    #extract data
    parc1Data=parc1.get_data().astype(int)
    parc2Data=parc2.get_data().astype(int)
    
    #find out which one is bigger, use datasize as the guide.  Datatype shouldn't
    #be a confound at this point because we converted to int
    parc1DataSize=sys.getsizeof(parc1Data)
    parc2DataSize=sys.getsizeof(parc2Data)
    #resample to the larger one
    print('matching data block sizes')
    #0 values may cause a confound with this?
    if parc1DataSize>parc2DataSize:
        print ('parc1 size of ' + str(np.round(parc1DataSize/(1024**2))) + ' MB greater than parc2 size of ' + str(np.round(parc2DataSize/(1024**2))) + ' MB')
        print ('resampling to parc1' )
        parc2=resample_img(parc2,target_affine=parc1.affine,target_shape=(parc1.shape),interpolation='nearest')
        parc2Data=parc2.get_data().astype(int)
    elif parc2DataSize>parc1DataSize:
        print ('parc2 size of ' + str(np.round(parc2DataSize/(1024**2))) + ' MB greater than parc1 size of ' + str(np.round(parc1DataSize/(1024**2))) + ' MB')
        print ('resampling to parc2' )
        parc1=resample_img(parc1,target_affine=parc2.affine,target_shape=(parc2.shape),interpolation='nearest')
        parc1Data=parc1.get_data().astype(int)
    elif parc1Data.shape==parc2Data.shape:
        print ('Input parcellations same data shape, proceeding without modification')
    else:
        raise ValueError('Unexpected edge case detected \n both parcellations same size, but different dimensions')
    
    #find locations of labels
    labelIndexes1=np.where(parc1Data)
    labelIndexes2=np.where(parc2Data)
    
    #convert them to lists of lists
    labelIndexes1_list=list(np.asarray(labelIndexes1).T)
    labelIndexes2_list=list(np.asarray(labelIndexes2).T)
    
    
    
    conflictLocations=[list(x) for x in set(tuple(x) for x in labelIndexes1_list).intersection(set(tuple(x) for x in labelIndexes2_list))]
    print(str(len(conflictLocations)) + ' total labeling conflicts detected')
    print('Conflicts detected for the following locations:')
    conflictDF=pd.DataFrame(columns=['coordinate','parc1ID','parc2ID','parcIDselected'])
    
    for iterator,iConflicts in enumerate(conflictLocations):
        #enter the conflict location
        conflictDF.at[iterator,'coordinate']=iConflicts
        #identify the labels in each
        parc1ID=parc1Data[iConflicts[0],iConflicts[1],iConflicts[2]]
        parc2ID=parc2Data[iConflicts[0],iConflicts[1],iConflicts[2]]
        #enter the information into the data frame
        conflictDF.at[iterator,'parc1ID']=parc1ID
        conflictDF.at[iterator,'parc2ID']=parc2ID
        #adjudicate the victor
        #but throw an error first if you've received inchoherent inputs
        if not parc1_override_labels==None and  parc2_override_labels==None:
            if parc1ID in parc1_override_labels and parc2ID in parc2_override_labels:
                raise ValueError('Parcellation1 label ' + str(parc1ID) + ' Parcellation2 label ' + str(parc2ID) + ' both entered as overrides.  Adjudicate discrepancy and rerun')
        
        if not parc2_override_labels==None:
            if parc2ID in parc2_override_labels:
                conflictDF.at[iterator,'parcIDselected']=parc2ID
        if not parc1_override_labels==None:
            if parc1ID in parc1_override_labels:
                conflictDF.at[iterator,'parcIDselected']=parc1ID
        #note this defaults to parc1 when conflicts are not adjudicated explicitly
        #else:
            #this is pointless, don't do this
            #conflictDF.at[iterator,'parcIDselected']=parc1ID
           
    #given that we are using the parc1 as the default, we can remove those values
    #from the conflict DF in order to expidite the merge
    toDeleteVec=np.zeros(len(conflictDF), dtype=bool)
    for iConflicts in range(len(conflictDF)):
        if conflictDF['parc1ID'].iloc[iConflicts]==conflictDF['parcIDselected'].iloc[iConflicts] or np.isnan(conflictDF['parcIDselected'].iloc[iConflicts]):
            toDeleteVec[iConflicts]=True
    
    #now remove them from the dataFrame
    toRemoveDF=conflictDF.loc[np.logical_not(toDeleteVec)]
        
    outData=parc1Data    
    print(str(len(toRemoveDF)) + ' voxel labels will be changed')
    #however, for the conflicts we have to iterate and adjudicate
    for iterator in range(len(toRemoveDF)):
        outData[toRemoveDF['coordinate'].iloc[iterator][0],toRemoveDF['coordinate'].iloc[iterator][1],toRemoveDF['coordinate'].iloc[iterator][2]]=toRemoveDF['parcIDselected'].iloc[iterator]
            
    #convert it back into a nifti
    outParcNifti=nib.Nifti1Image(outData, parc1.affine,parc1.header)
    #pass them out
    return outParcNifti , combinedLut , toRemoveDF
        
            
            
#I CANT FIGURE THIS OUT RIGHT NOW
# def resampleNoZero(img, target_affine=None, target_shape=None, interpolation='nearest', copy=True, order='F', clip=True, fill_value=0, force_resample=False):
#     #a wrapped version of nilearn.image.resample_img with interpolation='nearest' that doesn't let 0 values influence the interpolation
#     from nilearn.image import crop_img, resample_img
#     import numpy as np
#     imgData=np.asanyarray(img.dataobj)
#     continuousReampleNifti=resample_img(img,target_affine,target_shape,interpolation='continuous' )
#     continuousReampleData=np.asanyarray(continuousReampleNifti.dataobj)
#     nearestReampleNifti=resample_img(img,target_affine,target_shape,interpolation='nearest' )
#     nearestReampleData=np.asanyarray(nearestReampleNifti.dataobj)
#     dataDiff=continuousReampleData-nearestReampleData
    
    
#     #fill in with an absurd value
#     imgData[imgData]
    
    