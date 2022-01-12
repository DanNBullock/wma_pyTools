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
    atlas : TYPE
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
                      'medial':   np.min(fullVolumeBoundCoords[np.min(np.abs(fullVolumeBoundCoords[:,0]))==np.abs(fullVolumeBoundCoords[:,0]),0]),
                      'lateral': np.max(fullVolumeBoundCoords[np.max(np.abs(fullVolumeBoundCoords[:,0]))==np.abs(fullVolumeBoundCoords[:,0]),0]),
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


    planeCoords=[]
    #i guess the only way
    for iDims in list(range(len(inputROINifti.shape))):
        #set step size at half the voxel length in this dimension
        stepSize=fullMask.header.get_zooms()[iDims]*.5
        if iDims==dimensionDict[relativePosition]:
            #kind of wishy,washy, but because we halved the step size in the previous step
            #by taking the average of the coord bounds (in subject space) we should actually be fine
            #we'll use this as one of our two bounds
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

def removeIslandsFromAtlas(atlasNifti):
    """
    Removes island label values from atlas iteratively across values present.
    Important for when extracted atlas labels are being used as anatomical 
    markers.
    
    Potentially faster ways of doing this in other packages.
    
    Parameters
    ----------
    atlasNifti : TYPE
        A multi label atlas, with integer based labels.  Will load if string
        is passed

    Returns
    -------
    atlasNiftiOut : TYPE
        The input atlas nifti, but with islands removed
    removalReport : TYPE
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

def inflateAtlasIntoWMandBG(atlasNifti,iterations):
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
    import tqdm
    #import dipy.tracking.utils as ut
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti
    
    #first perform the island removal operation
    [noIslandAtlas,report]=removeIslandsFromAtlas(atlasNifti)
    
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
    inflationLabelTargets=list(presumedWMLabels)+list(np.atleast_1d(detectBackground))
    #also get remaining labels
    remaining=nonBGLabels[np.isin(nonBGLabels,inflationLabelTargets,invert=True)]
    
    #now iteratively perform the inflation
    
    #so long as nothing is on the edge this will probably work, due to coord weirdness
    #This was forshadowing, code updated on 1/12/22 to take care of this
    #get the image data
    noIslandAtlasData=noIslandAtlas.get_data()
    noIslandAtlasData=np.round(noIslandAtlasData).astype(int)
    
    for iInflations in range(iterations):
        print('inflation iteration ' + str(iInflations+1))
        #pad the data to avoid indexing issues
        noIslandAtlasData=np.pad(noIslandAtlasData,[1,1])
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
                window=window+1
            #print(str(noIslandAtlasData[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]))
            #print(str(voteWinner))
            noIslandAtlasData[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner)
        
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
    if not (1 in np.unique(intersectionNifti.get_data)) or (True in np.unique(intersectionNifti.get_data)):
        import warnings
        warnings.warn("Empty mask for intersection returned; likely no mutual intersection between input " + str(len(ROIs)) + "ROIs")
                      
    return intersectionNifti