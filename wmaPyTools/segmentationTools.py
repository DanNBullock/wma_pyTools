# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:51 2021

@author: Daniel
"""
def speedSortSegCriteria(criteriaROIs,inclusionCriteria,operations):
    """
    Sorts segmentation criteria in order to minimize segmentation time.
    Acheives this by sorting criteria by how fast/harsh they are.  Endpoint
    criteria applied first, because they are only computed on endpoints.  Then
    for standard traversal ROIS, smallest are placed before larger.  This is 
    because (1) fewer streamlines are likely to pass through smaller ROIs,
    and so this is expected to be fast and (2), when paired with a bounding
    box implementation, greatly reduces the numbr of vreticies that are
    computed against.

    The ordering of the input vectors is presumed to be locked to one another.

    Parameters
    ----------
    criteriaROIs : TYPE
        A list of nifti ROI criteria
    inclusionCriteria : TYPE
        The boolean vector indicating whether these rois will operate as
    exclusion or inclusion ROIs
    operations : TYPE
        The vector of strings indicating which operations will be performed

    Returns
    -------
    sortedCriteriaROIs : TYPE
        A sorted list of nifti ROI criteria
    sortedInclusionCriteria : TYPE
        The sorted boolean vector indicating whether these rois will operate as
    exclusion or inclusion ROIs
    sortedOperations : TYPE
        The sorted vector of strings indicating which operations will be 
    performed

    """
    #also remember all = [some criteria] == any != [some criteria]
    #we could auto convert those as well to speed things up further.
    
    
    import numpy as np
    
    #initialize an output order vector
    outOrder=list(range(len(criteriaROIs)))
    
    #for each of the input ROIs compute the number of voxels that will be checked
    volumesToCheck=np.zeros(len(criteriaROIs))
    for iROIs in range(len(criteriaROIs)):
        #compute the volume that needs to be checked
        volumesToCheck[iROIs]=np.sum(criteriaROIs[iROIs].get_fdata())

    print("volumes of input ROIs to check " + str(volumesToCheck))
    #find the criteria that only involve endpoints (you don't have to check all the nodes
    #for these, only the endpoints, so it's way faster to do cdist)
    endpointCriteria=['end' in x for x in operations]
    #where are these
    endpointCriteriaIndexes=np.where(endpointCriteria)[0]
    #sort these in accordance with the relevant volumes
    sortOrder=endpointCriteriaIndexes[np.argsort(volumesToCheck[endpointCriteriaIndexes]) ]
    
    #go ahead and place these in the output order vec
    for endpointCriteriaIndexes in range(len(endpointCriteriaIndexes)):
        outOrder[endpointCriteriaIndexes]=sortOrder[endpointCriteriaIndexes]
        
    #now do the same thing for the regular criteria
    allNodeCriteria=[('any' in x) or ('all' in x) for x in operations]
    #where are these
    allNodeCriteriaIndexes=np.where(allNodeCriteria)[0]
    #sort these in accordance with the relevant volumes
    sortOrder=allNodeCriteriaIndexes[np.argsort(volumesToCheck[allNodeCriteriaIndexes]) ]
    #find the index of where we should begin filling these in
    remainFillIndex=len(allNodeCriteriaIndexes)
    #fill in the remaining values
    for iRemainIndexes in range(len(outOrder)-remainFillIndex,len(outOrder)):
        outOrder[iRemainIndexes]=sortOrder[iRemainIndexes-(len(outOrder)-remainFillIndex)]
    
    #initialize output objects
    sortedCriteriaROIs=[]
    sortedInclusionCriteria=[]
    sortedOperations=[]
    #resort the inputs in accordance with predicted stringency
    volumesToCheckOut=np.zeros(len(outOrder)).astype(int)
    for iOutputs in range(len(outOrder)):
        sortedCriteriaROIs.append(criteriaROIs[outOrder[iOutputs]])
        sortedInclusionCriteria.append(inclusionCriteria[outOrder[iOutputs]])
        sortedOperations.append(operations[outOrder[iOutputs]])
        
        volumesToCheckOut[iOutputs]=np.sum(sortedCriteriaROIs[iOutputs].get_fdata()).astype(int)
        #if it is an exclusion, get the number of streamlines that are left out
    print("volumes of output ROIs to check " + str(volumesToCheckOut))
    
    return sortedCriteriaROIs,sortedInclusionCriteria, sortedOperations
    
def segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -roisvec: a list of nifti objects that will serve as your ROIs
    #
    # -includeVec: a boolean list indicating whether you want the associated roi to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # operationsVec: a list with any of the following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    # -outBoolVec: boolean vec indicating streamlines that survived ALL operations
    #
    # NOTE: roisvec, includeVec, and operationsVec should all be the same lenth
    # ADVICE: apply the harshest (fewest survivors) criteria first.  Will result 
    # in signifigant speed ups.
    # ADVICE: starting with specifying endpoints has the additional benefit of reducting
    # the number of nodes considered per streamline to 2.  This would be an effective way
    # of implementing a fast and harsh first criteria.
    """
    import numpy as np
    import itertools 
    
    if ~ len(np.unique([len(roisvec), len(includeVec), len(operationsVec)]))==1:
        raise ValueError('mismatch between lengths of roi, inclusion, and operation vectors')

    #ok, but, if you were clever, you would sort the criterion now in the following way:
    #First do the endpoint criteria, and of those, do those in the order of smallest ROI to largest
    #Also be sure to account for the negation logic as well
    #what you're aiming for is to do the criterion in order of harshest to least harsh
    [roisvec,includeVec,operationsVec]=speedSortSegCriteria(roisvec,includeVec,operationsVec)
    
    #ok, lets do this battle royale style
    #here's our starting "class"
    remainingStreamIndexes=list(range(len(streamlines)))
    
    for iOperations in list(range(len(roisvec))):
        
        #perform this segmentation operation
        curBoolVec=applyNiftiCriteriaToTract_DIPY_Test(streamlines[remainingStreamIndexes], roisvec[iOperations], includeVec[iOperations], operationsVec[iOperations])
        #after the cull
        remainingStreamIndexes=list(itertools.compress(remainingStreamIndexes,curBoolVec))
        
    #create blank output structure
    outBoolVec=np.zeros(len(streamlines),dtype=bool)
    #these are the winners
    outBoolVec[remainingStreamIndexes]=True
    return outBoolVec
   

def applyNiftiCriteriaToTract_DIPY(streamlines, maskNifti, includeBool, operationSpec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria, uses a range of dipy functions
    #and custom made functions to expedite the typically slow segmentation process
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -maskNifti: a nifti Mask containing only 1s and 0s
    #
    # -includeBool: a boolean indicator of whether you want the associated ROI to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # -operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    """
    #still learning how to import from modules
    from dipy.tracking.utils import near_roi
    import numpy as np
    import dipy.tracking.utils as ut
    import nibabel as nib
    from nilearn import masking 
    import scipy
    
    #perform some input checks
    validOperations=["any","all","either_end","both_end"]
    if np.logical_not(np.in1d(operationSpec, validOperations)):
         raise Exception("applyNiftiCriteriaToTract Error: input operationSpec not understood.")
    
    if np.logical_not(type(maskNifti).__name__=='Nifti1Image'):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not a nifti.")
    
    #the conversion to int may cause problems if the input isn't convertable to int.  Then again, the point of this is to raise an error, so...
    elif np.logical_not(np.all(np.unique(maskNifti.get_fdata()).astype(int)==[0, 1])): 
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not convertable to 0,1 int mask.  Likely not a mask.")
        
    if np.logical_not(isinstance(includeBool, bool )):
        raise Exception("applyNiftiCriteriaToTract Error: input includeBool not a bool.  See input description for usage")
        
    #in order to achieve a speedup we should minimize the number of coordinates we are testing.
    #let's intersect the ROI and a mask of the streamlines so that we can subset the relevant portions/voxels of the ROI
    tractMask=ut.density_map(streamlines, maskNifti.affine, maskNifti.shape)
    #dialate tract mask in order to include voxels that are outside of the 
    #tractmask, but nonetheless within the tolerance.
    tractMask=scipy.ndimage.binary_dilation(tractMask.astype(bool), iterations=1)
    #convert int-based voxel-wise count data to int because apparently that's the only way you can save with nibabel
    tractMask=nib.Nifti1Image(np.greater(tractMask,0).astype(int), affine=maskNifti.affine)

    #take the intersection of the two, the tract mask and the ROI mask nifti.
    #ideally we are reducing the number of points that each streamline node needs to be compared to
    #for example, a 256x 256 planar ROI would result in 65536 coords,
    #intersecting with a full brain tractogram reduces this by about 1/3
    #because we are not bothering with the roi coordinates that are outside
    #the whitematter mask
    tractMaskROIIntersection=masking.intersect_masks([tractMask,maskNifti], threshold=1, connected=False)

    #we can obtain a speedup with respect to the tractogram as well, by only
    #considering those streamlines that plausably occupy the same bounding box
    #as the ROI.  The use of a bounding box is predicated upon the assumption that
    #assessing whether any node coordinate is within a specified bounds
    # (i.e. B_D_1>C_I_N_D>B_D_2; I=streamline index, N=node index, D=dimension index, b=bound)
    #is sufficiently fast *and* specifc (in that it exclusdes a sufficient number of stremalines)
    #to justify this additional round of computation
    
    #if we could reduce this to only those streamline-nodes that are within the bounding box
    #we could speed this up further.
    
    #find the streamlines that are within the bounding box of the maskROI,
    #NOTE: this isn't necessarily the full mask input by the user
    boundedStreamsBool=subsetStreamsByROIboundingBox(streamlines, tractMaskROIIntersection)
    
    #subset them
    boundedStreamSubset=streamlines[np.where(boundedStreamsBool)[0]]
    #second track mask application doesn't seem to do anything    
    
    #use dipy's near roi function to generate bool
    criteriaStreamsBool=near_roi(boundedStreamSubset, tractMaskROIIntersection.affine, tractMaskROIIntersection.get_fdata().astype(bool), mode=operationSpec)
       
    #find the indexes of the original streamlines that the survivors correspond to
    boundedStreamsIndexes=np.where(boundedStreamsBool)[0]
    originalIndexes=boundedStreamsIndexes[np.where(criteriaStreamsBool)[0]]
    
    if includeBool==True:
        #initalize an out bool vec
        outBoolVec=np.zeros(len(streamlines))
        #set the relevant entries to true
        outBoolVec[originalIndexes]=True
    elif includeBool==False:          
        #initalize an out bool vec
        outBoolVec=np.ones(len(streamlines))
        #set the relevant entries to true
        outBoolVec[originalIndexes]=False
    
    return outBoolVec

def applyNiftiCriteriaToTract_DIPY_Test(streamlines, maskNifti, includeBool, operationSpec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria, uses a range of dipy functions
    #and custom made functions to expedite the typically slow segmentation process
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -maskNifti: a nifti Mask containing only 1s and 0s
    #
    # -includeBool: a boolean indicator of whether you want the associated ROI to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # -operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    """
    #still learning how to import from modules
    from dipy.tracking.utils import near_roi
    import numpy as np
    import dipy.tracking.utils as ut
    import nibabel as nib
    from nilearn import masking 
    import scipy
    import time
    
    #perform some input checks
    validOperations=["any","all","either_end","both_end"]
    if np.logical_not(np.in1d(operationSpec, validOperations)):
         raise Exception("applyNiftiCriteriaToTract Error: input operationSpec not understood.")
    
    if np.logical_not(type(maskNifti).__name__=='Nifti1Image'):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not a nifti.")
    
    #the conversion to int may cause problems if the input isn't convertable to int.  Then again, the point of this is to raise an error, so...
    elif np.logical_not(np.all(np.unique(maskNifti.get_fdata()).astype(int)==[0, 1])): 
        if np.all(np.unique(maskNifti.get_fdata()).astype(int)[0]==0):
            import warnings
            warnings.warn("input mask nifti empty.")
        else:
            raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not convertable to 0,1 int mask.  Likely not a mask.")
        
    if np.logical_not(isinstance(includeBool, bool )):
        raise Exception("applyNiftiCriteriaToTract Error: input includeBool not a bool.  See input description for usage")
        
    #in order to achieve a speedup we should minimize the number of coordinates we are testing.
    #let's intersect the ROI and a mask of the streamlines so that we can subset the relevant portions/voxels of the ROI
    tractMask=ut.density_map(streamlines, maskNifti.affine, maskNifti.shape)
    #dialate tract mask in order to include voxels that are outside of the 
    #tractmask, but nonetheless within the tolerance.
    tractMask=scipy.ndimage.binary_dilation(tractMask.astype(bool), iterations=1)
    #convert int-based voxel-wise count data to int because apparently that's the only way you can save with nibabel
    tractMask=nib.Nifti1Image(np.greater(tractMask,0).astype(int), affine=maskNifti.affine)

    #take the intersection of the two, the tract mask and the ROI mask nifti.
    #ideally we are reducing the number of points that each streamline node needs to be compared to
    #for example, a 256x 256 planar ROI would result in 65536 coords,
    #intersecting with a full brain tractogram reduces this by about 1/3
    #because we are not bothering with the roi coordinates that are outside
    #the whitematter mask
    tractMaskROIIntersection=masking.intersect_masks([tractMask,maskNifti], threshold=1, connected=False)

    #we can obtain a speedup with respect to the tractogram as well, by only
    #considering those streamlines that plausably occupy the same bounding box
    #as the ROI.  The use of a bounding box is predicated upon the assumption that
    #assessing whether any node coordinate is within a specified bounds
    # (i.e. B_D_1>C_I_N_D>B_D_2; I=streamline index, N=node index, D=dimension index, b=bound)
    #is sufficiently fast *and* specifc (in that it exclusdes a sufficient number of stremalines)
    #to justify this additional round of computation
    
    
    #find the streamlines that are within the bounding box of the maskROI,
    #NOTE: this isn't necessarily the full mask input by the user, as a result of the
    #intersection with the tract mask
    

    #maybe not catching the streams will save ram?
    #ok, but you can probably do this differently if you're just doing the endpoints
    if 'end' in operationSpec:
        #or [[streamline[0,:],streamline[-1,:]] for streamline in streamlines]
        [outIndexes, outStreams]   =subsetStreamsNodesByROIboundingBox([np.asarray([streamline[0,:],streamline[-1,:]]) for streamline in streamlines], tractMaskROIIntersection)
    else:
        [outIndexes, outStreams]   =subsetStreamsNodesByROIboundingBox(streamlines, tractMaskROIIntersection)
    #use dipy's near roi function to generate bool
    #start timing
    t1_start=time.process_time()
    criteriaStreamsBool=near_roi(outStreams, tractMaskROIIntersection.affine, tractMaskROIIntersection.get_fdata().astype(bool), mode=operationSpec)
    #stop timing
    t1_stop=time.process_time()
    # get the elapsed time
    modifiedTime=t1_stop-t1_start
    if includeBool:
        print('Tractogram segmentation complete in ' +str(modifiedTime) +', '+str(np.sum(criteriaStreamsBool)) + ' of ' + str(len(streamlines)) + ' met INCLUSION criterion')
    else:
        print('Tractogram segmentation complete in ' +str(modifiedTime) +', '+str(np.sum(criteriaStreamsBool)) + ' of ' + str(len(streamlines)) + ' met EXCLUSION criterion')
   

    originalIndexes=outIndexes[np.where(criteriaStreamsBool)[0]]
    
    if includeBool==True:
        #initalize an out bool vec
        outBoolVec=np.zeros(len(streamlines)).astype(bool)
        #set the relevant entries to true
        outBoolVec[originalIndexes]=True
    elif includeBool==False:          
        #initalize an out bool vec
        outBoolVec=np.ones(len(streamlines)).astype(bool)
        #set the relevant entries to true
        outBoolVec[originalIndexes]=False
    
    return outBoolVec.astype(bool)

def subsetStreamsByROIboundingBox(streamlines, maskNifti):
    """subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, which will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    """
    #compute distance tolerance
    from dipy.core.geometry import dist_to_corner
    import time
    import wmaPyTools.roiTools 
    
    #begin timing
    t1_start=time.process_time()
    
    #use distance to corner to set tolerance
    dtc = dist_to_corner(maskNifti.affine)
    
    #convert them to subject space
    subjectSpaceBounds=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(maskNifti)
    #expand to accomidate tolerance
    subjectSpaceBounds[0,:]=subjectSpaceBounds[0,:]-dtc
    subjectSpaceBounds[1,:]=subjectSpaceBounds[1,:]+dtc
    
    #map and lambda function to determine whether each streamline is within the bounds
    criteriaVec=list(map(lambda streamline: streamlineWithinBounds(streamline,subjectSpaceBounds), streamlines))
    
    #stop timing
    t1_stop=time.process_time()
    # get the elapsed time
    modifiedTime=t1_stop-t1_start
    
    print('Tractogram subseting complete in ' +str(modifiedTime) +', '+str(sum(criteriaVec)) + ' of ' + str(len(streamlines)) + ' within mask boundaries')
    return criteriaVec

def subsetStreamsNodesByROIboundingBox(streamlines, maskNifti):
    """subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, which will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    """
    #compute distance tolerance
    from dipy.core.geometry import dist_to_corner
    from dipy.tracking.streamline import Streamlines
    import numpy as np
    import time
    import wmaPyTools.roiTools  
    
    #begin timing
    t1_start=time.process_time()
    
    #use distance to corner to set tolerance
    dtc = dist_to_corner(maskNifti.affine)
    
    #convert them to subject space
    subjectSpaceBounds=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(maskNifti)
    #expand to accomidate tolerance
    subjectSpaceBounds[0,:]=subjectSpaceBounds[0,:]-dtc
    subjectSpaceBounds[1,:]=subjectSpaceBounds[1,:]+dtc
    
    #map and lambda function to extract the nodes within the bounds
    criteriaVec=list(map(lambda streamline: streamlineNodesWithinBounds(streamline,subjectSpaceBounds), streamlines))
    outIndexes=np.where(list(map(lambda x: x.size>0, criteriaVec)))[0]
    outStreams=Streamlines(criteriaVec)
    
    #stop timing
    t1_stop=time.process_time()
    # get the elapsed time
    modifiedTime=t1_stop-t1_start
    
    print('Tractogram subseting complete in ' +str(modifiedTime) +', '+str(len(outIndexes)) + ' of ' + str(len(streamlines)) + ' within mask boundaries')
    return outIndexes, outStreams

def streamlineWithinBounds(streamline,bounds):
    """ determine whether **any** node of the input streamline is within the specified bounds.
    Returns a single bool
    Args:
        -streamline: an n by d shaped array where n=the number of nodes and d = the dimensionality of the streamline
        
        -bounds: a 2 x d array specifying the coordinate boundaries (in the pertinant space of the streamline) for assesment

    Output:
        withinBoundsBool:  a boolean value indicating whether the input streamline satisfies the within-bounds criteria
    
    """
    import numpy as np

    #see which nodes are between the bounds
    nodeCriteria=np.asarray([np.logical_and(streamline[:,iDems]>bounds[0,iDems],streamline[:,iDems]<bounds[1,iDems]) for iDems in list(range(bounds.shape[1])) ])
    
    #return true if any single node is between all three sets of bounds
    return np.any(np.all(nodeCriteria,axis=0))

def streamlineNodesWithinBounds(streamline,bounds):
    """ determine whether **any** node of the input streamline is within the specified bounds.
    Returns the nodes within the bounds.
    Args:
        streamline: an n by d shaped array where n=the number of nodes and d = the dimensionality of the streamline
        bounds: a 2 x d array specifying the coordinate boundaries (in the pertinant space of the streamline) for assesment

    Output:
        withinBoundsNodes:  an array of the nodes 
    
    """
    import numpy as np
    
    #see which nodes are between the bounds
    nodeCriteria=np.asarray([np.logical_and(streamline[:,iDems]>bounds[0,iDems],streamline[:,iDems]<bounds[1,iDems]) for iDems in list(range(bounds.shape[1])) ])
    
    #return return the viable nodes, could and often will be empty
    return streamline[np.all(nodeCriteria,axis=0)]

def applyEndpointCriteria(streamlines,planarROI,requirement,whichEndpoints):
    """ apply a relative location criteria to the endpoints of all streamlines in a collection of streamlines
    Args:
        streamlines: streamlines to be segmented from
        planarROI: the planar ROI relative to which the streamlines' endpoints' locations should be assessed
        requirement:  A relative anatomical positional term
        whichEndpoints:  whether this criteria should apply to 'both', 'one', or 'neither' endpoint
    
    Output:
        streamBool:  a boolean vector indicating which streamlines meet the specified criteria.
    """
    import numpy as np
    import nibabel as nib
    import wmaPyTools.roiTools 
    
    fullMask = nib.nifti1.Nifti1Image(np.ones(planarROI.get_fdata().shape), planarROI.affine, planarROI.header)
    #obtain boundary coords in subject space in order set max min values for interactive visualization
    convertedBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(fullMask)

    #implement test to determine if input planar roi is indeed planar
    #get coordinates of mask voxels in image space
    
    from dipy.tracking.utils import apply_affine
    planeSubjCoords=apply_affine(planarROI.affine, np.array(np.where(planarROI.get_fdata())).T)
    #find the unique values of img space coordinates for each dimension
    uniqueCoordCounts=[len(np.unique(planeSubjCoords[:,iCoords])) for iCoords in list(range(planeSubjCoords.shape[1]))]
    #one of them should be singular in the case of a planar roi, throw an error if not
    if ~np.any(np.isin(uniqueCoordCounts,1)):
        raise ValueError('input ROI not planar (i.e. single voxel thick for True values)')
    
    planeCoord=planeSubjCoords[0,np.where(np.equal(uniqueCoordCounts,1))[0][0]]

    #set up the dictionary for boundaries
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
    
    #throw an error if there's a mismatch 
    if  np.logical_not(np.where(np.equal(uniqueCoordCounts,1))[0][0]==dimensionDict[requirement]):
        raise Exception("applyEndpointCriteria Error: input relative position " + requirement + " not valid for input plane.")
    
    #create blank structure for endpoints
    endpoints=np.zeros((len(streamlines),6))
    #get the endpoints, taken from
    #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
    for iStreamline in range(len(streamlines)):
        #remember, first 3 = endpoint 1, last 3 = endpoint 2    
        endpoints[iStreamline,:]= np.concatenate([streamlines[iStreamline][0,:], streamlines[iStreamline][-1,:]])
    
    Endpoints1=endpoints[:,0:3]
    Endpoints2=endpoints[:,3:7]
    
    #sort the bounds
    sortedBounds=np.sort([planeCoord,positionTermsDict[requirement]])
    #get the relevant image dimension
    spaceDim=np.where(np.equal(uniqueCoordCounts,1))[0][0]
    
    #apply the criteria to both endpoints
    endpoints1Criteria=np.logical_and(np.greater(Endpoints1[:,spaceDim],sortedBounds[0]),np.less(Endpoints1[:,spaceDim],sortedBounds[1]))
    endpoints2Criteria=np.logical_and(np.greater(Endpoints2[:,spaceDim],sortedBounds[0]),np.less(Endpoints2[:,spaceDim],sortedBounds[1]))
    
    whichEndpointsDict={'neither': 0,
                        'one': 1,
                        'both':   2}
    
    
    #sum the two endpoint criterion vectors
    sumVec=np.add(endpoints1Criteria,endpoints2Criteria,dtype=int)
    
    #see where the target value is met
    targetMetVec=sumVec==whichEndpointsDict[whichEndpoints]
    
    return targetMetVec
    
def applyMidpointCriteria(streamlines,planarROI,requirement):
    """ apply a relative location criteria to the midpoints of all streamlines in a collection of streamlines
    Args:
        streamlines: streamlines to be segmented from
        planarROI: the planar ROI relative to which the streamlines' midpoints' locations should be assessed
        requirement:  A relative anatomical positional term

    Output:
        streamBool:  a boolean vector indicating which streamlines meet the specified criteria.
    """
    import numpy as np
    import nibabel as nib
    import wmaPyTools.roiTools  
    
    fullMask = nib.nifti1.Nifti1Image(np.ones(planarROI.get_fdata().shape), planarROI.affine, planarROI.header)
    #obtain boundary coords in subject space in order set max min values for interactive visualization
    convertedBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(fullMask)

    #implement test to determine if input planar roi is indeed planar
    #get coordinates of mask voxels in image space
    
    from dipy.tracking.utils import apply_affine
    planeSubjCoords=apply_affine(planarROI.affine, np.array(np.where(planarROI.get_fdata())).T)
    #find the unique values of img space coordinates for each dimension
    uniqueCoordCounts=[len(np.unique(planeSubjCoords[:,iCoords])) for iCoords in list(range(planeSubjCoords.shape[1]))]
    #one of them should be singular in the case of a planar roi, throw an error if not
    if ~np.any(np.isin(uniqueCoordCounts,1)):
        raise ValueError('input ROI not planar (i.e. single voxel thick for True values)')
    
    planeCoord=planeSubjCoords[0,np.where(np.equal(uniqueCoordCounts,1))[0][0]]

    #set up the dictionary for boundaries
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
    
    #throw an error if there's a mismatch 
    if  np.logical_not(np.where(np.equal(uniqueCoordCounts,1))[0][0]==dimensionDict[requirement]):
        raise Exception("applyEndpointCriteria Error: input relative position " + requirement + " not valid for input plane.")

    #use dipy to get the midpoints    
    from dipy.segment.metric import MidpointFeature
    feature = MidpointFeature()
    midpoints = np.squeeze(np.asarray(list(map(feature.extract, streamlines))))
    
    
    #sort the bounds
    sortedBounds=np.sort(planeCoord,positionTermsDict[requirement])
    #get the relevant image dimension
    spaceDim=np.where(np.equal(uniqueCoordCounts,1))[0][0]
    
    #apply the criteria to both endpoints
    midpointsCriteria=np.logical_and(np.greater(midpoints[:,spaceDim],sortedBounds[0]),np.less(midpoints[:,spaceDim],sortedBounds[1]))

    return midpointsCriteria

def maskMatrixByBoolVec(dipyGrouping,boolVec):
    """ recompute a connectivity matrix for a specified subset of the streamlines
    Args:
        dipyGrouping: the grouping output of the dipy utils.connectivity_matrix [WITH symmetric=False]
        boolVec: a boolean vector indicating which streamlines to consider in this computation
        

    Output:
        matrixSubset:  a matrix whose entries have been altered in accordance with the input
    """

    import numpy as np 

    #get the number of unique labels
    uniqueLabels=np.unique(np.asarray(list(dipyGrouping.keys())))
    #get the indexes of the valid streamlines from the input boolVec
    #concatenate to force to 1d array
    validStreams=np.concatenate(np.where(boolVec))
    #inatialize blank matrix object
    matrixSubset=np.zeros((len(uniqueLabels),len(uniqueLabels)))
    
    #iterate over the dictionary keys/ pairings
    for iPairings in range(len(list(dipyGrouping.keys()))):
        #get the current dictionary key entry
        currKey=list(dipyGrouping.keys())[iPairings]
        #get the length of the intersction of the boolVec streamlines and the current streamlines
        #and place it in the matrix location
        matrixSubset[currKey[0],currKey[1]]=len(np.intersect1d(dipyGrouping[currKey],validStreams))
    
    return matrixSubset.astype(int)

def subsetStreamsNodesByROIboundingBox_test(streamlines, maskNifti):
    """subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, which will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    """
    #compute distance tolerance
    from dipy.core.geometry import dist_to_corner
    from dipy.tracking.streamline import Streamlines
    import numpy as np
    import wmaPyTools.roiTools  
    
    import time
    
    #begin timing
    t1_start=time.process_time()
    
    #use distance to corner to set tolerance
    dtc = dist_to_corner(maskNifti.affine)
    
    #convert them to subject space
    subjectSpaceBounds=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(maskNifti)
    #expand to accomidate tolerance
    subjectSpaceBounds[0,:]=subjectSpaceBounds[0,:]-dtc
    subjectSpaceBounds[1,:]=subjectSpaceBounds[1,:]+dtc
    
    nodeArray=[]
    for iStreamlines in streamlines:
        nodeArray.append(iStreamlines[np.all(np.asarray([np.logical_and(iStreamlines[:,iDems]>subjectSpaceBounds[0,iDems],iStreamlines[:,iDems]<subjectSpaceBounds[1,iDems]) for iDems in list(range(subjectSpaceBounds.shape[1])) ]),axis=0)])
 
    
    #map and lambda function to extract the nodes within the bounds
    outIndexes=np.where(list(map(lambda x: x.size>0, nodeArray)))[0]
    outStreams=Streamlines(nodeArray)
    
        
    #map and lambda function to extract the nodes within the bounds
    #criteriaVec=list(map(lambda streamline: streamlineNodesWithinBounds(streamline,subjectSpaceBounds), streamlines))
    #outIndexes=np.where(list(map(lambda x: x.size>0, criteriaVec)))[0]
    #outStreams=Streamlines(criteriaVec)
    
    
    #stop timing
    t1_stop=time.process_time()
    # get the elapsed time
    modifiedTime=t1_stop-t1_start
    
    print('Tractogram subseting complete in ' +str(modifiedTime) +', '+str(len(outIndexes)) + ' of ' + str(len(streamlines)) + ' within mask boundaries')
    return outIndexes, outStreams   
    

def applyNiftiCriteriaToTract_DIPY_Cython(streamlines, maskNifti, includeBool, operationSpec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria, uses a range of dipy functions
    #and custom made functions to expedite the typically slow segmentation process
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -maskNifti: a nifti Mask containing only 1s and 0s
    #
    # -includeBool: a boolean indicator of whether you want the associated ROI to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # -operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    """
    #still learning how to import from modules
    from dipy.tracking.utils import near_roi
    import numpy as np
    import dipy.tracking.utils as ut
    import nibabel as nib
    from nilearn import masking 
    import scipy
    from dipy.tracking.vox2track import _streamlines_in_mask
    
    #perform some input checks
    validOperations=["any","all","either_end","both_end"]
    if np.logical_not(np.in1d(operationSpec, validOperations)):
         raise Exception("applyNiftiCriteriaToTract Error: input operationSpec not understood.")
    
    if np.logical_not(type(maskNifti).__name__=='Nifti1Image'):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not a nifti.")
    
    #the conversion to int may cause problems if the input isn't convertable to int.  Then again, the point of this is to raise an error, so...
    elif np.logical_not(np.all(np.unique(maskNifti.get_fdata()).astype(int)==[0, 1])): 
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not convertable to 0,1 int mask.  Likely not a mask.")
        
    if np.logical_not(isinstance(includeBool, bool )):
        raise Exception("applyNiftiCriteriaToTract Error: input includeBool not a bool.  See input description for usage")
        
    lin_T, offset =ut._mapping_to_voxel(maskNifti.affine)
    criteriaStreamsBool=_streamlines_in_mask( list(streamlines), maskNifti.get_fdata().astype(np.uint8), lin_T, offset)
    
    if includeBool==True:
        #initalize an out bool vec
        outBoolVec=np.zeros(len(streamlines), dtype=bool)
        #set the relevant entries to true
        outBoolVec[criteriaStreamsBool.astype(bool)]=True
    elif includeBool==False:          
        #initalize an out bool vec
        outBoolVec=np.ones(len(streamlines), dtype=bool)
        #set the relevant entries to true
        outBoolVec[criteriaStreamsBool.astype(bool)]=False
    
    return outBoolVec

def segmentTractUsingTractMask(streamlines,singleTractMask):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.
    singleTractMask : TYPE
        DESCRIPTION.

    Returns
    -------
    outBoolVec : TYPE
        DESCRIPTION.

    """
    import wmaPyTools.roiTools
    
    tractProbabilityMap2SegCriteria(singleTractMask)
    
    outBoolVec='incomplete'
    
    return outBoolVec

def tractProbabilityMap2SegCriteria(singleTractProbMap):
    """
    This function converts a nifti-based probablity map (presumably of an atlas
    maping of a tract) to a series of segmentation critersegmentedTract = StatefulTractogram(testStreamlines.streamlines[comboROIBool], testAnatomy, Space.RASMM)
    save_tractogram( segmentedTract,'testSegmentedTract.trk')ia
    
    You know what would make this process a lot easier?
    If atlases included endpoint masks, and not just allNode masks

    Parameters
    ----------
    singleTractProbMap : TYPE
        DESCRIPTION.

    Returns
    -------
    criteriaDict : dictionary
        DESCRIPTION.

    """
    import nibabel as nib
    import numpy as np
    import wmaPyTools.roiTools  
    
    if isinstance(singleTractProbMap,str):
        singleTractProbMap=nib.load(singleTractProbMap)
        
    #if it turns out we need to inflate the input mask, do it here using scipy dilate
    
    #create a bounding box of planarROIS.  These will be exclusion criteria as
    #nothing from the tract can traverse them.  Theoretically we could require
    #that all streamlines contain all nodes within the mask, but this is (1)
    #probably overly constrictive and (2) computationally intensive    
    boundaryPlanes=wmaPyTools.roiTools.boundaryROIPlanesFromMask(singleTractProbMap)
  
    
    #go ahead and get the boundaries in subject space as well
    #we'll be using these to find the primary axis along which we'll be making 
    #planar ROIs.  Alternatively, you could make ~3 spherical ROIS that span the
    #entrity of the mask, one at each end of the thresheld density, and one in the middle
    # but this presumes a bit about the morphology of the tract
    # the arcuate is going to be a real test case for this method
    maskBoundaries=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(singleTractProbMap)
    #compute the span of values for each of these
    dimSpans=np.abs(maskBoundaries[:,0]-maskBoundaries[:,1])
    primaryDim=np.where(dimSpans==np.max(dimSpans))[0][0]
    #kind of an inference here, but ok
    dimDict={0:'x',1:'y',2:'z'}
    letterDim=dimDict[primaryDim]
    
    #here we're arbitraily going to set a span frequency which will serve as 
    #the spacing between ROIs
    planarROISpacing=7
    #establish the coordinates of the planar ROIs
    planeCoords=np.arrange(dimSpans[primaryDim,0],dimSpans[primaryDim,1],planarROISpacing)
    #maybe throw an error here if you end up with less than three values.  This method
    #is probably ill advised for tracts with length of 21 or less (i.e. ufibers)
    if len(planeCoords)<3:
        raise ValueError('dimensional span of ' + str(dimSpans[primaryDim]) + ' for dimension ' + letterDim + ' too small to create reasonable planar ROIs')
    #This is going to cause problems for curved tracts like the arcuate
    #you could imagine the plane hitting the extreme rear of the arc, and imposing
    #an inclusion criterion that ends up shaving everything that curved ahead 
    #of that point.  That's a problem
    #So here's what's going to happen in that case:  There will be a plane
    #which is outside the bounds formed by BOTH endpoint clusters
    
    
    #ACTUALLY, MAYBE IT DOESN'T MATTER IF WE ARE PASSED A DENSITY/PROBABILITY MASK
    #if we're doing this cleverly, we can assume that the input is density/probabilty
    #based, so here we can appply some sort of thesholding to extract the
    # "core" of the tract.
    
    #here we'll arbitraily set a threshold value
    #thresholdProportion=.7
    #next we find where this is in the unique values of the datablock
    #and set it as our threshold value
    #threshVal=np.unique(singleTractProbMap.get_fdata())[np.floor(len(np.unique(singleTractProbMap.get_fdata()))*thresholdProportion).astype(int)]
    
    #lets get to making the planar coordinates
    planarInclusionROIs=[]
    for iROIcords in planeCoords[1:-1]:
        planarInclusionROIs.append(wmaPyTools.roiTools.makePlanarROI(singleTractProbMap, iROIcords, letterDim))
        
    criteriaDict={}
    criteriaDict['any']['include']=planarInclusionROIs
    #there's abit of weirdness here, in that the exclusion rois have names.  This will
    #undoubtedly cause standardization errors later
    criteriaDict['any']['exclude']=boundaryPlanes

    return criteriaDict

def densityMaskToSegCriteria(densityMask):
    
    import numpy as np
    import nibabel as nib
    import scipy
    import copy
    
    #MAJOR INSIGHT:  it is never the case that endpoints can be in the "core" of a
    #tract.  Ergo, you can erode the mask a bit and use it as an endpoint exclusion criterion, 
    # but also as a traversal criterion
    coreMask=scipy.ndimage.binary_erosion(densityMask.get_fdata(), iterations=2)
    inflatedMask=scipy.ndimage.binary_dilation(densityMask.get_fdata(), iterations=1)
    endpointsMask=copy.deepcopy(inflatedMask)
    endpointsMask[coreMask]=False
    
    endpointCriteria=nib.Nifti1Image(endpointsMask, densityMask.affine, densityMask.header)
    coreIntersectCriteria=nib.Nifti1Image(coreMask, densityMask.affine, densityMask.header)
    
    criteriaDict={}
    
    
    
    return criteriaDict

def voxelwiseAtlasConnectivity(streamlines,atlasNifti,mask=None):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.
    atlasNifti : TYPE
        DESCRIPTION.
    mask : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    voxelAtlasConnectivityArray : TYPE
        DESCRIPTION.

    """
    import numpy as np
    from dipy.tracking import utils
    import wmaPyTools.streamlineTools
    from dipy.tracking.vox2track import streamline_mapping
    
    #napkin math to predict size of output array:
    #16 bit * 200 labels * 2000000 streamlines =~ 800 megabytes, not too bad

    
    #NOTE astype(int) is causing all sorts of problems, BE WARNED
    #using round as a workaround
    [relabeledAtlasData, labelMappings]=utils.reduce_labels(np.round(atlasNifti.get_fdata()).astype(int))
    
    #I guess we can go ahead and create this output structure
    voxelAtlasConnectivityArray=np.zeros([len(streamlines),len(np.unique(relabeledAtlasData))]).astype(np.uintc)
    
    #if an input mask is actually provided
    if not mask==None:
        streamsInMaskBool=applyNiftiCriteriaToTract_DIPY_Test(streamlines, mask, True, 'either_end')
        #we don't care about the label identity of *endpoints* in these img-space
        #voxels
        WithinMaskVoxels=np.nonzero(mask)
        #may need to manipulate this to make it a list of lists
    else:
        #I guess we're doing it for all of them!
        streamsInMaskBool=np.ones(len(streamlines))
        #there are no voxels that we aren't considering in this case
        WithinMaskVoxels=[]
    
    #perform dipy connectivity analysis.  Don't need to downsample to endpoints
    #yet because dipy only computes on endpoints anyways
    M, grouping=utils.connectivity_matrix(streamlines[streamsInMaskBool], atlasNifti.affine, label_volume=relabeledAtlasData,
                            return_mapping=True,
                            mapping_as_streamlines=False)
    
    #get just the endpoints
    streamEndpoints=wmaPyTools.streamlineTools.downsampleToEndpoints(streamlines[streamsInMaskBool])
    #perform the mapping of these endpoints
    #maybe we need the flipped version of this
    streamlineEndpointMapping=streamline_mapping(streamEndpoints, atlasNifti.affine)
    #extract the dictionary keys as coordinates
    imgSpaceEndpointVoxels = list(streamlineEndpointMapping.keys())
    
    for iGroupings in list(grouping):
        currentStreams=grouping[iGroupings]
        if not 
        voxelAtlasConnectivityArray
        
        
        
        
        
        

    
    
    return voxelAtlasConnectivityArray