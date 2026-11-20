# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:01 2021

@author: Daniel
"""
#Claude note: module level imports for the private speedup helpers below
#(the public functions in this module keep their own local imports)
import numpy as np
from scipy.ndimage import find_objects as _find_objects


# Claude note: --------------------------------------------------------------
# Plane / voxel mapping helpers (added 2026-11).  makePlanarROI,
# sliceROIwithPlane and alignROItoReference used to map subject-space points
# to voxels with dipy's _to_voxel_coordinates and then subtract 1 from every
# index "because of zero indexing".  dipy's indices are already 0-based (the
# -1 is a leftover from the MATLAB originals, which index from 1), so every
# plane landed one voxel too low along each axis, and a point on the lowest
# voxel wrapped round to the opposite edge of the image (index -1).  Separately,
# dipy's bounding_box (used by subjectSpaceMaskBoundaryCoords) reports its
# upper bound one voxel past the mask, so planeAtMaskBorder put its low-side
# planes one voxel outside the mask and only got its high-side planes onto
# the mask because the two errors cancelled.  These helpers do the mapping
# directly, and subjectSpaceMaskBoundaryCoords now returns the centres of the
# mask's outermost voxels on both sides (see _imageFaceBounds for the outer
# faces of a whole image).
# ---------------------------------------------------------------------------

def _pointsToVoxelMask(coords, affine, shape):
    """
    Claude note: mark the voxels containing subject-space points.

    A point is assigned to the voxel whose centre is nearest along each axis
    (the same rounding dipy's _to_voxel_coordinates uses), with no index
    offset.  Points falling outside the image are dropped instead of wrapping
    round to the far edge.

    Parameters
    ----------
    coords : numpy.ndarray
        N x 3 subject-space (mm) coordinates.
    affine : numpy.ndarray
        4 x 4 voxel-to-subject affine of the target image.
    shape : tuple of int
        Shape of the target image (first three dimensions are used).

    Returns
    -------
    outMask : numpy.ndarray of bool
        Mask of the target shape, True in voxels containing a point.
    """
    import nibabel as nib
    #Claude note: move the points into (continuous) voxel space
    voxCoords = nib.affines.apply_affine(np.linalg.inv(affine), np.asarray(coords, dtype=float))
    #Claude note: round to the nearest voxel centre, as dipy does (floor(x+0.5))
    inds = np.floor(voxCoords + 0.5).astype(np.int64)
    #Claude note: drop any point that falls outside the image
    shape3 = np.asarray(shape[:3])
    inImage = np.all((inds >= 0) & (inds < shape3), axis=1)
    inds = inds[inImage]
    #Claude note: set those voxels
    outMask = np.zeros(tuple(shape3), dtype=bool)
    outMask[inds[:, 0], inds[:, 1], inds[:, 2]] = True
    return outMask


def _voxelCentreBounds(affine, indexMins, indexMaxs):
    """
    Claude note: subject-space min/max of the voxel centres in an (inclusive)
    index box.

    Parameters
    ----------
    affine : numpy.ndarray
        4 x 4 voxel-to-subject affine.
    indexMins, indexMaxs : sequence of int
        Inclusive lowest and highest voxel index along each of the 3 axes.

    Returns
    -------
    bounds : numpy.ndarray
        2 x 3 array: row 0 the minimum, row 1 the maximum subject-space
        coordinate in each dimension, taken over the box's corner voxels.
    """
    import itertools
    import nibabel as nib
    #Claude note: all 8 corner voxels of the box
    corners = np.asarray(list(itertools.product(*[[indexMins[i], indexMaxs[i]] for i in range(3)])), dtype=float)
    #Claude note: their subject-space positions
    cornerCoords = nib.affines.apply_affine(affine, corners)
    return np.vstack([cornerCoords.min(axis=0), cornerCoords.max(axis=0)])


def _imageFaceBounds(affine, shape):
    """
    Claude note: subject-space bounds of an image's outer voxel faces.

    The image occupies voxel coordinates -0.5 .. n-0.5 along each axis (voxel
    centres sit on whole numbers), so its extent is the box spanned by those
    corners, half a voxel beyond the first and last voxel centres.

    Parameters
    ----------
    affine : numpy.ndarray
        4 x 4 voxel-to-subject affine.
    shape : tuple of int
        Image shape (first three dimensions are used).

    Returns
    -------
    bounds : numpy.ndarray
        2 x 3 array: row 0 the minimum, row 1 the maximum subject-space
        coordinate in each dimension.
    """
    return _voxelCentreBounds(affine, [-.5, -.5, -.5], [n - .5 for n in shape[:3]])


# Claude note: --------------------------------------------------------------
# Private speedup helpers (added 2026-11).  The atlas processing functions
# previously ran morphological operations (label, erosion, propagation) over
# the whole (e.g. 256^3) volume once per atlas label, although each label
# occupies a small corner of it.  Cropping each such operation to the label's
# bounding box gives identical results (voxels outside the box are not part
# of the label, which is exactly how scipy.ndimage treats voxels outside the
# array), at a small fraction of the cost.  See
# tests/test_speedupEquivalence.py.
# ---------------------------------------------------------------------------

def _labelBoundingBoxes(data, labelValues):
    """
    Claude note: for each requested label value, a tuple of slices (one per
    dimension) that contains every voxel where data==label.  The box is
    tight when this is cheap to compute, but callers must only rely on it
    containing all of the label's voxels.

    Parameters
    ----------
    data : np.ndarray
        label volume
    labelValues : iterable of numbers

    Returns
    -------
    boxes : dict
        label value -> tuple of slices.  A label that is absent gets an empty
        box (zero size slices).
    """
    fullBox = tuple(slice(0, iDim) for iDim in data.shape)
    emptyBox = tuple(slice(0, 0) for iDim in data.shape)
    boxes = {}
    labelValues = list(labelValues)
    if np.issubdtype(data.dtype, np.integer) and data.size > 0 and data.min() >= 0 and data.max() < 10**7:
        #Claude note: one pass over the volume for all labels
        objectSlices = _find_objects(data)
        for iLabel in labelValues:
            if iLabel == 0 or iLabel != int(iLabel):
                #Claude note: find_objects ignores 0 (and non-integer values
                #can't be equal to integer data anyway)
                boxes[iLabel] = fullBox
            elif 0 < int(iLabel) <= len(objectSlices) and objectSlices[int(iLabel) - 1] is not None:
                boxes[iLabel] = objectSlices[int(iLabel) - 1]
            else:
                boxes[iLabel] = emptyBox
    else:
        for iLabel in labelValues:
            boxes[iLabel] = fullBox
    return boxes


def _resolveBackgroundLabel(labels, counts, backgroundLabel, functionName):
    """
    Claude note: decides which label is the atlas background.

    If backgroundLabel is given, it is used.  Otherwise the original
    heuristic is kept (the most common label), but because that heuristic
    cannot be reliable in principle (on a cropped atlas the most common label
    may well be white matter, and nothing in the image itself says otherwise)
    a warning is issued whenever its answer departs from the near universal
    convention that background is 0.

    Parameters
    ----------
    labels, counts : np.ndarray
        as returned by np.unique(atlasData, return_counts=True)
    backgroundLabel : number or None
        the caller's explicit background label, if any
    functionName : str
        for the warning message

    Returns
    -------
    detectBackground : int
    """
    import warnings
    if backgroundLabel is not None:
        if not np.any(labels==backgroundLabel):
            warnings.warn(functionName + ": requested backgroundLabel " + str(backgroundLabel) + " is not present in the atlas.")
        return int(backgroundLabel)
    detectBackground=int(labels[np.where(np.max(counts)==counts)[0][0]])
    if detectBackground!=0:
        zeroNote=(" Label 0 is present (" + str(int(counts[labels==0][0])) + " voxels) but is not the most common.") if np.any(labels==0) else ""
        warnings.warn(functionName + ": inferred background label " + str(detectBackground) + " (the most common label), not 0." + zeroNote + " If the atlas is cropped, or its background is not its most common label, this is likely wrong; pass backgroundLabel explicitly.")
    return detectBackground


def _inflationVoteForVoxel(noIslandAtlasData, iInflationTargetCoords, inflationLabelTargets):
    """
    Claude note: the per-voxel vote of inflateAtlasIntoWMandBG, moved here
    verbatim (minus the commented-out lines) so it can be reused for the
    (rare) voxels whose immediate neighborhood gives a tied vote.  Grows the
    window until a single label wins.
    """
    #this is the voxel-wise radius of the expans you're considering
    window=1
    #if you set the vector to 2 or longer by default, it will run until fixed
    voteWinner=[0,0]
    while len(voteWinner)>1:
        #we need to consider those cases in which the voxels are adjacent to the edge of the volume
        xRange=np.asarray(range((iInflationTargetCoords[0]-window),(iInflationTargetCoords[0]+1+window)))
        yRange=np.asarray(range((iInflationTargetCoords[1]-window),(iInflationTargetCoords[1]+1+window)))
        zRange=np.asarray(range((iInflationTargetCoords[2]-window),(iInflationTargetCoords[2]+1+window)))
        #remove invalid indexes
        #Claude note: >=1 (was >=0), so that the padding is excluded on both sides
        xRange=xRange[np.all([xRange<noIslandAtlasData.shape[0]-1,  xRange>=1],axis=0)]
        yRange=yRange[np.all([yRange<noIslandAtlasData.shape[1]-1,  yRange>=1],axis=0)]
        zRange=zRange[np.all([zRange<noIslandAtlasData.shape[2]-1,  zRange>=1],axis=0)]
        x, y, z = np.meshgrid(xRange, yRange, zRange,indexing='ij')
        #squeeze the output (maybe not necessary)          
        windowBlock= np.squeeze([x, y, z])
        #convert to array of coordinates
        windowIndexes=np.vstack(windowBlock).reshape(3,-1).T
        voxelVotes=np.asarray([noIslandAtlasData[iVoxel[0],iVoxel[1],iVoxel[2]] for iVoxel in list(windowIndexes)])
        #get the vote counts
        (labels,counts)=np.unique(voxelVotes[np.isin(voxelVotes,inflationLabelTargets,invert=True)], return_counts=True)
        #Claude note: no votes at all: widen
        #the window, and if it already spans the volume, leave the voxel as is
        if len(counts)==0:
            if window>max(noIslandAtlasData.shape):
                return int(noIslandAtlasData[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]])
            window=window+1
            voteWinner=[0,0]
            continue
        voteWinner=labels[np.where(np.max(counts)==counts)[0]]
        window=window+1
    return int(voteWinner[0])


def _inflationVotes(noIslandAtlasData, inflationTargetCoords, inflationLabelTargets, chunkSize=50000):
    """
    Claude note: vectorised equivalent of running _inflationVoteForVoxel for
    every target voxel.  The 3x3x3 (window=1) vote is computed for all target
    voxels at once; only voxels where that vote is tied (or empty) fall back
    to the original per-voxel loop, which then widens the window exactly as
    before.  The data are not modified while voting (the original also
    collected all votes before applying them), so the order in which voxels
    are processed does not matter.

    Returns
    -------
    winners : np.ndarray of int, one per row of inflationTargetCoords
    """
    winners=np.zeros(len(inflationTargetCoords),dtype=np.int64)
    if len(inflationTargetCoords)==0:
        return winners
    dataShape=np.asarray(noIslandAtlasData.shape)
    #Claude note: the 27 offsets of the window=1 neighborhood, in the same (x,y,z) order
    offsets=np.stack(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1],indexing='ij'),axis=-1).reshape(-1,3)
    for iChunk in range(0,len(inflationTargetCoords),chunkSize):
        chunkCoords=np.asarray(inflationTargetCoords[iChunk:iChunk+chunkSize])
        neighborCoords=chunkCoords[:,None,:]+offsets[None,:,:]
        #Claude note: same validity rule as _inflationVoteForVoxel: >=1 and < shape-1
        valid=np.all(np.logical_and(neighborCoords>=1,neighborCoords<dataShape-1),axis=2)
        clipped=np.clip(neighborCoords,0,dataShape-1)
        votes=noIslandAtlasData[clipped[:,:,0],clipped[:,:,1],clipped[:,:,2]]
        valid=np.logical_and(valid,np.isin(votes,inflationLabelTargets,invert=True))
        #Claude note: for each vote, how many valid votes share its label
        sameLabel=np.logical_and(votes[:,:,None]==votes[:,None,:],valid[:,None,:])
        voteCounts=np.where(valid,np.sum(sameLabel,axis=2),0)
        maxCounts=np.max(voteCounts,axis=1)
        firstMax=np.argmax(voteCounts,axis=1)
        winnerLabels=votes[np.arange(len(votes)),firstMax]
        #Claude note: tied if a different label reaches the same maximum count
        tied=np.any(np.logical_and(voteCounts==maxCounts[:,None],votes!=winnerLabels[:,None]),axis=1)
        needsLoop=np.logical_or(tied,maxCounts==0)
        chunkWinners=winnerLabels.astype(np.int64)
        for iLoop in np.where(needsLoop)[0]:
            chunkWinners[iLoop]=_inflationVoteForVoxel(noIslandAtlasData,chunkCoords[iLoop],inflationLabelTargets)
        winners[iChunk:iChunk+chunkSize]=chunkWinners
    return winners


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
    
    #Claude note: the image's extent, as the subject-space bounds of its first
    #and last voxel centres (previously the upper bound was one voxel past the
    #image, see subjectSpaceMaskBoundaryCoords)
    refShape=reference.shape[:3]
    convertedBoundCoords=_voxelCentreBounds(reference.affine, [0,0,0], [n-1 for n in refShape])
    #Claude note: half a voxel either side of those centres is still inside the image
    halfVoxel=np.asarray(reference.header.get_zooms()[:3])*.5

    #create a dict to interpret input
    dimDict={'x':0,'y':1,'z':2}
    selectedDim=dimDict[dimension]

    #if the requested planar coordinate is outside of the image, throw a full on error
    if convertedBoundCoords[0,selectedDim]-halfVoxel[selectedDim]>mmPlane or convertedBoundCoords[1,selectedDim]+halfVoxel[selectedDim]<mmPlane:
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
            stepSize=halfVoxel[iDims]
            #create a vector with the coords for this dimension
            #Claude note: running up to and including the last voxel centre
            dimCoords=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims]+stepSize*.5,stepSize)
            #append it to the planeCoords list object
            planeCoords.append(list(dimCoords))
    x, y, z = np.meshgrid(planeCoords[0], planeCoords[1], planeCoords[2],indexing='ij')
    #squeeze the output (maybe not necessary)
    planeCloud= np.squeeze([x, y, z])
    #convert to coordinate vector
    testSplit=np.vstack(planeCloud).reshape(3,-1).T
    #Claude note: mark the voxels holding those points (see _pointsToVoxelMask)
    outData=_pointsToVoxelMask(testSplit, reference.affine, refShape)
    #Claude note: a coordinate exactly half a voxel past the last voxel centre
    #passes the check above but rounds to the voxel beyond the image; treat
    #it as outside rather than silently returning an empty plane
    if not np.any(outData):
        raise ValueError('Requested planar coordinate outside of reference image \n coordinate ' + str(mmPlane) + ' requested for bounds ' + str(convertedBoundCoords[:,selectedDim]))
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
#     atlasData = np.round(np.asanyarray(atlas.dataobj)).astype(int)
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
    if not np.isin(relativePosition.lower(),validPositionTerms):
         raise Exception("planeAtROIborder Error: input relative position " + relativePosition + " not valid.")
    
    #convert the boundary coords of the mask to subject space in order to interpret positional terms
    #Claude note: the outermost mask voxel centres, so every border plane lies
    #on the mask's edge voxels (before 2026-11 the upper bound was one voxel
    #past the mask, which with makePlanarROI's old -1 offset put the
    #inferior / posterior / left planes one voxel outside the mask)
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
        (Claude note: these are the image's outer voxel faces.)

    """
  
    import itertools
    import numpy as np
    import nibabel as nib
  
    #Claude note: the outer voxel faces, as min / max rows, then flipped
    #to this function's historical max-first order
    faceBounds=_imageFaceBounds(inputNifti.affine, inputNifti.shape)
    boundingBox=faceBounds[::-1].copy()
    
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
    
    #Claude note: the image's extent, as its outer voxel faces
    convertedBoundCoords=_imageFaceBounds(reference.affine, reference.shape)
    
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
    
    #Claude note: round, not floor: nibabel voxel *centers* sit at integer indexes, so the
    #nearest voxel to p is the rounded index (floor biased this by up to 1 voxel)
    imgCoord=np.round(nib.affines.apply_affine(np.linalg.inv(reference.affine),p))
    
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
    #Claude note: a previous version computed x*x*voxelDims[0], i.e. scaled the
    #*squared* voxel distance by the voxel size only once.  That is only
    #correct for 1mm voxels; for 2mm voxels the sphere radius came out ~41%
    #too large (sqrt(2)), and for 1.25mm (HCP) ~12% too large.  The
    #distance in mm is (voxelCount*voxelSize), and *that* gets squared.
    voxelDims=reference.header.get_zooms()
    mask_r = (x*voxelDims[0])**2 + (y*voxelDims[1])**2 + (z*voxelDims[2])**2 <= r*r

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
    
    #Claude note: the atlas data are read and rounded once, and the
    #presence check only considers the requested labels' voxels: a requested
    #label is in np.unique(roundedAtlas) iff it is among the labels of the
    #voxels that matched a request.
    #ATLASES ARE ACTING WEIRD THESE DAYS, gotta do round then int, not other way
    #if you don't do it this way, somehow you get a reduced number of labels
    roundedAtlasData=np.round(np.asanyarray(atlas.dataobj)).astype(int)
    labelMask=np.isin(roundedAtlasData,roiNumsInArray)
    presentRequestedLabels=np.unique(roundedAtlasData[labelMask])
    
    if  np.logical_not(np.all(np.isin(roiNumsInArray,presentRequestedLabels))):
        import warnings
        warnings.warn("WMA.multiROIrequestToMask WARNING: ROI label " + str(list(roiNumsInArray[np.logical_not(np.isin(roiNumsInArray,presentRequestedLabels))])) + " not found in input Nifti structure.")
        
    #obtain coordiantes of all relevant label 
    #create blank data structure
    concatData=np.zeros(atlas.shape)
    #set all appropriate values to true
    #Claude note: boolean mask indexing, same voxels as np.where(labelMask)
    concatData[labelMask]=True
    
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
    inputROINiftiData=np.asanyarray(inputROINifti.dataobj)
    inputPlanarROIData=np.asanyarray(inputPlanarROI.dataobj)
    
    #boolean to check if intersection
    intersectBool=np.any(np.logical_and(inputROINiftiData!=0,inputPlanarROIData!=0))
    if not intersectBool:
        import warnings
        warnings.warn("sliceROIwithPlane WARNING: input planar ROI does not intersect with input ROI.")

    #implement test to determine if input planar roi is indeed planar
    #get coordinates of mask voxels in image space
    planeVoxCoords=np.where(np.asanyarray(inputPlanarROI.dataobj))
    #find the unique values of img space coordinates for each dimension
    uniqueCoordCounts=[len(np.unique(iCoords)) for iCoords in planeVoxCoords]
    #one of them should be singular in the case of a planar roi, throw an error if not
    if not np.any(np.isin(uniqueCoordCounts,1)):
        raise ValueError('input cut ROI not planar (i.e. single voxel thick for True values)')
    
    #Claude note: the keep region is decided directly from each
    #voxel's subject-space position.  The plane's own voxels
    #are kept on whichever side is requested, and nothing wraps.
    
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
    if relativePosition not in dimensionDict:
        raise ValueError('sliceROIwithPlane: relativePosition ' + str(relativePosition) + ' not understood')
    cutDim=dimensionDict[relativePosition]
    
    #Claude note: subject-space position of the plane along the cut dimension,
    #from the centres of its voxels
    planeSubjCoords=nib.affines.apply_affine(inputPlanarROI.affine, np.asarray(planeVoxCoords[:3]).T)
    planeCoord=np.mean(planeSubjCoords[:,cutDim])
    #Claude note: tolerance for "on the plane", far below a voxel
    zooms=np.asarray(inputROINifti.header.get_zooms()[:3])
    onPlaneTol=1e-3*np.min(zooms)
    if np.ptp(planeSubjCoords[:,cutDim])>onPlaneTol:
        import warnings
        warnings.warn('sliceROIwithPlane WARNING: planar ROI is not perpendicular to the ' + relativePosition + ' direction; cutting at its mean position.')
    
    #Claude note: the kept interval [lowBound, highBound] along cutDim
    if relativePosition in ['superior','anterior','rostral','right']:
        lowBound, highBound = planeCoord, np.inf
    elif relativePosition in ['inferior','posterior','caudal','left']:
        lowBound, highBound = -np.inf, planeCoord
    elif relativePosition=='medial':
        #between the plane and the midline (x=0)
        lowBound, highBound = np.sort([planeCoord, 0.0])
    else:
        #lateral: between the plane and whichever x edge of the image is nearer
        #to it (as before)
        #Claude note: from the image's first and last voxel centres
        imageBounds=_voxelCentreBounds(inputROINifti.affine, [0,0,0], [n-1 for n in inputROINifti.shape[:3]])
        xEdges=imageBounds[:,0]
        nearestEdge=xEdges[np.argmin(np.abs(xEdges-planeCoord))]
        if nearestEdge>=planeCoord:
            lowBound, highBound = planeCoord, np.inf
        else:
            lowBound, highBound = -np.inf, planeCoord
    
    #Claude note: subject-space coordinate of every voxel centre along cutDim,
    #built by broadcasting one affine row over the three index axes
    roiShape=inputROINifti.shape[:3]
    affineRow=inputROINifti.affine[cutDim]
    voxelCutCoords=(affineRow[0]*np.arange(roiShape[0])[:,None,None]
                    +affineRow[1]*np.arange(roiShape[1])[None,:,None]
                    +affineRow[2]*np.arange(roiShape[2])[None,None,:]
                    +affineRow[3])
    #Claude note: keep voxels inside the interval, counting the plane itself
    keepArea=(voxelCutCoords>=lowBound-onPlaneTol) & (voxelCutCoords<=highBound+onPlaneTol)
    
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
        blankNiftiData=np.add(blankNiftiData,np.asanyarray(currentSphere.dataobj))
        
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
    roiCoords=seeds_from_mask(np.asanyarray(inputROI.dataobj), inputROI.affine, density=densityKernel)
    
    #Claude note: mark the reference voxels holding those points.  This used
    #to map them with dipy and subtract 1 from each index "because of zero
    #indexing", which moved the ROI one voxel low along every axis (and
    #wrapped voxels on the lowest index round to the far edge); points outside
    #the reference image are now dropped (dipy raised on negative indices)
    outData=_pointsToVoxelMask(roiCoords, reference.affine, reference.shape[:3])
    
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
    
    #Claude note: each image's extent as its outer voxel faces
    convertedBoundCoords1=_imageFaceBounds(nifti1.affine, nifti1.shape)
    convertedBoundCoords2=_imageFaceBounds(nifti2.affine, nifti2.shape)
    
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
        for each dimension (Claude note: the subject-space centres of the
        outermost mask voxels; before 2026-11 the maximum was one voxel past
        the mask)

    """
    
    #Claude note: the bounds are the subject-space centres of the
    #mask's outermost voxels on BOTH sides.  For the outer faces of a whole image
    #see _imageFaceBounds.
    maskData=np.asanyarray(maskNifti.dataobj)!=0
    #fold any dimensions past the third into the first three
    if maskData.ndim>3:
        maskData=maskData.reshape(maskData.shape[:3]+(-1,)).any(axis=3)
    indexMins=[]
    indexMaxs=[]
    for iAxis in range(3):
        #indexes along this axis holding any mask voxel
        otherAxes=tuple(iOther for iOther in range(3) if iOther!=iAxis)
        occupied=np.flatnonzero(maskData.any(axis=otherAxes))
        if len(occupied)==0:
            #empty mask: as before (dipy returned 0 bounds, with a warning)
            import warnings
            warnings.warn('subjectSpaceMaskBoundaryCoords: no data found in mask; returning the position of voxel 0.')
            indexMins, indexMaxs = [0,0,0], [0,0,0]
            break
        indexMins.append(int(occupied[0]))
        indexMaxs.append(int(occupied[-1]))
    #the corners' subject-space min and max in each dimension
    subjectSpaceBounds=_voxelCentreBounds(maskNifti.affine, indexMins, indexMaxs)
    
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
#     nifti1=nib.nifti1.Nifti1Image(np.nan_to_num(np.asanyarray(nifti1.dataobj)), nifti1.affine, nifti1.header)
#     nifti2=nib.nifti1.Nifti1Image(np.nan_to_num(np.asanyarray(nifti2.dataobj)), nifti2.affine, nifti2.header)
    
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

def removeIslandsFromAtlas_viaLabelCount(atlasNifti,ignoreLabelsRequest=None):
    """
    Removes island label values from atlas iteratively across values present.
    Important for when extracted atlas labels are being used as anatomical 
    markers.  Instead of using erosion, we'll use scipy.ndimage.label
    
    Potentially faster ways of doing this in other packages.
    
    Parameters
    ----------
    atlasNifti : string path to file loadable via nib.load or nifti-like object
        A multi label atlas/parc, with integer based labels.  Will load if string
        is passed
    ignoreLabelsRequest :  list of int, optional
        Label values that should NOT be de-islanded (e.g. labels that are
        legitimately discontinuous, like some cerebellar or ventricular
        labels).  These are still reported on in removalReport, but left
        unmodified.  0 (background) is always ignored, whether or not it is
        included.  The default is None, i.e. only ignore 0.

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
    #Claude note: scipy.ndimage.measurements is deprecated namespace; label lives
    #directly in scipy.ndimage
    from scipy.ndimage import label
    import numpy as np
    import copy
    import pandas as pd
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti
    
    #by default we ignore the background
    #Claude note: build a new list rather than appending to the input, so that
    #neither the caller's list nor a shared default is mutated across calls
    if ignoreLabelsRequest is None:
        ignoreLabelsRequest=[]
    #just in case they included 0 or repeats
    ignoreLabels=np.unique(list(ignoreLabelsRequest)+[0])

    #do something to ensure that it is int based
    atlasData=np.round( np.asanyarray(atlas.dataobj)).astype(int)
    
    #create a seprate object to begin de-islanding
    noIslandData=copy.deepcopy(atlasData)
    
    #get counts of the unique lables
    [labels,counts]=np.unique(atlasData, return_counts=True)
    
    #Claude note: rather than writing into the report table cell-by-cell inside the loop
    #(which is slow, and which pandas >=3 silently turns into a no-op due to
    #copy-on-write) we accumulate per-label values here, and build the
    #report once at the end
    islandCounts=np.zeros(len(labels),dtype=int)
    islandVoxCounts=np.zeros(len(labels),dtype=int)
    
    #Claude note: first pass: remove islands from all non-ignored labels
    #second pass (below) : report on, but don't modify, the ignored labels
    #Claude note: connected components are found within each label's bounding
    #box (see _labelBoundingBoxes) rather than across the whole volume: a
    #label's components all lie inside its box, and scipy numbers them in the
    #same scan order, so the result is identical.
    labelBoxes=_labelBoundingBoxes(atlasData, labels)
    for labelIterator,iLabels in enumerate(labels):
        labelBox=labelBoxes[iLabels]
        #label the distinct elements of the parcellation matching the current label
        #~.5 of .9 seconds per loop here
        labeled_array, num_features = label(atlasData[labelBox]==iLabels)
        #get the counts for these elements
        #Claude note: equivalent to np.unique(labeled_array, return_counts=True)
        #over the whole volume: components 1..num_features are all present,
        #and 0 (background) is present iff the label doesn't fill the volume
        componentCounts=np.bincount(labeled_array.ravel(),minlength=num_features+1)[1:]
        backgroundCount=atlasData.size-counts[labelIterator]
        if backgroundCount>0:
            curr_counts=np.concatenate([[backgroundCount],componentCounts])
        else:
            curr_counts=componentCounts
        
        #Claude note: if the label occupies the entire volume (e.g. an all-zero
        #background-less image), there is no "0" component to skip, and thus
        #nothing to compare against
        if len(curr_counts)<2:
            continue

        #always ignore 0, add 1 because you're ignoring it
        maxSizeIndex=np.where(curr_counts[1:]==np.max(curr_counts[1:]))[0][0]+1
        #create a mask of the locations to be zeroed out
        currIslandMask=np.logical_and(labeled_array>0,labeled_array!=maxSizeIndex)
        
        #compute and populate the data for the report
        #-2 because we don't include 0 or the largest island
        islandCounts[labelIterator]=len(curr_counts)-2
        islandVoxCounts[labelIterator]=np.sum(currIslandMask)
        
        #Claude note: use the mask to set the relevant voxels to 0, but only if this
        #label isn't one we've been asked to leave alone
        if not iLabels in ignoreLabels:
            noIslandData[labelBox][currIslandMask]=0
    
    #Claude note: count the new total voxel count for each label
    endCountsDict=dict(zip(*np.unique(noIslandData, return_counts=True)))
    endVoxCounts=[endCountsDict.get(iLabels,0) for iLabels in labels]
    
    #create a report of the island removal process
    removalReport=pd.DataFrame({'labelVal':labels,
                                'startVoxCount':counts,
                                'islandCount':islandCounts,
                                'totalIslandVox':islandVoxCounts,
                                'endVoxCount':endVoxCounts})
            
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
    

def removeIslandsFromAtlas(atlasNifti,backgroundLabel=None):
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
    backgroundLabel : int, optional
        (Claude note.)  The atlas's background label, which
        removed island voxels are set to.  If None (the default), it is
        inferred as the most common label, with a warning if that
        isn't 0 (the inference is unreliable for e.g. cropped atlases).

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


    # atlasData=np.round(np.asanyarray(atlas.dataobj)).astype(int)
    # #COSIDER REPLACING ALL OR MOST OF THIS WITH
    #Doesn't seem to work?
    # relabelConnectedData=scipy.ndimage.measurements.label(atlasData.astype(bool))

    #count the label instances, and make an inference as to the background value
    #it's almost certianly the most common value, eg. 0
    (labels,counts)=np.unique(np.asanyarray(atlas.dataobj), return_counts=True)
    #Claude note: explicit backgroundLabel, or the heuristic with a warning
    detectBackground=_resolveBackgroundLabel(labels,counts,backgroundLabel,'removeIslandsFromAtlas')
    
    #get the list of labels without the background
    atlasLabels=labels.astype(int)[labels.astype(int)!=detectBackground]
    
    #Claude note: work on a copy, so the caller's atlas is never
    #modified in place.
    atlasData=np.array(np.asanyarray(atlas.dataobj))
    #Claude note: a separate, unmodified copy of the label data, so a
    #file-backed nifti isn't re-read and re-decompressed for each label.
    #Only the current label and background are ever modified, so
    #reading from this copy is equivalent.
    origData=np.array(atlasData)
    #Claude note: each label's morphology is computed within its bounding box
    #(see _labelBoundingBoxes); identical results, far smaller arrays
    labelBoxes=_labelBoundingBoxes(origData, atlasLabels)
    removalReport=pd.DataFrame(columns=['label','max_erosion','islands_removed'])
    #remove the background values
    for iLabels in atlasLabels:
        #Claude note: the label mask within its bounding box, and its voxel count
        labelBox=labelBoxes[iLabels]
        labelMask=origData[labelBox]==iLabels
        labelVoxCount=np.sum(labelMask)
        #create a blank vector to add to the report
        reportRow=[]
        #add the current label
        reportRow.append(iLabels)
        erosionIterations=0
        #the default max iterations, will be reduced in cases where implementation
        #results in complete loss
        forcedMaxIterations=3
        currentErosion=labelMask
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
                currentErosion=labelMask
        #print how much was eroded
        print(str(np.abs(np.sum(currentErosion) - labelVoxCount))+ ' of ' + str(labelVoxCount) + ' voxels eroded label ' + str(iLabels) )
        #add that value to the report list for this label
        reportRow.append(np.abs(np.sum(currentErosion) - labelVoxCount))
        #perform the propigation function, wherein the eroded label is inflated back
        #islandRemoved=scipy.ndimage.binary_propagation(currentErosion, structure=np.ones([3,3,3]).astype(int), mask=np.asanyarray(atlas.dataobj)==iLabels)
        islandRemoved=scipy.ndimage.binary_propagation(currentErosion, mask=labelMask)
        #print the resulting difference between the intitial label and this eroded+inflated one
        print(str(np.abs(np.sum(islandRemoved) - labelVoxCount))+ ' of ' + str(labelVoxCount) + ' voxels removed for label ' + str(iLabels) )
        #add that value to the report list for this label
        reportRow.append(np.abs(np.sum(islandRemoved) - labelVoxCount))
        #add the row to the output report
        removalReport.loc[len(removalReport)]=reportRow

        #now actually modify the data
        #what you want are the areas that are true in the origional atlas but false in the eroded+inflated version
        #set those to background
        atlasData[labelBox][np.logical_and(labelMask,np.logical_not(islandRemoved))]=detectBackground
    
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
    except ImportError:
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

                newIDentityHolder[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner[0])
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

                newIDentityHolder[iInflationTargetCoords[0],iInflationTargetCoords[1],iInflationTargetCoords[2]]=int(voteWinner[0])
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
    

def inflateAtlasIntoWMandBG(atlasNifti,iterations,inferWM=False,reportSavePath='de-island_report.csv',backgroundLabel=None):
    """
    Inflates label values of input atlas into (INFERRED, SEE IN CODE NOTES)
    white matter labels, and background 0 labels.  Will perform island removal
    first.
    
    NOTE:  Ordering effects likely in inflation outcomes (i.e. lower labels
    have higher priority)
    
    Potentially faster methods in other packages

    Parameters
    ----------
    atlasNifti : string path to file loadable via nib.load or nifti-like object
        An integer based, multi label nifti atlas.  Will load if string is
        passed.
    iterations : int
        The number of iterations to perform the inflation.  Proceeds voxelwise.
    inferWM : bool, optional
        (Claude note: description derived from the in-code notes below.)
        Whether to infer which labels are white matter, from the proportion
        of total labeled volume each label occupies, and also inflate into
        those.  The default is False, i.e. inflate only into background.
    reportSavePath : string path, optional
        (Claude note.)  Where to save the de-islanding report
        (see removeIslandsFromAtlas) as a csv.  This report is a provenance /
        QC product: per-label island counts and voxel counts, useful for
        checking whether inflation went awry, and for diagnosing problems in
        the source (e.g. FreeSurfer) segmentation.  The default is
        'de-island_report.csv' in the current working directory.  Pass a path to save it elsewhere (e.g. an app's output
        directory), or None to not save it.
        The default is 'de-island_report.csv'.
    backgroundLabel : int, optional
        (Claude note.)  The atlas's background label, i.e. the
        label inflated into.  If None (the default), it is inferred as the
        most common label, with a warning if that isn't 0 (the
        inference is unreliable for e.g. cropped atlases, where the most
        common label may be white matter).

    Returns
    -------
    atlasNiftiOut : nibabel.nifti1.Nifti1Image
        The input atlas nifti, but with islands removed and then inflated into
        background (and, if inferWM, presumed white matter) labels.

    """
    import nibabel as nib
    import numpy as np
    #from scipy.interpolate import NearestNDInterpolator
    import scipy
    #conditional import of tqdm
    try: 
        import tqdm
        tqdmFlag=True
    except ImportError:
        tqdmFlag=False
        
    from nilearn.image import crop_img, resample_img
    from scipy import ndimage
    #import dipy.tracking.utils as ut
    
    if isinstance(atlasNifti,str):
        atlas=nib.load(atlasNifti)
    else:
        atlas=atlasNifti
    
    #first perform the island removal operation
    #Claude note: the background label is passed on, so both steps agree
    [noIslandAtlas,report]=removeIslandsFromAtlas(atlasNifti,backgroundLabel=backgroundLabel)
    #just save it
    #Claude note: the location is caller-controlled; None skips saving
    if reportSavePath is not None:
        report.to_csv(reportSavePath)
    
    
    #now we need to detect the wm and bg labels
    #lots of ways we could do this, I guess
    #bg we can detect in the same way that we detected in the deIsland code
        
    #count the label instances, and make an inference as to the background value
    #it's almost certianly the most common value, eg. 0
    (labels,counts)=np.unique(np.asanyarray(atlas.dataobj), return_counts=True)
    #Claude note: explicit backgroundLabel, or the heuristic.  removeIslandsFromAtlas
    #above has already warned if the heuristic's answer looks doubtful, so the
    #warning is not repeated here.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        detectBackground=_resolveBackgroundLabel(labels,counts,backgroundLabel,'inflateAtlasIntoWMandBG')
    
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
    noIslandAtlasData=np.asanyarray(noIslandAtlas.dataobj)
    noIslandAtlasData=np.round(noIslandAtlasData).astype(int)
    
    
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
        #Claude note: the one-voxel padding is not data.  It is neither a
        #source nor a target, and is excluded from votes on both sides (else,
        #if the background label isn't 0, the padding's 0s could count as a
        #label to inflate from, and a target voxel could end up with no votes).
        paddingShell=np.ones(noIslandAtlasData.shape,dtype=bool)
        paddingShell[1:-1,1:-1,1:-1]=False
        dataMask[paddingShell]=False
        inflationTargetsMask[paddingShell]=False
        #inflate the datamask
        inflatedLabelsMask=scipy.ndimage.binary_dilation(dataMask)
        #find the intersection of the inflated data mask and the inflation targets
        infationArrayTargets=np.logical_and(inflatedLabelsMask,inflationTargetsMask)
        #find the coordinates for these targets
        inflationTargetCoords=np.asarray(np.where(infationArrayTargets)).T
    
        #this manuver is going to cost us .gif
        #NOTE, sequencing of coordinates may influence votes
        #create a dummy array for each inflation iteration to avoid this
        #Claude note: the per-voxel voting loop runs vectorised over all target
        #voxels in _inflationVotes; only tied votes go through the per-voxel
        #window-growing loop (_inflationVoteForVoxel).
        voteWinners=_inflationVotes(noIslandAtlasData,inflationTargetCoords,inflationLabelTargets)
        if len(inflationTargetCoords)>0:
            newIDentityHolder[inflationTargetCoords[:,0],inflationTargetCoords[:,1],inflationTargetCoords[:,2]]=voteWinners
            
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
    import numpy as np
    
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
    concatData=ndimage.binary_dilation(np.asanyarray(roiNifti.dataobj), iterations=iterations)
    
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
    import numpy as np
    
    if isinstance(inputMask,str):
        inputMask=nib.load(inputMask)
        
    #create a binarized copy of the mask
    binarizedMask=nib.Nifti1Image(np.asanyarray(inputMask.dataobj)>0, inputMask.affine, inputMask.header)
    
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
    if not np.any(np.asanyarray(intersectionNifti.dataobj)):
        import warnings
        warnings.warn("Empty mask for intersection returned; likely no mutual intersection between input " + str(len(ROIs)) + "ROIs")
                      
    return intersectionNifti

def changeLabelValue(inputParc,targetLabelNum,newLabelNum):
    import numpy as np
    import nibabel as nib
    parcData=np.asanyarray(inputParc.dataobj).astype(int)
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
    parc1Data=np.asanyarray(parc1.dataobj).astype(int)
    parc2Data=np.asanyarray(parc2.dataobj).astype(int)
    
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
        parc2Data=np.asanyarray(parc2.dataobj).astype(int)
    elif parc2DataSize>parc1DataSize:
        print ('parc2 size of ' + str(np.round(parc2DataSize/(1024**2))) + ' MB greater than parc1 size of ' + str(np.round(parc1DataSize/(1024**2))) + ' MB')
        print ('resampling to parc2' )
        parc1=resample_img(parc1,target_affine=parc2.affine,target_shape=(parc2.shape),interpolation='nearest')
        parc1Data=np.asanyarray(parc1.dataobj).astype(int)
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
    
    