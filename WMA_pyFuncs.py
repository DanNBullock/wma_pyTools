#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
essentially, a python refactoring of the relevant parts of 
https://github.com/DanNBullock/wma_tools

Dan Bullock, Nov 11, 2020
"""

def makePlanarROI(referenceNifti, mmPlane, dimension):
    #makePlanarROI(referenceNifti, mmPlane, dimension):
    #
    # INPUTS:
    # -referenceNifti:  the nifti that the ROI will be
    # applied to, also functions as the source of affine transform.
    #
    # -mmPlane: the ACPC (i.e. post affine application) mm plane that you would like to generate a planar ROI
    # at.  i.e. mmPlane=0 and dimension= x would be a planar roi situated along
    # the midsaggital plane.
    #
    # -dimension: either 'x', 'y', or 'z', to indicate the plane that you would
    # like the roi generated along
    #
    # OUTPUTS:
    # -planarROI: the roi structure of the planar ROI
    #
    #  Daniel Bullock 2020 Bloomington
    #this plane will be oblique to the subject's *actual anatomy* if they aren't
    #oriented orthogonally. As such, do this only after acpc-alignment
    #
    #adapted from https://github.com/DanNBullock/wma_tools/blob/master/ROI_Tools/bsc_makePlanarROI_v3.m
    
    #import nibabel and numpy
    import nibabel as nib
    import numpy as np
    #get the affine from the input nibabel nifti
    referenceAffine=referenceNifti.affine
    refrenceHeader=referenceNifti.header.copy()
    blankData=np.zeros(referenceNifti.shape)
    
    #set dimInt in accordance with input dim
    if dimension=='x':
        dimInt=0
    if dimension=='y':
        dimInt=1
    if dimension=='z':
        dimInt=2
        
    coord=[0,0,0]
    cordPlus=[0,0,0]

    #set value in appropriate dimension
    #but also find the adjacent coords
    cordPlus[dimInt]=mmPlane+1
    coord[dimInt]=mmPlane
    
    #apply affine to specific coord
    #floor because of how voxel indexing works
    convertedSlice=np.floor(nib.affines.apply_affine(np.linalg.inv(referenceNifti.affine),coord)).astype(int)
    convertedPlusSlice=np.floor(nib.affines.apply_affine(np.linalg.inv(referenceNifti.affine),cordPlus)).astype(int)
    diffDim=np.where(np.logical_not(convertedSlice==convertedPlusSlice))[0]

    #set all slice values to 1 to create planar ROI
    if diffDim==0:
        blankData[convertedSlice[diffDim],:,:]=1
    if diffDim==1:
        blankData[:,convertedSlice[diffDim],:]=1
    if diffDim==2:
        blankData[:,:,convertedSlice[diffDim]]=1
    
    #create description string for this 
    #cant figure out s80
    #descriptionString=referenceNifti.header['descrip']
    #testConcat=descriptionString+descriptionString
    #refrenceHeader['descrip']=
    
    #format output
    returnedNifti=nib.nifti1.Nifti1Image(blankData, referenceAffine, header=refrenceHeader)
    
    return returnedNifti

def roiFromAtlas(atlas,roiNum):
    #roiFromAtlas(atlas,roiNum)
    #creates a nifti structure mask for the input atlas image of the specified label
    #
    # INPUTS:
    # -atlas:  an atlas nifti
    #
    # -roiNum: an int input indicating the SINGLE label that is to be extracted.  Will throw warning if not present
    #
    # OUTPUTS:
    # -outImg:  a mask with int(1) in those voxels where the associated label was found.  If the label wasn't found, an empty nifti structure is output.

    import numpy as np
    import nibabel as nib
    outHeader = atlas.header.copy()
    atlasData = atlas.get_data()
    outData = np.zeros((atlasData.shape)).astype(int)
    #check to make sure it is in the atlas
    #not entirely sure how boolean array behavior works here
    if  np.isin(roiNum,atlasData):
            outData[atlasData==roiNum]=int(1)
    else:
        import warnings
        warnings.warn("WMA.roiFromAtlas WARNING: ROI label " + str(roiNum) + " not found in input Nifti structure.")
                
    outImg = nib.nifti1.Nifti1Image(outData, atlas.affine, outHeader)
    return outImg

def returnMaskBoundingBoxVoxelIndexes(maskNifti):
    #returnMaskBoundingBoxVoxelIndexes(atlas,roiNum):
    #returns the 8 verticies corresponding to the vertices of the mask's bounding box IN IMAGE SPACE
    #(i.e. indicating the voxel indexes that correspond to these points)
    #
    # INPUTS:
    # -maskNifti:  a nifti with ONLY 1 and 0 (int) as the content, a boolean mask, in essence
    #
    # OUTPUTS:
    # -boundingCoordArray:  an 8 by 3 array indicating the VoxelIndexes of the bounding box
    
    import nibabel as nib
    import numpy as np
    
    #get the data
    maskData = maskNifti.get_data()
    #check to make sure its a mask
    if len(np.unique(maskData))> 2:
      raise Exception("returnMaskBoundingBoxVoxelIndexes Error: Non mask input detected.  " + str(len(np.unique(maskData))) + " unique values detected in input NIfTI structure.")
    
    #find the indexes of the nonzero entries
    nonZeroIndexes=np.nonzero(maskData)
    #find the min and max for each dimension
    maxDim1=np.max(nonZeroIndexes[0])
    minDim1=np.min(nonZeroIndexes[0])
    maxDim2=np.max(nonZeroIndexes[1])
    minDim2=np.min(nonZeroIndexes[1])  
    maxDim3=np.max(nonZeroIndexes[2])
    minDim3=np.min(nonZeroIndexes[2])
    
    #user itertools and cartesian product
    import itertools
    outCoordnates=np.asarray(list(itertools.product([maxDim1,minDim1], [maxDim2,minDim2], [maxDim3,minDim3])))
    return outCoordnates

def planeAtMaskBorder(inputMask,relativePosition):
    #planeAtMaskBorder(inputNifti,roiNum,relativePosition):
    #creates a planar roi at the specified border of the specified ROI.
    #
    # INPUTS:
    # -inputMask:  a nifti with ONLY 1 and 0 (int) as the content, a boolean mask, in essence
    #
    # -relativePosition: string input indicating which border to obtain planar roi at
    # Valid inputs: 'superior','inferior','medial','lateral','anterior','posterior','rostral','caudal','left', or 'right'
    #
    # OUTPUTS:
    # outPlaneNifti: planar ROI as Nifti at specified border
    
    import nibabel as nib
    import numpy as np
    
    #establish valid positional terms
    validPositionTerms=['superior','inferior','medial','lateral','anterior','posterior','rostral','caudal','left','right']
    #cased relativePosition check
    #again, don't know how arrays work with booleans
    if ~np.isin(relativePosition.lower(),validPositionTerms):
         raise Exception("planeAtROIborder Error: input relative position " + relativePosition + " not valid.")
    
    #get the input NIfTI data
    #inputData = inputNifti.get_data()
    #check to ensure that inputNifti is a label structure of some type (mask or atlas)
    #using np. mod modulo
    #moduloCheckOut=np.mod(inputData,1)
    #NOTE, APPEARS T1 ONLY HAS INT AS WELL, SO NO REAL GOOD WAY TO TELL. 
    
    maskBoundCoords=returnMaskBoundingBoxVoxelIndexes(inputMask)
    
    #convert the coords to subject space in order to interpret positional terms
    convertedBoundCoords=nib.affines.apply_affine(inputMask.affine,maskBoundCoords)
    
    #switch cases for input term    
    def interpretPoisitionTerm(x):
        #might do some weird stuff for equal values
        return {
            'superior': np.max(convertedBoundCoords[:,2]),
            'inferior': np.min(convertedBoundCoords[:,2]),
            'medial':   np.min(convertedBoundCoords[np.min(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
            'lateral': np.max(convertedBoundCoords[np.max(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
            'anterior': np.max(convertedBoundCoords[:,1]),
            'posterior': np.min(convertedBoundCoords[:,1]),
            'rostral': np.max(convertedBoundCoords[:,1]),
            'caudal': np.min(convertedBoundCoords[:,1]),
            'left': np.min(convertedBoundCoords[:,0]),
            'right': np.max(convertedBoundCoords[:,0]),
        }[x]
    
    #switch cases to infer dimension of interest    
    def interpretDimension(x):
        #might do some weird stuff for equal values
        return {
            'superior': 'z',
            'inferior': 'z',
            'medial':   'x',
            'lateral': 'x',
            'anterior': 'y',
            'posterior': 'y',
            'rostral': 'y',
            'caudal': 'y',
            'left': 'x',
            'right': 'x',
        }[x]

    outPlaneNifti=makePlanarROI(inputMask,interpretPoisitionTerm(relativePosition) , interpretDimension(relativePosition))
    
    #return the planar roi you hvae created
    return outPlaneNifti

def createSphere(r, p, reference):
    """ create a sphere of given radius at some point p in the brain mask
    Args:
    r: radius of the sphere
    p: point (in subject coordinates of the brain mask, i.e. not the image space) of the center of the
    sphere)
    reference:  The reference nifti whose space the sphere will be in
    
    
    modified version of nltools sphere function which outputs the sphere ROI in the
    coordinate space of the input reference
    """
    import nibabel as nib
    import numpy as np

    dims = reference.shape
    
    imgCoord=np.floor(nib.affines.apply_affine(np.linalg.inv(reference.affine),p))

    dim1Vals= np.abs(np.arange(0, dims[0], reference.header.get_zooms()[0])-imgCoord[0])
    dim2Vals= np.abs(np.arange(0, dims[1], reference.header.get_zooms()[1])-imgCoord[1])
    dim3Vals= np.abs(np.arange(0, dims[2], reference.header.get_zooms()[2])-imgCoord[2])
    #ogrid doesnt work?
    x, y, z = np.meshgrid(np.floor(dim1Vals).astype(int), np.floor(dim2Vals).astype(int), np.floor(dim3Vals).astype(int),indexing='ij')          
    
    #maybe this works?
    mask_r = x*x + y*y + z*z <= r*r

    activation = np.zeros(dims)
    activation[mask_r] = 1
    #not sure of robustness to strange input affines, but seems to work

    return nib.Nifti1Image(activation, affine=reference.affine)

def multiROIrequestToMask(atlas,roiNums):
    #multiROIrequestToMask(atlas,roiNums):
    #creates a nifti structure mask for the input atlas image of the specified labels
    #
    # INPUTS:
    # -atlas:  an atlas nifti
    #
    # -roiNums: an 1d int array input indicating the labels that are to be extracted.  Singleton request (single int) will work fine.  Will throw warning if not present
    #
    # OUTPUTS:
    # -outImg:  a mask with int(1) in those voxels where the associated labels were found.  If the label wasn't found, an empty nifti structure is output.
    
    ##NOTE REPLACE WITH nil.masking.intersect_masks when you get a chance
    
    import numpy as np
    import nibabel as nib
    
    #force input roiNums to array, don't want to deal with lists and dicts
    roiNumsInArray=np.asarray(roiNums)
    
    #if its a singleton request, just use the standard roiFromAtlas function
    #done this way because for loop wasn't handling singleton case well
    if roiNumsInArray.size==1:
        concatOutNifti=roiFromAtlas(atlas,roiNumsInArray)
    
    #if its a multi request
    else:
    #setup blank nifti struc
        referenceAffine=atlas.affine
        refrenceHeader=atlas.header.copy()
        #create blank data structure
        concatData=np.zeros(np.append(np.asarray(atlas.shape),len(roiNumsInArray)))
    
        #create an initially blank nifti to serve as the basis of a concat
    
        #again, .size use might cause issues
        for iRequests in range(roiNumsInArray.size):
            #make a mask for the current request
            currentMask=roiFromAtlas(atlas,roiNumsInArray[iRequests])
            #get the data
            currentMaskData=currentMask.get_data()
            #set current dimension entries to maskData
            concatData[:,:,:,iRequests]=currentMaskData
    
        #make a blank structure for the output
        concatDataCondensed=np.zeros(atlas.shape)
        #check across the 4th dimension for non zero entries, create a mask using this
        concatDataCondensed[np.any(concatData!=0,3)]=1
        #insert this data into an output nifti structure
        concatOutNifti=nib.nifti1.Nifti1Image(concatDataCondensed, referenceAffine, header=refrenceHeader)
    
    return concatOutNifti

def planarROIFromAtlasLabelBorder(inputAtlas,roiNums, relativePosition):
    #planarROIFromAtlasLabelBorder(referenceNifti, mmPlane, dimension):
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
    #this plane will be oblique to the subject's *actual anatomy* if they aren't
    #oriented orthogonally. As such, do this only after acpc-alignment
    
    import nibabel as nib
    #merge the inputs if necessary
    mergedRequest=multiROIrequestToMask(inputAtlas,roiNums)
    
    #use that mask to generate a planar border
    planeOut=planeAtMaskBorder(mergedRequest,relativePosition)
    
    return(planeOut)
    
def sliceROIwithPlane(inputROINifti,inputPlanarROI,relativePosition):
    #sliceROIwithPlane(inputROINifti,planarROI,relativePosition):
    #slices input ROI Nifti using input planarROI and returns portion specified by relativePosition
    #
    # inputROINifti:  a (presumed ROI) nifti with ONLY 1 and 0 (int) as the content, a boolean mask, in essence
    #
    # planarROI: a planar roi (nifti) that is to be used to perform the slicing operation on the inputROINifti
    # 
    # relativePosition: which portion of the sliced ROI to return
    # Valid inputs: 'superior','inferior','medial','lateral','anterior','posterior','rostral','caudal','left', or 'right'
   
    #test for intersection between the ROIS
    import nibabel as nib
    import numpy as np
    import nilearn as nil
    from nilearn.masking import intersect_masks
    
    #get the data
    inputROINiftiData=inputROINifti.get_data()
    inputPlanarROIData=inputPlanarROI.get_data()
    
    #boolean to check if intersection
    intersectBool=np.any(np.logical_and(inputROINiftiData!=0,inputPlanarROIData!=0))
    if ~intersectBool:
        import warnings
        warnings.warn("WMA.sliceROIwithPlane WARNING: input planar ROI does not intersect with input ROI.")
    
    #find bounds for current planar ROI    
    boundsTable=findMaxMinPlaneBound(inputPlanarROI)

    #fix exception error for bad calls later
    #if boundsTable['boundLabel'].str.contains(relativePosition).any():
    #    print(boundsTable)
    #    raise Exception("sliceROIwithPlane Error: plane-term mismatch detected.  Input relativePosition term uninterpretable relative to inputPlanarROI.")
    

    #find the boundary that we will be filling to
    fillBound=int(float(boundsTable['boundValue'].loc[boundsTable['boundLabel']==relativePosition].to_numpy()[0]))
    sortedBounds=np.sort((fillBound,int(float(boundsTable['boundValue'].loc[0]))))
    #get the dimension indicator
    dimIndicator=int(float(boundsTable['dimIndex'].loc[0]))
    #create blank output structure    
    sliceKeepData=np.zeros(inputPlanarROIData.shape)  
    
    #there's probably a better way to do this,
    if dimIndicator==0:
        sliceKeepData[sortedBounds[0]:sortedBounds[1],:,:]=1
    elif dimIndicator==1:
        sliceKeepData[:,sortedBounds[0]:sortedBounds[1],:]=1
    elif dimIndicator==2:
        sliceKeepData[:,:,sortedBounds[0]:sortedBounds[1]]=1
    
    #create a nifti structure for this object
    sliceKeepNifti=nib.nifti1.Nifti1Image(sliceKeepData, inputPlanarROI.affine, header=inputPlanarROI.header)
    
    #intersect the ROIs to return the remaining portion
    #will cause a problem if Niftis have different affines.
    remainingROI=intersect_masks([sliceKeepNifti,inputROINifti], threshold=1, connected=False)
    #consider throwing an error here if the output Nifti is empty
    
    return remainingROI

def alignROItoReference(inputROI,reference):
    """ extracts the coordinates of an ROI and reinstantites them as an ROI in the refernce space of the reference input
    Helps avoid affine weirdness.
    Args:
    inputROI: an input ROI in nifti format
    reference: the reference nifti that you would like the ROI moved to.
        
    Outputs:
    outROI: output nifti ROI in the reference space of the input reference nifti

    """   
    import numpy as np
    import nibabel as nib
    from dipy.tracking.utils import seeds_from_mask
    
    densityKernel=np.asarray(reference.header.get_zooms())
    
    roiCoords=seeds_from_mask(inputROI.get_fdata(), inputROI.affine, density=densityKernel)
    
    outROI=pointCloudToMask(roiCoords,reference)
    return outROI
    
def findMaxMinPlaneBound(inputPlanarROI):
    #indMaxMinPlaneBound(inputPlanarROI):
    #finds the min and max (sensible) boundaries for the input planar ROI
    #
    # INPUTS
    #
    # -inputPlanarROI: a planar ROI nifti
    #
    # OUTPUTS
    # -boundsTable: a pandas table containing the boundary labels and boundary values for this plane
    import numpy as np
    import pandas as pd
    import nibabel as nib
    from dipy.tracking.utils import seeds_from_mask
    
    #specify a kernel for the point cloud
    densityKernel=np.asarray(inputPlanarROI.header.get_zooms())
    
    planeCoords=seeds_from_mask(inputPlanarROI.get_fdata(), inputPlanarROI.affine, density=densityKernel)
    
    #planeCoords=np.asarray(np.where(inputPlanarROIData!=0))
    #python indexing?
    findSingletonDimension=np.where(np.equal([len(np.unique(planeCoords[:,0])),len(np.unique(planeCoords[:,1])),len(np.unique(planeCoords[:,2]))],1))[0]
  
    singletonCoord=planeCoords[0,findSingletonDimension]
    
    centerPointImg=nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),[0,0,0])
    
    samplePlaneCoordSubj=[0,0,0]
    samplePlaneCoordSubj[findSingletonDimension[0]]=singletonCoord[0]
    samplePlaneCoordImg=nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),samplePlaneCoordSubj)
    
    relativeImgPosition=samplePlaneCoordImg-centerPointImg
    
    #we need to translate the axes in case the affine is weird, and it often is
    superiorPointTest=centerPointImg-nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),[0,0,10])
    leftPointTest=centerPointImg-nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),[-10,0,0])
    anteriorPointTest=centerPointImg-nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),[0,10,0])
    
    #use abs because sometimes there are flips and such.
    xDimIndex=np.where(np.abs(leftPointTest)==np.max(np.abs(leftPointTest)))[0]
    yDimIndex=np.where(np.abs(anteriorPointTest)==np.max(np.abs(anteriorPointTest)))[0]
    zDimIndex=np.where(np.abs(superiorPointTest)==np.max(np.abs(superiorPointTest)))[0]
    
    
    #NOTE: we're establishing all of these even if they aren't sensible
    #conversion of logic for bad affines
    #superiorBound = inputPlanarROI.shape[zDimIndex[0]]
    #inferiorBound = 0
    
    #casing for flipped axes
    if superiorPointTest[zDimIndex[0]]<0:
       superiorBound = inputPlanarROI.shape[zDimIndex[0]]
       inferiorBound = 0
    elif superiorPointTest[zDimIndex[0]]>0:
       superiorBound = 0
       inferiorBound = inputPlanarROI.shape[zDimIndex[0]]

    medialBound=   np.floor(centerPointImg[xDimIndex[0]])
    #now do a check for lateral bound for a planar roi?  Only matters in case of plane in x dimension, and if x is a particular side
    
    #if the singleton dimension is x and the plane is closer to the max end of the array
    if np.logical_and(findSingletonDimension[0] == 0, relativeImgPosition[xDimIndex[0]]<0):
        lateralBound=inputPlanarROI.shape[xDimIndex[0]]
        #if the singleton dimension is x and the plane is closer to the min end of the array, the lateral bound is zero
    elif np.logical_and(findSingletonDimension[0] == 0, relativeImgPosition[xDimIndex[0]]>0):
        lateralBound=0
        #if the plane of interest isnt an x plane, then the medial/lateral bound thing doesn't make sense, but we'll output the max val anyways
    elif np.logical_not(findSingletonDimension[0] == 0):
        lateralBound=inputPlanarROI.shape[xDimIndex[0]]
        
    if leftPointTest[xDimIndex[0]]>0:
        leftBound=inputPlanarROI.shape[xDimIndex[0]]
        rightBound=0
    elif leftPointTest[xDimIndex[0]]<0:
        leftBound=0
        rightBound=inputPlanarROI.shape[xDimIndex[0]]

    
       #casing for flipped axes
    if anteriorPointTest[yDimIndex[0]]<0:
       anteriorBound= inputPlanarROI.shape[yDimIndex[0]]
       posteriorBound = 0
       rostralBound= inputPlanarROI.shape[yDimIndex[0]]
       caudalBound= 0
    elif anteriorPointTest[yDimIndex[0]]>0:
       anteriorBound= 0
       posteriorBound = inputPlanarROI.shape[yDimIndex[0]]
       rostralBound= 0
       caudalBound= inputPlanarROI.shape[yDimIndex[0]]

    #conditional assignment of validStrings and boundVals
    #these dimensional inferences are sound
    
    #allbounds=['posterior','anterior','caudal','rostral','medial','lateral','left', 'right','inferior','superior']
    #also all bounds= [posteriorBound,anteriorBound,caudalBound,rostralBound,medialBound,lateralBound,leftBound,rightBound, inferiorBound,superiorBound]

    borderStrings=['posterior','anterior','caudal','rostral', 'medial','lateral','left', 'right','inferior','superior']
    boundVals=[posteriorBound,anteriorBound,caudalBound,rostralBound,medialBound,lateralBound,leftBound,rightBound, inferiorBound,superiorBound]
    
    dimIndexColumn=[findSingletonDimension[0],yDimIndex[0],yDimIndex[0],yDimIndex[0],yDimIndex[0],xDimIndex[0],xDimIndex[0],xDimIndex[0],xDimIndex[0],zDimIndex[0],zDimIndex[0]]
    
    labelColumn=['exactDim_'+str(findSingletonDimension[0])]+borderStrings
    valueColumn=np.concatenate((samplePlaneCoordImg[findSingletonDimension],np.asarray(boundVals)))
    #fill the out data structure
    boundsTable=pd.DataFrame(np.array([labelColumn, valueColumn,dimIndexColumn]).T,
                              columns=['boundLabel', 'boundValue','dimIndex'])
    return boundsTable
    
def segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #because we want (1) distinct "mode" inputs, (2) nifti inputs instead of coordinate cloud ROI inputs (3)a boolean output rather than a generator
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
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
    #
    import numpy as np
    
    #create an array to store the boolean result of each round of segmentation
    
    
    outBoolArray=np.zeros(streamlines,len(roisvec))
    
    for iOperations in range(len(roisvec)):
        
        curBoolVec=applyNiftiCriteriaToTract(streamlines, roisvec[iOperations], includeVec[iOperations], operationsVec[iOperations])
        
        #if this is the first segmentation application
        if iOperations == 0:
            #set the relevant column to the current segmentation bool vec
            outBoolArray[:,iOperations]=curBoolVec
        #otherwise
        else:
            lastRoundSurvivingIndexes=np.where(outBoolArray[:,iOperations-1])[0]
            thisRoundSurvivingIndexes=lastRoundSurvivingIndexes[np.where(outBoolArray[:,iOperations])[0]]
        
            #set relevant array entries to true
            outBoolArray[thisRoundSurvivingIndexes,iOperations]=1
        
        #in either case, subsegment the streamlines to speed up the next iteration
        streamlines=streamlines[np.where(curBoolVec)[0]]
        
    #when all iterations are complete collapse across the colums and return only those streams that met all criteria
    outBoolVec=np.all(outBoolArray,axis=1)
    
    return outBoolVec
   
def applyNiftiCriteriaToTract(streamlines, maskNifti, includeBool, operationSpec):
    #segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #because we want (1) nifti inputs instead of coordinate cloud ROI inputs (2)a boolean output rather than a generator
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
    # operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    
    #still learning how to import from modules
    from dipy.tracking.utils import seeds_from_mask
    import numpy as np
    from dipy.core.geometry import dist_to_corner
    import dipy.tracking.utils as ut
    import nilearn as nil
    import nibabel as nib
    from nilearn import masking 
    
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
    #lets intersect the ROI and a mask of the streamlines so that we can subset the relevant portions of the ROI
    
    tractMask=pointCloudToMask(np.concatenate(streamlines),maskNifti)
    
    #take the intersection of the two
    tractMaskROIIntersection=masking.intersect_masks([tractMask,maskNifti], threshold=1, connected=False)
    #ideally we are reducing the number of points that teach node needs to be compared to
    #for example, a 256x 256 planar ROI would result in 65536 coords, intersecting with a full brain tractogram reduces this by about 1/3
    
    #find the streamlines that are within the bounding box of the maskROI
    boundedStreamsBool=subsetStreamsByROIboundingBox(streamlines, tractMaskROIIntersection)
    
    #subset them
    boundedStreamSubset=streamlines[np.where(boundedStreamsBool)[0]]
    #second track mask application doesn't seem to do anything    
    
    #dipy interprets ROIs as point clouds, which necessitates following conversion.  Kind of
    #similar to how vistasoft used to use point clouds instead of niftis.  I/We are 
    #making the switch to using Niftis here because that's how Nibabel and the other stuff
    #we have been using has done it.
    
    #here we are going to take the input mask at its word and use the resolution
    # i.e. (diagonal of the affine, but taken from get_zooms due to unreliability of diagonal itself) to infer the actual data granularity of the mask
    # we're assuming that this is the floor of the sensitivity of the source data
    # and so it would be "making up data" to specify a point cloud spacing kernel
    # smaller than this
    densityKernel=np.asarray(tractMaskROIIntersection.header.get_zooms())
    
    roiPointCloud=seeds_from_mask(tractMaskROIIntersection.get_fdata(), tractMaskROIIntersection.affine, density=densityKernel)
    
    # https://github.com/dipy/dipy/blob/558adb604865877e1835cd03433f71cb6d851d21/dipy/tracking/streamline.py#L302
    # This calculates the maximal distance to a corner of the voxel:
    dtc = dist_to_corner(maskNifti.affine)
    #going to use this calculation to impose tolerance.  Can't see why users would need to specify this
    
    #we can use the mask bounds to subset the nodes in each streamline which warrant consideration.
    #this nets an even more signifigant speed-up per streamline, and is possible because our output is bool rather than streamline
    maskBounds=nib.affines.apply_affine(maskNifti.affine,returnMaskBoundingBoxVoxelIndexes(maskNifti))
    #due to how this output is structured, vertex 0 and vertex 8 are always opposing
    maskBoundTolerance=[np.min(maskBounds[[0,7],0])-dtc,np.max(maskBounds[[0,7],0])+dtc,np.min(maskBounds[[0,7],1])-dtc,np.max(maskBounds[[0,7],1])+dtc,np.min(maskBounds[[0,7],2])-dtc,np.max(maskBounds[[0,7],2])+dtc]
    

    criteriaVec=np.zeros(len(boundedStreamSubset)).astype(int)
    for iStreamline in range(len(boundedStreamSubset)):
        
        #Here's where
        #we can use the mask bounds to subset the nodes in each streamline which warrant consideration.
        #this nets an even more signifigant speed-up per streamline, and is possible because our output is bool rather than streamline
        #Why?
        #because the distance computation is (1) an all to all computation and
        # (2) because it involves a multi-step euclidian distance computation for each of those
        #obtain current streamline
        curStreamline=boundedStreamSubset[iStreamline]
        #create a boolean array for each boundary indicating criteria satisfaction for each node
        curBoundsBoolArray=np.vstack((curStreamline[:,0]>maskBoundTolerance[0], \
                                      curStreamline[:,0]<maskBoundTolerance[1], \
                                      curStreamline[:,1]>maskBoundTolerance[2], \
                                      curStreamline[:,1]<maskBoundTolerance[3], \
                                      curStreamline[:,2]>maskBoundTolerance[4], \
                                      curStreamline[:,2]<maskBoundTolerance[5])).T                                 
        #extract current nodes which satisfy the boundary criteria                              
        curNodes=curStreamline[np.all(curBoundsBoolArray,axis=1),:]
        
        #if there are nodes for this streamline within the bounding box
        if curNodes.size>0:
            criteriaVec[iStreamline] = ut.streamline_near_roi(curNodes, roiPointCloud, tol=dtc, mode=operationSpec)
        #otherwise, if there are no nodes within the bounding box, you don't need to do the computation, you know that none of them will satisfy the criteria
        else:
            criteriaVec[iStreamline]=0
    
    #if the input instruction is NOT, negate the output
    if np.logical_not(includeBool):
        criteriaVec=np.logical_not(criteriaVec)
    
    #find the indexes of the original streamlines that the survivors correspond to
    boundedStreamsIndexes=np.where(boundedStreamsBool)[0]
    originalIndexes=boundedStreamsIndexes[np.where(criteriaVec.astype(int))[0]]
    
    #initalize an out bool vec
    outBoolVec=np.zeros(len(streamlines))
    #set the relevant entries to true
    outBoolVec[originalIndexes]=1
    
    return outBoolVec
    
    
def pointCloudToMask(pointCloudArray,referenceNifti):
    #pointCloudToMask(streamlines, referenceNifti)
    #creates a mask of a point cloud in the space and resolution of the input referenceNifti
    #basically the reverse of seeds_from_mask
    # 
    # INPUTS
    # -pointCloudArray: N X 3 array of points, corresponding to a point cloud
    #
    # -referenceNifti: reference nifti to obtain affine and data structure size
    #
    # OUTPUTS
    #
    #  cloudMaskNifti: mask of cloud, in nifti format
    #
    # NOTE: will throw error if tract mask doesn't fit in nifti bounds
    
    import nibabel as nib
    import numpy as np
    
    referenceAffine=referenceNifti.affine
    refrenceHeader=referenceNifti.header.copy()
    maskData=np.zeros(referenceNifti.shape)
    
    #there may be an offset issue here due to the smaller than 1 offset of the affine
    maskImgCoords=np.unique(np.floor(nib.affines.apply_affine(np.linalg.inv(referenceNifti.affine),pointCloudArray)),axis=0).astype(int)
    
    #throw exception if mask bounds exceed niftiBounds
    maskBounds=np.asarray([np.min(maskImgCoords[:,0]),np.max(maskImgCoords[:,0]),np.min(maskImgCoords[:,1]),np.max(maskImgCoords[:,1]),np.min(maskImgCoords[:,2]),np.max(maskImgCoords[:,2])])
    if np.logical_or(np.any(maskBounds[0:6:2]<0),np.any([maskBounds[1]>referenceNifti.shape[0],maskBounds[3]>referenceNifti.shape[1],maskBounds[5]>referenceNifti.shape[2]])):
        raise Exception("tractMask Error: cloud mask exceends reference nifti bounding box.  Possible mismatch")
    
    #fill in the mask data.
    #array indexing misbehaves when you use  maskData[maskImgCoords], don't know why
    maskData[maskImgCoords[:,0],maskImgCoords[:,1],maskImgCoords[:,2]]=1
    
    #create out structure
    cloudMaskNifti=nib.nifti1.Nifti1Image(maskData, referenceAffine, header=referenceNifti.header)
    
    return cloudMaskNifti

def subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, whcih will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    import nibabel as nib
    import numpy as np
    #compute distance tolerance
    from dipy.core.geometry import dist_to_corner
    dtc = dist_to_corner(maskNifti.affine)
    
    maskBounds=nib.affines.apply_affine(maskNifti.affine,returnMaskBoundingBoxVoxelIndexes(maskNifti))
    #due to how this output is structured, vertex 0 and vertex 8 are always opposing
    
    maskBoundTolerance=[np.min(maskBounds[[0,7],0])-dtc,np.max(maskBounds[[0,7],0])+dtc,np.min(maskBounds[[0,7],1])-dtc,np.max(maskBounds[[0,7],1])+dtc,np.min(maskBounds[[0,7],2])-dtc,np.max(maskBounds[[0,7],2])+dtc]
    
    
    criteriaVec=np.zeros(len(streamlines))
    #iterating across streamlines seems inefficient...
    #but i guess this avoids doing trig on these points
    #seems WAAAAY faster than the standard per-streamline intersect code
    for iStreamline in range(len(streamlines)):
        
        #is there an all() version of this that would be faster?
        #checking for being bounded for each streamline
        criteriaVec[iStreamline]=np.all([np.logical_and(np.any(streamlines[iStreamline][:,0]>maskBoundTolerance[0]),np.any(streamlines[iStreamline][:,0]<maskBoundTolerance[1])), \
                                                 np.logical_and(np.any(streamlines[iStreamline][:,1]>maskBoundTolerance[2]),np.any(streamlines[iStreamline][:,1]<maskBoundTolerance[3])), \
                                                 np.logical_and(np.any(streamlines[iStreamline][:,2]>maskBoundTolerance[4]),np.any(streamlines[iStreamline][:,2]<maskBoundTolerance[5]))
                                                ])
   
    return criteriaVec.astype(int)



def findTractNeckNode(streamlines):
    #findTractNeckNode(streamlines):
    # finds the node index for each streamline which corresponds to the most tightly constrained
    # portion of the tract (i.e. the "neck).
    #
    # INPUTS
    #
    #-streamlines: appropriately formatted list of candidate streamlines, presumably corresponding to a coherent anatomical SUBSTRUCTURE (i.e. tract)
    # NOTE: this computation isn't really sensible for a whole brain tractome, and would probably take a long time as well 
    #
    # OUTPUTS
    #
    # -neckNodeVec:  a 1d int vector array that indicates, for each streamline, the node that is associated with the "neck" of the input streamline collection.
    #
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    from scipy.spatial.distance import cdist
    import numpy as np
    
    #throw warning for singleton input
    if len(streamlines)==1:
        import warnings
        warnings.warn("WMA.findTractNeckNode WARNING: singleton streamline input detected.")  
    
    #arbitrarily set the number of nodes to sample the centroid at.  Possible parameter input
    centroidNodesNum=35
    # Streamlines will be resampled to 24 points on the fly.
    feature = ResampleFeature(nb_points=35)
    #?
    metric = AveragePointwiseEuclideanMetric(feature=feature)  # a.k.a. MDF
    #threshold set very high to return 1 bundle
    qb = QuickBundles(threshold=100., metric=metric)
    #obtain a single cluser on the input streamlines
    clusters = qb.cluster(streamlines)
    
    #create vectors for the average and standard deviation of the neck point distance
    neckPointDistAvg=np.zeros(centroidNodesNum)
    neckPointDistStDev=np.zeros(centroidNodesNum)
    
    #iterate across centroid nodes.  To speed this up for future implementations, could 
    #only target middle ones, or sample some minimal number and continue to sample if min is fewer than 2 nodes from edge
    # i.e. looking for local min
    for iCentroidNodes in range(centroidNodesNum):

        #I don't understand 2d array generation with 1 row, so this is how we do it
        middleCentroidVec=np.zeros((1,3))
        middleCentroidVec[0,:]=clusters.centroids[0][iCentroidNodes,:]
    
        #create a vector to catch the indexes and the node values
        neckNodeIndexVec=np.zeros(len(streamlines)).astype(int)
        neckNodeVec=np.zeros((len(streamlines),3))
        
        #iterate across streamlines
        for iStreamline in range(len(streamlines)):
            #distances for all nodes 
            curNodesDist = cdist(streamlines[iStreamline], middleCentroidVec, 'euclidean')
            neckNodeIndexVec[iStreamline]=np.where(curNodesDist==np.min(curNodesDist))[0].astype(int)
            neckNodeVec[iStreamline,:]=streamlines[iStreamline][neckNodeIndexVec[iStreamline]]
    
        avgNeckPoint=np.zeros((1,3))
        avgNeckPoint[0,:]=np.mean(neckNodeVec,axis=0)
        curNearDistsFromAvg=cdist(neckNodeVec, avgNeckPoint, 'euclidean')
        neckPointDistAvg[iCentroidNodes]=np.mean(curNearDistsFromAvg)
        neckPointDistStDev[iCentroidNodes]=np.std(curNearDistsFromAvg)
    
    #find the neckNode on the centroid    
    centroidNeckNode=np.where(neckPointDistAvg==np.min(neckPointDistAvg))[0]
    
    #run the computation one last time to get the nearest streamline nodes for the determined centroid node
    neckNodeIndexVecOut=np.zeros(len(streamlines)).astype(int)
    neckNodeVec=np.zeros((len(streamlines),3))
        
    #iterate across streamlines
    for iStreamline in range(len(streamlines)):
        #distances for all nodes 
        curNodesDist = cdist(streamlines[iStreamline], clusters.centroids[0][centroidNeckNode,:], 'euclidean')
        neckNodeIndexVecOut[iStreamline]=np.where(curNodesDist==np.min(curNodesDist))[0].astype(int)
        neckNodeVec[iStreamline,:]=streamlines[iStreamline][neckNodeIndexVec[iStreamline]]
        
    return neckNodeIndexVecOut

def removeStreamlineOutliersAtNeck(streamlines,cutStDev):
    # removeStreamlineOutliersAtNeck(streamlines,cutStDev):
    # INPUTS
    #
    #-streamlines: appropriately formatted list of candidate streamlines, presumably corresponding to a coherent anatomical SUBSTRUCTURE (i.e. tract)
    # NOTE: this computation isn't really sensible for a whole brain tractome, and would probably take a long time as well 
    #
    # OUTPUTS
    #
    # -streamlinesCleaned:  a subset of the input streamlines, corresponding to those which have survived the cleaning process.
    #
    # NOTE: given that we are using a lognormal distribution and the 
    #
    import scipy.stats as stats  
    import numpy as np
    from scipy.spatial.distance import cdist
    
    
    neckNodeIndexVecOut=findTractNeckNode(streamlines)
    
    #recover the nodes
    neckNodes=np.zeros((len(streamlines),3))
    for iStreamline in range(len(streamlines)):
        #distances for all nodes 
        neckNodes[iStreamline,:] =streamlines[iStreamline][neckNodeIndexVecOut[iStreamline]]
    
    #compute the statistics on the average neck point
    avgNeckPoint=np.zeros((1,3))
    avgNeckPoint[0,:]=np.mean(neckNodes,axis=0)
    curNearDistsFromAvg=cdist(neckNodes, avgNeckPoint, 'euclidean')
    #neckPointDistAvg=np.mean(curNearDistsFromAvg)
    #neckPointDistStDev=np.std(curNearDistsFromAvg)
    
    #deviation from centroid is typical lognorm?
    #lognormOut[0]=shape param, lognormOut[2]=scaleParam
    sSigma, loc, scale = stats.lognorm.fit(curNearDistsFromAvg, floc=0)
    muVar=np.log(scale)
    #https://www.mathworks.com/help/stats/lognormal-distribution.html
    computedMean=np.exp(muVar+np.square(sSigma)/2)
    computedStDev=muVar
    
    #confidence interval
    #np.sum(np.logical_or(curNearDistsFromAvg<computedMean-2*computedStDev,curNearDistsFromAvg<computedMean+2*computedStDev))
    #curNearDistsFromAvg[curNearDistsFromAvg>computedMean+5*computedStDev]
    
    #finish later, maybe not necessary
    
    streamlinesCleaned=streamlines[curNearDistsFromAvg.flatten()<computedMean+cutStDev*computedStDev]
    
    return streamlinesCleaned

    
def shiftBundleAssignment(clusters,targetCluster,streamIndexesToMove):
    #shiftBundleAssignment(clusters,targetCluster,streamIndexesToMove)
    #
    # This function is, in essence, a workaround for quickbundles, which has a method
    # for assigning streamlines to a cluster, but doesn't also take the additional
    # step of removing those streamlines from existing clusters.
    # https://github.com/dipy/dipy/blob/ed71831f6a9e048961b00af10f1f381e2da63efe/dipy/segment/clustering.py#L103
    #
    # clusters: the clusters object output from qb.cluster
    #
    # targetCluster: the index of the cluster that we will be adding stream indexes to
    #
    # streamIndexesToMove: the indexes that we will be adding to the targetCluster and removing from all other clusters
    from dipy.segment.clustering import QuickBundles
    import numpy as np
    
    indexesRecord=[]
    
    #lets begin by removing the indexes from all clusters
    for iClusters in range(len(clusters)):
        #if any of the to remove indexes are in the current cluster
        if np.any(np.isin(streamIndexesToMove,clusters.clusters[iClusters].indices)):
            #extract the current cluster indexes
            currIndexes=clusters.clusters[iClusters].indices
            
            #add to the record
            indexesRecord.append(currIndexes)
            currToRemoveIndexes=np.where(np.isin(clusters.clusters[iClusters].indices ,streamIndexesToMove))[0]
            fixedIndexes=np.delete(currIndexes,currToRemoveIndexes)
            clusters.clusters[iClusters].indices=fixedIndexes
            #is this necessary?
            clusters.clusters[iClusters].update()
            
    #now perform a check to ensure that the request was sensible
    #ugly way to program this check
    #if there are any indexes that you requested to move that ARE NOT in any of the associated clusters
    flattenIndexes = lambda t: [item for sublist in t for item in sublist]
    if len(np.where(np.logical_not(np.isin(streamIndexesToMove,np.asarray(flattenIndexes(indexesRecord)))))[0])>0:
        #throw an error, because something probably went wrong with your request
        #probably not formatted correctly to begin with, and will throw an error while throwing an error.  Dog.
     raise Exception("shiftBundleAssignment Error: requested indexes" + str(streamIndexesToMove[np.where(np.logical_not(np.isin(streamIndexesToMove,indexesRecord)))[0]]) +  " not found in any cluster.  Malformed request, possible track/tractome-cluster mismatch.")
     
    #indexes removed and request viability confirmed, now add requested streams to relevant cluster
    #luckily we have a built-in method for this
    #clusters.clusters[targetCluster].asign(streamIndexesToMove)
    
    #except I cant figure out how to get it to work so, brute force
    
    clusters.clusters[targetCluster].indices=np.union1d(clusters[targetCluster].indices,streamIndexesToMove)
    clusters.clusters[targetCluster].update()
    
    #fixed?
    return clusters

def neckmentation(streamlines):
    #neckmentation(streamlines)
    #
    # a neck-based segmentation using findTractNeckNode and quickbundles
    #
    # INPUTS
    #
    # -streamlines: an input tractome to be segmented
    #
    
    import numpy as np
    from scipy.spatial.distance import cdist
    import itertools
    
    #set tolerance for how far apart bundles can be to be merged
    #initial investigations indicate that mean distance from centroid is ~ 3, so given that this is on both sides,
    #we should half it to ensure that they are close to one another
    #we'll start with 2, just to be generous
    distanceThresh=2
    
    
    #import dipy and perform quickBundles
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    centroidNodesNum=100
    # Streamlines will be resampled to 24 points on the fly.
    feature = ResampleFeature(nb_points=centroidNodesNum)
    #?
    metric = AveragePointwiseEuclideanMetric(feature=feature)  # a.k.a. MDF
    #threshold set very high to return 1 bundle
    qb = QuickBundles(threshold=5., metric=metric)
    #get the centroid clusters or clusters, dont know which this is
    clusters = qb.cluster(streamlines)
    
    #do it twice
    clusters=mergeBundlesViaNeck(streamlines,clusters,distanceThresh)
    
    clusters=mergeBundlesViaNeck(streamlines,clusters,distanceThresh)
    
    return clusters
    
    #create a blank vector for the neck node coordinates
  
    
def mergeBundlesViaNeck(streamlines,clusters,distanceThresh):
    #mergeBundlesViaNeck(clusters,distanceThresh):
    #
    #merges bundles based on how close the necks of the bundles are
    #
    # -streamlines: an input tractome to be segmented
    #
    # -clusters:  a clusters object, an output from qb.cluster
    #
    # distanceThresh the threshold between neck centroids that is accepted for merging
    
    import numpy as np
    from scipy.spatial.distance import cdist
    import itertools
    
    
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    
    neckNodes=np.zeros((len(clusters),3))
    for iClusters in range(len(clusters)):
        print(iClusters)
        currentBundle=streamlines[clusters.clusters[iClusters].indices]
        if len (currentBundle)>1:
            currentNeckIndexes=findTractNeckNode(currentBundle)
            currentNeckNodes=np.zeros((len(currentNeckIndexes),3))
        
            for iStreams in range(len(currentBundle)):
                currentNeckNodes[iStreams,:]=currentBundle[iStreams][currentNeckIndexes[iStreams]]
        
            
            #if it's a singleton streamline, just guess the midpoint?
        else:
            currentNeckNodes=np.zeros((1,3))
            currentNeckNodes[0,:]=currentBundle[0][np.floor(currentBundle[0].shape[0]/2).astype(int),:]
        neckNodes[iClusters,:]=np.mean(currentNeckNodes,axis=0)
        currentNeckNodes=[]
        
    #now do the distance computation
    neckNodeDistanceArray=cdist(neckNodes,neckNodes,metric='euclidean')
    withinThreshNecks=np.asarray(np.where(neckNodeDistanceArray<distanceThresh))        
    withinThreshNecksNotIdent=withinThreshNecks[:,np.where(~np.equal(withinThreshNecks[0,:],withinThreshNecks[1,:]))[0]]
    #perform a check to ensure that we do not waste time on singleton merge attempts
    #now we need to find the clusters that are empty
    streamCountVec=np.zeros(len(clusters))    
    for iClusters in range(len(clusters)):
        streamCountVec[iClusters]=len(clusters.clusters[iClusters].indices)
    
    singletonIndexes=np.where(streamCountVec==1)
    np.where(np.all(~np.isin(withinThreshNecksNotIdent,singletonIndexes),axis=0))
    
    
    
    
    possibleMerges=list([])    
    for iMerges in range(withinThreshNecksNotIdent.shape[1]):
        currentCandidates=withinThreshNecksNotIdent[:,iMerges]
        clusterOneInstances=np.asarray(np.where(withinThreshNecksNotIdent==currentCandidates[0]))[1,:]
        clusterTwoInstances=np.asarray(np.where(withinThreshNecksNotIdent==currentCandidates[1]))[1,:]
        
        withinThreshIndexesToCheck=np.union1d(clusterOneInstances,clusterTwoInstances)
        
        currentCentroidIndexes=np.unique(withinThreshNecksNotIdent[:,withinThreshIndexesToCheck])
        
        newCentroidsArray=np.vstack((neckNodes[currentCentroidIndexes,:],np.mean(neckNodes[currentCentroidIndexes,:],axis=0)))
        
        curDistArray=cdist(newCentroidsArray,newCentroidsArray,metric='euclidean')
        
        #its within the bounds
        if np.max(curDistArray[:,-1])<np.max(curDistArray[:,-2]):
            possibleMerges.append(currentCentroidIndexes)
        else:
            possibleMerges.append(currentCandidates)
            
    possibleMerges.sort(key=len)     
    possibleMerges.reverse()
    
    mergedBundles=[]
    for iMerges in range(len(possibleMerges)):
        
        currentCandidates=possibleMerges[iMerges]
        remainToMerge=np.setdiff1d(currentCandidates,mergedBundles)
        if len(remainToMerge)>1:
            print(iMerges)
            print(remainToMerge)
            for iBundles in range(1,len(remainToMerge)):
                clusters=shiftBundleAssignment(clusters,currentCandidates[0],clusters.clusters[remainToMerge[iBundles]].indices)  
            mergedBundles=np.append(mergedBundles,remainToMerge)
        #otherwise
        #do nothing, there is no bundle to merge
    
    #now we need to find the clusters that are empty
    streamCountVec=np.zeros(len(clusters))    
    for iClusters in range(len(clusters)):
        streamCountVec[iClusters]=len(clusters.clusters[iClusters].indices)
        
    #now that we have those indicies we have to go through them IN REVERSE ORDER
    #in order to remove them, because deleting them changes the index sequence for all subsequent clusters.
    #there has to be a better implementation of this process, but this is where we are.
    #clusters.remove_cluster DOESNT WORK, due to "only integer scalar arrays can be converted to a scalar index"
    toDeleteClusters=np.where(streamCountVec==0)[0]

    flippedToDelete=np.flip(toDeleteClusters)
    #this is asinine, but I can't figure out another way to do it.
    for iDeletes in range(len(flippedToDelete)):
        currentToDelete=flippedToDelete[iDeletes]
        currentCluster=clusters.clusters[currentToDelete]
        clusters.remove_cluster(currentCluster)
    
    return clusters

