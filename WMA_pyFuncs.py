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
    #set value in appropriate dimension
    coord[dimInt]=mmPlane
    
    #apply affine to specific coord
    #floor because of how voxel indexing works
    convertedSlice=np.floor(nib.affines.apply_affine(referenceNifti.affine,coord)).astype(int)


    #set all slice values to 1 to create planar ROI
    if dimension=='x':
        blankData[convertedSlice[dimInt],:,:]=1
    if dimension=='y':
        blankData[:,convertedSlice[dimInt],:]=1
    if dimension=='z':
        blankData[:,:,convertedSlice[dimInt]]=1
    
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
    if len(np.unique(maskData))!=2:
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

    if boundsTable['boundLabel'].str.contains(relativePosition).any():
        raise Exception("sliceROIwithPlane Error: plane-term mismatch detected.  Input relativePosition term uninterpretable relative to inputPlanarROI.")
    
    #find the boundary that we will be filling to
    fillBound=int(boundsTable['boundValue'].loc[boundsTable['boundLabel']==relativePosition].to_numpy()[0])
    sortedBounds=np.sort((fillBound,int(boundsTable['boundValue'].loc[0])))
    #get the dimension indicator
    dimIndicator=int(boundsTable['boundLabel'].loc[0][-1])
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
    remainingROI=nil.masking.intersect_masks([sliceKeepNifti,inputROINifti], threshold=0, connected=False)
    #consider throwing an error here if the output Nifti is empty
    
    return remainingROI
    
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

    
    inputPlanarROIData=inputPlanarROI.get_data()
    
    planeCoords=np.asarray(np.where(inputPlanarROIData!=0))
    #python indexing?
    findSingletonDimension=np.where(np.equal([len(np.unique(planeCoords[0,:])),len(np.unique(planeCoords[1,:])),len(np.unique(planeCoords[2,:]))],1))[0]
  
    singletonCoord=planeCoords[findSingletonDimension,0]
    
    centerPoint=nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),[0,0,0])
    
    #NOTE: we're establishing all of these even if they aren't sensible

    superiorBound = inputPlanarROI.shape[2]
    inferiorBound = 0
    medialBound=   np.floor(centerPoint[0])
    #now do a check for lateral bound
    if centerPoint[0] < singletonCoord:
        lateralBound=inputPlanarROI.shape[0]
    elif centerPoint[0] > singletonCoord:
        lateralBound=0
    
    anteriorBound= inputPlanarROI.she[1]
    posteriorBound = 0
    rostralBound= inputPlanarROI.shape[1]
    caudalBound= 0
    #matbe find a way to detect this from the header info or affine
    #leftTest, theoretically negative is always left with RAS
    leftPoint=nib.affines.apply_affine(np.linalg.inv(inputPlanarROI.affine),[-10,0,0])
    if centerPoint[0]<leftPoint[0]:
        leftBound=inputPlanarROI.shape[0]
        rightBound=0
    elif centerPoint[0]>leftPoint[0]:
        leftBound=0
        rightBound=inputPlanarROI.shape[0]

    #conditional assignment of validStrings and boundVals
    if findSingletonDimension==0:
        validStrings=['medial','lateral','left', 'right']
        boundVals=[medialBound,lateralBound,leftBound,rightBound]
    elif findSingletonDimension==1:
        validStrings=['posterior','anterior','caudal','rostral']
        boundVals=[posteriorBound,anteriorBound,caudalBound,rostralBound]
    elif findSingletonDimension==2:
         validStrings=['inferior','superior']
         boundVals=[inferiorBound,superiorBound]
         
    labelColumn=['exactDim'+str(findSingletonDimension[0])]+validStrings
    valueColumn=np.concatenate((singletonCoord,np.asarray(boundVals)))
    #fill the out data structure
    boundsTable=pd.DataFrame(np.array([labelColumn, valueColumn]).T,
                              columns=['boundLabel', 'boundValue'])
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
    # NOTE: roisvec, includeVec, and operationsVec should all be the same lenth
    
    from dipy import tracking, utils
    testCloud=dipy.tracking.utils.seeds_from_mask(inputPlanarROI.get_data(), inputPlanarROI.affine, density=[1, 1, 1])