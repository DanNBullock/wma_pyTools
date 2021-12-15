# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:22 2021

@author: Daniel
"""
def streamGeomQuantifications(tractogram):
    #streamGeomQuantifications(tractogram)
    #
    #This function quantifies a number of streamline-based quantities
    #in the same fashion as wma_tools's  ConnectomeTestQ
    #
    # INPUTS
    #
    # tractogram: an input stateful tractogram
    #
    # OUTPUTS
    # 
    # quantificationTable: a pandas table documenting the streamline based
    #                      quantificaton.  
    # see https://github.com/DanNBullock/wma_tools#connectometestq
    # for more details.
    #
    # begin code
    import pandas as pd
    import numpy as np
    #establish the dataframe
    column_names = ["length", "fullDisp", "efficiencyRat", "asymRat", "bioPriorCost"]
    quantificationTable = pd.DataFrame(columns = column_names)
    #begin the iteration
    from dipy.tracking.streamline import length
    import math
    for iStreamlines in tractogram.streamlines:
        #compute lengths
        streamLength=length(iStreamlines)
        firstHalfLength=length(iStreamlines[1:int(round(len(iStreamlines)/2)),:])
        secondHalfLength=length(iStreamlines[int(round(len(iStreamlines)/2))+1:-1,:])
    
        #compute displacements
        displacement=math.dist(iStreamlines[1,:],iStreamlines[-1,:]) 
        firstHalfDisp=math.dist(iStreamlines[1,:],iStreamlines[int(round(len(iStreamlines)/2)),:])
        secondHalfDisp=math.dist(iStreamlines[int(round(len(iStreamlines)/2))+1,:],iStreamlines[-1,:])
    
        #compute ratios
        efficiencyRatio=displacement/streamLength
        asymetryRatio=np.square((firstHalfDisp/firstHalfLength)-(secondHalfDisp/secondHalfLength))
        bioPriorCost=1/(1-asymetryRatio)
        
        #append to dataframe
        rowVector=[streamLength, displacement, efficiencyRatio, asymetryRatio, bioPriorCost]
        rowAsSeries = pd.Series(rowVector, index = quantificationTable.columns)
        quantificationTable.append(rowAsSeries,ignore_index=True)
    return quantificationTable

def endpointDispersionMapping(streamlines,referenceNifti,distanceParameter):
    """endpointDispersionMapping(streamlines,referenceNifti,distanceParameter)
    For each voxel in the streamline-derived white matter mask, computes the
    average distance of streamlines' (within some specified radial distance
    of the voxel) endpoints from the average coordinate of the endpoints.  
    Simply averages the metric for each of the two endpoint clusters.

    Parameters
    ----------
    streamlines : TYPE
        Steamlines which are to be subjected to this analyis, dervied from
        tractogram.streamlines
    referenceNifti : TYPE
        A reference nifti.  Possibly not necessary; see wmc2tracts for example
        dummy mechanism.
    distanceParameter : TYPE
        DESCRIPTION.

    Returns
    -------
    dispersionMeasurement [NiFTI image]
        A nifti object with the data block containing the measurements derived
        for each voxel in the corresponding locations

    """
    # To be determined:
    # should we be useing the mean centroid (i.e. raw averaged enpoint coordinate)
    # or the actual endpoint closest to this coordinate?
    
    
    
    import dipy.tracking.utils as ut
    import dipy.tracking.streamline as streamline
    import numpy as np
    import nibabel as nib
    from scipy.spatial.distance import cdist
    from dipy.tracking.vox2track import streamline_mapping
    import itertools
    from dipy.segment.clustering import QuickBundles
    import wmaPyTools.roiTools 
    
    
    # get a streamline index dict of the whole brain tract
    streamlineMapping=streamline_mapping(streamlines, referenceNifti.affine)
    #extract the dictionary keys as coordinates
    imgSpaceTractVoxels = list(streamlineMapping.keys())
    subjectSpaceTractCoords = nib.affines.apply_affine(referenceNifti.affine, np.asarray(imgSpaceTractVoxels))  
    
    print('computing statistics for ' + str(len(streamlines)) + ' occupying ' + str(len(imgSpaceTractVoxels)) + ' voxels.')
    
    returnValues=np.zeros(len(subjectSpaceTractCoords))
    #probably a more elegant way to do this
    for iCoords in range(len(subjectSpaceTractCoords)):
        #make a sphere
        currentSphere=wmaPyTools.roiTools.createSphere(distanceParameter, subjectSpaceTractCoords[iCoords,:], referenceNifti)
        
        #get the sphere coords in image space
        currentSphereImgCoords = np.array(np.where(currentSphere.get_fdata())).T
        
        #find the roi coords which correspond to voxels within the streamline mask
        validCoords=list(set(list(tuple([tuple(e) for e in currentSphereImgCoords]))) & set(imgSpaceTractVoxels))
        
        #return flattened list of indexes
        streamIndexes=list(itertools.chain(*[streamlineMapping[iCoords] for iCoords in validCoords]))
        
        #extract those streamlines as a subset
        streamsSubset=streamlines[streamIndexes]
        
        #not actually sure how this will work with a messy bundle
        #reorient streamlines so that endpoints 1 and endpoints 2 mean something
        qb = QuickBundles(threshold=100)
        cluster = qb.cluster(streamsSubset)
        
        #there should be only one with the distance setting this high
        orientedStreams=streamline.orient_by_streamline(streamsSubset, cluster.centroids[0])
        
        
        #create blank structure for endpoints
        endpoints=np.zeros((len(orientedStreams),6))
        #get the endpoints, taken from
        #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
        for iStreamline in range(len(orientedStreams)):
            #remember, first 3 = endpoint 1, last 3 = endpoint 2    
            endpoints[iStreamline,:]= np.concatenate([orientedStreams[iStreamline][0,:], orientedStreams[iStreamline][-1,:]])
        
        Endpoints1=endpoints[:,0:3]
        Endpoints2=endpoints[:,3:7]
        
        avgEndPoint1=np.mean(Endpoints1,axis=0)
        curNearDistsFromAvg1=cdist(Endpoints1, np.reshape(avgEndPoint1, (1,3)), 'euclidean')
        endPoint1DistAvg=np.mean(curNearDistsFromAvg1)
        
        avgEndPoint2=np.mean(Endpoints2,axis=0)
        curNearDistsFromAvg2=cdist(Endpoints2, np.reshape(avgEndPoint2, (1,3)), 'euclidean')
        endPoint2DistAvg=np.mean(curNearDistsFromAvg2)
        
        returnValues[iCoords]=np.mean([endPoint2DistAvg,endPoint1DistAvg])

    
    outDataArray=np.zeros(referenceNifti.shape,dtype='float')
    for iCoords in range(len(subjectSpaceTractCoords)):
        outDataArray[imgSpaceTractVoxels[iCoords]] = returnValues[iCoords]
        
    return nib.nifti1.Nifti1Image(outDataArray, referenceNifti.affine, referenceNifti.header)

def simpleEndpointDispersion_Bootstrap(streamlines,referenceNifti=None,distanceParameter=3,bootstrapNum=1):
    """
    uses Scipy's ndimage.generic_filter to perform the endpoint dispersion analysis
    Ideally this code is cleaner and may also be faster.

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.
    referenceNifti : TYPE
        DESCRIPTION.
    distanceParameter : TYPE
        DESCRIPTION.
    bootstrapNum : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import scipy
    from dipy.tracking.vox2track import streamline_mapping
    import nibabel as nib
    import numpy as np
    import itertools
    from multiprocessing import Pool
    import os
    import time
    from functools import partial
    import tqdm
    import wmaPyTools.streamTools 
    
    #create a dummy nifti if necessary in order to get a get an affine?
    if referenceNifti==None:
        referenceNifti=wmaPyTools.streamTools.dummyNiftiForStreamlines(streamlines)
    
    streamlines=wmaPyTools.streamTools.orientAllStreamlines(streamlines)
    
    #this is probably faster than geting the density map, turning that into a mask,
    #and then getting the voxel indexes for that
    # get a streamline index dict of the whole brain tract
    streamlineMapping=streamline_mapping(streamlines, referenceNifti.affine)
    #use this to get the size
    #sys.getsizeof(streamlineMapping)
    #use this to set the coordinate ordering
    imgCoords=sorted(list(streamlineMapping.keys()))
    
    # lets create a standard mask/kernel for the search window
    #standard vector, centered at the midpoint
    dimVoxelDistVals=np.arange(0, 1+distanceParameter*2) -distanceParameter
    #if your input was non iso-metric, you would use the voxeldims vector variale to adjust 
    #the size of the relevant dimensions.  we're assuming isometric for this kernel
    voxelDims=np.ones(3)
    #create a default "footprint" for the filter, e.g. this is the array form of the kernel
    x, y, z = np.meshgrid(dimVoxelDistVals, dimVoxelDistVals, dimVoxelDistVals,indexing='ij')          
    #I didn't abs, but apparently that doesn't cause a problem?
    mask_r = x*x*voxelDims[0] + y*y*voxelDims[1] + z*z*voxelDims[2] <= distanceParameter*distanceParameter
    
    #this is what we iterate over per coord
    def streamsInImgSpaceWindow(coord, streamlineMapping, mask_r):
        #listPosition=sorted(list(streamlineMapping.keys())).index(coord)/len(streamlineMapping)
        #print(listPosition,  end = "\r")
        x2, y2, z2 = np.meshgrid(dimVoxelDistVals+coord[0], dimVoxelDistVals+coord[1], dimVoxelDistVals+coord[2],indexing='ij')  
        
        coordsToCheck=np.asarray([ x2, y2, z2])[:,mask_r].T
        
        
        #try a few different methods here
        #set base intersection
        #find the roi coords which correspond to voxels within the streamline mask
        #validCoords=list(set(list(tuple([tuple(e) for e in coordsToCheck]))) & set(list(streamlineMapping.keys())))
        #get the streamline indexes for the streamlines that fall within these coords
        #streamIndexes=list(itertools.chain(*[streamlineMapping[iCoords] for iCoords in validCoords]))
        
        #for list
        # streamIndexes=[]
        # for iCoords in coordsToCheck:
        #     if tuple(iCoords) in list(streamlineMapping.keys()):
        #         streamIndexes=streamIndexes+streamlineMapping[tuple(iCoords)]
                
        #try except
        #COMICALLY FASTER
        streamIndexes=[]
        for iCoords in coordsToCheck:
            try:
                streamIndexes=streamIndexes+streamlineMapping[tuple(iCoords)]
            except:
                pass
        streamIndexes=np.unique(streamIndexes)
        return streamIndexes
        
    #now do it many times in a parallel pool 
    #processesCount=os.cpu_count()
    #processes_pool = Pool(processesCount)
    t1_start=time.process_time()
    #streamsListList=processes_pool.map(partial(streamsInImgSpaceWindow,streamlineMapping=streamlineMapping,mask_r=mask_r), imgCoords)
    print('computing voxel-wise collections of streamlines to check for while using a ' + str(distanceParameter) + 'mm spherical window')
    streamsListList=[streamsInImgSpaceWindow(iCoords,streamlineMapping,mask_r) for iCoords in tqdm.tqdm(imgCoords, position=0, leave=True)]
    t1_stop=time.process_time()
    modifiedTime=t1_stop-t1_start
    print(str(modifiedTime) + ' seconds to find streamline groups to check')  
    
    #create an output structure
    #remember it's endpoints1 Avg, endpoints1 Var, endpoints2 Avg, endpoints2 Var for 4th dim, 
    #then 5th dim = repeats
    outArray=np.zeros([referenceNifti.shape[0],referenceNifti.shape[1],referenceNifti.shape[2],4,bootstrapNum])
    #arbitrarily set that we'll be bootstrapping across half of the input streams
   
    #iterate across repeats
    
    
    t1_start=time.process_time()
    #compute the dispersion for each subset of streamlines, and output the quantificaitons
    #outValues=processes_pool.map(computeStreamsDispersion,streamlines[subsampledStreamLists])
    outValues=[computeStreamsDispersion_bootstrapNoOrient(streamlines[iStreamLists],bootstrapProportion=.5,bootstrapIter=1,refAnatT1=referenceNifti) for iStreamLists in tqdm.tqdm(streamsListList, position=0, leave=True)] 
    #A reminder of what's coming out of this:
    #endPoint1DistAvg, endPoint2DistAvg, endPoint1DistVar, endPoint2DistVar, X repeats
    
    #places those quantifications in the correct space in the output array
    outArray[np.asarray(imgCoords)[:,0],np.asarray(imgCoords)[:,1],np.asarray(imgCoords)[:,2],:,:]=np.asarray(outValues)
    t1_stop=time.process_time()
    modifiedTime=t1_stop-t1_start
    print(str(modifiedTime) + ' seconds to compute endpoint dispersions') 
     
    #now we actually have to do the output computations
    
    
    meanOfMeans=np.mean(outArray[:,:,:,[0,2],:],axis=(3,4),where=outArray[:,:,:,[0,2],:]>0)
    varianceOfMeans=np.var(outArray[:,:,:,[0,2],:],axis=(3,4))
    meanOfVariances=np.mean(outArray[:,:,:,[1,3],:],axis=(3,4))
    varianceOfVariances=np.var(outArray[:,:,:,[1,3],:],axis=(3,4))
    
    #asym
    #(endPoint1DistAvg-endPoint2DistAvg)/(endPoint1DistAvg+endPoint2DistAvg)
    meanOfMeansAsym=np.mean(np.divide(np.subtract(outArray[:,:,:,0,:],outArray[:,:,:,2,:]),np.add(outArray[:,:,:,0,:],outArray[:,:,:,2,:])),axis=3)
    varianceOfMeansAsym=np.var(np.divide(np.subtract(outArray[:,:,:,0,:],outArray[:,:,:,2,:]),np.add(outArray[:,:,:,0,:],outArray[:,:,:,2,:])),axis=3)
    meanOfVariancesAsym=np.mean(np.divide(np.subtract(outArray[:,:,:,1,:],outArray[:,:,:,3,:]),np.add(outArray[:,:,:,1,:],outArray[:,:,:,3,:])),axis=3)
    varianceOfVariancesAsym=np.var(np.divide(np.subtract(outArray[:,:,:,1,:],outArray[:,:,:,3,:]),np.add(outArray[:,:,:,1,:],outArray[:,:,:,3,:])),axis=3)
    
    #create nifti objects for each metric
    meanOfMeansNifti=nib.nifti1.Nifti1Image(np.nan_to_num(meanOfMeans), referenceNifti.affine, referenceNifti.header)
    varianceOfMeansNifti=nib.nifti1.Nifti1Image(np.nan_to_num(varianceOfMeans), referenceNifti.affine, referenceNifti.header)
    meanOfVariancesNifti=nib.nifti1.Nifti1Image(np.nan_to_num(meanOfVariances), referenceNifti.affine, referenceNifti.header)
    varianceOfVariancesNifti=nib.nifti1.Nifti1Image(np.nan_to_num(varianceOfVariances), referenceNifti.affine, referenceNifti.header)
    
    meanOfMeansAsymNifti=nib.nifti1.Nifti1Image(np.nan_to_num(meanOfMeansAsym), referenceNifti.affine, referenceNifti.header)
    varianceOfMeansAsymNifti=nib.nifti1.Nifti1Image(np.nan_to_num(varianceOfMeansAsym), referenceNifti.affine, referenceNifti.header)
    meanOfVariancesAsymNifti=nib.nifti1.Nifti1Image(np.nan_to_num(meanOfVariancesAsym), referenceNifti.affine, referenceNifti.header)
    varianceOfVariancesAsymNifti=nib.nifti1.Nifti1Image(np.nan_to_num(varianceOfVariancesAsym), referenceNifti.affine, referenceNifti.header)
    
    #propbably should output a dict for this
    outDict={'meanOfMeansNifti':meanOfMeansNifti, 'varianceOfMeansNifti':varianceOfMeansNifti, 'meanOfVariancesNifti':meanOfVariancesNifti, 'varianceOfVariancesNifti':varianceOfVariancesNifti, 'meanOfMeansAsymNifti':meanOfMeansAsymNifti, 'varianceOfMeansAsymNifti':varianceOfMeansAsymNifti, 'meanOfVariancesAsymNifti':meanOfVariancesAsymNifti, 'varianceOfVariancesAsymNifti':varianceOfVariancesAsymNifti}
    
    return outDict

def computeStreamsDispersion_bootstrapNoOrient(streamlines,bootstrapProportion=.5,bootstrapIter=1,refAnatT1=None):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.
    bootstrapProportion : TYPE, optional
        DESCRIPTION. The default is .5.
    bootstrapIter : TYPE, optional
        DESCRIPTION. The default is 1.  As in no actual boot strap

    Returns
    -------
    endPoint1DistAvg : TYPE
        DESCRIPTION.
    endPoint2DistAvg : TYPE
        DESCRIPTION.
    endPoint1DistVar : TYPE
        DESCRIPTION.
    endPoint2DistVar : TYPE
        DESCRIPTION.


    """
    import numpy as np
    from scipy.spatial.distance import cdist
    
    #check to see if theinput is singleton
    if type(streamlines) == np.ndarray:
        #bulk it up
        streamlines=[streamlines]
        
    #we can use the non-multi version of orientTractUsingNeck because we essentially
    #know that we are selecting these streamlines by their neck, at least insofar
    #as our (spatially defined) collection of streamlines is concerned

        
    #set number of streamlines to subsample
    subSampleNum=np.floor(len(streamlines)/2).astype(int)
    
    endpoints=np.zeros((len(streamlines),6))
    #get the endpoints, taken from
    #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
    for iStreamline in range(len(streamlines)):
        #remember, first 3 = endpoint 1, last 3 = endpoint 2    
        endpoints[iStreamline,:]= np.concatenate([streamlines[iStreamline][0,:], streamlines[iStreamline][-1,:]])
    
    #select the appropriate endpoints
    Endpoints1=endpoints[:,0:3]
    Endpoints2=endpoints[:,3:7]
    
    endPoint1DistAvg=np.zeros(bootstrapIter)
    endPoint1DistVar=np.zeros(bootstrapIter)
    endPoint2DistAvg=np.zeros(bootstrapIter)
    endPoint2DistVar=np.zeros(bootstrapIter)
    for iRepeats in range(bootstrapIter):
        subsampleIndexes=[np.random.randint(0,len(streamlines),subSampleNum)]
        if not len(subsampleIndexes)==0:
        
            #compute the subset mean distance and variance for endpoint cluster 1
            avgEndPoint1=np.mean(Endpoints1[subsampleIndexes],axis=0)
            curNearDistsFromAvg1=cdist(np.atleast_2d(Endpoints1[subsampleIndexes]), np.atleast_2d(avgEndPoint1), 'euclidean')
            endPoint1DistAvg[iRepeats]=np.mean(curNearDistsFromAvg1)
            endPoint1DistVar[iRepeats]=np.var(curNearDistsFromAvg1)
            
            #compute the subset mean distance and variance for endpoint cluster 2
            avgEndPoint2=np.mean(Endpoints2[subsampleIndexes],axis=0)
            curNearDistsFromAvg2=cdist(np.atleast_2d(Endpoints2[subsampleIndexes]), np.atleast_2d(avgEndPoint2), 'euclidean')
            endPoint2DistAvg[iRepeats]=np.mean(curNearDistsFromAvg2)
            endPoint2DistVar[iRepeats]=np.var(curNearDistsFromAvg2)
            
        else:
            endPoint1DistAvg[iRepeats]=float("NaN")
            endPoint1DistVar[iRepeats]=float("NaN")
            
            endPoint2DistAvg[iRepeats]=float("NaN")
            endPoint2DistVar[iRepeats]=float("NaN")
        
    return endPoint1DistAvg, endPoint2DistAvg, endPoint1DistVar, endPoint2DistVar

def computeStreamsDispersion_bootstrap(streamlines,bootstrapProportion=.5,bootstrapIter=1,refAnatT1=None):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.
    bootstrapProportion : TYPE, optional
        DESCRIPTION. The default is .5.
    bootstrapIter : TYPE, optional
        DESCRIPTION. The default is 1.  As in no actual boot strap

    Returns
    -------
    endPoint1DistAvg : TYPE
        DESCRIPTION.
    endPoint2DistAvg : TYPE
        DESCRIPTION.
    endPoint1DistVar : TYPE
        DESCRIPTION.
    endPoint2DistVar : TYPE
        DESCRIPTION.


    """
    import numpy as np
    from scipy.spatial.distance import cdist
    import wmaPyTools.streamTools 
    
    #check to see if theinput is singleton
    if type(streamlines) == np.ndarray:
        #bulk it up
        streamlines=[streamlines]
        
    #we can use the non-multi version of orientTractUsingNeck because we essentially
    #know that we are selecting these streamlines by their neck, at least insofar
    #as our (spatially defined) collection of streamlines is concerned
    if refAnatT1==None:
        orientedStreams=wmaPyTools.streamTools.orientTractUsingNeck_Robust(streamlines,surpressReport=True)
    else:
        orientedStreams=wmaPyTools.streamTools.orientTractUsingNeck_Robust(streamlines,refAnatT1, surpressReport=True)
        
    #set number of streamlines to subsample
    subSampleNum=np.floor(len(streamlines)/2).astype(int)
    
    
    endpoints=np.zeros((len(orientedStreams),6))
    #get the endpoints, taken from
    #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
    for iStreamline in range(len(orientedStreams)):
        #remember, first 3 = endpoint 1, last 3 = endpoint 2    
        endpoints[iStreamline,:]= np.concatenate([orientedStreams[iStreamline][0,:], orientedStreams[iStreamline][-1,:]])
    
    #select the appropriate endpoints
    Endpoints1=endpoints[:,0:3]
    Endpoints2=endpoints[:,3:7]
    
    endPoint1DistAvg=np.zeros(bootstrapIter)
    endPoint1DistVar=np.zeros(bootstrapIter)
    endPoint2DistAvg=np.zeros(bootstrapIter)
    endPoint2DistVar=np.zeros(bootstrapIter)
    for iRepeats in range(bootstrapIter):
        subsampleIndexes=[np.random.randint(0,len(streamlines),subSampleNum)]
        if not len(subsampleIndexes)==0:
        
            #compute the subset mean distance and variance for endpoint cluster 1
            avgEndPoint1=np.mean(Endpoints1[subsampleIndexes],axis=0)
            curNearDistsFromAvg1=cdist(np.atleast_2d(Endpoints1[subsampleIndexes]), np.atleast_2d(avgEndPoint1), 'euclidean')
            endPoint1DistAvg[iRepeats]=np.mean(curNearDistsFromAvg1)
            endPoint1DistVar[iRepeats]=np.var(curNearDistsFromAvg1)
            
            #compute the subset mean distance and variance for endpoint cluster 2
            avgEndPoint2=np.mean(Endpoints2[subsampleIndexes],axis=0)
            curNearDistsFromAvg2=cdist(np.atleast_2d(Endpoints2[subsampleIndexes]), np.atleast_2d(avgEndPoint2), 'euclidean')
            endPoint2DistAvg[iRepeats]=np.mean(curNearDistsFromAvg2)
            endPoint2DistVar[iRepeats]=np.var(curNearDistsFromAvg2)
            
        else:
            endPoint1DistAvg[iRepeats]=float("NaN")
            endPoint1DistVar[iRepeats]=float("NaN")
            
            endPoint2DistAvg[iRepeats]=float("NaN")
            endPoint2DistVar[iRepeats]=float("NaN")
        
    return endPoint1DistAvg, endPoint2DistAvg, endPoint1DistVar, endPoint2DistVar

def endpointDispersionMapping_Bootstrap(streamlines,referenceNifti,distanceParameter,bootstrapNum):
    """endpointDispersionMapping_Bootstrap(streamlines,referenceNifti,distanceParameter,bootstrapNum)    
       For each voxel in the streamline-derived white matter mask, computes the
       average distance of streamlines' (within some specified radial distance
       of the voxel) endpoints from the average coordinate of the endpoints.  
       Simply averages the metric for each of the two endpoint clusters.  
       
       Distinct from non bootstrap version:  performs some number of iterated
       bootstrap measurments from a subset of the whole input streamline group
       in order to ascertain variability of resultant metrics.  Performs
       bootstrap operations on a 1/2 subset of the total input streamlines

       Parameters
       ----------
       streamlines : TYPE
           Steamlines which are to be subjected to this analyis, dervied from
           tractogram.streamlines
       referenceNifti : TYPE
           A reference nifti.  Possibly not necessary; see wmc2tracts for example
           dummy mechanism.
       distanceParameter : TYPE
           DESCRIPTION.

       Returns
       -------
       [returns 4 distinct niftis]
       
       meanOfMeans [NiFTI image]
           A nifti object with the data block containing the per voxel averages
           of the averages derived from the boot strap operations
           
       varianceOfMeans [NiFTI image]
           A nifti object with the data block containing the per voxel variances
           of the averages derived from the boot strap operations
       
       meanOfVariances [NiFTI image]
           A nifti object with the data block containing the per voxel averages
           of the variances derived from the boot strap operations
       
       varianceOfVariances [NiFTI image]
           A nifti object with the data block containing the per voxel variances
           of the variances derived from the boot strap operations

       """
    import dipy.tracking.utils as ut
    import dipy.tracking.streamline as streamline
    import numpy as np
    import nibabel as nib
    from scipy.spatial.distance import cdist
    from dipy.tracking.vox2track import streamline_mapping
    import itertools
    from dipy.segment.clustering import QuickBundles
    import wmaPyTools.roiTools  
    
    # get a streamline index dict of the whole brain tract
    streamlineMapping=streamline_mapping(streamlines, referenceNifti.affine)
    #extract the dictionary keys as coordinates
    imgSpaceTractVoxels = list(streamlineMapping.keys())
    subjectSpaceTractCoords = nib.affines.apply_affine(referenceNifti.affine, np.asarray(imgSpaceTractVoxels))  
    
    print('computing statistics for ' + str(len(streamlines)) + ' occupying ' + str(len(imgSpaceTractVoxels)) + ' voxels.')
    
    bootstrapStreamNum=int(len(streamlines)/2)
    meanOfMeans=np.zeros(len(subjectSpaceTractCoords))
    varianceOfMeans=np.zeros(len(subjectSpaceTractCoords))
    meanOfVariances=np.zeros(len(subjectSpaceTractCoords))
    varianceOfVariances=np.zeros(len(subjectSpaceTractCoords))
    #probably a more elegant way to do this
    for iCoords in range(len(subjectSpaceTractCoords)):
        #make a sphere
        currentSphere=wmaPyTools.roiTools.createSphere(distanceParameter, subjectSpaceTractCoords[iCoords,:], referenceNifti)
        
        #get the sphere coords in image space
        currentSphereImgCoords = np.array(np.where(currentSphere.get_fdata())).T
        
        #find the roi coords which correspond to voxels within the streamline mask
        validCoords=list(set(list(tuple([tuple(e) for e in currentSphereImgCoords]))) & set(imgSpaceTractVoxels))
        
        #return flattened list of indexes
        streamIndexes=list(itertools.chain(*[streamlineMapping[iCoords] for iCoords in validCoords]))
        
        #extract those streamlines as a subset
        streamsSubset=streamlines[streamIndexes]
        
        #not actually sure how this will work with a messy bundle
        #reorient streamlines so that endpoints 1 and endpoints 2 mean something
        #using quickbundles to get a centroid, because the actual method
        #is buried in obscurity
        qb = QuickBundles(threshold=100)
        cluster = qb.cluster(streamsSubset)
        
        #there should be only one with the distance setting this high
        orientedStreams=streamline.orient_by_streamline(streamsSubset, cluster.centroids[0])
        
        #create blank structure for endpoints
        endpoints=np.zeros((len(orientedStreams),6))
        #get the endpoints, taken from
        #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
        for iStreamline in range(len(orientedStreams)):
            #remember, first 3 = endpoint 1, last 3 = endpoint 2    
            endpoints[iStreamline,:]= np.concatenate([orientedStreams[iStreamline][0,:], orientedStreams[iStreamline][-1,:]])
        
        #select the appropriate endpoints
        Endpoints1=endpoints[:,0:3]
        Endpoints2=endpoints[:,3:7]
        
        #create holders for both the dispersion means and the dispersion variances
        dispersionMeans=[]
        dispersionVariances=[]
        for iBoostrap in range (bootstrapNum):
            
            #select a subset of half the whole streamline group, then 
            #find the intersection fo that set and the current voxel's streamlines
            currentBootstrapStreamsAll=np.random.randint(0,len(streamlines),bootstrapStreamNum)
            currentBootstrapStreamsSubSelect=np.in1d(streamIndexes,currentBootstrapStreamsAll)
            
            #compute the subset mean distance and variance for endpoint cluster 1
            avgEndPoint1=np.mean(Endpoints1[currentBootstrapStreamsSubSelect],axis=0)
            curNearDistsFromAvg1=cdist(Endpoints1[currentBootstrapStreamsSubSelect], np.reshape(avgEndPoint1, (1,3)), 'euclidean')
            endPoint1DistAvg=np.mean(curNearDistsFromAvg1)
            endPoint1DistVar=np.var(curNearDistsFromAvg1)
            
            #compute the subset mean distance and variance for endpoint cluster 2
            avgEndPoint2=np.mean(Endpoints2[currentBootstrapStreamsSubSelect],axis=0)
            curNearDistsFromAvg2=cdist(Endpoints2[currentBootstrapStreamsSubSelect], np.reshape(avgEndPoint2, (1,3)), 'euclidean')
            endPoint2DistAvg=np.mean(curNearDistsFromAvg2)
            endPoint2DistVar=np.var(curNearDistsFromAvg2)
        
            #for this bootstrap iteration, compute the average distance and the variance
            dispersionMeans.append(np.mean([endPoint2DistAvg,endPoint1DistAvg]))
            dispersionVariances.append(np.mean([endPoint1DistVar,endPoint2DistVar]))
        
        #now place them in the appropriate location in their respective
        #storage vectors
        meanOfMeans[iCoords]=np.mean(dispersionMeans)
        varianceOfMeans[iCoords]=np.var(dispersionMeans)
        meanOfVariances[iCoords]=np.mean(dispersionVariances)
        varianceOfVariances[iCoords]=np.var(dispersionVariances)
    
    #Now that the metrics have been compute for all coordinates, create
    #3d arrays to store the output for the nifti object data
    outMeanOfMeansArray=np.zeros(referenceNifti.shape,dtype='float')
    outVarianceOfMeansArray=np.zeros(referenceNifti.shape,dtype='float')
    outMeanOfVariancesArray=np.zeros(referenceNifti.shape,dtype='float')
    outVarianceOfVariancesArray=np.zeros(referenceNifti.shape,dtype='float')
    
    #iterate across each voxel coordinate
    for iCoords in range(len(subjectSpaceTractCoords)):
        #fill in the corresponding voxel's value for each metric
        outMeanOfMeansArray[imgSpaceTractVoxels[iCoords]] = meanOfMeans[iCoords]
        outVarianceOfMeansArray[imgSpaceTractVoxels[iCoords]] = varianceOfMeans[iCoords]
        outMeanOfVariancesArray[imgSpaceTractVoxels[iCoords]] = meanOfVariances[iCoords]
        outVarianceOfVariancesArray[imgSpaceTractVoxels[iCoords]] = varianceOfVariances[iCoords]
    
    #create nifti objects for each metric
    meanOfMeansNifti=nib.nifti1.Nifti1Image(outMeanOfMeansArray, referenceNifti.affine, referenceNifti.header)
    varianceOfMeansNifti=nib.nifti1.Nifti1Image(outVarianceOfMeansArray, referenceNifti.affine, referenceNifti.header)
    meanOfVariancesNifti=nib.nifti1.Nifti1Image(outMeanOfVariancesArray, referenceNifti.affine, referenceNifti.header)
    varianceOfVariancesNifti=nib.nifti1.Nifti1Image(outVarianceOfVariancesArray, referenceNifti.affine, referenceNifti.header)
    
    return meanOfMeansNifti, varianceOfMeansNifti, meanOfVariancesNifti, varianceOfVariancesNifti

def endpointDispersionAsymmetryMapping_Bootstrap(streamlines,referenceNifti,distanceParameter,bootstrapNum):
    """endpointDispersionMapping_Bootstrap(streamlines,referenceNifti,distanceParameter,bootstrapNum)    
       For each voxel in the streamline-derived white matter mask, computes the
       average distance of streamlines' (within some specified radial distance
       of the voxel) endpoints from the average coordinate of the endpoints.  
       Simply averages the metric for each of the two endpoint clusters.  
       
       Distinct from non bootstrap version:  performs some number of iterated
       bootstrap measurments from a subset of the whole input streamline group
       in order to ascertain variability of resultant metrics.  Performs
       bootstrap operations on a 1/2 subset of the total input streamlines
       
       Asymmetry version:  in addition to computing the mean of the relevant
       values for each voxel-subset of streamlines, also computes the ratio
       of the endpoint clusters.
       This is computed thusly:
           
           (A-B) / (A+B) 
           
           Where A = endpointCluster1Value and B = endpointClusterBValue
       
       In this way, the span of possible (non inf, non nan) values is 1 to -1,
       where 1 is the case in which A is substantially larger than B (e.g. B~=0),
       -1 is the case in which B is substantially larger than A (e.g. A~=0),
       and 0 is the case in which A and B are roughly equivalent. 

       Parameters
       ----------
       streamlines : TYPE
           Steamlines which are to be subjected to this analyis, dervied from
           tractogram.streamlines
       referenceNifti : TYPE
           A reference nifti.  Possibly not necessary; see wmc2tracts for example
           dummy mechanism.
       distanceParameter : TYPE
           DESCRIPTION.

       Returns
       -------
       [returns 8 distinct niftis]
       
       meanOfMeans [NiFTI image]
           A nifti object with the data block containing the per voxel averages
           of the averages derived from the boot strap operations
           
       varianceOfMeans [NiFTI image]
           A nifti object with the data block containing the per voxel variances
           of the averages derived from the boot strap operations
       
       meanOfVariances [NiFTI image]
           A nifti object with the data block containing the per voxel averages
           of the variances derived from the boot strap operations
       
       varianceOfVariances [NiFTI image]
           A nifti object with the data block containing the per voxel variances
           of the variances derived from the boot strap operations
           
       meanOfMeansAsym [NiFTI image]
           A nifti object with the data block containing the asymmetry measurment of
           per voxel averages of the averages derived from the boot strap operations
           
       varianceOfMeansAsym [NiFTI image]
           A nifti object with the data block containing the asymmetry measurment of
           the per voxel variances of the averages derived from the boot strap operations
       
       meanOfVariancesAsym [NiFTI image]
           A nifti object with the data block containing the asymmetry measurment of
           the per voxel averages of the variances derived from the boot strap operations
       
       varianceOfVariancesAsym [NiFTI image]
           A nifti object with the data block containing the asymmetry measurment of
           the per voxel variances of the variances derived from the boot strap operations

       """
    import dipy.tracking.utils as ut
    import dipy.tracking.streamline as streamline
    import numpy as np
    import nibabel as nib
    from scipy.spatial.distance import cdist
    from dipy.tracking.vox2track import streamline_mapping
    import itertools
    from dipy.segment.clustering import QuickBundles
    import wmaPyTools.roiTools  
    
    # get a streamline index dict of the whole brain tract
    streamlineMapping=streamline_mapping(streamlines, referenceNifti.affine)
    #extract the dictionary keys as coordinates
    imgSpaceTractVoxels = list(streamlineMapping.keys())
    subjectSpaceTractCoords = nib.affines.apply_affine(referenceNifti.affine, np.asarray(imgSpaceTractVoxels))
    
    print('computing statistics for ' + str(len(streamlines)) + ' occupying ' + str(len(imgSpaceTractVoxels)) + ' voxels.')
    
    bootstrapStreamNum=int(len(streamlines)/2)
    meanOfMeans=np.zeros(len(subjectSpaceTractCoords))
    varianceOfMeans=np.zeros(len(subjectSpaceTractCoords))
    meanOfVariances=np.zeros(len(subjectSpaceTractCoords))
    varianceOfVariances=np.zeros(len(subjectSpaceTractCoords))
    
    #asym
    meanOfMeansAsym=np.zeros(len(subjectSpaceTractCoords))
    varianceOfMeansAsym=np.zeros(len(subjectSpaceTractCoords))
    meanOfVariancesAsym=np.zeros(len(subjectSpaceTractCoords))
    varianceOfVariancesAsym=np.zeros(len(subjectSpaceTractCoords))
    #probably a more elegant way to do this
    for iCoords in range(len(subjectSpaceTractCoords)):
        #make a sphere
        currentSphere=wmaPyTools.roiTools.createSphere(distanceParameter, subjectSpaceTractCoords[iCoords,:], referenceNifti)
        
        #get the sphere coords in image space
        currentSphereImgCoords = np.array(np.where(currentSphere.get_fdata())).T
        
        #find the roi coords which correspond to voxels within the streamline mask
        validCoords=list(set(list(tuple([tuple(e) for e in currentSphereImgCoords]))) & set(imgSpaceTractVoxels))
        
        #return flattened list of indexes
        streamIndexes=list(itertools.chain(*[streamlineMapping[iCoords] for iCoords in validCoords]))
        
        #extract those streamlines as a subset
        streamsSubset=streamlines[streamIndexes]
        
        #not actually sure how this will work with a messy bundle
        #reorient streamlines so that endpoints 1 and endpoints 2 mean something
        #using quickbundles to get a centroid, because the actual method
        #is buried in obscurity
        qb = QuickBundles(threshold=100)
        cluster = qb.cluster(streamsSubset)
        
        #there should be only one with the distance setting this high
        orientedStreams=streamline.orient_by_streamline(streamsSubset, cluster.centroids[0])
        
        #create blank structure for endpoints
        endpoints=np.zeros((len(orientedStreams),6))
        #get the endpoints, taken from
        #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
        for iStreamline in range(len(orientedStreams)):
            #remember, first 3 = endpoint 1, last 3 = endpoint 2    
            endpoints[iStreamline,:]= np.concatenate([orientedStreams[iStreamline][0,:], orientedStreams[iStreamline][-1,:]])
        
        #select the appropriate endpoints
        Endpoints1=endpoints[:,0:3]
        Endpoints2=endpoints[:,3:7]
        
        #create holders for both the dispersion means and the dispersion variances
        dispersionMeans=[]
        dispersionVariances=[]
        #asym
        dispersionMeansAsym=[]
        dispersionVariancesAsym=[]
        for iBoostrap in range (bootstrapNum):
            
            #select a subset of half the whole streamline group, then 
            #find the intersection fo that set and the current voxel's streamlines
            currentBootstrapStreamsAll=np.random.randint(0,len(streamlines),bootstrapStreamNum)
            currentBootstrapStreamsSubSelect=np.in1d(streamIndexes,currentBootstrapStreamsAll)
            
            #compute the subset mean distance and variance for endpoint cluster 1
            avgEndPoint1=np.mean(Endpoints1[currentBootstrapStreamsSubSelect],axis=0)
            curNearDistsFromAvg1=cdist(Endpoints1[currentBootstrapStreamsSubSelect], np.reshape(avgEndPoint1, (1,3)), 'euclidean')
            endPoint1DistAvg=np.mean(curNearDistsFromAvg1)
            endPoint1DistVar=np.var(curNearDistsFromAvg1)
            
            #compute the subset mean distance and variance for endpoint cluster 2
            avgEndPoint2=np.mean(Endpoints2[currentBootstrapStreamsSubSelect],axis=0)
            curNearDistsFromAvg2=cdist(Endpoints2[currentBootstrapStreamsSubSelect], np.reshape(avgEndPoint2, (1,3)), 'euclidean')
            endPoint2DistAvg=np.mean(curNearDistsFromAvg2)
            endPoint2DistVar=np.var(curNearDistsFromAvg2)
        
            #for this bootstrap iteration, compute the average distance and the variance
            dispersionMeans.append(np.mean([endPoint2DistAvg,endPoint1DistAvg]))
            dispersionVariances.append(np.mean([endPoint1DistVar,endPoint2DistVar]))
            #asym
            dispersionMeansAsym.append((endPoint1DistAvg-endPoint2DistAvg)/(endPoint1DistAvg+endPoint2DistAvg))
            dispersionVariancesAsym.append((endPoint1DistVar-endPoint2DistVar)/(endPoint1DistVar+endPoint2DistVar))
        
        #now place them in the appropriate location in their respective
        #storage vectors
        meanOfMeans[iCoords]=np.mean(dispersionMeans)
        varianceOfMeans[iCoords]=np.var(dispersionMeans)
        meanOfVariances[iCoords]=np.mean(dispersionVariances)
        varianceOfVariances[iCoords]=np.var(dispersionVariances)
        
        #asym
        meanOfMeansAsym[iCoords]=np.mean(dispersionMeansAsym)
        varianceOfMeansAsym[iCoords]=np.var(dispersionMeansAsym)
        meanOfVariancesAsym[iCoords]=np.mean(dispersionVariancesAsym)
        varianceOfVariancesAsym[iCoords]=np.var(dispersionVariancesAsym)
    
    #Now that the metrics have been compute for all coordinates, create
    #3d arrays to store the output for the nifti object data
    outMeanOfMeansArray=np.zeros(referenceNifti.shape,dtype='float')
    outVarianceOfMeansArray=np.zeros(referenceNifti.shape,dtype='float')
    outMeanOfVariancesArray=np.zeros(referenceNifti.shape,dtype='float')
    outVarianceOfVariancesArray=np.zeros(referenceNifti.shape,dtype='float')
    
    #asym
    outMeanOfMeansAsymArray=np.zeros(referenceNifti.shape,dtype='float')
    outVarianceOfMeansAsymArray=np.zeros(referenceNifti.shape,dtype='float')
    outMeanOfVariancesAsymArray=np.zeros(referenceNifti.shape,dtype='float')
    outVarianceOfVariancesAsymArray=np.zeros(referenceNifti.shape,dtype='float')
    
    #iterate across each voxel coordinate
    for iCoords in range(len(subjectSpaceTractCoords)):
        #fill in the corresponding voxel's value for each metric
        outMeanOfMeansArray[imgSpaceTractVoxels[iCoords]] = meanOfMeans[iCoords]
        outVarianceOfMeansArray[imgSpaceTractVoxels[iCoords]] = varianceOfMeans[iCoords]
        outMeanOfVariancesArray[imgSpaceTractVoxels[iCoords]] = meanOfVariances[iCoords]
        outVarianceOfVariancesArray[imgSpaceTractVoxels[iCoords]] = varianceOfVariances[iCoords]
        
        #asym
        outMeanOfMeansAsymArray[imgSpaceTractVoxels[iCoords]] = meanOfMeansAsym[iCoords]
        outVarianceOfMeansAsymArray[imgSpaceTractVoxels[iCoords]] = varianceOfMeansAsym[iCoords]
        outMeanOfVariancesAsymArray[imgSpaceTractVoxels[iCoords]] = meanOfVariancesAsym[iCoords]
        outVarianceOfVariancesAsymArray[imgSpaceTractVoxels[iCoords]] = varianceOfVariancesAsym[iCoords]
    
    #create nifti objects for each metric
    meanOfMeansNifti=nib.nifti1.Nifti1Image(outMeanOfMeansArray, referenceNifti.affine, referenceNifti.header)
    varianceOfMeansNifti=nib.nifti1.Nifti1Image(outVarianceOfMeansArray, referenceNifti.affine, referenceNifti.header)
    meanOfVariancesNifti=nib.nifti1.Nifti1Image(outMeanOfVariancesArray, referenceNifti.affine, referenceNifti.header)
    varianceOfVariancesNifti=nib.nifti1.Nifti1Image(outVarianceOfVariancesArray, referenceNifti.affine, referenceNifti.header)
    
    meanOfMeansAsymNifti=nib.nifti1.Nifti1Image(outMeanOfMeansArray, referenceNifti.affine, referenceNifti.header)
    varianceOfMeansAsymNifti=nib.nifti1.Nifti1Image(outVarianceOfMeansArray, referenceNifti.affine, referenceNifti.header)
    meanOfVariancesAsymNifti=nib.nifti1.Nifti1Image(outMeanOfVariancesArray, referenceNifti.affine, referenceNifti.header)
    varianceOfVariancesAsymNifti=nib.nifti1.Nifti1Image(outVarianceOfVariancesArray, referenceNifti.affine, referenceNifti.header)
    
    return meanOfMeansNifti, varianceOfMeansNifti, meanOfVariancesNifti, varianceOfVariancesNifti, meanOfMeansAsymNifti, varianceOfMeansAsymNifti, meanOfVariancesAsymNifti, varianceOfVariancesAsymNifti

def cumulativeTraversalStream(streamline):
    import numpy as np
    deltas=[np.abs(streamline[iNodes,:]-streamline[iNodes-1,:]) for iNodes in  range(1,streamline.shape[0])]
    totalDelta=np.sum(deltas,axis=0)
    return totalDelta

def quantifyTractEndpoints(tractStreamlines,atlas,atlasLookupTable):
    """rquantifyTractEndpoints(tractStreamlines,atlas,atlasLookupTable)
    A function used to quantify the terminations of a tract and output a 
    table for both endpoint clusters.  Basically a complicated wrapper around
    dipy's connectivity_matrix that outputs organized tables

    Parameters
    ----------
    tractStreamlines : streamlines type
        Streamlines corresponding to the tract of interest
    atlas : Nifti, int based
        A nifti atlas that will be used to determine the endpoint connectivity
    atlasLookupTable : pandas dataframe or file loadable to pandas dataframe
        A dataframe of the atlas lookup table which includes the labels featured
        in the atlas and their identities.  These identities will be used
        to label the periphery of the radial plot.
    Returns
    -------
    Two pandas dataframes, one for each endpoint group

    """ 
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import nibabel as nib
    import dipy
    import numpy as np
    from dipy.segment.clustering import QuickBundles
    from dipy.tracking.utils import reduce_labels
    from dipy.tracking import utils
    import dipy.tracking.streamline as streamline
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    from dipy.segment.metric import MinimumAverageDirectFlipMetric
    import itertools
    import wmaPyTools.streamTools  
    
    #use dipy function to reduce labels to contiguous values
    if isinstance(atlas,str):
        atlas=nib.load(atlas)
    #NOTE astype(int) is causing all sorts of problems, BE WARNED
    [relabeledAtlasData, labelMappings]=reduce_labels(atlas.get_fdata())
    #create new nifti object
    renumberedAtlasNifti=nib.Nifti1Image(relabeledAtlasData, atlas.affine, atlas.header)
    
    #take care of tractogram and treamline issues
    if isinstance(tractStreamlines,str):
        loadedTractogram=nib.streamlines.load(tractStreamlines)
        tractStreamlines=loadedTractogram.streamlines
    elif  isinstance(tractStreamlines,nib.streamlines.tck.TckFile):
        tractStreamlines=tractStreamlines.streamlines
    
    #obtains quantifications of neck properties for this collection of streamlines
    neckQuantifications=wmaPyTools.streamTools.bundleTest(tractStreamlines)
    if neckQuantifications['mean'] <5:
        print('submitted streamlines appear to be coherent bundle via neck criterion')
        tractStreamlines=wmaPyTools.streamTools.orientTractUsingNeck(tractStreamlines)
    else:
        print('submitted streamlines DO NOT appear to be coherent bundle via neck criterion')
        tractStreamlines=wmaPyTools.streamTools.dipyOrientStreamlines(tractStreamlines)
    
    #segment tractome into connectivity matrix from parcellation
    M, grouping=utils.connectivity_matrix(tractStreamlines, atlas.affine, label_volume=renumberedAtlasNifti.get_fdata().astype(int),
                            return_mapping=True,
                            mapping_as_streamlines=False)
    #get the keys so that you can iterate through them later
    keyTargets=list(grouping.keys())
    keyTargetsArray=np.asarray(keyTargets)
    
    #work with the input lookup table
    if isinstance(atlasLookupTable,str):
        if atlasLookupTable[-4:]=='.csv':
            atlasLookupTable=pd.read_csv(atlasLookupTable)
        elif (np.logical_or(atlasLookupTable[-4:]=='.xls',atlasLookupTable[-5:]=='.xlsx')):
            atlasLookupTable=pd.read_excel(atlasLookupTable)

    # if the lookup table has more entries than the atlas has items, we'll need to do some readjusting
    # probably need to do some readjusting in any case
    if len(atlasLookupTable)>len(labelMappings):
        #infer which column contains the original identities
        #presumably, this would be the LUT column with the largest number of matching labels with the original atlas.
        matchingLabelsCount=[len(list(set(atlasLookupTable[iColumns]).intersection(set(np.unique(atlas.get_fdata()))))) for iColumns in atlasLookupTable.columns.to_list()]
        #there's an edge case here relabeled atlas == the original atlas AND the provided LUT was larger (what would the extra entries be?)
        #worry about that later
        columnBestGuess=atlasLookupTable.columns.to_list()[matchingLabelsCount.index(np.max(matchingLabelsCount))]
        #we can also take this opportunity to pick the longest average column, which is likely the optimal label name
        entryLengths=atlasLookupTable.applymap(str).applymap(len)
        labelColumnGuess=entryLengths.mean(axis=0).idxmax()
        
        #lets get the correspondances of the labels in the two atlases, we can 
        #assume that ordering has been preserved, it's just that identity has been 
        #shifted to a contiguous integer sequence
                
        #now that we have the guess, get the corresponding row entries, and reset the index.
        #This should make the index match the renumbered label values.
        #NOTE np.round results in a very different outcome than .astype(int) WHY
        LUTWorking=atlasLookupTable[atlasLookupTable[columnBestGuess].isin(np.round(labelMappings))].reset_index(drop=True)
        
        #indexMappings=np.asar[np.where(labelMappings[iLabels]==atlasLookupTable[columnBestGuess].values)[0] for iLabels in range(len(labelMappings))]
      
    
    #iterate across both sets of endpoints
    for iEndpoints in range(keyTargetsArray.shape[1]):
        uniqueLabelValues=np.unique(keyTargetsArray[:,iEndpoints])
        plotLabels=[]
        plotValues=[]
        for iUniqueLabelValues in uniqueLabelValues:
            targetKeys=list(itertools.compress(keyTargets,[keyTargetsArray[:,iEndpoints]==iUniqueLabelValues][0]))
            #could also probably just sum up the column in the matrix
            counts=[len(grouping[iKeys]) for iKeys in targetKeys]
            #append to the plotValue list
            plotValues.append(np.sum(counts))
            plotLabels.append(LUTWorking[labelColumnGuess].iloc[iUniqueLabelValues])
        
        if iEndpoints==0:
            endpoints1DF=pd.DataFrame(data ={'labelNames': plotLabels, 'endpointCounts':plotValues})
        else:
            endpoints2DF=pd.DataFrame(data ={'labelNames': plotLabels, 'endpointCounts':plotValues})
    
    return endpoints1DF, endpoints2DF

def streamlinePrototypicalityMeasure(streamlines,sumOrMean='sum'):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    measuresVec : TYPE
        DESCRIPTION.

    """
    
    import numpy as np
    import dipy.tracking.utils as ut
    from dipy.tracking.vox2track import streamline_mapping
    import wmaPyTools.streamTools
    
    dummyNifti= wmaPyTools.streamTools.dummyNiftiForStreamlines(streamlines)
    streamlineMapping=streamline_mapping(streamlines, dummyNifti.affine)
    imgSpaceTractVoxels = list(streamlineMapping.keys())
    #probably a terribly inefficient way to do this
    traversalValues=[[] for i in range(len(streamlines))]
    #get the density map
    densityMap=ut.density_map(streamlines, dummyNifti.affine, dummyNifti.shape)
    for iVoxels in imgSpaceTractVoxels:
        currentStreams=streamlineMapping[iVoxels]
        currentValue=densityMap[iVoxels]
        for iStreams in currentStreams:
            traversalValues[iStreams].append(currentValue)
    
    #sum it all up
    if sumOrMean=='sum':
        perStreamSums=[np.sum(iStreams) for iStreams in traversalValues]
    elif sumOrMean=='mean':
        perStreamSums=[np.mean(iStreams) for iStreams in traversalValues]
    #normalize it
    measuresVec=np.divide(perStreamSums,np.max(perStreamSums))
    return measuresVec

def quantifyOverlapBetweenNiftis(ROI1,ROI2):
    """
    

    Parameters
    ----------
    ROI1 : TYPE
        DESCRIPTION.
    ROI2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    
    
    import wmaPyTools.roiTools
    from dipy.core.geometry import dist_to_corner
    import numpy as np
    import scipy.spatial.distance
    
    #start by aligning the ROIs
    [ROI1,ROI2]=wmaPyTools.roiTools.alignNiftis(ROI1,ROI2)
    
    #identify their types
    #kind of involves an assumption, in that we are inferring that a 2 unique 
    #value ROI is binary, and thus 0 and 1 or some equivalent
    if len(np.unique(ROI1.get_fdata()))<=2:
           ROI1type=bool
    else:
        ROI1type=ROI1.get_fdata().dtype
        
    if len(np.unique(ROI2.get_fdata()))<=2:
           ROI2type=bool
    else:
        ROI2type=ROI2.get_fdata().dtype
    
    #if either of these are bool, you've got to force them both to boolean
    #so that you can perform dice coefficient
    if ROI1type==bool or ROI2type==bool:
        #looks like we are doing bool
        #convert both to 1D boolean vectors.
        #I'm assuming ravel ravels in a standard fashion and that this is safe.
        #also, we can use ravel given that these ROIs are presumably the same size
        #thanks to wmaPyTools.roiTools.alignNiftis
        ROI1DataVec=np.ravel(ROI1.get_fdata().astype(bool))
        ROI2DataVec=np.ravel(ROI2.get_fdata().astype(bool))
        #now compute the dice coefficient as the distance measure
        print('Computing dice coefficient')
        distanceMeasure=scipy.spatial.distance.dice(ROI1DataVec,ROI2DataVec)
    else: 
        #I'm assuming ravel ravels in a standard fashion and that this is safe.
        #also, we can use ravel given that these ROIs are presumably the same size
        #thanks to wmaPyTools.roiTools.alignNiftis
        #no need to modify datatype as before, though maybe an issue if different 
        #types, i.e. int vs float.
        ROI1DataVec=np.ravel(ROI1.get_fdata())
        ROI2DataVec=np.ravel(ROI2.get_fdata())
        #now compute the dice coefficient as the distance measure
        print('Computing cosine distance')
        distanceMeasure=scipy.spatial.distance.cosine(ROI1DataVec,ROI2DataVec)
    
    #return the quantification
    return distanceMeasure
