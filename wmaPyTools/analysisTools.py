# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:22 2021

@author: Daniel
"""
def streamGeomQuantifications(tractogram):
    """
    This function quantifies a number of streamline-based quantities
    in the same fashion as wma_tools's  ConnectomeTestQ
    
    see https://github.com/DanNBullock/wma_tools#connectometestq
    for more details.

    Parameters
    ----------
    tractogram : TYPE
        An input stateful tractogram

    Returns
    -------
    quantificationTable : TYPE
        A pandas table documenting the streamline based quantificaton. 

    """
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
        currentSphereImgCoords = np.array(np.where(currentSphere.get_data())).T
        
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
    import wmaPyTools.streamlineTools 
    
    #create a dummy nifti if necessary in order to get a get an affine?
    if referenceNifti==None:
        referenceNifti=wmaPyTools.streamlineTools.dummyNiftiForStreamlines(streamlines)
    
    streamlines=wmaPyTools.streamlineTools.orientAllStreamlines(streamlines)
    
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
    import wmaPyTools.streamlineTools 
    
    #check to see if theinput is singleton
    if type(streamlines) == np.ndarray:
        #bulk it up
        streamlines=[streamlines]
        
    #we can use the non-multi version of orientTractUsingNeck because we essentially
    #know that we are selecting these streamlines by their neck, at least insofar
    #as our (spatially defined) collection of streamlines is concerned
    if refAnatT1==None:
        orientedStreams=wmaPyTools.streamlineTools._Robust(streamlines,surpressReport=True)
    else:
        orientedStreams=wmaPyTools.streamlineTools.orientTractUsingNeck_Robust(streamlines,refAnatT1, surpressReport=True)
        
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
        currentSphereImgCoords = np.array(np.where(currentSphere.get_data())).T
        
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
        currentSphereImgCoords = np.array(np.where(currentSphere.get_data())).T
        
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
    import wmaPyTools.streamlineTools  
    
    #use dipy function to reduce labels to contiguous values
    if isinstance(atlas,str):
        atlas=nib.load(atlas)
    #NOTE astype(int) is causing all sorts of problems, BE WARNED
    [relabeledAtlasData, labelMappings]=reduce_labels(atlas.get_data())
    #create new nifti object
    renumberedAtlasNifti=nib.Nifti1Image(relabeledAtlasData, atlas.affine, atlas.header)
    
    #take care of tractogram and treamline issues
    if isinstance(tractStreamlines,str):
        loadedTractogram=nib.streamlines.load(tractStreamlines)
        tractStreamlines=loadedTractogram.streamlines
    elif  isinstance(tractStreamlines,nib.streamlines.tck.TckFile):
        tractStreamlines=tractStreamlines.streamlines
    
    #obtains quantifications of neck properties for this collection of streamlines
    neckQuantifications=wmaPyTools.streamlineTools.bundleTest(tractStreamlines)
    if neckQuantifications['mean'] <5:
        print('submitted streamlines appear to be coherent bundle via neck criterion')
        #tractStreamlines=wmaPyTools.streamlineTools.orientTractUsingNeck(tractStreamlines)
        tractStreamlines=wmaPyTools.streamlineTools.orientAllStreamlines(tractStreamlines)
    else:
        print('submitted streamlines DO NOT appear to be coherent bundle via neck criterion')
        #tractStreamlines=wmaPyTools.streamlineTools.dipyOrientStreamlines(tractStreamlines)
        #the above line appears to be unstable and can lead to crashes, perhaps
        #when the tract contains an extremely small number of streamlines?
        #in any case, we can use a generalized version for this
        #this kind of makes the above if statement/check redundant, but 
        #hopefully in the future we might have a more robust orientation method
        tractStreamlines=wmaPyTools.streamlineTools.orientAllStreamlines(tractStreamlines)
    
    #segment tractome into connectivity matrix from parcellation
    M, grouping=utils.connectivity_matrix(tractStreamlines, atlas.affine, label_volume=renumberedAtlasNifti.get_data().astype(int),
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
        matchingLabelsCount=[len(list(set(atlasLookupTable[iColumns]).intersection(set(np.unique(atlas.get_data()))))) for iColumns in atlasLookupTable.columns.to_list()]
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
    Generate a per-streamline measure of how 'typical' each streamline is for this
    collection of input streamlines, as indicated by the relative density of
    streamlines in the traversal of those streamlines.
    
    Parameters
    ----------
    streamlines : TYPE
        Theoretically, any collection of streamlines.  However, this computation
    probably only makes sense for coherent "tracts"

    sumOrMean : string
        Either "sum" or "mean".  Indicates what the final aggregate quantifiaction
    will be.  Presumably, "mean" takes care of length-related confounds better.

    Returns
    -------
    measuresVec : TYPE
        A vector containing a normalized measure of the density values traversed
    by each straemline 

    """
    
    import numpy as np
    import dipy.tracking.utils as ut
    from dipy.tracking.vox2track import streamline_mapping
    import wmaPyTools.streamlineTools
    
    #create a dummy referencec nifti for the input straemlines
    dummyNifti= wmaPyTools.streamlineTools.dummyNiftiForStreamlines(streamlines)
    #compute the voxelwise mapping for the streamlines
    streamlineMapping=streamline_mapping(streamlines, dummyNifti.affine)
    #find the voxels occupied by streamlines and get a list of these
    imgSpaceTractVoxels = list(streamlineMapping.keys())
    #probably a terribly inefficient way to do this
    #initalize a blank list of lists
    traversalValues=[[] for i in range(len(streamlines))]
    #get the density map
    densityMap=ut.density_map(streamlines, dummyNifti.affine, dummyNifti.shape)
    #iterate across the voxels with streamlines in them
    for iVoxels in imgSpaceTractVoxels:
        #get the indexes of the streams that traverse this voxel
        currentStreams=streamlineMapping[iVoxels]
        #get the density for this voxel
        currentValue=densityMap[iVoxels]
        #iterate across the streamline indexes for this voxel, and add
        #this voxels density value to the list of lists 
        #(because we're just going to sum and mean, the ordering doesn't matter)
        for iStreams in currentStreams:
            #append the current value to the list of lists for the relevant
            #streamlines
            traversalValues[iStreams].append(currentValue)
    
    #sum it all up
    if sumOrMean=='sum':
        perStreamSums=[np.sum(iStreams) for iStreams in traversalValues]
    elif sumOrMean=='mean':
        perStreamSums=[np.mean(iStreams) for iStreams in traversalValues]
    #normalize it, divide by the highest number for this collection of streamlines
    measuresVec=np.divide(perStreamSums,np.max(perStreamSums))
    return measuresVec

def quantifyOverlapBetweenNiftis(ROI1,ROI2):
    """
    Computes either the dice coefficient or cosine distance for two input niftis
    based on whether the inputs are bool or float.  Pair with DIPY's 
    ut.density_map to perform this analysis for tractograms.

    Parameters
    ----------
    ROI1 : TYPE
        DESCRIPTION.
    ROI2 : TYPE
        DESCRIPTION.

    Returns
    -------
    distanceMeasure:  float
        The single value quantification associated with the overlap measure

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
    if len(np.unique(ROI1.get_data()))<=2:
           ROI1type=bool
    else:
        ROI1type=ROI1.get_data().dtype
        
    if len(np.unique(ROI2.get_data()))<=2:
           ROI2type=bool
    else:
        ROI2type=ROI2.get_data().dtype
    
    #if either of these are bool, you've got to force them both to boolean
    #so that you can perform dice coefficient
    if ROI1type==bool or ROI2type==bool:
        #looks like we are doing bool
        #convert both to 1D boolean vectors.
        #I'm assuming ravel ravels in a standard fashion and that this is safe.
        #also, we can use ravel given that these ROIs are presumably the same size
        #thanks to wmaPyTools.roiTools.alignNiftis
        ROI1DataVec=np.ravel(ROI1.get_data().astype(bool))
        ROI2DataVec=np.ravel(ROI2.get_data().astype(bool))
        #now compute the dice coefficient as the distance measure
        print('Computing dice coefficient')
        distanceMeasure=scipy.spatial.distance.dice(ROI1DataVec,ROI2DataVec)
    else: 
        #I'm assuming ravel ravels in a standard fashion and that this is safe.
        #also, we can use ravel given that these ROIs are presumably the same size
        #thanks to wmaPyTools.roiTools.alignNiftis
        #no need to modify datatype as before, though maybe an issue if different 
        #types, i.e. int vs float.
        ROI1DataVec=np.ravel(ROI1.get_data())
        ROI2DataVec=np.ravel(ROI2.get_data())
        #now compute the dice coefficient as the distance measure
        print('Computing cosine distance')
        distanceMeasure=scipy.spatial.distance.cosine(ROI1DataVec,ROI2DataVec)
    
    #return the quantification
    return distanceMeasure

def quantifyTractsOverlap(streamlines1, streamlines2):
    """
    Computes the overlap between two collections of streamlines, presumably
    representing distinct tracts.

    Parameters
    ----------
    streamlines1 : TYPE
        Collection of streamlines presumably representing a tract of interest
    streamlines2 : TYPE
        Collection of streamlines presumably representing a tract of interest

    Returns
    -------
    overlapMeasure : Float
        The single value quantification associated with the overlap measure

    """
    
    import dipy.tracking.utils as ut
    from dipy.tracking.vox2track import streamline_mapping
    import wmaPyTools.streamlineTools
    
    #create a dummy referencec nifti for the input streamlines
    dummyNifti1= wmaPyTools.streamlineTools.dummyNiftiForStreamlines(streamlines1)
    #use that dummy nifti as an input for the density computation
    densityMap1=ut.density_map(streamlines1, dummyNifti1.affine, dummyNifti1.shape)
    
    dummyNifti2= wmaPyTools.streamlineTools.dummyNiftiForStreamlines(streamlines2)
    densityMap2=ut.density_map(streamlines2, dummyNifti2.affine, dummyNifti2.shape)
    
    overlapMeasure=quantifyOverlapBetweenNiftis(densityMap1,densityMap2)
    
    return overlapMeasure

def multiTractOverlapAnalysis(streamlinesList, namesList=None):
    """
    Comprehensively and iteratively asses the overlap between 
    collections of streammlines.  Will load tracts if paths provided.  Recomend
    efficient indexing practices if passing streamlines.

    Parameters
    ----------
    streamlinesList : List
        A list of either paths to tracts of interest or a list of streamline
    collections thesmselves presumably representing tracts
    namesList : TYPE, optional
        A list of strings indicating the name of the tracts. The default is None.
    Used in the creation of the output table.  The ordering in the names 
    is presumed to match the ordering of the input structures.

    Returns
    -------
    overlapTable : Pandas table
        A N x N pandas table, where N is the number of input tracts/paths.
    Each I,J entry indicates the overlap measure for those tracts/structures.

    """
    import pandas as pd
    import numpy as np
    import nibabel as nib
    
    #use meshgrid to create indexing array
    x, y = np.meshgrid(range(len(streamlinesList)), range(len(streamlinesList))  ,indexing='i')
    #create a list for iteration
    pairingList=[[x[iPairings],y[iPairings]] for iPairings in range(len(x)) ]
    
    #create blank outut array
    overlapArray=np.zeros ([len(streamlinesList),len(streamlinesList)])
    
    if namesList==None:
        namesList=['Tract_'+str(iTract) for iTract in range (streamlinesList)]
    
    #iterate across the structure pairings
    for iPairs in pairingList:
        
        #get objects from input list
        currStreams1=streamlinesList[iPairs[0]]
        currStreams2=streamlinesList[iPairs[1]]
    
        #load them and extract streamlines if input
        if isinstance(currStreams1,str):
            sftHolder1=nib.streamlines.load(currStreams1)
            currStreams1=sftHolder1.streamlines
        
        if isinstance(currStreams2,str):
            sftHolder2=nib.streamlines.load(currStreams2)
            currStreams2=sftHolder2.streamlines
        
        #perform overlap analysis
        currentOverlap=quantifyTractsOverlap(currStreams1, currStreams2)
        #place value into the array
        overlapArray[iPairs[0],iPairs[1]]=currentOverlap
        
    #make a table out of it
    overlapTable=pd.DataFrame(data=overlapArray, columns=namesList)      
    
    return overlapTable

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
    import wmaPyTools.segmentationTools
    from dipy.tracking.vox2track import streamline_mapping
    import itertools
    import pandas as pd
    import tqdm
    
    #napkin math to predict size of output array:
    #16 bit * 200 labels * 300000 voxels =~ 120 MB, not too bad

    
    #NOTE astype(int) is causing all sorts of problems, BE WARNED
    #using round as a workaround
    [relabeledAtlasData, labelMappings]=utils.reduce_labels(np.round(atlasNifti.get_data()).astype(int))
    
   
    #if an input mask is actually provided
    if not mask==None:
        streamsInMaskBool=wmaPyTools.segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(streamlines, mask, True, 'either_end')
        #we don't care about the label identity of *endpoints* in these img-space
        #voxels
        #WithinMaskVoxels=np.nonzero(mask)
        #may need to manipulate this to make it a list of lists
    else:
        #I guess we're doing it for all of them!
        streamsInMaskBool=np.ones(len(streamlines),dtype=bool)
        #there are no voxels that we aren't considering in this case
        #WithinMaskVoxels=[]
    
    #perform dipy connectivity analysis.  Don't need to downsample to endpoints
    #yet because dipy only computes on endpoints anyways
    print('computing voxel-label profiles for ' + str(np.sum(streamsInMaskBool)) + ' streamlines')
    
    M, grouping=utils.connectivity_matrix(streamlines[streamsInMaskBool], atlasNifti.affine, label_volume=relabeledAtlasData,
                            return_mapping=True,
                            mapping_as_streamlines=False)
    
    #get just the endpoints
    streamEndpoints=wmaPyTools.streamlineTools.downsampleToEndpoints(streamlines[streamsInMaskBool])
    #perform the mapping of these endpoints
    #maybe we need the flipped version of this
    streamlineEndpointMapping=streamline_mapping(streamEndpoints, atlasNifti.affine)
    #extract the dictionary keys as coordinates
    
    #I guess we can go ahead and create this output structure
    voxelAtlasConnectivityArray=np.zeros([len(list(streamlineEndpointMapping.keys())),len(np.unique(relabeledAtlasData))]).astype(np.uintc)
    
    
    # for iGroupings in list(grouping):
    #     currentStreams=grouping[iGroupings]
    #     if not 
    #     voxelAtlasConnectivityArray
    allEndpointMappings=list(streamlineEndpointMapping.keys())
    allGroups=list(grouping.keys())
    for iEndpointVoxels in  tqdm.tqdm(range(len(allEndpointMappings))):
        
        currentVoxLabelValue=relabeledAtlasData[allEndpointMappings[iEndpointVoxels]]
        currentVoxStreams=streamlineEndpointMapping[allEndpointMappings[iEndpointVoxels]]
        
        groupKeysWithLabelBool=np.asarray([ currentVoxLabelValue in currentGrouping for currentGrouping in allGroups])
        
        currentGroupings=list(itertools.compress(allGroups,groupKeysWithLabelBool))
        
        for iGroupings in range(len(currentGroupings)):
            #if the first group label is the current voxel value, go with the
            #other label value.  Essentially, ignores ordering and deals with same
            #label mappings adequately
            if currentGroupings[iGroupings][0]==currentVoxLabelValue:
                voxelAtlasConnectivityArray[iEndpointVoxels,currentGroupings[iGroupings][1]]=len(np.intersect1d(grouping[currentGroupings[iGroupings]],currentVoxStreams))
            else:
                voxelAtlasConnectivityArray[iEndpointVoxels,currentGroupings[iGroupings][0]]=len(np.intersect1d(grouping[currentGroupings[iGroupings]],currentVoxStreams))
    
    #set up the pandas output
    
    columnLabels=['Label_' + str(iLabel) for iLabel in range(M.shape[0])]
    rowLabels=allEndpointMappings
    voxelAtlasConnectivityTable=pd.DataFrame(data=voxelAtlasConnectivityArray,index=rowLabels,columns=columnLabels)
    
    return voxelAtlasConnectivityTable

def voxelAtlasDistanceMatrix(voxelAtlasConnectivityTable):
    
    import pandas as pd
    import numpy as np
    import os
    from scipy.spatial.distance import cdist
    import tqdm
    
    #get the indexes, which also serve as the voxel labels
    voxelIndexes=voxelAtlasConnectivityTable.index
    #determine if the output has been normalized
    #if you did the tracking using trackStreamsInMask then each voxel should
    #have the same number of streamlines
    # if you didn't, each voxel could have distinct numbers of streamlines
    # this controls for this
    # could probably be tripped up if the first voxel only has one streamline in it
    firstRowSum=np.sum(voxelAtlasConnectivityTable.iloc[0])
    if not firstRowSum==1 :
        normalizedVoxelAtlasConnectivityTable=voxelAtlasConnectivityTable.apply(lambda x: np.divide(x,np.sum(x)),axis=1)
    else:
        normalizedVoxelAtlasConnectivityTable=voxelAtlasConnectivityTable
        
    # do some math to determine if the output size on this is sensible
    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    #expected memory usage
    #THERE'S A PROBABLY A WAY TO DO THIS WITH A SPARSE MATRIX
    #the above method returns MB it looks like
    expectedMemUsage=np.divide(np.multiply(len(voxelIndexes)^2,32), 1024^2)
    #use 1.3 for a bit of extra buffer
    if expectedMemUsage*1.3>=free_memory:
        raise Exception('Expected memory usage for float 32 square array of size ' + str(len(voxelIndexes)) + ' exceeds available RAM of ' + str(free_memory) + 'MB')
    
    #otherwise go ahead
    #create output array
    cosineDistanceMatrix=np.zeros([len(voxelIndexes),len(voxelIndexes)])
    #may actually be pretty fast/efficient using cdist
    #it's not, it's extremely slow
    #maybe we can parallelize it
    #at the very least we can only compute this on the upper or lower diagonal.
    for iVoxelIndexesRow in tqdm.tqdm(range(len(voxelIndexes))):
        for iVoxelIndexesCol in range(len(voxelIndexes)):
            cosineDistanceMatrix[iVoxelIndexesRow,iVoxelIndexesCol]=cdist(np.atleast_2d(normalizedVoxelAtlasConnectivityTable.to_numpy()[iVoxelIndexesRow]),np.atleast_2d(normalizedVoxelAtlasConnectivityTable.to_numpy()[iVoxelIndexesCol]),metric='cosine')[0][0]
            
    #I guess we're done
    return voxelIndexes, cosineDistanceMatrix
    