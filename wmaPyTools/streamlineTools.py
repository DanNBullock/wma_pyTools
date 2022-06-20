# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:04:06 2021

@author: Daniel
"""

def combineTracts(tractsORstreamlines):
    #import numpy as np
    import nibabel as nib
    #from dipy.tracking.streamline import Streamlines
    #import wmaPyTools.streamlineTools
    import os
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.tracking.streamline import Streamlines
    import warnings 
    import numpy as np

    #create blank holder for tracts / streamlines
    tractHolder=[[] for iTracts in tractsORstreamlines ]
    #iterate across inputs
    
    for iterator,iTracts in enumerate(tractsORstreamlines):
        if isinstance(iTracts, str):
            #I guess we're just throwing out the stateful tractogram info here?
            tempHold=nib.streamlines.load(iTracts)
            tractHolder[iterator]=tempHold.streamlines
        elif isinstance( iTracts, nib.streamlines.tck.TckFile):
            #I guess we're just throwing out the stateful tractogram info here?
            tractHolder[iterator]=iTracts.streamlines
        #i guess it can come in a different flavor if its from dipy?
        #yeay standardization
        elif isinstance( iTracts, StatefulTractogram):
            #I guess we're just throwing out the stateful tractogram info here?
            tractHolder[iterator]=iTracts.streamlines
        elif isinstance( iTracts, nib.streamlines.array_sequence.ArraySequence):  
        #probably a bit brittle and inelegant but, we'll do it this ugly way
            tractHolder[iterator]=iTracts
    

    #get something to count the number of streamlines
    streamsCounts=[[] for iTracts in tractHolder]
    #create blank holder
    completeTractsHolder=[]
    #now we have a collection of streamline inputs, lets merge them using .extend
    for iterator,iTract in enumerate(tractHolder):
        
        #do a clean check for inf, because these cause all kinds of problems
        cleanStreams=[x for x in iTract if not np.any(np.isinf(x)) ]
        infFlags=[np.any(np.isinf(x)) for x in iTract]
        if np.any(infFlags):
            warnings.warn('Streamlines ' + str (list(np.where(infFlags)[0]))+ ' removed for containing inf' )
        
        #get the length of this
        streamsCounts[iterator]=len(cleanStreams)
        if iterator==0:
            completeTractsHolder=iTract
        else:
            completeTractsHolder.extend(iTract)
    
    outStatefulTractogram=stubbornSaveTractogram(Streamlines(completeTractsHolder),'tractogramToDelete.tck')
    print (str(len(tractsORstreamlines)) + ' input tract-like objects merged into a single, ' + str(len(outStatefulTractogram.streamlines)) + ' streamline long stateful tractogram')
   
    os.remove("tractogramToDelete.tck") 
    return outStatefulTractogram, streamsCounts


def dummyNiftiForStreamlines(streamlines):
    import numpy as np
    import nibabel as nib
    import dipy.tracking.utils as ut
    
    #dipy is stubborn and wants a reference nifti for some reason
    #fineI'llDoItMyself.jpg
    tractBounds=np.asarray([np.min(streamlines._data,axis=0),np.max(streamlines._data,axis=0)])
    #tractBounds2=np.asarray([np.min(streamlines.data,axis=0),np.max(streamlines.data,axis=0)])
    roundedTractBounds=np.asarray([np.floor(tractBounds[0,:]),np.ceil(tractBounds[1,:])])
    constructedAffine=np.eye(4)
    constructedAffine[0:3,3]=tractBounds[0,:]

    lin_T, offset =ut._mapping_to_voxel(constructedAffine)
    inds = ut._to_voxel_coordinates(streamlines._data, lin_T, offset)
        
    #testBounds=np.asarray([np.min(inds,axis=0),np.max(inds,axis=0)])
        
    #now create a dummy nifit, because that's what dipy demands
    dataShape=(roundedTractBounds[1,:]-roundedTractBounds[0,:]).astype(int)
    #adding a +1 pad because it yells otherwise?
    #should I pad here?
    dummyData=np.zeros(dataShape+1)
    dummyNifti= nib.nifti1.Nifti1Image(dummyData, constructedAffine)
    return dummyNifti

def findTractNeckNode(streamlines, refAnatT1=None):
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

    from scipy.spatial.distance import cdist
    import numpy as np
    import dipy.tracking.utils as ut
    from scipy.ndimage import gaussian_filter
    import nibabel as nib
    import itertools
    
    #TODO 
    #fix this, its too slow
    
    #lets presuppose that the densest point in the density mask ocurrs within
    #the neck.  Thus we need a
    if refAnatT1==None:
        dummyNifti=dummyNiftiForStreamlines(streamlines)
        tractMask=ut.density_map(streamlines, dummyNifti.affine, dummyNifti.shape)
    else:
        tractMask=ut.density_map(streamlines, refAnatT1.affine, refAnatT1.shape)
        
    #BEFORE WE SMOOTH
    #run a sanity check to ensure that there's some sort of overlap between the streamlines
    #if the density is NOT only 1
    if not np.array_equal(np.unique(tractMask),np.array([0,1])):    

        #we smooth it just in case there are weird local maxima
        #that sigma may need to be worked on
        #The move to float thouls fix things
        smoothedDensity=tractMask.astype(float)
        #for iSmooths in range(1):
        smoothedDensity=gaussian_filter(np.square(smoothedDensity),3)
            
        #now find the max point
        maxDensityLoc=np.asarray(np.where(smoothedDensity==np.max(smoothedDensity)))
        #pick the first one arbitrarily in case there are multiple
        maxDensityImgCoord=maxDensityLoc[:,0]
        #get the coordinate in subject space
        if refAnatT1==None:
            subjCoord = nib.affines.apply_affine(dummyNifti.affine,maxDensityImgCoord)
        else:
            subjCoord = nib.affines.apply_affine(refAnatT1.affine,maxDensityImgCoord)
    else:
        #how else should we approximate this
        #just take the average of the inner half of the streamlines, I guess
        coordsToAverage=[[] for iStreams in streamlines]
        for iterator,iStreams in enumerate(streamlines):
            #yes it will be slightly biased due to odd numbers, but whatever,
            #no need to overengineer this at the moment
            #TODO way to overengineer: take the weighted average
            #with the weight being the absolute value of the distance from
            #the middle node index
            #these fibers should be culled anyways
            curLength=len(iStreams)
            curHalfLength=np.round(curLength*.5).astype(int)
            curQuarter=np.round(curHalfLength*.5).astype(int)
            #get the middle half
            coordsToAverage[iterator]=iStreams[curQuarter:curQuarter+curHalfLength,:]
        #average them, I guess
        subjCoord=np.mean(np.vstack(np.asarray(list(itertools.chain(*coordsToAverage)))),axis=0)
                

    #iterate across streamlines
    neckNodeIndexVecOut=np.zeros(len(streamlines),dtype=int)
    for iterator,iStreamline in enumerate(streamlines):
        #distances for all nodes 
        curNodesDist = cdist(iStreamline, np.atleast_2d(subjCoord), 'euclidean')
        #presumably the nodes most directly tangent to the highest density point would be the neck?
        neckNodeIndexVecOut[iterator]=np.argmin(curNodesDist)
        
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

def smoothStreamlines(tractogram):
    #smoothStreamlines(tractogram):
    #
    #Smooths streamlines using spline method.  Replaces tractogram's streamlines
    #with the smoothed versions

    #
    # -tractogram: an input stateful tractogram 
    #
    #
    # distanceThresh the threshold between neck centroids that is accepted for merging
    import dipy
    import nibabel as nib
    from dipy.tracking.streamlinespeed import set_number_of_points
    #extract the streamlines, but not really because this is just linking
    #inputStreamlines=tractogram.streamlines
    #get the count before anything is done
    initialLength=len(tractogram.streamlines)
    for iStreams in range(initialLength):
        #use the spline method to get the smoothed streamline
        tractogram.streamlines[iStreams]=set_number_of_points(dipy.tracking.metrics.spline(tractogram.streamlines[iStreams]),len(tractogram.streamlines[iStreams]))
        
    return tractogram

# def cullViaClusters(clusters,tractogram,streamThresh):
#     #cullViaClusters(clusters,tractogram,streamThresh)
#     #
#     #This function culls streamlines from a tractogram
#     #based on the number of streamlines in their clusters
#     #
#     # INPUTS
#     #
#     # clusters: the output cluster object from quickbundles
#     #
#     # tractogram: a tractogram associated with the input clusters object
#     #
#     #streamThresh:  the minimum number of streamlines in a cluster bundle
#     #               needed to survive the culling process
#     #
#     # OUTPUTS
#     #
#     # tractogram: the cleaned tractogram
#     #
#     # culledTractogram: a tractogram containing those streamlines which have
#     # been culled.
#     #
#     # begin code    
#     import numpy as np
#     import copy
#     #apparently this can cause some issues on linux machines with dtype u21?
#     clustersSurviveThresh=np.greater(np.asarray(list(map(len, clusters))),streamThresh)
#     survivingStreams=[]
#     for iclusters in clusters[clustersSurviveThresh]:
#         survivingStreams=survivingStreams + iclusters.indices
#     culledStreamIndicies=list(set(list(range(1,len(tractogram.streamlines))))-set(survivingStreams))
#     culledTractogram=copy.deepcopy(tractogram)
#     culledTractogram.streamlines=culledTractogram.streamlines[culledStreamIndicies]
#     #cull those streamlines
#     #don't know what to do about those warnings
#     tractogram.streamlines=tractogram.streamlines[survivingStreams]
#     return tractogram, culledTractogram


# def qbCullStreams(tractogram,qbThresh,streamThresh):
#     #qbCullStreams(tractogram,qbThresh,streamThresh)
#     #
#     #this function uses dipy quickbundles to filter out streamlines which exhibt
#     #unusual/extremely uncommon trajectories using a interstreamline distance
#     #measure
#     #
#     # INPUTS
#     #
#     # tractogram: an input tractogram to be cleaned
#     #
#     # qbThresh: the distance parameter to be used for the quickbundles algorithm
#     #
#     #streamThresh:  the minimum number of streamlines in a cluster bundle
#     #               needed to survive the culling process
#     # OUTPUTS
#     #
#     # tractogram: the cleaned tractogram
#     #
#     # culledTractogram: a tractogram containing those streamlines which have
#     # been culled.
#     # 
#     # Begin code
#     from dipy.segment.clustering import QuickBundles
#     #get the number of input streamlines
#     inputStreamNumber=len(tractogram.streamlines)
#     #good default value for quick clustering
#     #qbThresh=15
#     #perform quickBundles
#     qb = QuickBundles(threshold=qbThresh)
#     clusters = qb.cluster(tractogram.streamlines)
#     #perform cull
#     [outTractogram,culledTractogram]=cullViaClusters(clusters,tractogram,streamThresh)
#     #report cull count
#     numberCulled=inputStreamNumber-len(outTractogram.streamlines)
#     print(str(numberCulled) + ' streamlines culled')
#     return outTractogram, culledTractogram

def dipyOrientStreamlines(streamlines):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.metric import Feature
    from dipy.tracking.streamline import length
    from dipy.segment.metric import SumPointwiseEuclideanMetric
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    import numpy as np
    import dipy.tracking.streamline as streamline
    import wmaPyTools.analysisTools
   
    
    # feature = ResampleFeature(nb_points=60)
    # metric = AveragePointwiseEuclideanMetric(feature)

    # qb = QuickBundles(threshold=2,metric=metric, max_nb_clusters = len(streamlines)/100)
    # cluster = qb.cluster(streamlines)
    # print(str(len(cluster)) + ' clusters generated for input with ' + str(len(streamlines)) + ' streamlines')
            
    cluster=quickbundlesClusters(streamlines, thresholds=[30,20,10,5],nb_pts=50,verbose=True)
    
    #create the dummy nifti here to save time
    dummyNifti=wmaPyTools.roiTools.dummyNiftiForStreamlines(cluster.refdata)
    
    for iBundles in cluster:
        [orientedStreams, clusterIndexes]=orientDipyCluster(iBundles,dummyNifti)
        
        #ok, now this is super ugly/iffy
        for streamsIterator,iIndex in enumerate(clusterIndexes):
            streamlines[iIndex]=orientedStreams[streamsIterator]
               
        # streamIndexes=list(cluster[iBundles].indices)
        # curCentroid=cluster[iBundles].centroid
        # #but you have to make sure that this is oriented correctly
        # centroidNeck=curCentroid[int(np.round(len(curCentroid)/2))-3:int(np.round(len(curCentroid)/2))+3,:]
        # deltas=wmaPyTools.analysisTools.cumulativeTraversalStream(centroidNeck)
        
        # maxTraversalDim=np.where(np.max(deltas)==deltas)[0][0]
        # #get the current endpoints
        # endpoint1=centroidNeck[0,:]
        # endpoint2=centroidNeck[-1,:]
    
        # #if the coordinate of endpoint1 in the max traversal dimension
        # #is less than the coordinate of endpoint1 in the max traversal dimension
        # #flip it
        # if endpoint1[maxTraversalDim]<endpoint2[maxTraversalDim]:
        #     curCentroid= curCentroid[::-1]

        # orientedStreams=streamline.orient_by_streamline(streamlines[streamIndexes], curCentroid)
        # for iStreams in range(len(streamIndexes)):
        #     if not np.all(streamlines[streamIndexes[iStreams]][0,:]==orientedStreams[iStreams][0,:]):
        #         streamlines[streamIndexes[iStreams]]= streamlines[streamIndexes[iStreams]][::-1]

    return streamlines

def orientDipyCluster(cluster,refNifti=None):
    import numpy as np
    import wmaPyTools.analysisTools
    from scipy.spatial.distance import cdist
    
    import dipy.tracking.streamline as streamline
    
    streamIndexes=list(cluster.indices)
    clusterStreams=cluster.refdata[streamIndexes]
    curCentroid=cluster.centroid
    #but you have to make sure that this is oriented correctly
    
    #find the neck for this tract
    neckNodes=findTractNeckNode(clusterStreams,refAnatT1=refNifti)
    #get the neck coords for each
    neckCoords=[iStream[neckNodes[iterator],:] for iterator,iStream in enumerate(clusterStreams)]
    #mean neck coord
    meanNeckCord=np.mean(np.asarray(neckCoords),axis=0)
    #find the cetroid node closest to this
    nodeDistances=cdist(curCentroid,np.atleast_2d(meanNeckCord))
    
    centroidNeckNode=np.argmin(nodeDistances)
    #get safe neck nodes to index
    safeNeckNodes=list(set(list(range(len(curCentroid)))).intersection(set(range(centroidNeckNode-3,centroidNeckNode+3))))
    centroidNeck=curCentroid[safeNeckNodes,:]
    deltas=wmaPyTools.analysisTools.cumulativeTraversalStream(centroidNeck)
    
    maxTraversalDim=np.where(np.max(deltas)==deltas)[0][0]
    #get the current endpoints
    endpoint1=centroidNeck[0,:]
    endpoint2=centroidNeck[-1,:]

    #if the coordinate of endpoint1 in the max traversal dimension
    #is less than the coordinate of endpoint1 in the max traversal dimension
    #flip it
    if endpoint1[maxTraversalDim]<endpoint2[maxTraversalDim]:
        curCentroid= curCentroid[::-1]

    orientedStreams=streamline.orient_by_streamline(clusterStreams, curCentroid)
    
    #I don't think I need this any more
    # refStreams=cluster.refdata
    # for iStreams in range(len(streamIndexes)):
    #     if not np.all(streamlines[streamIndexes[iStreams]][0,:]==orientedStreams[iStreams][0,:]):
    #         streamlines[streamIndexes[iStreams]]= streamlines[streamIndexes[iStreams]][::-1]
    
    #would I rather do this in place?
    return orientedStreams, cluster.indices
    

def bundleTest(streamlines):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from scipy.spatial.distance import cdist
    import numpy as np
    
    neckNodes=findTractNeckNode(streamlines)
    neckCoords=[streamlines[iStreamline][neckNodes[iStreamline],:] for iStreamline in range(len(streamlines)) ]
    
    meanNeckCoord=np.mean(np.squeeze(np.asarray(neckCoords)),axis=0)
    
    neckNodeDistances=np.squeeze(cdist(np.atleast_2d(meanNeckCoord),np.squeeze(np.asarray(neckCoords))))
    
    #if you want 95% of your streamlines within 
    #does mirroring across the y axis turn an exponential distributon into a normal distribution
    #with the same standard deviation (except maybe the standard deivation is doubled?)
    outReport={'mean':np.mean(neckNodeDistances),'median':np.median(neckNodeDistances),'std':np.std(neckNodeDistances),'mirrored_std':np.std(np.hstack([neckNodeDistances,-neckNodeDistances]))}
    
    return outReport

def iteratedStreamsInWindow(streamlines,referenceNifti=None):
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
    import copy
    
    #create a dummy nifti if necessary in order to get a get an affine?
    if referenceNifti==None:
        referenceNifti=dummyNiftiForStreamlines(streamlines)
    
    #this is probably faster than geting the density map, turning that into a mask,
    #and then getting the voxel indexes for that
    # get a streamline index dict of the whole brain tract
    streamlineMapping=streamline_mapping(streamlines, referenceNifti.affine)
    streamMaskCoords=list(streamlineMapping.keys())
    
    dataShape=list(referenceNifti.shape)
    blankNiftiData=np.zeros((dataShape.append(7)))
    for iIterations in range(1,8):

        
        dialatedStreamMappings=streamlinesInWindowFromMapping(streamlines,streamlineMapping,referenceNifti=None,distanceParameter=iIterations)
        proportionInWindow=[np.divide(len(iCoordinates),len(streamlines)) for iCoordinates in dialatedStreamMappings]
        #now map those on to the output nifti
        for iCoordinates in range(len(streamMaskCoords)):
            blankNiftiData[streamMaskCoords[iCoordinates][0],streamMaskCoords[iCoordinates][1],streamMaskCoords[iCoordinates][2],iIterations-1]=proportionInWindow[iCoordinates]

    niftiOut=nib.nifti1.Nifti1Image(np.asarray(blankNiftiData), referenceNifti.affine, referenceNifti.header)
    
    return niftiOut

def streamlinesInWindowFromMapping(streamlines,streamlineMapping,referenceNifti=None,distanceParameter=3):
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
    try: 
        import tqdm
        tqdmFlag=True
    except:
        tqdmFlag=False
    
    
    #create a dummy nifti if necessary in order to get a get an affine?
    if referenceNifti==None:
        referenceNifti=dummyNiftiForStreamlines(streamlines)
    
    #streamlines=orientAllStreamlines(streamlines)
    
    #this is probably faster than geting the density map, turning that into a mask,
    #and then getting the voxel indexes for that
    # get a streamline index dict of the whole brain tract
   
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
    if tqdmFlag:
        streamsListList=[streamsInImgSpaceWindow(iCoords,streamlineMapping,mask_r) for iCoords in tqdm.tqdm(imgCoords, position=0, leave=True)]
    else:
        streamsListList=[streamsInImgSpaceWindow(iCoords,streamlineMapping,mask_r) for iCoords in imgCoords]
  
    t1_stop=time.process_time()
    modifiedTime=t1_stop-t1_start
    print(str(modifiedTime) + ' seconds to find streamline groups to check')  
    
    return streamsListList


# def parseInputWMC(classification):
    
#     import os
#     import json
    
    
#     #route it properly if it is a string 
#     if isinstance(classification,str):
#         #get the file extension
#         splitFile=os.path.splitext(classification)
#         if splitFile[1]=='.json':
#             classificationDict='test'
#         if splitFile[1]=='.mat':
#             classificationDict=matWMC2dict(classification)
#         #.mat

def inputTcks_to_WMCandTCK(tcksORPaths,names=None):
    from glob import glob
    import os
    import nibabel as nib
    import numpy as np
    
    
    if isinstance(tcksORPaths,str):
        if os.path.isdir(tcksORPaths):
            print('input directory detected')
            tckFilePaths=glob(os.path.join(tcksORPaths,'*.tck'))
            #set the variable to this newly generated list
            tcksORPaths=tckFilePaths
            

        else:
            raise Exception('singleton, non directory input not understood')

    
    streamlinesHolders=[[] for iInputs in tcksORPaths]
    #create a holder for inferred names if necessary
    if names==None:
        names=[[] for iInputs in tcksORPaths]
        meaningfulNamesFlag=np.any([isinstance(itcksORPaths,str) for itcksORPaths in tcksORPaths])
        #but names weren't provided
        namesProvidedFlag=False
        #I don't know what to do in mixed cases
        #because of combineTracts(tractsORstreamlines), all we need is names
        for iterator,itcksORPaths in enumerate(tcksORPaths):
            if isinstance(itcksORPaths,str):
                #get the name
                #could cause all kinds of problems if . in name
                nameParts=os.path.basename(itcksORPaths).split('.')
                names[iterator]=nameParts[0]
            else:
                #just generate some names, I guess...
                names[iterator]='tract_'+str(iterator)
    #if names were passed in            
    else:
        meaningfulNamesFlag=True
        #but also, maybe trust the user that the specified order is the one desired
        namesProvidedFlag=True
    
    #if the names are menaingful, lets try and sort the names
    if meaningfulNamesFlag and not namesProvidedFlag:
        flippedStrings=[iNames[::-1] for iNames in names]
        sortOrdering=sorted(range(len(flippedStrings)), key=lambda k: flippedStrings[k])
        sortedNames=[names[iOrder] for iOrder in sortOrdering]
        sortedFiles=[tcksORPaths[iOrder] for iOrder in sortOrdering]
        #this would only ever be triggered if file paths were passed, so its safe to do this
        tcksORPaths=sortedFiles
        names=sortedNames
            
    #run the combine method/algorithm        
    [outStatefulTractogram, streamsCounts]=combineTracts(tcksORPaths)
    indexVec=[]
    for iterator, iCounts in enumerate(streamsCounts):
        indexVec.extend([iterator+1]*iCounts)
        
    wmc_Dict={}
    wmc_Dict['names']=names
    wmc_Dict['index']=indexVec
    
    return outStatefulTractogram, wmc_Dict
    

def wmc2tracts(inputTractogram,classification,outdir):
    """wmc2tracts(trk_file,classification,outdir):
     convert a wmc .mat + tract input into separate files for tracts 
    based on @giulia-berto's
 https://github.com/FBK-NILab/app-classifyber-segmentation/blob/1.3/wmc2trk.py
 
    INPUTS
    
    inputTractogram: an input tractogram
    
    classification: a .mat WMC input that corresponds to the input tractogram
    WMC described here: https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b
    
    outdir: the output directory in which to save the output files
 
    """
    import nibabel as nib
    from scipy.io import loadmat
    import dipy
    import numpy as np
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import load_tractogram, save_tractogram
    import dipy.tracking.utils as ut
    import os
    
    if isinstance(inputTractogram,str):
        inputTractogram=inputTractogram=nib.streamlines.load(inputTractogram)
        print('input tractogram loaded')
    
    if isinstance(classification,str):
        #load the .mat object
        classification=matWMC2dict(classification)
    else:
        pass
    
    #it comes back as an eldridch horror, so parse it appropriately
    #get the index vector
    indices=classification['index']
    #get the names vector
    tractIdentities=classification['names']
    
    for tractID in range(len(tractIdentities)):
        #remove unncessary characters, adds unnecessary '[]'
        tract_name = tractIdentities[tractID]
        idx_tract = np.array(np.where(indices==tractID+1))[0]
        
        #save it in the same format as the input
        if isinstance(inputTractogram, nib.streamlines.tck.TckFile):
            out_filename=os.path.join(outdir,tract_name + '.tck')
        elif  isinstance(inputTractogram, nib.streamlines.trk.TrkFile):
            out_filename=os.path.join(outdir,tract_name + '.trk')
        print('saving '+ str(len(idx_tract)) + ' streamlines for' + tract_name+ ' to')
        print(out_filename)
        if isinstance( inputTractogram,nib.streamlines.array_sequence.ArraySequence):
            #maybe they input just streamlines
            stubbornSaveTractogram(inputTractogram[idx_tract],out_filename)
        else:
            #better hope its a stateful tractogram
            stubbornSaveTractogram(inputTractogram.streamlines[idx_tract],out_filename)
  
def matWMC2dict(classification):
    """
    "wmc2tracts(trk_file,classification,outdir):
     convert a .mat wmc to a dict / .json variant
    based on @giulia-berto's
 https://github.com/FBK-NILab/app-classifyber-segmentation/blob/1.3/wmc2trk.py

    Parameters
    ----------

    classification : TYPE
        DESCRIPTION.

    Returns
    -------
    wmc_Dict: a dictionary object with "names" and "index" field, in accordance
    with the wmc datatype
    https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme


    """
    from scipy.io import loadmat
    import json
    import numpy as np
    
    if isinstance(classification,str):
        print('loading .mat input')
        #load the .mat object
        classification=loadmat(classification)
        #it comes back as an eldridch horror, so parse it appropriately
        #get the index vector
    print('digesting .mat object')
    indices=classification['classification'][0][0]['index'][0]
        #get the names vector
        #this is... the worst thing ever
    tractIdentities=[str(iIdenties[0]) for iIdenties in classification['classification'][0][0]['names'].flatten()]
    tractNames=[]
    for iterator,inName in enumerate(tractIdentities):
        #remove unncessary characters, adds unnecessary '[]'
        #this has been solved
        #t_name = tractIdentities[tractID][2:-2]
        #standard practice: get rid of all spaces
        tractNames.append(inName.replace(' ', '_'))
        #also consider using this opportunity to fix/enforce naming conventions
    
    
     #create the dictionary object 
    wmc_Dict={}
    wmc_Dict['names']=tractNames
    wmc_Dict['index']=indices.tolist()
    #and the fix
    wmc_Dict['names']=np.array(wmc_Dict['names'], dtype=np.object) 
    print('input classification structure represents ' + str(len(wmc_Dict['names'])) + ' structures composed of ' + str(np.sum(np.greater(wmc_Dict['index'],0))) + ' out of ' + str(len(indices)) + 'total available streamlines.')
    
    return wmc_Dict

def updateClassification(boolOrIndexesIn,name,existingClassification=None):
    """
    Updates or creates a dict-based wmc-type classification structure.
        See link below for format specification:
        https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme. 

    Parameters
    ----------
    boolOrIndexesIn : Bool or list/array of INTs
        Either a boolean vector or a list/array of ents.  The entries in this
        input are presumed to indicate the streamlines corresponding to the
        identity specified in the "name" input.
    name : string
        The name corresponding to the current structure.  This identity is
        ascribed to the streamlines specified in the "boolOrIndexesIn" input.
    existingClassification : dict, optional
        A dictionary object with "names" and "index" field, in accordance
        with the wmc datatype
        https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme. 
        If one is not provided, an attempt to create a new one will be made.
        The default is None.

    Raises
    ------
    ValueError
        Will throw an error if you ask it to create a new classification
        while inputting only indexes.  The lenght of the source tractogram
        cannot be inferred from such information.
        
    Warning
        Will throw a warning if you enter a name that already exists in the 
        classification.  Then will overwrite those entries.

    Returns
    -------
    wmc_Dict : dict
        A dictionary object with "names" and "index" field, in accordance
        with the wmc datatype
        https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme

    """
    
    import numpy as np
    from warnings import warn
    #determine whether the input vector is bool or indexes
    uniqueInputVals=np.unique(boolOrIndexesIn)
    #if the unique values are limited to some combination of ones and zeros
    if np.any([np.array_equal(uniqueInputVals,[0,1]), np.array_equal(uniqueInputVals,[1]),np.array_equal(uniqueInputVals,[0])]):
        #its a bool
        #for the sake of expediency and standardization, lets get the length
        #and convert this to indexes
        currentIndexes=np.where(boolOrIndexesIn)[0]
        streamlinesLength=len(boolOrIndexesIn)
    else:
        #otherwise, they are indexes 
        currentIndexes=boolOrIndexesIn
        streamlinesLength=None 
    
    #determine if input classification exists
    if existingClassification==None:
        #need to make a new one
        wmc_Dict={}
        
        #if there is a valid streamline lenth, create a blank index structure
        if isinstance(streamlinesLength,int):
            wmc_Dict['index']=np.zeros(streamlinesLength,dtype=int)
        else:
            raise ValueError('Input indexes do not indicate TOTAL number of streamlines in input tract \ncCan not create new wmc structure without this information.')
        
        #now that we have sorted that out, we can add the name and set the indexes
        wmc_Dict['names']=np.array([name])
        #because we are creating this new, it is safe to assume that the new
        #index for this name is 1
        wmc_Dict['index'][currentIndexes]=1
    
    else:
        wmc_Dict=existingClassification
        #check to see if the name is already in there
        #if it is...
        if name in wmc_Dict['names']:
            #throw a warning that you're about to overwrite that listing
            warn('Input name ' + name + ' detected in input classification structure \nOverwriting previous record(s)')
            #we have to add one because we can't use 0 indexing
            currentIndex=np.where(wmc_Dict['names']==name)[0][0]+1
            #find the locations of where this is
            currentIndexMatches=wmc_Dict['index']==currentIndex
            #reset those entries to 0
            wmc_Dict['index'][currentIndexMatches]=0
            #now set the index entries with the current information
            wmc_Dict['index'][currentIndexes]=currentIndex
        #otherwise
        else:
            #add it to the list of names 
            wmc_Dict['names']=np.append(wmc_Dict['names'],name)
            #get the new lenght of this vector
            #we don't need to add 1 because length reflects appropriate index
            currentIndex=len(wmc_Dict['names'])
            wmc_Dict['index'][currentIndexes]=currentIndex
            
    #BEFORE WE LEAVE THOUGH, we have to convert the list to an np array for some reason:
    #https://stackoverflow.com/questions/7464632/python-to-mat-file-export-list-of-string-to-ordinar-matrix-of-chars-not-a-cell
    
    wmc_Dict['names']=np.array(wmc_Dict['names'], dtype=np.object)        
    return wmc_Dict
    
  
def matWMC2jsonWMC(classification):
    """
    "wmc2tracts(trk_file,classification,outdir):
     convert a .mat wmc to a dict / .json variant
    based on @giulia-berto's
 https://github.com/FBK-NILab/app-classifyber-segmentation/blob/1.3/wmc2trk.py

    Parameters
    ----------

    classification : TYPE
        DESCRIPTION.

    Returns
    -------
    None.


    """
    from scipy.io import loadmat
    import json
    
    if isinstance(classification,str):
        #load the .mat object
        classification=loadmat(classification)
        #it comes back as an eldridch horror, so parse it appropriately
        #get the index vector
    indices=classification['classification'][0][0]['index'][0]
        #get the names vector
    tractIdentities=[str(iIdenties) for iIdenties in classification['classification'][0][0][0][0]]
    tractNames=[]
    for tractID in range(len(tractIdentities)):
        #remove unncessary characters, adds unnecessary '[]'
        t_name = tractIdentities[tractID][2:-2]
        #standard practice: get rid of all spaces
        tractNames.append(t_name.replace(' ', '_'))
        #also consider using this opportunity to fix/enforce naming conventions
    
    #create the dictionary object 
    wmc_Dict={}
    wmc_Dict['names']=tractNames
    wmc_Dict['index']=indices.tolist()
    
    outJson=json.dumps(wmc_Dict)
    return outJson

def stubbornSaveTractogram(streamlines,savePath=None):
    """
    Why shouuld i supply a reference nifti?
    
    NOTE this only works for the sensible format of streamlines, i.e. RASMM

    Returns
    -------
    None.

    """
    import nibabel as nib
    import dipy
    import numpy as np
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import load_tractogram, save_tractogram
    import dipy.tracking.utils as ut
    
    from dipy.tracking.streamline import Streamlines

    #dipy is stubborn and wants a reference nifti for some reason
    #fineI'llDoItMyself.jpg
    tractBounds=np.asarray([np.min(streamlines._data,axis=0),np.max(streamlines._data,axis=0)])
    roundedTractBounds=np.asarray([np.floor(tractBounds[0,:]),np.ceil(tractBounds[1,:])])
    constructedAffine=np.eye(4)
    constructedAffine[0:3,3]=tractBounds[0,:]

    lin_T, offset =ut._mapping_to_voxel(constructedAffine)
    inds = ut._to_voxel_coordinates(streamlines._data, lin_T, offset)
    
    testBounds=np.asarray([np.min(inds,axis=0),np.max(inds,axis=0)])
    
    #now create a dummy nifit, because that's what dipy demands
    dataShape=(roundedTractBounds[1,:]-roundedTractBounds[0,:]).astype(int)
    #adding a +1 pad because it yells otherwise?
    dummyData=np.zeros(dataShape+1)
    dummyNifti= nib.nifti1.Nifti1Image(dummyData, constructedAffine)
   
    #voxStreams=dipy.tracking.streamline.transform_streamlines(streamlines,np.linalg.inv(constructedAffine))
    #statefulTractogramOut=StatefulTractogram(voxStreams, dummyNifti, Space.VOX)
    #note we have to force a conversion here
    statefulTractogramOut=StatefulTractogram(Streamlines(streamlines), dummyNifti, Space.RASMM)
    if not savePath==None:
        save_tractogram(statefulTractogramOut,savePath, bbox_valid_check=False)
    return statefulTractogramOut

def orientTractUsingNeck(streamlines,refAnatT1=None,surpressReport=False):
    """orientTractUsingNeck(streamlines)
    A function which uses the neck of a (presumed) tract to flip consituent
    streamlines so that they are all in the same orientation.  This function
    exists because dipy's streamline.orient_by_streamline doesn't work, at
    at least not when used with a centroid streamline

    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        A collection of streamlines presumably corresponding to a tract.
        Unknown functionality if a random collection of streamlines is used

    Returns
    -------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence 
        The input streamlines, but with the appropriate streamlines flipped 
        such that all streamlines proceed in the same orientation

    """
    import numpy as np
    from scipy.spatial.distance import cdist
    import copy
    import wmaPyTools.analysisTools  
    
    #this has to be run a couple times to ensure that streamlines that are initially 
    #flipped aren't throwing off the group average too bad
    flipRuns=1
    flipCount=0
    exitTrigger=False
    #here lets arbitrarily aim for 5 mm, and compute that distance per streamline.  Probably expensive
    #but we can't always assume that 
    
    #set the value here, the mm space target distance for neck evaluation
    targetMM=2.0
    
    #we only need to do this once per orientation
    #get the neck nodes for the tract
    #if you're doing this a lot, it speeds it up to pass a ref anat
    if refAnatT1==None:
        neckNodes=findTractNeckNode(streamlines)
    else:
        neckNodes=findTractNeckNode(streamlines,refAnatT1)
    
    #obtain the coordinates for each of these neck nodes
    neckCoords=np.zeros([len(streamlines),3])
    for iStreamlines in range(len(streamlines)):
        neckCoords[iStreamlines,:]=streamlines[iStreamlines][neckNodes[iStreamlines]]
        
    #establish correspondance of lookdistance
    nodeDists1=np.sqrt(np.sum(np.square(np.diff(streamlines[0],axis=0)),axis=1))
    nodeDists2=np.sqrt(np.sum(np.square(np.diff(streamlines[len(streamlines)-1],axis=0)),axis=1))
    #if the streamlines have a fairly regular internode distance
    if np.abs(np.mean(nodeDists1)-np.mean(nodeDists2))/np.mean([np.mean(nodeDists1),np.mean(nodeDists2)]) <.10:
        #compute the look distance based on this average for 5mm
        lookDistance=np.round(targetMM/np.mean([np.mean(nodeDists1),np.mean(nodeDists2)])).astype(int)
    else:
        
        raise Exception('streamlines have variable internode distances, \nplease resample to a standard spacing')

        
    aheadNodes=np.zeros(len(streamlines)).astype(int)
    behindNodes=np.zeros(len(streamlines)).astype(int)
    for iStreamlines in range(len(streamlines)):

       #A check to make sure you've got room to do this indexing on both sides
        if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)[0]>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)[0]>lookDistance):
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+lookDistance
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-lookDistance
       #otherwise do the best you can
        else:
           #if there's a limit to how many nodes are available ahead, do the best you can
           spaceAhead=np.abs(neckNodes[iStreamlines][0]-len(streamlines[iStreamlines]))
           if spaceAhead<=lookDistance:
               aheadWindow=spaceAhead-1
           else:
               aheadWindow=lookDistance
           #if there's a limit to how many nodes are available behind, do the best you can
           spaceBehind=np.abs(len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines][0]))
           if spaceBehind<=lookDistance:
               behindWindow=spaceBehind-1
           else:
               behindWindow=lookDistance
           
           #append the relevant values
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+(aheadWindow)
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-(behindWindow)
    
    aheadCoords=[ streamlines[iStreamlines][behindNodes[iStreamlines]] for iStreamlines in range(len(streamlines))]
    behindCoords=[ streamlines[iStreamlines][aheadNodes[iStreamlines]] for iStreamlines in range(len(streamlines))]
    
    streamTraversals=[wmaPyTools.analysisTools.cumulativeTraversalStream(istreamlines) for istreamlines in streamlines]
    avgTraversals=np.mean(streamTraversals,axis=0)
    
    if not surpressReport:
        print('initial orientation complete')
    
    while not exitTrigger: 
      
        #however, if the algorithm gets stuck (and keeps flipping one streamline back and forth, for example)
        #expand the window a bit to hopefully introduce more variability in to the system
        #(under the presumption that streamlines get more variable as you move away from the neck)
        #the modulo operator here will expand the window everytime the flip run counter reaches a multiple of 10
        if flipRuns % 12 == 0 :
            print('algorithm stuck, expanding consideration window and randomizing streamline orientations')
            targetMM=targetMM+.5
            
            for iStreamlines in range(len(streamlines)):
                if np.random.choice([True,False]):
                    streamlines[iStreamlines]= streamlines[iStreamlines][::-1]
            
            #you will also need to recompute the ahead and behind nodes    
            aheadNodes=np.zeros(len(streamlines)).astype(int)
            behindNodes=np.zeros(len(streamlines)).astype(int)
            for iStreamlines in range(len(streamlines)):

               #A check to make sure you've got room to do this indexing on both sides
                if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)[0]>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)[0]>lookDistance):
                   aheadNodes[iStreamlines]=neckNodes[iStreamlines]+lookDistance
                   behindNodes[iStreamlines]=neckNodes[iStreamlines]-lookDistance
               #otherwise do the best you can
                else:
                   #if there's a limit to how many nodes are available ahead, do the best you can
                   spaceAhead=np.abs(neckNodes[iStreamlines][0]-len(streamlines[iStreamlines]))
                   if spaceAhead<=lookDistance:
                       aheadWindow=spaceAhead-1
                   else:
                       aheadWindow=lookDistance
                   #if there's a limit to how many nodes are available behind, do the best you can
                   spaceBehind=np.abs(len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines][0]))
                   if spaceBehind<=lookDistance:
                       behindWindow=spaceBehind-1
                   else:
                       behindWindow=lookDistance
                   
                   #append the relevant values
                   aheadNodes[iStreamlines]=neckNodes[iStreamlines]+(aheadWindow)
                   behindNodes[iStreamlines]=neckNodes[iStreamlines]-(behindWindow)
            
            aheadCoords=[ streamlines[iStreamlines][behindNodes[iStreamlines]] for iStreamlines in range(len(streamlines))]
            behindCoords=[ streamlines[iStreamlines][aheadNodes[iStreamlines]] for iStreamlines in range(len(streamlines))]
        
        # use the coords that are at the heart of the tract to establish the orientation guide
        
        #first set empty vectors for neck, ahead, and behind coords
        if not surpressReport:
            print('coord corespondances extracted')
        #establish 
        aheadDistances=np.zeros(len(streamlines))
        behindDistances=np.zeros(len(streamlines))
        averageAheadCoord=np.mean(np.asarray(aheadCoords),axis=0)
        averageBehindCoord=np.mean(np.asarray(behindCoords),axis=0)
        for iStreamlines in range(len(streamlines)):
             aheadDistances[iStreamlines]=np.squeeze(cdist(np.atleast_2d(averageAheadCoord),np.atleast_2d(streamlines[iStreamlines][behindNodes[iStreamlines]])))
             behindDistances[iStreamlines]=np.squeeze(cdist(np.atleast_2d(averageBehindCoord),np.atleast_2d(streamlines[iStreamlines][behindNodes[iStreamlines]])))
        if not surpressReport:
            print('coord distances from mean computed')
        orientationGuideAheadCoord=aheadCoords[np.where(np.min(aheadDistances)==aheadDistances)[0][0]].flatten()
        orientationGuideBehindCoord=behindCoords[np.where(np.min(behindDistances)==behindDistances)[0][0]].flatten()
        
        
        
        
        #if you wanted to force RAS / LPI on your tractogram orientatiuon, now would be the time to do it
        #first store the current values in a separate variable
        #DISPALCEMENT IS ACTUALLY BAD FOR THIS, WHAT YOU WANT IS TRAVERSAL
        currentAheadNode=copy.deepcopy(orientationGuideAheadCoord)
        currentBehindNode=copy.deepcopy(orientationGuideBehindCoord)
        #find the displacement for each dimension
        currentDiaplacement=currentAheadNode-currentBehindNode
        maxTraversalDim=np.where(np.max(avgTraversals)==avgTraversals)[0]
        #if that value is negative, switch the identity of the ahead and behind nodes
        if currentDiaplacement[maxTraversalDim]<0:
            orientationGuideAheadCoord=currentBehindNode
            orientationGuideBehindCoord=currentAheadNode
        
        #print(str(orientationGuideAheadCoord))
        #print(str(orientationGuideBehindCoord))
        if not surpressReport:
            print('orentation nodes specified')
        #iterate across streamlines     
        for iStreamlines in range(len(streamlines)):
            #compute the distances from the comparison orientation for both possible
            #orientations
            sumDistanceOrientation1=np.sum([cdist(np.atleast_2d(orientationGuideAheadCoord),np.atleast_2d(aheadCoords[iStreamlines])),cdist(np.atleast_2d(orientationGuideBehindCoord),np.atleast_2d(behindCoords[iStreamlines]))])
            sumDistanceOrientation2=np.sum([cdist(np.atleast_2d(orientationGuideAheadCoord),np.atleast_2d(behindCoords[iStreamlines])),cdist(np.atleast_2d(orientationGuideBehindCoord),np.atleast_2d(aheadCoords[iStreamlines]))])
            #flip if necessary
            if sumDistanceOrientation2<sumDistanceOrientation1:
                #print(str(iStreamlines))
                #print(str(sumDistanceOrientation1))
                #print(str(sumDistanceOrientation2))
                
                #flip the nodes and coords if necessary
                currentAheadNode=copy.deepcopy(aheadNodes[iStreamlines])
                currentBehindNode=copy.deepcopy(behindNodes[iStreamlines])
                aheadNodes[iStreamlines]=currentBehindNode
                behindNodes[iStreamlines]=currentAheadNode
                aheadCoords[iStreamlines]=streamlines[iStreamlines][aheadNodes[iStreamlines]]
                behindCoords[iStreamlines]=streamlines[iStreamlines][behindNodes[iStreamlines]]
                
                streamlines[iStreamlines]= streamlines[iStreamlines][::-1]
                flipCount=flipCount+1
        if not surpressReport:
            print('flip run ' + str(flipRuns) +': ' + str(flipCount) + ' of ' + str(len(streamlines)) + ' streamlines flipped')
            
        if np.logical_and(flipRuns!=1,flipCount==0) :
            exitTrigger=True
        else:
            #set the counters
            flipRuns=flipRuns+1
            flipCount=0
            
    #I don't know that I trust the whole in place flipping mechanism, so
    #we will return the modified streamline object from this function
    return streamlines

def findAheadAndBehindNodes(streamlines,neckNodes,lookDistance):    
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    aheadNodes : TYPE
        DESCRIPTION.
    behindNodes : TYPE
        DESCRIPTION.

    """
    import numpy as np
    aheadNodes=np.zeros(len(streamlines)).astype(int)
    behindNodes=np.zeros(len(streamlines)).astype(int)
    for iStreamlines in range(len(streamlines)):

       #A check to make sure you've got room to do this indexing on both sides
        if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)[0]>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)[0]>lookDistance):
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+lookDistance
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-lookDistance
       #otherwise do the best you can
        else:
           #if there's a limit to how many nodes are available ahead, do the best you can
           spaceAhead=np.abs(neckNodes[iStreamlines][0]-len(streamlines[iStreamlines]))
           if spaceAhead<=lookDistance:
               aheadWindow=spaceAhead-1
           else:
               aheadWindow=lookDistance
           #if there's a limit to how many nodes are available behind, do the best you can
           spaceBehind=np.abs(len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines][0]))
           if spaceBehind<=lookDistance:
               behindWindow=spaceBehind-1
           else:
               behindWindow=lookDistance
           
           #append the relevant values
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+(aheadWindow)
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-(behindWindow)
           
    return aheadNodes, behindNodes

def orientTractUsingNeck_Robust(streamlines,refAnatT1=None,surpressReport=False):
    """orientTractUsingNeck(streamlines)
    A function which uses the neck of a (presumed) tract to flip consituent
    streamlines so that they are all in the same orientation.  This function
    exists because dipy's streamline.orient_by_streamline doesn't work, at
    at least not when used with a centroid streamline

    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        A collection of streamlines presumably corresponding to a tract.
        Unknown functionality if a random collection of streamlines is used

    Returns
    -------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence 
        The input streamlines, but with the appropriate streamlines flipped 
        such that all streamlines proceed in the same orientation

    """
    import numpy as np
    from scipy.spatial.distance import cdist
    import copy
    from dipy.segment.metric import SumPointwiseEuclideanMetric
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    from dipy.segment.clustering import QuickBundles
    
    #this has to be run a couple times to ensure that streamlines that are initially 
    #flipped aren't throwing off the group average too bad
    flipRuns=1
    flipCount=0
    expandCount=0
    splitCount=0
    exitTrigger=False
    #here lets arbitrarily aim for 5 mm, and compute that distance per streamline.  Probably expensive
    #but we can't always assume that 
    
    #set the value here, the mm space target distance for neck evaluation
    targetMM=2.0
    
    #we only need to do this once per orientation
    #get the neck nodes for the tract
    #if you're doing this a lot, it speeds it up to pass a ref anat
    if refAnatT1==None:
        neckNodes=findTractNeckNode(streamlines)
    else:
        neckNodes=findTractNeckNode(streamlines,refAnatT1)
    
    #obtain the coordinates for each of these neck nodes
    neckCoords=np.zeros([len(streamlines),3])
    for iStreamlines in range(len(streamlines)):
        neckCoords[iStreamlines,:]=streamlines[iStreamlines][neckNodes[iStreamlines]]
        
    #establish correspondance of lookdistance
    nodeDists1=np.sqrt(np.sum(np.square(np.diff(streamlines[0],axis=0)),axis=1))
    nodeDists2=np.sqrt(np.sum(np.square(np.diff(streamlines[len(streamlines)-1],axis=0)),axis=1))
    #if the streamlines have a fairly regular internode distance
    if np.abs(np.mean(nodeDists1)-np.mean(nodeDists2))/np.mean([np.mean(nodeDists1),np.mean(nodeDists2)]) <.10:
        #compute the look distance based on this average for 5mm
        lookDistance=np.round(targetMM/np.mean([np.mean(nodeDists1),np.mean(nodeDists2)])).astype(int)
    else:
        
        raise Exception('streamlines have variable internode distances, \nplease resample to a standard spacing')

    #find the initial arrangement of ahead and behind nodes (i.e. naieve)
    [aheadNodes, behindNodes]=findAheadAndBehindNodes(streamlines,neckNodes,lookDistance)
    
    if not surpressReport:
        print('initial orientation complete')
    
    #at first, try and flip all of the streamlines
    tryTheseIndexes=[range(len(streamlines))]
    exitTriggerAll=False
    while not exitTriggerAll:
        try:
            for iSubsets in range(len(tryTheseIndexes)):
                exitTriggerOne=False
                flipRuns=1
                flipCount=0
                expandCount=0
                while not exitTriggerOne:                   
                    #if we've expanded a full 5mm and we still can't get anywhere
                    if expandCount==10:
                        print('splitting streamlines into subgroups')
                        #iterate the split count
                        splitCount=splitCount+1
                        #clear the tryThese structure
                        tryTheseIndexes=[]
                        #lets split the tract up and try again
                        feature = ResampleFeature(nb_points=100)
                        metric = AveragePointwiseEuclideanMetric(feature)
                        qb = QuickBundles(threshold=2,metric=metric, max_nb_clusters = splitCount+1)
                        cluster = qb.cluster(streamlines)
                        for iClusters in range(len(cluster)):
                            tryTheseIndexes.append(cluster.clusters[iClusters].indices)
                        
                        raise Exception
                        
                    
                  
                    #however, if the algorithm gets stuck (and keeps flipping one streamline back and forth, for example)
                    #expand the window a bit to hopefully introduce more variability in to the system
                    #(under the presumption that streamlines get more variable as you move away from the neck)
                    #the modulo operator here will expand the window everytime the flip run counter reaches a multiple of 10
                    if flipRuns % 12 == 0 :
                        if not surpressReport:
                            print('algorithm stuck, expanding consideration window and randomizing streamline orientations')
                        targetMM=targetMM+.5
                        expandCount=expandCount+1
                        lookDistance=np.round(targetMM/np.mean([np.mean(nodeDists1),np.mean(nodeDists2)])).astype(int)
        
                        
                        for iStreamlines in range(len(streamlines[tryTheseIndexes[iSubsets]])):
                            if np.random.choice([True,False]):
                                streamlines[tryTheseIndexes[iSubsets]][iStreamlines]= streamlines[tryTheseIndexes[iSubsets]][iStreamlines][::-1]
                        
                        [aheadNodes, behindNodes]=findAheadAndBehindNodes(streamlines[tryTheseIndexes[iSubsets]],neckNodes[tryTheseIndexes[iSubsets]],lookDistance)
                       
                        
                       
                    # use the coords that are at the heart of the tract to establish the orientation guide
                    
                    [streamlines[tryTheseIndexes[iSubsets]], aheadNodes, behindNodes, flipCount]=flipStreamstoAB_OrientOnce(streamlines[tryTheseIndexes[iSubsets]], aheadNodes, behindNodes,surpressReport=True)
                    if not surpressReport:
                        print('flip run ' + str(flipRuns) +': ' + str(flipCount) + ' of ' + str(len(streamlines)) + ' streamlines flipped')
                
                    if np.all([flipRuns!=1,flipCount==0]) :
                        exitTriggerOne=True
                    else:
                      #set the counters
                          flipRuns=flipRuns+1
                          flipCount=0
        except:
            pass
        
        if np.all([flipRuns!=1,flipCount==0, iSubsets==len(tryTheseIndexes)-1]) :
            exitTriggerAll=True
        else:
            #set the counters
            flipRuns=flipRuns+1
            flipCount=0
            
    #I don't know that I trust the whole in place flipping mechanism, so
    #we will return the modified streamline object from this function
    return streamlines




def flipStreamstoAB_OrientOnce(streamlines, aheadNodes, behindNodes,surpressReport=False):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import numpy as np
    from scipy.spatial.distance import cdist
    import copy
    import wmaPyTools.analysisTools  
    
    flipCount=0
    #compute the coordinates
    aheadCoords=[ streamlines[iStreamlines][behindNodes[iStreamlines]] for iStreamlines in range(len(streamlines))]
    behindCoords=[ streamlines[iStreamlines][aheadNodes[iStreamlines]] for iStreamlines in range(len(streamlines))]

    # use the coords that are at the heart of the tract to establish the orientation guide

    #first set empty vectors for neck, ahead, and behind coords
    if not surpressReport:
        print('coord corespondances extracted')
    #establish 
    aheadDistances=np.zeros(len(streamlines))
    behindDistances=np.zeros(len(streamlines))
    averageAheadCoord=np.mean(np.asarray(aheadCoords),axis=0)
    averageBehindCoord=np.mean(np.asarray(behindCoords),axis=0)
    for iStreamlines in range(len(streamlines)):
         #print(str(iStreamlines))
         aheadDistances[iStreamlines]=np.squeeze(cdist(np.atleast_2d(averageAheadCoord),np.atleast_2d(streamlines[iStreamlines][behindNodes[iStreamlines]])))
         behindDistances[iStreamlines]=np.squeeze(cdist(np.atleast_2d(averageBehindCoord),np.atleast_2d(streamlines[iStreamlines][behindNodes[iStreamlines]])))
    if not surpressReport:
        print('coord distances from mean computed')
    orientationGuideAheadCoord=aheadCoords[np.where(np.min(aheadDistances)==aheadDistances)[0][0]].flatten()
    orientationGuideBehindCoord=behindCoords[np.where(np.min(behindDistances)==behindDistances)[0][0]].flatten()
    
    streamTraversals=[wmaPyTools.analysisTools.cumulativeTraversalStream(istreamlines) for istreamlines in streamlines]
    avgTraversals=np.mean(streamTraversals,axis=0)
    
    
    #if you wanted to force RAS / LPI on your tractogram orientatiuon, now would be the time to do it
    #first store the current values in a separate variable
    #DISPALCEMENT IS ACTUALLY BAD FOR THIS, WHAT YOU WANT IS TRAVERSAL
    currentAheadNode=copy.deepcopy(orientationGuideAheadCoord)
    currentBehindNode=copy.deepcopy(orientationGuideBehindCoord)
    #find the displacement for each dimension
    currentDiaplacement=currentAheadNode-currentBehindNode
    maxTraversalDim=np.where(np.max(avgTraversals)==avgTraversals)[0]
    #if that value is negative, switch the identity of the ahead and behind nodes
    if currentDiaplacement[maxTraversalDim]<0:
        orientationGuideAheadCoord=currentBehindNode
        orientationGuideBehindCoord=currentAheadNode
    
    #print(str(orientationGuideAheadCoord))
    #print(str(orientationGuideBehindCoord))
    if not surpressReport:
        print('orentation nodes specified')
    #iterate across streamlines     
    for iStreamlines in range(len(streamlines)):
        #compute the distances from the comparison orientation for both possible
        #orientations
        sumDistanceOrientation1=np.sum(cdist(np.atleast_2d(orientationGuideAheadCoord),np.atleast_2d(aheadCoords[iStreamlines])))+np.sum(cdist(np.atleast_2d(orientationGuideBehindCoord),np.atleast_2d(behindCoords[iStreamlines])))
        sumDistanceOrientation2=np.sum(cdist(np.atleast_2d(orientationGuideAheadCoord),np.atleast_2d(behindCoords[iStreamlines])))+np.sum(cdist(np.atleast_2d(orientationGuideBehindCoord),np.atleast_2d(aheadCoords[iStreamlines])))
        #flip if necessary
        if sumDistanceOrientation2<sumDistanceOrientation1:
            #print(str(iStreamlines))
            #print(str(sumDistanceOrientation1))
            #print(str(sumDistanceOrientation2))
            
            #flip the nodes and coords if necessary
            currentAheadNode=copy.deepcopy(aheadNodes[iStreamlines])
            currentBehindNode=copy.deepcopy(behindNodes[iStreamlines])
            aheadNodes[iStreamlines]=currentBehindNode
            behindNodes[iStreamlines]=currentAheadNode
            #aheadCoords[iStreamlines]=streamlines[iStreamlines][aheadNodes[iStreamlines]]
            #behindCoords[iStreamlines]=streamlines[iStreamlines][behindNodes[iStreamlines]]
            
            streamlines[iStreamlines]= streamlines[iStreamlines][::-1]
            flipCount=flipCount+1
    return streamlines, aheadNodes, behindNodes, flipCount

def orientAllStreamlines(tractStreamlines):
    """
    Ok, so here's a philosophical quandry:  do you actually need a specific
    tract in order to orient a streamline.  That is, do you need the reference
    of the larger tract in order to determine the appropriate RAS-LPI orientation
    of all constituent streamlines.  I'd suggest no.  For any given streamline
    there can be a computation for traversals, and thus an oppriate orientation
    for THAT particular streamline, relative to he maximal dimension of traversal

    Jan 13 2022 Note:  Apparently the use of the variable name "streamlines"
    was causing kernel crashes, but it is unclear why this was.

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import numpy as np
    #conditional import of tqdm
    try: 
        import tqdm
        tqdmFlag=True
    except:
        tqdmFlag=False
    import wmaPyTools.analysisTools
    
    #create a counter, for fun
    flipCount=0
    if tqdmFlag:
        for iStreamlines in tqdm.tqdm(range(len(tractStreamlines))):
            
            #compute traversals for streamline
            traversals=wmaPyTools.analysisTools.cumulativeTraversalStream(tractStreamlines[iStreamlines])
            #find which dimension has max traversal
            maxTraversalDim=np.where(np.max(traversals)==traversals)[0][0]
            #get the current endpoints
            endpoint1=tractStreamlines[iStreamlines][0,:]
            endpoint2=tractStreamlines[iStreamlines][-1,:]
        
            #if the coordinate of endpoint1 in the max traversal dimension
            #is less than the coordinate of endpoint1 in the max traversal dimension
            #flip it
            if endpoint1[maxTraversalDim]<endpoint2[maxTraversalDim]:
                tractStreamlines[iStreamlines]= tractStreamlines[iStreamlines][::-1]
                flipCount=flipCount+1
    else:
        for iStreamlines in range(len(tractStreamlines)):
            
            #compute traversals for streamline
            traversals=wmaPyTools.analysisTools.cumulativeTraversalStream(tractStreamlines[iStreamlines])
            #find which dimension has max traversal
            maxTraversalDim=np.where(np.max(traversals)==traversals)[0][0]
            #get the current endpoints
            endpoint1=tractStreamlines[iStreamlines][0,:]
            endpoint2=tractStreamlines[iStreamlines][-1,:]
        
            #if the coordinate of endpoint1 in the max traversal dimension
            #is less than the coordinate of endpoint1 in the max traversal dimension
            #flip it
            if endpoint1[maxTraversalDim]<endpoint2[maxTraversalDim]:
                tractStreamlines[iStreamlines]= tractStreamlines[iStreamlines][::-1]
                flipCount=flipCount+1
    #add a report        
    print( str(flipCount) + ' of ' + str(len(tractStreamlines)) + ' streamlines flipped')

    return  tractStreamlines

def wmc_from_DIPY_connectome(grouping,lookupTable):
    """
    Generates a wmc-type classification structure
    https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme
    from the output of dipy.tracking.utils.connectivity_matrix

    Parameters
    ----------
    grouping : dict, output of dipy.tracking.utils.connectivity_matrix
        The direct dictionary output of the grouping return object from
        dipy.tracking.utils.connectivity_matrix
    lookupTable : pandas.DataFrame
        A lookup table derived either from wmaPyTools.analysisTools.reduceAtlasAndLookupTable
        or wmaPyTools.genUtils.parcJSON_to_LUT, but probably the former, given
        the requirements of dipy.tracking.utils.connectivity_matrix

    Returns
    -------
    classification : a dictionary object with "names" and "index" field, in accordance
    with the wmc datatype
    https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme

    """
    import numpy as np
    import itertools
    #ugly way to get max index value from grouping
    #np.max(grouping[list(grouping.keys())])
    maxStreamIndex=np.max(list(itertools.chain(*[grouping[ikeys] for ikeys in list(grouping.keys())])))
    #remember, zero indexing
        
    for iterator,iConnections in enumerate(list( grouping.keys())):
        boolvec=np.zeros(maxStreamIndex+1,dtype=bool)
        currentName1=lookupTable['labelNames'].iloc[iConnections[0]]
        currentName2=lookupTable['labelNames'].iloc[iConnections[1]]
        connectionName=currentName1 + '_TO_' + currentName2
        currentIndexes=grouping[iConnections]
        boolvec[currentIndexes]=True
        if not 'classification' in locals():
            classification=updateClassification(boolvec,connectionName,existingClassification=None)
        else:
            classification=updateClassification(boolvec,connectionName,existingClassification=classification)
        
    return classification

def downsampleToEndpoints(streamlines):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    endpointsAsStreams : TYPE
        DESCRIPTION.

    """      
    import numpy as np
    from dipy.tracking.streamline import Streamlines
    
    #create blank structure for endpoints
    endpoints=np.zeros((len(streamlines),6))
    #get the endpoints, taken from
    #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
    for iStreamline in range(len(streamlines)):
        #remember, first 3 = endpoint 1, last 3 = endpoint 2    
        endpoints[iStreamline,:]= np.concatenate([streamlines[iStreamline][0,:], streamlines[iStreamline][-1,:]])
    
    
    endpoints1=endpoints[:,0:3]
    endpoints2=endpoints[:,3:7]
    #horzcat or vertcat?
    #i'm asuming it's 3xn
    twoPointStreams=[np.vstack((endpoints1[iStreams],endpoints2[iStreams])) for iStreams in range(len(streamlines)) ]
    
    endpointsAsStreams=Streamlines(twoPointStreams)

    return endpointsAsStreams 

 
    

def orientTractUsingNeck_multi(streamlines):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.metric import Feature
    from dipy.tracking.streamline import length
    from dipy.segment.metric import SumPointwiseEuclideanMetric
    from dipy.segment.metric import ResampleFeature
    from dipy.segment.metric import AveragePointwiseEuclideanMetric
    import numpy as np
   
    cluster=quickbundlesClusters(streamlines,nb_pts=50)

    
    for iBundles in range(len(cluster)):
        streamIndexes=list(cluster[iBundles].indices)
        orientedStreams=orientTractUsingNeck(streamlines[streamIndexes])
        for iStreams in range(len(streamIndexes)):
            if not np.all(streamlines[streamIndexes[iStreams]][0,:]==orientedStreams[iStreams][0,:]):
                streamlines[streamIndexes[iStreams]]= streamlines[streamIndexes[iStreams]][::-1]
                
    return streamlines

def trackStreamsInMask(targetMask,seed_density,wmMask,dwi,bvecs,bvals):
    """
    

    Returns
    -------
    streamlines : TYPE
        DESCRIPTION.

    """
    from dipy.core.geometry import dist_to_corner
    from dipy.core.gradients import gradient_table
    from dipy.data import get_fnames, default_sphere
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.image import load_nifti
    from dipy.reconst.csdeconv import (auto_response,
                                   auto_response_ssst,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst)
    import nibabel as nib
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    from dipy.data import default_sphere
    from dipy.data import small_sphere
    from dipy.direction import ProbabilisticDirectionGetter
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.reconst.shm import CsaOdfModel
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines
    from nilearn import image

    
    if isinstance(dwi,str):
        dwi=nib.load(dwi)
        
    if isinstance(bvecs,str) or  isinstance(bvals,str) :
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)


    gtab = gradient_table(bvals, bvecs)

    response, ratio = auto_response_ssst(gtab, dwi.get_data(), roi_radii=10, fa_thr=0.7)
    #response, ratio = auto_response(gtab, dwi.get_data(), roi_radii=10, fa_thr=0.7)
    
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    
    
   
    #no white matter mask for now?
    wmOnDwi=image.resample_to_img(wmMask,dwi,interpolation='nearest')
    
    csd_fit = csd_model.fit(dwi.get_data(), mask=wmOnDwi.get_data())

    # prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
    #                                                 max_angle=30.,
    #                                                 sphere=default_sphere)
    
    
    csa_model = CsaOdfModel(gtab, sh_order=6)
    gfa = csa_model.fit(dwi.get_data(), mask=wmOnDwi.get_data()).gfa
    stopping_thr= 0.2
    stopping_criterion = ThresholdStoppingCriterion(gfa, stopping_thr)
    
    fod = csd_fit.odf(default_sphere)
    pmf = fod.clip(min=0)
    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                    sphere=default_sphere)   

   
    seeds = utils.seeds_from_mask(targetMask.get_data(), targetMask.affine, density=seed_density)
    step_size= dist_to_corner(targetMask.affine)
    
    
    streamline_generator = LocalTracking(prob_dg, stopping_criterion, seeds,
                                     targetMask.affine, step_size=step_size)
    streamlines = Streamlines(streamline_generator)
    #tracking_method= "probabilistic"
    #use_binary_mask= False
    #stopping_thr= 0.2


    #pmf_threshold= 0.1
    #max_angle= 30.0
    #peaks= #somefilepath
    #stopping_files= #testdata/fa.nii.gz
    #seeding_files= #testdata/mask.nii.gz
  
    return streamlines

def quickbundlesClusters(streamlines, thresholds = [30,20,10,5], nb_pts=20,verbose=False):
    """
    (quickly?, via qbx_and_merge?) perform a quick-bundling of an input
    collection of streamlines, 
    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        The input streamlines 
    **kwargs : keyword arguments for the qbx_and_merge
        Currently only supports [thresholds] and [nb_pts]
        See dipy.segment.bundles.qbx_and_merge for more details
    Returns
    -------
    clusters : dipy.segment.clustering.ClusterMapCentroid
        The clusters resulting from the quickBundle-ification of the input 
        streamlines
    """
    from dipy.segment.bundles import qbx_and_merge
    from dipy.tracking.streamline import Streamlines
    
    #this didn't work, and was causing all kinds of problems
    # #fill in parameters if they are there.
    # if not 'thresholds' in locals():
    #     thresholds = [30,20,10,5]
    # if not 'nb_points' in locals():
    #     nb_pts=20
    # if not 'verbose' in locals():
    #     verbose=False
    #perform the quick, iterave bundling
    clusters=qbx_and_merge(streamlines,thresholds , nb_pts, select_randomly=None, rng=None, verbose=verbose)
    
    return clusters

def cullViaClusters(clusters,streamlines,streamThresh):
    import itertools
    import numpy as np
    #get the cluster lengths
    clusterLengths=[len(iCluster) for iCluster in clusters]
    
    #find which have meet the thresh criterion
    clustersSurviveThresh=np.greater(clusterLengths,streamThresh)
    
    #get a list of the clusters
    survivingClusters=list(itertools.compress(clusters,clustersSurviveThresh))
    #get the indexes of the streamlines from each
    survivingClusterLists=[iCluster.indices for iCluster in survivingClusters]
    #cat them all together
    survivingStreamsIndicies=list(itertools.chain(*survivingClusterLists))
    
    #find the obverse of the surviving stream set
    culledStreamIndicies=list(set(list(range(0,len(streamlines))))-set(survivingStreamsIndicies))

    return survivingStreamsIndicies, culledStreamIndicies

def cullStreamsByBundling(streamlines,streamThresh,qbThresholds=[30,20,10,5],qbResmaple=50):
    
    clusters=quickbundlesClusters(streamlines, thresholds=qbThresholds,nb_points=qbResmaple,verbose=True)
    
    survivingStreamsIndicies, culledStreamIndicies=cullViaClusters(clusters,streamlines,streamThresh)
    
    return survivingStreamsIndicies, culledStreamIndicies