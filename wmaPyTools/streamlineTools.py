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
    
    from nibabel.streamlines import ArraySequence
    from wmaPyTools.segmentationTools import _arraySequenceRowIndexes
    
    #dipy is stubborn and wants a reference nifti for some reason
    #fineI'llDoItMyself.jpg
    #Claude note: the bounds are taken from the input streamlines' own
    #nodes, not from streamlines._data, which for an indexed subset
    #(streamlines[idx]) is the *parent* tractogram's entire node buffer; the
    #dummy grid's origin and extent (and hence density binning, e.g. the neck
    #found by findTractNeckNode) must not depend on unrelated streamlines.
    if isinstance(streamlines, ArraySequence):
        rowIndexes, lengths = _arraySequenceRowIndexes(streamlines)
        tractNodes=streamlines._data[rowIndexes]
    else:
        tractNodes=np.concatenate([np.asarray(iStream) for iStream in streamlines])
    tractBounds=np.asarray([np.min(tractNodes,axis=0),np.max(tractNodes,axis=0)])
    #tractBounds2=np.asarray([np.min(streamlines.data,axis=0),np.max(streamlines.data,axis=0)])
    roundedTractBounds=np.asarray([np.floor(tractBounds[0,:]),np.ceil(tractBounds[1,:])])
    constructedAffine=np.eye(4)
    constructedAffine[0:3,3]=tractBounds[0,:]

    #Claude note: a call here mapped every node of _data to voxel coordinates
    #and discarded the result (its only possible effect, an error for
    #negative coordinates, can't occur with the origin at the minimum), so it
    #has been removed
        
    #testBounds=np.asarray([np.min(inds,axis=0),np.max(inds,axis=0)])
        
    #now create a dummy nifit, because that's what dipy demands
    dataShape=(roundedTractBounds[1,:]-roundedTractBounds[0,:]).astype(int)
    #adding a +1 pad because it yells otherwise?
    #should I pad here?
    dummyData=np.zeros(dataShape+1)
    dummyNifti= nib.nifti1.Nifti1Image(dummyData, constructedAffine)
    return dummyNifti

def tractNeckSection(streamlines, minCoverage=0.9, tolerance=1.1, sectionHalfWidthMM=0.75, maxSectionRadiusMM=15.0, nCoreCandidates=200, seed=0):
    """
    Claude note: locates a tract's neck from its measured
    cross-sections, rather than from a density peak (see findTractNeckNode,
    method='density').

    Procedure
    ---------
    1. Core: a real streamline, the medoid (by MDF) of a sample of the longest
       quartile, smoothed and resampled to 1 mm.  Being a real streamline it is
       central and spans the tract even when many streamlines are truncated.
       (Re-centring the core on its own sections - a principal curve - was
       tried and is unstable: it grew 125 -> 257 mm on AF in 5 iterations.)
    2. Curved sections: each node belongs to its nearest core point, and counts
       only within sectionHalfWidthMM of it ALONG the core and within
       maxSectionRadiusMM of it ACROSS.  Sections are thin slices perpendicular
       to the local course; they follow curved tracts, never overlap, and
       cannot cut a curved tract twice (flat planes did both).
    3. Per section: how many streamlines pass through it, and its radius (the
       median distance of the per-streamline section centroids from their
       median; robust to off-centre streamlines).
    4. Neck: among sections that at least minCoverage of the streamlines pass
       through, take the narrowest, then the contiguous stretch around it whose
       radius is within `tolerance` of that minimum, and choose the MIDDLE of
       that stretch.  A long uniform stretch (e.g. the arcuate's stem) has no
       single narrowest point; its middle is the principled choice.

    Priors (parameters): minCoverage=0.9 and tolerance=1.1.  On HCP 100206
    (TractSeg TOM tracks) this places the neck at the visible waist of UF,
    IFO, CG and CST (internal capsule, where the density method chose the
    brainstem tip, reached by ~2/3 of streamlines), and mid-stem on AF.  On
    CC_7 it lands ~6 mm right of the midline: the callosal body is uniformly
    narrow, and cross-section alone cannot single out the midline.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
        The streamlines of a single tract.  Orientation does not matter.
    minCoverage : float, optional
        Fraction of streamlines a section must contain to be a neck candidate.
        The default is 0.9.
    tolerance : float, optional
        Sections within this factor of the minimum radius belong to the narrow
        stretch.  The default is 1.1.
    sectionHalfWidthMM, maxSectionRadiusMM : float, optional
        Section thickness (half) along the core, and reach across it.
    nCoreCandidates : int, optional
        Streamlines sampled when choosing the medoid core.
    seed : int, optional
        Seed for that sample, so results are reproducible.

    Returns
    -------
    neckNodes : numpy.ndarray of int
        For each streamline, the index of its node in the neck section (its
        node closest to the neck's core point if it never enters the section).
    traversing : numpy.ndarray of bool
        Whether each streamline passes through the neck section.
    neckInfo : dict
        'core' (N x 3, 1 mm), 'neckIndex', 'narrowestIndex', 'stretch'
        (start, end core indexes), 'count' and 'radius' per section.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from dipy.segment.metric import mdf
    from dipy.tracking.streamline import Streamlines, set_number_of_points

    streams=[np.asarray(iStreamline,dtype=float) for iStreamline in streamlines]

    def cumulativeArc(points):
        return np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))])

    def resampleByArc(points,step=1.0):
        arcLength=cumulativeArc(points)
        samples=np.arange(0,arcLength[-1],step)
        return np.column_stack([np.interp(samples,arcLength,points[:,iDim]) for iDim in range(3)])

    def smoothCurve(points,width=9):
        kernel=np.ones(width)/width
        padded=np.vstack([np.repeat(points[:1],width//2,0),points,np.repeat(points[-1:],width//2,0)])
        return np.column_stack([np.convolve(padded[:,iDim],kernel,mode='valid') for iDim in range(3)])

    #Claude note: 1. the core
    rng=np.random.default_rng(seed)
    lengths=np.array([cumulativeArc(iStream)[-1] for iStream in streams])
    longest=np.where(lengths>=np.percentile(lengths,75))[0]
    candidates=rng.choice(longest,min(nCoreCandidates,len(longest)),replace=False)
    resampled=[np.asarray(set_number_of_points(Streamlines([streams[iIndex]]),50)[0]) for iIndex in candidates]
    totalDistance=[sum(mdf(iCandidate,jCandidate) for jCandidate in resampled) for iCandidate in resampled]
    core=resampleByArc(smoothCurve(resampleByArc(streams[candidates[int(np.argmin(totalDistance))]])))
    tangent=np.gradient(core,axis=0)
    tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    coreTree=cKDTree(core)

    def assign(nodes):
        distance,section=coreTree.query(nodes)
        along=np.einsum('ij,ij->i',nodes-core[section],tangent[section])
        return section,along,(np.abs(along)<=sectionHalfWidthMM)&(distance<=maxSectionRadiusMM)

    #Claude note: sections are 2*sectionHalfWidthMM thick, so nodes must be
    #closer than that or whole sections come out empty (and streamlines skip
    #the neck section).  Work on copies densified to <=0.5 mm, and subsample
    #fine tractography steps (e.g. 0.125 mm) down to ~0.5 mm for the profile.
    stepMM=np.median([np.median(np.linalg.norm(np.diff(iStream,axis=0),axis=1)) for iStream in streams if len(iStream)>1])
    if stepMM>0.5:
        dense=[resampleByArc(iStream,0.5) if len(iStream)>1 else iStream for iStream in streams]
        stride=1
    else:
        dense=streams
        stride=max(1,int(0.5/max(stepMM,1e-6)))

    #Claude note: 2-3. section profile
    subsampled=[iStream[::stride] for iStream in dense]
    owner=np.concatenate([np.full(len(iSub),iIndex) for iIndex,iSub in enumerate(subsampled)])
    nodes=np.vstack(subsampled)
    section,along,inside=assign(nodes)
    key=owner[inside]*len(core)+section[inside]
    uniqueKeys,inverse=np.unique(key,return_inverse=True)
    sums=np.zeros((len(uniqueKeys),3))
    np.add.at(sums,inverse,nodes[inside])
    centroids=sums/np.bincount(inverse)[:,None]
    sectionOf=uniqueKeys%len(core)
    count=np.bincount(sectionOf,minlength=len(core))
    radius=np.full(len(core),np.nan)
    for iSection in range(len(core)):
        members=sectionOf==iSection
        if members.sum()>=20:
            radius[iSection]=np.median(np.linalg.norm(centroids[members]-np.median(centroids[members],axis=0),axis=1))
    if np.all(np.isnan(radius)):
        raise ValueError('tractNeckSection: no section contains 20 streamlines; too few streamlines to measure cross-sections')
    #Claude note: pad with edge values; zero padding made the end sections look narrow
    radius=np.convolve(np.pad(np.nan_to_num(radius,nan=np.nanmax(radius)),1,mode='edge'),np.ones(3)/3,mode='valid')

    #Claude note: 4. the neck: middle of the narrow, well-covered stretch
    covered=count>=minCoverage*len(streams)
    if not covered.any():
        import warnings
        warnings.warn('tractNeckSection: no section contains '+str(minCoverage)+' of the streamlines; using sections with at least half the maximum coverage')
        covered=count>=0.5*count.max()
    narrowest=int(np.where(covered)[0][np.argmin(radius[covered])])
    nearMinimum=covered&(radius<=tolerance*radius[narrowest])
    stretchStart=narrowest
    while stretchStart-1>=0 and nearMinimum[stretchStart-1]:
        stretchStart-=1
    stretchEnd=narrowest
    while stretchEnd+1<len(nearMinimum) and nearMinimum[stretchEnd+1]:
        stretchEnd+=1
    neckIndex=(stretchStart+stretchEnd)//2

    #Claude note: per-streamline neck node: its node in the neck section closest to the
    #section's centre plane; otherwise its node nearest the neck's core point
    neckNodes=np.zeros(len(streams),dtype=int)
    traversing=np.zeros(len(streams),dtype=bool)
    for iIndex,(iStream,iDense) in enumerate(zip(streams,dense)):
        iSection,iAlong,iInside=assign(iDense)
        inNeck=np.where(iInside&(iSection==neckIndex))[0]
        if len(inNeck):
            neckPoint=iDense[inNeck[np.argmin(np.abs(iAlong[inNeck]))]]
            traversing[iIndex]=True
        else:
            neckPoint=core[neckIndex]
        #Claude note: map back to the ORIGINAL streamline's nodes
        neckNodes[iIndex]=int(np.argmin(np.linalg.norm(iStream-neckPoint,axis=1)))
    neckInfo={'core':core,'neckIndex':neckIndex,'narrowestIndex':narrowest,'stretch':(stretchStart,stretchEnd),'count':count,'radius':radius}
    return neckNodes,traversing,neckInfo

def findTractNeckNode(streamlines, refAnatT1=None, method='density'):
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

    # Claude note: method='density' (default, original) takes the peak of the
    # smoothed squared streamline-point density and gives each streamline its
    # node nearest that single point.  method='section' uses tractNeckSection
    # (measured cross-sections; middle of the narrowest well-covered stretch),
    # which fixes necks the density peak misplaces (e.g. CST: brainstem tip
    # instead of the internal capsule).  The default stays 'density' until the
    # orientation / outlier / bundle-test code has been checked with it.
    if method=='section':
        return tractNeckSection(streamlines)[0]
    elif method!='density':
        raise ValueError("method must be 'density' or 'section', got "+repr(method))

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
    #Claude note: was computedStDev=muVar, i.e. log(scale), the log-space
    #*mean* rather than any standard deviation.  It depends on the distance
    #unit and goes negative whenever the median distance is under 1 mm, so
    #the cut fell below the mean and most of the tract was removed (measured
    #2026-11-11 on the 72 TractSeg TOM tracts, density neck, cutStDev=2:
    #1.4-95% removed; FX_right 95%, and 100% at cutStDev=3).  The distribution
    #actually fitted is a lognormal (shape sSigma, scale exp(muVar)); its
    #standard deviation, from the same MathWorks page as the mean above, is
    #sqrt(exp(2*muVar+sSigma^2)*(exp(sSigma^2)-1)) = computedMean*sqrt(exp(sSigma^2)-1).
    #That keeps cutStDev meaning "standard deviations above the mean", as its
    #name says.  With it, cutStDev=2 removes 0-10% (median 2.5%) on those
    #tracts, and the fitted lognormal itself puts 3.9-4.5% beyond the cut.
    #(Alternative, not chosen: a log-space cut exp(muVar+cutStDev*sSigma),
    #which removes less, median 0.3%, and changes what cutStDev counts.)
    computedStDev=computedMean*np.sqrt(np.exp(np.square(sSigma))-1)
    
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
    from dipy.segment.featurespeed import ResampleFeature
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
    from dipy.segment.featurespeed import ResampleFeature
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
    Claude note: orients each QuickBundles sub-bundle independently
    (orientDipyCluster); sub-bundles can disagree with each other, and on a
    whole tractogram this per-cluster approach leaves 18.6% of streamlines
    locally reversed after segmentation.  For a tract, prefer
    orientTractUsingCentroids or orientTractUsingNeck_Robust.  See docs/orientationMethods.md for
    measured tradeoffs between the orientation methods.
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.featurespeed import Feature
    from dipy.tracking.streamline import length
    from dipy.segment.metric import SumPointwiseEuclideanMetric
    from dipy.segment.featurespeed import ResampleFeature
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
    #Claude note: was wmaPyTools.roiTools.dummyNiftiForStreamlines, which doesn't
    #exist (AttributeError); the function lives in this module
    dummyNifti=dummyNiftiForStreamlines(cluster.refdata)
    
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
    except ImportError:
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
            #Claude note: KeyError only: a voxel with no streamlines simply isn't in the mapping
            except KeyError:
                pass
        streamIndexes=np.unique(streamIndexes)
        return streamIndexes
        
    #now do it many times in a parallel pool 
    #processesCount=os.cpu_count()
    #processes_pool = Pool(processesCount)
    #Claude note: timed with genUtils.StageTimer (wall-clock; silenced by
    #genUtils.setTimingVerbose(False)) instead of time.process_time
    import wmaPyTools.genUtils
    stageTimer=wmaPyTools.genUtils.StageTimer('find streamline groups')
    #streamsListList=processes_pool.map(partial(streamsInImgSpaceWindow,streamlineMapping=streamlineMapping,mask_r=mask_r), imgCoords)
    print('computing voxel-wise collections of streamlines to check for while using a ' + str(distanceParameter) + 'mm spherical window')
    if tqdmFlag:
        streamsListList=[streamsInImgSpaceWindow(iCoords,streamlineMapping,mask_r) for iCoords in tqdm.tqdm(imgCoords, position=0, leave=True)]
    else:
        streamsListList=[streamsInImgSpaceWindow(iCoords,streamlineMapping,mask_r) for iCoords in imgCoords]
  
    stageTimer.finish(lambda modifiedTime: str(modifiedTime) + ' seconds to find streamline groups to check')  
    
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
    #Claude note: force to an array.  matWMC2dict returns 'index' as a python list,
    #and list==int is simply False (not elementwise), so previously no
    #streamlines were ever found for any tract when a .mat path was passed
    indices=np.asarray(classification['index'])
    #get the names vector
    tractIdentities=classification['names']
    
    for tractID in range(len(tractIdentities)):
        #remove unncessary characters, adds unnecessary '[]'
        tract_name = tractIdentities[tractID]
        idx_tract = np.array(np.where(indices==tractID+1))[0]
        
        #save it in the same format as the input
        if  isinstance(inputTractogram, nib.streamlines.trk.TrkFile):
            out_filename=os.path.join(outdir,tract_name + '.trk')
        #Claude note: .tck for tck input, and also as the default for already-loaded
        #streamlines (previously out_filename was simply undefined then)
        else:
            out_filename=os.path.join(outdir,tract_name + '.tck')
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
    #Claude note: flatten rather than [0]: MATLAB may save index as either a 1xN row or an
    #Nx1 column vector; [0] silently kept only the first element of the latter
    indices=classification['classification'][0][0]['index'].flatten()
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
    wmc_Dict['names']=np.array(wmc_Dict['names'], dtype=object) 
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
    
    wmc_Dict['names']=np.array(wmc_Dict['names'], dtype=object)        
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
    #Claude note: flatten rather than [0]: MATLAB may save index as either a 1xN row or an
    #Nx1 column vector; [0] silently kept only the first element of the latter
    indices=classification['classification'][0][0]['index'].flatten()
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
    Claude note: not recommended: its flip decision is position-based
    (42-48% against consensus on synthetic bundles).  Use
    orientTractUsingNeck_Robust or orientTractUsingCentroids.  See docs/orientationMethods.md for
    measured tradeoffs between the orientation methods.

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
    #Claude note: findTractNeckNode returns a flat vector of ints (either
    #method); the code below used to index it as a column ((n,1) array) with
    #neckNodes[i][0] and (...)[0], which raised IndexError ("invalid index to
    #scalar variable") on every call, and so also broke
    #orientTractUsingNeck_multi.  Flatten once here and index it as a vector.
    neckNodes=np.asarray(neckNodes).ravel()
    
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
        if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)>lookDistance):
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+lookDistance
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-lookDistance
       #otherwise do the best you can
        else:
           #if there's a limit to how many nodes are available ahead, do the best you can
           spaceAhead=np.abs(neckNodes[iStreamlines]-len(streamlines[iStreamlines]))
           if spaceAhead<=lookDistance:
               aheadWindow=spaceAhead-1
           else:
               aheadWindow=lookDistance
           #if there's a limit to how many nodes are available behind, do the best you can
           spaceBehind=np.abs(len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))
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
                if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)>lookDistance):
                   aheadNodes[iStreamlines]=neckNodes[iStreamlines]+lookDistance
                   behindNodes[iStreamlines]=neckNodes[iStreamlines]-lookDistance
               #otherwise do the best you can
                else:
                   #if there's a limit to how many nodes are available ahead, do the best you can
                   spaceAhead=np.abs(neckNodes[iStreamlines]-len(streamlines[iStreamlines]))
                   if spaceAhead<=lookDistance:
                       aheadWindow=spaceAhead-1
                   else:
                       aheadWindow=lookDistance
                   #if there's a limit to how many nodes are available behind, do the best you can
                   spaceBehind=np.abs(len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))
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
                #Claude note: reversing a streamline moves the node at index k to
                #index len-1-k.  Previously the two indexes were only swapped, which
                #left them pointing at the wrong nodes (in the reversed streamline) unless
                #the neck was exactly mid-streamline; on the next pass those streamlines
                #looked misoriented again and were flipped back, forever (the "algorithm
                #stuck" loop).  The node behind the neck in the new orientation is the
                #reflection of the old ahead node, and vice versa.
                #(the coordinates above are physically correct as swapped; only
                #the indexes need re-expressing in the reversed numbering)
                lastIndex=len(streamlines[iStreamlines])-1
                aheadNodes[iStreamlines]=lastIndex-aheadNodes[iStreamlines]
                behindNodes[iStreamlines]=lastIndex-behindNodes[iStreamlines]
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
    #Claude note: findTractNeckNode returns a flat int vector now; the [0]
    #indexing below was written for an older per-streamline array output and
    #crashed (IndexError), which made orientTractUsingNeck_Robust unusable.
    neckNodes=np.asarray(neckNodes).ravel()
    for iStreamlines in range(len(streamlines)):

       #A check to make sure you've got room to do this indexing on both sides
        if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)>lookDistance):
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+lookDistance
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-lookDistance
       #otherwise do the best you can
        else:
           #if there's a limit to how many nodes are available ahead, do the best you can
           spaceAhead=np.abs(neckNodes[iStreamlines]-len(streamlines[iStreamlines]))
           if spaceAhead<=lookDistance:
               aheadWindow=spaceAhead-1
           else:
               aheadWindow=lookDistance
           #if there's a limit to how many nodes are available behind, do the best you can
           spaceBehind=np.abs(len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))
           if spaceBehind<=lookDistance:
               behindWindow=spaceBehind-1
           else:
               behindWindow=lookDistance
           
           #append the relevant values
           aheadNodes[iStreamlines]=neckNodes[iStreamlines]+(aheadWindow)
           behindNodes[iStreamlines]=neckNodes[iStreamlines]-(behindWindow)
           
    return aheadNodes, behindNodes

def orientNonTraversingByEndpoints(streamlines,neckNodes,lookDistance,minSideFraction=0.15,surpressReport=False):
    """
    (Claude note.)  Orients the streamlines of a (mostly)
    already-oriented tract that do not pass through the tract's neck, but
    instead end at (or near) it, using where their endpoints land.
    
    The traversing streamlines (those with at least minSideFraction of their
    nodes, and at least lookDistance nodes, on each side of their neck node)
    define two terminal clouds: their first
    nodes (cloud A) and their last nodes (cloud B).  Each non-traversing
    streamline is flipped if its endpoints sit closer to the clouds in the
    reversed order, measured as the summed distance from each endpoint to the
    NEAREST point of the corresponding cloud (nearest-neighbour rather than
    centroid, so that fanning terminations are handled).  For a branch whose
    one end is at the neck, the far end effectively decides.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
        The tract, already oriented by a neck-based method.  Modified in place
        and also returned.
    neckNodes : array-like of int
        The neck node index of each streamline, in its CURRENT orientation.
    lookDistance : int
        Minimum number of nodes required on each side of the neck for a
        streamline to count as traversing it.
    minSideFraction : float, optional
        Minimum fraction of a streamline's nodes that must lie on EACH side of
        its neck node for it to count as traversing the neck.  This is a prior:
        on HCP 100206 AF_left, branch streamlines that end at the neck had it
        within the first ~10% of their length, ordinary streamlines at >=20%.
        The default is 0.15.
    surpressReport : bool, optional
        If False, prints how many streamlines were re-oriented.  The default
        is False.

    Returns
    -------
    streamlines : same type as input
        The streamlines, with non-traversing ones re-oriented as needed.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    neckNodes=np.asarray(neckNodes).ravel()
    lengths=np.array([len(iStreamline) for iStreamline in streamlines])
    #Claude note: a streamline traverses the neck if it has a meaningful share of its
    #length on both sides of it
    minSideNodes=np.maximum(lookDistance,np.ceil(minSideFraction*(lengths-1)))
    traversing=np.logical_and(neckNodes>=minSideNodes,(lengths-1-neckNodes)>=minSideNodes)
    #Claude note: nothing to do if all traverse, and no reference if none do
    if np.all(traversing) or not np.any(traversing):
        return streamlines
    #Claude note: the two terminal clouds of the traversing majority
    cloudA=cKDTree(np.array([streamlines[iIndex][0] for iIndex in np.where(traversing)[0]]))
    cloudB=cKDTree(np.array([streamlines[iIndex][-1] for iIndex in np.where(traversing)[0]]))
    reorientedCount=0
    for iIndex in np.where(~traversing)[0]:
        firstNode=streamlines[iIndex][0]
        lastNode=streamlines[iIndex][-1]
        currentFit=cloudA.query(firstNode)[0]+cloudB.query(lastNode)[0]
        flippedFit=cloudA.query(lastNode)[0]+cloudB.query(firstNode)[0]
        if flippedFit<currentFit:
            streamlines[iIndex]=streamlines[iIndex][::-1]
            reorientedCount=reorientedCount+1
    if not surpressReport:
        print(str(np.sum(~traversing))+' streamlines do not traverse the neck; '+str(reorientedCount)+' re-oriented by endpoint location')
    return streamlines

def orientTractUsingNeck_Robust(streamlines,refAnatT1=None,surpressReport=False):
    """orientTractUsingNeck(streamlines)
    Claude note: on 72 TractSeg tracts, locally reversed mean 0.52%, end-to-end
    opposite mean 0.79%, 3.5-7 s per tract; fails where the neck sits near one
    end of the tract (e.g. ST_OCC_right).  orientTractUsingCentroids is faster
    and needs no neck, and fails on different tracts.  See docs/orientationMethods.md for
    measured tradeoffs between the orientation methods.

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
    from dipy.segment.featurespeed import ResampleFeature
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
    
    #Claude note: a dedicated exception for the deliberate "split into
    #subgroups and start over" jump below.  Previously a bare Exception was
    #raised and *every* exception was caught and retried, so any real error
    #(or an external timeout) produced an infinite loop.
    class _SplitIntoSubgroups(Exception):
        pass
    #Claude note: hard cap on total passes, so a non-converging input warns
    #instead of hanging
    maxTotalPasses=500
    totalPasses=0
    neckNodes=np.asarray(neckNodes).ravel().copy()
    
    #Claude note: reflect the neck index of every streamline that was reversed
    #(the node at index k moves to len-1-k), so later ahead/behind windows are
    #computed around the right node
    def reflectNecksOfFlipped(subsetIndexes,firstNodesBefore):
        for iterator,iIndex in enumerate(subsetIndexes):
            if not np.array_equal(np.asarray(streamlines[iIndex][0]),firstNodesBefore[iterator]):
                neckNodes[iIndex]=len(streamlines[iIndex])-1-neckNodes[iIndex]
    
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
                        
                        raise _SplitIntoSubgroups
                        
                    
                  
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
                                #Claude note: keep the neck index in step with the reversal
                                iIndex=list(tryTheseIndexes[iSubsets])[iStreamlines]
                                neckNodes[iIndex]=len(streamlines[iIndex])-1-neckNodes[iIndex]
                        
                        [aheadNodes, behindNodes]=findAheadAndBehindNodes(streamlines[tryTheseIndexes[iSubsets]],neckNodes[tryTheseIndexes[iSubsets]],lookDistance)
                       
                        
                       
                    # use the coords that are at the heart of the tract to establish the orientation guide
                    
                    #Claude note: guard against non-convergence
                    totalPasses=totalPasses+1
                    if totalPasses>maxTotalPasses:
                        import warnings
                        warnings.warn('orientTractUsingNeck_Robust WARNING: orientation did not converge within '+str(maxTotalPasses)+' passes; returning the current (possibly inconsistent) orientation')
                        return streamlines
                    subsetIndexes=list(tryTheseIndexes[iSubsets])
                    firstNodesBefore=[np.array(streamlines[iIndex][0]) for iIndex in subsetIndexes]
                    [streamlines[tryTheseIndexes[iSubsets]], aheadNodes, behindNodes, flipCount]=flipStreamstoAB_OrientOnce(streamlines[tryTheseIndexes[iSubsets]], aheadNodes, behindNodes,surpressReport=True)
                    reflectNecksOfFlipped(subsetIndexes,firstNodesBefore)
                    if not surpressReport:
                        print('flip run ' + str(flipRuns) +': ' + str(flipCount) + ' of ' + str(len(streamlines)) + ' streamlines flipped')
                
                    if np.all([flipRuns!=1,flipCount==0]) :
                        exitTriggerOne=True
                    else:
                      #set the counters
                          flipRuns=flipRuns+1
                          flipCount=0
        except _SplitIntoSubgroups:
            pass
        
        if np.all([flipRuns!=1,flipCount==0, iSubsets==len(tryTheseIndexes)-1]) :
            exitTriggerAll=True
        else:
            #set the counters
            flipRuns=flipRuns+1
            flipCount=0
            
    #Claude note: streamlines that END at the neck rather than passing
    #through it (e.g. a branch leaving the tract at its bend) have no
    #well-defined crossing direction, so the neck rule above orients them by
    #the direction of a streamline end, i.e. arbitrarily.  Re-orient those by
    #where their endpoints land relative to the traversing majority's two
    #terminal clouds.  (On HCP 100206 AF_left: 42 of 1500, a coherent superior
    #branch.)
    streamlines=orientNonTraversingByEndpoints(streamlines,neckNodes,lookDistance,surpressReport=surpressReport)
    
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
    if not surpressReport:
        print('coord corespondances extracted')
    #Claude note: the guide is the principal axis of all the streamlines'
    #own local directions at the neck.  A principal axis is sign-free, so it
    #is the same whichever way each streamline currently runs (points picked
    #per side nearest the mean coords would usually belong to different
    #streamlines, and when about half are reversed the guide would encode the
    #offset between two streamlines rather than the tract's direction); its
    #sign is then set by the rule below.
    #Claude note: the local direction of each streamline at its neck (behind
    #point -> ahead point)
    localDirections=np.asarray(aheadCoords,dtype=float).reshape(len(streamlines),-1)-np.asarray(behindCoords,dtype=float).reshape(len(streamlines),-1)
    #Claude note: their lengths, so they can be made unit vectors (each
    #streamline then counts equally, whatever its node spacing)
    localDirectionNorms=np.linalg.norm(localDirections,axis=1)
    #Claude note: streamlines whose two neck points coincide carry no direction
    hasDirection=localDirectionNorms>0
    unitDirections=localDirections[hasDirection]/localDirectionNorms[hasDirection,None]
    if len(unitDirections)==0:
        #Claude note: nothing to orient by; a zero guide flips nothing
        guideDirection=np.zeros(localDirections.shape[1])
    else:
        #Claude note: the principal axis is the first right singular vector of
        #the (uncentred) stack of unit directions: the axis along which they
        #line up, regardless of their signs
        guideDirection=np.linalg.svd(unitDirections,full_matrices=False)[2][0]
    if not surpressReport:
        print('orientation guide computed from the principal axis of local directions')
    
    streamTraversals=[wmaPyTools.analysisTools.cumulativeTraversalStream(istreamlines) for istreamlines in streamlines]
    avgTraversals=np.mean(streamTraversals,axis=0)
    
    
    #if you wanted to force RAS / LPI on your tractogram orientatiuon, now would be the time to do it
    #DISPALCEMENT IS ACTUALLY BAD FOR THIS, WHAT YOU WANT IS TRAVERSAL
    #find the dimension of greatest average traversal
    maxTraversalDim=np.where(np.max(avgTraversals)==avgTraversals)[0][0]
    #if the guide is negative along it, reverse the guide (the sign rule is
    #unchanged: the guide points positive along the dimension of greatest
    #average traversal)
    if guideDirection[maxTraversalDim]<0:
        guideDirection=-guideDirection
    
    #print(str(orientationGuideAheadCoord))
    #print(str(orientationGuideBehindCoord))
    if not surpressReport:
        print('orentation nodes specified')
    #iterate across streamlines     
    for iStreamlines in range(len(streamlines)):
        #compute the distances from the comparison orientation for both possible
        #orientations
        #Claude note: the decision compares the local *direction* (behind
        #point -> ahead point) with the guide's direction, not positions: in
        #real tracts the cross-section is many mm wide, so position offsets
        #swamp the ~2*lookDistance mm that encodes direction.  Direction is
        #unaffected by how far apart streamlines are.
        #Claude note: the guide direction is computed once, above
        localDirection=localDirections[iStreamlines]
        #flip if necessary
        if np.dot(localDirection,guideDirection)<0:
            #print(str(iStreamlines))
            #print(str(sumDistanceOrientation1))
            #print(str(sumDistanceOrientation2))
            
            #flip the nodes and coords if necessary
            currentAheadNode=copy.deepcopy(aheadNodes[iStreamlines])
            currentBehindNode=copy.deepcopy(behindNodes[iStreamlines])
            #Claude note: reversing a streamline moves the node at index k to
            #index len-1-k.  Previously the two indexes were only swapped, which
            #left them pointing at the wrong nodes (in the reversed streamline) unless
            #the neck was exactly mid-streamline; on the next pass those streamlines
            #looked misoriented again and were flipped back, forever (the "algorithm
            #stuck" loop).  The node behind the neck in the new orientation is the
            #reflection of the old ahead node, and vice versa.
            lastIndex=len(streamlines[iStreamlines])-1
            aheadNodes[iStreamlines]=lastIndex-currentBehindNode
            behindNodes[iStreamlines]=lastIndex-currentAheadNode
            #aheadCoords[iStreamlines]=streamlines[iStreamlines][aheadNodes[iStreamlines]]
            #behindCoords[iStreamlines]=streamlines[iStreamlines][behindNodes[iStreamlines]]
            
            streamlines[iStreamlines]= streamlines[iStreamlines][::-1]
            flipCount=flipCount+1
    return streamlines, aheadNodes, behindNodes, flipCount

def _orientAllStreamlinesVectorised(tractStreamlines):
    """
    Claude note: vectorised core of orientAllStreamlines, for nibabel
    ArraySequence / dipy Streamlines input.  Computes, for every streamline
    at once, exactly what the per-streamline loop computes (the same float
    arithmetic: |node differences| summed sequentially per dimension, in the
    streamlines' own dtype), and reverses, in place, the streamlines whose
    first endpoint has the lower coordinate in their dimension of maximal
    traversal.

    Returns
    -------
    flipCount : int, or None
        The number of streamlines flipped, or None if the input was not
        handled here (not an ArraySequence, streamlines with fewer than 2
        nodes, NaNs, or streamlines sharing rows of data), in which case the
        input is left untouched.
    """
    import numpy as np
    from nibabel.streamlines import ArraySequence
    from wmaPyTools.segmentationTools import _arraySequenceRowIndexes, _streamlineChunks
    
    if not isinstance(tractStreamlines, ArraySequence) or len(tractStreamlines)==0:
        return None
    lengths=np.asarray(tractStreamlines._lengths,dtype=np.intp)
    if np.any(lengths<2) or tractStreamlines._data.ndim!=2:
        return None
    #Claude note: streamlines sharing rows would see each other's flips in the loop
    offsetOrder=np.argsort(tractStreamlines._offsets,kind='stable')
    sortedOffsets=np.asarray(tractStreamlines._offsets)[offsetOrder]
    sortedEnds=sortedOffsets+lengths[offsetOrder]
    if np.any(sortedOffsets[1:]<sortedEnds[:-1]):
        return None
    #Claude note: decide every flip first (in bounded-memory chunks, see
    #_streamlineChunks), so that nothing is modified if a NaN sends us back
    #to the original loop
    chunkFlips=[]
    for firstIndex, chunk in _streamlineChunks(tractStreamlines):
        rowIndexes, chunkLengths = _arraySequenceRowIndexes(chunk)
        nodes=tractStreamlines._data[rowIndexes]
        if np.any(np.isnan(nodes)):
            return None
        starts=np.cumsum(chunkLengths)-chunkLengths
        #Claude note: absolute node-to-node differences, dropping those spanning two streamlines
        absDeltas=np.abs(np.diff(nodes,axis=0))
        withinStream=np.ones(len(absDeltas),dtype=bool)
        withinStream[(starts[1:]-1)]=False
        absDeltas=absDeltas[withinStream]
        #Claude note: each streamline contributes lengths-1 differences, summed sequentially
        #per dimension, as np.sum(deltas,axis=0) does
        deltaStarts=starts-np.arange(len(chunkLengths))
        traversals=np.add.reduceat(absDeltas,deltaStarts,axis=0)
        if np.any(np.isnan(traversals)):
            return None
        #Claude note: find which dimension has max traversal (the first, in case of ties)
        maxTraversalDim=np.argmax(traversals,axis=1)
        #get the current endpoints
        endpoint1=nodes[starts,maxTraversalDim]
        endpoint2=nodes[starts+chunkLengths-1,maxTraversalDim]
        chunkFlips.append(np.where(endpoint1<endpoint2)[0]+firstIndex)
    toFlip=np.concatenate(chunkFlips)
    #Claude note: reverse the rows of each streamline to be flipped, in place
    offsets=np.asarray(tractStreamlines._offsets,dtype=np.intp)
    for iFlip in range(0,len(toFlip),100000):
        flipIndexes=toFlip[iFlip:iFlip+100000]
        flipLengths=lengths[flipIndexes]
        flipOffsets=offsets[flipIndexes]
        withinFlipped=np.arange(int(np.sum(flipLengths)))-np.repeat(np.cumsum(flipLengths)-flipLengths,flipLengths)
        targetRows=np.repeat(flipOffsets,flipLengths)+withinFlipped
        sourceRows=np.repeat(flipOffsets+flipLengths-1,flipLengths)-withinFlipped
        tractStreamlines._data[targetRows]=tractStreamlines._data[sourceRows]
    return int(len(toFlip))

def orientAllStreamlines(tractStreamlines):
    """
    Claude note: orients each streamline on its own; very fast, fine on
    straight tracts, but on 72 TractSeg tracts 19 had more than 5% of
    streamlines locally reversed (curved, hooked, commissural tracts).  For a
    tract, prefer orientTractUsingCentroids or orientTractUsingNeck_Robust.  See docs/orientationMethods.md for
    measured tradeoffs between the orientation methods.
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
    except ImportError:
        tqdmFlag=False
    import wmaPyTools.analysisTools
    
    #create a counter, for fun
    flipCount=0
    #Claude note: for a Streamlines / ArraySequence input, the traversals,
    #flip decisions and flips are computed for all streamlines at once (see
    #_orientAllStreamlinesVectorised), with identical results.  Other inputs
    #(and degenerate ones) go through the original per-streamline loop.
    vectorisedFlipCount=_orientAllStreamlinesVectorised(tractStreamlines)
    if vectorisedFlipCount is not None:
        flipCount=vectorisedFlipCount
    elif tqdmFlag:
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
    Claude note: orients each QuickBundles sub-bundle independently with
    orientTractUsingNeck, so sub-bundles can disagree with each other; not
    recommended for orienting a tract (use orientTractUsingCentroids).  See docs/orientationMethods.md for
    measured tradeoffs between the orientation methods.
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.featurespeed import Feature
    from dipy.tracking.streamline import length
    from dipy.segment.metric import SumPointwiseEuclideanMetric
    from dipy.segment.featurespeed import ResampleFeature
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

def _orientCentroidsByTangentAgreement(centroids, clusterSizes, radiusMM=4.0, minAbsCos=0.5, maxNeighbours=16, excludeFolded=True, minFoldSeparationMM=10.0, foldRadiusMM=8.0):
    """
    Claude note: decides, for each QuickBundles centroid,
    whether it must be reversed so that all centroids of a tract run the same
    way.  Used by orientTractUsingCentroids; see its docstring for the method
    and its tradeoffs.

    Wherever two centroids run close to each other (within radiusMM) and
    roughly parallel (|cos| >= minAbsCos between their local directions),
    the sign of the dot product of their local directions says whether they
    currently run the same way (+) or opposite ways (-).  These signs are
    summed per pair of centroids.  The centroids are then oriented greedily,
    like growing a spanning tree: start from the largest cluster, and
    repeatedly take the not-yet-oriented centroid with the largest
    size-weighted overlap with the already-oriented ones, reversing it if its
    size-weighted agreement with them is negative.  When no unoriented
    centroid overlaps the oriented ones within radiusMM (an isolated
    outlier, or a separate piece), the largest unoriented one is oriented by
    direction agreement with the nearest oriented centroid points at any
    distance (kept as it is if those are not roughly parallel), and the
    greedy growth continues from there.

    Parameters
    ----------
    centroids : list of (n, 3) arrays
        The cluster centroids.
    clusterSizes : array-like of int
        The number of streamlines in each cluster (the weights).
    radiusMM : float, optional
        Distance within which two centroids' points count as overlapping.
        The default is 4.0.
    minAbsCos : float, optional
        Minimum |cos| between two overlapping points' local directions for
        the pair to count (excludes crossing, non-parallel pairs).  The
        default is 0.5.
    maxNeighbours : int or None, optional
        If an int, each centroid point is compared only with its
        maxNeighbours nearest points within radiusMM, which bounds memory and
        time on dense input.  None compares every pair within radiusMM.  The
        default is 16.

    Returns
    -------
    flipCentroid : numpy.ndarray of bool
        True for each centroid that must be reversed.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    #Claude note: resample each centroid to ~1 mm spacing, and compute its
    #unit local direction (tangent) at every point
    allPoints=[]
    allTangents=[]
    pointOwner=[]
    for iCentroid,currentCentroid in enumerate(centroids):
        currentCentroid=np.asarray(currentCentroid,dtype=float)
        #Claude note: arc length of this centroid
        segmentLengths=np.linalg.norm(np.diff(currentCentroid,axis=0),axis=1)
        cumulativeArc=np.concatenate([[0.0],np.cumsum(segmentLengths)])
        #Claude note: number of ~1 mm samples (at least 3)
        nSamples=max(int(np.ceil(cumulativeArc[-1])),3)
        sampleArc=np.linspace(0,cumulativeArc[-1],nSamples)
        if cumulativeArc[-1]>0:
            resampled=np.column_stack([np.interp(sampleArc,cumulativeArc,currentCentroid[:,iDim]) for iDim in range(3)])
        else:
            #Claude note: a zero-length centroid has no direction; repeat its point
            resampled=np.repeat(currentCentroid[:1],nSamples,axis=0)
        #Claude note: tangents by central differences, normalized
        tangents=np.gradient(resampled,axis=0)
        tangents=tangents/np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-9)
        allPoints.append(resampled)
        allTangents.append(tangents)
        pointOwner.append(np.full(len(resampled),iCentroid))
    allPoints=np.vstack(allPoints)
    allTangents=np.vstack(allTangents)
    pointOwner=np.concatenate(pointOwner)
    #Claude note: find overlapping point pairs
    pointTree=cKDTree(allPoints)
    if maxNeighbours is None:
        #Claude note: every pair within radiusMM
        pointPairs=pointTree.query_pairs(radiusMM,output_type='ndarray')
    else:
        #Claude note: each point's maxNeighbours nearest within radiusMM, so
        #memory stays O(points * maxNeighbours) in dense regions
        neighbourDistances,neighbourIndexes=pointTree.query(allPoints,k=maxNeighbours+1,distance_upper_bound=radiusMM)
        queryRows=np.repeat(np.arange(len(allPoints)),maxNeighbours+1)
        queryColumns=neighbourIndexes.ravel()
        #Claude note: missing neighbours are reported as index len(allPoints)
        foundNeighbour=queryColumns<len(allPoints)
        pointPairs=np.column_stack([queryRows[foundNeighbour],queryColumns[foundNeighbour]])
    #Claude note: hairpin (folded) centroids.  A centroid that comes back
    #alongside itself (two of its own points within radiusMM, at least
    #minFoldSeparationMM apart along it, running opposite ways) has two arms
    #that touch different parts of the tract in opposite directions, so it
    #passes contradictory evidence between them.  (Diagnosed 2026-11-12 on
    #HCP 100206 SLF_III_left and ST_PREM_right, where ~8% of streamlines fold
    #back and whole layers were reversed through them.)  With excludeFolded,
    #their point pairs are dropped from the agreement evidence; they are
    #then oriented individually against their nearest oriented neighbours
    #(the isolated-centroid rule below) and do not bridge other centroids.
    #Arcs and U shapes (AF, CC_7) are not affected: their arms are far apart.
    isFolded=np.zeros(len(centroids),dtype=bool)
    if excludeFolded:
        for iCentroid in range(len(centroids)):
            #Claude note: this centroid's own points (~1 mm apart, so index
            #separation is ~mm along the centroid)
            ownPoints=np.where(pointOwner==iCentroid)[0]
            #Claude note: arms closer than foldRadiusMM (default: radiusMM) count as a fold
            if foldRadiusMM is None:
                currentFoldRadius=radiusMM
            else:
                currentFoldRadius=foldRadiusMM
            selfPairs=cKDTree(allPoints[ownPoints]).query_pairs(currentFoldRadius,output_type='ndarray')
            if len(selfPairs)==0:
                continue
            selfPairs=selfPairs[np.abs(selfPairs[:,0]-selfPairs[:,1])>=minFoldSeparationMM]
            if len(selfPairs)==0:
                continue
            selfCosines=np.einsum('ij,ij->i',allTangents[ownPoints[selfPairs[:,0]]],allTangents[ownPoints[selfPairs[:,1]]])
            isFolded[iCentroid]=np.any(selfCosines<=-minAbsCos)
    #Claude note: only pairs from two different centroids carry information,
    #and none from folded centroids (if excluded)
    pointPairs=pointPairs[pointOwner[pointPairs[:,0]]!=pointOwner[pointPairs[:,1]]]
    pointPairs=pointPairs[~isFolded[pointOwner[pointPairs[:,0]]]&~isFolded[pointOwner[pointPairs[:,1]]]]
    #Claude note: cosine between the two points' local directions
    pairCosines=np.einsum('ij,ij->i',allTangents[pointPairs[:,0]],allTangents[pointPairs[:,1]])
    #Claude note: keep roughly parallel pairs (crossings say nothing about direction)
    parallelPair=np.abs(pairCosines)>=minAbsCos
    pointPairs=pointPairs[parallelPair]
    pairCosines=pairCosines[parallelPair]
    #Claude note: per pair of centroids, the summed agreement signs (S) and the
    #number of overlapping point pairs (N), as symmetric sparse matrices
    nCentroids=len(centroids)
    ownerA=pointOwner[pointPairs[:,0]]
    ownerB=pointOwner[pointPairs[:,1]]
    agreementSigns=np.sign(pairCosines)
    agreementMatrix=coo_matrix((np.concatenate([agreementSigns,agreementSigns]),(np.concatenate([ownerA,ownerB]),np.concatenate([ownerB,ownerA]))),shape=(nCentroids,nCentroids)).tocsr()
    overlapMatrix=coo_matrix((np.ones(2*len(ownerA)),(np.concatenate([ownerA,ownerB]),np.concatenate([ownerB,ownerA]))),shape=(nCentroids,nCentroids)).tocsr()
    #Claude note: greedy orientation
    clusterSizes=np.asarray(clusterSizes,dtype=float)
    flipCentroid=np.zeros(nCentroids,dtype=bool)
    isOriented=np.zeros(nCentroids,dtype=bool)
    #Claude note: running, size-weighted overlap and signed agreement of every
    #centroid with the oriented set
    overlapWithOriented=np.zeros(nCentroids)
    agreementWithOriented=np.zeros(nCentroids)
    for iStep in range(nCentroids):
        #Claude note: the unoriented centroid with the largest overlap
        candidateOverlap=np.where(isOriented,-1.0,overlapWithOriented)
        nextCentroid=int(np.argmax(candidateOverlap))
        if candidateOverlap[nextCentroid]<=0:
            #Claude note: nothing unoriented overlaps the oriented set within
            #radiusMM.  Take the largest unoriented cluster, and orient it by
            #direction agreement with the NEAREST oriented centroid points at
            #any distance (e.g. a lone outlier streamline a few mm off the
            #bundle); if nothing is oriented yet (the first centroid), or no
            #nearest pair is roughly parallel, keep it as it is.  (Leaving
            #isolated centroids as they were misoriented 9 of 4,000 single-
            #streamline outliers on synthetic truncated U bundles.)
            nextCentroid=int(np.argmax(np.where(isOriented,-1.0,clusterSizes)))
            flipCentroid[nextCentroid]=False
            if np.any(isOriented):
                #Claude note: the oriented centroids' points, with their
                #directions in their current (possibly reversed) orientation
                orientedPoint=isOriented[pointOwner]
                orientedTangents=allTangents[orientedPoint]*np.where(flipCentroid[pointOwner[orientedPoint]],-1.0,1.0)[:,None]
                ownPoint=pointOwner==nextCentroid
                nearestOriented=cKDTree(allPoints[orientedPoint]).query(allPoints[ownPoint])[1]
                nearestCosines=np.einsum('ij,ij->i',allTangents[ownPoint],orientedTangents[nearestOriented])
                parallelNearest=np.abs(nearestCosines)>=minAbsCos
                if np.any(parallelNearest):
                    flipCentroid[nextCentroid]=np.sum(np.sign(nearestCosines[parallelNearest]))<0
        else:
            #Claude note: reverse it if it mostly disagrees with the oriented set
            flipCentroid[nextCentroid]=agreementWithOriented[nextCentroid]<0
        isOriented[nextCentroid]=True
        #Claude note: fold it into its neighbours' running totals; a reversed
        #centroid's agreements change sign
        overlapRow=overlapMatrix.getrow(nextCentroid)
        agreementRow=agreementMatrix.getrow(nextCentroid)
        if flipCentroid[nextCentroid]:
            orientationSign=-1.0
        else:
            orientationSign=1.0
        overlapWithOriented[overlapRow.indices]+=overlapRow.data*clusterSizes[nextCentroid]
        agreementWithOriented[agreementRow.indices]+=agreementRow.data*orientationSign*clusterSizes[nextCentroid]
    return flipCentroid


def orientTractUsingCentroids(streamlines, thresholds=[30,20,10,5], nb_pts=50, seed=0, radiusMM=4.0, minAbsCos=0.5, maxNeighbours=16, nConsensusSeeds=5, surpressReport=False):
    """
    Claude note: orients the streamlines of a tract so that
    they all run the same way, through the tract's QuickBundles sub-bundles
    (Dan's idea), without using a neck.

    Method
    ------
    1. Split the tract into sub-bundles with quickbundlesClusters (seeded, so
       the result is reproducible).
    2. Orient each streamline to its own sub-bundle's centroid, by local
       direction agreement: reverse it if its local directions mostly
       oppose the centroid's at the centroid's nearest points.
    3. Orient the centroids against each other by local direction agreement
       where they overlap (_orientCentroidsByTangentAgreement; hairpin,
       i.e. self-folding, centroids are excluded from the evidence): wherever two
       centroids run within radiusMM and roughly parallel, the sign of their
       local directions' dot product says whether they agree.  This is
       insensitive to where along the tract a sub-bundle starts or stops,
       which matters for truncated streamlines.  (Comparing nodes at fixed
       fractions of length, as in MATLAB wma_tools endpointClusterProto.m,
       was tried first; it failed on HCP 100206 CC_7 in 7-10 of 10 seeds,
       ~17% locally reversed, because 125 mm-truncated sub-bundles sit
       offset along the callosal U.)
    4. Reverse each sub-bundle's streamlines with its centroid.  Steps 1-4
       are repeated for nConsensusSeeds clustering seeds and each streamline
       takes the majority decision.
    5. Whole-tract sign rule (as flipStreamstoAB_OrientOnce): the tract
       runs positive along its dimension of greatest average traversal, i.e.
       if most streamlines' last node lies below their first node in that
       dimension, every streamline is reversed.

    Choosing an orientation method (measured 2026-11-12 on HCP 100206; see
    docs/orientationMethods.md for the full tables)
    ---------------------------------------------------------------------
    orientTractUsingCentroids (this function)
        On 72 TractSeg tracts (1,500 streamlines each): locally reversed mean
        0.17%, no tract above 5% (measureOrientationLocalReversal);
        end-to-end opposite mean 0.27% (measureOrientationEndToEnd; worst
        ST_PREF_left 6.0%, a fan); ~1-2 s per tract with the default 5-seed
        vote.  Works with any node spacing, without a neck, and with as few
        as 20 streamlines.  Weak spot: fan-shaped tracts (striatal
        projections) keep 1-3% whose direction is ambiguous (they reach
        neither terminal region).  Hairpin streamlines (folding back on
        themselves) once reversed whole groups through contradictory
        evidence (SLF_III_left 30.8%, ST_PREM_right 23.0%); folded
        sub-bundles are now excluded from the evidence (see
        _orientCentroidsByTangentAgreement), and the 5-seed vote removes the
        rare clustering-dependent reversals.
    orientTractUsingNeck_Robust
        Local mean 0.52%, end-to-end mean 0.79%; 3.5-7 s per tract.  Needs
        node spacing regular to within 10% (raises otherwise) and a
        meaningful neck.  Failure mode: tracts whose neck sits near one end
        (ST_OCC_right 25.8% end-to-end, ST_PREM_left 7.8%, T_OCC_right 6.9%).
        A useful second opinion: where the two methods disagree on more than
        ~5% of streamlines, render the tract before trusting either.
    orientAllStreamlines
        Per streamline; very fast; local mean 3.89% with 19 of 72 tracts
        above 5% (fails on curved, hooked and commissural tracts).
    orientTractUsingNeck, orientTractUsingNeck_multi, dipyOrientStreamlines
        Not recommended (position-based flips; independently oriented
        sub-bundles).
    Whole-brain use: orienting a whole tractogram before segmentation was
    much worse than orienting each tract after segmentation (first 100k
    streamlines of the HCP 100206 3M tractogram, 39 tracts: mean 7.4%
    locally reversed with a prototype of this method on the whole
    tractogram, 18.6% with the per-cluster path of
    visTools.orientAndColormapStreams, vs 0.16% with this method per
    tract).  Orient per tract.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
        The streamlines of ONE tract.  Modified in place and also returned.
    thresholds : list of float, optional
        QuickBundles distance thresholds (mm), coarse to fine.  The default
        is [30,20,10,5].
    nb_pts : int, optional
        Points per streamline used for clustering and for the
        streamline-to-centroid comparison.  The default is 50.
    seed : int or None, optional
        Clustering seed.  The default is 0 (reproducible); None gives dipy's
        unseeded behavior.
    radiusMM : float, optional
        Distance within which centroids count as overlapping.  The default
        is 4.0.
    minAbsCos : float, optional
        Minimum |cos| between overlapping centroid directions for the pair
        to count.  The default is 0.5.
    maxNeighbours : int or None, optional
        Nearest overlapping points considered per centroid point (None: all
        pairs; slower and more memory on dense input).  The default is 16.
    nConsensusSeeds : int, optional
        Number of clustering seeds (seed, seed+1, ...) whose per-streamline
        decisions are combined by majority vote.  1 is fastest (~0.3 s per
        1,500 streamlines) but occasionally reverses a block of a tract; 5
        (the default, ~1.5 s) removed every such failure seen.
    surpressReport : bool, optional
        If False, prints the number of sub-bundles and of reversed
        streamlines.  The default is False.

    Returns
    -------
    streamlines : same type as input
        The streamlines, reoriented in place.
    """
    import numpy as np
    from dipy.tracking.streamline import set_number_of_points, Streamlines
    #Claude note: nothing to orient for fewer than two streamlines
    if len(streamlines)<2:
        return streamlines
    #Claude note: steps 1-4 are repeated for nConsensusSeeds clustering seeds
    #(seed, seed+1, ...), and each streamline takes the majority decision.
    #A single clustering occasionally reverses a block of a tract (measured:
    #1 of 40 runs on 4 hard tracts, ST_PREM_right 40%); the vote removed it
    #in every seed set tried.
    from scipy.spatial import cKDTree
    #Claude note: every streamline resampled to nb_pts points, all at once
    #(shared by all seeds)
    resampledStreamlines=np.asarray(set_number_of_points(Streamlines(streamlines),nb_pts),dtype=float)
    #Claude note: unit local directions of the resampled streamlines
    streamlineTangents=np.gradient(resampledStreamlines,axis=1)
    streamlineTangents=streamlineTangents/np.maximum(np.linalg.norm(streamlineTangents,axis=2,keepdims=True),1e-9)
    if seed is None:
        consensusSeeds=[None]*max(int(nConsensusSeeds),1)
    else:
        consensusSeeds=[seed+iSeed for iSeed in range(max(int(nConsensusSeeds),1))]
    reverseBySeed=[]
    for currentSeed in consensusSeeds:
        #Claude note: 1. sub-bundles (seeded)
        clusters=quickbundlesClusters(streamlines,thresholds=list(thresholds),nb_pts=nb_pts,seed=currentSeed)
        #Claude note: the members of each cluster, in a fixed (sorted) order
        clusterMembers=[sorted(int(iIndex) for iIndex in currentCluster.indices) for currentCluster in clusters]
        clusterCentroids=[np.asarray(currentCluster.centroid,dtype=float) for currentCluster in clusters]
        #Claude note: 2. each streamline against its own centroid, by local
        #direction agreement: at each of the streamline's points, the dot product
        #of its local direction with the centroid's local direction at the
        #centroid's nearest point; reversed if the sum is negative.  (Comparing
        #positions instead, i.e. mean pointwise distance as is vs reversed, is
        #ambiguous for a streamline that sits symmetrically around a bend: on a
        #synthetic truncated U it left 1 of 200 reversed.)
        reverseStreamline=np.zeros(len(streamlines),dtype=bool)
        for currentMembers,currentCentroid in zip(clusterMembers,clusterCentroids):
            currentMembers=np.asarray(currentMembers)
            #Claude note: the centroid's unit local directions
            centroidTangents=np.gradient(currentCentroid,axis=0)
            centroidTangents=centroidTangents/np.maximum(np.linalg.norm(centroidTangents,axis=1,keepdims=True),1e-9)
            #Claude note: each member point's nearest centroid point
            memberPoints=resampledStreamlines[currentMembers].reshape(-1,3)
            nearestCentroidPoint=cKDTree(currentCentroid).query(memberPoints)[1]
            #Claude note: summed agreement per member
            pointAgreement=np.einsum('ij,ij->i',streamlineTangents[currentMembers].reshape(-1,3),centroidTangents[nearestCentroidPoint])
            memberAgreement=pointAgreement.reshape(len(currentMembers),-1).sum(axis=1)
            reverseStreamline[currentMembers]=memberAgreement<0
        #Claude note: 3. centroids against each other
        flipCentroid=_orientCentroidsByTangentAgreement(clusterCentroids,[len(currentMembers) for currentMembers in clusterMembers],radiusMM=radiusMM,minAbsCos=minAbsCos,maxNeighbours=maxNeighbours)
        #Claude note: 4. a streamline's final reversal: its own, toggled by its centroid's
        for iCluster,currentMembers in enumerate(clusterMembers):
            if flipCentroid[iCluster]:
                reverseStreamline[currentMembers]=~reverseStreamline[currentMembers]
        reverseBySeed.append(reverseStreamline.copy())
    #Claude note: align every run's overall sign to the first run's (a run may
    #differ from it by a whole-tract reversal, which carries no information),
    #then take the per-streamline majority; ties keep the first run's decision
    reverseBySeed=np.array(reverseBySeed)
    for iRun in range(1,len(reverseBySeed)):
        if np.mean(reverseBySeed[iRun]==reverseBySeed[0])<0.5:
            reverseBySeed[iRun]=~reverseBySeed[iRun]
    reverseVotes=reverseBySeed.sum(axis=0)
    reverseStreamline=reverseBySeed[0].copy()
    reverseStreamline[2*reverseVotes>len(reverseBySeed)]=True
    reverseStreamline[2*reverseVotes<len(reverseBySeed)]=False
    #Claude note: 5. whole-tract sign rule: positive along the dimension of
    #greatest average traversal, decided by the majority of streamlines in
    #their new orientation
    #Claude note: each streamline's summed absolute node-to-node displacement
    #per axis, i.e. analysisTools.cumulativeTraversalStream, vectorized (that
    #function loops over nodes in Python and was 60% of this function's time)
    streamTraversals=[np.sum(np.abs(np.diff(np.asarray(iStreamline,dtype=float),axis=0)),axis=0) for iStreamline in streamlines]
    maxTraversalDim=int(np.argmax(np.mean(streamTraversals,axis=0)))
    #Claude note: first-to-last displacement along that dimension, in the new orientation
    displacement=np.array([np.asarray(iStreamline)[-1,maxTraversalDim]-np.asarray(iStreamline)[0,maxTraversalDim] for iStreamline in streamlines])
    displacement[reverseStreamline]=-displacement[reverseStreamline]
    if np.sum(displacement<0)>np.sum(displacement>0):
        reverseStreamline=~reverseStreamline
    #Claude note: apply, in place
    for iIndex in np.where(reverseStreamline)[0]:
        streamlines[iIndex]=streamlines[iIndex][::-1]
    if not surpressReport:
        print(str(len(consensusSeeds))+' clustering seed(s), last with '+str(len(clusters))+' sub-bundles; '+str(int(np.sum(reverseStreamline)))+' of '+str(len(streamlines))+' streamlines reversed')
    return streamlines


def measureOrientationLocalReversal(streamlines, stepMM=2.0, radiusMM=2.0, minAbsCos=0.8, returnFlags=False):
    """
    Claude note: measures how consistently a tract's
    streamlines are oriented, using only local information, so that it does
    not favor any particular orientation method.  Every streamline is sampled
    every stepMM (a point and its unit local direction).  For each sample,
    the samples of OTHER streamlines within radiusMM whose direction is
    roughly parallel (|cos| >= minAbsCos) are its neighbours.  A
    consistently oriented tract has neighbours running the same way.

    Validated on synthetic bundles: with 10.5% / 27.5% of a straight bundle
    reversed it reports 10.5% / 27.5%; on an arc, 14.0% / 30.5% -> 14.5% /
    30.0%.  Near 50% reversed the two directions are indistinguishable and it
    saturates.  Limits: a coherent group reversed as a whole can be nearly
    invisible to it, because most of each member's neighbours are its own
    groupmates (measureOrientationEndToEnd covers this case: on
    SLF_III_left such a 33% group read 5.2% here and 30.8% there); review
    such cases with renders.  An earlier endpoint-region score
    (2-means on all endpoints) proved unreliable on fan-shaped tracts and the
    whole corpus callosum (0.8-28.6% on the same oriented output depending on
    the 2-means initialization) and should not be used.

    Parameters
    ----------
    streamlines : sequence of (n, 3) arrays
        The streamlines of one tract.
    stepMM : float, optional
        Sampling step along each streamline.  The default is 2.0.
    radiusMM : float, optional
        Neighbourhood radius.  The default is 2.0.
    minAbsCos : float, optional
        Minimum |cos| for two samples to count as parallel.  The default is 0.8.
    returnFlags : bool, optional
        If True, return the per-streamline flags instead of the summary.
        The default is False.

    Returns
    -------
    summary : dict
        'streamPct' : % of streamlines (with neighbours) most of whose
            neighbour pairs run the opposite way.
        'pairPct' : % of neighbour pairs that run opposite ways.
        'nPairs' : number of neighbour pairs.
        'noNeighbours' : number of streamlines with no neighbours.
    flags : numpy.ndarray of bool (if returnFlags)
        True for each locally reversed streamline.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    samplePoints=[]
    sampleTangents=[]
    sampleOwner=[]
    for iStreamline,currentStreamline in enumerate(streamlines):
        currentStreamline=np.asarray(currentStreamline,dtype=float)
        #Claude note: arc length along this streamline
        cumulativeArc=np.concatenate([[0.0],np.cumsum(np.linalg.norm(np.diff(currentStreamline,axis=0),axis=1))])
        #Claude note: too short to sample
        if cumulativeArc[-1]<2*stepMM:
            continue
        sampleArc=np.arange(stepMM/2,cumulativeArc[-1]-stepMM/2,stepMM)
        points=np.column_stack([np.interp(sampleArc,cumulativeArc,currentStreamline[:,iDim]) for iDim in range(3)])
        #Claude note: local direction over +-0.5 mm
        ahead=np.column_stack([np.interp(sampleArc+0.5,cumulativeArc,currentStreamline[:,iDim]) for iDim in range(3)])
        behind=np.column_stack([np.interp(sampleArc-0.5,cumulativeArc,currentStreamline[:,iDim]) for iDim in range(3)])
        tangents=ahead-behind
        tangents=tangents/np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-9)
        samplePoints.append(points)
        sampleTangents.append(tangents)
        sampleOwner.append(np.full(len(points),iStreamline))
    nStreamlines=len(streamlines)
    if len(samplePoints)==0:
        if returnFlags:
            return np.zeros(nStreamlines,dtype=bool)
        return {'streamPct':np.nan,'pairPct':np.nan,'nPairs':0,'noNeighbours':nStreamlines}
    samplePoints=np.vstack(samplePoints)
    sampleTangents=np.vstack(sampleTangents)
    sampleOwner=np.concatenate(sampleOwner)
    #Claude note: neighbouring samples from different streamlines
    samplePairs=cKDTree(samplePoints).query_pairs(radiusMM,output_type='ndarray')
    samplePairs=samplePairs[sampleOwner[samplePairs[:,0]]!=sampleOwner[samplePairs[:,1]]]
    pairCosines=np.einsum('ij,ij->i',sampleTangents[samplePairs[:,0]],sampleTangents[samplePairs[:,1]])
    #Claude note: keep roughly parallel pairs
    parallelPair=np.abs(pairCosines)>=minAbsCos
    samplePairs=samplePairs[parallelPair]
    pairCosines=pairCosines[parallelPair]
    runsOpposite=pairCosines<0
    #Claude note: per streamline, its neighbour pairs and how many run opposite
    pairsPerStreamline=np.bincount(sampleOwner[samplePairs[:,0]],minlength=nStreamlines)+np.bincount(sampleOwner[samplePairs[:,1]],minlength=nStreamlines)
    oppositePerStreamline=np.bincount(sampleOwner[samplePairs[:,0]],weights=runsOpposite,minlength=nStreamlines)+np.bincount(sampleOwner[samplePairs[:,1]],weights=runsOpposite,minlength=nStreamlines)
    hasNeighbours=pairsPerStreamline>0
    locallyReversed=hasNeighbours&(oppositePerStreamline>0.5*np.maximum(pairsPerStreamline,1))
    if returnFlags:
        return locallyReversed
    if np.any(hasNeighbours):
        streamPct=float(100*np.mean(locallyReversed[hasNeighbours]))
    else:
        streamPct=np.nan
    if len(runsOpposite):
        pairPct=float(100*np.mean(runsOpposite))
    else:
        pairPct=np.nan
    return {'streamPct':streamPct,'pairPct':pairPct,'nPairs':int(len(runsOpposite)),'noNeighbours':int(np.sum(~hasNeighbours))}


def measureOrientationEndToEnd(streamlines, minProjectionMM=10.0):
    """
    Claude note: a global orientation-consistency check,
    complementary to measureOrientationLocalReversal.  Each streamline's
    end-to-end vector (last node minus first node) is projected onto the
    tract's principal end-to-end axis (the first singular vector of all
    end-to-end vectors, which does not depend on their signs).  Among
    streamlines whose projection is at least minProjectionMM long, it
    reports the % whose sign opposes the majority.

    Why both measures: the local measure can miss a coherent group that is
    reversed as a whole, because most of each member's neighbours are its
    own (equally reversed) groupmates.  This happened: on HCP 100206
    SLF_III_left, a pre-release version of orientTractUsingCentroids
    reversed a 489-streamline layer (33%) that was interleaved with the rest
    (median 0.7 mm away) but whose members' neighbours were 90% layer-mates;
    the local measure reported 5.2%, this one 30.8%.  Conversely,
    this measure assumes the tract runs between two terminal regions; it is
    unreliable for U-shaped tracts whose streamlines are truncated (CC_7:
    ~5% from midline stubs), for fans, and for short streamlines (excluded
    by minProjectionMM).

    Parameters
    ----------
    streamlines : sequence of (n, 3) arrays
        The streamlines of one tract.
    minProjectionMM : float, optional
        Streamlines whose end-to-end projection is shorter are excluded
        (their direction along the axis is ambiguous).  The default is 10.0.

    Returns
    -------
    summary : dict
        'oppositePct' : % of the included streamlines running against the
            majority along the principal end-to-end axis.
        'nIncluded' : number of streamlines included.
        'axis' : the principal end-to-end axis (unit vector).
    """
    import numpy as np
    #Claude note: end-to-end vectors
    endToEnd=np.array([np.asarray(iStreamline,dtype=float)[-1]-np.asarray(iStreamline,dtype=float)[0] for iStreamline in streamlines]).reshape(-1,3)
    if len(endToEnd)==0:
        return {'oppositePct':np.nan,'nIncluded':0,'axis':np.full(3,np.nan)}
    #Claude note: principal axis (sign-free)
    principalAxis=np.linalg.svd(endToEnd,full_matrices=False)[2][0]
    projection=endToEnd@principalAxis
    included=np.abs(projection)>=minProjectionMM
    if not np.any(included):
        return {'oppositePct':np.nan,'nIncluded':0,'axis':principalAxis}
    positiveFraction=np.mean(projection[included]>0)
    return {'oppositePct':float(100*min(positiveFraction,1-positiveFraction)),'nIncluded':int(np.sum(included)),'axis':principalAxis}


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
    import numpy as np
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

    response, ratio = auto_response_ssst(gtab, np.asanyarray(dwi.dataobj), roi_radii=10, fa_thr=0.7)
    #response, ratio = auto_response(gtab, np.asanyarray(dwi.dataobj), roi_radii=10, fa_thr=0.7)
    
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    
    
   
    #no white matter mask for now?
    wmOnDwi=image.resample_to_img(wmMask,dwi,interpolation='nearest')
    
    csd_fit = csd_model.fit(np.asanyarray(dwi.dataobj), mask=np.asanyarray(wmOnDwi.dataobj))

    # prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
    #                                                 max_angle=30.,
    #                                                 sphere=default_sphere)
    
    
    csa_model = CsaOdfModel(gtab, sh_order=6)
    gfa = csa_model.fit(np.asanyarray(dwi.dataobj), mask=np.asanyarray(wmOnDwi.dataobj)).gfa
    stopping_thr= 0.2
    stopping_criterion = ThresholdStoppingCriterion(gfa, stopping_thr)
    
    fod = csd_fit.odf(default_sphere)
    pmf = fod.clip(min=0)
    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle=30.,
                                                    sphere=default_sphere)   

   
    seeds = utils.seeds_from_mask(np.asanyarray(targetMask.dataobj), targetMask.affine, density=seed_density)
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

def quickbundlesClusters(streamlines, thresholds = [30,20,10,5], nb_pts=100,verbose=False,seed=None):
    """
    (quickly?, via qbx_and_merge?) perform a quick-bundling of an input
    collection of streamlines.  Perhaps an unnecessary wrapper.
    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        The input streamlines 
    **kwargs : keyword arguments for the qbx_and_merge
        Currently only supports [thresholds] and [nb_pts]
        See dipy.segment.bundles.qbx_and_merge for more details
    seed : int or None, optional
        Claude note: seed for qbx_and_merge's random generator.  None (the
        default, and the original behavior) passes rng=None, so dipy draws an
        unseeded generator and the clusters (their number, membership and
        centroids) differ from run to run.  Pass an int for reproducible
        clusters.  (Added 2026-11-12; measured: on HCP 100206 CC_7, 1,500
        streamlines, the cluster count varied between 91 and 96 across runs.)
    Returns
    -------
    clusters : dipy.segment.clustering.ClusterMapCentroid
        The clusters resulting from the quickBundle-ification of the input 
        streamlines
    """
    from dipy.segment.bundles import qbx_and_merge
    from dipy.tracking.streamline import Streamlines
    import numpy as np
    
    #this didn't work, and was causing all kinds of problems
    # #fill in parameters if they are there.
    # if not 'thresholds' in locals():
    #     thresholds = [30,20,10,5]
    # if not 'nb_points' in locals():
    #     nb_pts=20
    # if not 'verbose' in locals():
    #     verbose=False
    #perform the quick, iterave bundling
    #Claude note: a seeded generator when a seed is given; otherwise rng=None,
    #as before (dipy then uses an unseeded generator)
    if seed is None:
        clusteringRng=None
    else:
        clusteringRng=np.random.default_rng(seed)
    #Claude note: nb_pts is passed by keyword (dipy deprecates passing it positionally)
    clusters=qbx_and_merge(streamlines,thresholds , nb_pts=nb_pts, select_randomly=None, rng=clusteringRng, verbose=verbose)
    
    return clusters

def cullViaClusters(clusters,streamlines,streamThresh):
    """
    Determine from an input collection of streamlines which ought to be culled
    in accordance with how "unique" the trajectory of individual streamlines
    are, as assessed by the number of streamlines they are bundled with using
    DIPY's quickbundles.
    
    Returns indexes for both the surviving and (to be) culled streamlines.  As
    such, does not actually perform the streamline cull itself, which would
    be performed with cullStreamsByBundling

    Parameters
    ----------
    clusters : ClusterMapCentroid object of dipy.segment.clustering module
        Output clustring of dipy's quickbundles, e.g. from quickbundlesClusters
    streamlines : nib.streamlines.array_sequence.ArraySequence
        The source streamlines. Technically redundant with clusters._refdata.
        Will be removed in future versions
    streamThresh : int. Positive, greater than 0
        The *minimum* number of streamlines needed in a quick bundles cluster
        in order for *all* of the streamlines in that cluster to survive
        the culling process.  All streamlines in all clusters having fewer than
        this number will be interpreted as being excessively distinct from
        other trajectories in the input collection of streamlines and will be
        removed.

    Returns
    -------
    survivingStreamsIndicies : list of int
        The indexes of the surviving / filtered streamlines.
        set(survivingStreamsIndicies).union(culledStreamIndicies)==range(len(clusters._refdata))==range(len(streamlines))
    culledStreamIndicies : list of int
        The indexes of the culled / (to be) removed streamlines.
        set(survivingStreamsIndicies).union(culledStreamIndicies)==range(len(clusters._refdata))==range(len(streamlines))

    """

    import itertools
    import numpy as np
    import nibabel as nib
    import dipy
    
    #implement robustness for case in which streamlines instead of cluster
    #is passed.  Implicitly, this would indicate the user's desire to perform
    #clustering inside this function, instead of performing it ahead of time.
    #could be sensible in a case where you wouldn't be using the clustering 
    #information for other purposes.
    if isinstance(clusters, nib.streamlines.array_sequence.ArraySequence):
        clusters=quickbundlesClusters(streamlines, thresholds=[30,20,10,5],nb_pts=100)
    elif isinstance(clusters, dipy.segment.clustering.ClusterMapCentroid):
        pass
        #do nothing, it's in the format that is needed
    
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
    culledStreamIndicies=list(set(list(range(0,len(clusters._refdata))))-set(survivingStreamsIndicies))

    return survivingStreamsIndicies, culledStreamIndicies

def manualSelectStreams_byEndpoint(streamlines,parc,lookupTable,targetLabels):
    """
    Determines which streamlines end in the requested collections of endpoints.
    In other words, reduces the output of 
    dipy.tracking.utils.connectivity_matrix to the indexes of *only* the
    streamlines terminating in each of the specified collections of termination
    areas.
    
    Returns a dictionary of classification structures such that each key-value
    pair corresponds to the requested label value(s) and the streamline indexes 
    associated with those termination areas.

    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        The streamlines that are to be considered
    parc : nibabel.nifti1.Nifti1Image or compatible
        The parcellation to be used for dipy's utils.connectivity_matrix
    lookupTable : csv or pandas.DataFrame
        The lookup table for the associated parc
    targetLabels : list of (int or list of int)
        The requested termination areas whose streamline endpoint associations
        are to be determined.  Each element of the list is treated as a
        separate request, and has a corresponding output classification
        structure.

    Raises
    ------
    ValueError
        Raises a value error of any of the requested labels are not found in
        the input parcellation

    Returns
    -------
    outClassificationDicts : dict
        A dictionary of classification structures such that each key-value
        pair corresponds to the requested label value(s) and the streamline
        indexes associated with those termination areas.

    """
    
    
    import numpy as np
    from dipy.tracking.utils import reduce_labels
    from dipy.tracking import utils
    import wmaPyTools.analysisTools
    import itertools
    
    #throw an error at the outset if the requested labels aren't in the provided table
    #get the unique label indexes requested
    uniqueRequestLabels=np.unique(targetLabels)
    #find out which labels are avaialbe in the input parc
    availableLabels=np.unique(np.asanyarray(parc.dataobj))
    #find which were requested but not avaialbe
    requestedButNotFound=list(set(list(uniqueRequestLabels))-set(list(availableLabels)) )
    if len(requestedButNotFound)>0:
        raise ValueError('Requested indexes: \n '+ requestedButNotFound + '\n not found in input parcellation.' )
    
    #perform initial dipy based endpoint analysis
    #begin by reducing the input atlas
    [renumberedAtlasNifti,reducedLookupTable]=wmaPyTools.analysisTools.reduceAtlasAndLookupTable(parc,lookupTable,removeAbsentLabels=True)
    #then do the endpoint connectivity matrix
    M, grouping=utils.connectivity_matrix(streamlines, np.round(renumberedAtlasNifti.affine,2), label_volume=np.asanyarray(renumberedAtlasNifti.dataobj).astype(int),
                            symmetric=False,
                            return_mapping=True,
                            mapping_as_streamlines=False)
    #from the DIPY connectivity dictionary structure, obtain all of the keys
    #these keys are the indexes of *existing* connections, as indexed by
    #their *reduced* index number. The value pairings for these are the
    #streamline indexes    
    groupingKeys=grouping.keys()
    
    outClassificationDicts={}

    for iIterator,iSubdivisions in enumerate(targetLabels):
        
        #if iSubdivisions is a singleton, convert it to a list
        if isinstance(iSubdivisions,int):
            iSubdivisions=[iSubdivisions]
        
        #convert the requested label, presumably from the original parcellation,
        #to the reduced version
        currReducedIndexes=[reducedLookupTable.index[reducedLookupTable['#No.']==iIndexes].values[0] for iIndexes in iSubdivisions]
        
        #get the name or names for the current request, could be multiple
        #so cat them together with and to obtain a name for this streamline collection
        #find the entries in the lookup table
        names=reducedLookupTable['LabelName:'].iloc[currReducedIndexes].values
        #join them all together
        namesJoined='_and_'.join(names)
        
        #iterate across the keys and determine if any match the current subdivision
        keyBoolVec=np.zeros(len(groupingKeys),dtype=bool)
        for iKeyIndex,iKeyPairs in enumerate(groupingKeys):
            keyBoolVec[iKeyIndex]=iKeyPairs[0] in iSubdivisions or iKeyPairs[1] in iSubdivisions
        
        #extract the connectivity matrix keys that meet this criterion
        validKeys=list(itertools.compress(groupingKeys,keyBoolVec))
        
        #extract the list of streamline indexes for each of these keys
        indexLists=[grouping[iKey] for iKey in validKeys]
        
        #cat them all together, but be sure to get the unique values, as it
        #is possible in cases where multiple labels have been entered for this
        #iSubdivisions that *both* endpoint areas have been chosen, and thus
        #the a streamline index might show up more than once
        allStreamIndicies=np.unique(list(itertools.chain(*indexLists)))
        
        #now that we have the unique indexes for this, we need to create a WMC
        #structure
        #because we are creating a new wmc for each iSubdivisions requested, we
        #need to create a boolean vector mask of the input streamlines for the
        #current indexes
        currentBoolVec=np.zeros(len(streamlines),dtype=bool)
        #do I need to do it this way?
        currentBoolVec[allStreamIndicies]=[True]*len(allStreamIndicies)
        #currentBoolVec[allStreamIndicies]=True
        currentClassificationStructure=updateClassification(currentBoolVec,namesJoined,existingClassification=None)
        
        outClassificationDicts[iSubdivisions]=currentClassificationStructure
    
    return outClassificationDicts

# def cullStreamsByBundling(streamlines,streamThresh,qbThresholds=[30,20,10,5],qbResmaple=100):
#     """
#     Actually perform the streamline cull 

#     Parameters
#     ----------
#     streamlines : TYPE
#         DESCRIPTION.
#     streamThresh : TYPE
#         DESCRIPTION.
#     qbThresholds : TYPE, optional
#         DESCRIPTION. The default is [30,20,10,5].
#     qbResmaple : TYPE, optional
#         DESCRIPTION. The default is 100.

#     Returns
#     -------
#     survivingStreamsIndicies : TYPE
#         DESCRIPTION.
#     culledStreamIndicies : TYPE
#         DESCRIPTION.

#     """
    
#     clusters=quickbundlesClusters(streamlines, thresholds=qbThresholds,nb_pts=qbResmaple,verbose=True)
    
#     survivingStreamsIndicies, culledStreamIndicies=cullViaClusters(clusters,streamlines,streamThresh)
    
#     return survivingStreamsIndicies, culledStreamIndicies