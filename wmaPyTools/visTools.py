# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:14 2021

@author: Daniel
"""
# def crossSectionGIFsFromTract(streamlines,refAnatT1,saveDir):
    
    #DEPRICATED IN FAVOR OF densityGifsOfTract, which utilizes the existing crossSectionGIFsFromNifti
    
#     import nibabel as nib
#     #use dipy to create the density mask
#     from dipy.tracking import utils
#     import numpy as np
    
#     import wmaPyTools.roiTools 
    
#     from nilearn.image import crop_img 
#     #nilearn.image.resample_img ? to resample output
    
#     croppedReference=crop_img(refAnatT1)
    
#     densityMap=utils.density_map(streamlines, croppedReference.affine, croppedReference.shape)
#     densityNifti=nib.nifti1.Nifti1Image(densityMap,croppedReference.affine, croppedReference.header)
    
#     #refuses to plot single slice, single image
#     #from nilearn.plotting import plot_stat_map
#     #outImg=plot_stat_map(stat_map_img=densityNifti,bg_img=refAnatT1, cut_coords= 1,display_mode='x',cmap='viridis')
   
    
#     #obtain boundary coords in subject space in order to
#     #use plane generation function
#     convertedBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(croppedReference)
#     #crossSectionGIFsFromNifti
#     dimsList=['x','y','z']
#     #brute force with matplotlib
#     import matplotlib.pyplot as plt
#     for iDims in list(range(len(croppedReference.shape))):
#         #this assumes that get_zooms returns zooms in subject space and not image space orientation
#         # which may not be a good assumption if the orientation is weird
#         subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],refAnatT1.header.get_zooms()[iDims])
#         #get the desired broadcast shape and delete current dim value
#         broadcastShape=list(croppedReference.shape)
#         del broadcastShape[iDims]
        
#         #iterate across slices
#         for iSlices in list(range(len(subjectSpaceSlices))):
#             #set the slice list entry to the appropriate singular value
#             currentSlice=wmaPyTools.roiTools.makePlanarROI(croppedReference, subjectSpaceSlices[iSlices], dimsList[iDims])

#             #set up the figure
#             fig,ax = plt.subplots()
#             ax.axis('off')
#             #kind of overwhelming to do this in one line
#             refData=np.rot90(np.reshape(croppedReference.get_data()[currentSlice.get_data().astype(bool)],broadcastShape),3)
#             plt.imshow(refData, cmap='gray', interpolation='nearest')
#             #kind of overwhelming to do this in one line
#             densityData=np.rot90(np.reshape(densityNifti.get_data()[currentSlice.get_data().astype(bool)],broadcastShape),3)
#             plt.imshow(np.ma.masked_where(densityData<1,densityData), cmap='viridis', alpha=.5, interpolation='nearest')
#             figName='dim_' + str(iDims) +'_'+  str(iSlices).zfill(3)
#             plt.savefig(figName,bbox_inches='tight')
#             plt.clf()
    
#     import os        
#     from PIL import Image
#     import glob
#     for iDims in list(range(len(croppedReference.shape))):
#         dimStem='dim_' + str(iDims)
#         img, *imgs = [Image.open(f) for f in sorted(glob.glob(dimStem+'*.png'))]
#         img.save(os.path.join(saveDir,dimStem+'.gif'), format='GIF', append_images=imgs,
#                  save_all=True, duration=len(imgs)*2, loop=0)
#         #this worked for a time, not sure why it doesn't take lists now
#         [os.remove(iImgFiles) for iImgFiles in sorted(glob.glob(dimStem+'*.png'))]
#         #os.remove()
        
def plotMultiGifsFrom4DNifti(fourDNifti,referenceAnatomy,saveDir):
    
    
    import os
    import nibabel as nib
    
    for iSlices in range(fourDNifti.shape[3]):
        
        outPath=os.path.join(saveDir, str(iSlices))
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        #just try it, maybe it will work
        densityNifti=nib.nifti1.Nifti1Image(fourDNifti.get_data()[:,:,:,iSlices],referenceAnatomy.affine,referenceAnatomy.header)
        crossSectionGIFsFromNifti(densityNifti,referenceAnatomy,outPath) 
        
def dispersionReport(outDict,streamlines,saveDir,refAnatT1,distanceParameter=3):
    """
    

    Parameters
    ----------
    outDict : TYPE
        DESCRIPTION.
    saveDir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import dipy.tracking.utils as ut
    from dipy.tracking.vox2track import streamline_mapping
    import nibabel as nib
    
    streamlineMapping=streamline_mapping(streamlines, refAnatT1.affine)
    
    
    ### ugly stuff, but needed for later
    
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
    
    
    outputFiles=list(outDict.keys())
    for iOutFiles in outputFiles:
        outPath=os.path.join(saveDir, iOutFiles)
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        
        currentData=outDict[iOutFiles].get_data()
        smallestNotZero=np.unique(currentData)[1]
        crossSectionGIFsFromNifti(outDict[iOutFiles],refAnatT1,outPath, blendOption=False)
        #lets mask out the background
        [unique, counts] = np.unique(currentData, return_counts=True)
        backgroundVal=unique[np.where(np.max(counts)==counts)]        
        fig = plt.hist(np.ravel(currentData[currentData!=backgroundVal]),bins=2000)
        #fig = plt.hist(np.ravel(currentData[currentData!=backgroundVal]))
      
        plt.title(iOutFiles)
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(outPath,iOutFiles+".png"),dpi=300)
        #plot lowest and highest 
        densityMap=ut.density_map(streamlines, refAnatT1.affine, refAnatT1.shape)
        countWeightedValues=np.nan_to_num(np.divide(currentData,densityMap))
        #highest first
        #find where that location is
        maxLocation=np.where(countWeightedValues==np.max(countWeightedValues))
        #find the streams in that window
        maxLocationList=[maxLocation[0][0],maxLocation[1][0],maxLocation[2][0]]
        maxLocationIndexes= streamsInImgSpaceWindow(maxLocationList, streamlineMapping, mask_r)
        #name the plot
        tractName=os.path.join(outPath,'_'.join([str(x) for x in maxLocationList]) +'_maxValStreams')
        dipyPlotTract(streamlines[maxLocationIndexes],refAnatT1=None, tractName=tractName,endpointColorDensityKernel=7)
        #
        smallestNotZero=np.unique(countWeightedValues)[1]
        minLocation=np.where(countWeightedValues==np.min(countWeightedValues[countWeightedValues>=smallestNotZero]))
        #find the streams in that window
        minLocationList=[minLocation[0][0],minLocation[1][0],minLocation[2][0]]
        minLocationIndexes= streamsInImgSpaceWindow(minLocationList, streamlineMapping, mask_r)
        #name the plot
        tractName=os.path.join(outPath,'_'.join([str(x) for x in minLocationList]) +'_minValStreams')
        dipyPlotTract(streamlines[minLocationIndexes],refAnatT1=None, tractName=tractName,endpointColorDensityKernel=7)
        
        #save the nfitis down
        nib.save(outDict[iOutFiles],os.path.join(outPath, iOutFiles+'.nii.gz'))
        
        
        
        #find highest and lowest streamline-count-weighted values
        

        
        # get a streamline index dict of the whole brain tract
        
        
def generateAnatOverlayXSections(overlayNifti,refAnatT1,saveDir):
    import nibabel as nib
    #use dipy to create the density mask
    from dipy.tracking import utils
    import numpy as np
    from glob import glob
    import os
    from nilearn.image import reorder_img  
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import figure
    from nilearn.image import crop_img, resample_img 
    import matplotlib.pyplot as plt
    import dipy.tracking.utils as ut
    import wmaPyTools.roiTools  
    
    #do some testing here for binary vs count data
    
    #RAS reoreintation
    refAnatT1=reorder_img(refAnatT1)
    overlayNifti=reorder_img(overlayNifti)
    
    #get the resolution of each nifti
    overlayRes=overlayNifti.header.get_zooms()
    refT1Res=refAnatT1.header.get_zooms()
    #if they aren't the same resolution
    if not np.array_equal(overlayRes,refT1Res):
        #resample the *T1* not the overlay
        #this is because this is just for visualization purposes, not quantative
        #and resampling the typically sparse overlay leads to substantial losses
        refAnatT1=resample_img(refAnatT1,target_affine=overlayNifti.affine[0:3,0:3])
        
    #get their shapes as well
    overlayShape=overlayNifti.shape
    refT1Shape=refAnatT1.shape
    
    #obtain boundary coords in subject space in order to
    #use plane generation function
    convertedBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(refAnatT1)
    
    dimsList=['x','y','z']
    #brute force with matplotlib
   
    for iDims in list(range(len(refT1Shape))):
   
        if refT1Shape[iDims]<=overlayShape[iDims]:
            subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],refT1Res[iDims])
        else:
            subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],overlayRes[iDims])
        #get the desired broadcast shape and delete current dim value
        broadcastShape=list(refAnatT1.shape)
        del broadcastShape[iDims]
            
            #iterate across slices
        for iSlices in list(range(len(subjectSpaceSlices))):
            #set the slice list entry to the appropriate singular value
            #THE SOLUTION WAS SO OBVIOUS. DONT USE A SINGLE SLICE FOR BOTH THE
            #REFERNCE AND THE OVERLAY.  DUH!
            #actually this doesn't matter if we resample
            currentRefSlice=wmaPyTools.roiTools.makePlanarROI(refAnatT1, subjectSpaceSlices[iSlices], dimsList[iDims])
            #could be an issue if overlay nifti is signifigantly smaller
            currentOverlaySlice=wmaPyTools.roiTools.makePlanarROI(overlayNifti, subjectSpaceSlices[iSlices], dimsList[iDims])
    
            #set up the figure
            fig,ax = plt.subplots()
            ax.axis('off')
            #kind of overwhelming to do this in one line
            refData=np.rot90(np.reshape(refAnatT1.get_data()[currentRefSlice.get_data().astype(bool)],broadcastShape),1)
            plt.imshow(refData, cmap='gray', interpolation='antialiased')
            #kind of overwhelming to do this in one line
            overlayData=np.rot90(np.reshape(overlayNifti.get_data()[currentOverlaySlice.get_data().astype(bool)],broadcastShape),1)
            #lets mask out the background
            [unique, counts] = np.unique(overlayData, return_counts=True)
            backgroundVal=unique[np.where(np.max(counts)==counts)[0]]
            
            plt.imshow(np.ma.masked_where(overlayData==backgroundVal,overlayData), cmap='jet', alpha=.75, interpolation='antialiased',vmin=0,vmax=np.nanmax(overlayNifti.get_data()))
            curFig=plt.gcf()
            cbaxes = inset_axes(curFig.gca(), width="5%", height="80%", loc=5) 
            plt.colorbar(cax=cbaxes, ticks=[0.,np.nanmax(overlayNifti.get_data())], orientation='vertical')
            
            #put some text about the current dimension and slice
            yLims=curFig.gca().get_ylim()
            #xLims=curFig.gca().get_xlim()
            #this changes the resolution of the figure itself, and breaks everything
            #fix later
            # curFig.gca().text(0, yLims[0]-1, dimsList[iDims] + ' = ' + str(subjectSpaceSlices[iSlices]),
            # bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})

            
            curFig.gca().yaxis.set_ticks_position('left')
            curFig.gca().tick_params( colors='white')
            # we use *2 in order to afford room for the subsequent blended images
            figName='dim_' + str(iDims) +'_'+  str(iSlices).zfill(3)+'.png'
            plt.savefig(os.path.join(saveDir,figName),bbox_inches='tight',pad_inches=0.0)
            plt.close()

def multiTileOverlay_wrap(overlayNifti,refAnatT1,saveDir,figName,noEmpties=True,postClean=True):
    
    import nibabel as nib
    #use dipy to create the density mask
    from dipy.tracking import utils
    import numpy as np
    from glob import glob
    import os
    from nilearn.image import reorder_img  
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import figure
    from nilearn.image import crop_img, resample_img 
    import matplotlib.pyplot as plt
    import dipy.tracking.utils as ut
    import wmaPyTools.roiTools  
    from dipy.segment.mask import bounding_box
    
    refAnatT1=reorder_img(refAnatT1)
    overlayNifti=reorder_img(overlayNifti)
    
    generateAnatOverlayXSections(overlayNifti,refAnatT1,saveDir)
    
    imgSpaceMins, imgSpaceMaxs=bounding_box(overlayNifti.get_data())
    
    overlayShape=overlayNifti.shape
    refT1Shape=refAnatT1.shape
    
    

    
    import os        
    from PIL import Image
    from glob import glob
    #create blened images to smooth transitions between slices
  
    for iDims in list(range(len(refAnatT1.shape))):
        #presumes continuous slice numbering
        if noEmpties:
            if refT1Shape[iDims]<=overlayShape[iDims]:
                imgSpaceSlices=np.arange(imgSpaceMins[iDims],imgSpaceMaxs[iDims])
            else:
                #I don't think this can happen
                imgSpaceSlices=np.arange(0,refT1Shape[iDims])
        else:
            
            imgSpaceSlices=np.arange(0,refT1Shape[iDims])
        
        
        dimStem='dim_' + str(iDims)
        #open the images
        img, *imgs = [Image.open(os.path.join(saveDir,dimStem+'_'+str(f).zfill(3)+'.png')) for f in  imgSpaceSlices]
        #find the number you'll be working with
        numImagesToTile=len(imgSpaceSlices)
        #find the lenght of the aggregated sides
        squareSide=np.ceil(np.sqrt(numImagesToTile)).astype(int)
        #get the dimensions of the images in this dimension
        imgPix=img.size
        #create a blank output array
        tileOut=np.zeros([imgPix[0]*squareSide,imgPix[1]*squareSide,np.asarray(img).shape[2]-1]).astype(np.uint8)
        #create the iterated bounds
        colBounds=np.arange(0,tileOut.shape[0],imgPix[0])
        rowBounds=np.arange(0,tileOut.shape[1],imgPix[1])
        #get the appropriate number of these to iterate across all imgs
        y, x=np.meshgrid(colBounds,rowBounds,indexing='ij')
        #turn into 1d vectors
        yBoundVec=np.ravel(y)
        xBoundVec=np.ravel(x)
        for iImgs in range(len(imgs)):
            #tileOut[yBoundVec[iImgs]:yBoundVec[iImgs]+imgPix[1],xBoundVec[iImgs]:xBoundVec[iImgs]+imgPix[0],:]=np.transpose(np.asarray(imgs[iImgs]),[1,0,2]).astype(np.uint8)[:,:,0:3]
            tileOut[yBoundVec[iImgs]:yBoundVec[iImgs]+imgPix[0],xBoundVec[iImgs]:xBoundVec[iImgs]+imgPix[1],:]=np.transpose(np.asarray(imgs[iImgs]),[1,0,2]).astype(np.uint8)[:,:,0:3]

        im=Image.fromarray(np.fliplr(np.rot90(tileOut,3)))
        
        im.save(os.path.join(saveDir,figName+'_'+dimStem+'.png'), format='png')
        plt.close('all')
        #conditionally remove the generated files
        if postClean:
            [os.remove(ipaths) for ipaths in sorted(glob(os.path.join(saveDir,dimStem+'_*.png')))]    
        
def multiTileDensity(streamlines,refAnatT1,saveDir,tractName,densityThreshold=0,noEmpties=True):
    """
    

    Parameters
    ----------
    overlayNifti : TYPE
        DESCRIPTION.
    refAnatT1 : TYPE
        DESCRIPTION.
    saveDir : TYPE
        DESCRIPTION.
    noEmpties : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    import nibabel as nib
    #use dipy to create the density mask
    from dipy.tracking import utils
    import numpy as np
    from glob import glob
    import os
    from nilearn.image import reorder_img  
    
    #has to be done before matplotlib import
    #check to see if in docker container
    import wmaPyTools.genUtils
    if wmaPyTools.genUtils.is_docker():
        import matplotlib as mpl
        mpl.use('Agg')
        # print('Docker execution detected\nUsing xvfbwrapper for virtual display')
        # #borrowing from
        # #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
        # from xvfbwrapper import Xvfb

        # vdisplay = Xvfb()
        # vdisplay.start()
    #stop code    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import figure
    from nilearn.image import crop_img, resample_img 
    import matplotlib.pyplot as plt
    import dipy.tracking.utils as ut
    import wmaPyTools.roiTools    
    
    #Crop initial T1
    #crop the anatomy back down in case it has gotten overly widened
    refAnatT1=crop_img(refAnatT1)
    
    #get tract density nifti
    tractDensityNifti=ut.density_map(streamlines, refAnatT1.affine, refAnatT1.shape)
    #apply the density threshold
    tractDensityNifti[tractDensityNifti<(densityThreshold*np.max(tractDensityNifti))]=0
    #regenerate the nifti
    densityNifti = nib.nifti1.Nifti1Image(tractDensityNifti, refAnatT1.affine, refAnatT1.header)
    
    #RAS reoreintation
    refAnatT1=reorder_img(refAnatT1)
    overlayNifti=reorder_img(densityNifti)

    #obtain boundary coords in subject space in order to
    #use plane generation function
    convertedBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(refAnatT1)
    #overlayBoundCoords=subjectSpaceMaskBoundaryCoords(overlayNifti)
    
    #find which input has the highest resolution
    refAnatT1Resolution=refAnatT1.header.get_zooms()
    overlayResolution=overlayNifti.header.get_zooms()
    
    #not necessary with density mask
    # #change datatype so that nilearn doesn't mess up
    # #it doesn't like bool so we have to do something about that
    # #somehow the data can be set to nifti, but can't be extracted as nifti, so...
    # overlayNifti=nib.nifti1.Nifti1Image(overlayNifti.get_data(),overlayNifti.affine,overlayNifti.header)
    # overlayNifti.set_data_dtype(overlayNifti.get_data().dtype)
    
    #resample to the best resolution
    #but also assuming that the refernce anatomy is ultimately the shape that we want
    # #this is going to cause huge problems if the user passes in a super cropped overlay
    # THIS SHOULDN'T BE NECESSARY DUE TO THE USE OF THE REFT1 FOR DENSITY MAP
    # if np.prod(refAnatT1Resolution)>np.prod(overlayResolution):
    #     print('resampling reference anatomy to overlay')
    #     refAnatT1=resample_img(refAnatT1,target_affine=overlayNifti.affine[0:3,0:3])
    #     overlayNifti=resample_img(overlayNifti,target_affine=overlayNifti.affine[0:3,0:3])
    # else:
    #     print('resampling overlay to reference anatomy')
    #     refAnatT1=resample_img(refAnatT1,target_affine=refAnatT1.affine[0:3,0:3])
    #     overlayNifti=resample_img(overlayNifti,target_affine=refAnatT1.affine[0:3,0:3])
    
    # #crop the anatomy back down in case it has gotten overly widened
    # refAnatT1=crop_img(refAnatT1)
    
    # #now crop the overlay to the dimensions of the reference anatomy
    # overlayNifti=resample_img(overlayNifti,target_affine=refAnatT1.affine, target_shape=refAnatT1.shape)
    
    #compute the bounds of the actual density data   
    streamDensityBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(overlayNifti)
    
    dimsList=['x','y','z']
    #brute force with matplotlib
    
    for iDims in list(range(len(refAnatT1.shape))):
        #this assumes that get_zooms returns zooms in subject space and not image space orientation
        # which may not be a good assumption if the orientation is weird
        if noEmpties:
            # + and - 1 in odrer to provide a pad
            #NO PADDING, PADDING IS BAD
            if refAnatT1Resolution[iDims]<=overlayResolution[iDims]:
                subjectSpaceSlices=np.arange(streamDensityBoundCoords[0,iDims],streamDensityBoundCoords[1,iDims],refAnatT1Resolution[iDims])
            else:
                subjectSpaceSlices=np.arange(streamDensityBoundCoords[0,iDims],streamDensityBoundCoords[1,iDims],overlayResolution[iDims])
     
            
        #pick whichever input has the best resolution in this dimension
        else:
                if refAnatT1Resolution[iDims]<=overlayResolution[iDims]:
                    subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],refAnatT1Resolution[iDims])
                else:
                    subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],overlayResolution[iDims])
                #get the desired broadcast shape and delete current dim value
        broadcastShape=list(refAnatT1.shape)
        del broadcastShape[iDims]
            
            #iterate across slices
        for iSlices in list(range(len(subjectSpaceSlices))):
            #set the slice list entry to the appropriate singular value
            #THE SOLUTION WAS SO OBVIOUS. DONT USE A SINGLE SLICE FOR BOTH THE
            #REFERNCE AND THE OVERLAY.  DUH!
            #actually this doesn't matter if we resample
            currentRefSlice=wmaPyTools.roiTools.makePlanarROI(refAnatT1, subjectSpaceSlices[iSlices], dimsList[iDims])
            #could be an issue if overlay nifti is signifigantly smaller
            currentOverlaySlice=wmaPyTools.roiTools.makePlanarROI(overlayNifti, subjectSpaceSlices[iSlices], dimsList[iDims])
    
            #set up the figure
            fig,ax = plt.subplots()
            ax.axis('off')
            #kind of overwhelming to do this in one line
            refData=np.rot90(np.reshape(refAnatT1.get_data()[currentRefSlice.get_data().astype(bool)],broadcastShape),1)
            plt.imshow(refData, cmap='gray', interpolation='gaussian')
            #kind of overwhelming to do this in one line
            overlayData=np.rot90(np.reshape(overlayNifti.get_data()[currentOverlaySlice.get_data().astype(bool)],broadcastShape),1)
            #lets mask out the background
            [unique, counts] = np.unique(overlayData, return_counts=True)
            backgroundVal=unique[np.where(np.max(counts)==counts)[0]]
            
            plt.imshow(np.ma.masked_where(overlayData==backgroundVal,overlayData), cmap='jet', alpha=.75, interpolation='antialiased',vmin=0,vmax=np.nanmax(overlayNifti.get_data()))
            curFig=plt.gcf()
            yLims=sorted(curFig.gca().get_ylim())
            xLims=sorted(curFig.gca().get_xlim())
            #this changes the resolution of the figure itself, and breaks everything
            #fix later
            
            #also, it's all wonky, so the min and max as well as axis orienations are flipped
            curFig.gca().text((yLims[1])*.02, (yLims[1])*.08, dimsList[iDims] + ' = ' + str(np.round(subjectSpaceSlices[iSlices],1)), color='white', fontsize=15)
            # bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
            
            #be careful here, modifying the cbaxes  modifies the current figure
            cbaxes = inset_axes(curFig.gca(), width="5%", height="80%", loc=5) 
            plt.colorbar(cax=cbaxes, ticks=[0.,np.nanmax(overlayNifti.get_data())], orientation='vertical')
            
            
            #put some text about the current dimension and slice
        
            
            
            
            curFig.gca().yaxis.set_ticks_position('left')
            curFig.gca().tick_params(colors='white')
            
            
            # we use *2 in order to afford room for the subsequent blended images
            figName='dim_' + str(iDims) +'_'+  str(iSlices*2).zfill(3)+'.png'
            plt.savefig(figName,bbox_inches='tight',pad_inches=0.0)
            plt.clf()
                
    import os        
    from PIL import Image
    from glob import glob
    #create blened images to smooth transitions between slices
  
    for iDims in list(range(len(refAnatT1.shape))):
        dimStem='dim_' + str(iDims)
        #open the images
        img, *imgs = [Image.open(f) for f in sorted(glob(dimStem+'_*.png'))]
        #find the number you'll be working with
        numImagesToTile=len(sorted(glob(dimStem+'_*.png')))
        #find the lenght of the aggregated sides
        squareSide=np.ceil(np.sqrt(numImagesToTile)).astype(int)
        #get the dimensions of the images in this dimension
        imgPix=img.size
        #create a blank output array
        tileOut=np.zeros([imgPix[0]*squareSide,imgPix[1]*squareSide,np.asarray(img).shape[2]-1]).astype(np.uint8)
        #create the iterated bounds
        colBounds=np.arange(0,tileOut.shape[0],imgPix[0])
        rowBounds=np.arange(0,tileOut.shape[1],imgPix[1])
        #get the appropriate number of these to iterate across all imgs
        y, x=np.meshgrid(colBounds,rowBounds,indexing='ij')
        #turn into 1d vectors
        yBoundVec=np.ravel(y)
        xBoundVec=np.ravel(x)
        for iImgs in range(len(imgs)):
            #tileOut[yBoundVec[iImgs]:yBoundVec[iImgs]+imgPix[1],xBoundVec[iImgs]:xBoundVec[iImgs]+imgPix[0],:]=np.transpose(np.asarray(imgs[iImgs]),[1,0,2]).astype(np.uint8)[:,:,0:3]
            tileOut[yBoundVec[iImgs]:yBoundVec[iImgs]+imgPix[0],xBoundVec[iImgs]:xBoundVec[iImgs]+imgPix[1],:]=np.transpose(np.asarray(imgs[iImgs]),[1,0,2]).astype(np.uint8)[:,:,0:3]

        im=Image.fromarray(np.fliplr(np.rot90(tileOut,3)))
        
        im.save(os.path.join(saveDir,tractName+'_'+dimStem+'.png'), format='png')
        plt.close('all')

        [os.remove(ipaths) for ipaths in sorted(glob(dimStem+'*.png'))]
    
    #stop code    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()
    
    
def crossSectionGIFsFromOverlay(overlayNifti,refAnatT1,saveDir,figName,postClean=True):
    
    import matplotlib.pyplot as plt
    generateAnatOverlayXSections(overlayNifti,refAnatT1,saveDir)
    
    import os        
    from PIL import Image
    from glob import glob

    for iDims in list(range(len(refAnatT1.shape))):
        dimStem='dim_' + str(iDims)
        img, *imgs = [Image.open(f) for f in sorted(glob(os.path.join(saveDir,dimStem+'_*.png')))]
        img.save(os.path.join(saveDir,figName+'_'+dimStem+'.gif'), format='GIF', append_images=imgs,
                 save_all=True, duration=1, loop=0)
        plt.close('all')

        if postClean:
            [os.remove(ipaths) for ipaths in sorted(glob(os.path.join(saveDir,dimStem+'_*.png')))]   
        
def crossSectionGIFsFromNifti(overlayNifti,refAnatT1,saveDir, blendOption=False):
    import nibabel as nib
    #use dipy to create the density mask
    from dipy.tracking import utils
    import numpy as np
    from glob import glob
    import os
    from nilearn.image import reorder_img  
        
    #has to be done before matplotlib import
    #check to see if in docker container
    import wmaPyTools.genUtils
    if wmaPyTools.genUtils.is_docker():
        import matplotlib as mpl
        mpl.use('Agg')
        # print('Docker execution detected\nUsing xvfbwrapper for virtual display')
        # #borrowing from
        # #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
        # from xvfbwrapper import Xvfb

        # vdisplay = Xvfb()
        # vdisplay.start()
    #stop code    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import figure
    from nilearn.image import crop_img, resample_img 
    import matplotlib.pyplot as plt
    import wmaPyTools.roiTools  
    
   
    #resample crop (doesn't seem to work)
    #[refAnatT1,overlayNifti]=dualCropNifti(refAnatT1,overlayNifti)
    #ok, but if we assume that the overlay is *always* going to be smaller than the 
    #reference, we can crop the overlay safely and work adaptively.  Why would
    #you even have overlay **outside** of the refernce
    #nilearn doesn't handle NAN gracefully, so we have to be inelegant
    #overlayNifti=nib.nifti1.Nifti1Image(np.nan_to_num(overlayNifti.get_data()), overlayNifti.affine, overlayNifti.header)
    #croppedOverlayNifti=crop_img(overlayNifti)
    
    #RAS reoreintation
    refAnatT1=reorder_img(refAnatT1)
    overlayNifti=reorder_img(overlayNifti)
                 
    
    #refuses to plot single slice, single image
    #from nilearn.plotting import plot_stat_map
    #outImg=plot_stat_map(stat_map_img=densityNifti,bg_img=refAnatT1, cut_coords= 1,display_mode='x',cmap='viridis')
   
    
    #obtain boundary coords in subject space in order to
    #use plane generation function
    convertedBoundCoords=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(refAnatT1)
    #overlayBoundCoords=subjectSpaceMaskBoundaryCoords(overlayNifti)
    
    #find which input has the highest resolution
    refAnatT1Resolution=refAnatT1.header.get_zooms()
    overlayResolution=overlayNifti.header.get_zooms()
    
    #change datatype so that nilearn doesn't mess up
    #it doesn't like bool so we have to do something about that
    #somehow the data can be set to nifti, but can't be extracted as nifti, so...
    overlayNifti=nib.nifti1.Nifti1Image(overlayNifti.get_data(),overlayNifti.affine,overlayNifti.header)
    overlayNifti.set_data_dtype(overlayNifti.get_data().dtype)
    
    #resample to the best resolution
    #but also assuming that the refernce anatomy is ultimately the shape that we want
    #this is going to cause huge problems if the user passes in a super cropped overlay
    if np.prod(refAnatT1Resolution)>np.prod(overlayResolution):
        print('resampling reference anatomy to overlay')
        refAnatT1=resample_img(refAnatT1,target_affine=overlayNifti.affine[0:3,0:3])
        overlayNifti=resample_img(overlayNifti,target_affine=overlayNifti.affine[0:3,0:3])
    else:
        print('resampling overlay to reference anatomy')
        refAnatT1=resample_img(refAnatT1,target_affine=refAnatT1.affine[0:3,0:3])
        overlayNifti=resample_img(overlayNifti,target_affine=refAnatT1.affine[0:3,0:3])
    
    #crop the anatomy back down in case it has gotten overly widened
    refAnatT1=crop_img(refAnatT1)
    
    #now crop the overlay to the dimensions of the reference anatomy
    overlayNifti=resample_img(overlayNifti,target_affine=refAnatT1.affine, target_shape=refAnatT1.shape)
    #WARNING THIS INTRODUCES NIQUEST SAMPLING ERRORS INTO THE OVERLAY
    
    dimsList=['x','y','z']
    #brute force with matplotlib
    
    for iDims in list(range(len(refAnatT1.shape))):
        #this assumes that get_zooms returns zooms in subject space and not image space orientation
        # which may not be a good assumption if the orientation is weird
        
        #pick whichever input has the best resolution in this dimension
        if refAnatT1Resolution[iDims]<=overlayResolution[iDims]:
            subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],refAnatT1Resolution[iDims])
        else:
            subjectSpaceSlices=np.arange(convertedBoundCoords[0,iDims],convertedBoundCoords[1,iDims],overlayResolution[iDims])
        #get the desired broadcast shape and delete current dim value
        broadcastShape=list(refAnatT1.shape)
        del broadcastShape[iDims]
        
        #iterate across slices
        for iSlices in list(range(len(subjectSpaceSlices))):
            #set the slice list entry to the appropriate singular value
            #THE SOLUTION WAS SO OBVIOUS. DONT USE A SINGLE SLICE FOR BOTH THE
            #REFERNCE AND THE OVERLAY.  DUH!
            #actually this doesn't matter if we resample
            currentRefSlice=wmaPyTools.roiTools.makePlanarROI(refAnatT1, subjectSpaceSlices[iSlices], dimsList[iDims])
            #could be an issue if overlay nifti is signifigantly smaller
            currentOverlaySlice=wmaPyTools.roiTools.makePlanarROI(overlayNifti, subjectSpaceSlices[iSlices], dimsList[iDims])

            #set up the figure
            fig,ax = plt.subplots()
            ax.axis('off')
            #kind of overwhelming to do this in one line
            refData=np.rot90(np.reshape(refAnatT1.get_data()[currentRefSlice.get_data().astype(bool)],broadcastShape),1)
            plt.imshow(refData, cmap='gray', interpolation='gaussian')
            #kind of overwhelming to do this in one line
            overlayData=np.rot90(np.reshape(overlayNifti.get_data()[currentOverlaySlice.get_data().astype(bool)],broadcastShape),1)
            #lets mask out the background
            [unique, counts] = np.unique(overlayData, return_counts=True)
            backgroundVal=unique[np.where(np.max(counts)==counts)[0]]
            
            plt.imshow(np.ma.masked_where(overlayData==backgroundVal,overlayData), cmap='jet', alpha=.75, interpolation='gaussian',vmin=0,vmax=np.nanmax(overlayNifti.get_data()))
            curFig=plt.gcf()
            cbaxes = inset_axes(curFig.gca(), width="5%", height="80%", loc=5) 
            plt.colorbar(cax=cbaxes, ticks=[0.,np.nanmax(overlayNifti.get_data())], orientation='vertical')
            curFig.gca().yaxis.set_ticks_position('left')
            curFig.gca().tick_params( colors='white')
            # we use *2 in order to afford room for the subsequent blended images
            figName='dim_' + str(iDims) +'_'+  str(iSlices*2).zfill(3)+'.png'
            plt.savefig(figName,bbox_inches='tight',pad_inches=0.0)
            plt.clf()
            
    import os        
    from PIL import Image
    from glob import glob
    #create blened images to smooth transitions between slices
    if blendOption:
        for iDims in list(range(len(refAnatT1.shape))):
            dimStem='dim_' + str(iDims)
            imageList=sorted(glob(dimStem+'*.png'))
            for iImages in list(range(len(imageList)-1)):
                thisImage=Image.open(imageList[iImages])
                nextImage=Image.open(imageList[iImages+1])
                blendedImage = Image.blend(thisImage, nextImage, alpha=0.5)
                # 1 + 2 * iImages fills in the name space we left earlier
                figName='dim_' + str(iDims) +'_'+  str(1+iImages*2).zfill(3)+'.png'
                blendedImage.save(figName,'png')
  
    for iDims in list(range(len(refAnatT1.shape))):
        dimStem='dim_' + str(iDims)
        img, *imgs = [Image.open(f) for f in sorted(glob(dimStem+'*.png'))]
        img.save(os.path.join(saveDir,dimStem+'.gif'), format='GIF', append_images=imgs,
                 save_all=True, duration=1, loop=0)
        plt.close('all')

        [os.remove(ipaths) for ipaths in sorted(glob(dimStem+'*.png'))]
        
    # #stop code    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()

def densityGifsOfTract(tractStreamlines,referenceAnatomy,saveDir,tractName):
    import os
    import nibabel as nib
    from glob import glob
    import numpy as np
    import dipy   

    import dipy.tracking.utils as ut
    
    tractMask=ut.density_map(tractStreamlines, referenceAnatomy.affine, referenceAnatomy.shape)
    densityNifti = nib.nifti1.Nifti1Image(tractMask, referenceAnatomy.affine, referenceAnatomy.header)
    
    #now make the niftiGifs
    crossSectionGIFsFromNifti(densityNifti,referenceAnatomy,saveDir)   

    filesToRename=[os.path.join(saveDir,'dim_'+ str(iDim) +'.gif') for iDim in range(3)]
    
    for iFiles in filesToRename:
        [path, file]=os.path.split(iFiles)
        os.rename(iFiles,os.path.join(path,tractName+'_'+file))

#def densityGifsOfOverlay(tractStreamlines,referenceAnatomy,saveDir,tractName):
        
def radialTractEndpointFingerprintPlot(tractStreamlines,atlas,atlasLookupTable,tractName='tract',forcePlotLabels=None,saveDir=None,color=False):
    """radialTractEndpointFingerprintPlot(tractStreamlines,atlas,atlasLookupTable,tractName=None,saveDir=None)
    A function used to generate radial fingerprint plots for input tracts, as 
    found in Folloni 2021.  Plots absolute streamline count

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
    tractName : string, optional
        The name of the tract to be used as a label in the output figure. No 
        input will result in there being no corresponding label. The default is None.
    forcePlotLabels: array-like / list of int, optional
        A list or array of integer values corresponding to labels in the input
        atlas / atlasLookupTable that will be included in the output plot 
        REGARDLESS of whether streamlines connect to this region or not (e.g.
        this option will force the plotting of 0 values for specificed labels)
    saveDir : TYPE, optional
        The directory in which to save the resultant radial plot. No input
        will result in no save. The default is None.

    Returns
    -------
    The figure generated

    """ 
    import os
    import matplotlib.pyplot as plt
    import wmaPyTools.analysisTools
    import pandas as pd
    
    #just use this to get the column names, you can't be sure that all the names
    #are there
    [renumberedAtlasNifti,reducedLookUpTable]=wmaPyTools.analysisTools.reduceAtlasAndLookupTable(atlas,atlasLookupTable,removeAbsentLabels=False)
    columnNames=reducedLookUpTable.columns
    
    #don't pass the output of this to the next part, because the label numbers no longer match the atlas
    [endpointsDF1, endpointsDF2]=wmaPyTools.analysisTools.quantifyTractEndpoints(tractStreamlines,atlas,atlasLookupTable)
    
    
    if color==True:
       endpointsDF1['color']=''
       endpointsDF2['color']=''
       streamColors, colorLut=colorStreamEndpointsFromParc(tractStreamlines,atlas)
       #place label names in the color lut
       colorLut['name']=''
       for iRows,iLabels in enumerate(colorLut['labels'].to_list()):
            colorLut['name'].iloc[iRows]=reducedLookUpTable.iloc[(reducedLookUpTable.iloc[:,0]==iLabels).to_numpy(),1].values[0]
       #now do it the other way for the label values  
       for iRows,iNames in enumerate(colorLut['name'].to_list()):
           if iNames in endpointsDF1['labelNames'].to_list():
               endpointsDF1['color'].iloc[(endpointsDF1['labelNames']==iNames).to_numpy()]=pd.Series([colorLut['rgb_Color'].iloc[iRows]])
           if iNames in endpointsDF2['labelNames'].to_list():
               endpointsDF2['color'].iloc[(endpointsDF2['labelNames']==iNames).to_numpy()]=pd.Series([colorLut['rgb_Color'].iloc[iRows]])
      
        
        
    #get the requested labels, if any
    if not forcePlotLabels==None:
        #get the sub table for the requeseted labels
        forceTable=atlasLookupTable[atlasLookupTable[columnNames[0]].isin(forcePlotLabels)]
        #find the labels that aren't already there
        missingLabels1=forceTable[columnNames[1]][~forceTable[columnNames[1]].isin(endpointsDF1['labelNames'])]
        missingLabels2=forceTable[columnNames[1]][~forceTable[columnNames[1]].isin(endpointsDF2['labelNames'])]
        
        #now append them to the tables
        endpointsDF1=endpointsDF1.append(pd.DataFrame(missingLabels1,columns=['labelNames']),ignore_index=True)
        endpointsDF2=endpointsDF2.append(pd.DataFrame(missingLabels2,columns=['labelNames']),ignore_index=True)
        
        #now set the nans to zero
        endpointsDF1['endpointCounts']= endpointsDF1['endpointCounts'].fillna(0)
        endpointsDF2['endpointCounts']= endpointsDF2['endpointCounts'].fillna(0)
    
    if color==True:
        figure1=basicRadarPlot(list(endpointsDF1['labelNames']),list(endpointsDF1['endpointCounts']),COLORS=endpointsDF1['color'].to_list())
        figure2=basicRadarPlot(list(endpointsDF2['labelNames']),list(endpointsDF2['endpointCounts']),COLORS=endpointsDF2['color'].to_list())
    else:
        figure1=basicRadarPlot(list(endpointsDF1['labelNames']),list(endpointsDF1['endpointCounts']))
        figure2=basicRadarPlot(list(endpointsDF2['labelNames']),list(endpointsDF2['endpointCounts']))
    
    figure1.get_axes()[0].set_title(tractName+'\nRAS endpoints')
    figure1.get_axes()[0].set_facecolor([0,0,1,.15])
    #figure1.patch.set_facecolor([0,0,1,.2])
    figure2.get_axes()[0].set_title(tractName+'\nLPI endpoints')
    figure2.get_axes()[0].set_facecolor([1,0,0,.15])
    #figure2.patch.set_facecolor([1,0,0,.2])
    
    if saveDir==None:
        saveDir=os.getcwd()
        
    figure1.savefig(os.path.join(saveDir,tractName+'_RAS_endpointMap.eps'))
    figure2.savefig(os.path.join(saveDir,tractName+'_LPI_endpointMap.eps'))
    
def radialTractEndpointFingerprintPlot_MultiSubj(tractTractogramList,atlasList,atlasLookupTable,tractName='tract',forcePlotLabels=None,saveDir=None):
    """
    A function used to generate radial fingerprint plots for input tracts, as 
    found in Folloni 2021.  Plots proportional streamline count, averaged
    across input variants, with error bars.  Secret Trick:  if you input two
    instances of the same tract/atlas (as opposed to inputs for different
    subjects), will simply plot the proportion for that single subject.
    

    Parameters
    ----------
    tractTractogramList : list of streamlines type
        Streamlines corresponding to the tract of interest
    atlasList: list of  Nifti, int based
        A nifti atlas that will be used to determine the endpoint connectivity
    atlasLookupTable : pandas dataframe or file loadable to pandas dataframe
        A dataframe of the atlas lookup table which includes the labels featured
        in the atlas and their identities.  These identities will be used
        to label the periphery of the radial plot.
    tractName : string, optional
        The name of the tract to be used as a label in the output figure. No 
        input will result in there being no corresponding label. The default is None.
    forcePlotLabels: array-like / list of int, optional
        A list or array of integer values corresponding to labels in the input
        atlas / atlasLookupTable that will be included in the output plot 
        REGARDLESS of whether streamlines connect to this region or not (e.g.
        this option will force the plotting of 0 values for specificed labels)
    saveDir : TYPE, optional
        The directory in which to save the resultant radial plot. No input
        will result in no save. The default is None.
    

    Returns
    -------
    None.

    """
    import pandas as pd
    import numpy as np
    import os
    from matplotlib import pyplot as plt
    import wmaPyTools.analysisTools  
    
    #just use this to get the column names, you can't be sure that all the names
    #are there
    [renumberedAtlasNifti,reducedLookUpTable]=wmaPyTools.analysisTools.reduceAtlasAndLookupTable(atlasList[0],atlasLookupTable,removeAbsentLabels=False)
    columnNames=reducedLookUpTable.columns

    
    RASendpointData=[]
    LPIendpointData=[]
    for iTracts in range(len(tractTractogramList)):
        [currentRAS,currentLPI]=wmaPyTools.analysisTools.quantifyTractEndpoints(tractTractogramList[iTracts],atlasList[iTracts],atlasLookupTable)
        RASendpointData.append(currentRAS)
        LPIendpointData.append(currentLPI)
        
    #normalize them
    for iTracts in range(len(tractTractogramList)):
        RASendpointData[iTracts]['endpointCounts']=RASendpointData[iTracts]['endpointCounts'].divide(RASendpointData[iTracts]['endpointCounts'].sum())
        LPIendpointData[iTracts]['endpointCounts']=LPIendpointData[iTracts]['endpointCounts'].divide(LPIendpointData[iTracts]['endpointCounts'].sum())
    
    firstRASDF=RASendpointData[0]
    firstLPIDF=LPIendpointData[0]
    for iTracts in range(1,len(tractTractogramList)):
        firstRASDF=pd.merge(firstRASDF,RASendpointData[iTracts], on='labelNames', how='outer')
        firstLPIDF=pd.merge(firstLPIDF,LPIendpointData[iTracts], on='labelNames', how='outer')
    
    #set NaNs to 0
    firstRASDF=firstRASDF.fillna(0)
    firstLPIDF=firstLPIDF.fillna(0)
    
    #compute means and variances
    firstLPIDF[['meanProportion','proportionSTD']]=pd.DataFrame(np.atleast_2d(np.squeeze(np.asarray([np.mean(firstLPIDF.iloc[:,1:-1],axis=1),np.std(firstLPIDF.iloc[:,1:-1],axis=1)])).T), index=firstLPIDF.index)
    firstRASDF[['meanProportion','proportionSTD']]=pd.DataFrame(np.atleast_2d(np.squeeze(np.asarray([np.mean(firstRASDF.iloc[:,1:-1],axis=1),np.std(firstRASDF.iloc[:,1:-1],axis=1)])).T), index=firstRASDF.index)
    
    #arbitrary criteria for mean proportion
    minThresh=.01
    #split the dataframe in ordert to get the common and uncommon endpoints
    firstLPIDFCommon= firstLPIDF[firstLPIDF['meanProportion'] >= minThresh]
    firstLPIDFUnCommon= firstLPIDF[firstLPIDF['meanProportion'] <= minThresh]
    firstRASDFCommon= firstRASDF[firstRASDF['meanProportion'] >= minThresh]
    firstRASDFUnCommon= firstRASDF[firstRASDF['meanProportion'] <= minThresh]
    
    #here we enforce the required labels by switching them over or filling them in
    #get the requested labels, if any
    if not forcePlotLabels==None:
        #get the sub table for the requeseted labels
        forceTable=atlasLookupTable[atlasLookupTable[columnNames[0]].isin(forcePlotLabels)]
        #check to see if they are in BOTH tables
        missingLabels1=forceTable[columnNames[1]][~forceTable[columnNames[1]].isin(firstLPIDF['labelNames'])]
        missingLabels2=forceTable[columnNames[1]][~forceTable[columnNames[1]].isin(firstRASDF['labelNames'])]
        
        #now append them to the uncommon tables
        firstLPIDFUnCommon=firstLPIDFUnCommon.append(pd.DataFrame(data=missingLabels1.tolist(),columns=['labelNames']),ignore_index=True)
        firstRASDFUnCommon=firstRASDFUnCommon.append(pd.DataFrame(data=missingLabels2.tolist(),columns=['labelNames']),ignore_index=True)
        
        #now set the nans to zero
        firstLPIDFUnCommon= firstLPIDFUnCommon.fillna(0)
        firstRASDFUnCommon= firstRASDFUnCommon.fillna(0)
        
        #then move over the relevant rows to the common table, a clever move
        firstLPIDFCommon=firstLPIDFCommon.append(firstLPIDFUnCommon[firstLPIDFUnCommon['labelNames'].isin(forceTable[columnNames[1]])],ignore_index=True)
        firstRASDFCommon=firstRASDFCommon.append(firstRASDFUnCommon[firstRASDFUnCommon['labelNames'].isin(forceTable[columnNames[1]])],ignore_index=True)
        
    #Plot the common endpoints
    figure1=basicRadarPlot(list(firstRASDFCommon['labelNames']),list(firstRASDFCommon['meanProportion']),metaValues=list(firstRASDFCommon['proportionSTD']))
    figure2=basicRadarPlot(list(firstLPIDFCommon['labelNames']),list(firstLPIDFCommon['meanProportion']),metaValues=list(firstLPIDFCommon['proportionSTD']))
    
    figure1.get_axes()[0].set_title(tractName+' RAS\n common endpoints\n',size=16)
    figure1.get_axes()[0].set_facecolor([0,0,1,.15])
    
    #find optimal location for axis label
    clearSpaceSizes=firstRASDFCommon['meanProportion']+firstRASDFCommon['proportionSTD']
    lowestLocation=np.where(np.min(clearSpaceSizes)==clearSpaceSizes)[0]
    #generate the angle locations
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(firstRASDFCommon['labelNames']), endpoint=False)
    #the .2 and .4 here need to be adaptive to the number of labels, not sure how yet though.
    locationSelect=ANGLES[lowestLocation[0]]-.2*((ANGLES[lowestLocation[0]]+1)-ANGLES[lowestLocation[0]])
    rotationAngle=(((ANGLES[lowestLocation[0]]+1)-.4*((ANGLES[lowestLocation[0]]+1)-ANGLES[lowestLocation[0]]))*57.295)/2
    
    figure1.get_axes()[0].text(locationSelect, np.max(firstRASDFCommon['meanProportion']*.5), "avg proportion\n of streamlines", rotation=rotationAngle+180, 
        ha="center", va="center", size=14, zorder=12)

    figure2.get_axes()[0].set_title(tractName+' LPI\n common endpoints')
    figure2.get_axes()[0].set_facecolor([1,0,0,.15])
    
    #find optimal location for axis label
    clearSpaceSizes=firstLPIDFCommon['meanProportion']+firstLPIDFCommon['proportionSTD']
    lowestLocation=np.where(np.min(clearSpaceSizes)==clearSpaceSizes)[0]
    #generate the angle locations
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(firstLPIDFCommon['labelNames']), endpoint=False)
    #the .2 and .4 here need to be adaptive to the number of labels, not sure how yet though.
    locationSelect=ANGLES[lowestLocation[0]]-.2*((ANGLES[lowestLocation[0]]+1)-ANGLES[lowestLocation[0]])
    rotationAngle=(((ANGLES[lowestLocation[0]]+1)-.4*((ANGLES[lowestLocation[0]]+1)-ANGLES[lowestLocation[0]]))*57.295)/2
    
    figure2.get_axes()[0].text(locationSelect, np.max(firstLPIDFCommon['meanProportion']), "avg proportion\n of streamlines", rotation=rotationAngle+180, 
        ha="center", va="center", size=12, zorder=12)
 
    
    if saveDir==None:
        saveDir=os.getcwd()
        
    figure1.savefig(os.path.join(saveDir,tractName+'_group_RAS_commonEndpointMap.svg'))
    figure2.savefig(os.path.join(saveDir,tractName+'_group_LPI_commonEndpointMap.svg'))
    plt.close()
    
    
    #Plot the UNcommon endpoints
    #but use try except
    
    try: 
    
        figure1=basicRadarPlot(list(firstRASDFUnCommon['labelNames']),list(firstRASDFUnCommon['meanProportion']),metaValues=list(firstRASDFUnCommon['proportionSTD']))
       
        figure1.get_axes()[0].set_title(tractName+' RAS\n *UN*common endpoints')
        figure1.get_axes()[0].set_facecolor([0,0,1,.15])
        figure1.get_axes()[0].text(0, np.max(firstRASDFUnCommon['meanProportion']), "avg proportion\n of streamlines", rotation=-69, 
            ha="center", va="center", size=12, zorder=12)
        
        figure1.savefig(os.path.join(saveDir,tractName+'_group_RAS_UNcommonEndpointMap.svg'))
        plt.close()
    except:
        print('no uncommon for RAS')
    
    try:
        figure2=basicRadarPlot(list(firstLPIDFUnCommon['labelNames']),list(firstLPIDFUnCommon['meanProportion']),metaValues=list(firstLPIDFUnCommon['proportionSTD']))
       
        figure2.get_axes()[0].set_title(tractName+' LPI\n *UN*common endpoints')
        figure2.get_axes()[0].set_facecolor([1,0,0,.15])
        figure2.get_axes()[0].text(0, np.max(firstLPIDFUnCommon['meanProportion']), "avg proportion\n of streamlines", rotation=-69, 
            ha="center", va="center", size=12, zorder=12)
        figure2.savefig(os.path.join(saveDir,tractName+'_group_LPI_UNcommonEndpointMap.svg'))
        plt.close() 
    except:
        print('no uncommon for LPI')
        
            
def radialTractEndpointFingerprintPlot_Norm(tract,atlas,atlasLookupTable,tractName='tract',forcePlotLabels=None,saveDir=None,color=False):
    """
    A function used to generate radial fingerprint plots for input tracts, as 
    found in Folloni 2021.  Norms by proportion of tract streamlines.  Outputs
    a 2x2 plot between common and uncommon.
    

    Parameters
    ----------
    tractTractogramList : list of streamlines type
        Streamlines corresponding to the tract of interest
    atlasList: list of  Nifti, int based
        A nifti atlas that will be used to determine the endpoint connectivity
    atlasLookupTable : pandas dataframe or file loadable to pandas dataframe
        A dataframe of the atlas lookup table which includes the labels featured
        in the atlas and their identities.  These identities will be used
        to label the periphery of the radial plot.
    tractName : string, optional
        The name of the tract to be used as a label in the output figure. No 
        input will result in there being no corresponding label. The default is None.
    forcePlotLabels: array-like / list of int, optional
        A list or array of integer values corresponding to labels in the input
        atlas / atlasLookupTable that will be included in the output plot 
        REGARDLESS of whether streamlines connect to this region or not (e.g.
        this option will force the plotting of 0 values for specificed labels)
    saveDir : TYPE, optional
        The directory in which to save the resultant radial plot. No input
        will result in no save. The default is None.
    

    Returns
    -------
    None.

    """
    import pandas as pd
    import numpy as np
    import os
        
    #has to be done before matplotlib import
    #check to see if in docker container
    import wmaPyTools.genUtils
    if wmaPyTools.genUtils.is_docker():
        import matplotlib as mpl
        mpl.use('Agg')
        # print('Docker execution detected\nUsing xvfbwrapper for virtual display')
        # #borrowing from
        # #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
        # from xvfbwrapper import Xvfb

        # vdisplay = Xvfb()
        # vdisplay.start()
    #stop code    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()
    
    from matplotlib import pyplot as plt
    import wmaPyTools.analysisTools  
  
     
    #just use this to get the column names, you can't be sure that all the names
    #are there
    [renumberedAtlasNifti,reducedLookUpTable]=wmaPyTools.analysisTools.reduceAtlasAndLookupTable(atlas,atlasLookupTable,removeAbsentLabels=False)
    columnNames=reducedLookUpTable.columns

       
    [currentRAS,currentLPI]=wmaPyTools.analysisTools.quantifyTractEndpoints(tract,atlas,atlasLookupTable)
    RASendpointData=currentRAS
    LPIendpointData=currentLPI
        
    #normalize them
    RASendpointData['endpointCounts']=RASendpointData['endpointCounts'].divide(RASendpointData['endpointCounts'].sum())
    LPIendpointData['endpointCounts']=LPIendpointData['endpointCounts'].divide(LPIendpointData['endpointCounts'].sum())


    #get the colors
    if color==True:
       RASendpointData['color']=''
       LPIendpointData['color']=''
       streamColors, colorLut=colorStreamEndpointsFromParc(tract,atlas)
       #place label names in the color lut
       colorLut['name']=''
       for iRows,iLabels in enumerate(colorLut['labels'].to_list()):
            colorLut['name'].iloc[iRows]=reducedLookUpTable.iloc[(reducedLookUpTable.iloc[:,0]==iLabels).to_numpy(),1].values[0]
       #now do it the other way for the label values  
       for iRows,iNames in enumerate(colorLut['name'].to_list()):
           if iNames in RASendpointData['labelNames'].to_list():
               RASendpointData['color'].iloc[(RASendpointData['labelNames']==iNames).to_numpy()]=pd.Series([colorLut['rgb_Color'].iloc[iRows]])
           if iNames in LPIendpointData['labelNames'].to_list():
               LPIendpointData['color'].iloc[(LPIendpointData['labelNames']==iNames).to_numpy()]=pd.Series([colorLut['rgb_Color'].iloc[iRows]])
      
        
        

    #arbitrary criteria for mean proportion
    minThresh=.01
    #split the dataframe in ordert to get the common and uncommon endpoints
    firstLPIDFCommon= LPIendpointData[LPIendpointData['endpointCounts'] >= minThresh]
    firstLPIDFUnCommon= LPIendpointData[LPIendpointData['endpointCounts'] <= minThresh]
    firstRASDFCommon= RASendpointData[RASendpointData['endpointCounts'] >= minThresh]
    firstRASDFUnCommon= RASendpointData[RASendpointData['endpointCounts'] <= minThresh]
    
    #here we enforce the required labels by switching them over or filling them in
    #get the requested labels, if any
    if not forcePlotLabels==None:
        #get the sub table for the requeseted labels
        forceTable=atlasLookupTable[atlasLookupTable[columnNames[0]].isin(forcePlotLabels)]
        #check to see if they are in BOTH tables
        missingLabels1=forceTable[columnNames[1]][~forceTable[columnNames[1]].isin(LPIendpointData['labelNames'])]
        missingLabels2=forceTable[columnNames[1]][~forceTable[columnNames[1]].isin(RASendpointData['labelNames'])]
        
        #now append them to the uncommon tables
        firstLPIDFUnCommon=firstLPIDFUnCommon.append(pd.DataFrame(data=missingLabels1.tolist(),columns=['labelNames']),ignore_index=True)
        firstRASDFUnCommon=firstRASDFUnCommon.append(pd.DataFrame(data=missingLabels2.tolist(),columns=['labelNames']),ignore_index=True)
        
        #now set the nans to zero
        firstLPIDFUnCommon= firstLPIDFUnCommon.fillna(0)
        firstRASDFUnCommon= firstRASDFUnCommon.fillna(0)
        
        #then move over the relevant rows to the common table, a clever move
        firstLPIDFCommon=firstLPIDFCommon.append(firstLPIDFUnCommon[firstLPIDFUnCommon['labelNames'].isin(forceTable[columnNames[1]])],ignore_index=True)
        firstRASDFCommon=firstRASDFCommon.append(firstRASDFUnCommon[firstRASDFUnCommon['labelNames'].isin(forceTable[columnNames[1]])],ignore_index=True)
        
        
    fig, axs = plt.subplots(2, 2,subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(wspace=-.4,hspace=0.7)
    fig.figsize=(12, 12)
    fontsize=6
    #TODO
    #https://stackoverflow.com/questions/71427116/matplotlib-align-subplot-titles-to-top-of-figure
    
    #Plot the common endpoints
    if color==True:
        axs[0,0]=basicRadarPlot_ax(list(firstRASDFCommon['labelNames']),list(firstRASDFCommon['endpointCounts']),ax=axs[0,0],COLORS=firstRASDFCommon['color'].to_list())
        axs[0,1]=basicRadarPlot_ax(list(firstLPIDFCommon['labelNames']),list(firstLPIDFCommon['endpointCounts']),ax=axs[0,1],COLORS=firstLPIDFCommon['color'].to_list())
        axs[0,0].set_title('RAS common endpoints',size=fontsize)
        axs[0,0].set_facecolor([0,0,1,.15])
        axs[0,1].set_title('LPI common endpoints',size=fontsize)
        axs[0,1].set_facecolor([1,0,0,.15])  
    else:
        axs[0,0]=basicRadarPlot_ax(list(firstRASDFCommon['labelNames']),list(firstRASDFCommon['endpointCounts']),ax=axs[0,0])
        axs[0,1]=basicRadarPlot_ax(list(firstLPIDFCommon['labelNames']),list(firstLPIDFCommon['endpointCounts']),ax=axs[0,1])
        axs[0,0].set_title('RAS common endpoints',size=fontsize)
        axs[0,0].set_facecolor([0,0,1,.15])
        axs[0,1].set_title('LPI common endpoints',size=fontsize)
        axs[0,1].set_facecolor([1,0,0,.15])  


    
    #Plot the UNcommon endpoints
    #but use try except
    
    try: 
    
        if color==True:
            axs[1,0]=basicRadarPlot_ax(list(firstRASDFUnCommon['labelNames']),list(firstRASDFUnCommon['endpointCounts']),ax=axs[1,0],COLORS=firstRASDFUnCommon['color'].to_list())
        else:
            axs[1,0]=basicRadarPlot_ax(list(firstRASDFUnCommon['labelNames']),list(firstRASDFUnCommon['endpointCounts']),ax=axs[1,0])
        
        #figure3=basicRadarPlot(list(firstRASDFUnCommon['labelNames']),list(firstRASDFUnCommon['endpointCounts']))
       
        axs[1,0].set_title('RAS *UN*common endpoints',size=fontsize)
        axs[1,0].set_facecolor([0,0,1,.15])
        
    except:
        print('no uncommon for RAS')
    
    try:
        
        if color==True:
            axs[1,1]=basicRadarPlot_ax(list(firstLPIDFUnCommon['labelNames']),list(firstLPIDFUnCommon['endpointCounts']),ax=axs[1,1],COLORS=firstLPIDFUnCommon['color'].to_list())
        else:
            axs[1,1]=basicRadarPlot_ax(list(firstLPIDFUnCommon['labelNames']),list(firstLPIDFUnCommon['endpointCounts']),ax=axs[1,1])
       
        axs[1,1].set_title('LPI *UN*common endpoints',size=fontsize)
        axs[1,1].set_facecolor([1,0,0,.15])

   
    except:
        print('no uncommon for LPI')    
        
    fig.suptitle(tractName+' endpoints fingerprint', fontsize=12)
    fig.tight_layout()
    plt.show()
    #save the overarching figure
    if not saveDir == None:
        fig.savefig(os.path.join(saveDir,tractName+'_endpointFingerprint_Normed.svg'),dpi=800,bbox_inches='tight')
    else:
        fig.savefig(tractName+'_endpointFingerprint_Normed.svg',dpi=800,bbox_inches='tight')
    plt.close() 
    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()

    
def basicRadarPlot(labels,values, metaValues=None,COLORS=None):
    """
    https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib
    #maybe also consider
    https://www.python-graph-gallery.com/circular-barplot/
    and group with lobes?

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    figure handle

    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from textwrap import wrap
    
    #convert the values to log scale
    #figure out a way to do this conditionally
    #maybe if standard deviation of values is greater than 100 ?
    #values=np.log10(values)
    
    # Values for the x axis
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(labels), endpoint=False)
    
    # Set default font to Bell MT
    plt.rcParams.update({"font.family": "Bell MT"})

    GREY12 = "#1f1f1f"
    # Set default font color to GREY12
    plt.rcParams["text.color"] = GREY12

    # The minus glyph is not available in Bell MT
    # This disables it, and uses a hyphen
    plt.rc("axes", unicode_minus=False)

    # Colors
    if not np.any(COLORS):
        COLORS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]

    # Colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

    # Normalizer
    #norm = mpl.colors.Normalize(vmin=TRACKS_N.min(), vmax=TRACKS_N.max())

    # Normalized colors. Each number of tracks is mapped to a color in the 
    # color scale 'cmap'
    #COLORS = cmap(norm(TRACKS_N))

    # Some layout stuff ----------------------------------------------
    # Initialize layout in polar coordinates
    fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})

    # Set background color to white, both axis and figure.
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(1.2 * np.pi / 2)
    #find adaptive way to set min value
    if np.logical_or(metaValues==None, not metaValues):
        ax.set_ylim(np.min(values)*-1.3, np.max(values)*1.3)
    else:
        ax.set_ylim((np.min(values)+np.max(metaValues))*-1.3, (np.max(values)+np.max(metaValues))*1.3)
    

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # Add bars to represent the cumulative track lengths
    #make conditional on metavalues
    if metaValues==None:
        ax.bar(ANGLES, values, color=COLORS, alpha=0.9, width=(3.1415/(len(values)))*1.5 )
    else:
        ax.bar(ANGLES, values, yerr=metaValues,color=COLORS, alpha=0.9, width=(3.1415/(len(values)))*1.5 )
        
    #overly specific to aparcaseg, fix later
    #try and do split lines
    for iREGION in range(len(labels)):
        if 'ctx_lh_G_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_lh_G_','ctx_lh_G\n')
        elif 'ctx_lh_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_lh_S_','ctx_lh_S\n')
        elif 'ctx_lh_G\nand_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_lh_G\nand_S_','ctx_lh_G_and_S\n')
            
    for iREGION in range(len(labels)):
        if 'ctx_rh_G_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_rh_G_','ctx_rh_G\n')
        elif 'ctx_rh_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_rh_S_','ctx_rh_S\n')
        elif 'ctx_rh_G\nand_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_rh_G\nand_S_','ctx_rh_G_and_S\n')

    REGION = ["\n".join(wrap(r, 5, break_long_words=False)) for r in labels]
    
    #XTICKS = ax.xaxis.get_major_ticks()
    #for tick in XTICKS:
    #    tick.set_pad(10)
    
    YTICKS = ax.yaxis.get_major_ticks()
    YTICKS[-2].set_visible(False)
    YTICKS[-1].set_visible(False)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    
    ax.xaxis.grid(False)
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(REGION, size=14);
    #ax.text(0, np.max(values)-.5, "Log10  # \n of streamlines", rotation=-69, 
    #    ha="center", va="center", size=12, zorder=12)
    return plt.gcf()

def basicRadarPlot_ax(labels,values, metaValues=None,COLORS=None,ax=None):
    """
    https://www.python-graph-gallery.com/web-circular-barplot-with-matplotlib
    #maybe also consider
    https://www.python-graph-gallery.com/circular-barplot/
    and group with lobes?

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    figure handle

    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from textwrap import wrap
    from matplotlib.ticker import MaxNLocator
    
    fontSize=4
    
    #convert the values to log scale
    #figure out a way to do this conditionally
    #maybe if standard deviation of values is greater than 100 ?
    #values=np.log10(values)
    
    # Values for the x axis
    ANGLES = np.linspace(0.05, 2 * np.pi - 0.05, len(labels), endpoint=False)
    
    # Set default font to Bell MT
    plt.rcParams.update({"font.family": "Bell MT"})

    GREY12 = "#1f1f1f"
    # Set default font color to GREY12
    plt.rcParams["text.color"] = GREY12

    # The minus glyph is not available in Bell MT
    # This disables it, and uses a hyphen
    plt.rc("axes", unicode_minus=False)

    # Colors
    if not np.any(COLORS):
        COLORS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]

    # Colormap
    cmap = mpl.colors.LinearSegmentedColormap.from_list("my color", COLORS, N=256)

    # Normalizer
    #norm = mpl.colors.Normalize(vmin=TRACKS_N.min(), vmax=TRACKS_N.max())

    # Normalized colors. Each number of tracks is mapped to a color in the 
    # color scale 'cmap'
    #COLORS = cmap(norm(TRACKS_N))

    # Some layout stuff ----------------------------------------------
    # Initialize layout in polar coordinates
    if ax==None:
        fig, ax = plt.subplots(figsize=(9, 12.6), subplot_kw={"projection": "polar"})
    
        # Set background color to white, both axis and figure.
        fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_theta_offset(1.2 * np.pi / 2)
    #find adaptive way to set min value
    valueRange=np.max(values)-np.min(values)
    if np.logical_or(metaValues==None, not metaValues):
        ax.set_ylim(0-np.max(values)*.05, np.max(values)*1.3)
    else:
        ax.set_ylim(0-np.max(values)*.05, (np.max(values)+np.max(metaValues))*1.3)
    

    # Add geometries to the plot -------------------------------------
    # See the zorder to manipulate which geometries are on top

    # Add bars to represent the cumulative track lengths
    #make conditional on metavalues
    if metaValues==None:
        ax.bar(ANGLES, values, color=COLORS, alpha=0.9, width=(3.1415/(len(values)))*1.5 )
    else:
        ax.bar(ANGLES, values, yerr=metaValues,color=COLORS, alpha=0.9, width=(3.1415/(len(values)))*1.5 )
        
    #overly specific to aparcaseg, fix later
    #try and do split lines
    for iREGION in range(len(labels)):
        if 'ctx_lh_G_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_lh_G_','ctx_lh_G\n')
        elif 'ctx_lh_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_lh_S_','ctx_lh_S\n')
        elif 'ctx_lh_G\nand_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_lh_G\nand_S_','ctx_lh_G_and_S\n')
            
    for iREGION in range(len(labels)):
        if 'ctx_rh_G_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_rh_G_','ctx_rh_G\n')
        elif 'ctx_rh_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_rh_S_','ctx_rh_S\n')
        elif 'ctx_rh_G\nand_S_' in labels[iREGION]:
            labels[iREGION]=labels[iREGION].replace('ctx_rh_G\nand_S_','ctx_rh_G_and_S\n')

    REGION = ["\n".join(wrap(r, 5, break_long_words=False)) for r in labels]
    
    
    
    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(0)
    ax.yaxis.labelpad=0
    
    YTICKS = ax.yaxis.get_major_ticks()
    #we only need half th
    ax.yaxis.set_major_locator(MaxNLocator(5)) 

    
    ax.tick_params(axis='y', labelsize=fontSize )
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #ax.spines["start"].set_color("none")
    ax.spines["polar"].set_color("none")
    
    ax.xaxis.grid(False)
    ax.set_xticks(ANGLES)
    ax.set_xticklabels(REGION, size=fontSize);
    #ax.text(0, np.max(values)-.5, "Log10  # \n of streamlines", rotation=-69, 
    #    ha="center", va="center", size=12, zorder=12)
    return ax

def dipyPlotPrototypicality(streamlines,filename):
    
    import matplotlib
    import wmaPyTools.analysisTools  
    prototypeMeasure=wmaPyTools.analysisTools.streamlinePrototypicalityMeasure(streamlines,'mean')
    
    cmap = matplotlib.cm.get_cmap('jet')
    
    colors = [cmap(prototypeMeasure[iStreamline]) for iStreamline in range(len(streamlines))]
   
    
    dipyBasicPlotTract(streamlines,colors,'testPrototype2')
    
    


def dipyBasicPlotTract(streamlines,colors,tractName):
    import numpy as np
    from fury import actor, window
    import matplotlib
    import nibabel as nib
    from scipy.spatial.distance import cdist
    import dipy.tracking.utils as ut
    from scipy import ndimage
    from nilearn.image import crop_img, resample_to_img 
    
    scene = window.Scene()
    scene.clear()
    
    
    stream_actor = actor.line(streamlines, colors, linewidth=10,fake_tube=True,opacity=1)
    #stream_actor = actor.line(streamlines[1:10000], colors[1:10000], linewidth=10,fake_tube=True,opacity=1)
    #stream_actor = actor.line(streamlines[1:3000], colors[1:3000])
    #scene.clear()
    scene.add(stream_actor)

    scene.set_camera(position=(-176.42, 118.52, 128.20),
                 focal_point=(113.30, 128.31, 76.56),
                 view_up=(0.18, 0.00, 0.98))    

    scene.reset_camera()
 

    # window.show(scene, size=(600, 600), reset_camera=False)
    if not tractName==None:
        outName=tractName
    else:
        outName='tractFigure'
        
    #window.record(scene, out_path=outName+'.png', size=(6000, 6000))
     
    outArray=window.snapshot(scene, fname=None, size=(6000, 6000))
    #for whatever reason, the output of window.snapshot is upside down
    #the record function inverts this
    #https://github.com/fury-gl/fury/blob/0ff2c0ad98b92d9d2a80dc2bcc4e5c2e12f600a7/fury/window.py#L754
    #but the record function opens an unclosable window, so we can't use that
    #so here we flip it and rotate 90
    outArray=np.flipud(outArray)
    #outArray=np.rot90(outArray)
    #now crop it; 0,0,0 is the rgb color for black, so those pixels sum to 0
    colorsum=np.sum(outArray,axis=2)
    pixelsWithColor=np.asarray(np.where(colorsum>0))
    #find the borders
    minVals=np.min(pixelsWithColor,axis=1)
    maxVals=np.max(pixelsWithColor,axis=1)
    #arbitrarily establish a desired border width
    borderPixels=20
    #subselect
    croppedArray=outArray[(minVals[0]-borderPixels):(maxVals[0]+borderPixels),(minVals[1]-borderPixels):(maxVals[1]+borderPixels),:]
    
    matplotlib.pyplot.imsave(outName + '.png',croppedArray)

def dipyPlotTract(streamlines,refAnatT1=None, tractName=None,endpointColorDensityKernel=7):
    import numpy as np
    from fury import actor, window
    import matplotlib
    import nibabel as nib
    from scipy.spatial.distance import cdist
    import dipy.tracking.utils as ut
    from scipy import ndimage
    from nilearn.image import crop_img, resample_to_img 
    import matplotlib.pyplot as plt
    
    import wmaPyTools.streamlineTools
    import wmaPyTools.roiTools
    
    if not refAnatT1==None:
        if isinstance(refAnatT1,str):
            refAnatT1=nib.load(refAnatT1)

    scene = window.Scene()
    scene.clear()
    
    #fix this later
    if not refAnatT1==None:
        #borrowed from app-
        # compute mean and standard deviation of t1 for brigthness adjustment
        print("setting brightness")
        t1Data=refAnatT1.get_data()
        mean, std = t1Data[t1Data > 0].mean(), t1Data[t1Data > 0].std()

        img_min = 0.5

        img_max = 3

        # set brightness range
        value_range = (mean - img_min * std, mean + img_max * std)
        
        
        
        # vol_actor = actor.slicer(refAnatT1.get_data())
        # vol_actor.display(x=0)
        # scene.add(vol_actor)
        print('skipping')
    else:
        refAnatT1=wmaPyTools.streamlineTools.dummyNiftiForStreamlines(streamlines)
    
    
    
    
    
    #colormap for main tract
    cmap = matplotlib.cm.get_cmap('seismic')
    #colormap for neck
    #apparently this doesn't work for old versions of matplotlib
    try:
        neckCmap = matplotlib.cm.get_cmap('twilight')
    except:
        neckCmap = matplotlib.cm.get_cmap('jet')
    #jet could work too
    
    endpoints1Cmap=matplotlib.cm.get_cmap('winter')
    endpoints2Cmap=matplotlib.cm.get_cmap('autumn')
       
    colors = [cmap(np.array(range(streamline.shape[0]))/streamline.shape[0]) for streamline in streamlines]
    
    #find the neck nodes
    neckNodes=wmaPyTools.streamlineTools.findTractNeckNode(streamlines) 
    
    lin_T, offset =ut._mapping_to_voxel(refAnatT1.affine)
    
    #lets set some parameters for the exponential decay computation we'll be doing
    #for the endpoint coloration here
    #in essence we'll be altering the color of endpoints such that the "rarer"
    #(e.g. more outlier-like) they are, the more they will diverge from the standard
    #coloration scheme
    #the endpointColorDensityKernel variable is used to set the distance at which
    #the "value" of an endpoint is equal to one, closer and it is worth more, 
    #further and it is worth less.
    #in a standard exponential decay function a= the y intercept, b = the fraction being exponentiated
    #we'll just stipulate that it is 1/2 here, but this value could be modified, if desired
    expFrac=1/2
    expIntercept=(np.power(np.power(expFrac,-1),endpointColorDensityKernel))
    
    #steal come code from orientTractUsingNeck to color the neck
    #lets aim for a realspace distance when orienting our streamlines
    #we'll arbitrarily say 5 for the moment
    #we'll assume that all streamlines have the same internode distance
    #it's a TERRIBLE assumption, but really, people who violate it are the ones in error...
    avgNodeDistance=np.mean(np.sqrt(np.sum(np.square(np.diff(streamlines[0],axis=0)),axis=1)))
    lookDistanceMM=2.5
    #find the number of nodes this is equivalent to
    lookDistance=np.round(lookDistanceMM/avgNodeDistance).astype(int)
    aheadNodes=np.zeros(len(streamlines)).astype(int)
    behindNodes=np.zeros(len(streamlines)).astype(int)
    #set an empty vector for the density values, and also pre-extract the endpoints
    endpoints1Density=np.zeros(len(streamlines))
    endpoints2Density=np.zeros(len(streamlines))
    endpoints1=np.asarray([iStreams[0,:] for iStreams in streamlines])
    endpoints2=np.asarray([iStreams[-1,:] for iStreams in streamlines])
    #this gets you all endpoints in an array, will be used later
    allEndpoints=np.vstack((endpoints1,endpoints2))
    #do a conditional here
    densityMap=ut.density_map(allEndpoints, refAnatT1.affine, refAnatT1.shape)
    #densityNifti=nib.nifti1.Nifti1Image(summedDensityMap.astype(int), refAnatT1.affine, refAnatT1.header)
    #nib.save(densityNifti,'summeddensityNifti100206.nii.gz')
    #create a default sphere kernel to use in the filter.
    #no longer hard coding 0,0,0, now using the average endpoint.  Theoretically
    #has to be somewhere in the middle, right?
    kernelNifti=wmaPyTools.roiTools.createSphere(endpointColorDensityKernel, np.mean(allEndpoints,axis=0), refAnatT1)
    kernelNifti=nib.nifti1.Nifti1Image(kernelNifti.get_data().astype(int), kernelNifti.affine, kernelNifti.header)
    croppedKernel=crop_img(kernelNifti)
   
    summedDensityMap=ndimage.generic_filter(densityMap,function=np.sum,footprint=croppedKernel.get_data().astype(bool))

    for iStreamlines in range(len(streamlines)):
        #A check to make sure you've got room to do this indexing on both sides
        if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)[0]>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)[0]>lookDistance):
            aheadNodes[iStreamlines]=(neckNodes[iStreamlines]+lookDistance)
            behindNodes[iStreamlines]=(neckNodes[iStreamlines]-lookDistance)
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
            aheadNodes[iStreamlines]=neckNodes[iStreamlines]+aheadWindow
            behindNodes[iStreamlines]=neckNodes[iStreamlines]-behindWindow
        #compute the endpoint desnity for this streamline's endpoints
        #note that we are being agnostic about the endpoint identity when we compare
        #to allEndpoints as opposed to just this side's endpoints.  This is
        #reasonable because (1), ideally the other side's endpoint are far away and will 
        #essentially contribute nothing to this value (2), if they are close, something
        #has probably gone wrong with our orientation methods in the first place.
        inds = ut._to_voxel_coordinates(streamlines[iStreamlines][0,:], lin_T, offset)
        #no longer decay based, but probably faster
        endpoints1Density[iStreamlines]=summedDensityMap[inds[0],inds[1],inds[2]] 

        #no longer decay based, but probably faster
        inds = ut._to_voxel_coordinates(streamlines[iStreamlines][-1,:], lin_T, offset)
        #no longer decay based, but probably faster
        endpoints2Density[iStreamlines]=summedDensityMap[inds[0],inds[1],inds[2]] 
        
    
    #invert normalize the density vectors
    invNormEndpoints1Density=np.power(endpoints1Density,-1)*np.min(endpoints1Density)
    invNormEndpoints2Density=np.power(endpoints2Density,-1)*np.min(endpoints2Density)
    
    #short for the short middle
    #could proably clean up discreapancy between using len of arrayas and the lookdistance
    whiteArray=np.ones([np.round(lookDistance).astype(int),4])
    blendWeightsArray=np.asarray([np.arange(np.round(lookDistance).astype(int))/len(whiteArray),np.flip(np.arange(np.round(lookDistance).astype(int))/len(whiteArray))])
    blendWeightsArray=np.stack([blendWeightsArray]*4,axis=-1)
    #make a long array too for blending
    longBlendWeightsArray=np.asarray([np.arange(np.round(lookDistance).astype(int))/np.round(lookDistance).astype(int),np.flip(np.arange(np.round(lookDistance).astype(int))/np.round(lookDistance).astype(int))])
    longBlendWeightsArray=np.stack([longBlendWeightsArray]*4,axis=-1)
 
    
    for iStreamlines in range(len(streamlines)):
        #do the neck
        colors[iStreamlines][behindNodes[iStreamlines]:aheadNodes[iStreamlines]]=neckCmap(np.array(range(aheadNodes[iStreamlines]-behindNodes[iStreamlines]))/(aheadNodes[iStreamlines]-behindNodes[iStreamlines]))
        #but also blend it a bit
        #don't need to do an if check, because streamlines can always be shorter
        #if len(streamlines[iStreamlines])
        #ok, so what we are doing here: blending the existing streamline color with white using a weighed average along the streamline
        #this turned out extremely nice, so lets also do it for the endpoints as well
        #if there's room, don't bother with it otherwise
        if 0<=behindNodes[iStreamlines]-(np.round(lookDistance).astype(int)):
            behindColors=colors[iStreamlines][behindNodes[iStreamlines]-(np.round(lookDistance).astype(int)):behindNodes[iStreamlines]]
            blendColorsStack=np.asarray([behindColors,whiteArray])
            blendedReplacement=np.average(blendColorsStack,axis=0,weights=np.flip(blendWeightsArray,axis=1))
            colors[iStreamlines][behindNodes[iStreamlines]-(np.round(lookDistance).astype(int)):behindNodes[iStreamlines]]=blendedReplacement
        #now do the other side
        #if there's room, don't bother with it otherwise
        if len(colors[iStreamlines])>=aheadNodes[iStreamlines]+(np.round(lookDistance).astype(int)):
            aheadColors=colors[iStreamlines][aheadNodes[iStreamlines]:aheadNodes[iStreamlines]+(np.round(lookDistance).astype(int))]
            blendColorsStack=np.asarray([aheadColors,whiteArray])
            blendedReplacement=np.average(blendColorsStack,axis=0,weights=blendWeightsArray)
            colors[iStreamlines][aheadNodes[iStreamlines]:aheadNodes[iStreamlines]+(np.round(lookDistance).astype(int))]=blendedReplacement
        
        #also, this will recolor the streamlines if they had a neck point near the end of the streamline
        #do endpoints 1
        #no need to flip in order to ensure correct sequencing, apprently?
        end1Colors=colors[iStreamlines][0:lookDistance]
        #you know, like a telomere
        newEndpoint1Cap=np.flip(endpoints1Cmap(np.linspace(0,invNormEndpoints1Density[iStreamlines],num=lookDistance)),axis=0)
        blendColorsStack=np.asarray([end1Colors,newEndpoint1Cap])
        blendedReplacement=np.average(blendColorsStack,axis=0,weights=longBlendWeightsArray)
        colors[iStreamlines][0:lookDistance]=blendedReplacement
        #do endpoints 2
        #we do need to flip here in order to get sequecning at end of streamline correct
        #actually it was for the other one
        end2Colors=colors[iStreamlines][-(lookDistance+1):-1]
        #you know, like a telomere
        newEndpoint2Cap=endpoints2Cmap(np.linspace(0,invNormEndpoints2Density[iStreamlines],num=lookDistance))
        blendColorsStack=np.asarray([end2Colors,newEndpoint2Cap])
        blendedReplacement=np.average(blendColorsStack,axis=0,weights=np.flip(longBlendWeightsArray,axis=1))
        colors[iStreamlines][-(lookDistance+1):-1]=blendedReplacement
      
        
    stream_actor = actor.line(streamlines, colors, linewidth=10,fake_tube=True,opacity=1)
    #stream_actor = actor.line(streamlines[1:10000], colors[1:10000], linewidth=10,fake_tube=True,opacity=1)
    #stream_actor = actor.line(streamlines[1:3000], colors[1:3000])
    #scene.clear()
    scene.add(stream_actor)

    scene.set_camera(position=(-176.42, 118.52, 128.20),
                 focal_point=(113.30, 128.31, 76.56),
                 view_up=(0.18, 0.00, 0.98))    

    scene.reset_camera()
 

    # window.show(scene, size=(600, 600), reset_camera=False)
    if tractName!=None:
        outName=tractName
    else:
        outName='tractFigure'
        
    #window.record(scene, out_path=outName+'.png', size=(6000, 6000))
     
    outArray=window.snapshot(scene, fname=None, size=(6000, 6000))
    #for whatever reason, the output of window.snapshot is upside down
    #the record function inverts this
    #https://github.com/fury-gl/fury/blob/0ff2c0ad98b92d9d2a80dc2bcc4e5c2e12f600a7/fury/window.py#L754
    #but the record function opens an unclosable window, so we can't use that
    #so here we flip it and rotate 90
    outArray=np.flipud(outArray)
    #outArray=np.rot90(outArray)
    #now crop it; 0,0,0 is the rgb color for black, so those pixels sum to 0
    colorsum=np.sum(outArray,axis=2)
    pixelsWithColor=np.asarray(np.where(colorsum>0))
    #find the borders
    minVals=np.min(pixelsWithColor,axis=1)
    maxVals=np.max(pixelsWithColor,axis=1)
    #arbitrarily establish a desired border width
    borderPixels=20
    #subselect
    croppedArray=outArray[(minVals[0]-borderPixels):(maxVals[0]+borderPixels),(minVals[1]-borderPixels):(maxVals[1]+borderPixels),:]
    
    plt.imsave(outName + '.png',croppedArray)
    #now lets crop it a bit
    
def dipyPlotTract_clean(streamlines,refAnatT1=None, tractName=None, parcNifti=None):
    
        
    #has to be done before matplotlib import
    #check to see if in docker container
    import wmaPyTools.genUtils
    if wmaPyTools.genUtils.is_docker():
        #import matplotlib as mpl
        #mpl.use('Agg')
         print('Docker execution detected\nUsing xvfbwrapper for virtual display')
         #borrowing from
         #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
         from xvfbwrapper import Xvfb

         vdisplay = Xvfb()
         vdisplay.start()
    #stop code    
    # if wmaPyTools.genUtils.is_docker():
    #     #borrowing from
    #     #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
    #     vdisplay.stop()
    
    
    import numpy as np
    from fury import actor, window
    import matplotlib
    import nibabel as nib
    from scipy.spatial.distance import cdist
    import dipy.tracking.utils as ut
    from scipy import ndimage
    from nilearn.image import crop_img, resample_to_img 
    import matplotlib.pyplot as plt
    
    import wmaPyTools.streamlineTools
    import wmaPyTools.roiTools

    if not refAnatT1==None:
        if isinstance(refAnatT1,str):
            refAnatT1=nib.load(refAnatT1) 

    scene = window.Scene()
    scene.clear()
    
    #fix this later
    # if not refAnatT1==None:
    #     # #borrowed from app-
    #     # # compute mean and standard deviation of t1 for brigthness adjustment
    #     # print("setting brightness")
    #     # t1Data=refAnatT1.get_data()
    #     # mean, std = t1Data[t1Data > 0].mean(), t1Data[t1Data > 0].std()

    #     # img_min = 0.5

    #     # img_max = 3

    #     # # set brightness range
    #     # value_range = (mean - img_min * std, mean + img_max * std)
        
        
        
    #     # vol_actor = actor.slicer(refAnatT1.get_data(), refAnatT1.affine, value_range)
    #     # vol_actor.display(x=0)
    #     # scene.add(vol_actor)
    #     # print('skipping')
    # else:
    #     refAnatT1=wmaPyTools.streamlineTools.dummyNiftiForStreamlines(streamlines)
        
    if not parcNifti==None:
        streamColors, colorLut=colorStreamEndpointsFromParc(streamlines,parcNifti,streamColors=None,colorWindowMM=2.5)
    else:
        streamColors=colorGradientForStreams(streamlines)
        
    
    
    stream_actor = actor.line(streamlines, streamColors, linewidth=10,fake_tube=True,opacity=1)
    #stream_actor = actor.line(streamlines[1:10000], colors[1:10000], linewidth=10,fake_tube=True,opacity=1)
    #stream_actor = actor.line(streamlines[1:3000], colors[1:3000])
    #scene.clear()
    scene.add(stream_actor)


    #try and find the mean x coord and use this to establish the view position
    #and focal point
    # #use the neck to find thsi
    # neckNodes=wmaPyTools.streamlineTools.findTractNeckNode(streamlines)
    # #decompress it?
    # neckNodes=[iNodes[0] for iNodes in neckNodes]
    # neckCoords=[iStreamlines[neckNodes[iIndex],:] for iIndex,iStreamlines in enumerate(streamlines)]
    # #no idea whether this will return long or wide, will assume wide for now
    # meanNeckCoord=np.mean(np.asarray(neckCoords),axis=0)

    # if meanNeckCoord[0] < 0:
    #     #if the mean neck coord is on the negative side
    #     scene.set_camera(position=(-176.42, 118.52, 128.20),
    #                  focal_point=(113.30, 128.31, 76.56),
    #                  view_up=(0.18, 0.00, 0.98))    
    # else:
    scene.set_camera(position=(-176.42, 118.52, 128.20),
                     focal_point=(113.30, 128.31, 76.56),
                     view_up=(0.18, 0.00, 0.98))  

    scene.reset_camera()
 

    # window.show(scene, size=(600, 600), reset_camera=False)
    if not tractName==None:
        outName=tractName
    else:
        outName='tractFigure'
        
    #window.record(scene, out_path=outName+'.png', size=(6000, 6000))
     
    outArray=window.snapshot(scene, fname=None, size=(6000, 6000))
    #for whatever reason, the output of window.snapshot is upside down
    #the record function inverts this
    #https://github.com/fury-gl/fury/blob/0ff2c0ad98b92d9d2a80dc2bcc4e5c2e12f600a7/fury/window.py#L754
    #but the record function opens an unclosable window, so we can't use that
    #so here we flip it and rotate 90
    outArray=np.flipud(outArray)
    #outArray=np.rot90(outArray)
    #now crop it; 0,0,0 is the rgb color for black, so those pixels sum to 0
    colorsum=np.sum(outArray,axis=2)
    pixelsWithColor=np.asarray(np.where(colorsum>0))
    #find the borders
    minVals=np.min(pixelsWithColor,axis=1)
    maxVals=np.max(pixelsWithColor,axis=1)
    #arbitrarily establish a desired border width
    borderPixels=20
    #subselect
    croppedArray=outArray[(minVals[0]-borderPixels):(maxVals[0]+borderPixels),(minVals[1]-borderPixels):(maxVals[1]+borderPixels),:]
    
    plt.imsave(outName + '.png',croppedArray) 
    if wmaPyTools.genUtils.is_docker():
        #borrowing from
        #https://github.com/brainlife/app-wmc_figures/blob/76c4cf6448a72299f2d70195f9177b75e3310934/main.py#L32-L38
        vdisplay.stop()
    
    #try this?
    scene.clear()
    
    if not parcNifti==None:
        colorLut.to_csv(outName+'.csv')
    #now lets crop it a bit

def colorGradientForStreams(streamlines, streamCmap='seismic', neckCmap='twilight' ,endpointCmaps=['winter','autumn']):
    """
    Computes and implements a colormapping for an input collection of streamlines
    applies a separate colormap at the "neck" of the tract, and also
    applies separate color maps at the endpoints of the tract.
    
    NOTE:  DOES NOT REORIENT THE TRACT BEFORE DOING THIS.  PREORIENT THE
    STREAMLINES BEFORE APPLYING THIS FUNCTION

    Parameters
    ----------
    streamlines : collection of streamlines
        The streamlines for which the colormapping (for each streamline) is to
        be computed,
    streamCmap : string, specifying matplotlib colormap (or colormap itself), optional
        The colormap for the overall streamline. The default is 'seismic'.
    neckCmap : string, specifying matplotlib colormap (or colormap itself), optional, optional
        The colormap that is specific to the neck. The default is 'twilight'.
    endpointCmaps : string, specifying matplotlib colormap (or colormap itself), optional, optional
        The colormap that is specific to the endpoints. The default is ['winter','autumn'].

    Returns
    -------
    streamColors : a list of lists, with foats inside
        For each node of each streamline, this list of list contains the color
        value for that node.

    """
    
    import wmaPyTools.streamlineTools
    import matplotlib
    import numpy as np
    if isinstance(neckCmap, str):
        #sometimes twlight isn't avaialble.  lets think of a way around this.
        try:
            neckCmap = matplotlib.cm.get_cmap(neckCmap)
        except:
            neckCmap = matplotlib.cm.get_cmap(streamCmap)
            #create a linspace to index into colormap
            linspaceFill=np.linspace(0.0, 1.0, 255)
            #use it to extract a temp colormap
            tempColormap=neckCmap(linspaceFill)
            #roll it by half
            cycleRollColormap=np.roll(tempColormap,int(len(tempColormap) *.5),axis=0)
            #now do the devilishly clever thing of flipping, adding and averaging the new colormap
            doubleAvgCmap=np.divide(np.add(np.flip(cycleRollColormap,axis=0),cycleRollColormap),2)
            #use that to create a new colormap           
            neckCmap=matplotlib.colors.LinearSegmentedColormap.from_list('cycle_'+streamCmap,doubleAvgCmap,len(doubleAvgCmap))
    if isinstance(streamCmap, str):
        streamCmap = matplotlib.cm.get_cmap(streamCmap)

    if isinstance(endpointCmaps[0], str):
        endpointCmap1 = matplotlib.cm.get_cmap(endpointCmaps[0])
        endpointCmap2 = matplotlib.cm.get_cmap(endpointCmaps[1])
    else:
        endpointCmap1 = endpointCmaps[0]
        endpointCmap2 = endpointCmaps[1]
        
    #print out the names, just because fury seems to be erroring at some point
    print('Selected colormaps:')
    print('NeckCmap = ' + neckCmap.name )
    print('StreamCmap = ' + streamCmap.name )
    print('endpointCmap1 = ' + endpointCmap1.name )
    print('endpointCmap2 = ' + endpointCmap2.name )
        
    #go ahead and apply the full streamline colorscheme
    streamColors = [streamCmap(np.array(range(streamline.shape[0]))/streamline.shape[0]) for streamline in streamlines]
    
    #steal come code from orientTractUsingNeck to color the neck
    #lets aim for a realspace distance when orienting our streamlines
    #we'll arbitrarily say 5 for the moment
    #we'll assume that all streamlines have the same internode distance
    #it's a TERRIBLE assumption, but really, people who violate it are the ones in error...
    avgNodeDistance=np.mean(np.sqrt(np.sum(np.square(np.diff(streamlines[0],axis=0)),axis=1)))
    lookDistanceMM=2.5
    #find the number of nodes this is equivalent to
    lookDistance=np.round(lookDistanceMM/avgNodeDistance).astype(int)
    aheadNodes=np.zeros(len(streamlines)).astype(int)
    behindNodes=np.zeros(len(streamlines)).astype(int)

    neckNodes=wmaPyTools.streamlineTools.findTractNeckNode(streamlines)
    
    for iStreamlines in range(len(streamlines)):
        #A check to make sure you've got room to do this indexing on both sides
        if np.logical_and((len(streamlines[iStreamlines])-neckNodes[iStreamlines]-1)[0]>lookDistance,((len(streamlines[iStreamlines])-(len(streamlines[iStreamlines])-neckNodes[iStreamlines]))-1)[0]>lookDistance):
            aheadNodes[iStreamlines]=(neckNodes[iStreamlines]+lookDistance)
            behindNodes[iStreamlines]=(neckNodes[iStreamlines]-lookDistance)
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
            aheadNodes[iStreamlines]=neckNodes[iStreamlines]+aheadWindow
            behindNodes[iStreamlines]=neckNodes[iStreamlines]-behindWindow
    
    #create some arrays to use for blending
    #could proably clean up discreapancy between using len of arrayas and the lookdistance
    whiteArray=np.ones([np.round(lookDistance).astype(int),4])
    blendWeightsArray=np.asarray([np.arange(np.round(lookDistance).astype(int))/len(whiteArray),np.flip(np.arange(np.round(lookDistance).astype(int))/len(whiteArray))])
    blendWeightsArray=np.stack([blendWeightsArray]*4,axis=-1)
    #make a long array too for blending
    longBlendWeightsArray=np.asarray([np.arange(np.round(lookDistance).astype(int))/np.round(lookDistance).astype(int),np.flip(np.arange(np.round(lookDistance).astype(int))/np.round(lookDistance).astype(int))])
    longBlendWeightsArray=np.stack([longBlendWeightsArray]*4,axis=-1)
    
    
    #do the endpoints
    for iStreamlines in range(len(streamlines)):
        #do the neck
        streamColors[iStreamlines][behindNodes[iStreamlines]:aheadNodes[iStreamlines]]=neckCmap(np.array(range(aheadNodes[iStreamlines]-behindNodes[iStreamlines]))/(aheadNodes[iStreamlines]-behindNodes[iStreamlines]))
        #but also blend it a bit
        #don't need to do an if check, because streamlines can always be shorter
        #if len(streamlines[iStreamlines])
        #ok, so what we are doing here: blending the existing streamline color with white using a weighed average along the streamline
        #this turned out extremely nice, so lets also do it for the endpoints as well
        #if there's room, don't bother with it otherwise
        if 0<=behindNodes[iStreamlines]-(np.round(lookDistance).astype(int)):
            behindColors=streamColors[iStreamlines][behindNodes[iStreamlines]-(np.round(lookDistance).astype(int)):behindNodes[iStreamlines]]
            blendColorsStack=np.asarray([behindColors,whiteArray])
            blendedReplacement=np.average(blendColorsStack,axis=0,weights=np.flip(blendWeightsArray,axis=1))
            streamColors[iStreamlines][behindNodes[iStreamlines]-(np.round(lookDistance).astype(int)):behindNodes[iStreamlines]]=blendedReplacement
        #now do the other side
        #if there's room, don't bother with it otherwise
        if len(streamColors[iStreamlines])>=aheadNodes[iStreamlines]+(np.round(lookDistance).astype(int)):
            aheadColors=streamColors[iStreamlines][aheadNodes[iStreamlines]:aheadNodes[iStreamlines]+(np.round(lookDistance).astype(int))]
            blendColorsStack=np.asarray([aheadColors,whiteArray])
            blendedReplacement=np.average(blendColorsStack,axis=0,weights=blendWeightsArray)
            streamColors[iStreamlines][aheadNodes[iStreamlines]:aheadNodes[iStreamlines]+(np.round(lookDistance).astype(int))]=blendedReplacement
        
        #also, this will recolor the streamlines if they had a neck point near the end of the streamline
        #do endpoints 1
        #no need to flip in order to ensure correct sequencing, apprently?
        end1Colors=streamColors[iStreamlines][0:lookDistance]
        #you know, like a telomere
        newEndpoint1Cap=np.flip(endpointCmap1(np.linspace(0,1,num=lookDistance)),axis=0)
        blendColorsStack=np.asarray([end1Colors,newEndpoint1Cap])
        blendedReplacement=np.average(blendColorsStack,axis=0,weights=longBlendWeightsArray)
        streamColors[iStreamlines][0:lookDistance]=blendedReplacement
        #do endpoints 2
        #we do need to flip here in order to get sequecning at end of streamline correct
        #actually it was for the other one
        end2Colors=streamColors[iStreamlines][-(lookDistance+1):-1]
        #you know, like a telomere
        newEndpoint2Cap=endpointCmap2(np.linspace(0,1,num=lookDistance))
        blendColorsStack=np.asarray([end2Colors,newEndpoint2Cap])
        blendedReplacement=np.average(blendColorsStack,axis=0,weights=np.flip(longBlendWeightsArray,axis=1))
        streamColors[iStreamlines][-(lookDistance+1):-1]=blendedReplacement
    
    return streamColors
    
def colorStreamEndpointsFromParc(streamlines,parcNifti,streamColors=None,colorWindowMM=2.5):
    """
    Assigns colors to the end sections of streamlines in accordance with what
    labels from the input parcNifti they terminate in
    

    Parameters
    ----------
    streamlines : collection of streamlines
        The streamlines for which the colormapping (for each streamline) is to
        be computed,
    parcNifti : nifti
        a nifti parcellation file, with integer values indicating volumetric
        labels
    streamColors : list of 3xN arrays
        A precomputed, per node color map, such that, in each sublist, each 
        column corresponds to the RBG color assigned to that node.
        If no such colormapping is input, one will be computed using  the default
        settings from colorGradientForStreams. The default is None.
    colorWindowMM : float, optional
        The length, in mm, of each streamline's end sections that you would
        like colored in correspondance with the endpoint location.
        The default is 2.5.

    Returns
    -------
    streamColors : TYPE
        DESCRIPTION.
    colorLut : TYPE
        DESCRIPTION.

    """
    
    import numpy as np
    import matplotlib
    import dipy
    from dipy.tracking import utils
    import pandas as pd
    
    
    avgNodeDistance=np.mean(np.sqrt(np.sum(np.square(np.diff(streamlines[0],axis=0)),axis=1)))
    #colorWindowMM=2.5
    #find the number of nodes this is equivalent to
    colorNodeDistance=np.round(colorWindowMM/avgNodeDistance).astype(int)
    
    #get the number of colors required
    uniqueLabels=np.unique(np.round(parcNifti.get_data()).astype(int))
    neededColors=len(uniqueLabels)
    #generate a color selection for these
    endpointsColormap=matplotlib.cm.get_cmap('gist_rainbow')
    
    #establish a parameter to force the color selection to jump about the colormap
    jitterParam=12
    #this tells us how many times we will cycle through the colormap
    floorDivideResult=np.floor_divide(neededColors,jitterParam)+1
    #this establishes an offset to add for each added cycle through the loop
    cycleJitter=np.divide(1,floorDivideResult)
    
    #find a color index value for each unique label identity
    colorIndexes=[np.add(np.divide(np.remainder(iColors,jitterParam),jitterParam),np.multiply(np.divide(1,jitterParam),np.multiply(np.floor_divide(iColors,jitterParam)+1,cycleJitter)))  for iColors in range (neededColors) ]
    #index into the color map with these
    endpointColors = endpointsColormap(colorIndexes)
    
    #perform an initial colormapping if necessary
    if not np.any(streamColors):
        print('Running default color assignment for streamlines via colorGradientForStreams')
        streamColors=colorGradientForStreams(streamlines)
    
    #get the endpoint identities
    M, grouping=utils.connectivity_matrix(streamlines, parcNifti.affine, label_volume=np.round(parcNifti.get_data()).astype(int),
                            symmetric=False,
                            return_mapping=True,
                            mapping_as_streamlines=False)
    #get the keys so that you can iterate through them later
    keyTargets=list(grouping.keys())
    keyTargetsArray=np.asarray(keyTargets)
      
    #iterate across both sets of endpoints
    for iIndexes,iPairs in enumerate(keyTargets):
        #get the indexes of the relevant streams
        currentStreams=grouping[iPairs]
        
        for iStreams in currentStreams:
            #get the color value for this index
            colorSequenceIndex1=np.where(np.isin(uniqueLabels,keyTargetsArray[iIndexes,0]))[0][0]
            #set endpoint 1 colors
            streamColors[iStreams][0:colorNodeDistance,:]=endpointColors[colorSequenceIndex1,:]
            #get the color value for this index
            colorSequenceIndex2=np.where(np.isin(uniqueLabels,keyTargetsArray[iIndexes,1]))[0][0]
            #set endpoint 2 colors
            streamColors[iStreams][-colorNodeDistance:,:]=endpointColors[colorSequenceIndex2,:]
        
    colorLut=pd.DataFrame(columns=['labels','rgb_Color'])
    colorLut['rgb_Color']=pd.Series(list(endpointColors))
    colorLut['labels']=uniqueLabels
    
    return streamColors, colorLut

def multiPlotsForTract(streamlines,atlas=None,atlasLookupTable=None,refAnatT1=None,outdir=None,tractName=None,makeGifs=True,makeTiles=True,makeFingerprints=True,makeSpagetti=True):
    """
    Create any number of image plots for the input collection of streamlines

    Parameters
    ----------
    streamlines : nibabel.streamlines.array_sequence.ArraySequence
        A collection of streamlines presumably corresponding to a tract.
        Unknown functionality if a random collection of streamlines is used
    atlas : nifti1.Nifti1Image, optionalp
        An atlas parcellation nifti which will be used in the relevant visualizations.
        The default is None.
    atlasLookupTable : atlasLookupTable : pandas dataframe or file loadable to pandas dataframe, optional
        A dataframe of the atlas lookup table which includes the labels featured
        in the atlas and their identities.  These identities will be used
        to label the relevant plot elements. The default is None.
    refAnatT1 : TYPE, optional
        DESCRIPTION. The default is None.
    outdir : TYPE, optional
        DESCRIPTION. The default is None.
    tractName : TYPE, optional
        DESCRIPTION. The default is None.
    makeGifs : TYPE, optional
        DESCRIPTION. The default is True.
    makeTiles : TYPE, optional
        DESCRIPTION. The default is True.
    makeFingerprints : TYPE, optional
        DESCRIPTION. The default is True.
    makeSpagetti : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    import nibabel as nib
    import os
    import numpy as np
    import wmaPyTools.streamlineTools  
    from warnings import warn

    if isinstance(streamlines,str):
        streamlinesLoad=nib.streamlines.load(streamlines)
        streamlines=streamlinesLoad.streamlines
    #orient them for plotting
    streamlines=wmaPyTools.streamlineTools.orientAllStreamlines(streamlines)
        
    if isinstance(refAnatT1,str):
        refAnatT1=nib.load(refAnatT1)
    
    #probably unnecessary
    if outdir==None:
        outdir=os.getcwd()
    
    if tractName==None:
        tractName=str(len(streamlines))+'_streamsTract'
    
    if  makeSpagetti:
        #plots currently work better without reference T1
        print('creating anatomy plot')
        streamsPlotPathName=os.path.join(outdir,'streamsPlot',tractName+'streams')
        #make it if necessary
        if not os.path.exists(os.path.join(outdir,'streamsPlot')):
            os.makedirs(os.path.join(outdir,'streamsPlot'))
        dipyPlotTract_clean(streamlines,refAnatT1=None, tractName=streamsPlotPathName, parcNifti=None)
        #dipyPlotTract(streamlines,refAnatT1=refAnatT1, tractName=os.path.join(outdir,tractName))
    
    if  makeFingerprints:   
    #we use the group variant because it normalizes by proportion and splits out the endpoints into common and uncommon 
        #if an atlas + lookup table isn't provided, skip this
        if np.logical_and(np.any(atlas),np.any(atlasLookupTable)):
            print('creating endpoint fingerprint plot')
            #set out dir
            fingerprintPlotDir=os.path.join(outdir,'fingerprintPlot')
            #make it if necessary
            if not os.path.exists(fingerprintPlotDir):
                os.makedirs(fingerprintPlotDir)
            
            radialTractEndpointFingerprintPlot_Norm(streamlines,atlas,atlasLookupTable,tractName=tractName,forcePlotLabels=None,saveDir=fingerprintPlotDir,color=False)
            #radialTractEndpointFingerprintPlot(tractStreamlines,atlas,atlasLookupTable,tractName='tract',forcePlotLabels=None,saveDir=None,color=False)
            #radialTractEndpointFingerprintPlot_MultiSubj([, streamlines],[atlas, atlas],atlasLookupTable,tractName=tractName,saveDir=outdir)
        else:
            warn('Unable to produce fingerprint plots due to lack of appropriate inputs')
            
    #do the cross section gif if a reference anatomy has been provided and requested
    if makeGifs:
        if  not np.logical_or(refAnatT1=='',refAnatT1==None):
            print('creating cross section density gifs')
            #set out dir
            gifsDir=os.path.join(outdir,'gifs')
            #make it if necessary
            if not os.path.exists(gifsDir):
                os.makedirs(gifsDir)
            densityGifsOfTract(streamlines,refAnatT1,saveDir=gifsDir,tractName=tractName)
        else:
            warn('Unable to produce density gif plots due to lack of appropriate inputs')

    if makeTiles: 
        print('creating multi-tile density plots')
        if  not np.logical_or(refAnatT1=='',refAnatT1==None):
            #set out dir
            multiTileDir=os.path.join(outdir,'multiTiles')
            #make it if necessary
            if not os.path.exists(multiTileDir):
                os.makedirs(multiTileDir)
            multiTileDensity(streamlines,refAnatT1,multiTileDir,tractName,noEmpties=True)
        else:
            warn('Unable to produce multi-tile density plots due to lack of appropriate inputs')
            
    print('Plotting for ' + tractName + ' complete.')
    
def jsonFor_multiPlotsForTract(saveDir,tractName=None,makeGifs=True,makeTiles=True,makeFingerprints=True,makeSpagetti=True):
    
    import copy
    import os
    #Soichi example:
    #{ "images": [ 
    #               { "filename": "images/brainmask.svg", "name": "brainmask", "desc": "TODO" },
    #               { "filename": "images/carpetplot.svg", "name": "carpetplot", "desc": "TODO" },
    #               { "filename": "images/sampling_scheme.gif", "name": "sampling_scheme", "desc": "TODO" }
    #            ]
    #}    
    
    
    #create a list to hold the fig info
    figInfoList=[]
    blankFigInfo={'filename': '',"name": '','desc': ''}
    
    #create a list for convenience
    dimLabels=['x','y','z']
    
    if makeGifs:
        gifDirStem=os.path.join(saveDir,tractName,'gifs')
        for iDims in range(3):
            currFigInfo=copy.deepcopy(blankFigInfo)
            currFigInfo['filename']=os.path.join(gifDirStem + 'dim_'+str(iDims)+'.gif') 
            currFigInfo['name']='dim_'+str(iDims)
            currFigInfo['desc']='gif of streamline density as viewed by moving through the '+ dimLabels[iDims] +' dimension.'
            figInfoList.append(currFigInfo)
    if makeTiles:
        tileDirStem=os.path.join(saveDir,tractName,'multiTiles')
        for iDims in range(3):
            currFigInfo=copy.deepcopy(blankFigInfo)
            currFigInfo['filename']=os.path.join(tileDirStem + 'dim_'+str(iDims)+'.png') 
            currFigInfo['name']='dim_'+str(iDims)
            currFigInfo['desc']='tiled images of streamline density as viewed by moving through the '+ dimLabels[iDims] +' dimension.'
            figInfoList.append(currFigInfo)
    
    if makeFingerprints:
        fingerprintDirStem=os.path.join(saveDir,tractName,'fingerprint')
        currFigInfo=copy.deepcopy(blankFigInfo)
        currFigInfo['filename']=os.path.join(fingerprintDirStem, tractName+'_endpointFingerprint_Normed.svg') 
        currFigInfo['name']=tractName+'_endpointFingerprint_Normed'
        currFigInfo['desc']='Normed (by count) proportion of streamlines in the RAS and LPI endpoint clusters, divided into common (>1%) and uncommon (<1%) categories.'
        figInfoList.append(currFigInfo)
    
    if makeSpagetti:
        streamsPlotDirStem=os.path.join(saveDir,tractName,'streamsPlot')
        currFigInfo=copy.deepcopy(blankFigInfo)
        currFigInfo['filename']=os.path.join(streamsPlotDirStem, tractName+'.png') 
        currFigInfo['name']=tractName+'.png'
        currFigInfo['desc']='Streamline plot of ' + tractName + ', with stream body colors corresponding to proximity to endpoint cluster & distance from "neck", and endpoints colored either corresponding to atlas terminations or density'
        figInfoList.append(currFigInfo)
    
    #make a new dictionary
    outDict={}
    outDict['images']=figInfoList
    
    return outDict

        
def plotFullyConnectedRelations(squareformDistTable):
    """
    Plots a circular connectivity map 

    Parameters
    ----------
    squareformDistTable : pandas table
        A squareform pandas table (i.e. N x N) with data indicating the
    or distances between each item/entity i,j

    Raises
    ------
    ValueError
        In the event that a non-square table is detected, will throw error

    Returns
    -------
    fig : TYPE
        Figure handle for the output figure
    axes : TYPE
        axes handle for the output figure

    """
    
    
    from mne.viz import plot_connectivity_circle
    import numpy as np
    import matplotlib
    import warnings
    
    #test to ensure input table is squareform
    if not squareformDistTable.shape[0]==squareformDistTable.shape[1]:
        raise ValueError('input distance table does not appear to be squareform')
        
    if squareformDistTable.shape[0]>8:
        warnings.warning('Input data table features more than 8 items.  Output plot may be cluttered or uninterpretable')
        
    #get the column names, presumably this is the name of the pertinent entity
    #as is the case in multiTractOverlapAnalysis
    nodeNames=list(squareformDistTable.columns)
    nodeColorMap = matplotlib.cm.get_cmap('winter')
    
    #connectionColorMap= matplotlib.cm.get_cmap('autumn')
    
    fig, axes = plot_connectivity_circle(squareformDistTable.to_numpy(), nodeNames, colormap='autumn',
    node_colors=nodeColorMap(range(len(nodeNames))), facecolor='white', textcolor='black', colorbar_size=20, colorbar=True,
    linewidth=10,fontsize_names=15)
    
    return fig,axes

def iteratedTractSubComponentCrossSec(streamlines,atlas,lookupTable,refAnatT1,outDir,proportionThreshold=.01,densityThreshold=0):
    
    from dipy.tracking import utils
    import numpy as np
    import wmaPyTools.analysisTools
    import os
    
    
    [renumberedAtlasNifti,reducedLookupTable]=wmaPyTools.analysisTools.reduceAtlasAndLookupTable(atlas,lookupTable,removeAbsentLabels=True)
    #still have to guess the column for the string name
    #we could standardize the output of reduceAtlasAndLookupTable later
    entryLengths=reducedLookupTable.applymap(str).applymap(len)
    nameColumnGuess=entryLengths.mean(axis=0).idxmax()
    
   
    #get the endpoint identities
    M, grouping=utils.connectivity_matrix(streamlines, renumberedAtlasNifti.affine, label_volume=np.round(renumberedAtlasNifti.get_data()).astype(int),
                            return_mapping=True,
                            mapping_as_streamlines=False)
    #get the keys so that you can iterate through them later
    keyTargets=list(grouping.keys())
    keyTargetsArray=np.asarray(keyTargets)
    
    saveDir=os.path.join(outDir,'subCompTiles')
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    #iterate across both sets of endpoints
    for iIndexes,iPairs in enumerate(keyTargets):
        #get the indexes of the relevant streams
        print('visualising ROI ' + str(iPairs[0]) + ' to ROI ' + str(iPairs[1])  )
        
        currentStreams=grouping[iPairs]
        name1=reducedLookupTable[nameColumnGuess].iloc[iPairs[0]]
        name2=reducedLookupTable[nameColumnGuess].iloc[iPairs[1]]
        
        tckName=name1+'_TO_'+name2
        # implement the thresholding
        currentProportion=len(currentStreams)/len(streamlines)
        if currentProportion >=  proportionThreshold: 
           multiTileDensity(streamlines[currentStreams],refAnatT1,saveDir,tckName,densityThreshold,noEmpties=True)
