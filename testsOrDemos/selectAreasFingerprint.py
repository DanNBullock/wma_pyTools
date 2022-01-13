#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:01:47 2022

@author: dan
"""

import nibabel as nib
from dipy.tracking.streamline import select_by_rois
import numpy as np
import time
import pandas as pd
import dipy.tracking.utils as ut
import random
import pandas as pd
import os

#you'll need to be in the right directory for this to work
import wmaPyTools.roiTools
import wmaPyTools.segmentationTools
import wmaPyTools.visTools
import wmaPyTools.streamlineTools

# esablish path to tractogram
iTCK =2
pathToTractogram = '/media/dan/storage/data/exploreTractography/test_1M_'+str(iTCK)+'.tck'
#pathToTractogram = '/media/dan/storage/data/exploreTractography/test_1M_2.tck'
testTractogram = nib.streamlines.load(pathToTractogram)
#orient the streamlines
streamlines=wmaPyTools.streamlineTools.orientAllStreamlines(testTractogram.streamlines)
# establish path to reference T1
pathToT1 ='/media/dan/storage/data/exploreTractography/T1_in_b0_unifized_GM.nii.gz'
T1 = nib.load(pathToT1)
#path to charm atlas
pathToCharmAtlas ='/media/dan/storage/data/exploreTractography/CHARM_6_nodes_in_Chandler_DWI.nii.gz'
charmAtlas = nib.load(pathToCharmAtlas)
#deisland and inflate atlas
#now we iterate
for iInflate in range(3):

    modifiedCharmAtlas=wmaPyTools.roiTools.inflateAtlasIntoWMandBG(charmAtlas,iInflate)
    #get the lookup table for charm 6, to get plot labels from
    pathToCharmLUT='/media/dan/storage/data/exploreTractography/CHARM_Nodes_6_LUT.txt'
    charmLUT=pd.read_table(pathToCharmLUT, header=None)
    
    #email text: (NOTE: The indexes provided don't match the provided LUT, modifications made accordingly)
    # We would like the connectivity strength between macaque area_24c (in CHARM6, #10) and the following regions:
    # V1 (#246)
    # clearly_v2 (#245)
    # AI (#223)
    # area_7m (#126)
    # preSMA (#88)
    # SMA (#89)
    # M1 (#79)
    # area_46d (#63)
    # area_46f (#67)
    # area 11l (#29)
    # area _11m (#28)
    # area_12o (#36)
    
    #set up the label indexes
    leftSourceLabel=[144]
    rightSourceLabel=[5]
    leftTargetLabels=[278,277,265,210,188,183,175,176,154,153,159]
    rightTargetLabels=[139,138,126,71,49,44,36,37,14,15,20]
    
    #convert these to ROIs
    leftSourceROI=wmaPyTools.roiTools.multiROIrequestToMask(modifiedCharmAtlas,leftSourceLabel,inflateIter=0)
    rightSourceROI=wmaPyTools.roiTools.multiROIrequestToMask(modifiedCharmAtlas,rightSourceLabel,inflateIter=0)
    leftTargetROI=wmaPyTools.roiTools.multiROIrequestToMask(modifiedCharmAtlas,leftTargetLabels,inflateIter=0)
    rightTargetROI=wmaPyTools.roiTools.multiROIrequestToMask(modifiedCharmAtlas,rightTargetLabels,inflateIter=0)
    bgElim=wmaPyTools.roiTools.multiROIrequestToMask(modifiedCharmAtlas,[0],inflateIter=0)
    
    #if we subset the tractogram from the outset, we can probably clean up the visualization
    #and speed up the overall processing
    leftStreamsSubset=wmaPyTools.segmentationTools.segmentTractMultiROI(streamlines, [leftSourceROI, leftTargetROI,bgElim], [True,True,False], ['either_end','either_end','either_end'])
    rightStreamsSubset=wmaPyTools.segmentationTools.segmentTractMultiROI(streamlines, [rightSourceROI, rightTargetROI,bgElim], [True,True,False], ['either_end','either_end','either_end'])
    
    #leftStreamsTractogram=wmaPyTools.streamlineTools.stubbornSaveTractogram(leftStreamsSubset,'/media/dan/storage/data/exploreTractography/24_C_investigation/leftStreamsSubset.tck')
    #rightStreamsTractogram=wmaPyTools.streamlineTools.stubbornSaveTractogram(rightStreamsSubset,'/media/dan/storage/data/exploreTractography/24_C_investigation/rightStreamsSubset.tck')
    
    leftName='leftQuery_tck'+str(iTCK)+'_inflate'+str(iInflate)
    rightName='rightQuery_tck'+str(iTCK)+'_inflate'+str(iInflate)
    
    wmaPyTools.visTools.radialTractEndpointFingerprintPlot(streamlines[leftStreamsSubset],modifiedCharmAtlas,charmLUT,tractName=leftName,saveDir='/media/dan/storage/data/exploreTractography/24_C_investigation')
    wmaPyTools.visTools.radialTractEndpointFingerprintPlot(streamlines[rightStreamsSubset],modifiedCharmAtlas,charmLUT,tractName=rightName,saveDir='/media/dan/storage/data/exploreTractography/24_C_investigation')
    
    wmaPyTools.visTools.dipyPlotTract(streamlines[leftStreamsSubset],refAnatT1=None, tractName=os.path.join('/media/dan/storage/data/exploreTractography/24_C_investigation',leftName))
    wmaPyTools.visTools.dipyPlotTract(streamlines[rightStreamsSubset],refAnatT1=None, tractName=os.path.join('/media/dan/storage/data/exploreTractography/24_C_investigation',rightName))
    
    wmaPyTools.visTools.densityGifsOfTract(streamlines[leftStreamsSubset],T1,saveDir='/media/dan/storage/data/exploreTractography/24_C_investigation',tractName=leftName)
    wmaPyTools.visTools.densityGifsOfTract(streamlines[rightStreamsSubset],T1,saveDir='/media/dan/storage/data/exploreTractography/24_C_investigation',tractName=rightName)
    
    #a dirty trick to get it to do the norm.  Because it divides by number of "subjects" double entry of single subject just returns single subject normed.
    wmaPyTools.visTools.radialTractEndpointFingerprintPlot_MultiSubj([streamlines[leftStreamsSubset], streamlines[leftStreamsSubset]],[modifiedCharmAtlas, modifiedCharmAtlas],charmLUT,tractName=leftName+'Norm',saveDir='/media/dan/storage/data/exploreTractography/24_C_investigation')
    wmaPyTools.visTools.radialTractEndpointFingerprintPlot_MultiSubj([streamlines[rightStreamsSubset], streamlines[rightStreamsSubset]],[modifiedCharmAtlas, modifiedCharmAtlas],charmLUT,tractName=rightName+'Norm',saveDir='/media/dan/storage/data/exploreTractography/24_C_investigation')

    wmaPyTools.visTools.multiTileDensity(streamlines[leftStreamsSubset],T1,'/media/dan/storage/data/exploreTractography/24_C_investigation',leftName,noEmpties=True)
    wmaPyTools.visTools.multiTileDensity(streamlines[rightStreamsSubset],T1,'/media/dan/storage/data/exploreTractography/24_C_investigation',rightName,noEmpties=True)