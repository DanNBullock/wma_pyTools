#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:09:31 2021

@author: dan
"""

import nibabel as nib
from dipy.tracking.streamline import select_by_rois
import numpy as np
import time
import pandas as pd
import dipy.tracking.utils as ut
import random

# esablish path to tractogram
pathToTestTractogram = '/media/dan/storage/data/exploreTractography/test_1M_1.tck'
testTractogram = nib.streamlines.load(pathToTestTractogram)
# establish path to reference T1
pathToTestT1 ='/media/dan/storage/data/exploreTractography/T1_in_b0_unifized_GM.nii.gz'
testT1 = nib.load(pathToTestT1)



# get a mask of the whole brain tract
tractMask=ut.density_map(testTractogram.streamlines, testT1.affine, testT1.shape)
#extract these as coordinates
imgSpaceTractVoxels = np.array(np.where(tractMask)).T
subjectSpaceTractCoords = nib.affines.apply_affine(testT1.affine, imgSpaceTractVoxels)

# create a dataframe for the output
outputDataframe=pd.DataFrame(columns=['dipyTime','modifiedTime','dipyCount','modifiedCount','testRadius'])

#arbitrarily set the number of rois per test.
#Increasing this beyond 2 isn't that helpful because its highly unlikely that 
#any given streamline passes though 3 randomly placed spheres
roiNumToTest=1

for iTests in list(range(1,20)):
    #initalize list structures at the start of each test
    roisData = []
    roisNifti =[]
    include = []
    operations=[]
    
    # arbitrarily set radius for test spheres
    testRadius = np.random.randint(4,10)
    # generate some number of spheres to use in a test segmentation
    for iRois in list(range(roiNumToTest)):
        # randomly select a centroid coordinate within the tract mask
        # NOTE: This could be on the edge or deep in the tractogram
        testCentroid = subjectSpaceTractCoords[np.random.randint(0,len(subjectSpaceTractCoords))]
        # obtain that data array as bool
        sphereNifti=createSphere(testRadius, testCentroid, testT1)
        # add that and a True to the list vector for each
        roisData.append(sphereNifti.get_fdata().astype(bool))
        roisNifti.append(sphereNifti)
        # randomly select include or exclude
        include.append(bool(random.getrandbits(1)))
        operations.append('any')
        
    # start timing
    t1_start=time.process_time()
    # specify segmentation
    dipySegmented=ut.near_roi(testTractogram.streamlines, testT1.affine, roisData[0], mode='any')
    # actually perform segmentation and get count (cant do indexes here for whatever reason)
    if [include[0]]:
        dipyCount=np.sum(dipySegmented)
    else:
        dipyCount=np.sum(np.logical_not(dipySegmented))
    # stop time
    t1_stop=time.process_time()
    # get the elapsed time
    dipyTime=t1_stop-t1_start
    
    #restart time
    t1_start=time.process_time()
    #perform segmentation again, but with the modified version
    modifiedSegmented=segmentTractMultiROI(testTractogram.streamlines, roisNifti, include, operations)
    # stop time
    t1_stop=time.process_time()
    # get the elapsed time
    modifiedTime=t1_stop-t1_start
    #get the count
    modifiedCount=np.sum(modifiedSegmented)
    
    #set the dataframe entries
    outputDataframe.at[iTests,'dipyTime']=dipyTime
    outputDataframe.at[iTests,'modifiedTime']=modifiedTime
    outputDataframe.at[iTests,'dipyCount']=dipyCount
    outputDataframe.at[iTests,'modifiedCount']=modifiedCount
    outputDataframe.at[iTests,'testRadius']=testRadius