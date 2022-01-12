#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:06:18 2022

@author: dan
"""

#some how set a path to wma pyTools
wmaToolsDir='/media/dan/storage/gitDir/wma_pyTools'
import os
os.chdir(wmaToolsDir)
import wmaPyTools
import nibabel as nib
import pandas as pd
import wmaPyTools.roiTools
import wmaPyTools.streamlineTools 
import numpy as np

seed_density=200
#load atlas
atlas=nib.load('/media/dan/storage/data/proj-5ffc884d2ba0fba7a7e89132/sub-100307/dt-neuro-freesurfer.tag-acpc_aligned.id-6074dd05daf98925f029a156/output/mri/aparc.a2009s+aseg.nii.gz')
wmLabels=[2,41]
#parietal 3 number loactions
targetMaskLabels=[11157, 11127, 11168, 11136, 11126, 11125, 12157, 12127, 12168, 12136, 12126, 12125]

targetMask=wmaPyTools.roiTools.multiROIrequestToMask(atlas,targetMaskLabels,inflateIter=0)
wmMask=wmaPyTools.roiTools.multiROIrequestToMask(atlas,wmLabels,inflateIter=0)

dwi=nib.load('/media/dan/storage/data/proj-5941a225f876b000210c11e5/sub-100307/dt-neuro-dwi.tag-preprocessed.id-598a28ec19ee5b6f80cba0b8/dwi.nii.gz')
bvecsfile=pd.read_table('/media/dan/storage/data/proj-5941a225f876b000210c11e5/sub-100307/dt-neuro-dwi.tag-preprocessed.id-598a28ec19ee5b6f80cba0b8/dwi.bvecs', header=None, delim_whitespace=True)
bvalsfile=pd.read_table('/media/dan/storage/data/proj-5941a225f876b000210c11e5/sub-100307/dt-neuro-dwi.tag-preprocessed.id-598a28ec19ee5b6f80cba0b8/dwi.bvals', header=None, delim_whitespace=True)


bvecs=np.asarray(bvecsfile)
bvals=np.squeeze(np.asarray(bvalsfile))


testOutput=wmaPyTools.streamlineTools.trackStreamsInMask(targetMask,seed_density,wmMask,dwi,bvecs,bvals)