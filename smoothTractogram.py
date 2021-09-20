# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:00:27 2021

@author: Daniel
"""
#streamline smoothing
import json
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import copy
import os
import scipy

refPath='D:\\Documents\\gitDir\\monkeyTracts\\D99_in_Chandler_Diffusion_Space.nii.gz'
tractPath='D:\\Documents\\gitDir\\monkeyTracts\\cleaned_1M_Tractogram\\culled.tck'
reference = nib.load(refPath)
tractogram = load_tractogram(tractPath,reference)

streamlines=tractogram.streamlines
def smoothStreamlines(tractogram):
    import dipy
    import nibabel as nib
    inputStreamlines=tractogram.streamlines
    initialLength=len(inputStreamlines)
    for iStreams in range(initialLength):
        dipySplineOut=dipy.tracking.metrics.spline(inputStreamlines[iStreams])
        #this is an ugly way to do this
        inputStreamlines.append(dipySplineOut)
    outStreamlines=streamlines[initialLength-1:-1]
    out_tractogram = nib.streamlines.tractogram.Tractogram(outStreamlines)
    return out_tractogram
    

resultObj=smoothStreamlines(tractogram)
save_tractogram()
nib.sa