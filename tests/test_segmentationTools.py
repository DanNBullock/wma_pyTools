# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Tests for wmaPyTools.segmentationTools.

The core invariant tested here is that the various "fast" / reordered
segmentation implementations agree with each other (and with the plain dipy
reference) on the same request.  This mirrors the author's own
segmentationSpeedTest*.py A/B design, but as assertions.
"""

import numpy as np
import nibabel as nib
import pytest
from dipy.tracking.utils import near_roi

import wmaPyTools.roiTools as roiTools
import wmaPyTools.segmentationTools as segmentationTools


@pytest.fixture
def leftAndRightROIs(synthAtlas):
    leftROI = roiTools.multiROIrequestToMask(synthAtlas, [10])
    rightROI = roiTools.multiROIrequestToMask(synthAtlas, [20])
    return leftROI, rightROI


def test_segmentTractMultiROI_lengthMismatchRaises(synthStreamlines, leftAndRightROIs):
    """
    Regression: the guard was written `if ~ len(...)==1`, and since ~1 == -2
    the condition was never true, so mismatched inputs were never caught.
    """
    leftROI, rightROI = leftAndRightROIs
    with pytest.raises(ValueError):
        segmentationTools.segmentTractMultiROI(synthStreamlines, [leftROI, rightROI], [True], ['any', 'any'])


def test_segmentTractMultiROI_bothROIsAny(synthStreamlines, leftAndRightROIs):
    leftROI, rightROI = leftAndRightROIs
    outBool = segmentationTools.segmentTractMultiROI(
        synthStreamlines, [leftROI, rightROI], [True, True], ['any', 'any'])
    #the 5 left-right streamlines, plus the reversed one, cross both boxes;
    #the A-P streamlines and the short left-only one do not
    assert np.array_equal(np.where(outBool)[0], [0, 1, 2, 3, 4, 9])


@pytest.mark.parametrize('includeVec,operationsVec', [
    ([True, True], ['any', 'any']),
    ([True, False], ['any', 'any']),
    ([True, True], ['either_end', 'either_end']),
    ([True, True], ['both_end', 'any']),
])
def test_segmentation_implementationsAgree(synthStreamlines, leftAndRightROIs, includeVec, operationsVec):
    leftROI, rightROI = leftAndRightROIs
    rois = [leftROI, rightROI]
    standard = segmentationTools.segmentTractMultiROI(synthStreamlines, rois, includeVec, operationsVec)
    fast = segmentationTools.segmentTractMultiROI_fast(synthStreamlines, rois, includeVec, operationsVec)
    assert np.array_equal(np.asarray(standard).astype(bool), np.asarray(fast).astype(bool))


def test_segmentation_matchesDipyReference(synthStreamlines, leftAndRightROIs):
    leftROI, rightROI = leftAndRightROIs
    rois = [leftROI, rightROI]
    ours = segmentationTools.segmentTractMultiROI(synthStreamlines, rois, [True, True], ['any', 'any'])
    #dipy reference: plain near_roi against each full ROI, AND-ed.  Our
    #implementation pre-subsets the ROI by a dilated streamline density mask
    #for speed; this checks that shortcut doesn't change the answer
    dipyBool = np.ones(len(synthStreamlines), dtype=bool)
    for iROI in rois:
        dipyBool &= near_roi(synthStreamlines, iROI.affine,
                             np.asarray(iROI.dataobj).astype(bool), mode='any')
    assert np.array_equal(np.asarray(ours).astype(bool), dipyBool)


def test_applyEndpointCriteria_left(synthStreamlines, synthReference):
    plane = roiTools.makePlanarROI(synthReference, 0, 'x')
    #'both' endpoints left of the midline: only the short left streamline
    bothLeft = segmentationTools.applyEndpointCriteria(synthStreamlines, plane, 'left', 'both')
    assert np.array_equal(np.where(bothLeft)[0], [8])
    #'one' endpoint left of the midline: the left-right crossing ones
    oneLeft = segmentationTools.applyEndpointCriteria(synthStreamlines, plane, 'left', 'one')
    assert set(np.where(oneLeft)[0]) == {0, 1, 2, 3, 4, 9}


def test_applyMidpointCriteria_anterior(synthStreamlines, synthReference):
    plane = roiTools.makePlanarROI(synthReference, 0, 'y')
    anteriorMid = segmentationTools.applyMidpointCriteria(synthStreamlines, plane, 'anterior')
    #A-P streamlines run y=-20..25, so their midpoint (~2.5) is anterior of 0;
    #the left-right streamlines at y=2,4 also have anterior midpoints
    assert set(np.where(anteriorMid)[0]) >= {5, 6, 7}
    assert 0 not in set(np.where(anteriorMid)[0])


def test_tractProbabilityMap2SegCriteria_elongatedMask():
    """Claude note: an elongated mask gets inclusion planes every 7 mm along
    its longest dimension (the end planes dropped) and its six border planes as
    exclusions."""
    affine = np.diag([1., 1., 1., 1.])
    affine[:3, 3] = [-20., -40., -20.]
    data = np.zeros((40, 80, 40), dtype=np.uint8)
    #a bar 6 voxels wide in x and z, and 50 voxels long in y
    data[17:23, 15:65, 17:23] = 1
    maskNifti = nib.Nifti1Image(data, affine)

    criteria = segmentationTools.tractProbabilityMap2SegCriteria(maskNifti)

    #the bar's voxel centres run from y=-25 to y=24, so the planes sit at
    #-25, -18, ..., 17 and the inner ones are -18 to 10
    expectedPlanes = np.arange(-25., 24., 7)[1:-1]
    includes = criteria['any']['include']
    assert len(includes) == len(expectedPlanes)
    for iPlane, iExpected in zip(includes, expectedPlanes):
        voxelIndexes = np.array(np.where(np.asanyarray(iPlane.dataobj))).T
        subjectY = nib.affines.apply_affine(iPlane.affine, voxelIndexes)[:, 1]
        np.testing.assert_allclose(np.unique(subjectY), [iExpected])
    assert sorted(criteria['any']['exclude'].keys()) == sorted(['medial', 'lateral', 'anterior', 'posterior', 'superior', 'inferior'])
