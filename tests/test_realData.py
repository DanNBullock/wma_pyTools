# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Real-data tests against HCP sub-100206 (see tests/README.md for layout).
Skipped unless WMAPYTOOLS_TESTDATA is set.

NOTE: the parcellation (FreeSurfer conformed, 1mm, LIA orientation) and the
T1 (1.25mm, RAS) deliberately live on different voxel grids.  Everything is
expected to agree in mm (RAS) space, which is exactly what these tests check.
"""

import numpy as np
import nibabel as nib
import pytest
from dipy.tracking.utils import near_roi

import wmaPyTools.roiTools as roiTools
import wmaPyTools.segmentationTools as segmentationTools
import wmaPyTools.streamlineTools as streamlineTools
import wmaPyTools.analysisTools as analysisTools

pytestmark = pytest.mark.realdata

#FreeSurfer aseg labels
LEFT_THALAMUS = 10
RIGHT_THALAMUS = 49
LEFT_CEREBRAL_WM = 2
RIGHT_CEREBRAL_WM = 41


@pytest.fixture(scope='module')
def loaded(realData):
    return {
        't1': nib.load(realData['t1']),
        'parc': nib.load(realData['parc']),
        'streamlines': nib.streamlines.load(realData['tck']).streamlines,
    }


def test_wmcFromMat_matchesFullTractogram(realData):
    wmc = streamlineTools.matWMC2dict(realData['classification'])
    assert len(wmc['names']) == 61
    assert len(wmc['index']) == 3000000
    assert np.max(wmc['index']) == len(wmc['names'])


def test_parcAndTractogramShareMMSpace(loaded):
    """Sanity check: most streamline nodes should fall inside labeled brain."""
    parc = loaded['parc']
    nodes = np.concatenate([np.asarray(s) for s in loaded['streamlines'][:2000]])
    vox = np.round(nib.affines.apply_affine(np.linalg.inv(parc.affine), nodes)).astype(int)
    labels = np.asanyarray(parc.dataobj)[vox[:, 0], vox[:, 1], vox[:, 2]]
    assert np.mean(np.isin(labels, [LEFT_CEREBRAL_WM, RIGHT_CEREBRAL_WM])) > 0.5


def test_createSphere_onHCPgrid(loaded):
    t1 = loaded['t1']
    radius = 8
    sphere = roiTools.createSphere(radius, [0, -18, 18], t1)
    expectedVox = (4 / 3) * np.pi * radius ** 3 / np.prod(t1.header.get_zooms())
    assert 0.85 * expectedVox < np.asanyarray(sphere.dataobj).sum() < 1.15 * expectedVox


def test_segmentation_thalamicInterhemispheric(loaded):
    parc = loaded['parc']
    streamlines = loaded['streamlines'][:5000]
    leftThal = roiTools.multiROIrequestToMask(parc, [LEFT_THALAMUS])
    rightThal = roiTools.multiROIrequestToMask(parc, [RIGHT_THALAMUS])
    rois = [leftThal, rightThal]
    standard = np.asarray(segmentationTools.segmentTractMultiROI(
        streamlines, rois, [True, True], ['any', 'any'])).astype(bool)
    fast = np.asarray(segmentationTools.segmentTractMultiROI_fast(
        streamlines, rois, [True, True], ['any', 'any'])).astype(bool)
    reference = np.ones(len(streamlines), dtype=bool)
    for iROI in rois:
        reference &= near_roi(streamlines, iROI.affine,
                              np.asanyarray(iROI.dataobj).astype(bool), mode='any')
    assert np.array_equal(standard, reference)
    assert np.array_equal(fast, reference)


def test_planarEndpointSegmentation_onLIAparc(loaded):
    """
    Planes derived from an LIA-oriented parc must still mean "left" in RAS.
    Streamlines with both endpoints left of the left-thalamus medial border
    should have both endpoint x coordinates below that plane's mm x.
    (NOTE: in this subject the left thalamus label reaches x=0, so the plane
    sits at x=+1mm; "left of it" is therefore not the same as x<0.)
    """
    parc = loaded['parc']
    streamlines = loaded['streamlines'][:20000]
    plane = roiTools.planarROIFromAtlasLabelBorder(parc, [LEFT_THALAMUS], 'medial')
    bothLeft = segmentationTools.applyEndpointCriteria(streamlines, plane, 'left', 'both')
    assert bothLeft.sum() > 0
    planeX = nib.affines.apply_affine(plane.affine, np.argwhere(np.asanyarray(plane.dataobj)))[:, 0]
    assert len(np.unique(planeX)) == 1
    endpointX = np.asarray([[s[0][0], s[-1][0]] for s in streamlines[np.where(bothLeft)[0]]])
    assert np.all(endpointX < planeX[0])
    #and the mirror request should select a disjoint set
    bothRight = segmentationTools.applyEndpointCriteria(streamlines, plane, 'right', 'both')
    assert not np.any(np.logical_and(bothLeft, bothRight))


def test_streamGeomQuantifications_onRealStreamlines(loaded):
    table = analysisTools.streamGeomQuantifications(loaded['streamlines'][:2000])
    assert len(table) == 2000
    valid = table.dropna()
    assert len(valid) > 1900
    assert valid['efficiencyRat'].between(0, 1 + 1e-6).all()
    assert valid['asymRat'].between(0, 1 + 1e-6).all()
