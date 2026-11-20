# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Tests for wmaPyTools.roiTools.  Function selection is driven by what the
downstream brainlife apps / notebooks actually call (see tests/README.md),
plus regression tests for specific bugs fixed on this branch.
"""

import warnings

import numpy as np
import nibabel as nib
import pytest

import wmaPyTools.roiTools as roiTools


def _maskData(nifti):
    return np.asarray(nifti.dataobj)


def test_makePlanarROI_isSingleVoxelThick(synthReference):
    planeNifti = roiTools.makePlanarROI(synthReference, 0, 'x')
    planeData = _maskData(planeNifti).astype(bool)
    #a plane in x should occupy exactly one x index and every y,z index
    xIndexes = np.unique(np.where(planeData)[0])
    assert len(xIndexes) == 1
    assert planeData.sum() == synthReference.shape[1] * synthReference.shape[2]
    #and that x index should map back to ~0mm
    mmX = nib.affines.apply_affine(synthReference.affine, [xIndexes[0], 0, 0])[0]
    assert abs(mmX) <= synthReference.header.get_zooms()[0]


def test_createSphere_radiusIsInMillimeters(synthReference):
    """
    Regression: the distance mask used x*x*voxDim rather than (x*voxDim)**2,
    inflating spheres on non-1mm grids (by ~41% radius at 2mm voxels).
    """
    radius = 10
    sphere = roiTools.createSphere(radius, [0, 0, 0], synthReference)
    sphereCoordsMM = nib.affines.apply_affine(
        synthReference.affine, np.asarray(np.where(_maskData(sphere))).T)
    distances = np.linalg.norm(sphereCoordsMM, axis=1)
    #every voxel center must fall within the requested mm radius (plus
    #tolerance of one voxel for the centroid snapping to the grid)
    assert distances.max() <= radius + synthReference.header.get_zooms()[0]
    #and the voxel count should be close to the analytic volume
    expectedVox = (4 / 3) * np.pi * radius ** 3 / np.prod(synthReference.header.get_zooms())
    assert 0.7 * expectedVox < _maskData(sphere).sum() < 1.3 * expectedVox


def test_createSphere_outsideImageRaises(synthReference):
    with pytest.raises(ValueError):
        roiTools.createSphere(5, [500, 0, 0], synthReference)


def test_multiROIrequestToMask_selectsLabels(synthAtlas):
    mask = roiTools.multiROIrequestToMask(synthAtlas, [10, 20])
    maskData = _maskData(mask).astype(bool)
    atlasData = _maskData(synthAtlas)
    assert np.array_equal(maskData, np.isin(atlasData, [10, 20]))


def test_multiROIrequestToMask_acceptsScalar(synthAtlas):
    mask = roiTools.multiROIrequestToMask(synthAtlas, 30)
    assert _maskData(mask).astype(bool).sum() == (_maskData(synthAtlas) == 30).sum()


def test_planeAtMaskBorder_anterior(synthAtlas):
    mask = roiTools.multiROIrequestToMask(synthAtlas, [30])
    plane = roiTools.planeAtMaskBorder(mask, 'anterior')
    planeIdx = np.where(_maskData(plane))
    assert len(np.unique(planeIdx[1])) == 1


def test_planeAtMaskBorder_invalidTermRaises(synthAtlas):
    mask = roiTools.multiROIrequestToMask(synthAtlas, [30])
    with pytest.raises(Exception):
        roiTools.planeAtMaskBorder(mask, 'sideways')


def test_planarROIFromAtlasLabelBorder(synthAtlas):
    plane = roiTools.planarROIFromAtlasLabelBorder(synthAtlas, [10], 'medial')
    planeIdx = np.where(_maskData(plane))
    assert len(np.unique(planeIdx[0])) == 1


def test_sliceROIwithPlane_keepsCorrectSide(synthAtlas, synthReference):
    roi = roiTools.multiROIrequestToMask(synthAtlas, [10, 20])
    plane = roiTools.makePlanarROI(synthReference, 0, 'x')
    leftOnly = roiTools.sliceROIwithPlane(roi, plane, 'left')
    leftData = _maskData(leftOnly).astype(bool)
    #everything retained should be label 10, all of label 10 retained
    assert np.array_equal(leftData, _maskData(synthAtlas) == 10)


def test_findROISintersection(synthAtlas):
    roiA = roiTools.multiROIrequestToMask(synthAtlas, [10, 30])
    roiB = roiTools.multiROIrequestToMask(synthAtlas, [30, 20])
    intersection = roiTools.findROISintersection([roiA, roiB])
    assert np.array_equal(_maskData(intersection).astype(bool), _maskData(synthAtlas) == 30)


def test_removeIslandsFromAtlas_viaLabelCount_defaultNotShared(synthAtlas):
    """
    Regression: ignoreLabelsRequest=[] was a mutable default that was
    appended to on every call.
    """
    import inspect
    default = inspect.signature(roiTools.removeIslandsFromAtlas_viaLabelCount).parameters['ignoreLabelsRequest'].default
    assert default is None or default == []
    import pandas as pd
    beforeOption = pd.options.mode.chained_assignment
    roiTools.removeIslandsFromAtlas_viaLabelCount(synthAtlas)
    roiTools.removeIslandsFromAtlas_viaLabelCount(synthAtlas)
    if default is not None:
        assert default == []
    #regression: also used to globally mutate pandas options
    assert pd.options.mode.chained_assignment == beforeOption


def test_removeIslandsFromAtlas_viaLabelCount_removesIsland(synthAtlas):
    atlasData = _maskData(synthAtlas).copy()
    #plant a 1-voxel island of label 20 far from the main label-20 body
    atlasData[2, 2, 2] = 20
    islandAtlas = nib.Nifti1Image(atlasData, synthAtlas.affine)
    cleaned, report = roiTools.removeIslandsFromAtlas_viaLabelCount(islandAtlas)
    assert _maskData(cleaned)[2, 2, 2] != 20
    assert len(report) > 0


def test_inflateAtlasIntoWMandBG_runs(synthAtlas):
    inflated = roiTools.inflateAtlasIntoWMandBG(synthAtlas, 1)
    #inflation should only grow labeled territory
    assert (_maskData(inflated) > 0).sum() >= (_maskData(synthAtlas) > 0).sum()


def test_boundaryROIPlanesFromMask(synthAtlas):
    mask = roiTools.multiROIrequestToMask(synthAtlas, [30])
    planes = roiTools.boundaryROIPlanesFromMask(mask)
    assert len(planes) == 6


def test_inflateAtlasIntoWMandBG_reportPath(synthAtlas, tmp_path, monkeypatch):
    """
    The de-islanding QC report defaults to ./de-island_report.csv (historical
    behavior), can be redirected, and can be disabled with None.
    """
    monkeypatch.chdir(tmp_path)
    roiTools.inflateAtlasIntoWMandBG(synthAtlas, 1)
    assert (tmp_path / 'de-island_report.csv').exists()
    (tmp_path / 'de-island_report.csv').unlink()
    roiTools.inflateAtlasIntoWMandBG(synthAtlas, 1, reportSavePath=None)
    assert list(tmp_path.iterdir()) == []
    qcDir = tmp_path / 'qc'
    qcDir.mkdir()
    roiTools.inflateAtlasIntoWMandBG(synthAtlas, 1, reportSavePath=str(qcDir / 'report.csv'))
    assert (qcDir / 'report.csv').exists()


# Claude note: -------------------------------------------------------------
# Regression tests for the voxel-offset bugs in makePlanarROI,
# planeAtMaskBorder, sliceROIwithPlane and alignROItoReference (2026-11).
# Each used to subtract 1 from dipy's (already 0-based) voxel indices, so
# planes and kept regions sat one voxel low and indices of -1 wrapped round to
# the opposite edge of the image.  The affines include 2 mm voxels and a
# flipped x axis, since the offset was in index space, not mm space.
# --------------------------------------------------------------------------

_OFFSET_SHAPE = (20, 24, 16)
_OFFSET_AFFINES = [
    np.eye(4),
    np.diag([2., 2., 2., 1.]),
    np.array([[-1., 0, 0, 10], [0, 1, 0, -12], [0, 0, 1, -8], [0, 0, 0, 1]]),
]


def _indexesAlong(data, axis):
    """Claude note: sorted voxel indexes along axis that hold any True voxel."""
    otherAxes = tuple(i for i in range(3) if i != axis)
    return np.where(np.asarray(data).astype(bool).any(axis=otherAxes))[0].tolist()


@pytest.mark.parametrize('affine', _OFFSET_AFFINES)
def test_makePlanarROI_landsOnRequestedVoxel(affine):
    """
    Claude note: a plane requested at a voxel centre's mm coordinate must
    occupy exactly that voxel slice (including the first and last slices,
    where the old -1 offset wrapped round), and cover the whole slice.
    """
    ref = nib.Nifti1Image(np.zeros(_OFFSET_SHAPE, dtype=np.uint8), affine)
    for axis, dimension in enumerate('xyz'):
        for voxelIndex in [0, 5, _OFFSET_SHAPE[axis] - 1]:
            #Claude note: mm coordinate of that slice's voxel centres
            ijk = np.zeros(3)
            ijk[axis] = voxelIndex
            mm = nib.affines.apply_affine(affine, ijk)[axis]
            planeData = _maskData(roiTools.makePlanarROI(ref, mm, dimension)).astype(bool)
            assert _indexesAlong(planeData, axis) == [voxelIndex]
            #Claude note: the whole slice is filled
            assert planeData.sum() == np.prod(_OFFSET_SHAPE) // _OFFSET_SHAPE[axis]


def test_makePlanarROI_outsideImageRaises(synthReference):
    with pytest.raises(ValueError):
        roiTools.makePlanarROI(synthReference, 10000, 'x')


@pytest.mark.parametrize('affine', _OFFSET_AFFINES)
def test_planeAtMaskBorder_liesOnMaskEdgeVoxels(affine):
    """
    Claude note: every border plane must sit on the mask's outermost voxels
    (previously the inferior / posterior / left planes were one voxel
    outside the mask).
    """
    data = np.zeros(_OFFSET_SHAPE, dtype=np.uint8)
    #Claude note: mask spans x 3-8, y 5-11, z 4-9 (voxel indexes)
    data[3:9, 5:12, 4:10] = 1
    mask = nib.Nifti1Image(data, affine)
    #Claude note: which index is on which anatomical side depends on the sign
    #of the affine's diagonal
    xLowIsLeft = affine[0, 0] > 0
    expected = {'superior': (2, 9), 'inferior': (2, 4),
                'anterior': (1, 11), 'posterior': (1, 5),
                'left': (0, 3 if xLowIsLeft else 8),
                'right': (0, 8 if xLowIsLeft else 3)}
    for border, (axis, voxelIndex) in expected.items():
        planeData = _maskData(roiTools.planeAtMaskBorder(mask, border))
        assert _indexesAlong(planeData, axis) == [voxelIndex], border


@pytest.mark.parametrize('affine', _OFFSET_AFFINES)
def test_sliceROIwithPlane_keepsPlaneAndNoWrap(affine):
    """
    Claude note: cutting a full-volume ROI keeps exactly the voxels on the
    requested side, the plane's own slice included on either side, and never
    the opposite edge slice.
    """
    full = nib.Nifti1Image(np.ones(_OFFSET_SHAPE, dtype=np.uint8), affine)
    for axis, planeIndex, highSide, lowSide in [(0, 7, 'right', 'left'),
                                               (1, 9, 'anterior', 'posterior'),
                                               (2, 5, 'superior', 'inferior')]:
        planeData = np.zeros(_OFFSET_SHAPE, dtype=np.uint8)
        sliceIndex = [slice(None)] * 3
        sliceIndex[axis] = planeIndex
        planeData[tuple(sliceIndex)] = 1
        plane = nib.Nifti1Image(planeData, affine)
        #Claude note: the anatomical side of higher voxel indexes depends on
        #the sign of the affine along this axis
        if affine[axis, axis] < 0:
            highSide, lowSide = lowSide, highSide
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            highKept = _maskData(roiTools.sliceROIwithPlane(full, plane, highSide)).astype(bool)
            lowKept = _maskData(roiTools.sliceROIwithPlane(full, plane, lowSide)).astype(bool)
        assert _indexesAlong(highKept, axis) == list(range(planeIndex, _OFFSET_SHAPE[axis]))
        assert _indexesAlong(lowKept, axis) == list(range(0, planeIndex + 1))
        #Claude note: kept slices are kept whole
        assert highKept.sum() == (_OFFSET_SHAPE[axis] - planeIndex) * np.prod(_OFFSET_SHAPE) // _OFFSET_SHAPE[axis]


def test_alignROItoReference_identityRoundTrip():
    """
    Claude note: aligning an ROI to its own grid must return it unchanged
    (it used to come back shifted one voxel along every axis, with the
    lowest slices wrapped round to the far edge).
    """
    affine = np.diag([1., 1., 1., 1.])
    data = np.zeros(_OFFSET_SHAPE, dtype=np.uint8)
    data[0:4, 6:10, 3:7] = 1
    roi = nib.Nifti1Image(data, affine)
    aligned = _maskData(roiTools.alignROItoReference(roi, roi)).astype(bool)
    assert np.array_equal(aligned, data.astype(bool))


@pytest.mark.parametrize('thickness', [6, 1])
def test_maskStats_bordersAreOuterFaces(thickness):
    """
    Claude note: analysisTools.maskStats reports a box mask's borders at its
    outer voxel faces and a boxyness of 1, including for a mask one voxel
    thick (which briefly gave boxyness=inf once border planes were moved
    onto the mask's edge voxels).
    """
    import wmaPyTools.analysisTools as analysisTools
    affine = np.array([[1., 0, 0, -30], [0, 1, 0, -12], [0, 0, 1, -8], [0, 0, 0, 1]])
    data = np.zeros((40, 24, 16), dtype=np.uint8)
    #Claude note: x voxels 3-8 (mm -27..-22), y 5-11 (mm -7..-1), z 4..4+thickness-1
    data[3:9, 5:12, 4:4 + thickness] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stats = analysisTools.maskStats(nib.Nifti1Image(data, affine))
    row = stats.iloc[0]
    assert row['lateralBorder'] == -27.5 and row['medial_border'] == -21.5
    assert row['posteriorBorder'] == -7.5 and row['anteriorBorder'] == -0.5
    assert row['inferiorBorder'] == -4.5 and row['superiorBorder'] == -4.5 + thickness
    assert np.isclose(float(row['boxyness']), 1.0)


def test_maskStats_rightHemisphereBorders():
    """
    Claude note: in the right hemisphere the medial border is the low-x face.
    """
    import wmaPyTools.analysisTools as analysisTools
    affine = np.array([[1., 0, 0, -30], [0, 1, 0, -12], [0, 0, 1, -8], [0, 0, 0, 1]])
    data = np.zeros((40, 24, 16), dtype=np.uint8)
    #Claude note: x voxels 33-35 (mm 3..5)
    data[33:36, 5:12, 4:6] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        row = analysisTools.maskStats(nib.Nifti1Image(data, affine)).iloc[0]
    assert row['medial_border'] == 2.5 and row['lateralBorder'] == 5.5
    assert np.isclose(float(row['boxyness']), 1.0)


def test_makePlanarROI_halfVoxelPastEdgeRaises():
    """
    Claude note: exactly half a voxel past the last voxel centre rounds to
    the voxel beyond the image; that must raise, not return an empty plane.
    """
    ref = nib.Nifti1Image(np.zeros(_OFFSET_SHAPE, dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError):
        roiTools.makePlanarROI(ref, _OFFSET_SHAPE[2] - 1 + .5, 'z')
    #Claude note: half a voxel before the first centre is still voxel 0
    assert _indexesAlong(_maskData(roiTools.makePlanarROI(ref, -.5, 'z')), 2) == [0]


@pytest.mark.parametrize('affine', _OFFSET_AFFINES)
def test_subjectSpaceMaskBoundaryCoords_isOutermostVoxelCentres(affine):
    """
    Claude note: both rows are the subject-space centres of the mask's
    outermost voxels (the upper row used to be one voxel past the mask,
    inherited from dipy's exclusive bounding_box).
    """
    data = np.zeros(_OFFSET_SHAPE, dtype=np.uint8)
    data[3:9, 5:12, 4:10] = 1
    data[5, 5, 4] = 0
    bounds = roiTools.subjectSpaceMaskBoundaryCoords(nib.Nifti1Image(data, affine))
    #Claude note: expected = min / max over the mask's own voxel centres
    centres = nib.affines.apply_affine(affine, np.argwhere(data))
    assert np.allclose(bounds, np.vstack([centres.min(axis=0), centres.max(axis=0)]))


def test_imageFaceBounds_andReturnNiftiBounds():
    """
    Claude note: an image's extent is half a voxel beyond its first and last
    voxel centres; returnNiftiBounds reports it max-first, as before.
    """
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = [-10, -20, -30]
    faces = roiTools._imageFaceBounds(affine, _OFFSET_SHAPE)
    expectedLow = np.array([-10, -20, -30]) - 1
    expectedHigh = expectedLow + 2 * np.asarray(_OFFSET_SHAPE)
    assert np.allclose(faces, np.vstack([expectedLow, expectedHigh]))
    niftiBounds = roiTools.returnNiftiBounds(nib.Nifti1Image(np.zeros(_OFFSET_SHAPE), affine))
    assert np.allclose(niftiBounds, np.vstack([expectedHigh, expectedLow]))


def test_visToolsSliceRangeCoversEveryVoxel():
    """
    Claude note: the slice sweeps in visTools build their positions as
    np.arange(bounds[0], bounds[1] + step/2, step) from
    subjectSpaceMaskBoundaryCoords; that must give one position per voxel,
    first to last (checked here without importing visTools' plotting stack).
    """
    affine = np.diag([2., 2., 2., 1.])
    data = np.zeros(_OFFSET_SHAPE, dtype=np.uint8)
    data[2:11, :, :] = 1
    bounds = roiTools.subjectSpaceMaskBoundaryCoords(nib.Nifti1Image(data, affine))
    step = 2.
    positions = np.arange(bounds[0, 0], bounds[1, 0] + step * .5, step)
    assert np.allclose(positions, 2. * np.arange(2, 11))
