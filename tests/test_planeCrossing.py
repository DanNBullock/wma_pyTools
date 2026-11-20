# -*- coding: utf-8 -*-
"""
Claude note: tests for the exact, tolerance-free plane crossing test
(segmentationTools.planarROICrossings and the 'cross' operation), added
2026-11.

The defining properties tested here:
  * crossing vs. kissing is decided by SIDE (consecutive nodes on opposite
    sides of the plane), not by distance or by how many nodes are near it;
  * the result does not depend on how densely the nodes sample the path
    (adding nodes along a segment, or removing nodes that lie exactly on a
    segment, changes nothing);
  * planes may be bounded (only crossings through the mask's in-plane cells
    count) and the image may be oblique.
"""

import io
import contextlib
import warnings

import numpy as np
import nibabel as nib
import pytest
from dipy.tracking.streamline import Streamlines

import wmaPyTools.roiTools as roiTools
import wmaPyTools.segmentationTools as segmentationTools


def _quiet(func, *args, **kwargs):
    """Claude note: run a library call without its narration and warnings."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args, **kwargs)


@pytest.fixture
def planeY0():
    """Claude note: the full coronal plane y = 0 mm on a 2 mm grid."""
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = [-20, -20, -20]
    reference = nib.Nifti1Image(np.zeros((21, 21, 21), dtype=np.uint8), affine)
    return reference, _quiet(roiTools.makePlanarROI, reference, 0, 'y')


def test_crossVersusKiss(planeY0):
    """
    Claude note: each case's expected crossing count, including the ones a
    distance-based test gets wrong (a crossing with no node near the plane; a
    kiss; a touch; a streamline ending near the plane).
    """
    reference, plane = planeY0
    cases = [
        ([[0, -5, 0], [0, 5, 0]], 1),                        # crosses, 10 mm segment, no node near
        ([[0, -5, 0], [0, -0.1, 0], [0, -5, 1]], 0),         # kiss: approaches to 0.1 mm, turns back
        ([[0, -5, 0], [0, 0, 0], [0, -5, 1]], 0),            # touches the plane exactly, returns
        ([[0, -5, 0], [0, 0, 0], [0, 0, 0.5], [0, 5, 1]], 1),  # passes through, running in the plane a while
        ([[0, -5, 0], [0, 5, 0], [0, -5, 0]], 2),            # crosses and comes back
        ([[0, -5, 0], [0, -0.3, 0]], 0),                     # ends near the plane
        ([[0, 0, 0], [0, 0, 5]], 0),                         # lies entirely in the plane
        ([[30, -5, 0], [30, 5, 0]], 0),                      # crosses the plane outside the image
    ]
    streamlines = Streamlines([np.asarray(points, dtype=float) for points, _ in cases])
    counts = _quiet(segmentationTools.planarROICrossings, streamlines, plane)
    assert list(counts) == [expected for _, expected in cases]


def _upsampled(streamline, factor, rng):
    """Claude note: add factor-1 random points on every segment (same polyline)."""
    pieces = []
    for iSegment in range(len(streamline) - 1):
        fractions = np.sort(rng.random(factor - 1))
        pieces.append(streamline[iSegment])
        pieces.extend(streamline[iSegment] + fractions[:, None] * (streamline[iSegment + 1] - streamline[iSegment]))
    pieces.append(streamline[-1])
    return np.asarray(pieces)


def test_invariantToNodeSampling(planeY0):
    """
    Claude note: the same polylines sampled differently give identical
    counts: random extra points on each segment, and removing every other
    point of a straight streamline (those points lie exactly on the line).
    """
    reference, plane = planeY0
    rng = np.random.default_rng(0)
    #Claude note: random wiggly streamlines around the plane
    original = [np.cumsum(rng.normal(0, 2.5, size=(rng.integers(3, 30), 3)), axis=0) + rng.uniform(-8, 8, size=3) for _ in range(300)]
    upsampled = [_upsampled(s, 7, rng) for s in original]
    countsOriginal = _quiet(segmentationTools.planarROICrossings, Streamlines(original), plane)
    countsUpsampled = _quiet(segmentationTools.planarROICrossings, Streamlines(upsampled), plane)
    assert np.array_equal(countsOriginal, countsUpsampled)
    assert countsOriginal.sum() > 50
    #Claude note: straight streamlines; drop collinear interior points
    starts = rng.uniform(-15, 15, size=(200, 3))
    ends = rng.uniform(-15, 15, size=(200, 3))
    dense = [np.linspace(a, b, 41) for a, b in zip(starts, ends)]
    sparse = [d[::8] for d in dense]
    countsDense = _quiet(segmentationTools.planarROICrossings, Streamlines(dense), plane)
    countsSparse = _quiet(segmentationTools.planarROICrossings, Streamlines(sparse), plane)
    assert np.array_equal(countsDense, countsSparse)
    #Claude note: and it agrees with the analytic answer for a full plane
    assert np.array_equal(countsDense > 0, np.sign(starts[:, 1]) != np.sign(ends[:, 1]))


def test_boundedPlane(planeY0):
    """
    Claude note: a plane cut to its left half only counts crossings through
    that half.
    """
    reference, plane = planeY0
    leftHalf = _quiet(roiTools.sliceROIwithPlane, plane, _quiet(roiTools.makePlanarROI, reference, 0, 'x'), 'left')
    streamlines = Streamlines([np.array([[-6., -5, 0], [-6, 5, 0]]), np.array([[6., -5, 0], [6, 5, 0]])])
    assert list(_quiet(segmentationTools.planarROICrossings, streamlines, leftHalf)) == [1, 0]


def test_obliqueImage():
    """
    Claude note: on an image rotated 30 degrees about z, the counts equal the
    sign changes of the mm-space distance to the plane through the slab's
    voxel centres.
    """
    angle = np.deg2rad(30)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = 2 * rotation
    affine[:3, 3] = -nib.affines.apply_affine(affine, [15, 15, 15])
    data = np.zeros((31, 31, 31), dtype=np.uint8)
    data[:, 15, :] = 1
    plane = nib.Nifti1Image(data, affine)
    #Claude note: the plane in mm: through the image centre, normal = image's 2nd axis
    normal = rotation[:, 1]
    rng = np.random.default_rng(1)
    streamlines = [np.cumsum(rng.normal(0, 3, size=(rng.integers(3, 25), 3)), axis=0) + rng.uniform(-10, 10, size=3) for _ in range(300)]
    #Claude note: expected = sign changes of the mm distance to the plane, counting
    #only crossings whose crossing point lies inside the image's in-plane extent
    #(the plane ROI ends where the image ends)
    inverseAffine = np.linalg.inv(affine)
    expected = []
    for s in streamlines:
        distance = s @ normal
        count = 0
        for iSegment in np.flatnonzero(np.diff(np.sign(distance)) != 0):
            fraction = distance[iSegment] / (distance[iSegment] - distance[iSegment + 1])
            crossingPoint = s[iSegment] + fraction * (s[iSegment + 1] - s[iSegment])
            voxelPoint = nib.affines.apply_affine(inverseAffine, crossingPoint)
            count += int(np.all(np.floor(voxelPoint + 0.5) >= 0) and np.all(np.floor(voxelPoint + 0.5) <= 30))
        expected.append(count)
    expected = np.array(expected)
    counts = _quiet(segmentationTools.planarROICrossings, Streamlines(streamlines), plane)
    assert np.array_equal(counts, expected)
    assert expected.sum() > 50


def test_nonPlanarMasksRaise(planeY0):
    reference, plane = planeY0
    slab = np.zeros((21, 21, 21), dtype=np.uint8)
    slab[:, 9:11, :] = 1
    line = np.zeros((21, 21, 21), dtype=np.uint8)
    line[:, 10, 10] = 1
    streamlines = Streamlines([np.array([[0., -5, 0], [0, 5, 0]])])
    for badData in [slab, line]:
        with pytest.raises(ValueError):
            _quiet(segmentationTools.planarROICrossings, streamlines, nib.Nifti1Image(badData, plane.affine))


def test_crossOperationInSegmentation(planeY0):
    """
    Claude note: 'cross' works through every segmentation entry point and
    combines with the other operations.
    """
    reference, plane = planeY0
    streamlines = Streamlines([np.array([[0., -5, 0], [0, 5, 0]]),
                               np.array([[0., -5, 0], [0, -0.1, 0], [0, -5, 1]]),
                               np.array([[0., -5, 0], [0, 5, 0], [0, -5, 0]])])
    expected = [True, False, True]
    assert list(_quiet(segmentationTools.segmentTractMultiROI, streamlines, [plane], [True], ['cross'])) == expected
    assert list(_quiet(segmentationTools.segmentTractMultiROI_fast, streamlines, [plane], [True], ['cross'])) == expected
    assert list(_quiet(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test, streamlines, plane, True, 'cross')) == expected
    assert list(_quiet(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test, streamlines, plane, False, 'cross')) == [not e for e in expected]
    assert list(_quiet(segmentationTools.near_roi_fast, streamlines, plane.affine, np.asanyarray(plane.dataobj), mode='cross')) == expected
    #Claude note: combined: crosses the plane AND does not come within tolerance
    #of it otherwise (the third streamline's nodes are all 5 mm away)
    combined = _quiet(segmentationTools.segmentTractMultiROI, streamlines, [plane, plane], [True, False], ['cross', 'any'])
    assert list(combined) == [True, False, True]


def test_speedSortKeepsEveryCriterion(planeY0):
    """
    Claude note: speedSortSegCriteria returns every criterion exactly once,
    in the order endpoint / cross / other node criteria, smallest ROI first
    within each (it previously misplaced operations outside its two groups).
    """
    reference, plane = planeY0
    small = nib.Nifti1Image(np.zeros((21, 21, 21), dtype=np.uint8), plane.affine)
    small.dataobj[10, 10, 10] = 1
    big = nib.Nifti1Image(np.zeros((21, 21, 21), dtype=np.uint8), plane.affine)
    big.dataobj[5:15, 5:15, 5:15] = 1
    rois = [big, plane, small, big, small]
    includes = [True, True, False, True, True]
    operations = ['any', 'cross', 'any', 'either_end', 'both_end']
    sortedROIs, sortedIncludes, sortedOperations = _quiet(segmentationTools.speedSortSegCriteria, rois, includes, operations)
    assert sortedOperations == ['both_end', 'either_end', 'cross', 'any', 'any']
    assert [id(r) for r in sortedROIs] == [id(small), id(big), id(plane), id(small), id(big)]
    assert sortedIncludes == [True, True, True, False, True]


# Claude note: -------------------------------------------------------------
# Tests on generated edge cases (wmaPyTools.syntheticBundles), whose true
# crossing counts come from their construction.
# --------------------------------------------------------------------------

import wmaPyTools.syntheticBundles as syntheticBundles

_PLACEMENTS = [
    # rotation, spacing, spacingJitter, voxelSize
    (None, 1., 0., 1.),
    ((20, 35, -50), 1., 0., 1.),
    ((0, 0, 30), 2., 0.5, 2.),
    ((70, 10, 5), 0.5, 0.8, 1.25),
    ((45, 45, 45), 3., 0.3, 1.),
]


@pytest.mark.parametrize('rotation,spacing,spacingJitter,voxelSize', _PLACEMENTS)
def test_generatedCases_countsMatchTruth(rotation, spacing, spacingJitter, voxelSize):
    """
    Claude note: every generated case class (steep to very shallow crossings,
    1-4 crossings, kisses and parallel runs down to 0.001 mm from the plane,
    streamlines ending near it), on oblique grids and with coarse and uneven
    node spacing: the counts equal the truth.
    """
    streamlines, truth, plane = syntheticBundles.generatePlaneCrossingCases(
        nPerClass=40, rotation=rotation, translation=(5., -12., 30.), spacing=spacing,
        spacingJitter=spacingJitter, voxelSize=voxelSize, seed=3)
    counts = _quiet(segmentationTools.planarROICrossings, streamlines, plane)
    assert np.array_equal(counts, truth['crossings'])
    #Claude note: the cases are resolved by their sampling (so the construction's
    #count is the polyline's count)
    assert len(truth['unresolved']) == 0


def test_crossingFamilyPredicates():
    """
    Claude note: each operation in the family selects exactly the streamlines
    its count predicate describes, and each not_ operation equals its
    partner with includeBool=False.
    """
    streamlines, truth, plane = syntheticBundles.generatePlaneCrossingCases(nPerClass=15, rotation=(10, 20, 30), seed=4)
    counts = truth['crossings']
    expected = {'cross': counts >= 1, 'not_cross': counts == 0,
                'single_cross': counts == 1, 'not_single_cross': counts != 1,
                'double_cross': counts == 2, 'not_double_cross': counts != 2}
    assert set(expected) == set(segmentationTools.CROSSING_OPERATIONS)
    for operation, wanted in expected.items():
        got = _quiet(segmentationTools.segmentTractMultiROI, streamlines, [plane], [True], [operation])
        assert np.array_equal(np.asarray(got), wanted), operation
    for positive in ['cross', 'single_cross', 'double_cross']:
        negated = _quiet(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test, streamlines, plane, False, positive)
        named = _quiet(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test, streamlines, plane, True, 'not_' + positive)
        assert np.array_equal(np.asarray(negated), np.asarray(named))
    #Claude note: a loop through the plane (3 or 4 crossings) is a cross but
    #neither a single nor a double cross
    loops = counts >= 3
    assert loops.any()
    assert not np.any(expected['single_cross'][loops] | expected['double_cross'][loops])


def test_generatedBundles_invariantToSampling():
    """
    Claude note: the same bundle sampled at 0.5 mm and at 2 mm with uneven
    steps (the same curves, different nodes) crosses a plane through its neck
    identically, up to streamlines whose crossing pair falls within one
    coarse segment (none here: the bundle crosses its neck plane once).
    """
    for shape in ['straight', 'arc', 'hook', 'commissural', 'fan']:
        fine, truth = syntheticBundles.generateBundle(shape, nStreamlines=80, spacing=0.5, seed=7)
        coarse, _ = syntheticBundles.generateBundle(shape, nStreamlines=80, spacing=2., spacingJitter=0.6, seed=7)
        reference = syntheticBundles.referenceForStreamlines(fine, voxelSize=1.)
        #Claude note: a plane through the neck, perpendicular to the voxel axis
        #closest to the neck tangent
        axis = int(np.argmax(np.abs(truth['neckTangent'])))
        neckVoxel = np.round(nib.affines.apply_affine(np.linalg.inv(reference.affine), truth['neck'])).astype(int)
        data = np.zeros(reference.shape, dtype=np.uint8)
        index = [slice(None)] * 3
        index[axis] = neckVoxel[axis]
        data[tuple(index)] = 1
        plane = nib.Nifti1Image(data, reference.affine)
        countsFine = _quiet(segmentationTools.planarROICrossings, fine, plane)
        countsCoarse = _quiet(segmentationTools.planarROICrossings, coarse, plane)
        assert np.array_equal(countsFine > 0, countsCoarse > 0), shape
        assert (countsFine > 0).mean() > 0.8, shape
