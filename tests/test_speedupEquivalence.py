# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan (Claude note: written by Claude)

Claude note: equivalence tests for the 2026-11 speedups.

Each sped-up function (or private helper) is compared against a *reference*
implementation of what the code did before the speedup: either the library
function that used to be called (e.g. dipy's near_roi / density_map, nilearn's
intersect_masks, np.unique), or a compact copy of the original per-streamline
/ per-voxel loop.  The requirement is IDENTICAL output (np.array_equal, same
dtype), not approximate agreement.

The exceptions are the bugs deliberately fixed afterwards (2026-11): the
nifti-criterion segmentation functions are now compared against plain dipy
near_roi (the old code over-passed 'both_end' / 'all'), and the bug fixes
have their own regression tests at the end of this file.

The synthetic streamlines deliberately include nodes on the voxel grid's
corners, edges and faces (where a node is *exactly* the tolerance distance
from a voxel center), degenerate streamlines (single node, all-zero, NaN), and
an oblique affine, since those are where a faster method is most likely to
disagree with the original.
"""

import warnings

import numpy as np
import nibabel as nib
import pandas as pd
import pytest
from dipy.tracking.utils import near_roi, density_map
from dipy.tracking.streamline import Streamlines
from dipy.core.geometry import dist_to_corner
from nilearn import masking
from scipy.ndimage import binary_dilation, binary_erosion, binary_propagation, label
from scipy.spatial.distance import cdist

import wmaPyTools.roiTools as roiTools
import wmaPyTools.segmentationTools as segmentationTools
import wmaPyTools.streamlineTools as streamlineTools
import wmaPyTools.analysisTools as analysisTools

MODES = ['any', 'all', 'either_end', 'both_end']


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _randomWalkStreamlines(rng, affine, shape, nStreams=300, dtype=np.float32):
    """Random 1mm-step walks that stay inside the volume."""
    invAffine = np.linalg.inv(affine)
    lowMM = nib.affines.apply_affine(affine, [2, 2, 2])
    highMM = nib.affines.apply_affine(affine, np.asarray(shape) - 3)
    lowMM, highMM = np.minimum(lowMM, highMM), np.maximum(lowMM, highMM)
    streams = []
    for iStream in range(nStreams):
        nNodes = rng.integers(2, 60)
        steps = rng.normal(size=(nNodes, 3))
        steps /= np.linalg.norm(steps, axis=1, keepdims=True)
        start = rng.uniform(lowMM, highMM)
        nodes = np.clip(start + np.cumsum(steps, axis=0), lowMM, highMM)
        #Claude note: snap some streamlines to the integer mm grid, which on a
        #2mm grid with an even origin puts nodes on voxel faces/edges/corners
        if iStream % 3 == 0:
            nodes = np.round(nodes)
        streams.append(nodes.astype(dtype))
    #Claude note: a streamline whose nodes are exactly on voxel corners
    corner = nib.affines.apply_affine(affine, np.asarray(shape) // 2 + 0.5)
    streams.append(np.asarray([corner, corner + [2, 0, 0], corner + [2, 2, 0]], dtype=dtype))
    return streams


@pytest.fixture
def randomStreamlines(synthAffine):
    from tests.conftest import SYNTH_SHAPE
    rng = np.random.default_rng(1234)
    return Streamlines(_randomWalkStreamlines(rng, synthAffine, SYNTH_SHAPE))


@pytest.fixture
def obliqueAffine():
    #Claude note: rotated, anisotropic, shifted
    from scipy.spatial.transform import Rotation
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler('xyz', [10, -7, 23], degrees=True).as_matrix() @ np.diag([1.5, 2.0, 2.5])
    affine[:3, 3] = [-30, -40, -35]
    return affine


@pytest.fixture
def synthROIs(synthAtlas):
    return [roiTools.multiROIrequestToMask(synthAtlas, [iLabel]) for iLabel in [10, 20, 30]]


# ---------------------------------------------------------------------------
# reference (pre-speedup) implementations
# ---------------------------------------------------------------------------

def _refCriterion(streamlines, roiNifti, includeBool, mode):
    """The intended behavior of the nifti-criterion functions: dipy's near_roi
    on the full streamlines and the full mask (negated for exclusion)."""
    out = near_roi([np.asarray(s) for s in streamlines], roiNifti.affine, np.asanyarray(roiNifti.dataobj), mode=mode)
    return out if includeBool else np.logical_not(out)


def _refRemoveIslandsViaLabelCount(atlasNifti, ignoreLabelsRequest=None):
    """The pre-speedup body of removeIslandsFromAtlas_viaLabelCount."""
    ignoreLabels = np.unique(list(ignoreLabelsRequest or []) + [0])
    atlasData = np.round(np.asanyarray(atlasNifti.dataobj)).astype(int)
    noIslandData = atlasData.copy()
    labels, counts = np.unique(atlasData, return_counts=True)
    islandCounts = np.zeros(len(labels), dtype=int)
    islandVoxCounts = np.zeros(len(labels), dtype=int)
    for labelIterator, iLabels in enumerate(labels):
        labeled_array, num_features = label(atlasData == iLabels)
        curr_labels, curr_counts = np.unique(labeled_array, return_counts=True)
        if len(curr_counts) < 2:
            continue
        maxSizeIndex = np.where(curr_counts[1:] == np.max(curr_counts[1:]))[0][0] + 1
        currIslandMask = np.logical_and(labeled_array > 0, labeled_array != maxSizeIndex)
        islandCounts[labelIterator] = len(curr_counts) - 2
        islandVoxCounts[labelIterator] = np.sum(currIslandMask)
        if not iLabels in ignoreLabels:
            noIslandData[currIslandMask] = 0
    endCountsDict = dict(zip(*np.unique(noIslandData, return_counts=True)))
    report = pd.DataFrame({'labelVal': labels, 'startVoxCount': counts, 'islandCount': islandCounts,
                           'totalIslandVox': islandVoxCounts,
                           'endVoxCount': [endCountsDict.get(i, 0) for i in labels]})
    return noIslandData, report


def _refRemoveIslandsFromAtlas(atlasNifti):
    """The pre-speedup body of removeIslandsFromAtlas (prints dropped)."""
    labels, counts = np.unique(np.asanyarray(atlasNifti.dataobj), return_counts=True)
    detectBackground = int(labels[np.where(np.max(counts) == counts)[0][0]])
    atlasLabels = labels.astype(int)[labels.astype(int) != detectBackground]
    atlasData = np.array(np.asanyarray(atlasNifti.dataobj))
    origData = atlasData.copy()
    reportRows = []
    for iLabels in atlasLabels:
        erosionIterations = 0
        forcedMaxIterations = 3
        currentErosion = origData == iLabels
        while np.logical_and(erosionIterations < forcedMaxIterations, np.sum(currentErosion) > 28):
            currentErosion = binary_erosion(currentErosion)
            erosionIterations += 1
            if np.sum(currentErosion) == 0:
                forcedMaxIterations -= 1
                erosionIterations = 0
                currentErosion = origData == iLabels
        islandRemoved = binary_propagation(currentErosion, mask=origData == iLabels)
        reportRows.append([iLabels, np.abs(np.sum(currentErosion) - np.sum(origData == iLabels)),
                           np.abs(np.sum(islandRemoved) - np.sum(origData == iLabels))])
        atlasData[np.logical_and(origData == iLabels, np.logical_not(islandRemoved))] = detectBackground
    return atlasData, reportRows


def _refOrientAllStreamlines(streams):
    """The pre-speedup per-streamline loop of orientAllStreamlines."""
    flipCount = 0
    for i in range(len(streams)):
        traversals = analysisTools.cumulativeTraversalStream(streams[i])
        maxDim = np.where(np.max(traversals) == traversals)[0][0]
        if streams[i][0, :][maxDim] < streams[i][-1, :][maxDim]:
            streams[i] = streams[i][::-1]
            flipCount += 1
    return streams, flipCount


def _refStreamGeomQuantifications(streams):
    """The pre-speedup per-streamline loop of streamGeomQuantifications."""
    columns = ["length", "fullDisp", "efficiencyRat", "asymRat", "bioPriorCost"]
    arcLength = lambda nodes: np.sum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))
    rows = []
    for s in streams:
        s = np.asarray(s)
        if len(s) < 5:
            rows.append([np.nan] * 5)
            continue
        mid = int(np.floor(len(s) / 2 + .5)) - 1
        fl, sl = arcLength(s[:mid + 1, :]), arcLength(s[mid:, :])
        disp = np.linalg.norm(s[0, :] - s[-1, :])
        fd, sd = np.linalg.norm(s[0, :] - s[mid, :]), np.linalg.norm(s[-1, :] - s[mid, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            asym = np.square(np.divide(fd, fl) - np.divide(sd, sl))
            rows.append([fl + sl, disp, np.divide(disp, fl + sl), asym, np.divide(1, (1 - asym))])
    return pd.DataFrame(rows, columns=columns)


def _assertIdentical(new, ref):
    new, ref = np.asarray(new), np.asarray(ref)
    assert new.dtype == ref.dtype
    assert new.shape == ref.shape
    assert np.array_equal(new, ref, equal_nan=new.dtype.kind == 'f')


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('data', [
    np.array([0., 1., 1., 0.]), np.array([0, 1], dtype=np.int32), np.array([True, False]),
    np.zeros(5), np.ones(5), np.array([0., 2.]), np.array([0., 1., 2.]), np.array([0., np.nan, 1.]),
    np.array([-0., 1.]), np.array([0.5, 1.]), np.zeros(0)])
def test_uniqueMaskValues_matchesNpUnique(data):
    _assertIdentical(segmentationTools._uniqueMaskValues(data), np.unique(data))


def test_nodesNearCoords_matchesCdist(synthAffine, randomStreamlines):
    rng = np.random.default_rng(5)
    nodes = np.concatenate([np.asarray(s) for s in randomStreamlines]).astype(np.float64)
    roiCoords = nib.affines.apply_affine(synthAffine, rng.integers(5, 35, size=(400, 3)))
    tol = dist_to_corner(synthAffine)
    expected = np.min(cdist(nodes, roiCoords), axis=1) <= tol
    _assertIdentical(segmentationTools._nodesNearCoords(nodes, roiCoords, tol), expected)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('useOblique', [False, True])
def test_nearROIExact_matchesDipyNearRoi(mode, useOblique, synthAffine, obliqueAffine, randomStreamlines, synthAtlas):
    affine = obliqueAffine if useOblique else synthAffine
    roi = np.asanyarray(synthAtlas.dataobj) > 0
    streams = [np.asarray(s) for s in randomStreamlines]
    #Claude note: degenerate cases dipy treats specially
    streams += [np.zeros((4, 3), dtype=np.float32),
                streams[0][:1].copy(),
                np.vstack([streams[1], [[np.nan, 0, 0]]]).astype(np.float32)]
    _assertIdentical(segmentationTools._nearROIExact(streams, affine, roi, mode=mode), near_roi(streams, affine, roi, mode=mode))
    _assertIdentical(segmentationTools._nearROIExact(Streamlines(streams), affine, roi, mode=mode), near_roi(streams, affine, roi, mode=mode))


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('includeBool', [True, False])
def test_applyNiftiCriteria_matchNearRoi(mode, includeBool, randomStreamlines, synthROIs):
    for iROI in synthROIs:
        expected = _refCriterion(randomStreamlines, iROI, includeBool, mode)
        _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(randomStreamlines, iROI, includeBool, mode), expected)
        _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY(randomStreamlines, iROI, includeBool, mode).astype(bool), expected)
        _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test([np.asarray(x) for x in randomStreamlines], iROI, includeBool, mode), expected)


@pytest.mark.parametrize('mode', MODES)
def test_near_roi_fast_matchesNearRoi(mode, randomStreamlines, synthROIs):
    for iROI in synthROIs:
        roiData = np.asanyarray(iROI.dataobj)
        new = segmentationTools.near_roi_fast(randomStreamlines, iROI.affine, roiData, mode=mode)
        assert isinstance(new, list)
        _assertIdentical(np.asarray(new, dtype=bool), _refCriterion(randomStreamlines, iROI, True, mode))


def test_segmentTractMultiROI_bothVariants_matchNearRoiAndEachOther(randomStreamlines, synthROIs):
    rois = [synthROIs[0], synthROIs[2], synthROIs[1]]
    includes = [True, False, True]
    modes = ['either_end', 'any', 'any']
    expected = np.ones(len(randomStreamlines), dtype=bool)
    for iROI, iInclude, iMode in zip(rois, includes, modes):
        expected &= _refCriterion(randomStreamlines, iROI, iInclude, iMode)
    _assertIdentical(segmentationTools.segmentTractMultiROI(randomStreamlines, rois, includes, modes), expected)
    _assertIdentical(segmentationTools.segmentTractMultiROI_fast(randomStreamlines, rois, includes, modes), expected)


def test_subsetStreamsByROIboundingBox_matchReference(randomStreamlines, synthROIs):
    iROI = synthROIs[0]
    bounds = roiTools.subjectSpaceMaskBoundaryCoords(iROI)
    dtc = dist_to_corner(iROI.affine)
    bounds[0, :] -= dtc
    bounds[1, :] += dtc
    refSubsets = [segmentationTools.streamlineNodesWithinBounds(s, bounds) for s in randomStreamlines]
    outIndexes, outStreams = segmentationTools.subsetStreamsNodesByROIboundingBox(randomStreamlines, iROI)
    _assertIdentical(outIndexes, np.where([x.size > 0 for x in refSubsets])[0])
    refStreams = Streamlines(refSubsets)
    assert len(outStreams) == len(refStreams)
    for iNew, iRef in zip(outStreams, refStreams):
        _assertIdentical(iNew, iRef)
    newBool = segmentationTools.subsetStreamsByROIboundingBox(randomStreamlines, iROI)
    _assertIdentical(newBool, [segmentationTools.streamlineWithinBounds(s, bounds) for s in randomStreamlines])


@pytest.mark.parametrize('relativePosition,whichEndpoints', [('anterior', 'one'), ('posterior', 'both'), ('anterior', 'neither')])
def test_applyEndpointAndMidpointCriteria_matchReference(relativePosition, whichEndpoints, randomStreamlines, synthAtlas):
    plane = roiTools.planarROIFromAtlasLabelBorder(synthAtlas, [30], relativePosition)
    new = segmentationTools.applyEndpointCriteria(randomStreamlines, plane, relativePosition, whichEndpoints)
    #Claude note: recompute the endpoint criterion from per-streamline endpoints,
    #as the original loop gathered them
    asList = [np.asarray(s) for s in randomStreamlines]
    endpoints = np.asarray([np.concatenate([s[0, :], s[-1, :]]) for s in asList]).astype(np.float64)
    planeCoords = nib.affines.apply_affine(plane.affine, np.array(np.where(np.asanyarray(plane.dataobj))).T)
    #Claude note: the criteria cap at the image's outer voxel faces (2026-11)
    fullBounds = roiTools._imageFaceBounds(plane.affine, plane.shape)
    dim = 1
    bound = np.max(fullBounds[:, 1]) if relativePosition == 'anterior' else np.min(fullBounds[:, 1])
    sortedBounds = np.sort([planeCoords[0, dim], bound])
    inRange = lambda vals: np.logical_and(vals > sortedBounds[0], vals < sortedBounds[1])
    endpointSum = np.add(inRange(endpoints[:, dim]), inRange(endpoints[:, 3 + dim]), dtype=int)
    _assertIdentical(new, endpointSum == {'neither': 0, 'one': 1, 'both': 2}[whichEndpoints])
    from dipy.segment.featurespeed import MidpointFeature
    midpoints = np.squeeze(np.asarray(list(map(MidpointFeature().extract, randomStreamlines))))
    newMid = segmentationTools.applyMidpointCriteria(randomStreamlines, plane, relativePosition)
    #Claude note: recompute the criterion from dipy's own midpoints
    _assertIdentical(newMid, inRange(midpoints[:, dim]))


# ---------------------------------------------------------------------------
# ROI / atlas
# ---------------------------------------------------------------------------

def test_multiROIrequestToMask_matchesReference(synthAtlas):
    atlasData = np.asanyarray(synthAtlas.dataobj)
    for request in [[10], [10, 30], 20, [10, 99]]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            new = roiTools.multiROIrequestToMask(synthAtlas, request)
        expected = np.zeros(synthAtlas.shape)
        expected[np.where(np.isin(np.round(atlasData).astype(int), np.asarray(request)))] = True
        _assertIdentical(np.asanyarray(new.dataobj), expected)
        if request == [10, 99]:
            assert any('[np.int64(99)]' in str(w.message) or '[99]' in str(w.message) for w in caught)
        else:
            assert not caught


@pytest.fixture
def islandAtlas(synthAffine, synthAtlas):
    """The synthetic atlas plus islands, a label touching the volume edge,
    and two equal-size components of one label (a tie)."""
    data = np.array(np.asanyarray(synthAtlas.dataobj))
    data[2:4, 2:4, 2:4] = 10       # island of label 10
    data[30, 40, 30] = 20          # single voxel island of 20
    data[0:3, 0:3, 30:35] = 40     # label on the volume edge
    data[20:22, 2:4, 2:4] = 50     # label 50, two equal components
    data[20:22, 10:12, 2:4] = 50
    return nib.Nifti1Image(data, synthAffine)


def test_removeIslandsFromAtlas_viaLabelCount_matchesReference(islandAtlas):
    for ignore in [None, [20]]:
        newAtlas, newReport = roiTools.removeIslandsFromAtlas_viaLabelCount(islandAtlas, ignore)
        refData, refReport = _refRemoveIslandsViaLabelCount(islandAtlas, ignore)
        _assertIdentical(np.asanyarray(newAtlas.dataobj), refData)
        pd.testing.assert_frame_equal(newReport, refReport)


def test_removeIslandsFromAtlas_matchesReference(islandAtlas, capsys):
    refData, refRows = _refRemoveIslandsFromAtlas(islandAtlas)
    #Claude note: the input atlas is no longer edited in place (bug fix)
    inputCopy = nib.Nifti1Image(np.array(np.asanyarray(islandAtlas.dataobj)), islandAtlas.affine)
    newAtlas, newReport = roiTools.removeIslandsFromAtlas(inputCopy)
    _assertIdentical(np.asanyarray(newAtlas.dataobj), refData)
    _assertIdentical(np.asanyarray(inputCopy.dataobj), np.asanyarray(islandAtlas.dataobj))
    assert newReport.values.tolist() == [list(r) for r in refRows]


def test_inflationVotes_matchPerVoxelLoop(islandAtlas):
    data = np.pad(np.round(np.asanyarray(islandAtlas.dataobj)).astype(int), [1, 1])
    targets = [0]
    remaining = [10, 20, 30, 40, 50]
    candidates = np.logical_and(binary_dilation(np.isin(data, remaining)), np.isin(data, targets))
    coords = np.asarray(np.where(candidates)).T
    expected = [roiTools._inflationVoteForVoxel(data, iCoord, targets) for iCoord in coords]
    _assertIdentical(roiTools._inflationVotes(data, coords, targets, chunkSize=37), np.asarray(expected, dtype=np.int64))


# ---------------------------------------------------------------------------
# streamline geometry
# ---------------------------------------------------------------------------

def test_orientAllStreamlines_matchesReference(randomStreamlines):
    new = streamlineTools.orientAllStreamlines(randomStreamlines.copy())
    ref, refCount = _refOrientAllStreamlines(randomStreamlines.copy())
    assert len(new) == len(ref)
    for iNew, iRef in zip(new, ref):
        _assertIdentical(iNew, iRef)
    #Claude note: an indexed view writes through to its parent, as before
    parentNew, parentRef = randomStreamlines.copy(), randomStreamlines.copy()
    streamlineTools.orientAllStreamlines(parentNew[np.arange(0, len(parentNew), 3)])
    _refOrientAllStreamlines(parentRef[np.arange(0, len(parentRef), 3)])
    _assertIdentical(parentNew._data, parentRef._data)


def test_streamGeomQuantifications_matchesReference(randomStreamlines):
    #Claude note: add some short (<5 node) streamlines, which get NaN rows
    streams = Streamlines(list(randomStreamlines) + [np.zeros((3, 3), np.float32) + 1, np.arange(12, dtype=np.float32).reshape(4, 3)])
    pd.testing.assert_frame_equal(analysisTools.streamGeomQuantifications(streams), _refStreamGeomQuantifications(streams))
    pd.testing.assert_frame_equal(analysisTools.streamGeomQuantifications(randomStreamlines), _refStreamGeomQuantifications(randomStreamlines))


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def realLoaded(realData):
    return {'parc': nib.load(realData['parc']),
            'streamlines': nib.streamlines.load(realData['tck']).streamlines}


@pytest.mark.realdata
@pytest.mark.parametrize('mode', MODES)
def test_real_applyNiftiCriteria_matchNearRoi(mode, realLoaded):
    streams = realLoaded['streamlines'][:2000]
    for iLabel in [10, 12]:
        roi = roiTools.multiROIrequestToMask(realLoaded['parc'], [iLabel])
        expected = _refCriterion(streams, roi, True, mode)
        _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(streams, roi, True, mode), expected)
        newFast = segmentationTools.near_roi_fast(streams, roi.affine, np.asanyarray(roi.dataobj), mode=mode)
        _assertIdentical(np.asarray(newFast, dtype=bool), expected)


@pytest.mark.realdata
def test_real_streamlineGeometry_matchesReference(realLoaded):
    streams = realLoaded['streamlines'][:3000]
    pd.testing.assert_frame_equal(analysisTools.streamGeomQuantifications(streams), _refStreamGeomQuantifications(streams))
    new = streamlineTools.orientAllStreamlines(streams.copy())
    ref, _ = _refOrientAllStreamlines(streams.copy())
    _assertIdentical(new.get_data(), ref.get_data())


@pytest.mark.realdata
def test_real_removeIslandsViaLabelCount_matchesReference(realLoaded):
    #Claude note: a 96^3 crop around the thalami keeps the reference fast
    parc = realLoaded['parc']
    cropped = nib.Nifti1Image(np.ascontiguousarray(np.asanyarray(parc.dataobj)[80:176, 80:176, 80:176]), parc.affine)
    newAtlas, newReport = roiTools.removeIslandsFromAtlas_viaLabelCount(cropped)
    refData, refReport = _refRemoveIslandsViaLabelCount(cropped)
    _assertIdentical(np.asanyarray(newAtlas.dataobj), refData)
    pd.testing.assert_frame_equal(newReport, refReport)


# ---------------------------------------------------------------------------
# chunking (bounded memory) and voxel-boundary handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('chunkNodes', [1, 37, 500])
def test_chunkedProcessing_matchesReference(chunkNodes, monkeypatch, randomStreamlines, synthROIs):
    """Claude note: tiny chunk sizes put many chunk boundaries (including
    single-streamline chunks) inside the data; results must not change.
    Chunks always hold whole streamlines."""
    monkeypatch.setattr(segmentationTools, '_CHUNK_NODES', chunkNodes)
    chunks = list(segmentationTools._streamlineChunks(randomStreamlines))
    assert sum(len(c) for _, c in chunks) == len(randomStreamlines)
    for firstIndex, chunk in chunks:
        for iOffset, iStream in enumerate(chunk):
            _assertIdentical(iStream, randomStreamlines[firstIndex + iOffset])
    iROI = synthROIs[0]
    for mode in MODES:
        expected = _refCriterion(randomStreamlines, iROI, True, mode)
        _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(randomStreamlines, iROI, True, mode), expected)
        _assertIdentical(np.asarray(segmentationTools.near_roi_fast(randomStreamlines, iROI.affine, np.asanyarray(iROI.dataobj), mode=mode), dtype=bool), expected)
    new = streamlineTools.orientAllStreamlines(randomStreamlines.copy())
    ref, _ = _refOrientAllStreamlines(randomStreamlines.copy())
    _assertIdentical(new.get_data(), ref.get_data())
    pd.testing.assert_frame_equal(analysisTools.streamGeomQuantifications(randomStreamlines), _refStreamGeomQuantifications(randomStreamlines))


def test_orientAllStreamlines_nanLeavesInputForOriginalLoop(randomStreamlines):
    """Claude note: a NaN anywhere sends orientAllStreamlines back to the
    original loop *before* anything has been flipped (the loop then raises,
    as it always did)."""
    streams = Streamlines(list(randomStreamlines) + [np.array([[0, 0, 0], [np.nan, 1, 1], [2, 2, 2]], np.float32)])
    before = streams.get_data().copy()
    assert streamlineTools._orientAllStreamlinesVectorised(streams) is None
    _assertIdentical(streams.get_data(), before)


# ---------------------------------------------------------------------------
# regression tests for the bugs fixed in 2026-11
# ---------------------------------------------------------------------------

def test_bothEnd_requiresBothEndpoints(synthAffine, synthAtlas):
    """Claude note: a streamline starting in the left box (label 10) and
    ending far away used to pass 'both_end', because the bounding box cut it
    down to its in-box nodes first."""
    roi = roiTools.multiROIrequestToMask(synthAtlas, [10])
    xs = np.arange(-20, 30, 1.0)
    oneEnd = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]).astype(np.float32)
    bothEnds = np.vstack([oneEnd[:5], oneEnd[4::-1]])
    streams = Streamlines([oneEnd, bothEnds])
    _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(streams, roi, True, 'both_end'), np.array([False, True]))
    _assertIdentical(np.asarray(segmentationTools.near_roi_fast(streams, roi.affine, np.asanyarray(roi.dataobj), mode='both_end')), np.array([False, True]))
    #Claude note: the second streamline lies entirely inside the ROI
    _assertIdentical(segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(streams, roi, True, 'all'), np.array([False, True]))


def test_maskChecks(synthAffine, randomStreamlines):
    from tests.conftest import SYNTH_SHAPE
    fullMask = nib.Nifti1Image(np.ones(SYNTH_SHAPE), synthAffine)
    #Claude note: an all-ones mask used to raise "not convertable"
    out = segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(randomStreamlines, fullMask, True, 'any')
    _assertIdentical(out, _refCriterion(randomStreamlines, fullMask, True, 'any'))
    #Claude note: >2 values used to raise numpy's broadcasting ValueError; now a
    #warning, with non-zero voxels as the ROI
    multiData = np.zeros(SYNTH_SHAPE)
    multiData[10:15, 10:15, 10:15] = 1
    multiData[20:25, 20:25, 20:25] = 2
    multiMask = nib.Nifti1Image(multiData, synthAffine)
    with pytest.warns(UserWarning, match='other than 0 and 1'):
        out = segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(randomStreamlines, multiMask, True, 'any')
    _assertIdentical(out, _refCriterion(randomStreamlines, multiMask, True, 'any'))
    with pytest.warns(UserWarning, match='empty'):
        out = segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(randomStreamlines, nib.Nifti1Image(np.zeros(SYNTH_SHAPE), synthAffine), True, 'any')
    assert not np.any(out)
    nanData = multiData.copy()
    nanData[0, 0, 0] = np.nan
    with pytest.raises(Exception, match='non-finite'):
        segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(randomStreamlines, nib.Nifti1Image(nanData, synthAffine), True, 'any')


def test_dummyNiftiAndNeck_independentOfParentTractogram(randomStreamlines):
    """Claude note: an indexed subset now gets the same dummy grid, and the
    same neck nodes, as the same streamlines held in their own buffer."""
    subsetIndexes = np.arange(10, 60)
    asView = randomStreamlines[subsetIndexes]
    asOwnCopy = Streamlines([np.array(s) for s in asView])
    viewNifti = streamlineTools.dummyNiftiForStreamlines(asView)
    copyNifti = streamlineTools.dummyNiftiForStreamlines(asOwnCopy)
    _assertIdentical(viewNifti.affine, copyNifti.affine)
    assert viewNifti.shape == copyNifti.shape
    _assertIdentical(streamlineTools.findTractNeckNode(asView), streamlineTools.findTractNeckNode(asOwnCopy))
    #Claude note: and the grid spans the tract, not the parent tractogram
    tractMin = np.min(np.concatenate(list(asOwnCopy)), axis=0)
    _assertIdentical(viewNifti.affine[:3, 3], tractMin.astype(np.float64))


def test_inflateAtlas_nonZeroBackground(synthAffine):
    """Claude note: with a background label other than 0, the padding's 0s
    used to act as a label to inflate from, and target voxels could get no
    votes and crash."""
    data = np.full((20, 20, 20), 7, dtype=np.int32)
    data[5:10, 5:10, 5:10] = 3
    data[10:15, 5:10, 5:10] = 4
    data[0:3, 0:3, 0:3] = 5
    atlas = nib.Nifti1Image(data, synthAffine)
    inflated = np.asanyarray(roiTools.inflateAtlasIntoWMandBG(atlas, 1, reportSavePath=None).dataobj)
    #Claude note: no 0 (padding value) may appear, and only labels present
    assert set(np.unique(inflated)) <= {3, 4, 5, 7}
    assert np.sum(inflated == 3) > np.sum(data == 3)


def test_backgroundLabel_explicitAndWarned(synthAffine, islandAtlas):
    """Claude note: background inference is a heuristic; an explicit
    backgroundLabel overrides it, and a doubtful inference warns."""
    #a standard atlas (background 0 is the most common label): no warning,
    #and an explicit backgroundLabel=0 gives the same result
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        inferred = np.asanyarray(roiTools.inflateAtlasIntoWMandBG(islandAtlas, 1, reportSavePath=None).dataobj)
    explicit = np.asanyarray(roiTools.inflateAtlasIntoWMandBG(islandAtlas, 1, reportSavePath=None, backgroundLabel=0).dataobj)
    _assertIdentical(inferred, explicit)
    #a "cropped" atlas where white matter (2) outnumbers background (0)
    data = np.full((20, 20, 20), 2, dtype=np.int32)
    data[:, :, 15:] = 0
    data[5:10, 5:10, 5:10] = 10
    cropped = nib.Nifti1Image(data, synthAffine)
    with pytest.warns(UserWarning, match='inferred background label 2.*Label 0 is present'):
        guessed = np.asanyarray(roiTools.inflateAtlasIntoWMandBG(cropped, 1, reportSavePath=None).dataobj)
    #the heuristic inflates label 10 into the white matter...
    assert np.sum(guessed == 10) > np.sum(data == 10)
    #...while with the background declared, only background is inflated into
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        declared = np.asanyarray(roiTools.inflateAtlasIntoWMandBG(cropped, 1, reportSavePath=None, backgroundLabel=0).dataobj)
    #Claude note: label 10 (surrounded by white matter) is untouched, and white
    #matter, now an ordinary label, grows one voxel into the background
    expected = data.copy()
    expected[:, :, 15] = 2
    #(inflateAtlasIntoWMandBG returns int64 labels, as it always has)
    _assertIdentical(declared, expected.astype(declared.dtype))
    with pytest.warns(UserWarning, match='not present'):
        roiTools.removeIslandsFromAtlas(cropped, backgroundLabel=99)
