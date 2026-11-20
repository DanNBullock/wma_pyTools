# -*- coding: utf-8 -*-
"""
Claude note: tests for wmaPyTools.segmentationRecipes (the port of the
MATLAB wma_tools segmentation recipes).  The synthetic tests check the
MATLAB semantics of each building block on small hand-made volumes; the
real-data tests (marker realdata) run the pipeline and the priors on the
sub-100206 subset.  Agreement with MATLAB output is not a goal (Dan
Bullock, 2026-11-14): the segmentations are defined by the anatomy, and
MATLAB's unintended quirks are not reproduced.
"""

import numpy as np
import nibabel as nib
import pytest
from dipy.tracking.streamline import Streamlines

import wmaPyTools.segmentationRecipes as sr


# Claude note: --------------------------------------------------------------
# classification helpers
# ---------------------------------------------------------------------------

def test_reconcileClassifications_laterWinsAndNamesMerge():
    #Claude note: streamline 1 is claimed by both; the later one wins
    base = {'names': ['a', 'b'], 'index': np.array([1, 2, 0, 0])}
    add = {'names': ['c', 'a'], 'index': np.array([0, 1, 2, 0])}
    out = sr.reconcileClassifications(base, add)
    assert out['names'] == ['a', 'b', 'c']
    #Claude note: 0 -> a (base), 1 -> c (add wins), 2 -> a (add), 3 -> none
    np.testing.assert_array_equal(out['index'], [1, 3, 1, 0])
    #Claude note: inputs untouched
    np.testing.assert_array_equal(base['index'], [1, 2, 0, 0])


def test_addTractToClassification_andsCriteria():
    c = sr.emptyClassification(4)
    c = sr.addTractToClassification(c, 't1', [True, True, False, True], np.array([True, False, True, True]))
    c = sr.addTractToClassification(c, 't2', [False, False, False, True])
    assert c['names'] == ['t1', 't2']
    np.testing.assert_array_equal(c['index'], [1, 0, 0, 2])
    np.testing.assert_array_equal(sr.classificationBoolByName(c, 't1'), [True, False, False, False])
    #Claude note: absent names give all False rather than raising
    assert not np.any(sr.classificationBoolByName(c, 'nope'))


def test_renameMATLABTracts_opticRadiation():
    old = {'names': ['leftmeyer', 'rightbaum', 'leftArc', 'meyer'], 'index': np.array([1, 2, 3, 4, 0])}
    new = sr.renameMATLABTracts(old)
    assert new['names'] == ['leftInferior_OR', 'rightSuperior_OR', 'leftArc', 'Inferior_OR']
    np.testing.assert_array_equal(new['index'], old['index'])
    #Claude note: the input is not modified
    assert old['names'][0] == 'leftmeyer'


def test_renameMATLABTracts_implausibleSpelling():
    #Claude note: MATLAB v7's '_implausable' suffix becomes '_implausible'
    old = {'names': ['left_frontal_to_temporal_implausable', 'frontal_to_frontal_interHemi',
                     'left_frontal_to_temporal_implausible'], 'index': np.array([1, 2, 3])}
    new = sr.renameMATLABTracts(old)
    assert new['names'] == ['left_frontal_to_temporal_implausible', 'frontal_to_frontal_interHemi',
                            'left_frontal_to_temporal_implausible']


# Claude note: --------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def _plane(reference, mm, dim):
    import wmaPyTools.roiTools as roiTools
    return roiTools.makePlanarROI(reference, mm, dim)


def test_modifyROI_bsc_strictAndMedialCrossesMidline(synthReference, synthAffine):
    #Claude note: a full x-plane at y=10 mm, cut with x-planes
    yPlane = _plane(synthReference, 10, 'y')
    leftXPlane = _plane(synthReference, -10, 'x')
    xCoords = sr._maskVoxelCoords(yPlane)[:, 0]
    medial = sr.modifyROI_bsc(yPlane, leftXPlane, 'medial')
    medialX = sr._maskVoxelCoords(medial)[:, 0]
    #Claude note: strictly beyond the reference (x=-10 itself dropped) and
    #running on into the right hemisphere, up to the image edge
    assert np.min(medialX) > -10
    assert np.isclose(np.max(medialX), np.max(xCoords))
    lateral = sr.modifyROI_bsc(yPlane, leftXPlane, 'lateral')
    lateralX = sr._maskVoxelCoords(lateral)[:, 0]
    assert np.max(lateralX) < -10
    #Claude note: together they are the whole plane minus the reference line
    assert len(medialX) + len(lateralX) + np.sum(np.isclose(xCoords, -10)) == len(xCoords)
    #Claude note: a right hemisphere reference flips medial / lateral
    rightMedial = sr._maskVoxelCoords(sr.modifyROI_bsc(yPlane, [10, 0, 0], 'medial'))[:, 0]
    assert np.max(rightMedial) < 10 and np.isclose(np.min(rightMedial), np.min(xCoords))
    #Claude note: anterior / posterior against a coordinate
    zPlane = _plane(synthReference, 0, 'z')
    posterior = sr._maskVoxelCoords(sr.modifyROI_bsc(zPlane, [0, 4, 0], 'posterior'))
    assert np.max(posterior[:, 1]) < 4


def test_roiFromAtlasLabels_bsc_boxDilation(synthAtlas):
    plain = np.asanyarray(sr.roiFromAtlasLabels_bsc(synthAtlas, 10, 1).dataobj) > 0
    np.testing.assert_array_equal(plain, np.asanyarray(synthAtlas.dataobj) == 10)
    #Claude note: box 3 = one voxel of 26-connected growth
    grown = np.asanyarray(sr.roiFromAtlasLabels_bsc(synthAtlas, 10, 3).dataobj) > 0
    from scipy import ndimage
    np.testing.assert_array_equal(grown, ndimage.binary_dilation(plain, structure=np.ones((3, 3, 3))))


def test_planeFromROI_bsc_matchesLabelBorder(synthAtlas):
    import wmaPyTools.roiTools as roiTools
    fromLabels = sr.planeFromROI_bsc(10, 'anterior', synthAtlas)
    fromMask = sr.planeFromROI_bsc(roiTools.multiROIrequestToMask(synthAtlas, [10]), 'anterior', synthAtlas)
    np.testing.assert_array_equal(np.asanyarray(fromLabels.dataobj), np.asanyarray(fromMask.dataobj))
    #Claude note: on the label's most anterior voxel centre (box 10 spans y in [-10,10) mm)
    assert np.isclose(sr.roiMeanCoord(fromLabels)[1], 8)


# Claude note: --------------------------------------------------------------
# atlas voting
# ---------------------------------------------------------------------------

def test_deislandLabels_bsc_replacesSmallComponents(synthAffine):
    data = np.zeros((12, 12, 12), dtype=np.int32)
    data[2:6, 2:6, 2:6] = 7
    #Claude note: a 4-voxel island of 7 (< 5) and a 5-voxel one (kept)
    data[9, 9, 0:4] = 7
    data[9, 0:5, 9] = 7
    #Claude note: a diagonal neighbour is connected (26-connectivity)
    data[6, 6, 6] = 7
    out = np.asanyarray(sr.deislandLabels_bsc(nib.Nifti1Image(data, synthAffine)).dataobj)
    assert np.all(out[9, 9, 0:4] == 999)
    assert np.all(out[9, 0:5, 9] == 7)
    assert out[6, 6, 6] == 7
    assert np.all(out[2:6, 2:6, 2:6] == 7)


def test_prepareAtlas_bsc_islandsToZeroAndIdempotent(synthAffine):
    data = np.zeros((12, 12, 12), dtype=np.int32)
    data[2:6, 2:6, 2:6] = 7
    data[9, 9, 0:4] = 7
    atlas = nib.Nifti1Image(data, synthAffine)
    cleaned = sr.prepareAtlas_bsc(atlas)
    cleanedData = np.asanyarray(cleaned.dataobj)
    #Claude note: the 4-voxel island goes to 0 (not 999), the rest stays
    assert np.all(cleanedData[9, 9, 0:4] == 0)
    assert np.all(cleanedData[2:6, 2:6, 2:6] == 7)
    #Claude note: running it again changes nothing
    np.testing.assert_array_equal(np.asanyarray(sr.prepareAtlas_bsc(cleaned).dataobj), cleanedData)


def test_inflateAtlasLabels_bsc_growsIntoWhiteMatterOnly(synthAffine):
    #Claude note: a cortical label beside white matter, beside background
    data = np.zeros((14, 14, 14), dtype=np.int32)
    data[2:12, 2:12, 2:12] = 2
    data[2:12, 2:12, 2:5] = 11101
    #Claude note: an island of 999-to-be inside the white matter
    data[8, 8, 9] = 11101
    out = np.asanyarray(sr.inflateAtlasLabels_bsc(nib.Nifti1Image(data, synthAffine), 2).dataobj)
    #Claude note: two layers of white matter become cortex (away from the
    #edges of the box, where the background, 0, outvotes the cortex)
    assert np.all(out[3:11, 3:11, 5] == 11101)
    assert np.all(out[4:10, 4:10, 6] == 11101)
    assert np.all(out[4:10, 4:10, 7] == 2)
    #Claude note: at a box corner 0 wins the vote, as in MATLAB (0 is not an
    #inflation target, so it is a candidate)
    assert out[2, 2, 5] == 0
    #Claude note: background is never inflated into
    assert np.all(out[0, :, :] == 0) and np.all(out[:, :, 13] == 0)
    #Claude note: the single-voxel island became 999 and was then voted
    #back to its surroundings (white matter is excluded from winning, so
    #with no other candidate it takes the smallest label present, 2)
    assert out[8, 8, 9] == 2


def test_endpointAtlasLabels_bsc_voxelRuleAndMode(synthAffine):
    data = np.zeros((10, 10, 10), dtype=np.int32)
    data[3:7, 3:7, 3:7] = 5
    atlas = nib.Nifti1Image(data, synthAffine)
    #Claude note: voxel (5,5,5) is inside: label 5; a voxel at the label's
    #corner (3,3,3) has 8 of 27 neighbours labelled 5, so 0 wins
    centres = nib.affines.apply_affine(synthAffine, np.array([[5.2, 5.2, 5.2], [3.2, 3.2, 3.2]]))
    np.testing.assert_array_equal(sr.endpointAtlasLabels_bsc(atlas, centres), [5, 0])
    #Claude note: default: the voxel holding the point.  2.9 is in voxel 3
    #(window 18 of 27 label 5), 2.4 in voxel 2 (window 9 of 27: 0 wins)
    point = nib.affines.apply_affine(synthAffine, np.array([[2.9, 5.2, 5.2], [2.4, 5.2, 5.2]]))
    np.testing.assert_array_equal(sr.endpointAtlasLabels_bsc(atlas, point), [5, 0])


def test_vote_tieGrowsWindow():
    #Claude note: label 1 below x=4, label 2 above, and a centre slice whose
    #3x3x3 window is an exact 13 / 13 tie (plus the centre voxel, label 3)
    data = np.full((9, 9, 9), 2, dtype=np.int32)
    data[:4, :, :] = 1
    data[4, 3:6, 3:6] = np.array([[1, 1, 1], [1, 3, 2], [2, 2, 2]])
    window = data[3:6, 3:6, 3:6]
    assert np.sum(window == 1) == np.sum(window == 2) == 13
    #Claude note: one extra 2 in the 5x5x5 window breaks the tie
    data[2, 2, 2] = 2
    winner = sr._vote(data, np.array([[4, 4, 4]]))[0]
    assert winner == sr._voteLoop(data, [4, 4, 4]) == 2
    #Claude note: excluding label 2 leaves label 1
    assert sr._vote(data, np.array([[4, 4, 4]]), excludeLabels=[2])[0] == 1


# Claude note: --------------------------------------------------------------
# streamline criteria
# ---------------------------------------------------------------------------

def test_endpointCriteria_bsc(synthStreamlines):
    #Claude note: streamlines 0-4 run x=-25..25, 8 lies in x=-25..-15
    #Claude note: reference x=0 is not < 0, so 'lateral' means x > 0 here
    oneRight = sr.endpointCriteria_bsc(synthStreamlines, [0, 0, 0], 'lateral', 'one')
    assert oneRight[0] and oneRight[9] and not oneRight[8]
    neither = sr.endpointCriteria_bsc(synthStreamlines, [0, 0, 0], 'lateral', 'neither')
    assert neither[8] and not neither[0]
    #Claude note: anterior of y=0: the AP streamlines (5-7) have one end there
    anterior = sr.endpointCriteria_bsc(synthStreamlines, [0, 0, 0], 'anterior', 'either')
    assert np.all(anterior[5:8]) and not anterior[0]


def test_segmentByROIs_operationsAndEmptyROI(synthStreamlines, synthAtlas, synthReference):
    leftBox = sr.roiFromAtlasLabels_bsc(synthAtlas, 10)
    rightBox = sr.roiFromAtlasLabels_bsc(synthAtlas, 20)
    empty = nib.Nifti1Image(np.zeros(synthReference.shape, dtype=np.uint8), synthReference.affine)
    both = sr.segmentByROIs(synthStreamlines, [leftBox, rightBox], ['endpoints', 'endpoints'])
    np.testing.assert_array_equal(np.where(both)[0], [0, 1, 2, 3, 4, 9])
    notRight = sr.segmentByROIs(synthStreamlines, [leftBox, rightBox], ['and', 'not'])
    np.testing.assert_array_equal(np.where(notRight)[0], [8])
    #Claude note: an empty ROI passes every 'not' and fails every inclusion
    np.testing.assert_array_equal(sr.segmentByROIs(synthStreamlines, [leftBox, empty], ['and', 'not']),
                                  sr.segmentByROIs(synthStreamlines, [leftBox], ['and']))
    assert not np.any(sr.segmentByROIs(synthStreamlines, [leftBox, empty], ['and', 'endpoints']))


def test_tractByEndpointROIs_bsc(synthStreamlines, synthAtlas):
    leftBox = sr.roiFromAtlasLabels_bsc(synthAtlas, 10)
    rightBox = sr.roiFromAtlasLabels_bsc(synthAtlas, 20)
    #Claude note: one end in each box, either orientation (9 is reversed)
    np.testing.assert_array_equal(np.where(sr.tractByEndpointROIs_bsc(synthStreamlines, leftBox, rightBox))[0], [0, 1, 2, 3, 4, 9])
    np.testing.assert_array_equal(sr.tractByEndpointROIs_bsc(synthStreamlines, leftBox, rightBox),
                                  sr.tractByEndpointROIs_bsc(synthStreamlines, rightBox, leftBox))
    #Claude note: the same ROI twice = both ends in it (streamline 8)
    np.testing.assert_array_equal(np.where(sr.tractByEndpointROIs_bsc(synthStreamlines, leftBox, leftBox))[0], [8])
    empty = nib.Nifti1Image(np.zeros(synthAtlas.shape, dtype=np.uint8), synthAtlas.affine)
    assert not np.any(sr.tractByEndpointROIs_bsc(synthStreamlines, leftBox, empty))


def test_midpointCriteria_bsc(synthStreamlines):
    #Claude note: AP streamlines 5-7 run y=-20..25 (46 nodes): MATLAB's
    #midpoint is node 23 (1-based), y=2
    anterior = sr.midpointCriteria_bsc(synthStreamlines, [0, 1.5, 0], 'anterior')
    assert np.all(anterior[5:8])
    assert not np.any(sr.midpointCriteria_bsc(synthStreamlines, [0, 2.5, 0], 'anterior')[5:8])
    #Claude note: several criteria are ANDed
    both = sr.midpointCriteria_bsc(synthStreamlines, [0, 1.5, 0], 'anterior', [0, 0, 2], 'superior')
    np.testing.assert_array_equal(np.where(both)[0], [7])


def test_endpointAtlasCriteria_bsc(synthStreamlines, synthAtlas):
    labels = (np.array([10, 10, 0]), np.array([20, 0, 0]))
    three = synthStreamlines[:3]
    np.testing.assert_array_equal(sr.endpointAtlasCriteria_bsc(three, None, [10], 'either', labels), [True, True, False])
    np.testing.assert_array_equal(sr.endpointAtlasCriteria_bsc(three, None, [10, 20], 'both', labels), [True, False, False])
    np.testing.assert_array_equal(sr.endpointAtlasCriteria_bsc(three, None, [10], 'neither', labels), [False, False, True])


def test_intersectAndMergeROIs(synthAtlas):
    leftBox = sr.roiFromAtlasLabels_bsc(synthAtlas, 10)
    grownLeft = sr.roiFromAtlasLabels_bsc(synthAtlas, 10, 3)
    rightBox = sr.roiFromAtlasLabels_bsc(synthAtlas, 20)
    np.testing.assert_array_equal(np.asanyarray(sr.intersectROIs(leftBox, grownLeft).dataobj), np.asanyarray(leftBox.dataobj))
    assert not np.any(np.asanyarray(sr.intersectROIs(leftBox, rightBox).dataobj))
    merged = np.asanyarray(sr.mergeROIs(leftBox, rightBox).dataobj) > 0
    np.testing.assert_array_equal(merged, np.isin(np.asanyarray(synthAtlas.dataobj), [10, 20]))


def test_midlinePlane_bsc_LIAgridStopsShortOfTop():
    #Claude note: a FreeSurfer conformed (LIA) affine, 1 mm, 64^3 for speed
    affine = np.array([[-1, 0, 0, 32], [0, 0, 1, -31.5], [0, -1, 0, 30], [0, 0, 0, 1]], dtype=float)
    atlas = nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.int32), affine)
    coords = sr._maskVoxelCoords(sr.midlinePlane_bsc(atlas))
    assert np.allclose(coords[:, 0], 0)
    #Claude note: z points run 1-30 .. 64-30 = -29..34 while the image spans
    #z = -33..30, so the plane covers z = -29..30 only
    assert np.isclose(coords[:, 2].min(), -29) and np.isclose(coords[:, 2].max(), 30)


def test_dropEmptyTracts():
    c = {'names': ['a', 'b', 'c'], 'index': np.array([3, 0, 1, 3])}
    out = sr.dropEmptyTracts(c)
    assert out['names'] == ['a', 'c']
    np.testing.assert_array_equal(out['index'], [2, 0, 1, 2])


def test_segmentMajorTracts_familyOrderAndCerebellarOptIn(monkeypatch, synthStreamlines, synthAtlas):
    #Claude note: replace the expensive steps with stubs that record which
    #families run, in which order
    called = []
    monkeypatch.setattr(sr, 'inflateAtlasLabels_bsc', lambda atlas, iterations: atlas)
    monkeypatch.setattr(sr, 'streamlineCategoryPriors', lambda *args, **kwargs: sr.emptyClassification(len(synthStreamlines)))
    familyNames = ['segmentAntPostTracts', 'segmentCorpusCallosum', 'segmentThalamicRadiations', 'segmentAslant',
                   'segmentMdLF_ILF', 'segmentpArcTPC', 'segmentOpticRadiation', 'segmentCerebellarTracts',
                   'segmentVOF', 'segmentCST', 'segmentCingulum']
    for iName in familyNames:
        def stub(streamlines, atlas, iName=iName, **kwargs):
            called.append(iName)
            #Claude note: each family claims streamline 0 under its own name
            c = sr.emptyClassification(len(streamlines))
            return sr.addTractToClassification(c, iName + 'Tract', np.arange(len(streamlines)) == 0)
        monkeypatch.setattr(sr, iName, stub)
    out = sr.segmentMajorTracts(synthStreamlines, synthAtlas, deislandAtlas=False)
    assert called == [n for n in familyNames if n != 'segmentCerebellarTracts']
    #Claude note: the last family wins the shared streamline; empty tracts are dropped
    assert out['names'] == ['segmentCingulumTract'] and out['index'][0] == 1
    called.clear()
    sr.segmentMajorTracts(synthStreamlines, synthAtlas, deislandAtlas=False, includeCerebellar=True)
    assert called == familyNames


def test_streamlineCategoryPriors_names(synthAffine):
    #Claude note: a left frontal label (11116), the left thalamus (10), a
    #right frontal label (12116) and white matter (2) elsewhere
    data = np.full((40, 20, 20), 2, dtype=np.int32)
    data[0:8] = 11116
    data[16:20] = 10
    data[32:40] = 12116
    atlas = nib.Nifti1Image(data, np.eye(4))
    streams = Streamlines([
        np.array([[2, 10, 10], [18, 10, 10]], dtype=np.float32),
        np.array([[3, 10, 10], [36, 10, 10]], dtype=np.float32),
        np.array([[18, 10, 10], [26, 10, 10]], dtype=np.float32),
    ])
    #Claude note: pass the atlas as its own "inflated" atlas, and no
    #superficial fibres, to test only the naming
    noUfibers = sr.emptyClassification(3)
    prior = sr.streamlineCategoryPriors(streams, atlas, inflatedAtlas=atlas, superficialClassification=noUfibers)
    namesPerStream = [prior['names'][i - 1] for i in prior['index']]
    assert namesPerStream == ['leftfrontal_to_subcortical', 'frontal_to_frontal_interHemi', 'whiteMatter']


# Claude note: --------------------------------------------------------------
# version 7 priors (inflateAtlasLabels_v3_bsc, segmentSuperficialFibers_v3,
# streamlineCategoryPriors_v7)
# ---------------------------------------------------------------------------

def test_inflateAtlasLabels_v3_majorityNotSmallestLabel():
    #Claude note: cortex (11116) everywhere inside a zero border, one white
    #matter voxel (the only inflation target) with two ventricle (4) voxels
    #among its 26 neighbours
    data = np.zeros((9, 9, 9), dtype=np.int32)
    data[1:8, 1:8, 1:8] = 11116
    data[4, 4, 4] = 2
    data[5, 4, 4] = 4
    data[5, 5, 4] = 4
    atlas = nib.Nifti1Image(data, np.eye(4))
    #Claude note: the vote takes the most common neighbour, cortex (not the
    #smallest label, 4, as MATLAB's labels(find(max(counts))) did)
    fixed = sr._atlasData(sr.inflateAtlasLabels_v3_bsc(atlas, 1))
    assert fixed[4, 4, 4] == 11116
    #Claude note: nothing else changes
    changed = fixed != data
    assert np.sum(changed) == 1 and changed[4, 4, 4]


def test_inflateAtlasLabels_v3_fixedEqualsV1AwayFromEdges():
    #Claude note: a blocky random atlas (3-voxel blocks, so no islands), two
    #voxels of zero border so no target voxel is on the image edge
    rng = np.random.default_rng(7)
    blocks = rng.choice([2, 41, 1000, 11116, 12116, 11140, 10, 4, 24], size=(5, 5, 5))
    data = np.zeros((19, 19, 19), dtype=np.int32)
    data[2:17, 2:17, 2:17] = np.kron(blocks, np.ones((3, 3, 3), dtype=np.int64))
    atlas = nib.Nifti1Image(data, np.eye(4))
    v1 = sr._atlasData(sr.inflateAtlasLabels_bsc(atlas, 2))
    v3 = sr._atlasData(sr.inflateAtlasLabels_v3_bsc(atlas, 2))
    np.testing.assert_array_equal(v1, v3)
    #Claude note: and the inflation did something
    assert np.sum(v3 != data) > 0


def test_segmentSuperficialFibers_v3_criteria():
    #Claude note: left white matter (2) for x < 20, right (41) beyond; a
    #cortical slab at z 20-24 (11116 left, 12116 right)
    data = np.full((40, 30, 30), 2, dtype=np.int32)
    data[20:] = 41
    data[:20, :, 20:25] = 11116
    data[20:, :, 20:25] = 12116
    atlas = nib.Nifti1Image(data, np.eye(4))
    xs = np.arange(5, 16, dtype=np.float32)
    along = lambda x, z: np.column_stack([x, np.full(len(x), 15), z]).astype(np.float32)
    streams = Streamlines([
        #Claude note: 0, short, within the left cortex: a u-fibre
        along(xs, np.full(len(xs), 22)),
        #Claude note: 1, short but ending in the right cortex
        along(np.arange(15, 26, dtype=np.float32), np.full(11, 22)),
        #Claude note: 2, dips into the deep white matter (midpoint there too)
        along(xs, np.array([22, 20, 17, 14, 12, 12, 12, 14, 17, 20, 22])),
        #Claude note: 3, in the left cortex but 30 mm long (back and forth)
        along(np.r_[xs, xs[::-1][1:], xs[1:]], np.full(31, 22)),
    ])
    out = sr.segmentSuperficialFibers_v3(streams, atlas, inflatedAtlas=atlas)
    #Claude note: both tract names are always present (as in MATLAB)
    assert out['names'] == ['left_u-fibers', 'right_u-fibers']
    np.testing.assert_array_equal(out['index'], [1, 0, 0, 0])


def _v7Atlas():
    #Claude note: left frontal (11116), left thalamus (10), left
    #G_insular_short (11118), right frontal (12116), white matter elsewhere
    data = np.full((40, 20, 20), 2, dtype=np.int32)
    data[0:8] = 11116
    data[12:16] = 11
    data[16:20] = 10
    data[24:28] = 11118
    data[32:40] = 12116
    return nib.Nifti1Image(data, np.eye(4))


def test_streamlineCategoryPriors_v7_names():
    atlas = _v7Atlas()
    streams = Streamlines([
        np.array([[2, 10, 10], [18, 10, 10]], dtype=np.float32),
        np.array([[3, 10, 10], [36, 10, 10]], dtype=np.float32),
        np.array([[18, 10, 10], [22, 10, 10]], dtype=np.float32),
        np.array([[18, 10, 10], [26, 10, 10]], dtype=np.float32),
        #Claude note: both ends left frontal, but via the right hemisphere
        np.array([[2, 10, 10], [36, 10, 10], [3, 10, 10]], dtype=np.float32),
        np.array([[14, 10, 10], [4, 10, 10]], dtype=np.float32),
    ])
    #Claude note: the atlas as its own "inflated" atlas and no superficial
    #fibres, to test only the naming
    noUfibers = sr.emptyClassification(len(streams))
    prior = sr.streamlineCategoryPriors_v7(streams, atlas, inflatedAtlas=atlas, superficialClassification=noUfibers,
                                           deislandAtlas=False)
    namesPerStream = [prior['names'][i - 1] for i in prior['index']]
    assert namesPerStream == ['left_frontal_to_thalamic', 'frontal_to_frontal_interHemi', 'whiteMatter',
                              'left_insula_to_thalamic', 'left_frontal_to_frontal_implausible',
                              'left_caudateNAc_to_frontal']
    #Claude note: the same streamlines under v6, for contrast (no split, 118
    #frontal, no implausibility flag)
    prior6 = sr.streamlineCategoryPriors(streams, atlas, inflatedAtlas=atlas, superficialClassification=noUfibers)
    assert [prior6['names'][i - 1] for i in prior6['index']] == [
        'leftfrontal_to_subcortical', 'frontal_to_frontal_interHemi', 'whiteMatter',
        'leftfrontal_to_subcortical', 'leftfrontal_to_frontal', 'leftfrontal_to_subcortical']


def test_streamlineCategoryPriors_v7_caudateNotVentricle():
    #Claude note: a superficial fibre (flagged by hand) from left frontal
    #cortex to the caudate.  MATLAB v7 also lists the caudate as ventricle,
    #which would exclude it from '_ufiber'; the port does not
    atlas = _v7Atlas()
    streams = Streamlines([np.array([[4, 10, 10], [14, 10, 10]], dtype=np.float32)])
    ufibers = sr.addTractToClassification(sr.emptyClassification(1), 'left_u-fibers', [True])
    prior = sr.streamlineCategoryPriors_v7(streams, atlas, inflatedAtlas=atlas, superficialClassification=ufibers,
                                           deislandAtlas=False)
    assert prior['names'] == ['left_caudateNAc_to_frontal_ufiber']
    assert 11 not in sr._PRIOR_V7_VENTRICLE and 50 not in sr._PRIOR_V7_VENTRICLE


def _temporoOccipitalAtlas(dropLabel=None, degenerate=False):
    #Claude note: a 1 mm grid, x -20..19 (midline at 0), y -40..-1, z 0..9,
    #white matter (2 / 41) with single-voxel landmarks per hemisphere:
    #incisure x160 at (-14, -30, 0), calcarine x145 at (-5, -10, 0), pole
    #x143 at (-5, -38, 0) (right: x mirrored), and fusiform (x121) /
    #collateral sulcus (x162) 3x3x3 blocks (so they win the endpoint vote)
    #centred at z 1, posterior (y -35) and anterior (y -15) of the
    #incisure-calcarine line
    affine = np.eye(4)
    affine[:3, 3] = [-20, -40, 0]
    data = np.full((40, 40, 10), 2, dtype=np.int32)
    data[20:] = 41
    toVoxel = lambda x, y: (x + 20, y + 40, 0)
    for iOffset, iSign in ((11000, 1), (12000, -1)):
        #Claude note: degenerate: the calcarine tip straight above the
        #incisure tip (same x and y, z 5)
        placements = [(160, -14, -30), (145, -14, -30) if degenerate else (145, -5, -10), (143, -5, -38),
                      (121, -12, -35), (162, -12, -15), (162, -16, -35), (121, -16, -15)]
        for iBase, x, y in placements:
            if dropLabel == iOffset + iBase:
                continue
            if iBase in (121, 162):
                i, j, _ = toVoxel(iSign * x, y)
                data[i - 1:i + 2, j - 1:j + 2, 0:3] = iOffset + iBase
            elif degenerate and iBase == 145:
                data[toVoxel(iSign * x, y)[:2] + (5,)] = iOffset + iBase
            else:
                data[toVoxel(iSign * x, y)] = iOffset + iBase
    return nib.Nifti1Image(data, affine)


def test_temporoOccipitalLandmarks_synthetic():
    landmarks = sr.temporoOccipitalLandmarks(_temporoOccipitalAtlas())
    np.testing.assert_allclose(landmarks[11000]['incisureTip'], [-14, -30, 0])
    np.testing.assert_allclose(landmarks[11000]['calcarineTip'], [-5, -10, 0])
    np.testing.assert_allclose(landmarks[12000]['occipitalPole'], [5, -38, 0])
    #Claude note: a missing landmark label or a degenerate line raises
    with pytest.raises(ValueError, match='12145 missing'):
        sr.temporoOccipitalLandmarks(_temporoOccipitalAtlas(dropLabel=12145))
    with pytest.raises(ValueError, match='degenerate'):
        sr.temporoOccipitalLandmarks(_temporoOccipitalAtlas(degenerate=True))


def test_temporoOccipitalSplit_v6AndV7():
    #Claude note: fusiform posterior -> collateral sulcus anterior (left),
    #collateral sulcus posterior -> fusiform anterior (right).  MATLAB's
    #lists make both endpoints temporal; the split makes the posterior one
    #occipital
    atlas = _temporoOccipitalAtlas()
    streams = Streamlines([np.array([[-12, -35, 1], [-12, -25, 1], [-12, -15, 1]], dtype=np.float32),
                           np.array([[16, -35, 1], [16, -25, 1], [16, -15, 1]], dtype=np.float32)])
    kwargs = dict(inflatedAtlas=atlas, superficialClassification=sr.emptyClassification(2))
    namesOf = lambda c: [c['names'][i - 1] for i in c['index']]
    assert namesOf(sr.streamlineCategoryPriors(streams, atlas, **kwargs)) == ['lefttemporal_to_temporal',
                                                                              'righttemporal_to_temporal']
    assert namesOf(sr.streamlineCategoryPriors(streams, atlas, temporoOccipitalSplit=True, **kwargs)) == [
        'leftoccipital_to_temporal', 'rightoccipital_to_temporal']
    split7 = sr.streamlineCategoryPriors_v7(streams, atlas, deislandAtlas=False, temporoOccipitalSplit=True, **kwargs)
    assert namesOf(split7) == ['left_occipital_to_temporal', 'right_occipital_to_temporal']


# Claude note: --------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

@pytest.mark.realdata
def test_majorTracts_realData(realData):
    #Claude note: the whole pipeline (island cleaning, priors, 11 families)
    #on the 100k subset finds every one of its 61 tracts (the fixed
    #spino-cerebellar tracts are opt-in and not requested here).  It is no
    #longer compared with MATLAB's classification.mat: MATLAB agreement is
    #not a goal (Dan Bullock, 2026-11-14).  For the record: with MATLAB's
    #endpoint voxel rule the port reproduced all 61 MATLAB tracts (recall
    #>= 0.99, git history before 2026-11-14); with the correct rule the
    #lowest recall for tracts of 20+ MATLAB streamlines is 0.82
    #(leftMotorCerebellar) and the lowest Dice 0.79 (leftInferior_OR), the
    #half-voxel endpoint shift changing labels at region borders
    streamlines = nib.streamlines.load(realData['tck']).streamlines
    ours = sr.segmentMajorTracts(streamlines, nib.load(realData['parc']), includeCerebellar=True)
    assert len(ours['names']) == 61
    counts = np.bincount(ours['index'], minlength=len(ours['names']) + 1)[1:]
    assert np.all(counts > 0), [iName for iName, iCount in zip(ours['names'], counts) if iCount == 0]


@pytest.mark.realdata
def test_categoryPriors_v7_versusV6_realData(realData):
    #Claude note: v7 and v6 on the 100k subset.  There is no MATLAB v7
    #output to compare with (the committed v7 can't run, see
    #streamlineCategoryPriors_v7), so every streamline's v7 name is
    #checked against its v6 name, allowing only the documented v6 -> v7
    #changes: the 'left_' / 'right_' prefix, the category lists (subcortical
    #split, 118 and 19/55 to insula), '_implausible', and superficial fibre
    #membership (v1 -> v3)
    streamlines = nib.streamlines.load(realData['tck']).streamlines
    atlas = sr.prepareAtlas_bsc(nib.load(realData['parc']))
    #Claude note: with the vote fixed, the v3 inflation equals v1 (the
    #island-cleaned atlas reaches nowhere near the image edge)
    inflated = sr.inflateAtlasLabels_v3_bsc(atlas, 2)
    np.testing.assert_array_equal(sr._atlasData(inflated), sr._atlasData(sr.inflateAtlasLabels_bsc(atlas, 2)))
    superficial1 = sr.segmentSuperficialFibers(streamlines, atlas, inflated)
    superficial3 = sr.segmentSuperficialFibers_v3(streamlines, atlas, inflated)
    prior6 = sr.streamlineCategoryPriors(streamlines, atlas, inflated, superficial1)
    prior7 = sr.streamlineCategoryPriors_v7(streamlines, atlas, inflatedAtlas=inflated,
                                            superficialClassification=superficial3, deislandAtlas=False)
    names6 = [prior6['names'][i - 1] for i in prior6['index']]
    names7 = [prior7['names'][i - 1] for i in prior7['index']]
    #Claude note: each endpoint's category under the v6 and the v7 lists
    #(same endpoint labels; first-match order of endpoint 1 in both)
    first, last = sr._endpoints(streamlines)
    both = sr._bothSides
    order6 = [('frontal', both(sr._PRIOR_FRONTAL_BASE)), ('temporal', both(sr._PRIOR_TEMPORAL_BASE)),
              ('occipital', both(sr._PRIOR_OCCIPITAL_BASE)), ('parietal', both(sr._PRIOR_PARIETAL_BASE)),
              ('subcortical', sr.SUBCORTICAL_LABELS), ('spinal', sr._PRIOR_SPINE), ('insula', both([117, 149])),
              ('cerebellum', sr._PRIOR_CEREBELLUM), ('CorpusCallosum', sr._PRIOR_CC),
              ('ventricle', sr._PRIOR_VENTRICLE), ('unlabeled', sr._PRIOR_UNKNOWN), ('whiteMatter', sr.WM_LABELS),
              ('pericallosal', both([167])), ('OpticChi', sr._PRIOR_OPTIC_CHIASM)]
    order7 = [('frontal', both(sr._PRIOR_V7_FRONTAL_BASE)), ('temporal', both(sr._PRIOR_TEMPORAL_BASE)),
              ('occipital', both(sr._PRIOR_OCCIPITAL_BASE)), ('parietal', both(sr._PRIOR_PARIETAL_BASE)),
              ('subcortical', sr._PRIOR_V7_SUBCORTICAL), ('thalamic', sr._PRIOR_V7_THALAMIC),
              ('caudateNAc', sr._PRIOR_V7_CAUDATE_NAC), ('lenticularN', sr._PRIOR_V7_LENTICULAR_N),
              ('hippAmig', sr._PRIOR_V7_HIPP_AMIG), ('spinal', sr._PRIOR_SPINE), ('insula', sr._PRIOR_V7_INSULA),
              ('cerebellum', sr._PRIOR_CEREBELLUM), ('CorpusCallosum', sr._PRIOR_CC),
              ('ventricle', sr._PRIOR_V7_VENTRICLE), ('unlabeled', sr._PRIOR_UNKNOWN), ('whiteMatter', sr.WM_LABELS),
              ('pericallosal', both([167])), ('OpticChi', sr._PRIOR_OPTIC_CHIASM)]
    categories6, categories7 = [], []
    for iEndpoints in (first, last):
        endpointLabels = sr.endpointAtlasLabels_bsc(inflated, iEndpoints)
        categories6.append(sr._terminationCategories(endpointLabels, order6))
        categories7.append(sr._terminationCategories(endpointLabels, order7))
    superficialChanged = (superficial1['index'] > 0) != (superficial3['index'] > 0)
    pooled = ['CorpusCallosum', 'unlabeled', 'OpticChi', 'ventricle', 'whiteMatter', 'pericallosal']
    nImplausible, unexplained = 0, []
    for iStream, (name6, name7) in enumerate(zip(names6, names7)):
        #Claude note: undo '_implausible' and the prefix spelling
        name = name7
        if name.endswith('_implausible'):
            nImplausible += 1
            name = name[:-len('_implausible')]
        if name.startswith('left_') or name.startswith('right_'):
            name = name.replace('_', '', 1)
        #Claude note: undo the category lists: the sorted v7 pair becomes
        #the sorted v6 pair
        if name not in pooled and name != 'MaskFailure':
            pair7 = '_to_'.join(sorted([categories7[0][iStream], categories7[1][iStream]]))
            pair6 = '_to_'.join(sorted([categories6[0][iStream], categories6[1][iStream]]))
            name = name.replace(pair7, pair6, 1)
        #Claude note: what remains must be identical, unless the streamline's
        #superficial fibre membership changed
        if name != name6 and not superficialChanged[iStream]:
            unexplained.append((name6, name7))
    assert unexplained == []
    #Claude note: the plausibility flag fires, rarely (270 of 100k when
    #written)
    assert 0 < nImplausible < 0.01 * len(streamlines)


@pytest.mark.realdata
def test_temporoOccipitalSplit_realData(realData):
    #Claude note: the experimental Destrieux split on the 100k subset, v6
    #priors.  Only endpoints in x121 / x162 may change category, and only
    #from temporal to occipital (MATLAB's lists make them all temporal).
    #1,342 streamlines change name on this subset.
    streamlines = nib.streamlines.load(realData['tck']).streamlines
    atlas = sr.prepareAtlas_bsc(nib.load(realData['parc']))
    landmarks = sr.temporoOccipitalLandmarks(atlas)
    for iOffset in (11000, 12000):
        #Claude note: anatomical order along y: pole < incisure tip < calcarine tip
        iLandmarks = landmarks[iOffset]
        assert iLandmarks['occipitalPole'][1] < iLandmarks['incisureTip'][1] < iLandmarks['calcarineTip'][1]
    inflated = sr.inflateAtlasLabels_bsc(atlas, 2)
    superficial = sr.segmentSuperficialFibers(streamlines, atlas, inflated)
    default = sr.streamlineCategoryPriors(streamlines, atlas, inflated, superficial)
    split = sr.streamlineCategoryPriors(streamlines, atlas, inflated, superficial, temporoOccipitalSplit=True)
    namesDefault = np.array(default['names'], dtype=object)[default['index'] - 1]
    namesSplit = np.array(split['names'], dtype=object)[split['index'] - 1]
    changed = namesDefault != namesSplit
    assert np.sum(changed) == 1342
    #Claude note: every changed streamline has an endpoint in x121 / x162,
    #and its name differs only by temporal -> occipital
    first, last = sr._endpoints(streamlines)
    boundary = sr._bothSides(sr._TEMPORO_OCCIPITAL_BOUNDARY_BASE)
    touches = (np.isin(sr.endpointAtlasLabels_bsc(inflated, first), boundary)
               | np.isin(sr.endpointAtlasLabels_bsc(inflated, last), boundary))
    assert np.all(touches[changed])
    for iDefault, iSplit in zip(namesDefault[changed], namesSplit[changed]):
        assert iDefault.count('temporal') - iSplit.count('temporal') == iSplit.count('occipital') - iDefault.count('occipital') > 0, (iDefault, iSplit)
