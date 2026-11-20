# -*- coding: utf-8 -*-
"""
Claude note: tests for the synthetic streamline generators themselves
(wmaPyTools.syntheticBundles, added 2026-11): determinism, the requested
parameters being honoured, and the ground truth being right.
"""

import numpy as np
import pytest

import wmaPyTools.syntheticBundles as syntheticBundles


def _gaps(streamlines):
    return np.concatenate([np.linalg.norm(np.diff(np.asarray(s, dtype=float), axis=0), axis=1) for s in streamlines])


@pytest.mark.parametrize('shape', syntheticBundles.BUNDLE_SHAPES)
def test_bundleShapes_basicProperties(shape):
    streamlines, truth = syntheticBundles.generateBundle(shape, nStreamlines=40, length=60., spacing=1.,
                                                         reverseFraction=0.25, outlierFraction=0.1, seed=2)
    assert len(streamlines) == 40
    assert truth['outlier'].sum() == 4
    assert truth['shape'] == shape
    #Claude note: evenly spaced nodes, except one shorter final step (at the
    #start of the node list for reversed streamlines)
    for s, isReversed in zip(streamlines, truth['reversed']):
        gaps = _gaps([s])
        regular, final = (gaps[1:], gaps[0]) if isReversed else (gaps[:-1], gaps[-1])
        assert np.allclose(regular, 1., atol=0.05)
        assert final <= 1. + 1e-4  # streamlines are stored as float32
    #Claude note: non-outlier streamlines start near the start region, or
    #near the end region if reversed
    for s, isReversed, isOutlier in zip(streamlines, truth['reversed'], truth['outlier']):
        if isOutlier:
            continue
        first = np.asarray(s[0], dtype=float)
        nearStart = np.linalg.norm(first - truth['startRegion'])
        nearEnd = np.linalg.norm(first - truth['endRegion'])
        assert (nearEnd < nearStart) == bool(isReversed)


def test_deterministicForSeed():
    a, truthA = syntheticBundles.generateBundle('arc', nStreamlines=20, spacingJitter=0.5, reverseFraction=0.5, seed=11)
    b, truthB = syntheticBundles.generateBundle('arc', nStreamlines=20, spacingJitter=0.5, reverseFraction=0.5, seed=11)
    c, _ = syntheticBundles.generateBundle('arc', nStreamlines=20, spacingJitter=0.5, reverseFraction=0.5, seed=12)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(truthA['reversed'], truthB['reversed'])
    assert not all(np.array_equal(x, y) for x, y in zip(a, c) if len(x) == len(y))


def test_unevenSpacingWithinRequestedBounds():
    streamlines, _ = syntheticBundles.generateBundle('hook', nStreamlines=30, spacing=2., spacingJitter=0.6, seed=3)
    #Claude note: leave out each streamline's shorter final step (no reversal here)
    gaps = np.concatenate([_gaps([s])[:-1] for s in streamlines])
    assert gaps.min() >= 2. * 0.4 - 1e-6 and gaps.max() <= 2. * 1.6 + 1e-6
    assert gaps.std() > 0.3


def test_rigidPlacementPreservesGeometry():
    plain, truthPlain = syntheticBundles.generateBundle('commissural', nStreamlines=10, seed=5)
    placed, truthPlaced = syntheticBundles.generateBundle('commissural', nStreamlines=10, seed=5,
                                                          rotation=(30, -20, 60), translation=(10., 20., -5.))
    #Claude note: distances within each streamline are unchanged
    assert np.allclose(_gaps(plain), _gaps(placed), atol=1e-4)
    assert not np.allclose(truthPlain['neck'], truthPlaced['neck'])


def test_planeCaseClassesHaveTheirCounts():
    streamlines, truth, plane = syntheticBundles.generatePlaneCrossingCases(nPerClass=30, seed=8)
    for caseClass, count in syntheticBundles.PLANE_CASE_CLASSES.items():
        assert np.all(truth['crossings'][truth['class'] == caseClass] == count), caseClass
    #Claude note: the planar ROI is one voxel thick along exactly its second axis
    occupied = [len(np.unique(axisIndexes)) for axisIndexes in np.nonzero(np.asanyarray(plane.dataobj))]
    assert occupied[1] == 1 and occupied[0] > 1 and occupied[2] > 1


def test_referenceCoversStreamlines():
    streamlines, _ = syntheticBundles.generateBundle('loop', nStreamlines=15, rotation=(10, 50, 0), seed=6)
    reference = syntheticBundles.referenceForStreamlines(streamlines, voxelSize=2., rotation=(10, 50, 0))
    import nibabel as nib
    voxels = nib.affines.apply_affine(np.linalg.inv(reference.affine), np.concatenate(list(streamlines)))
    assert np.all(voxels >= -0.5) and np.all(voxels <= np.asarray(reference.shape) - 0.5)
