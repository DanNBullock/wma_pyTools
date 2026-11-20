# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Claude note: streamline orientation tests on synthetic bundles.  These are a
first seed of a synthetic bundle simulation suite.

Properties checked:
  consistency : every streamline of a coherent bundle ends up running the
                same way
  invariance  : randomly reversing input streamlines does not change the
                result
"""

import numpy as np
import pytest
from dipy.tracking.streamline import Streamlines, set_number_of_points

import wmaPyTools.streamlineTools as streamlineTools


def _straightBundle(nStreams=60, seed=0):
    """parallel-ish streamlines along x, pinched in the middle (a neck)"""
    rng = np.random.default_rng(seed)
    streams = []
    for _ in range(nStreams):
        x = np.linspace(-40, 40, 161)
        pinch = 1 - 0.8 * np.exp(-x ** 2 / 200)
        y = rng.normal(0, 3) * pinch
        z = rng.normal(0, 3) * pinch
        streams.append(np.column_stack([x, np.full_like(x, y), np.full_like(x, z)]))
    return streams


def _arcBundle(nStreams=60, seed=0):
    """
    a C-shaped arc (arcuate-like): starts running along y, bends through
    ~180 degrees, so the axis of greatest travel differs between streamlines
    of different lengths / radii
    """
    rng = np.random.default_rng(seed)
    streams = []
    for _ in range(nStreams):
        radius = 30 + rng.normal(0, 2)
        sweep = np.pi * rng.uniform(0.8, 1.15)
        offset = rng.normal(0, 1.5)
        #fixed 0.5mm step, as tractography produces (orientTractUsingNeck_Robust
        #refuses bundles whose node spacing varies by more than 10%)
        nNodes = int(round((radius + offset) * sweep / 0.5)) + 1
        theta = np.linspace(-np.pi / 2, -np.pi / 2 + sweep, nNodes)
        x = (radius + offset) * np.cos(theta)
        y = (radius + offset) * np.sin(theta)
        z = np.full_like(x, rng.normal(0, 1.5))
        streams.append(np.column_stack([x, y, z]))
    return streams


def _randomlyReversed(streams, seed=1):
    rng = np.random.default_rng(seed)
    flips = rng.random(len(streams)) < 0.5
    return [s[::-1] if f else s for s, f in zip(streams, flips)], flips


def _againstConsensus(streams):
    """fraction of streamlines oriented against the bundle's majority direction"""
    resampled = np.asarray(set_number_of_points(Streamlines(streams), 30))
    ref = resampled.mean(axis=0)
    for _ in range(3):
        dSame = np.mean(np.linalg.norm(resampled - ref, axis=2), axis=1)
        dFlip = np.mean(np.linalg.norm(resampled[:, ::-1] - ref, axis=2), axis=1)
        aligned = np.where((dFlip < dSame)[:, None, None], resampled[:, ::-1], resampled)
        ref = aligned.mean(axis=0)
    dSame = np.mean(np.linalg.norm(resampled - ref, axis=2), axis=1)
    dFlip = np.mean(np.linalg.norm(resampled[:, ::-1] - ref, axis=2), axis=1)
    return np.mean(dFlip < dSame)


def _runOrientation(method, streams):
    out = method(Streamlines([np.asarray(s, dtype=np.float32) for s in streams]))
    return [np.asarray(s) for s in out]


METHODS = {
    'orientAllStreamlines': lambda s: streamlineTools.orientAllStreamlines(s),
    'orientTractUsingNeck_Robust': lambda s: streamlineTools.orientTractUsingNeck_Robust(s, surpressReport=True),
    'orientTractUsingCentroids': lambda s: streamlineTools.orientTractUsingCentroids(s, surpressReport=True),
}


@pytest.mark.parametrize('methodName', METHODS.keys())
def test_straightBundle_consistentAndInvariant(methodName):
    streams = _straightBundle()
    scrambled, _ = _randomlyReversed(streams)
    out = _runOrientation(METHODS[methodName], scrambled)
    assert _againstConsensus(out) == 0
    #invariance: same final first-node positions regardless of input direction
    outPlain = _runOrientation(METHODS[methodName], streams)
    assert all(np.allclose(a[0], b[0]) for a, b in zip(out, outPlain))


def test_arcBundle_neckMethodConsistent():
    """
    Regression: orientTractUsingNeck_Robust used to loop forever (flipped
    streamlines kept stale neck-window indexes, and every exception was
    swallowed and retried).
    """
    streams = _arcBundle()
    scrambled, _ = _randomlyReversed(streams)
    out = _runOrientation(METHODS['orientTractUsingNeck_Robust'], scrambled)
    assert _againstConsensus(out) == 0


@pytest.mark.xfail(reason='known limitation: orientAllStreamlines orients each streamline by its own '
                          'axis of greatest travel, which disagrees between streamlines of a curved tract',
                   strict=False)
def test_arcBundle_orientAllStreamlinesConsistent():
    streams = _arcBundle()
    scrambled, _ = _randomlyReversed(streams)
    out = _runOrientation(METHODS['orientAllStreamlines'], scrambled)
    assert _againstConsensus(out) == 0


def test_orientNonTraversingByEndpoints_fixesBranchesEndingAtNeck():
    """
    Claude note: streamlines that END at the neck (a branch leaving the tract
    at its bend) have no crossing direction; the helper orients them so their
    far end lands in the terminal cloud it is closest to.
    """
    bundle = _straightBundle(nStreams=40)            # runs -x -> +x, neck at x=0
    rng = np.random.default_rng(3)
    branches = []
    for _ in range(8):
        #starts at the neck, heads up and toward the +x (end B) terminals
        t = np.linspace(0, 1, 81)
        start = np.array([0.0, rng.normal(0, 1), rng.normal(0, 1)])
        end = np.array([38.0, rng.normal(0, 3), 30.0 + rng.normal(0, 2)])
        branches.append(start + t[:, None] * (end - start))
    #give the branches the WRONG orientation (far end first)
    streams = bundle + [b[::-1] for b in branches]
    sl = Streamlines([np.asarray(s, dtype=np.float32) for s in streams])
    #neck index per streamline in its current orientation: middle for the
    #bundle, the node nearest x=0 for the branches (the last node, reversed)
    neck = np.array([len(s) // 2 for s in bundle] + [len(b) - 1 for b in branches])
    out = streamlineTools.orientNonTraversingByEndpoints(sl, neck, lookDistance=4, surpressReport=True)
    for i in range(len(bundle), len(streams)):
        #after re-orientation the branch starts at the neck (x~0) and ends at the +x terminals
        assert np.asarray(out[i])[0][0] < 5 and np.asarray(out[i])[-1][0] > 30
    #traversing streamlines untouched
    for i in range(len(bundle)):
        assert np.allclose(np.asarray(out[i]), bundle[i].astype(np.float32))


# Claude note: plotting orientation (visTools.orientStreamlinesForPlotting),
# which multiPlotsForTract now uses with orientMethod='neck' by default.
def test_orientStreamlinesForPlotting_neckDefaultIsConsistentOnArc():
    import wmaPyTools.visTools as visTools
    scrambled, _ = _randomlyReversed(_arcBundle())
    out = _runOrientation(visTools.orientStreamlinesForPlotting, scrambled)
    assert _againstConsensus(out) == 0


def test_orientStreamlinesForPlotting_fallsBackOnIrregularSpacing():
    import wmaPyTools.visTools as visTools
    #half the streamlines at 0.5 mm spacing, half at 1 mm: the neck method refuses these
    streams = _straightBundle(nStreams=20)
    streams = streams[:10] + [s[::2] for s in streams[10:]]
    scrambled, _ = _randomlyReversed(streams)
    with pytest.warns(UserWarning, match='falling back to orientAllStreamlines'):
        out = _runOrientation(visTools.orientStreamlinesForPlotting, scrambled)
    #the fallback still orients a straight bundle correctly
    assert _againstConsensus(out) == 0


def test_orientStreamlinesForPlotting_rejectsUnknownMethod():
    import wmaPyTools.visTools as visTools
    with pytest.raises(ValueError):
        visTools.orientStreamlinesForPlotting(Streamlines([np.zeros((5, 3))]), orientMethod='bogus')


def test_orientAndColormapStreams_singleTractIsConsistentOnArc():
    """
    Claude note: singleTract=True orients the whole tract at once (no per-
    cluster orientation that can disagree between clusters) and returns one
    RGB color row per node.
    """
    import wmaPyTools.visTools as visTools
    scrambled, _ = _randomlyReversed(_arcBundle())
    sl = Streamlines([np.asarray(s, dtype=np.float32) for s in scrambled])
    out, colors = visTools.orientAndColormapStreams(sl, singleTract=True)
    assert _againstConsensus([np.asarray(s) for s in out]) == 0
    assert all(np.asarray(c).shape[0] == len(s) for c, s in zip(colors, out))


def _wideFanningBundleHalfReversed(seed, nStreams=40):
    """
    Claude note: a wide, mildly fanning straight bundle along x (cross-section
    offsets ~4 mm, per-streamline tilts ~0.3 rad), every other streamline
    reversed, and each streamline's neck node jittered by up to 6 nodes.  This
    is the first-pass state in which the old orientation guide (ahead point and
    behind point from two separate argmins, i.e. two different streamlines)
    encoded the offset between two streamlines instead of the tract direction:
    on seeds 0-9 the old guide left 2-20 of the 40 streamlines backwards on 7
    seeds (measured 2026-11-11).

    Returns
    -------
    streams : list of (161, 3) arrays
    aheadNodes, behindNodes : lists of int
        the index pair passed to flipStreamstoAB_OrientOnce (4 nodes, i.e.
        2 mm, either side of each streamline's neck)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(-40, 40, 161)
    streams, aheadNodes, behindNodes = [], [], []
    for iStream in range(nStreams):
        # Claude note: cross-section offset (y, z) and a tilted unit direction
        offset = np.r_[0, rng.normal(0, 4, 2)]
        direction = np.r_[1, rng.normal(0, 0.3, 2)]
        direction = direction / np.linalg.norm(direction)
        stream = offset + t[:, None] * direction
        # Claude note: this streamline's neck node, jittered
        neckNode = 80 + int(rng.integers(-6, 7))
        if iStream % 2:
            # Claude note: reverse it, reflecting its neck node (k -> len-1-k)
            stream = stream[::-1].copy()
            neckNode = 160 - neckNode
        streams.append(stream)
        aheadNodes.append(neckNode - 4)
        behindNodes.append(neckNode + 4)
    return streams, aheadNodes, behindNodes


@pytest.mark.parametrize('seed', range(10))
def test_flipStreamstoAB_OrientOnce_principalAxisGuideOnWideFan(seed):
    """
    Claude note: regression test for the orientation guide.  One pass of
    flipStreamstoAB_OrientOnce must leave every streamline of a wide,
    half-reversed bundle running the same way (positive x, the dimension of
    greatest traversal), and the returned neck indexes must still point at the
    same physical nodes.
    """
    streams, aheadNodes, behindNodes = _wideFanningBundleHalfReversed(seed)
    # Claude note: the physical neck-window points before orientation
    pointsBefore = [{tuple(s[a]), tuple(s[b])} for s, a, b in zip(streams, aheadNodes, behindNodes)]
    out, aheadOut, behindOut, flipCount = streamlineTools.flipStreamstoAB_OrientOnce(
        streams, list(aheadNodes), list(behindNodes), surpressReport=True)
    # Claude note: every streamline now runs towards +x
    runsPositive = np.array([s[-1, 0] > s[0, 0] for s in out])
    assert runsPositive.all(), f'{np.sum(~runsPositive)} of {len(out)} streamlines left backwards'
    # Claude note: exactly the reversed half was flipped
    assert flipCount == len(out) // 2
    # Claude note: the reflected indexes still pick out the same two points
    pointsAfter = [{tuple(s[a]), tuple(s[b])} for s, a, b in zip(out, aheadOut, behindOut)]
    assert pointsAfter == pointsBefore


@pytest.mark.parametrize('functionName', ['orientTractUsingNeck', 'orientTractUsingNeck_multi'])
def test_orientTractUsingNeck_runsOnFlatNeckVector(functionName):
    """
    Claude note: regression test for the IndexError ("invalid index to scalar
    variable") that orientTractUsingNeck raised on every call, by indexing
    findTractNeckNode's flat output as a column (neckNodes[i][0]); it also
    broke orientTractUsingNeck_multi.  This checks only that both run and
    return every streamline: the non-robust method's orientation quality is a
    separate, known issue (position-based flip decision), so use
    orientTractUsingNeck_Robust for orientation.
    """
    streams = Streamlines(_randomlyReversed(_straightBundle())[0])
    np.random.seed(0)
    out = getattr(streamlineTools, functionName)(streams)
    assert len(out) == len(streams)



# Claude note: tests for orientTractUsingCentroids and measureOrientationLocalReversal (2026-11-12)

def _truncatedUBundle(nStreams=200, seed=0, windowFraction=0.6):
    """
    Claude note: a callosum-like U (a half circle of radius 40 mm in the x-y
    plane, 0.5 mm steps) where every streamline is a window covering only
    windowFraction of the U, starting at a random offset; the 125 mm
    truncation of the TractSeg TOM tracts produces this.  This was the
    failure case of comparing sub-bundle centroids at fixed fractions of
    their length (truncated sub-bundles sit offset along the U).
    """
    rng = np.random.default_rng(seed)
    fullNodes = int(round(40 * np.pi / 0.5)) + 1
    windowNodes = int(windowFraction * fullNodes)
    streams = []
    for _ in range(nStreams):
        radius = 40 + rng.normal(0, 2)
        z = rng.normal(0, 2)
        theta = np.linspace(np.pi, 0, fullNodes)
        start = int(rng.integers(0, fullNodes - windowNodes + 1))
        window = theta[start:start + windowNodes]
        streams.append(np.column_stack([radius * np.cos(window), radius * np.sin(window), np.full_like(window, z)]))
    return streams


def test_orientTractUsingCentroids_arcBundleConsistent():
    streams = _arcBundle(nStreams=120)
    scrambled, _ = _randomlyReversed(streams)
    out = _runOrientation(METHODS['orientTractUsingCentroids'], scrambled)
    assert _againstConsensus(out) == 0
    assert streamlineTools.measureOrientationLocalReversal(out)['streamPct'] == 0


@pytest.mark.parametrize('seed', range(5))
def test_orientTractUsingCentroids_truncatedUBundleConsistent(seed):
    """
    Claude note: offset, truncated sub-bundles along a U must still come out
    consistently oriented (all running the same way around the U).
    """
    streams = _truncatedUBundle(seed=seed)
    scrambled, _ = _randomlyReversed(streams, seed=seed + 1)
    out = _runOrientation(METHODS['orientTractUsingCentroids'], scrambled)
    # Claude note: around the U (theta from pi to 0), a consistently oriented
    # streamline's angle changes monotonically in one direction for all
    angleChange = np.array([np.unwrap(np.arctan2(s[:, 1], s[:, 0]))[-1] - np.unwrap(np.arctan2(s[:, 1], s[:, 0]))[0] for s in out])
    assert np.all(angleChange > 0) or np.all(angleChange < 0)


def test_orientTractUsingCentroids_isDeterministic():
    streams = _arcBundle(nStreams=120)
    scrambled, _ = _randomlyReversed(streams)
    outA = _runOrientation(METHODS['orientTractUsingCentroids'], scrambled)
    outB = _runOrientation(METHODS['orientTractUsingCentroids'], scrambled)
    assert all(np.array_equal(a, b) for a, b in zip(outA, outB))


def test_orientTractUsingCentroids_signRule():
    """
    Claude note: the tract runs positive along its dimension of greatest
    average traversal (x, for the straight bundle).
    """
    streams = _straightBundle()
    scrambled, _ = _randomlyReversed(streams)
    out = _runOrientation(METHODS['orientTractUsingCentroids'], scrambled)
    assert all(s[-1, 0] > s[0, 0] for s in out)


def test_orientTractUsingCentroids_irregularNodeSpacing():
    """
    Claude note: unlike orientTractUsingNeck_Robust, which raises when node
    spacing varies by more than 10%, this method works with any spacing.
    """
    rng = np.random.default_rng(3)
    streams = []
    for s in _straightBundle():
        # Claude note: keep a random, irregular subset of nodes (always both ends)
        keep = np.sort(np.unique(np.r_[0, rng.choice(len(s), len(s) // 3, replace=False), len(s) - 1]))
        streams.append(s[keep])
    scrambled, _ = _randomlyReversed(streams)
    out = _runOrientation(METHODS['orientTractUsingCentroids'], scrambled)
    assert all(s[-1, 0] > s[0, 0] for s in out)


@pytest.mark.parametrize('nStreams', [0, 1, 2, 5])
def test_orientTractUsingCentroids_tinyInputs(nStreams):
    streams = _straightBundle(nStreams=max(nStreams, 1))[:nStreams]
    scrambled = [s[::-1] for s in streams]
    out = streamlineTools.orientTractUsingCentroids(Streamlines([np.asarray(s, dtype=np.float32) for s in scrambled]), surpressReport=True)
    assert len(out) == nStreams
    if nStreams >= 2:
        assert all(np.asarray(s)[-1, 0] > np.asarray(s)[0, 0] for s in out)


def test_quickbundlesClusters_seedIsReproducible():
    streams = Streamlines([np.asarray(s, dtype=np.float32) for s in _arcBundle(nStreams=120)])
    clustersA = streamlineTools.quickbundlesClusters(streams, nb_pts=50, seed=4)
    clustersB = streamlineTools.quickbundlesClusters(streams, nb_pts=50, seed=4)
    assert [sorted(c.indices) for c in clustersA] == [sorted(c.indices) for c in clustersB]


@pytest.mark.parametrize('reversedFraction', [0.0, 0.1, 0.3])
def test_measureOrientationLocalReversal_tracksKnownReversal(reversedFraction):
    """
    Claude note: the measure must recover the fraction of a straight bundle
    that was reversed (the smaller of the two groups), within 3 points.
    """
    streams = _straightBundle(nStreams=200)
    rng = np.random.default_rng(0)
    flips = rng.random(len(streams)) < reversedFraction
    mixed = [s[::-1] if f else s for s, f in zip(streams, flips)]
    trueMinority = 100 * min(flips.mean(), 1 - flips.mean())
    measured = streamlineTools.measureOrientationLocalReversal(mixed)['streamPct']
    assert abs(measured - trueMinority) <= 3
    flags = streamlineTools.measureOrientationLocalReversal(mixed, returnFlags=True)
    assert flags.shape == (len(streams),)


def test_measureOrientationEndToEnd_seesSeparatedReversedLayer():
    """
    Claude note: two parallel layers of a straight bundle, 15 mm apart, the
    second (one third of the streamlines) entirely reversed.  The local
    measure cannot see it (no near-parallel neighbours across the gap); the
    end-to-end measure must.  (A separated version of the SLF_III_left
    case, where the reversed layer was interleaved but its members' neighbours
    were mostly layer-mates.)
    """
    upper = _straightBundle(nStreams=100, seed=0)
    lower = [s + np.array([0, 0, -15.0]) for s in _straightBundle(nStreams=50, seed=1)]
    mixed = upper + [s[::-1] for s in lower]
    assert streamlineTools.measureOrientationLocalReversal(mixed)['streamPct'] == 0
    endToEnd = streamlineTools.measureOrientationEndToEnd(mixed)
    assert abs(endToEnd['oppositePct'] - 100 * 50 / 150) < 1
    assert endToEnd['nIncluded'] == 150


def test_measureOrientationEndToEnd_consistentBundleIsZero():
    out = _runOrientation(METHODS['orientTractUsingCentroids'], _randomlyReversed(_straightBundle())[0])
    assert streamlineTools.measureOrientationEndToEnd(out)['oppositePct'] == 0


def test_orientStreamlinesForPlotting_centroidOption():
    import wmaPyTools.visTools as visTools
    scrambled, _ = _randomlyReversed(_arcBundle(nStreams=120))
    out = visTools.orientStreamlinesForPlotting(Streamlines([np.asarray(s, dtype=np.float32) for s in scrambled]), orientMethod='centroid')
    assert _againstConsensus([np.asarray(s) for s in out]) == 0


def _hairpinBridgedLayers(seed):
    """
    Claude note: two parallel layers along x (5 mm apart) plus hairpin
    streamlines that run out along the upper layer, turn, and come back along
    the lower one.  A hairpin's arms touch the two layers in opposite
    directions, so it passes contradictory direction evidence between them;
    without excluding folded sub-bundles, orientTractUsingCentroids reversed
    a whole layer on 3 of seeds 0-9 (the HCP 100206 SLF_III_left failure).
    Returns the scrambled streamlines and a label per streamline (0, 1:
    layers; 2: hairpins).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 80, 161)
    streams, label = [], []
    for zLayer, labelValue in [(0.0, 0), (-5.0, 1)]:
        for _ in range(80):
            y, z = rng.normal(0, 1.5), zLayer + rng.normal(0, 0.5)
            streams.append(np.column_stack([x, np.full_like(x, y), np.full_like(x, z)]))
            label.append(labelValue)
    for _ in range(30):
        y = rng.normal(0, 1.5)
        turnAt = rng.uniform(30, 70)
        out = np.linspace(turnAt - 40, turnAt, 81)
        theta = np.linspace(np.pi / 2, -np.pi / 2, 21)
        turn = np.column_stack([turnAt + 2.5 * np.cos(theta), np.full_like(theta, y), -2.5 + 2.5 * np.sin(theta)])
        streams.append(np.vstack([np.column_stack([out, np.full_like(out, y), np.zeros_like(out)]), turn[1:-1],
                                  np.column_stack([out[::-1], np.full_like(out, y), np.full_like(out, -5.0)])]))
        label.append(2)
    flips = rng.random(len(streams)) < 0.5
    return [s[::-1] if f else s for s, f in zip(streams, flips)], np.array(label)


@pytest.mark.parametrize('seed', range(10))
def test_orientTractUsingCentroids_hairpinsDoNotReverseALayer(seed):
    streams, label = _hairpinBridgedLayers(seed)
    out = _runOrientation(METHODS['orientTractUsingCentroids'], streams)
    dx = np.array([s[-1, 0] - s[0, 0] for s in out])[label < 2]
    assert np.all(dx > 0) or np.all(dx < 0)
