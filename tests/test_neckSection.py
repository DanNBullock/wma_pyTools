# -*- coding: utf-8 -*-
"""
Claude note: tests for streamlineTools.tractNeckSection /
findTractNeckNode(method='section') on synthetic bundles running along x.
"""
import numpy as np
import pytest
from dipy.tracking.streamline import Streamlines

import wmaPyTools.streamlineTools as streamlineTools


def _bundle(widthAt, nStreams=80, xStart=-40, xEnd=40, ends=None, seed=0):
    """streamlines along x at 0.5 mm steps; cross-section half-width widthAt(x);
    ends[i] optionally truncates streamline i at that x"""
    rng = np.random.default_rng(seed)
    streams = []
    for i in range(nStreams):
        stop = xEnd if ends is None else ends[i]
        x = np.arange(xStart, stop + 1e-9, 0.5)
        oy, oz = rng.uniform(-1, 1, 2)
        w = widthAt(x)
        streams.append(np.column_stack([x, oy * w, oz * w]).astype(np.float32))
    return Streamlines(streams)


def _neckX(streams, neckNodes):
    return np.median([np.asarray(s)[n][0] for s, n in zip(streams, neckNodes)])


def test_neckSection_findsPinch():
    sl = _bundle(lambda x: 8 - 6 * np.exp(-x ** 2 / 50))           # narrowest at x=0
    neckNodes, traversing, info = streamlineTools.tractNeckSection(sl)
    assert abs(_neckX(sl, neckNodes)) < 3
    assert traversing.all()


def test_neckSection_ignoresPoorlyCoveredTail():
    """
    CST-like: a waist at x=0 crossed by every streamline, and an even narrower
    tail (x>30) that only 60% of streamlines reach.  The neck must be the waist.
    """
    width = lambda x: np.where(x > 30, 1.0, 8 - 5 * np.exp(-x ** 2 / 50))
    ends = np.where(np.arange(80) < 48, 40, 25)                     # 40% stop at x=25
    sl = _bundle(width, ends=ends)
    neckNodes, traversing, info = streamlineTools.tractNeckSection(sl)
    assert abs(_neckX(sl, neckNodes)) < 3
    #same answer through the findTractNeckNode switch
    assert np.array_equal(streamlineTools.findTractNeckNode(sl, method='section'), neckNodes)


def test_neckSection_uniformStretchTakesItsMiddle():
    #flat-bottomed waist from x=-5 to x=+25: the neck is its middle, x~10
    width = lambda x: np.where((x >= -5) & (x <= 25), 3.0, 3.0 + 0.3 * np.minimum(np.abs(x + 5), np.abs(x - 25)))
    sl = _bundle(width)
    neckNodes, _, info = streamlineTools.tractNeckSection(sl)
    assert abs(_neckX(sl, neckNodes) - 10) < 4


def test_neckSection_flagsStreamlinesThatDoNotReachTheNeck():
    ends = np.where(np.arange(80) < 76, 40, -15)                    # 4 stop well short of x=0
    sl = _bundle(lambda x: 8 - 6 * np.exp(-x ** 2 / 50), ends=ends)
    _, traversing, _ = streamlineTools.tractNeckSection(sl)
    assert traversing[:76].all() and not traversing[76:].any()


def test_findTractNeckNode_rejectsUnknownMethod():
    with pytest.raises(ValueError):
        streamlineTools.findTractNeckNode(_bundle(lambda x: 3 + 0 * x, nStreams=10), method='bogus')


def test_neckSection_uniformTubeEndsAreNotNarrow():
    """Claude note: regression: zero-padded smoothing made a fully covered
    tube's end sections look narrowest, putting the neck at an end."""
    sl = _bundle(lambda x: 3.0 + 0 * x)
    neckNodes, _, _ = streamlineTools.tractNeckSection(sl)
    assert abs(_neckX(sl, neckNodes)) < 10
