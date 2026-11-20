# -*- coding: utf-8 -*-
"""
Claude note: regression test for visTools.colorGradientForStreams on
streamlines whose neck lies near their end (e.g. arcuate branches that end at
the bend).  The ahead-side white blend used to index one node past the end.
"""
import numpy as np
from dipy.tracking.streamline import Streamlines

import wmaPyTools.visTools as visTools


def test_colorGradientForStreams_neckNearStreamlineEnd():
    rng = np.random.default_rng(0)
    streams = []
    #a straight bundle along x at 0.5 mm spacing, pinched at x=0 (the neck)
    for _ in range(40):
        x = np.linspace(-40, 40, 161)
        pinch = 1 - 0.8 * np.exp(-x ** 2 / 200)
        streams.append(np.column_stack([x, np.full_like(x, rng.normal(0, 3)) * pinch,
                                        np.full_like(x, rng.normal(0, 3)) * pinch]))
    #branches that run in from -x and stop 0..12 nodes past the neck, so the
    #neck window ends at every offset from the streamline's last node
    for k in range(13):
        x = np.linspace(-40, k * 0.5, 81 + k)
        streams.append(np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]))
    sl = Streamlines([s.astype(np.float32) for s in streams])
    colors = visTools.colorGradientForStreams(sl)
    assert len(colors) == len(sl)
    assert all(len(c) == len(s) for c, s in zip(colors, sl))


def test_neckAnchoredArcPosition_basics():
    #uneven spacing on purpose: position follows arc length, not node count
    x = np.r_[np.linspace(0, 10, 11), np.linspace(10.5, 40, 4)]
    s = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    neck = 10                                   # at x=10
    pos = visTools.neckAnchoredArcPosition(s, neck)
    assert pos[0] == 0 and pos[-1] == 1 and pos[neck] == 0.5
    assert np.all(np.diff(pos) >= 0)
    assert np.isclose(pos[5], 0.25)             # halfway to the neck by length
    #neck at either end: no division by zero, range stays in [0,1]
    for n in (0, len(s) - 1):
        p = visTools.neckAnchoredArcPosition(s, n)
        assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1


def test_colorGradientForStreams_whiteSitsAtNeck():
    """
    Claude note: with bodyAnchor='neck' the palest seismic node of each
    streamline is at its neck, even when the neck is far from the streamline's
    halfway node; with 'index' (old behavior) it is at the halfway node.
    """
    import wmaPyTools.streamlineTools as streamlineTools
    rng = np.random.default_rng(1)
    streams = []
    for _ in range(30):
        #neck pinch at x=0, but streamlines run from -20 to +60: halfway is x=20
        x = np.arange(-20, 60.01, 0.5)
        pinch = 1 - 0.8 * np.exp(-x ** 2 / 200)
        streams.append(np.column_stack([x, np.full_like(x, rng.normal(0, 3)) * pinch,
                                        np.full_like(x, rng.normal(0, 3)) * pinch]))
    sl = Streamlines([s.astype(np.float32) for s in streams])
    neck = np.asarray(streamlineTools.findTractNeckNode(sl)).ravel()
    for anchor, expectX in (('neck', 0.0), ('index', 20.0)):
        #plain seismic body only: no neck/endpoint overlays to confuse "palest"
        colors = visTools.colorGradientForStreams(sl, bodyAnchor=anchor)
        for s, c, n in zip(sl, colors, neck):
            c = np.asarray(c)[:, :3]
            if anchor == 'neck':
                #the halfway node (x=20, 20 mm past the neck) is no longer white,
                #and paleness falls off monotonically away from the neck window
                half = np.argmin(np.abs(np.asarray(s)[:, 0] - 20))
                assert c[half].min() < 0.5
                assert c[n + 12].min() > c[n + 40].min() and c[n - 12].min() > c[n - 30].min()
            else:
                #old behavior: a second white patch at the halfway node, away
                #from the (also pale) neck overlay
                half = np.argmin(np.abs(np.asarray(s)[:, 0] - expectX))
                assert c[half].min() > 0.9


def test_bandedStreamCmap_bodyAnchoredAtNeck():
    """
    Claude note: the banded path (bandedStreamCmap, used by
    orientAndColormapStreams / makePrettyStreamsPlot) gets the same neck
    anchoring as colorGradientForStreams.
    """
    x = np.arange(-20, 60.01, 0.5)                  # neck at x=0, halfway at x=20
    s = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    neck = int(np.argmin(np.abs(x)))
    half = int(np.argmin(np.abs(x - 20)))
    anchored = np.asarray(visTools.bandedStreamCmap(s, neck, bodyAnchor='neck'))[:, :3]
    old = np.asarray(visTools.bandedStreamCmap(s, neck, bodyAnchor='index'))[:, :3]
    assert anchored.shape == old.shape == (len(s), 3)
    assert np.all(np.isfinite(anchored))
    #old: a white patch well away from the neck band (about the body's
    #halfway node; the neck band and end caps are excluded from the body, which
    #shifts it a few mm); anchored: well into the red side there
    outsideNeckBand = np.abs(x) > 10
    oldPalest = np.argmax(np.where(outsideNeckBand, old.min(axis=1), -1))
    assert old[oldPalest].min() > 0.9 and 15 < x[oldPalest] < 35
    assert anchored[half].min() < 0.5 and anchored[oldPalest].min() < 0.5
    #anchored: just outside the 10 mm neck band, still pale on both sides
    assert anchored[neck + 14].min() > anchored[neck + 60].min()
    assert anchored[neck - 14].min() > anchored[neck - 34].min()
    #1-element neck arrays (as findTractNeckNode can return) are accepted
    np.testing.assert_allclose(np.asarray(visTools.bandedStreamCmap(s, np.array([neck])))[:, :3], anchored)


def test_neckBandPaintedInNodeOrder():
    """
    Claude note: regression.  The neck window was built from a Python set,
    whose iteration order is not numeric across hash-table boundaries (a
    40-node window spanning node 512 iterated as 512..531, 492..511), so the
    neck colormap was painted out of order on ~20-40% of real streamlines.
    Here the neck is at node 512; the window's colors must be the neck
    colormap sampled in node order.
    """
    import matplotlib
    import wmaPyTools.streamlineTools as streamlineTools
    rng = np.random.default_rng(0)
    x = np.arange(-64, 64, 0.125)                           # 1024 nodes, x=0 at node 512
    pinch = 1 - 0.8 * np.exp(-x ** 2 / 200)
    sl = Streamlines([np.column_stack([x, np.full_like(x, rng.normal(0, 3)) * pinch,
                                       np.full_like(x, rng.normal(0, 3)) * pinch]).astype(np.float32)
                      for _ in range(40)])
    neck = visTools.neckNodesForPlotting(sl)
    look = 20                                               # round(2.5 mm / 0.125 mm)
    i = int(np.argmin(np.abs(neck - 512)))
    n = int(neck[i])
    window = list(range(n - look, n + look))
    assert window[0] < 512 <= window[-1], 'test setup: window must span node 512'
    c = np.asarray(visTools.colorGradientForStreams(sl)[i])[:, :3]
    expected = matplotlib.colormaps['twilight'](np.linspace(0, 1, len(window)))[:, :3]
    np.testing.assert_allclose(c[window], expected, atol=1e-6)
