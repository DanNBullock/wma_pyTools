# -*- coding: utf-8 -*-
"""
Claude note: tests for removeStreamlineOutliersAtNeck's cut (mean +
cutStDev * the fitted lognormal's standard deviation).
"""

import numpy as np
import pytest
from dipy.tracking.streamline import Streamlines

import wmaPyTools.streamlineTools as streamlineTools


def _tightStraightBundle(spreadMM, nStreams=200, seed=0):
    """
    Claude note: straight streamlines along x (0.5 mm spacing) with Gaussian
    cross-section offsets of spreadMM, so the neck points' distances from
    their mean scale with spreadMM.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-30, 30, 121)
    streams = []
    for _ in range(nStreams):
        y, z = rng.normal(0, spreadMM, 2)
        streams.append(np.column_stack([x, np.full_like(x, y), np.full_like(x, z)]))
    return Streamlines(streams)


@pytest.mark.parametrize('spreadMM', [0.3, 3.0])
def test_removeStreamlineOutliersAtNeck_removesOnlyTheTail(spreadMM):
    """
    Claude note: regression test.  The old "standard deviation" was log(scale),
    negative for sub-mm distances, so a tight bundle (spreadMM=0.3) lost most
    of its streamlines.  With the lognormal SD, cutStDev=2 removes only a tail
    (the fitted lognormal puts ~4% beyond it), whatever the bundle's scale.
    """
    streams = _tightStraightBundle(spreadMM)
    kept = streamlineTools.removeStreamlineOutliersAtNeck(streams, 2)
    removedFraction = 1 - len(kept) / len(streams)
    assert 0 <= removedFraction < 0.15


def test_removeStreamlineOutliersAtNeck_isScaleInvariant():
    """
    Claude note: the same bundle in different units (here, 10x wider) must
    lose the same streamlines; the old log(scale) rule did not.
    """
    narrow = _tightStraightBundle(0.3)
    wide = _tightStraightBundle(3.0)
    keptNarrow = streamlineTools.removeStreamlineOutliersAtNeck(narrow, 2)
    keptWide = streamlineTools.removeStreamlineOutliersAtNeck(wide, 2)
    assert len(keptNarrow) == len(keptWide)
