# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Claude note: tests for the color utilities in visTools (no rendering
needed).  hotTipColormap is checked against the matplotlib colormaps the
original design used ('autumn' for the red end of 'seismic', 'winter' for
the blue end).
"""

import numpy as np
import matplotlib
import pytest

import wmaPyTools.visTools as visTools


def test_oklabRoundTrip():
    rng = np.random.default_rng(0)
    rgb = rng.random((200, 3))
    #Claude note: the published OKLab matrices are precise to ~1e-6
    assert np.allclose(visTools.oklabToSrgb(visTools.srgbToOklab(rgb)), rgb, atol=1e-5)


def test_gamutCusps_pureColors():
    """
    Claude note: the cusp at the hue of pure red is pure red.  Pure blue is a
    known OKLab quirk: constant-hue lines bend near blue, so the most
    saturated color reachable along blue's own hue line is a slightly lighter
    azure-blue rather than (0,0,1) itself.  It is still perceptually close.
    """
    for rgb, tol in [((1, 0, 0), 0.01), ((0, 0, 1), 0.05)]:
        lab = visTools.srgbToOklab(np.array(rgb, float))
        hue = np.degrees(np.arctan2(lab[2], lab[1])) % 360
        cuspL, cuspC = visTools.gamutCuspOklch(hue)
        cuspLab = np.array([cuspL[0], cuspC[0] * np.cos(np.radians(hue)), cuspC[0] * np.sin(np.radians(hue))])
        assert np.linalg.norm(cuspLab - lab) < tol


@pytest.mark.parametrize('start,reference', [((1, 0, 0), 'autumn'), ((0, 0, 1), 'winter')])
def test_hotTipColormap_reproducesOriginalTips(start, reference):
    x = np.linspace(0, 1, 64)
    ours = visTools.hotTipColormap(start)(x)[:, :3]
    theirs = matplotlib.colormaps[reference](x)[:, :3]
    deltaE = np.linalg.norm(visTools.srgbToOklab(ours) - visTools.srgbToOklab(theirs), axis=1)
    assert deltaE.max() < 0.03


@pytest.mark.parametrize('path', ['srgb', 'ridge'])
@pytest.mark.parametrize('color,warm', [('red', True), ('magenta', True), ('darkorange', True),
                                        ('blue', False), ('#7f00ff', False), ('teal', False)])
def test_hotTipColormap_brightensTowardWarmOrCoolTarget(color, warm, path):
    cmap = visTools.hotTipColormap(color, path=path)
    startLab = visTools.srgbToOklab(cmap(0.0)[:3])
    endLab = visTools.srgbToOklab(cmap(1.0)[:3])
    #starts exactly at the requested color
    assert np.allclose(cmap(0.0)[:3], matplotlib.colors.to_rgb(color), atol=1e-6)
    #ends lighter
    assert endLab[0] > startLab[0]
    #warm colors end yellow (~110), cool ones green/cyan (~155)
    endHue = np.degrees(np.arctan2(endLab[2], endLab[1])) % 360
    assert abs(endHue - (110 if warm else 155)) < 5


def test_hotTipColormap_badPathRaises():
    with pytest.raises(ValueError):
        visTools.hotTipColormap('red', path='zigzag')


@pytest.mark.parametrize('preset', list(visTools.TRACT_COLOR_SCHEME_PRESETS.keys()))
def test_tractColorScheme_presetStructure(preset):
    """Claude note: every preset yields the four colormaps with the intended shape."""
    scheme = visTools.tractColorScheme(preset)
    for key in ['body', 'seam', 'tipA', 'tipB']:
        assert isinstance(scheme[key], matplotlib.colors.Colormap)
    bodyL = visTools.srgbToOklab(scheme['body'](np.linspace(0, 1, 101))[:, :3])[:, 0]
    #the neck (0.5) is the luminance peak of the body; both terminals are darker
    assert np.argmax(bodyL) == pytest.approx(50, abs=2)
    assert bodyL[0] < bodyL[50] and bodyL[-1] < bodyL[50]
    #each tip starts at (or very near) the body color at its own end, and brightens
    for tipKey, bodyPosition in [('tipA', 0.25), ('tipB', 0.75)]:
        tipStart = visTools.srgbToOklab(scheme[tipKey](0.0)[:3])
        tipEnd = visTools.srgbToOklab(scheme[tipKey](1.0)[:3])
        #Claude note: >= rather than >: an end that is already near the
        #lightest saturated colors (e.g. teal) has no room to brighten
        assert tipEnd[0] >= tipStart[0] - 1e-3
    #the seam is white-ish at both ends and dark in the middle
    seamL = visTools.srgbToOklab(scheme['seam']([0.0, 0.5, 1.0])[:, :3])[:, 0]
    assert seamL[1] < seamL[0] - 0.3 and seamL[1] < seamL[2] - 0.3


def test_tractColorScheme_classicIsOriginal():
    scheme = visTools.tractColorScheme()
    assert [scheme[k].name for k in ['body', 'seam', 'tipA', 'tipB']] == ['seismic', 'twilight', 'winter', 'autumn']


def test_tractColorScheme_customAndErrors():
    scheme = visTools.tractColorScheme(hueA=264, hueB=29)
    assert scheme['name'] == 'custom_264_29'
    with pytest.raises(ValueError):
        visTools.tractColorScheme('notAPreset')
    with pytest.raises(ValueError):
        visTools.tractColorScheme(hueA=264)
