# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Tests for wmaPyTools.streamlineTools, focused on the WMC (white matter
classification) round trips and tractogram I/O that the brainlife apps
(app-tcks_to_tck_and_WMC, app-segment_with_ROIs, app-wma_pyTools-TractFigs,
...) depend on.  WMC format:
https://brainlife.io/datatype/5cc1d64c44947d8aea6b2d8b/readme
"""

import os

import numpy as np
import nibabel as nib
import pytest
from scipy.io import savemat

import wmaPyTools.streamlineTools as streamlineTools


def test_stubbornSaveTractogram_roundTrip(synthStreamlines, tmp_path):
    outPath = str(tmp_path / 'out.tck')
    streamlineTools.stubbornSaveTractogram(synthStreamlines, outPath)
    reloaded = nib.streamlines.load(outPath).streamlines
    assert len(reloaded) == len(synthStreamlines)
    #coordinates should survive the trip in RASMM (float32 tolerance)
    assert np.allclose(reloaded[0], synthStreamlines[0], atol=1e-3)


def test_updateClassification_newThenAppendThenOverwrite(synthStreamlines):
    nStreams = len(synthStreamlines)
    lrBool = np.zeros(nStreams, dtype=bool)
    lrBool[[0, 1, 2, 3, 4, 9]] = True
    wmc = streamlineTools.updateClassification(lrBool, 'leftRight')
    assert list(wmc['names']) == ['leftRight']
    assert np.array_equal(np.where(wmc['index'] == 1)[0], [0, 1, 2, 3, 4, 9])
    #append a second structure by index
    wmc = streamlineTools.updateClassification(np.array([5, 6, 7]), 'antPost', wmc)
    assert list(wmc['names']) == ['leftRight', 'antPost']
    assert np.array_equal(np.where(wmc['index'] == 2)[0], [5, 6, 7])
    #overwrite the first, with a warning
    with pytest.warns(UserWarning):
        wmc = streamlineTools.updateClassification(np.array([8, 9, 2]), 'leftRight', wmc)
    assert np.array_equal(np.where(wmc['index'] == 1)[0], [2, 8, 9])


def test_updateClassification_newFromIndexesRaises():
    with pytest.raises(ValueError):
        streamlineTools.updateClassification(np.array([3, 4, 5]), 'x')


def test_inputTcks_to_WMCandTCK_fromDirectory(synthStreamlines, tmp_path):
    #write two tracts out as separate .tck files
    streamlineTools.stubbornSaveTractogram(synthStreamlines[[0, 1, 2]], str(tmp_path / 'left_tractA.tck'))
    streamlineTools.stubbornSaveTractogram(synthStreamlines[[5, 6]], str(tmp_path / 'right_tractB.tck'))
    tractogram, wmc = streamlineTools.inputTcks_to_WMCandTCK(str(tmp_path))
    assert len(tractogram.streamlines) == 5
    assert len(wmc['index']) == 5
    assert set(wmc['names']) == {'left_tractA', 'right_tractB'}
    #every streamline should be classified into one of the two
    assert set(np.unique(wmc['index'])) == {1, 2}


def test_matWMC2dict_and_wmc2tracts(synthStreamlines, tmp_path):
    #build a .mat classification the way MATLAB wma_tools would
    names = np.empty((1, 2), dtype=object)
    names[0, 0] = 'tract one'
    names[0, 1] = 'tract_two'
    index = np.zeros((len(synthStreamlines), 1))
    index[[0, 1, 2]] = 1
    index[[5, 6]] = 2
    classification = {'names': names, 'index': index}
    matPath = str(tmp_path / 'classification.mat')
    savemat(matPath, {'classification': classification})
    wmc = streamlineTools.matWMC2dict(matPath)
    #spaces are converted to underscores
    assert list(wmc['names']) == ['tract_one', 'tract_two']
    assert np.array_equal(np.asarray(wmc['index']), index.ravel())

    tckPath = str(tmp_path / 'whole.tck')
    streamlineTools.stubbornSaveTractogram(synthStreamlines, tckPath)
    outDir = tmp_path / 'tracts'
    outDir.mkdir()
    streamlineTools.wmc2tracts(tckPath, matPath, str(outDir))
    assert len(nib.streamlines.load(str(outDir / 'tract_one.tck')).streamlines) == 3
    assert len(nib.streamlines.load(str(outDir / 'tract_two.tck')).streamlines) == 2


def test_downsampleToEndpoints(synthStreamlines):
    endpoints = streamlineTools.downsampleToEndpoints(synthStreamlines)
    assert len(endpoints) == len(synthStreamlines)
    assert np.allclose(endpoints[0][0], synthStreamlines[0][0])
    assert np.allclose(endpoints[0][-1], synthStreamlines[0][-1])


def test_orientAllStreamlines_flipsReversed(synthStreamlines):
    #only the left-right streamlines, one of which (the last) is reversed
    lrStreams = synthStreamlines[[0, 1, 2, 3, 4, 9]]
    oriented = streamlineTools.orientAllStreamlines(lrStreams)
    firstX = np.asarray([s[0][0] for s in oriented])
    #all should now start on the same side
    assert np.all(np.sign(firstX) == np.sign(firstX[0]))
