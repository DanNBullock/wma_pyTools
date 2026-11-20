# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Regression tests for the genUtils (config / argument handling) and
analysisTools defects catalogued in the code assessment (items A1, A3-A6).
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

import wmaPyTools.genUtils as genUtils
import wmaPyTools.analysisTools as analysisTools


# ---------------------------------------------------------------- genUtils

def test_searchAndLoad_configJSON_fromOtherDirectory(tmp_path, monkeypatch):
    """A4: used to open the literal relative 'config.json' and then discard it."""
    (tmp_path / 'config.json').write_text(json.dumps({'tck': 'a.tck', 'n': 3}))
    #make sure we are NOT inside the config's directory
    monkeypatch.chdir(tmp_path.parent)
    assert genUtils.searchAndLoad_configJSON(str(tmp_path)) == {'tck': 'a.tck', 'n': 3}
    assert genUtils.searchAndLoad_configJSON(str(tmp_path / 'config.json')) == {'tck': 'a.tck', 'n': 3}


def test_searchAndLoad_configJSON_missing(tmp_path):
    assert genUtils.searchAndLoad_configJSON(str(tmp_path)) == []


def test_cmdORconfig_input_configOnly(tmp_path, monkeypatch):
    """A5: the config branch referenced nonPositionalArgs before assignment."""
    (tmp_path / 'config.json').write_text(json.dumps({'tck': 'a.tck'}))
    monkeypatch.setattr(sys, 'argv', ['script.py', str(tmp_path)])
    args, kwargs = genUtils.cmdORconfig_input()
    assert args == [] and kwargs == {'tck': 'a.tck'}


def test_cmdORconfig_input_noArgsUsesCwd(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'x': 1}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['script.py'])
    assert genUtils.cmdORconfig_input() == ([], {'x': 1})


def test_parseInputSysArgs_malformedWarnsNotCrashes(monkeypatch):
    """A3: warnings.warning does not exist; the warn branch crashed."""
    monkeypatch.setattr(sys, 'argv', ['script.py', 'pos1', 'key=', 'value'])
    with pytest.warns(UserWarning):
        args, kwargs = genUtils.parseInputSysArgs()
    assert kwargs['key'] == 'value'


def test_parcJSON_to_LUT_pathAndList(tmp_path):
    labels = [{'voxel_value': 2, 'name': 'L_WM'}, {'voxel_value': 41, 'name': 'R_WM'}]
    labelPath = tmp_path / 'label.json'
    labelPath.write_text(json.dumps(labels))
    fromPath = genUtils.parcJSON_to_LUT(str(labelPath))
    fromList = genUtils.parcJSON_to_LUT(labels)
    for iTable in [fromPath, fromList]:
        assert list(iTable.columns) == ['labelNumber', 'labelNames']
        assert iTable['labelNumber'].tolist() == [2, 41]


# ----------------------------------------------------------- analysisTools

def test_streamGeomQuantifications_straightLine():
    """A1/A6 + port drift: table used to be returned empty."""
    straight = np.column_stack([np.arange(0, 11.0), np.zeros(11), np.zeros(11)])
    table = analysisTools.streamGeomQuantifications([straight])
    assert len(table) == 1
    row = table.iloc[0]
    #endpoint-to-endpoint uses the true first node (0), not node 1
    assert row['length'] == pytest.approx(10)
    assert row['fullDisp'] == pytest.approx(10)
    assert row['efficiencyRat'] == pytest.approx(1)
    assert row['asymRat'] == pytest.approx(0)
    assert row['bioPriorCost'] == pytest.approx(1)


def test_streamGeomQuantifications_asymmetricAndShort():
    #first half straight, second half doubles back on itself: asymmetric
    firstHalf = np.column_stack([np.arange(0, 6.0), np.zeros(6), np.zeros(6)])
    secondHalf = np.column_stack([np.full(5, 5.0), np.arange(1, 6.0), np.zeros(5)])
    secondHalf = np.vstack([secondHalf, secondHalf[::-1][1:]])
    hooked = np.vstack([firstHalf, secondHalf])
    tooShort = np.zeros((3, 3))
    table = analysisTools.streamGeomQuantifications([hooked, tooShort])
    assert table.iloc[0]['asymRat'] > 0.1
    assert np.isfinite(table.iloc[0]['bioPriorCost'])
    #fewer than 5 nodes -> NaN, as in ConnectomeTestQ_v2.m
    assert table.iloc[1].isna().all()


# Claude note: tests for genUtils.StageTimer / collectStageTimings /
# setTimingVerbose (added 2026-11).

def test_stageTimer_recordsAndPrints(capsys):
    import time
    import wmaPyTools.genUtils as genUtils
    with genUtils.collectStageTimings() as outer:
        with genUtils.collectStageTimings() as inner:
            timer = genUtils.StageTimer('sleepStage')
            time.sleep(0.02)
            seconds = timer.finish(lambda s: 'slept ' + str(s))
        genUtils.StageTimer('second').finish()
    #Claude note: wall time is measured (sleep uses no CPU)
    assert seconds >= 0.015 and timer.cpuSeconds < seconds
    assert [r['stage'] for r in inner] == ['sleepStage']
    assert [r['stage'] for r in outer] == ['sleepStage', 'second']
    printed = capsys.readouterr().out
    assert 'slept ' in printed and 'second complete in' in printed
    #Claude note: blocks unregister themselves
    assert genUtils._ACTIVE_RECORDS == []


def test_setTimingVerbose_silences(capsys):
    import wmaPyTools.genUtils as genUtils
    previous = genUtils.setTimingVerbose(False)
    try:
        with genUtils.collectStageTimings() as records:
            genUtils.StageTimer('quiet').finish(lambda s: 'should not print')
        assert capsys.readouterr().out == ''
        assert len(records) == 1
    finally:
        genUtils.setTimingVerbose(previous)
    assert genUtils._TIMING_VERBOSE is True


def test_libraryStagesAreCollected(capsys):
    """
    Claude note: the library's own timed stages report to
    collectStageTimings (here a segmentation call on a tiny tractogram).
    """
    import numpy as np
    import nibabel as nib
    from dipy.tracking.streamline import Streamlines
    import wmaPyTools.genUtils as genUtils
    import wmaPyTools.segmentationTools as segmentationTools
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[4:6, 4:6, 4:6] = 1
    roi = nib.Nifti1Image(data, np.eye(4))
    streams = Streamlines([np.array([[0., 5, 5], [9, 5, 5]]), np.array([[0., 0, 0], [1, 1, 1]])])
    with genUtils.collectStageTimings() as records:
        segmentationTools.applyNiftiCriteriaToTract_DIPY_Test(streams, roi, True, 'any')
    assert 'applyNiftiCriteriaToTract_DIPY_Test' in [r['stage'] for r in records]
    assert 'met INCLUSION criterion' in capsys.readouterr().out
