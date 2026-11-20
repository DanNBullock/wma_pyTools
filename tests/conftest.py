# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 2026

@author: dan

Shared fixtures for the wma_pyTools test suite.

Two tiers of data are used here:

- **synthetic** fixtures, built in memory, small enough that the whole suite
  runs in seconds.  These deliberately use *non-1mm, non-origin* voxel grids
  (2mm voxels, shifted origin) because several historical bugs (e.g. the
  createSphere radius computation) were invisible on 1mm isotropic data.
- **realdata** fixtures, which point at an HCP subject (sub-100206) on local
  disk.  These are skipped unless the environment variable
  WMAPYTOOLS_TESTDATA points at a directory laid out as described in
  tests/README.md.
"""

import os
import subprocess

import numpy as np
import nibabel as nib
import pytest
from dipy.tracking.streamline import Streamlines

# ---------------------------------------------------------------------------
# keeping dev artifacts out of the repository
# ---------------------------------------------------------------------------
# Claude note: several library functions write files relative to the current
# working directory (e.g. inflateAtlasIntoWMandBG's de-island_report.csv).
# Running each test from its own temporary directory keeps such files out of
# the repo; the session-level check below fails the run if anything new
# shows up in the working tree anyway.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _runInTmpDir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _untrackedRepoFiles():
    try:
        out = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=all'],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {line[3:] for line in out.splitlines() if line.startswith('??')}


@pytest.fixture(scope='session', autouse=True)
def _noArtifactsLeftInRepo():
    before = _untrackedRepoFiles()
    yield
    after = _untrackedRepoFiles()
    if before is not None and after is not None:
        newFiles = sorted(after - before)
        assert not newFiles, 'tests left files in the repository: ' + str(newFiles)


# the synthetic image grid.  2mm voxels with the origin placed so that
# mm-space coordinate (0,0,0) falls at the center of the volume, which is
# loosely how an ACPC-aligned image is set up
SYNTH_SHAPE = (40, 48, 40)
SYNTH_VOXSIZE = 2.0


@pytest.fixture
def synthAffine():
    affine = np.diag([SYNTH_VOXSIZE, SYNTH_VOXSIZE, SYNTH_VOXSIZE, 1.0])
    #place mm (0,0,0) at the volume center
    affine[0:3, 3] = -np.asarray(SYNTH_SHAPE) * SYNTH_VOXSIZE / 2
    return affine


@pytest.fixture
def synthReference(synthAffine):
    """A blank float reference nifti with 2mm voxels."""
    return nib.Nifti1Image(np.zeros(SYNTH_SHAPE, dtype=np.float32), synthAffine)


@pytest.fixture
def synthAtlas(synthAffine):
    """
    A small integer-labeled atlas.  Labels (in mm space):
        10 : a left box  (x in [-30,-10))
        20 : a right box (x in [10,30))
        30 : a midline box, anterior (y in [10,30))
         0 : background everywhere else
    """
    atlasData = np.zeros(SYNTH_SHAPE, dtype=np.int32)
    #image-space indexes of mm coordinates along x, y
    xMM = np.arange(SYNTH_SHAPE[0]) * SYNTH_VOXSIZE + synthAffine[0, 3]
    yMM = np.arange(SYNTH_SHAPE[1]) * SYNTH_VOXSIZE + synthAffine[1, 3]
    zMM = np.arange(SYNTH_SHAPE[2]) * SYNTH_VOXSIZE + synthAffine[2, 3]
    zBool = np.abs(zMM) < 10
    yBool = np.abs(yMM) < 10
    atlasData[np.ix_((xMM >= -30) & (xMM < -10), yBool, zBool)] = 10
    atlasData[np.ix_((xMM >= 10) & (xMM < 30), yBool, zBool)] = 20
    atlasData[np.ix_(np.abs(xMM) < 6, (yMM >= 10) & (yMM < 30), zBool)] = 30
    return nib.Nifti1Image(atlasData, synthAffine)


@pytest.fixture
def synthStreamlines():
    """
    A tiny tractogram (mm space), 1mm node spacing:
        0-4  : left-right streamlines from x=-25 to x=25, at y in {-4..4}, z=0
        5-7  : anterior-posterior streamlines at x=0, from y=-20 to y=25
        8    : a short streamline entirely inside the left box
        9    : a right-to-left streamline (reverse orientation of 0-4)
    """
    streams = []
    for iY in [-4, -2, 0, 2, 4]:
        xs = np.arange(-25, 26, 1.0)
        streams.append(np.column_stack([xs, np.full(xs.shape, iY), np.zeros(xs.shape)]))
    for iZ in [-4, 0, 4]:
        ys = np.arange(-20, 26, 1.0)
        streams.append(np.column_stack([np.zeros(ys.shape), ys, np.full(ys.shape, iZ)]))
    xs = np.arange(-25, -14, 1.0)
    streams.append(np.column_stack([xs, np.zeros(xs.shape), np.zeros(xs.shape)]))
    xs = np.arange(25, -26, -1.0)
    streams.append(np.column_stack([xs, np.full(xs.shape, 1.0), np.zeros(xs.shape)]))
    return Streamlines([s.astype(np.float32) for s in streams])


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

def _realDataRoot():
    return os.environ.get('WMAPYTOOLS_TESTDATA')


@pytest.fixture(scope='session')
def realData():
    """
    Paths to the HCP sub-100206 test set.  Skips if not configured.
    See tests/README.md for the expected layout.
    """
    root = _realDataRoot()
    if root is None or not os.path.isdir(root):
        pytest.skip('WMAPYTOOLS_TESTDATA not set; skipping real-data test')
    paths = {
        't1': os.path.join(root, 't1.nii.gz'),
        'parc': os.path.join(root, 'aparc.a2009s+aseg.nii.gz'),
        'tck': os.path.join(root, 'track_subset.tck'),
        'classification': os.path.join(root, 'classification.mat'),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        pytest.skip('real test data incomplete, missing: ' + str(missing))
    return paths
