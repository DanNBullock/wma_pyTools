# wma_pyTools test suite

Unlike `testsOrDemos/`, which holds benchmarks and sanity-check scripts, this directory holds **assertions**: `pytest` runs them and they pass or fail.

```bash
pip install -e ".[test]"
pytest                      # synthetic tests only, a few seconds
WMAPYTOOLS_TESTDATA=/path/to/sub-100206 pytest   # also run real-data tests (~1 min)
```

## Two tiers

- **Synthetic** (`test_roiTools.py`, `test_segmentationTools.py`, `test_streamlineTools.py`, `test_genUtils_analysisTools.py`): small in-memory images and streamlines, built in `conftest.py`. The grid uses **2mm voxels with a shifted origin** on purpose: several historical bugs (e.g. the `createSphere` radius) were invisible on 1mm isotropic data.
- **Real data** (`test_realData.py`, marker `realdata`): HCP subject 100206. Skipped unless `WMAPYTOOLS_TESTDATA` is set.

## Real-data layout

`WMAPYTOOLS_TESTDATA` should point at a directory containing:

| file | source used during development |
|---|---|
| `t1.nii.gz` | brainlife `neuro/anat/t1w`, 1.25mm ACPC (RAS) |
| `aparc.a2009s+aseg.nii.gz` | brainlife `neuro/freesurfer` v5 output, `label/` (1mm, **LIA**, conformed) |
| `track_subset.tck` | the first 100,000 streamlines of the 3M-streamline brainlife `neuro/track/tck` |
| `classification.mat` | brainlife `neuro/wmc` (tag `cleaned`), 61 tracts over the full 3M tractogram |

The parcellation and T1 are on **different voxel grids and orientations**. This is deliberate: the tests check that everything agrees in mm (RAS) space.

To make `track_subset.tck` from a full tractogram:

```python
import itertools, numpy as np, nibabel as nib
tck = nib.streamlines.load('track.tck', lazy_load=True)
streams = list(itertools.islice(tck.tractogram.streamlines, 100000))
nib.streamlines.save(nib.streamlines.Tractogram(streams, affine_to_rasmm=np.eye(4)), 'track_subset.tck')
```

## What is covered, and why

Function selection follows what downstream code actually calls. Every brainlife app below vendors this repo as a git submodule **pinned to a specific commit**, so none of them is affected by changes here until its submodule is bumped. The public names and signatures they use are the compatibility surface to preserve.

Apps/notebooks importing `wmaPyTools` (as of 2026-11): app-wma_pyTools-TractFigs, app-DIPY-connectome_and_WMC_from_parc, app-de-island_parcellation, app-tcks_to_tck_and_WMC, app-segment_with_ROIs, app-filterViaQuickbundles, app-SNR_Report, app-track_aLIC, app-createMNI_ROIS_and_warp2subj, app-NHP_HCP_preproc, app-ProbtrackX_connectome-nipype, accSeg, atlasTractSegment, homologyTranslation, interactiveWMSegmentation, WiMSE. (The last three, plus WiMSE, pin pre-refactor commits and use the old `WMA_pyFuncs` module, now in `failedCode/`.)

Most-called functions, by number of call sites across those repos:

| calls | function |
|---|---|
| 65 | `roiTools.multiROIrequestToMask` |
| 19 | `roiTools.planarROIFromAtlasLabelBorder` |
| 16 | `roiTools.planeAtMaskBorder` |
| 15 | `visTools.dipyPlotTract` |
| 15 | `roiTools.sliceROIwithPlane` |
| 14 | `segmentationTools.segmentTractMultiROI` |
| 12 | `roiTools.findROISintersection` |
| 11 | `visTools.multiTileDensity` |
| 8 | `streamlineTools.updateClassification`, `stubbornSaveTractogram` |

`visTools` is **not yet covered**: it needs a rendering backend (fury/VTK, xvfb when headless), and its outputs are images.
