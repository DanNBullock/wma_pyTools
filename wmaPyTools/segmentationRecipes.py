# -*- coding: utf-8 -*-
"""
Claude note: white matter segmentation recipes, ported from the MATLAB
wma_tools repository (https://github.com/DanNBullock/wma_tools,
Segmentations/ and the helpers they call).

Each public ``segment*`` function corresponds to one MATLAB tract-family
function and returns a wmc-style classification dictionary,
``{'names': [str, ...], 'index': np.ndarray of int}``, where ``index[i]`` is
the 1-based position in ``names`` of the tract that streamline ``i`` belongs
to, or 0 for none (the brainlife wmc layout, as also produced by
``streamlineTools.updateClassification``).

The recipes expect a FreeSurfer aparc.a2009s+aseg parcellation (cortical
labels 11101-11175 left, 12101-12175 right) and streamlines in the same
subject (RAS mm) space.  As in the MATLAB code, anatomical directions
(anterior, superior, medial, ...) are read straight off the subject space
axes, so the data should be ACPC (or similarly) aligned.

Which MATLAB behaviours are reproduced, and where this differs from the
older general purpose functions in roiTools / segmentationTools:

- the MATLAB "inflated atlas" (bsc_inflateLabels) grows cortical and
  subcortical labels into white matter, and is ported here as
  inflateAtlasLabels_bsc.  It is a different operation from
  roiTools.inflateAtlasIntoWMandBG, which grows labels into background.
- MATLAB's bsc_modifyROI_v2 keeps the part of an ROI strictly beyond a
  plane, and its 'medial' side runs on across the midline; it is ported
  here as modifyROI_bsc, because roiTools.sliceROIwithPlane keeps the plane
  itself and stops 'medial' at x=0.
- streamline / ROI intersections are dipy near_roi (as in
  segmentationTools.segmentTractMultiROI), but with vistasoft's 0.87 mm
  tolerance rather than dipy's default 0.866 mm (see segmentByROIs).
- by default the atlas is first cleaned of islands (prepareAtlas_bsc), as
  the brainlife app-wmaSeg does before running the recipes.
- tract names follow the current naming; the optic radiation loops of
  older MATLAB output ('meyer', 'baum') are now 'Inferior_OR' and
  'Superior_OR' (see MATLAB_TRACT_RENAMES).
"""

import warnings

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from dipy.core.geometry import dist_to_corner

import wmaPyTools.roiTools
import wmaPyTools.segmentationTools

# Claude note: ---------------------------------------------------------------
# FreeSurfer label groups used by the MATLAB recipes (aparc.a2009s+aseg)
# ---------------------------------------------------------------------------

#Claude note: the hemisphere offsets of the a2009s cortical labels; the MATLAB
#code calls this "sidenum" (10000 + 1000*side, side 1 = left, 2 = right)
SIDE_NAMES = ('left', 'right')
SIDE_OFFSETS = (11000, 12000)

#Claude note: all cortical (grey matter) labels, both hemispheres
GREY_MATTER_LABELS = list(np.arange(101, 176) + 12000) + list(np.arange(101, 176) + 11000)
#Claude note: subcortical labels as listed in bsc_inflateLabels and the priors
SUBCORTICAL_LABELS = [10, 11, 12, 13, 17, 18, 19, 20, 26, 58, 27, 49, 50, 51, 52, 53, 54, 55, 56, 59]
#Claude note: cerebral white matter, left then right
WM_LABELS = [2, 41]
#Claude note: the labels bsc_inflateLabels grows other labels into: white
#matter, unknown cortex, and the de-islanding placeholder 999
INFLATION_TARGET_LABELS = [41, 2, 2000, 1000, 999]

#Claude note: tract names changed since the MATLAB recipes were written
#(name stems; prefixed by 'left' / 'right').  Meyer's and Baum's loops of the
#optic radiation are the inferior and superior optic radiation, as in
#Bullock et al. 2022, Cerebral Cortex (doi:10.1093/cercor/bhab500); the
#maintained wma_tools fork used by brainlife's app-wmaSeg renamed them the
#same way (2026-11).  Use renameMATLABTracts to compare with older output.
MATLAB_TRACT_RENAMES = {'meyer': 'Inferior_OR', 'baum': 'Superior_OR'}
#Claude note: name suffixes changed since the MATLAB code was written.
#bsc_streamlineCategoryPriors_v7 (and the v4 recipes written for it) spell
#the plausibility flag '_implausable'; the Python port spells it
#'_implausible' (Dan Bullock, 2026-11-14).
MATLAB_TRACT_SUFFIX_RENAMES = {'_implausable': '_implausible'}


# Claude note: ---------------------------------------------------------------
# classification structure helpers
# ---------------------------------------------------------------------------

def emptyClassification(nStreamlines):
    """
    Claude note: an empty wmc-style classification for nStreamlines
    streamlines.

    Parameters
    ----------
    nStreamlines : int
        The number of streamlines in the source tractogram.

    Returns
    -------
    classification : dict
        ``{'names': [], 'index': np.zeros(nStreamlines, int)}``
    """
    return {'names': [], 'index': np.zeros(int(nStreamlines), dtype=np.int64)}


def reconcileClassifications(baseClassification, addClassification):
    """
    Claude note: merge two classifications, as MATLAB's
    bsc_reconcileClassifications does.

    Names keep their order of first appearance (base names first).
    Streamlines assigned in addClassification take that assignment, even if
    baseClassification had assigned them to something else: the later
    classification wins.  A name present in both is merged into one entry.

    Parameters
    ----------
    baseClassification : dict
        wmc-style classification (``names``, ``index``).
    addClassification : dict
        wmc-style classification over the same streamlines.

    Returns
    -------
    classification : dict
        The merged classification (new object; inputs are not modified).
    """
    baseIndex = np.asarray(baseClassification['index'])
    addIndex = np.asarray(addClassification['index'])
    #Claude note: the two must describe the same tractogram
    if len(baseIndex) != len(addIndex):
        raise ValueError('reconcileClassifications: classifications of different lengths')
    #Claude note: union of the names, base names first, in order of appearance
    outNames = list(baseClassification['names'])
    for iName in addClassification['names']:
        if iName not in outNames:
            outNames.append(iName)
    #Claude note: translate each input's indexes into positions in outNames
    #(index 0, "unclassified", stays 0)
    baseMap = np.zeros(len(baseClassification['names']) + 1, dtype=np.int64)
    baseMap[1:] = [outNames.index(iName) + 1 for iName in baseClassification['names']]
    addMap = np.zeros(len(addClassification['names']) + 1, dtype=np.int64)
    addMap[1:] = [outNames.index(iName) + 1 for iName in addClassification['names']]
    outIndex = baseMap[baseIndex]
    #Claude note: the later classification overwrites where it assigns anything
    addAssigned = addIndex > 0
    outIndex[addAssigned] = addMap[addIndex[addAssigned]]
    return {'names': outNames, 'index': outIndex}


def renameMATLABTracts(classification):
    """
    Claude note: a copy of a classification with tract names from older
    MATLAB output (e.g. 'leftmeyer') changed to the current ones (e.g.
    'leftInferior_OR'), see MATLAB_TRACT_RENAMES, and old name suffixes
    changed to the current ones (e.g. 'left_frontal_to_temporal_implausable'
    to 'left_frontal_to_temporal_implausible'), see
    MATLAB_TRACT_SUFFIX_RENAMES.

    Parameters
    ----------
    classification : dict
        wmc-style classification (``names``, ``index``).

    Returns
    -------
    renamed : dict
        Same index, updated names.
    """
    renamedNames = []
    for iName in classification['names']:
        #Claude note: swap an old suffix first
        for iOldSuffix, iNewSuffix in MATLAB_TRACT_SUFFIX_RENAMES.items():
            if iName.endswith(iOldSuffix):
                iName = iName[:-len(iOldSuffix)] + iNewSuffix
                break
        #Claude note: swap the stem after a 'left' / 'right' prefix
        for iPrefix in ('left', 'right', ''):
            stem = iName[len(iPrefix):]
            if iName.startswith(iPrefix) and stem in MATLAB_TRACT_RENAMES:
                iName = iPrefix + MATLAB_TRACT_RENAMES[stem]
                break
        renamedNames.append(iName)
    return {'names': renamedNames, 'index': np.asarray(classification['index']).copy()}


def addTractToClassification(classification, name, *criteria):
    """
    Claude note: add a tract made of the streamlines meeting ALL of the
    given boolean criteria, as MATLAB's bsc_concatClassificationCriteria does
    (later tracts win over earlier ones, see reconcileClassifications).

    Parameters
    ----------
    classification : dict
        wmc-style classification to add to.
    name : str
        The tract name.
    *criteria : array-like of bool
        One or more boolean vectors over the streamlines; they are ANDed.

    Returns
    -------
    classification : dict
        The updated classification (a new object).
    """
    #Claude note: AND all criteria together
    tractBool = np.ones(len(classification['index']), dtype=bool)
    for iCriterion in criteria:
        tractBool &= np.asarray(iCriterion, dtype=bool).ravel()
    #Claude note: a one-tract classification, reconciled onto the input
    newClassification = {'names': [name], 'index': tractBool.astype(np.int64)}
    return reconcileClassifications(classification, newClassification)


def classificationBoolByName(classification, name):
    """
    Claude note: boolean vector of the streamlines classified as ``name``
    (all False if the name is absent), as MATLAB's bsc_extractStreamIndByName.

    Parameters
    ----------
    classification : dict
        wmc-style classification.
    name : str
        The tract / category name.

    Returns
    -------
    nameBool : np.ndarray of bool
    """
    index = np.asarray(classification['index'])
    if name not in classification['names']:
        return np.zeros(len(index), dtype=bool)
    return index == (classification['names'].index(name) + 1)


# Claude note: ---------------------------------------------------------------
# ROI helpers, MATLAB semantics
# ---------------------------------------------------------------------------

def _atlasData(atlas):
    """
    Claude note: the atlas's labels as a rounded int array (same rounding as
    roiTools.multiROIrequestToMask).
    """
    return np.round(np.asanyarray(atlas.dataobj)).astype(np.int64)


def roiFromAtlasLabels_bsc(atlas, labels, smoothKernel=1):
    """
    Claude note: mask of the given atlas labels, optionally grown, as
    MATLAB's bsc_roiFromAtlasNums(atlas, labels, smoothKernel).

    MATLAB grows the mask with ``~(smooth3(mask,'box',k)==0)``, i.e. every
    voxel with any mask voxel in the k x k x k box around it: a binary
    dilation by a k-cube.  k=1 (the most common call) leaves it unchanged.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        Integer label atlas.
    labels : int or list of int
        The labels to include.
    smoothKernel : int, optional
        The MATLAB box size k (odd).  The default is 1 (no growth).

    Returns
    -------
    roiNifti : nibabel.Nifti1Image
        0/1 mask on the atlas grid.
    """
    #Claude note: the union of the labels (warns about absent labels)
    roiNifti = wmaPyTools.roiTools.multiROIrequestToMask(atlas, np.atleast_1d(labels))
    maskData = np.asanyarray(roiNifti.dataobj) > 0
    #Claude note: the box smoothing, as a dilation by a k-cube.  A cube is
    #separable, so a maximum filter (three 1D passes) gives the same result
    #as binary_dilation with a k^3 structure, much faster for large k (MATLAB
    #recipes use k up to 27)
    if smoothKernel is not None and smoothKernel > 1:
        maskData = ndimage.maximum_filter(maskData, size=int(smoothKernel), mode='constant', cval=0)
    return nib.Nifti1Image(maskData.astype(np.uint8), atlas.affine)


def _maskNifti(maskData, reference):
    """
    Claude note: wrap a boolean array as a 0/1 nifti on reference's grid.
    """
    return nib.Nifti1Image(np.asarray(maskData, dtype=np.uint8), reference.affine)


def _maskVoxelCoords(maskNifti):
    """
    Claude note: subject space coordinates of a mask's non-zero voxel
    centres, (N,3).
    """
    voxelIndexes = np.array(np.nonzero(np.asanyarray(maskNifti.dataobj))).T
    return nib.affines.apply_affine(maskNifti.affine, voxelIndexes)


def roiMeanCoord(roiNifti):
    """
    Claude note: the mean subject space coordinate of an ROI's voxels; this
    is how the MATLAB code turns a (planar) ROI into a reference coordinate
    (e.g. in bsc_modifyROI_v2, bsc_applyEndpointCriteria).

    Parameters
    ----------
    roiNifti : nibabel.Nifti1Image
        A mask (typically a planar ROI).

    Returns
    -------
    meanCoord : np.ndarray, (3,)
    """
    coords = _maskVoxelCoords(roiNifti)
    if len(coords) == 0:
        raise ValueError('roiMeanCoord: empty ROI')
    return np.mean(coords, axis=0)


def modifyROI_bsc(roiNifti, reference, location):
    """
    Claude note: keep only the part of an ROI strictly beyond a reference
    coordinate, as MATLAB's bsc_modifyROI_v2(atlas, roi, refCoord, location).

    The reference is a coordinate, or an ROI (usually a plane) whose mean
    voxel coordinate is used.  Voxels exactly on the reference coordinate
    are dropped.  For 'medial' / 'lateral' the hemisphere is taken from the
    sign of the reference x coordinate (x<0 = left), and the kept part runs
    to the image edge: 'medial' of a left hemisphere plane includes the
    right hemisphere, as in MATLAB.  (roiTools.sliceROIwithPlane instead
    keeps the plane itself and stops 'medial' at x=0.)

    Parameters
    ----------
    roiNifti : nibabel.Nifti1Image
        The ROI (mask) to cut.
    reference : nibabel.Nifti1Image or array-like of 3 floats
        The reference ROI / plane, or a subject space coordinate.
    location : str
        'superior'/'top', 'inferior'/'bottom', 'anterior', 'posterior',
        'medial' or 'lateral': which side of the reference to keep.

    Returns
    -------
    roiOut : nibabel.Nifti1Image
        The cut ROI on roiNifti's grid (may be empty).
    """
    #Claude note: the reference coordinate
    if isinstance(reference, nib.Nifti1Image):
        refCoord = roiMeanCoord(reference)
    else:
        refCoord = np.asarray(reference, dtype=float)
    #Claude note: which axis, and which direction (+1 keeps larger values)
    leftFlag = refCoord[0] < 0
    locationDict = {'top': (2, 1), 'superior': (2, 1), 'bottom': (2, -1), 'inferior': (2, -1),
                    'anterior': (1, 1), 'posterior': (1, -1),
                    'medial': (0, 1 if leftFlag else -1), 'lateral': (0, -1 if leftFlag else 1)}
    if location.lower() not in locationDict:
        raise ValueError('modifyROI_bsc: location ' + str(location) + ' not understood')
    cutDim, valence = locationDict[location.lower()]
    #Claude note: subject space coordinate of each mask voxel along cutDim
    maskData = np.asanyarray(roiNifti.dataobj) > 0
    voxelIndexes = np.array(np.nonzero(maskData))
    voxelCoords = nib.affines.apply_affine(roiNifti.affine, voxelIndexes.T)
    #Claude note: strictly beyond the reference; the tolerance (far below a
    #voxel) keeps voxels lying on a reference plane out despite rounding
    tol = 1e-3 * np.min(np.asarray(roiNifti.header.get_zooms()[:3]))
    if valence > 0:
        keep = voxelCoords[:, cutDim] > refCoord[cutDim] + tol
    else:
        keep = voxelCoords[:, cutDim] < refCoord[cutDim] - tol
    #Claude note: write the kept voxels into a fresh mask
    outData = np.zeros(maskData.shape, dtype=np.uint8)
    keptIndexes = voxelIndexes[:, keep]
    outData[keptIndexes[0], keptIndexes[1], keptIndexes[2]] = 1
    return _maskNifti(outData, roiNifti)


def mergeROIs(*rois):
    """
    Claude note: union of masks on the same grid (MATLAB bsc_mergeROIs).

    Parameters
    ----------
    *rois : nibabel.Nifti1Image
        Masks on the same grid.

    Returns
    -------
    mergedNifti : nibabel.Nifti1Image
    """
    merged = np.zeros(rois[0].shape, dtype=bool)
    for iRoi in rois:
        merged |= np.asanyarray(iRoi.dataobj) > 0
    return _maskNifti(merged, rois[0])


def planeFromROI_bsc(roiOrLabels, location, atlas):
    """
    Claude note: a full-image planar ROI at an ROI's (or label set's)
    outermost voxel centres, as MATLAB's bsc_planeFromROI_v2.

    Parameters
    ----------
    roiOrLabels : nibabel.Nifti1Image, or int / list of int
        A mask, or labels of ``atlas`` to take the border of.
    location : str
        'superior', 'inferior', 'anterior', 'posterior', 'medial' or
        'lateral'.
    atlas : nibabel.Nifti1Image
        The atlas the labels refer to (ignored when a mask is given).

    Returns
    -------
    planeNifti : nibabel.Nifti1Image
    """
    #Claude note: MATLAB synonyms
    location = {'top': 'superior', 'bottom': 'inferior'}.get(location.lower(), location.lower())
    if isinstance(roiOrLabels, nib.Nifti1Image):
        return wmaPyTools.roiTools.planeAtMaskBorder(roiOrLabels, location)
    return wmaPyTools.roiTools.planarROIFromAtlasLabelBorder(atlas, np.atleast_1d(roiOrLabels), location)


# Claude note: ---------------------------------------------------------------
# atlas label voting (bsc_inflateLabels, bsc_atlasROINumsFromCoords_v3)
# ---------------------------------------------------------------------------

def _windowCounts(atlasData, center, radius, excludeLabels=None):
    """
    Claude note: the labels in a cubic window around a voxel and their
    counts, as computed in the MATLAB voting loops.

    Parameters
    ----------
    atlasData : np.ndarray of int
    center : array-like of 3 int
    radius : int
        The window half width.
    excludeLabels : array-like of int, optional
        Labels that cannot win (MATLAB's setdiff with the inflation targets).

    Returns
    -------
    labels : np.ndarray
    counts : np.ndarray
    """
    shape = np.asarray(atlasData.shape)
    center = np.asarray(center, dtype=np.intp)
    #Claude note: window bounds, clipped to the image (MATLAB would error
    #instead; this only matters within a few voxels of the edge)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius, shape - 1) + 1
    window = atlasData[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    #Claude note: candidate labels from the window
    candidates = np.unique(window)
    if excludeLabels is not None:
        candidates = candidates[np.isin(candidates, excludeLabels, invert=True)]
    windowLabels, windowCounts = np.unique(window, return_counts=True)
    counts = np.zeros(len(candidates), dtype=np.int64)
    present = np.isin(candidates, windowLabels)
    counts[present] = windowCounts[np.searchsorted(windowLabels, candidates[present])]
    return candidates, counts


def _voteLoop(atlasData, center, excludeLabels=None, maxRadius=50):
    """
    Claude note: the MATLAB per-voxel vote with tie-break: the most common
    label in the 3x3x3 window (minus excluded labels); while tied, grow the
    window by one voxel.

    Parameters
    ----------
    atlasData : np.ndarray of int
    center : array-like of 3 int
    excludeLabels : array-like of int, optional
        Labels that cannot win.
    maxRadius : int, optional
        Give up growing past this half width and take the smallest tied
        label (MATLAB would loop forever).  The default is 50.

    Returns
    -------
    label : int, or None if there are no candidate labels at all
    """
    #Claude note: first the 3x3x3 window
    labels, counts = _windowCounts(atlasData, center, 1, excludeLabels)
    if len(labels) == 0:
        return None
    radius = 1
    #Claude note: while tied, grow (MATLAB's loop starts again at radius 1,
    #which repeats the same tie, then grows)
    while np.sum(counts == np.max(counts)) > 1 and radius < maxRadius:
        radius += 1
        labels, counts = _windowCounts(atlasData, center, radius, excludeLabels)
        if len(labels) == 0:
            return None
    return int(labels[np.argmax(counts)])


def _vote(atlasData, centers, excludeLabels=None, chunkSize=20000):
    """
    Claude note: _voteLoop for many voxels at once.  The 3x3x3 vote is
    vectorised; only tied voxels go through the per-voxel loop.

    Parameters
    ----------
    atlasData : np.ndarray of int
    centers : np.ndarray, (N,3) int
        Voxel indexes (within the image).
    excludeLabels : array-like of int, optional

    Returns
    -------
    winners : np.ndarray of int, (N,)
        Winning labels; voxels with no candidate keep a value of -1.
    """
    centers = np.asarray(centers, dtype=np.intp).reshape(-1, 3)
    winners = np.full(len(centers), -1, dtype=np.int64)
    shape = np.asarray(atlasData.shape)
    #Claude note: the 27 offsets of the 3x3x3 window
    offsets = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij'), axis=-1).reshape(-1, 3)
    for iStart in range(0, len(centers), chunkSize):
        chunk = centers[iStart:iStart + chunkSize]
        #Claude note: neighbour labels; out-of-image neighbours are invalid
        neighbours = chunk[:, None, :] + offsets[None, :, :]
        valid = np.all((neighbours >= 0) & (neighbours < shape), axis=2)
        neighbours = np.clip(neighbours, 0, shape - 1)
        votes = atlasData[neighbours[:, :, 0], neighbours[:, :, 1], neighbours[:, :, 2]]
        if excludeLabels is not None:
            valid &= np.isin(votes, excludeLabels, invert=True)
        #Claude note: for each vote, how many valid votes share its label
        sameLabel = (votes[:, :, None] == votes[:, None, :]) & valid[:, None, :]
        voteCounts = np.where(valid, np.sum(sameLabel, axis=2), 0)
        maxCounts = np.max(voteCounts, axis=1)
        firstMax = np.argmax(voteCounts, axis=1)
        chunkWinners = votes[np.arange(len(votes)), firstMax].astype(np.int64)
        #Claude note: tied when a different label reaches the same count
        tied = np.any((voteCounts == maxCounts[:, None]) & (votes != chunkWinners[:, None]) & valid, axis=1)
        chunkWinners[maxCounts == 0] = -1
        #Claude note: ties go through the MATLAB loop
        for iTied in np.where(tied)[0]:
            loopWinner = _voteLoop(atlasData, chunk[iTied], excludeLabels)
            chunkWinners[iTied] = -1 if loopWinner is None else loopWinner
        winners[iStart:iStart + chunkSize] = chunkWinners
    return winners


def deislandLabels_bsc(atlas, maxIsleSize=5, replaceVal=999):
    """
    Claude note: replace small islands of each label, as MATLAB's
    fnDeislandLabels_v2(atlas, [], maxIsleSize, replaceVal).

    For each label > 0, its 26-connected components with fewer than
    maxIsleSize voxels (MATLAB bwareaopen) are set to replaceVal.  Voxels
    with labels <= 0 become 0.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        Integer label atlas.
    maxIsleSize : int, optional
        Components smaller than this are islands.  The default is 5.
    replaceVal : int, optional
        The value islands are set to.  The default is 999.

    Returns
    -------
    outAtlas : nibabel.Nifti1Image
    """
    atlasData = _atlasData(atlas)
    #Claude note: MATLAB starts from zeros and writes each label back
    outData = np.where(atlasData > 0, atlasData, 0)
    structure = np.ones((3, 3, 3), dtype=bool)
    #Claude note: one bounding box per label value, so each label's
    #components are found in a small crop rather than the whole volume
    positiveData = np.where(atlasData > 0, atlasData, 0)
    for iLabelMinus1, iSlice in enumerate(ndimage.find_objects(positiveData)):
        if iSlice is None:
            continue
        iLabel = iLabelMinus1 + 1
        labelCrop = positiveData[iSlice] == iLabel
        components, nComponents = ndimage.label(labelCrop, structure=structure)
        sizes = np.bincount(components.ravel())
        #Claude note: components (not the background, 0) below the size limit
        smallComponents = np.where(sizes < maxIsleSize)[0]
        smallComponents = smallComponents[smallComponents > 0]
        if len(smallComponents) > 0:
            outCrop = outData[iSlice]
            outCrop[np.isin(components, smallComponents)] = replaceVal
    return nib.Nifti1Image(outData.astype(np.int32), atlas.affine)


def prepareAtlas_bsc(atlas, maxIsleSize=5):
    """
    Claude note: remove label islands before running the recipes:
    components of each label smaller than maxIsleSize voxels are set to 0.

    This is the first step of brainlife's app-wmaSeg
    (``fnDeislandLabels_v3(atlas_raw, 5, 0)``, added there in 2020 by its
    maintainers, not part of wma_segMajTracks_BL_v2).  Every recipe then
    works on the cleaned atlas, so a stray voxel can't set a border plane.
    The recipes' own inflated atlas (inflateAtlasLabels_bsc) de-islands
    again, but to 999; after this step that finds nothing.  Repeating this
    step changes nothing either.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas.
    maxIsleSize : int, optional
        Components smaller than this are removed.  The default is 5.

    Returns
    -------
    cleanAtlas : nibabel.Nifti1Image
    """
    return deislandLabels_bsc(atlas, maxIsleSize, 0)


def inflateAtlasLabels_bsc(atlas, inflateIter=2):
    """
    Claude note: grow cortical and subcortical labels into white matter, as
    MATLAB's bsc_inflateLabels(atlas, inflateIter) (the "inflatedAtlas" used
    throughout the MATLAB recipes).

    1. de-island: components < 5 voxels of each label become 999
       (deislandLabels_bsc);
    2. inflateIter times: every voxel labelled white matter (2, 41), unknown
       cortex (1000, 2000) or 999 that touches (26-neighbourhood) a cortical
       or subcortical voxel takes the most common label among its 3x3x3
       neighbours, excluding those target labels (ties: grow the window).
       Voxels on the image edge are skipped.  All votes of an iteration use
       the labels from the start of that iteration;
    3. the 999 voxels left over take the same vote (against the labels from
       the start of the last iteration, as in MATLAB), or, with no candidate,
       the smallest label in their 3x3x3 window.

    MATLAB also smooths the grown mask and requires a 6/27 fraction, but a
    voxel touching a source voxel always has at least 8 of 27, so that test
    never removes anything and is omitted here.

    This differs from roiTools.inflateAtlasIntoWMandBG, which grows labels
    into background (0) and, optionally, inferred white matter.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        A FreeSurfer aparc.a2009s+aseg style atlas.
    inflateIter : int, optional
        The number of inflation iterations.  The default is 2 (as all the
        MATLAB recipes use).

    Returns
    -------
    inflatedAtlas : nibabel.Nifti1Image
    """
    #Claude note: step 1, de-islanding
    atlasIter = _atlasData(deislandLabels_bsc(atlas, 5, 999))
    sourceLabels = GREY_MATTER_LABELS + SUBCORTICAL_LABELS
    structure = np.ones((3, 3, 3), dtype=bool)
    shape = np.asarray(atlasIter.shape)
    #Claude note: kept for step 3 (MATLAB's atlasIterCur after the loop)
    atlasIterCur = atlasIter
    for iIter in range(inflateIter):
        atlasIterCur = atlasIter
        atlasIterNext = atlasIter.copy()
        #Claude note: the target voxels touching a source label
        grownSources = ndimage.binary_dilation(np.isin(atlasIterCur, sourceLabels), structure=structure)
        margin = grownSources & np.isin(atlasIterCur, INFLATION_TARGET_LABELS)
        marginCoords = np.array(np.nonzero(margin)).T
        #Claude note: MATLAB drops voxels on the image edge
        onEdge = np.any((marginCoords == 0) | (marginCoords == shape - 1), axis=1)
        marginCoords = marginCoords[~onEdge]
        #Claude note: vote, then write all winners at once
        winners = _vote(atlasIterCur, marginCoords, excludeLabels=INFLATION_TARGET_LABELS)
        hasWinner = winners >= 0
        atlasIterNext[tuple(marginCoords[hasWinner].T)] = winners[hasWinner]
        atlasIter = atlasIterNext
    #Claude note: step 3, the remaining 999 voxels
    ninesCoords = np.array(np.nonzero(atlasIter == 999)).T
    if len(ninesCoords) > 0:
        winners = _vote(atlasIterCur, ninesCoords, excludeLabels=INFLATION_TARGET_LABELS)
        #Claude note: no candidate: MATLAB takes the smallest label present
        for iNone in np.where(winners < 0)[0]:
            labels, _ = _windowCounts(atlasIterCur, ninesCoords[iNone], 1)
            winners[iNone] = labels[0]
        atlasIter = atlasIter.copy()
        atlasIter[tuple(ninesCoords.T)] = winners
    return nib.Nifti1Image(atlasIter.astype(np.int32), atlas.affine)


def endpointAtlasLabels_bsc(atlas, points):
    """
    Claude note: atlas labels at subject space points, after MATLAB's
    bsc_atlasROINumsFromCoords_v3(atlas, coords, 'acpc'): the most common
    label in the 3x3x3 window around the point's voxel, ties broken by
    growing the window.

    The voxel is the one containing the point (nearest voxel centre), and
    tied windows are grown symmetrically.  MATLAB instead took ``floor`` of
    the continuous voxel coordinate, which shifts every lookup by half a
    voxel towards lower indexes (in subject space a direction set by the
    image orientation), and counted tied windows over y-1 : y+r.  Neither is
    a deliberate choice, so neither is reproduced (Dan Bullock, 2026-11-14:
    no preserving unintentional idiosyncrasies).

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        Integer label atlas (in the MATLAB pipeline: the inflated atlas).
    points : np.ndarray, (N,3)
        Subject space coordinates (e.g. streamline endpoints).

    Returns
    -------
    labels : np.ndarray of int, (N,)
    """
    atlasData = _atlasData(atlas)
    #Claude note: continuous voxel coordinates, then the voxel holding the
    #point (round)
    voxelCoords = nib.affines.apply_affine(np.linalg.inv(atlas.affine), np.asarray(points, dtype=float))
    voxelIndexes = np.round(voxelCoords).astype(np.intp)
    #Claude note: points outside the image are clipped onto its edge
    voxelIndexes = np.clip(voxelIndexes, 0, np.asarray(atlasData.shape) - 1)
    return _vote(atlasData, voxelIndexes, excludeLabels=None)


# Claude note: ---------------------------------------------------------------
# streamline helpers
# ---------------------------------------------------------------------------

def _endpoints(streamlines):
    """
    Claude note: first and last node of each streamline, two (N,3) arrays.
    """
    first = wmaPyTools.segmentationTools._nodesAtPositions(streamlines, lambda lengths: np.zeros(len(lengths), dtype=np.intp))
    last = wmaPyTools.segmentationTools._nodesAtPositions(streamlines, lambda lengths: lengths - 1)
    return first, last


def _midpoints(streamlines):
    """
    Claude note: the MATLAB midpoint node, ``floor(n/2)`` in 1-based
    indexing (node 1 for one-node streamlines), i.e. 0-based
    ``max(n//2, 1) - 1``.
    """
    return wmaPyTools.segmentationTools._nodesAtPositions(streamlines, lambda lengths: np.maximum(lengths // 2, 1) - 1)


def _arcLengths(streamlines):
    """
    Claude note: the length in mm of each streamline (sum of its segment
    lengths).
    """
    nodes, lengths = wmaPyTools.segmentationTools._flattenStreamlines(streamlines)
    if len(nodes) == 0:
        return np.zeros(len(lengths))
    #Claude note: all consecutive node distances, then drop the ones that
    #bridge two streamlines
    stepLengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    streamIDs = np.repeat(np.arange(len(lengths)), lengths)
    withinStream = streamIDs[1:] == streamIDs[:-1]
    return np.bincount(streamIDs[1:][withinStream], weights=stepLengths[withinStream], minlength=len(lengths))


#Claude note: vistasoft's streamline-to-ROI distance (dtiIntersectFibersWithRoi
#as called by wma_SegmentFascicleFromConnectome, and bsc_tractByEndpointROIs)
VISTASOFT_MIN_DIST = 0.87
#Claude note: bsc_tractByEndpointROIs and bsc_midpointROISegment compare
#vistasoft nearpoints' output, which is the *squared* distance, with 0.87 and
#0.5, so their real distance limits are sqrt(0.87) = 0.933 mm and
#sqrt(0.5) = 0.707 mm.  (Found by validation: streamlines MATLAB assigned
#with an endpoint 0.87-0.93 mm from the ROI.)
ENDPOINT_PAIR_MIN_DIST = float(np.sqrt(0.87))
MIDPOINT_ROI_MAX_DIST = float(np.sqrt(0.5))


def segmentByROIs(streamlines, rois, operations, tol=VISTASOFT_MIN_DIST):
    """
    Claude note: MATLAB-style multi-ROI criteria, as
    wma_SegmentFascicleFromConnectome(fg, rois, operations).

    Each criterion is dipy's near_roi (computed by the exact KD-tree
    implementation segmentationTools uses), applied in turn to the
    streamlines that met the previous ones.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    rois : list of nibabel.Nifti1Image
        The ROI masks.
    operations : list of str
        One per ROI, MATLAB names: 'and' (any node near the ROI),
        'not' (no node near it), 'endpoints' (either endpoint near it),
        'both_endpoints' (both endpoints near it).
    tol : float or None, optional
        Distance (mm) from an ROI voxel centre that counts as "near".  The
        default, 0.87, is vistasoft's; None uses segmentTractMultiROI's
        (dipy's) default, the voxel centre-to-corner distance (0.866 mm for
        1 mm voxels).  A streamline endpoint 0.868 mm from the thalamus is
        an example (real data) where the two differ.  Values below the
        centre-to-corner distance (e.g. for 2 mm voxels) are raised to it,
        as dipy does, but without its warning.

    Returns
    -------
    keepBool : np.ndarray of bool
        Streamlines meeting all the criteria.
    """
    #Claude note: MATLAB operation -> (include, dipy mode)
    operationDict = {'and': (True, 'any'), 'not': (False, 'any'),
                     'endpoints': (True, 'either_end'), 'both_endpoints': (True, 'both_end')}
    if len(rois) != len(operations):
        raise ValueError('segmentByROIs: one operation per ROI is required')
    keepBool = np.ones(len(streamlines), dtype=bool)
    #Claude note: with tol=None this is exactly segmentTractMultiROI
    if tol is None:
        criteria = [(iRoi,) + operationDict[iOp.lower()] for iRoi, iOp in zip(rois, operations)]
        nonEmpty = [iCrit for iCrit in criteria if np.any(np.asanyarray(iCrit[0].dataobj))]
        #Claude note: an empty ROI meets no inclusion criterion and every
        #exclusion one (as in MATLAB)
        if any(iCrit[1] and not np.any(np.asanyarray(iCrit[0].dataobj)) for iCrit in criteria):
            return np.zeros(len(streamlines), dtype=bool)
        if len(nonEmpty) == 0:
            return keepBool
        return wmaPyTools.segmentationTools.segmentTractMultiROI(streamlines, [c[0] for c in nonEmpty],
                                                                 [c[1] for c in nonEmpty], [c[2] for c in nonEmpty]).astype(bool)
    #Claude note: endpoint criteria first (only two nodes per streamline),
    #then inclusions, then exclusions; the result does not depend on order
    rank = {'endpoints': 0, 'both_endpoints': 0, 'and': 1, 'not': 2}
    for iCriterion in sorted(range(len(rois)), key=lambda i: rank[operations[i].lower()]):
        include, mode = operationDict[operations[iCriterion].lower()]
        roi = rois[iCriterion]
        roiData = np.asanyarray(roi.dataobj) != 0
        #Claude note: never below the voxel's centre-to-corner distance
        roiTol = max(tol, dist_to_corner(roi.affine))
        survivors = np.where(keepBool)[0]
        if len(survivors) == 0:
            break
        #Claude note: near the ROI, per surviving streamline
        nearBool = wmaPyTools.segmentationTools._nearROIExact(streamlines[survivors], roi.affine, roiData, tol=roiTol, mode=mode)
        keepBool[survivors] = nearBool if include else ~nearBool
    return keepBool


def _segmentSubset(streamlines, subsetIndexes, rois, operations):
    """
    Claude note: segmentByROIs evaluated only on streamlines[subsetIndexes],
    returned as a boolean vector over all the streamlines (False outside
    the subset).  Used when every tract of a family shares a first
    criterion, so the rest need only be checked on its survivors.
    """
    outBool = np.zeros(len(streamlines), dtype=bool)
    if len(subsetIndexes) > 0:
        outBool[subsetIndexes] = segmentByROIs(streamlines[subsetIndexes], rois, operations)
    return outBool


def endpointCriteria_bsc(streamlines, reference, location, whichEndpoints):
    """
    Claude note: endpoints strictly beyond a reference coordinate, as
    MATLAB's bsc_applyEndpointCriteria(fg, ref, location, which) (one
    criterion; call repeatedly and AND for several).

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    reference : nibabel.Nifti1Image or array-like of 3 floats
        A (planar) ROI, whose mean voxel coordinate is used, or a coordinate.
    location : str
        'superior'/'top', 'inferior'/'bottom', 'anterior', 'posterior',
        'medial' or 'lateral' (hemisphere from the sign of the reference x).
    whichEndpoints : str
        'one' (exactly one), 'both', 'neither' or 'either' (at least one).

    Returns
    -------
    criteriaBool : np.ndarray of bool
    """
    if isinstance(reference, nib.Nifti1Image):
        refCoord = roiMeanCoord(reference)
    else:
        refCoord = np.asarray(reference, dtype=float)
    leftFlag = refCoord[0] < 0
    locationDict = {'top': (2, 1), 'superior': (2, 1), 'bottom': (2, -1), 'inferior': (2, -1),
                    'anterior': (1, 1), 'posterior': (1, -1),
                    'medial': (0, 1 if leftFlag else -1), 'lateral': (0, -1 if leftFlag else 1)}
    dimension, valence = locationDict[location.lower()]
    first, last = _endpoints(streamlines)
    #Claude note: is each endpoint beyond the reference?
    endpointCoords = np.column_stack([first[:, dimension], last[:, dimension]])
    beyond = endpointCoords > refCoord[dimension] if valence > 0 else endpointCoords < refCoord[dimension]
    nBeyond = np.sum(beyond, axis=1)
    whichDict = {'one': nBeyond == 1, 'both': nBeyond == 2, 'neither': nBeyond == 0, 'either': nBeyond >= 1}
    return whichDict[whichEndpoints.lower()]


# Claude note: ---------------------------------------------------------------
# priors (bsc_segmentSuperficialFibers, bsc_streamlineCategoryPriors_v6)
# ---------------------------------------------------------------------------

def segmentSuperficialFibers(streamlines, atlas, inflatedAtlas=None):
    """
    Claude note: short "u-fibers", as MATLAB's
    bsc_segmentSuperficialFibers(wbfg, atlas): per hemisphere, streamlines
    shorter than 30 mm whose midpoint is within 0.707 mm (MATLAB: squared
    distance 0.5) of an (inflated)
    cortical voxel centre, which avoid the hemisphere's (inflated) white
    matter and the other hemisphere's white matter grown by 2 voxels.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas.
    inflatedAtlas : nibabel.Nifti1Image, optional
        inflateAtlasLabels_bsc(atlas, 2), computed if not given.

    Returns
    -------
    classification : dict
        wmc-style, tracts 'leftUfibers', 'rightUfibers'.
    """
    if inflatedAtlas is None:
        inflatedAtlas = inflateAtlasLabels_bsc(atlas, 2)
    classification = emptyClassification(len(streamlines))
    #Claude note: midpoint near a cortical voxel centre: MATLAB's "0.5" is a
    #squared distance, i.e. within sqrt(0.5) = 0.707 mm
    greyCoords = _maskVoxelCoords(roiFromAtlasLabels_bsc(inflatedAtlas, GREY_MATTER_LABELS, 1))
    distances, _ = cKDTree(greyCoords).query(_midpoints(streamlines), distance_upper_bound=MIDPOINT_ROI_MAX_DIST * (1 + 1e-9))
    midpointBool = distances <= MIDPOINT_ROI_MAX_DIST
    #Claude note: shorter than 30 mm
    shortBool = _arcLengths(streamlines) < 30
    for iSide in range(2):
        ownWM = roiFromAtlasLabels_bsc(inflatedAtlas, WM_LABELS[iSide], 1)
        otherWM = roiFromAtlasLabels_bsc(atlas, WM_LABELS[1 - iSide], 5)
        #Claude note: only the streamlines already short with a cortical
        #midpoint need the (slower) ROI test
        candidates = np.where(shortBool & midpointBool)[0]
        avoidBool = np.zeros(len(streamlines), dtype=bool)
        if len(candidates) > 0:
            avoidBool[candidates] = segmentByROIs(streamlines[candidates], [ownWM, otherWM], ['not', 'not'])
        classification = addTractToClassification(classification, SIDE_NAMES[iSide] + 'Ufibers',
                                                   shortBool, midpointBool, avoidBool)
    return classification


#Claude note: the category label lists of bsc_streamlineCategoryPriors_v6
_PRIOR_LEFT = list(np.arange(101, 176) + 11000) + [26, 17, 18, 7, 8, 10, 11, 12, 13]
_PRIOR_RIGHT = list(np.arange(101, 176) + 12000) + [46, 47, 49, 50, 51, 52, 53, 54, 58]
_PRIOR_SPINE = [16, 28, 60]
_PRIOR_CEREBELLUM = [8, 47, 7, 46]
_PRIOR_VENTRICLE = [31, 63, 4, 43, 14, 24, 15, 44, 5, 62, 30, 80, 72]
_PRIOR_CC = [251, 252, 253, 254, 255]
_PRIOR_UNKNOWN = [0, 2000, 1000, 77, 78, 79, 80, 81, 82, 24, 42, 3]
_PRIOR_OPTIC_CHIASM = [85]
_PRIOR_FRONTAL_BASE = [124, 148, 118, 165, 101, 154, 105, 115, 154, 155, 115, 170, 129, 146, 153,
                       164, 106, 116, 108, 131, 171, 112, 150, 104, 169, 114, 113, 116, 107, 163, 139, 132, 140]
_PRIOR_TEMPORAL_BASE = [144, 134, 138, 137, 173, 174, 135, 175, 121, 151, 123, 162, 133]
_PRIOR_OCCIPITAL_BASE = [120, 119, 111, 158, 166, 143, 145, 159, 152, 122, 162, 161, 121, 160, 102]
_PRIOR_PARIETAL_BASE = [157, 127, 168, 136, 126, 125, 156, 128, 141, 172, 147, 109, 103, 130, 110]


def _bothSides(baseLabels):
    """
    Claude note: a2009s base label numbers (101-175) for both hemispheres.
    """
    return list(np.asarray(baseLabels) + 11000) + list(np.asarray(baseLabels) + 12000)


def _terminationCategories(labels, order):
    """
    Claude note: the endpoint category of each label, first match in
    ``order`` (a list of (category name, label list)); MATLAB raised an error
    for a label in none of them, as does this.
    """
    categories = np.full(len(labels), '', dtype=object)
    unassigned = np.ones(len(labels), dtype=bool)
    for iName, iLabels in order:
        match = unassigned & np.isin(labels, iLabels)
        categories[match] = iName
        unassigned &= ~match
    if np.any(unassigned):
        raise ValueError('streamline identity unaccounted for: labels ' + str(np.unique(np.asarray(labels)[unassigned])))
    return categories


#Claude note: the a2009s labels listed as both temporal and occipital in the
#category priors (v6, v7): G_oc-temp_lat-fusifor (121, fusiform gyrus) and
#S_oc-temp_med_and_Lingual (162, collateral and lingual sulci); first match
#makes them always temporal unless the Destrieux split is used
_TEMPORO_OCCIPITAL_BOUNDARY_BASE = [121, 162]


def temporoOccipitalLandmarks(atlas):
    """
    Claude note: EXPERIMENTAL.  The landmarks of the ventral temporo-occipital
    boundary of Destrieux et al. 2010 (NeuroImage 53:1-15,
    doi:10.1016/j.neuroimage.2010.06.010, section "Limits of temporal and
    occipital lobes"), per hemisphere, from an aparc.a2009s+aseg atlas.
    Destrieux et al. draw that boundary, on the inflated surface, as a
    virtual line from the infero-medial tip of the anterior occipital
    sulcus / temporo-occipital incisure (S_occipital_ant, x160) to the
    anterior tip of the calcarine sulcus (S_calcarine, x145).

    The tips are found with simple percentile heuristics chosen on one
    subject (sub-100206) and not yet validated on others:

    - incisure tip: the inferior 10% (by z) of the x160 voxels, then the
      most medial 10% (by abs(x)) of those, averaged;
    - calcarine tip: the anterior 1% (by y) of the x145 voxels, averaged;
    - occipital side: the centroid of the occipital pole (Pole_occipital,
      x143), used only to tell which side of the line is occipital.

    Coordinates are world (RAS, mm); the atlas is assumed to be roughly
    in a RAS-aligned space (x left-right, y posterior-anterior, z
    inferior-superior), as FreeSurfer output is.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas (e.g. prepareAtlas_bsc output).

    Returns
    -------
    landmarks : dict
        ``{11000: {...}, 12000: {...}}`` (left, right label offsets), each
        with 'incisureTip', 'calcarineTip' and 'occipitalPole' (3-vectors,
        mm).

    Raises
    ------
    ValueError
        If a landmark label (x160, x145, x143) is missing in a hemisphere,
        or the line is degenerate (the two tips less than 1 mm apart in the
        axial plane, or the occipital pole on the line).
    """
    atlasData = np.asarray(atlas.dataobj)
    landmarks = {}
    for iOffset in (11000, 12000):
        hemisphereName = 'left' if iOffset == 11000 else 'right'
        coordsByBase = {}
        for iBase in (160, 145, 143):
            voxels = np.argwhere(atlasData == iOffset + iBase)
            if len(voxels) == 0:
                raise ValueError('temporoOccipitalLandmarks: label ' + str(iOffset + iBase) + ' missing in the '
                                 + hemisphereName + ' hemisphere; the Destrieux temporo-occipital split needs '
                                 + 'S_occipital_ant (x160), S_calcarine (x145) and Pole_occipital (x143)')
            coordsByBase[iBase] = nib.affines.apply_affine(atlas.affine, voxels)
        #Claude note: incisure tip, inferior 10% then most medial 10%
        incisure = coordsByBase[160]
        inferior = incisure[incisure[:, 2] <= np.percentile(incisure[:, 2], 10)]
        medial = inferior[np.abs(inferior[:, 0]) <= np.percentile(np.abs(inferior[:, 0]), 10)]
        incisureTip = medial.mean(axis=0)
        #Claude note: calcarine tip, anterior 1%
        calcarine = coordsByBase[145]
        calcarineTip = calcarine[calcarine[:, 1] >= np.percentile(calcarine[:, 1], 99)].mean(axis=0)
        occipitalPole = coordsByBase[143].mean(axis=0)
        #Claude note: degeneracy checks
        lineVector = (calcarineTip - incisureTip)[:2]
        if np.linalg.norm(lineVector) < 1.0:
            raise ValueError('temporoOccipitalLandmarks: degenerate line in the ' + hemisphereName
                             + ' hemisphere (incisure and calcarine tips < 1 mm apart in the axial plane)')
        if abs(_sideOfLine(occipitalPole[np.newaxis, :], incisureTip, calcarineTip)[0]) < 1e-6:
            raise ValueError('temporoOccipitalLandmarks: the occipital pole lies on the temporo-occipital line in the '
                             + hemisphereName + ' hemisphere')
        landmarks[iOffset] = {'incisureTip': incisureTip, 'calcarineTip': calcarineTip, 'occipitalPole': occipitalPole}
    return landmarks


def _sideOfLine(points, lineStart, lineEnd):
    """
    Claude note: signed distance-like value of each point (n x 3, mm) from
    the vertical plane through lineStart and lineEnd (only x and y used).
    """
    lineVector = (np.asarray(lineEnd) - np.asarray(lineStart))[:2]
    normal = np.array([-lineVector[1], lineVector[0]])
    return (np.asarray(points)[:, :2] - np.asarray(lineStart)[:2]) @ normal


def _splitTemporoOccipitalCategories(categories, labels, points, landmarks):
    """
    Claude note: EXPERIMENTAL.  Reassign the endpoint categories of the
    temporo-occipital boundary labels (x121, x162, see
    _TEMPORO_OCCIPITAL_BOUNDARY_BASE) to 'occipital' or 'temporal' by the
    side of the hemisphere's Destrieux line (temporoOccipitalLandmarks) the
    endpoint is on: the occipital pole's side is occipital; points on the
    line count as temporal.  Only the category changes; the label (and so
    the hemisphere and grey matter flags) is untouched.

    Parameters
    ----------
    categories : numpy.ndarray
        _terminationCategories output (modified copy returned).
    labels : array-like
        Endpoint labels (inflated atlas).
    points : numpy.ndarray
        Endpoint coordinates, n x 3, world mm.
    landmarks : dict
        temporoOccipitalLandmarks output.

    Returns
    -------
    categories : numpy.ndarray
    """
    categories = np.array(categories, dtype=object)
    labels = np.asarray(labels)
    for iOffset, iLandmarks in landmarks.items():
        boundaryBool = np.isin(labels, np.asarray(_TEMPORO_OCCIPITAL_BOUNDARY_BASE) + iOffset)
        if not np.any(boundaryBool):
            continue
        sideArgs = (iLandmarks['incisureTip'], iLandmarks['calcarineTip'])
        poleSide = np.sign(_sideOfLine(iLandmarks['occipitalPole'][np.newaxis, :], *sideArgs)[0])
        occipitalBool = np.sign(_sideOfLine(np.asarray(points)[boundaryBool], *sideArgs)) == poleSide
        boundaryIndices = np.where(boundaryBool)[0]
        categories[boundaryIndices[occipitalBool]] = 'occipital'
        categories[boundaryIndices[~occipitalBool]] = 'temporal'
    return categories


def streamlineCategoryPriors(streamlines, atlas, inflatedAtlas=None, superficialClassification=None,
                             temporoOccipitalSplit=False):
    """
    Claude note: endpoint-category prior, as MATLAB's
    bsc_streamlineCategoryPriors_v6(wbfg, atlas, 2).  Each streamline is
    named after the regions its two endpoints are in, e.g.
    'leftfrontal_to_subcortical', 'frontal_to_frontal_interHemi',
    'rightoccipital_to_occipital_ufiber'.

    Endpoint labels come from the inflated atlas (endpointAtlasLabels_bsc).
    Each endpoint is frontal / temporal / occipital / parietal /
    subcortical / spinal / insula / cerebellum / CorpusCallosum / ventricle
    / unlabeled / whiteMatter / pericallosal / OpticChi (first match, in
    MATLAB's order), and the name is the alphabetically sorted pair
    ``a_to_b``, with ``_interHemi`` (one end in each hemisphere) or
    ``_ufiber`` (a superficial fibre ending in cortex) appended and
    ``left`` / ``right`` prepended for single hemisphere streamlines.  Names
    containing CorpusCallosum, unlabeled, OpticChi, ventricle, whiteMatter
    or pericallosal are pooled under that word.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas.
    inflatedAtlas : nibabel.Nifti1Image, optional
        inflateAtlasLabels_bsc(atlas, 2), computed if not given.
    superficialClassification : dict, optional
        segmentSuperficialFibers output, computed if not given.
    temporoOccipitalSplit : bool, optional
        EXPERIMENTAL (landmark heuristics tuned on one subject, pending
        multi-subject validation).  Assign endpoints
        in the fusiform gyrus (x121) and the collateral / lingual sulcus
        (x162), which MATLAB lists as both temporal and occipital (so
        first-match makes them always temporal), to occipital or temporal by
        the Destrieux et al. 2010 ventral temporo-occipital line (see
        temporoOccipitalLandmarks), per hemisphere, from ``atlas``.  The
        default is False (MATLAB's lists).

    Returns
    -------
    classification : dict
        wmc-style classification, one name per category.
    """
    if inflatedAtlas is None:
        inflatedAtlas = inflateAtlasLabels_bsc(atlas, 2)
    if superficialClassification is None:
        superficialClassification = segmentSuperficialFibers(streamlines, atlas, inflatedAtlas)
    superficialBool = np.asarray(superficialClassification['index']) > 0
    #Claude note: the endpoint labels
    first, last = _endpoints(streamlines)
    label1 = endpointAtlasLabels_bsc(inflatedAtlas, first)
    label2 = endpointAtlasLabels_bsc(inflatedAtlas, last)
    #Claude note: the label groups
    frontal, temporal = _bothSides(_PRIOR_FRONTAL_BASE), _bothSides(_PRIOR_TEMPORAL_BASE)
    occipital, parietal = _bothSides(_PRIOR_OCCIPITAL_BASE), _bothSides(_PRIOR_PARIETAL_BASE)
    pericallosal, insula = _bothSides([167]), _bothSides([117, 149])
    #Claude note: the two endpoints are checked in slightly different orders
    #in MATLAB (only matters for labels in two lists)
    order1 = [('frontal', frontal), ('temporal', temporal), ('occipital', occipital), ('parietal', parietal),
              ('subcortical', SUBCORTICAL_LABELS), ('spinal', _PRIOR_SPINE), ('insula', insula),
              ('cerebellum', _PRIOR_CEREBELLUM), ('CorpusCallosum', _PRIOR_CC), ('ventricle', _PRIOR_VENTRICLE),
              ('unlabeled', _PRIOR_UNKNOWN), ('whiteMatter', WM_LABELS), ('pericallosal', pericallosal),
              ('OpticChi', _PRIOR_OPTIC_CHIASM)]
    order2 = [('frontal', frontal), ('temporal', temporal), ('occipital', occipital), ('parietal', parietal),
              ('subcortical', SUBCORTICAL_LABELS), ('spinal', _PRIOR_SPINE), ('insula', insula),
              ('cerebellum', _PRIOR_CEREBELLUM), ('pericallosal', pericallosal), ('ventricle', _PRIOR_VENTRICLE),
              ('unlabeled', _PRIOR_UNKNOWN), ('whiteMatter', WM_LABELS), ('CorpusCallosum', _PRIOR_CC),
              ('OpticChi', _PRIOR_OPTIC_CHIASM)]
    term1 = _terminationCategories(label1, order1)
    term2 = _terminationCategories(label2, order2)
    #Claude note: optional (experimental) Destrieux split of x121 / x162
    if temporoOccipitalSplit:
        landmarks = temporoOccipitalLandmarks(atlas)
        term1 = _splitTemporoOccipitalCategories(term1, label1, first, landmarks)
        term2 = _splitTemporoOccipitalCategories(term2, label2, last, landmarks)
    #Claude note: hemisphere and validity flags, as in MATLAB
    excludeSide = _PRIOR_UNKNOWN + pericallosal + _PRIOR_CC + _PRIOR_OPTIC_CHIASM + WM_LABELS + _PRIOR_SPINE + _PRIOR_VENTRICLE
    validSide = _PRIOR_LEFT + _PRIOR_RIGHT
    left1, left2 = np.isin(label1, _PRIOR_LEFT), np.isin(label2, _PRIOR_LEFT)
    right1, right2 = np.isin(label1, _PRIOR_RIGHT), np.isin(label2, _PRIOR_RIGHT)
    excludeBool = np.isin(label1, excludeSide) | np.isin(label2, excludeSide)
    includeBool = np.isin(label1, validSide) | np.isin(label2, validSide)
    validU = (np.isin(label1, GREY_MATTER_LABELS) | np.isin(label2, GREY_MATTER_LABELS)) & ~excludeBool
    leftBool, rightBool = left1 & left2, right1 & right2
    interHemiBool = (left2 & right1) | (right2 & left1)
    singleLeft, singleRight = left1 ^ left2, right1 ^ right2
    interhemiFlag = interHemiBool & includeBool
    #Claude note: build each streamline's name
    streamNames = []
    for iStream in range(len(term1)):
        pairNames = sorted([term1[iStream], term2[iStream]])
        baseName = pairNames[0] + '_to_' + pairNames[1]
        streamName = baseName + '_interHemi' if interhemiFlag[iStream] else baseName
        if superficialBool[iStream] and validU[iStream]:
            streamName = baseName + '_ufiber'
        if (leftBool[iStream] or singleLeft[iStream]) and includeBool[iStream] and not interhemiFlag[iStream]:
            streamName = 'left' + streamName
        elif (rightBool[iStream] or singleRight[iStream]) and includeBool[iStream] and not interhemiFlag[iStream]:
            streamName = 'right' + streamName
        #Claude note: MATLAB's `~superficial==0` reads as `superficial~=0`
        if interhemiFlag[iStream] and validU[iStream] and superficialBool[iStream]:
            streamName = 'MaskFailure'
        streamNames.append(streamName)
    #Claude note: pool the names containing a summary word (first such word
    #in MATLAB's list)
    summarizeNames = ['CorpusCallosum', 'unlabeled', 'OpticChi', 'ventricle', 'whiteMatter', 'pericallosal']
    for iStream, iName in enumerate(streamNames):
        for iSummary in summarizeNames:
            if iSummary in iName:
                streamNames[iStream] = iSummary
                break
    #Claude note: the classification, names sorted
    uniqueNames, streamIndex = np.unique(np.asarray(streamNames, dtype=str), return_inverse=True)
    return {'names': [str(iName) for iName in uniqueNames], 'index': streamIndex.astype(np.int64) + 1}


# Claude note: ---------------------------------------------------------------
# priors, version 7 (bsc_inflateLabels_v3, bsc_segmentSuperficialFibers_v3,
# bsc_streamlineCategoryPriors_v7; wma_tools commits c9379a3 .. f2a1b2d,
# April-May 2020, as of b0792d8)
# ---------------------------------------------------------------------------

def inflateAtlasLabels_v3_bsc(atlas, inflateIter=2):
    """
    Claude note: grow cortical and subcortical labels into white matter, as
    MATLAB's bsc_inflateLabels_v3(atlas, inflateIter), used by the version 7
    priors (streamlineCategoryPriors_v7, segmentSuperficialFibers_v3).

    Each of inflateIter iterations relabels every white matter (2, 41),
    unknown cortex (1000, 2000) or 999 voxel that touches (26-neighbourhood)
    a cortical or subcortical voxel, from the labels in its 3x3x3 window
    other than those target labels.

    How v3 differs from bsc_inflateLabels (inflateAtlasLabels_bsc):

    - no de-islanding first (and so no 999 cleanup step): island-clean the
      atlas beforehand (prepareAtlas_bsc), as the v7 prior does by default;
    - voxels on the image edge are not skipped (MATLAB pads with zeros).

    The vote.  The most common non-target label in the window wins, ties are
    broken by growing the window, and all votes of an iteration use the
    labels from its start; that is exactly the vote of
    inflateAtlasLabels_bsc (bsc_inflateLabels, Dan Bullock's earlier
    implementation of the same idea).  On an island-cleaned atlas v3
    therefore equals inflateAtlasLabels_bsc except within one voxel of the
    image edge.  MATLAB's bsc_inflateLabels_v3 writes
    ``voteWinner=labels(find(max(counts)))``, which always picks the
    *smallest* non-target label in the window (so a target voxel next to
    background becomes background), and writes each winner into the array it
    is still reading, so its result depends on the voxel visiting order.
    That is an evident coding slip (the stated intent is a vote) and is not
    reproduced (Dan Bullock, 2026-11-14).

    Parameters
    ----------
    atlas : nibabel.Nifti1Image
        A FreeSurfer aparc.a2009s+aseg style atlas (island-cleaned).
    inflateIter : int, optional
        The number of inflation iterations.  The default is 2 (as
        bsc_segmentSuperficialFibers_v3 uses).

    Returns
    -------
    inflatedAtlas : nibabel.Nifti1Image
    """
    #Claude note: the labels that grow (cortex and subcortex) and the labels
    #they grow into (white matter, unknown cortex, 999), as in MATLAB
    sourceLabels = GREY_MATTER_LABELS + SUBCORTICAL_LABELS
    targetLabels = WM_LABELS + [2000, 1000, 999]
    workingData = _atlasData(atlas)
    structure = np.ones((3, 3, 3), dtype=bool)
    for iIter in range(inflateIter):
        #Claude note: the target voxels touching a source label
        grownSources = ndimage.binary_dilation(np.isin(workingData, sourceLabels), structure=structure)
        margin = grownSources & np.isin(workingData, targetLabels)
        marginCoords = np.array(np.nonzero(margin)).T
        #Claude note: the majority vote among the non-target labels of
        #each voxel's window (out-of-image neighbours don't vote), all
        #against the labels from the start of the iteration
        winners = _vote(workingData, marginCoords, excludeLabels=targetLabels)
        hasWinner = winners >= 0
        workingData = workingData.copy()
        workingData[tuple(marginCoords[hasWinner].T)] = winners[hasWinner]
    return nib.Nifti1Image(workingData.astype(np.int32), atlas.affine)


def segmentSuperficialFibers_v3(streamlines, atlas, inflatedAtlas=None):
    """
    Claude note: short "u-fibers", as MATLAB's
    bsc_segmentSuperficialFibers_v3(wbfg, atlas) (refactored 2020-04-20, used
    by the version 7 priors).  Per hemisphere, the streamlines that meet all
    four of MATLAB's criteria:

    1. the midpoint is within 0.707 mm (MATLAB: squared distance 0.5) of an
       inflated cortical voxel centre (either hemisphere);
    2. shorter than 30 mm (MATLAB: "for now we'll just arbitrarily select
       30 mm");
    3. both endpoints within 0.933 mm (MATLAB: squared distance 0.87) of
       this hemisphere's inflated cortex (bsc_tractByEndpointROIs with the
       same ROI twice); MATLAB notes that requiring one hemisphere is "a
       stipulation on our part";
    4. no node near this hemisphere's deep white matter (the white matter
       label left in the inflated atlas).

    Unlike bsc_segmentSuperficialFibers (segmentSuperficialFibers), v3 adds
    criterion 3 and no longer excludes streamlines near the other
    hemisphere's white matter.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas (island-cleaned).
    inflatedAtlas : nibabel.Nifti1Image, optional
        inflateAtlasLabels_v3_bsc(atlas, 2), computed if not given.

    Returns
    -------
    classification : dict
        wmc-style, tracts 'left_u-fibers', 'right_u-fibers'.
    """
    if inflatedAtlas is None:
        inflatedAtlas = inflateAtlasLabels_v3_bsc(atlas, 2)
    classification = emptyClassification(len(streamlines))
    #Claude note: each hemisphere's inflated cortex
    sideGreyROIs = [roiFromAtlasLabels_bsc(inflatedAtlas, list(np.arange(101, 176) + SIDE_OFFSETS[iSide]), 1)
                    for iSide in range(2)]
    #Claude note: criterion 1, midpoint near a cortical voxel centre of the
    #merged (both hemispheres) cortex; MATLAB's "0.5" is a squared distance
    greyCoords = _maskVoxelCoords(mergeROIs(sideGreyROIs[0], sideGreyROIs[1]))
    distances, _ = cKDTree(greyCoords).query(_midpoints(streamlines), distance_upper_bound=MIDPOINT_ROI_MAX_DIST * (1 + 1e-9))
    midpointBool = distances <= MIDPOINT_ROI_MAX_DIST
    #Claude note: criterion 2, shorter than 30 mm
    shortBool = _arcLengths(streamlines) < 30
    for iSide in range(2):
        #Claude note: criterion 3, both endpoints in this hemisphere's cortex
        corticalCriteriaBool = tractByEndpointROIs_bsc(streamlines, sideGreyROIs[iSide], sideGreyROIs[iSide])
        #Claude note: criterion 4, avoid this hemisphere's deep white matter;
        #only the streamlines meeting 1-3 need the (slower) ROI test
        wmROIDeep = roiFromAtlasLabels_bsc(inflatedAtlas, WM_LABELS[iSide], 1)
        candidates = np.where(midpointBool & shortBool & corticalCriteriaBool)[0]
        superficialFibersBool = np.zeros(len(streamlines), dtype=bool)
        if len(candidates) > 0:
            superficialFibersBool[candidates] = segmentByROIs(streamlines[candidates], [wmROIDeep], ['not'])
        classification = addTractToClassification(classification, SIDE_NAMES[iSide] + '_u-fibers',
                                                   midpointBool, shortBool, corticalCriteriaBool, superficialFibersBool)
    return classification


#Claude note: the label lists of bsc_streamlineCategoryPriors_v7 that differ
#from v6 (the others, i.e. left / right sides, spinal, cerebellum, corpus
#callosum, unknown, optic chiasm, white matter, and the temporal, occipital,
#parietal and pericallosal cortex, are the same lists as in v6)
#Claude note: v7 "subcortical" keeps only the aseg labels not given a
#category of their own below (20/56 and 27/59 in the FreeSurfer lookup
#table; usually absent from aparc.a2009s+aseg)
_PRIOR_V7_SUBCORTICAL = [20, 27, 56, 59]
#Claude note: v7's ventricle list without the caudate (11/50), which MATLAB
#v7 also lists here (Dan Bullock, 2026-11-14: "I would suggest fixing the
#caudate"); this is v6's ventricle list
_PRIOR_V7_VENTRICLE = [31, 63, 4, 43, 14, 24, 15, 44, 5, 62, 30, 80, 72]
#Claude note: v6's frontal list without G_insular_short (118), which v7
#moves to the insula
_PRIOR_V7_FRONTAL_BASE = [124, 148, 165, 101, 154, 105, 115, 154, 155, 115, 170, 129, 146, 153,
                          164, 106, 116, 108, 131, 171, 112, 150, 104, 169, 114, 113, 116, 107, 163, 139, 132, 140]
#Claude note: the insula: aseg Left/Right-Insula (19, 55) and the a2009s
#G_Ins_lg_and_S_cent_ins (117), G_insular_short (118) and
#S_circular_insula_inf (149) of each hemisphere
_PRIOR_V7_INSULA = [19] + list(np.array([117, 118, 149]) + 11000) + [55] + list(np.array([117, 118, 149]) + 12000)
#Claude note: the four subcortical groups v7 splits "subcortical" into,
#left then right: thalamus; caudate and accumbens; putamen and pallidum
#(the lenticular nucleus); hippocampus and amygdala
_PRIOR_V7_THALAMIC = [10, 49]
_PRIOR_V7_CAUDATE_NAC = [26, 11, 50, 58]
_PRIOR_V7_LENTICULAR_N = [12, 13, 51, 52]
_PRIOR_V7_HIPP_AMIG = [17, 18, 53, 54]


def streamlineCategoryPriors_v7(streamlines, atlas, inflateIter=2, inflatedAtlas=None, superficialClassification=None,
                                deislandAtlas=True,
                                temporoOccipitalSplit=False, verbose=False):
    """
    Claude note: endpoint-category prior, version 7, as MATLAB's
    bsc_streamlineCategoryPriors_v7(wbfg, atlas, inflateIter) (Dan Bullock's
    "plausibility segmentation"; wma_tools commits c9379a3, 31c202e,
    f2a1b2d, April-May 2020).  Each streamline is named after the regions its
    two endpoints are in, e.g. 'left_frontal_to_thalamic',
    'frontal_to_frontal_interHemi', 'right_occipital_to_occipital_ufiber',
    'left_frontal_to_temporal_implausible'.

    Differences from v6 (streamlineCategoryPriors), all from the MATLAB code:

    - the hemisphere prefix is 'left_' / 'right_' (v6: 'left' / 'right'),
      so the v6 recipes can't read v7 names;
    - "subcortical" is split into thalamic (10/49), caudateNAc (11/50,
      26/58), lenticularN (12/13, 51/52) and hippAmig (17/18, 53/54);
      'subcortical' keeps only 20/56 and 27/59;
    - the insula gains G_insular_short (118, v6: frontal) and the aseg
      insula (19/55, v6: subcortical);
    - MATLAB also lists the caudate (11/50) as ventricle (only used by the
      exclusion flag, so it can only block '_ufiber'); not here (Dan
      Bullock: "almost certainly a mistake");
    - '_implausible' (MATLAB spells it '_implausable'; renameMATLABTracts
      converts MATLAB output) is appended for streamlines that come near both
      the left and the right (inflated) hemisphere labels without being
      interhemispheric (one endpoint in each hemisphere);
    - superficial fibres come from segmentSuperficialFibers_v3 and both
      inflations from inflateAtlasLabels_v3_bsc;
    - a report of missing atlas labels (verbose) and a warning when more than
      5% of either endpoint set falls on an "unknown" label.

    Name construction, as in v6: each endpoint gets the first matching
    category (frontal / temporal / occipital / parietal / subcortical /
    thalamic / caudateNAc / lenticularN / hippAmig / spinal / insula /
    cerebellum / CorpusCallosum / ventricle / unlabeled / whiteMatter /
    pericallosal / OpticChi, in MATLAB's order, which differs slightly for
    the two endpoints), the name is the alphabetically sorted pair
    ``a_to_b``, with ``_interHemi`` or ``_ufiber`` appended, a side prefix
    for single hemisphere streamlines, 'MaskFailure' for interhemispheric
    superficial fibres, then ``_implausible``.  Names containing
    CorpusCallosum, unlabeled, OpticChi, ventricle, whiteMatter or
    pericallosal are pooled under that word.

    MATLAB defects and how they are handled (Claude, 2026-11-14):

    - the committed v7 (f2a1b2d) adds the undefined variable ``pericallosal``
      to the exclusion list, so it stops with an error before classifying
      anything; the evident intent is ``pericROI`` (v6 excluded it under that
      name, and the same commit made 'pericallosal' a pooled summary
      category), which is used here;
    - bsc_inflateLabels_v3's vote picks the smallest label (fixed, see
      inflateAtlasLabels_v3_bsc);
    - MATLAB's unknown-label printout lists the first groupcounts entries
      rather than the unknown labels; the report here lists the unknown
      labels;
    - the caudate is not listed as ventricle;
    - left as in MATLAB, pending a decision: S_oc-temp_med_and_Lingual (x162) and
      the fusiform gyrus (x121) listed as both temporal and occipital, so
      first-match makes them always temporal (temporoOccipitalSplit for an
      experimental landmark split).

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas.
    inflateIter : int, optional
        Inflation iterations for the endpoint labels; 0 uses the atlas as is.
        The default is 2.
    inflatedAtlas : nibabel.Nifti1Image, optional
        inflateAtlasLabels_v3_bsc(cleaned atlas, inflateIter), computed if
        not given.
    superficialClassification : dict, optional
        segmentSuperficialFibers_v3 output, computed if not given.
    deislandAtlas : bool, optional
        Island-clean the atlas first (prepareAtlas_bsc), as for every recipe.
        The default is True.
    temporoOccipitalSplit : bool, optional
        EXPERIMENTAL (landmark heuristics tuned on one subject, pending
        multi-subject validation).  Assign endpoints
        in the fusiform gyrus (x121) and the collateral / lingual sulcus
        (x162), which MATLAB lists as both temporal and occipital (so
        first-match makes them always temporal), to occipital or temporal by
        the Destrieux et al. 2010 ventral temporo-occipital line (see
        temporoOccipitalLandmarks), per hemisphere, from ``atlas``.  The
        default is False (MATLAB's lists).
    verbose : bool, optional
        Print MATLAB's missing-label and unknown-endpoint reports.  The
        default is False.

    Returns
    -------
    classification : dict
        wmc-style classification, one name per category.
    """
    if deislandAtlas:
        atlas = prepareAtlas_bsc(atlas)
    #Claude note: the label groups (both hemispheres where cortical)
    frontal, temporal = _bothSides(_PRIOR_V7_FRONTAL_BASE), _bothSides(_PRIOR_TEMPORAL_BASE)
    occipital, parietal = _bothSides(_PRIOR_OCCIPITAL_BASE), _bothSides(_PRIOR_PARIETAL_BASE)
    pericallosal = _bothSides([167])
    ventricle = _PRIOR_V7_VENTRICLE
    #Claude note: MATLAB's one-time report of the labels of each group that
    #are missing from the atlas
    if verbose:
        roiGroups = [('subcorticalROIS', _PRIOR_V7_SUBCORTICAL), ('spineROIS', _PRIOR_SPINE),
                     ('cerebellumROIS', _PRIOR_CEREBELLUM), ('ventricleROIS', ventricle), ('wmROIS', WM_LABELS),
                     ('ccROIS', _PRIOR_CC), ('unknownROIS', _PRIOR_UNKNOWN), ('OpticCROI', _PRIOR_OPTIC_CHIASM),
                     ('FrontalROIs', frontal), ('TemporalROIs', temporal), ('OccipitalROI', occipital),
                     ('ParietalROI', parietal), ('pericROI', pericallosal), ('insulaROI', _PRIOR_V7_INSULA),
                     ('thalamicROI', _PRIOR_V7_THALAMIC), ('caudateNAcROI', _PRIOR_V7_CAUDATE_NAC),
                     ('lenticularNROI', _PRIOR_V7_LENTICULAR_N), ('hippAmig', _PRIOR_V7_HIPP_AMIG)]
        labelsPresent = np.unique(_atlasData(atlas))
        for iGroupName, iGroupLabels in roiGroups:
            missing = np.asarray(iGroupLabels)[np.isin(iGroupLabels, labelsPresent, invert=True)]
            if len(missing) > 0:
                print('labels ' + str(sorted(set(missing.tolist()))) + ' missing for roi group ' + iGroupName)
    #Claude note: the inflated atlas, for the endpoint labels and the
    #hemisphere ROIs
    if inflatedAtlas is None:
        inflatedAtlas = inflateAtlasLabels_v3_bsc(atlas, inflateIter) if inflateIter > 0 else atlas
    #Claude note: the superficial fibres always use two inflation iterations
    if superficialClassification is None:
        superficialInflated = inflatedAtlas if inflateIter == 2 else inflateAtlasLabels_v3_bsc(atlas, 2)
        superficialClassification = segmentSuperficialFibers_v3(streamlines, atlas, superficialInflated)
    superficialBool = np.asarray(superficialClassification['index']) > 0
    #Claude note: streamlines coming near each hemisphere's labels anywhere
    #along their length (MATLAB: "to account for streamlines that cross and
    #then come back")
    leftROI = roiFromAtlasLabels_bsc(inflatedAtlas, _PRIOR_LEFT, 1)
    rightROI = roiFromAtlasLabels_bsc(inflatedAtlas, _PRIOR_RIGHT, 1)
    leftStreamsBool = segmentByROIs(streamlines, [leftROI], ['and'])
    rightStreamsBool = segmentByROIs(streamlines, [rightROI], ['and'])
    #Claude note: the endpoint labels
    first, last = _endpoints(streamlines)
    label1 = endpointAtlasLabels_bsc(inflatedAtlas, first)
    label2 = endpointAtlasLabels_bsc(inflatedAtlas, last)
    #Claude note: MATLAB's unknown-endpoint report and 5% warning, for each
    #endpoint set (MATLAB calls them the "RAS" and "LPI" groups)
    for iSetName, iLabels in (('RAS', label1), ('LPI', label2)):
        unknownBool = np.isin(iLabels, _PRIOR_UNKNOWN)
        if verbose:
            unknownLabels, unknownCounts = np.unique(iLabels[unknownBool], return_counts=True)
            for iLabel, iCount in zip(unknownLabels, unknownCounts):
                print(str(iCount) + ' endpoints for label ' + str(iLabel) + ' in ' + iSetName + ' group')
        if len(iLabels) > 0 and np.sum(unknownBool) / len(iLabels) > 0.05:
            warnings.warn('Proportion of unknown streamlines exceeds 5% for ' + iSetName + ' endpoints')
    #Claude note: the two endpoints are checked in slightly different orders
    #in MATLAB (only matters for labels in two lists)
    order1 = [('frontal', frontal), ('temporal', temporal), ('occipital', occipital), ('parietal', parietal),
              ('subcortical', _PRIOR_V7_SUBCORTICAL), ('thalamic', _PRIOR_V7_THALAMIC),
              ('caudateNAc', _PRIOR_V7_CAUDATE_NAC), ('lenticularN', _PRIOR_V7_LENTICULAR_N),
              ('hippAmig', _PRIOR_V7_HIPP_AMIG), ('spinal', _PRIOR_SPINE), ('insula', _PRIOR_V7_INSULA),
              ('cerebellum', _PRIOR_CEREBELLUM), ('CorpusCallosum', _PRIOR_CC), ('ventricle', ventricle),
              ('unlabeled', _PRIOR_UNKNOWN), ('whiteMatter', WM_LABELS), ('pericallosal', pericallosal),
              ('OpticChi', _PRIOR_OPTIC_CHIASM)]
    order2 = [('frontal', frontal), ('temporal', temporal), ('occipital', occipital), ('parietal', parietal),
              ('subcortical', _PRIOR_V7_SUBCORTICAL), ('thalamic', _PRIOR_V7_THALAMIC),
              ('caudateNAc', _PRIOR_V7_CAUDATE_NAC), ('lenticularN', _PRIOR_V7_LENTICULAR_N),
              ('hippAmig', _PRIOR_V7_HIPP_AMIG), ('spinal', _PRIOR_SPINE), ('insula', _PRIOR_V7_INSULA),
              ('cerebellum', _PRIOR_CEREBELLUM), ('pericallosal', pericallosal), ('ventricle', ventricle),
              ('unlabeled', _PRIOR_UNKNOWN), ('whiteMatter', WM_LABELS), ('CorpusCallosum', _PRIOR_CC),
              ('OpticChi', _PRIOR_OPTIC_CHIASM)]
    term1 = _terminationCategories(label1, order1)
    term2 = _terminationCategories(label2, order2)
    #Claude note: optional (experimental) Destrieux split of x121 / x162
    if temporoOccipitalSplit:
        landmarks = temporoOccipitalLandmarks(atlas)
        term1 = _splitTemporoOccipitalCategories(term1, label1, first, landmarks)
        term2 = _splitTemporoOccipitalCategories(term2, label2, last, landmarks)
    #Claude note: hemisphere and validity flags, as in MATLAB; the exclusion
    #list uses pericROI where the committed MATLAB names the undefined
    #variable `pericallosal`
    excludeSide = _PRIOR_UNKNOWN + _PRIOR_CC + _PRIOR_OPTIC_CHIASM + WM_LABELS + _PRIOR_SPINE + ventricle + pericallosal
    validSide = _PRIOR_LEFT + _PRIOR_RIGHT
    left1, left2 = np.isin(label1, _PRIOR_LEFT), np.isin(label2, _PRIOR_LEFT)
    right1, right2 = np.isin(label1, _PRIOR_RIGHT), np.isin(label2, _PRIOR_RIGHT)
    excludeBool = np.isin(label1, excludeSide) | np.isin(label2, excludeSide)
    includeBool = np.isin(label1, validSide) | np.isin(label2, validSide)
    validU = (np.isin(label1, GREY_MATTER_LABELS) | np.isin(label2, GREY_MATTER_LABELS)) & ~excludeBool
    leftBool, rightBool = left1 & left2, right1 & right2
    interHemiBool = (left2 & right1) | (right2 & left1)
    #Claude note: the v7 plausibility flag: near both hemispheres' labels,
    #but not one endpoint in each
    implausBool = ~interHemiBool & leftStreamsBool & rightStreamsBool
    singleLeft, singleRight = left1 ^ left2, right1 ^ right2
    interhemiFlag = interHemiBool & includeBool
    #Claude note: build each streamline's name
    streamNames = []
    for iStream in range(len(term1)):
        pairNames = sorted([term1[iStream], term2[iStream]])
        baseName = pairNames[0] + '_to_' + pairNames[1]
        streamName = baseName + '_interHemi' if interhemiFlag[iStream] else baseName
        if superficialBool[iStream] and validU[iStream]:
            streamName = baseName + '_ufiber'
        if (leftBool[iStream] or singleLeft[iStream]) and includeBool[iStream] and not interhemiFlag[iStream]:
            streamName = 'left_' + streamName
        elif (rightBool[iStream] or singleRight[iStream]) and includeBool[iStream] and not interhemiFlag[iStream]:
            streamName = 'right_' + streamName
        #Claude note: MATLAB's `~superficial==0` reads as `superficial~=0`
        if interhemiFlag[iStream] and validU[iStream] and superficialBool[iStream]:
            streamName = 'MaskFailure'
        if implausBool[iStream]:
            streamName = streamName + '_implausible'
        streamNames.append(streamName)
    #Claude note: pool the names containing a summary word (first such word
    #in MATLAB's list); this also pools their '_implausible' variants
    summarizeNames = ['CorpusCallosum', 'unlabeled', 'OpticChi', 'ventricle', 'whiteMatter', 'pericallosal']
    for iStream, iName in enumerate(streamNames):
        for iSummary in summarizeNames:
            if iSummary in iName:
                streamNames[iStream] = iSummary
                break
    #Claude note: the classification, names sorted
    uniqueNames, streamIndex = np.unique(np.asarray(streamNames, dtype=str), return_inverse=True)
    return {'names': [str(iName) for iName in uniqueNames], 'index': streamIndex.astype(np.int64) + 1}


# Claude note: ---------------------------------------------------------------
# tract families
# ---------------------------------------------------------------------------

#Claude note: a2009s base labels (add the side offset) used by bsc_segmentSubCortical_v2
_SUBCORT_FRONTAL = [124, 148, 118, 165, 101, 154, 105, 115, 154, 155, 170, 129, 146, 153,
                    164, 106, 116, 108, 131, 171, 112, 150, 104, 169, 114, 113, 103, 107, 117, 132, 139, 140, 142, 163]
_SUBCORT_TEMPORAL = [144, 134, 138, 137, 173, 174, 135, 175, 121, 151, 123, 162, 133, 149]
_SUBCORT_PARIETAL = [157, 127, 168, 136, 126, 125, 156, 128, 141, 172, 147, 109, 110, 130]
_MOTOR_BASE = [168, 128, 146, 129, 170, 103]
_THALAMUS = (10, 49)
_LENTICULAR = ((12, 13), (51, 52))
_PALLIDUM = (13, 52)


def _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas=True):
    """
    Claude note: clean the atlas (prepareAtlas_bsc, unless deislandAtlas is
    False) and compute the inflated atlas and category prior when they are
    not supplied (they are the expensive, shared inputs of every recipe;
    pass them in when running several families, computed from the cleaned
    atlas).
    """
    if deislandAtlas:
        atlas = prepareAtlas_bsc(atlas)
    if inflatedAtlas is None:
        inflatedAtlas = inflateAtlasLabels_bsc(atlas, 2)
    if categoryPrior is None:
        categoryPrior = streamlineCategoryPriors(streamlines, atlas, inflatedAtlas)
    if len(categoryPrior['index']) != len(streamlines):
        raise ValueError('categoryPrior does not match the streamlines')
    return atlas, categoryPrior, inflatedAtlas


def segmentThalamicRadiations(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: thalamic radiations, ported from MATLAB's
    bsc_segmentSubCortical_v2.  Per hemisphere: frontoThalamic,
    temporoThalamic, parietoThalamic, motorThalamic and spinoThalamic (in
    that order; a streamline meeting several takes the later one).

    All tracts end in the (inflated) thalamus and match an endpoint
    category prior; then:

    - frontoThalamic: an endpoint in frontal cortex, not in motor cortex;
      exactly one endpoint anterior of the thalamus; avoids planes behind
      the thalamus' anterior border (its inferior, superior and lateral
      borders), the lenticular nucleus' posterior border plane medial of the
      thalamus, and the superior frontal (112) border plane posterior of
      155's anterior border.  Prior: frontal_to_subcortical.
    - temporoThalamic: an endpoint in temporal cortex; exactly one endpoint
      below the pallidum; avoids the pericallosal (167) posterior plane, the
      pallidum's anterior plane lateral of the thalamus and the thalamus'
      superior plane.  Prior: subcortical_to_temporal.
    - parietoThalamic: an endpoint in parietal cortex; avoids the
      pallidum's anterior plane and planes anterior of / below the thalamus.
      Prior: parietal_to_subcortical.
    - motorThalamic: an endpoint in motor cortex (168 128 146 129 170 103,
      and 116 posterior of 170), not meeting the frontal criterion; avoids
      the thalamus' inferior plane and the lateral pallidum plane.  Prior:
      frontal_ or parietal_to_subcortical.
    - spinoThalamic: avoids the thalamus' superior and anterior planes.
      Prior: spinal_to_subcortical.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
        Whole brain tractogram, subject (RAS mm) space.
    atlas : nibabel.Nifti1Image
        aparc.a2009s+aseg atlas.
    categoryPrior : dict, optional
        streamlineCategoryPriors output, computed if not given.
    inflatedAtlas : nibabel.Nifti1Image, optional
        inflateAtlasLabels_bsc of the (cleaned) atlas, computed if not given.
    deislandAtlas : bool, optional
        Remove label islands from the atlas first (prepareAtlas_bsc), as
        brainlife's app-wmaSeg does.  The default is True.  If you pass
        categoryPrior / inflatedAtlas, compute them from the cleaned atlas.

    Returns
    -------
    classification : dict
        wmc-style classification with 10 tracts.
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        #Claude note: cortical and thalamic ROIs, from the inflated atlas
        frontalROI = roiFromAtlasLabels_bsc(inflatedAtlas, np.add(_SUBCORT_FRONTAL, sideNum))
        temporalROI = roiFromAtlasLabels_bsc(inflatedAtlas, np.add(_SUBCORT_TEMPORAL, sideNum))
        parietalROI = roiFromAtlasLabels_bsc(inflatedAtlas, np.add(_SUBCORT_PARIETAL, sideNum))
        thalamicROI = roiFromAtlasLabels_bsc(inflatedAtlas, _THALAMUS[iSide])
        #Claude note: motor ROI: motor labels plus 116 posterior of 170
        frontoMotorLimit = planeFromROI_bsc(170 + sideNum, 'anterior', inflatedAtlas)
        frontMedROI = roiFromAtlasLabels_bsc(inflatedAtlas, 116 + sideNum)
        frontoMedMotorROI = modifyROI_bsc(frontMedROI, frontoMotorLimit, 'posterior')
        motorROI = mergeROIs(frontoMedMotorROI, roiFromAtlasLabels_bsc(inflatedAtlas, np.add(_MOTOR_BASE, sideNum)))
        #Claude note: planes at the (inflated) thalamus' borders
        inferiorThalLimit = planeFromROI_bsc(thalamicROI, 'inferior', atlas)
        anteriorThalLimit = planeFromROI_bsc(thalamicROI, 'anterior', atlas)
        superiorThalLimit = planeFromROI_bsc(thalamicROI, 'superior', atlas)
        lateralThalamicLimit = planeFromROI_bsc(thalamicROI, 'lateral', atlas)
        #Claude note: planes at other structures' borders (raw atlas labels)
        posteriorLentiLimit = planeFromROI_bsc(list(_LENTICULAR[iSide]), 'posterior', atlas)
        antPalLimit = planeFromROI_bsc(_PALLIDUM[iSide], 'anterior', atlas)
        infPalLimit = planeFromROI_bsc(_PALLIDUM[iSide], 'inferior', atlas)
        superiorFrontalLimit = planeFromROI_bsc(112 + sideNum, 'superior', atlas)
        anteriorFrontalLimit = planeFromROI_bsc(155 + sideNum, 'anterior', atlas)
        anteriorSuperiorFrontalLimit = modifyROI_bsc(superiorFrontalLimit, anteriorFrontalLimit, 'posterior')
        postPeriC = planeFromROI_bsc(167 + sideNum, 'posterior', atlas)
        #Claude note: the exclusion planes, cut down from those
        posteriorInferiorExclude = modifyROI_bsc(inferiorThalLimit, anteriorThalLimit, 'posterior')
        posteriorSuperiorExclude = modifyROI_bsc(superiorThalLimit, anteriorThalLimit, 'posterior')
        posteriorLateralExclude = modifyROI_bsc(lateralThalamicLimit, anteriorThalLimit, 'posterior')
        anteriorLateralExclude = modifyROI_bsc(antPalLimit, lateralThalamicLimit, 'lateral')
        middleLateralExclude = modifyROI_bsc(posteriorLentiLimit, lateralThalamicLimit, 'medial')
        #Claude note: every tract ends in the thalamus, so all the other ROI
        #criteria only need checking on those streamlines
        thalamicIndexes = np.where(segmentByROIs(streamlines, [thalamicROI], ['endpoints']))[0]
        #Claude note: cortical endpoint criteria
        frontoMotorBool = _segmentSubset(streamlines, thalamicIndexes, [motorROI], ['endpoints'])
        frontalBool = _segmentSubset(streamlines, thalamicIndexes, [frontalROI, posteriorInferiorExclude, posteriorSuperiorExclude, posteriorLateralExclude],
                                     ['endpoints', 'not', 'not', 'not'])
        temporalBool = _segmentSubset(streamlines, thalamicIndexes, [temporalROI], ['endpoints'])
        parietalBool = _segmentSubset(streamlines, thalamicIndexes, [parietalROI, posteriorInferiorExclude, anteriorThalLimit],
                                      ['endpoints', 'not', 'not'])
        #Claude note: per-tract exclusions (the thalamic endpoint criterion
        #is already met by the whole subset)
        frontoThalamicBool = _segmentSubset(streamlines, thalamicIndexes, [posteriorLateralExclude, posteriorInferiorExclude, middleLateralExclude, anteriorSuperiorFrontalLimit],
                                            ['not', 'not', 'not', 'not'])
        temporoThalamicBool = _segmentSubset(streamlines, thalamicIndexes, [postPeriC, anteriorLateralExclude, superiorThalLimit],
                                             ['not', 'not', 'not'])
        parietoThalamicBool = _segmentSubset(streamlines, thalamicIndexes, [antPalLimit, inferiorThalLimit], ['not', 'not'])
        motorThalamicBool = _segmentSubset(streamlines, thalamicIndexes, [inferiorThalLimit, anteriorLateralExclude], ['not', 'not'])
        spinoThalamicBool = _segmentSubset(streamlines, thalamicIndexes, [superiorThalLimit, anteriorThalLimit], ['not', 'not'])
        #Claude note: endpoint position criteria
        frontoEndpointBool = endpointCriteria_bsc(streamlines, anteriorThalLimit, 'anterior', 'one')
        lateralThalEndpointBool = endpointCriteria_bsc(streamlines, infPalLimit, 'inferior', 'one')
        #Claude note: the category priors
        frontalSubcorticalBool = classificationBoolByName(categoryPrior, side + 'frontal_to_subcortical')
        parietalSubcorticalBool = classificationBoolByName(categoryPrior, side + 'parietal_to_subcortical')
        temporalSubcorticalBool = classificationBoolByName(categoryPrior, side + 'subcortical_to_temporal')
        spinalSubcorticalBool = classificationBoolByName(categoryPrior, side + 'spinal_to_subcortical')
        motorBoolPrior = parietalSubcorticalBool | frontalSubcorticalBool
        #Claude note: the tracts, in MATLAB's order (later ones win)
        classification = addTractToClassification(classification, side + 'frontoThalamic',
                                                  frontoThalamicBool, frontoEndpointBool, frontalBool, ~frontoMotorBool, frontalSubcorticalBool)
        classification = addTractToClassification(classification, side + 'temporoThalamic',
                                                  temporoThalamicBool, temporalBool, temporalSubcorticalBool, lateralThalEndpointBool)
        classification = addTractToClassification(classification, side + 'parietoThalamic',
                                                  parietoThalamicBool, parietalBool, parietalSubcorticalBool)
        classification = addTractToClassification(classification, side + 'motorThalamic',
                                                  motorThalamicBool, ~frontalBool, frontoMotorBool, motorBoolPrior)
        classification = addTractToClassification(classification, side + 'spinoThalamic',
                                                  spinoThalamicBool, spinalSubcorticalBool)
    return classification


# Claude note: ---------------------------------------------------------------
# more MATLAB criteria (used by the families below)
# ---------------------------------------------------------------------------

def _roiTolerance(roiNifti, tol=VISTASOFT_MIN_DIST):
    """
    Claude note: vistasoft's distance, but never below the voxel's
    centre-to-corner distance (as in segmentByROIs).
    """
    return max(tol, dist_to_corner(roiNifti.affine))


def tractByEndpointROIs_bsc(streamlines, roiA, roiB, tol=ENDPOINT_PAIR_MIN_DIST):
    """
    Claude note: streamlines with one endpoint near roiA and the other near
    roiB (either way round), as MATLAB's bsc_tractByEndpointROIs: distance to
    the nearest ROI voxel centre strictly below tol.  MATLAB compares the
    squared distance with 0.87, so the limit is sqrt(0.87) = 0.933 mm.
    Empty ROIs give no streamlines.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    roiA, roiB : nibabel.Nifti1Image
        The two endpoint ROIs (may be the same ROI: both endpoints in it).
    tol : float, optional
        Distance in mm.  The default is sqrt(0.87) = 0.933 (raised to the
        voxel's centre-to-corner distance on coarser grids).

    Returns
    -------
    keepBool : np.ndarray of bool
    """
    coordsA, coordsB = _maskVoxelCoords(roiA), _maskVoxelCoords(roiB)
    if len(coordsA) == 0 or len(coordsB) == 0:
        return np.zeros(len(streamlines), dtype=bool)
    first, last = _endpoints(streamlines)
    tolA, tolB = max(tol, dist_to_corner(roiA.affine)), max(tol, dist_to_corner(roiB.affine))
    #Claude note: nearest-voxel distances of both endpoints to both ROIs
    treeA, treeB = cKDTree(coordsA), cKDTree(coordsB)
    nearA1 = treeA.query(first, distance_upper_bound=tolA)[0] < tolA
    nearA2 = treeA.query(last, distance_upper_bound=tolA)[0] < tolA
    nearB1 = treeB.query(first, distance_upper_bound=tolB)[0] < tolB
    nearB2 = treeB.query(last, distance_upper_bound=tolB)[0] < tolB
    return (nearA1 & nearB2) | (nearB1 & nearA2)


def midpointCriteria_bsc(streamlines, *referencesAndLocations):
    """
    Claude note: midpoint node strictly beyond one or more references, as
    MATLAB's bsc_applyMidpointCriteria(fg, ref1, loc1, ref2, loc2, ...)
    (all criteria ANDed).  The midpoint is node floor(n/2) (1-based).

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    *referencesAndLocations
        Pairs of (planar ROI nifti or coordinate, location), location as in
        endpointCriteria_bsc.

    Returns
    -------
    criteriaBool : np.ndarray of bool
    """
    midpoints = _midpoints(streamlines)
    criteriaBool = np.ones(len(midpoints), dtype=bool)
    for iPair in range(0, len(referencesAndLocations), 2):
        reference, location = referencesAndLocations[iPair], referencesAndLocations[iPair + 1]
        refCoord = roiMeanCoord(reference) if isinstance(reference, nib.Nifti1Image) else np.asarray(reference, dtype=float)
        dimension, valence = _locationAxis(refCoord, location)
        if valence > 0:
            criteriaBool &= midpoints[:, dimension] > refCoord[dimension]
        else:
            criteriaBool &= midpoints[:, dimension] < refCoord[dimension]
    return criteriaBool


def _locationAxis(refCoord, location):
    """
    Claude note: MATLAB's (axis, direction) for a location term; medial /
    lateral take the hemisphere from the sign of the reference x.
    """
    leftFlag = refCoord[0] < 0
    locationDict = {'top': (2, 1), 'superior': (2, 1), 'bottom': (2, -1), 'inferior': (2, -1),
                    'anterior': (1, 1), 'posterior': (1, -1),
                    'medial': (0, 1 if leftFlag else -1), 'lateral': (0, -1 if leftFlag else 1)}
    if location.lower() not in locationDict:
        raise ValueError('location ' + str(location) + ' not understood')
    return locationDict[location.lower()]


def endpointAtlasCriteria_bsc(streamlines, inflatedAtlas, labels, criteria, endpointLabels=None):
    """
    Claude note: endpoint atlas labels in a list, as MATLAB's
    bsc_endpointAtlasCriteria (which labels endpoints in the inflated atlas,
    like the category prior).

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
    inflatedAtlas : nibabel.Nifti1Image
        inflateAtlasLabels_bsc of the recipe's atlas.
    labels : list of int
    criteria : str
        'one', 'both', 'neither' or 'either'.
    endpointLabels : tuple of two np.ndarray, optional
        Precomputed endpointAtlasLabels_bsc of the first and last nodes.

    Returns
    -------
    criteriaBool : np.ndarray of bool
    """
    if endpointLabels is None:
        first, last = _endpoints(streamlines)
        endpointLabels = (endpointAtlasLabels_bsc(inflatedAtlas, first), endpointAtlasLabels_bsc(inflatedAtlas, last))
    nIn = np.isin(endpointLabels[0], labels).astype(int) + np.isin(endpointLabels[1], labels).astype(int)
    return {'one': nIn == 1, 'both': nIn == 2, 'neither': nIn == 0, 'either': nIn > 0}[criteria.lower()]


def intersectROIs(*rois):
    """
    Claude note: voxels in all the masks (same grid), as MATLAB's
    bsc_intersectROIs / bsc_MultiIntersectROIs.

    Parameters
    ----------
    *rois : nibabel.Nifti1Image

    Returns
    -------
    intersectionNifti : nibabel.Nifti1Image
    """
    intersection = np.ones(rois[0].shape, dtype=bool)
    for iRoi in rois:
        intersection &= np.asanyarray(iRoi.dataobj) > 0
    return _maskNifti(intersection, rois[0])


def midlinePlane_bsc(atlas):
    """
    Claude note: the x=0 plane of MATLAB's old bsc_makePlanarROI(atlas, 0,
    'x') (used by bsc_segmentAslant).  Unlike bsc_makePlanarROI_v3 it does
    not span the image: its y / z points are ``pixdim*j - |qoffset|`` for
    j = 1..n along the 2nd / 3rd *array* axes, whatever their orientation.
    For the FreeSurfer conformed (LIA) grid that stops at z = 109 mm.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image

    Returns
    -------
    planeNifti : nibabel.Nifti1Image
        The plane's points marked on the atlas grid.
    """
    zooms = np.asarray(atlas.header.get_zooms()[:3])
    offsets = np.abs(atlas.affine[:3, 3])
    yCoords = zooms[1] * np.arange(1, atlas.shape[1] + 1) - offsets[1]
    zCoords = zooms[2] * np.arange(1, atlas.shape[2] + 1) - offsets[2]
    yGrid, zGrid = np.meshgrid(yCoords, zCoords, indexing='ij')
    points = np.column_stack([np.zeros(yGrid.size), yGrid.ravel(), zGrid.ravel()])
    #Claude note: mark the voxels holding those points, dropping any outside
    voxelIndexes = np.round(nib.affines.apply_affine(np.linalg.inv(atlas.affine), points)).astype(int)
    inside = np.all((voxelIndexes >= 0) & (voxelIndexes < np.asarray(atlas.shape[:3])), axis=1)
    planeData = np.zeros(atlas.shape[:3], dtype=np.uint8)
    planeData[tuple(voxelIndexes[inside].T)] = 1
    return _maskNifti(planeData, atlas)


def _priorBool(categoryPrior, *names):
    """
    Claude note: streamlines whose category prior is any of the names.
    """
    outBool = np.zeros(len(categoryPrior['index']), dtype=bool)
    for iName in names:
        outBool |= classificationBoolByName(categoryPrior, iName)
    return outBool


def _gate(streamlines, gateBool):
    """
    Claude note: a function evaluating segmentByROIs only on the streamlines
    in gateBool (False elsewhere).  Each family gates on the union of the
    category priors its tracts require: every tract ANDs one of them, so no
    tract can differ, and the ROI tests run on far fewer streamlines.
    """
    gateIndexes = np.where(gateBool)[0]
    return lambda rois, operations: _segmentSubset(streamlines, gateIndexes, rois, operations)


def dropEmptyTracts(classification):
    """
    Claude note: remove names with no streamlines (as MATLAB's
    wma_resortClassificationStruc does at the end of the pipeline),
    renumbering the index.

    Parameters
    ----------
    classification : dict

    Returns
    -------
    classification : dict
    """
    index = np.asarray(classification['index'])
    present = np.isin(np.arange(1, len(classification['names']) + 1), index)
    newNumber = np.zeros(len(classification['names']) + 1, dtype=np.int64)
    newNumber[1:][present] = np.arange(1, np.sum(present) + 1)
    return {'names': [iName for iName, iPresent in zip(classification['names'], present) if iPresent],
            'index': newNumber[index]}


# Claude note: ---------------------------------------------------------------
# remaining tract families (MATLAB order of wma_segMajTracks_BL_v2)
# ---------------------------------------------------------------------------

_AMYGDALA = (18, 54)
_HIPPOCAMPUS = (17, 53)
_PUTAMEN = (12, 51)
_LATERAL_VENTRICLE = (4, 43)
_CEREBELLUM = ((7, 8), (46, 47))


def segmentCingulum(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: cingulum, ported from MATLAB's bsc_segmentCingulum_v3.
    Per hemisphere: streamlines crossing the plane at the anterior border of
    CC 252 medial of the pericallosal sulcus' (167) lateral border, avoiding
    the same plane above the 156 superior border and the thalamus' top
    plane between the CC's posterior (251) and anterior (254) borders.
    Prior: frontal_to_parietal or frontal_to_temporal.

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
        'leftcingulum', 'rightcingulum'.
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        priorBool = _priorBool(categoryPrior, side + 'frontal_to_parietal', side + 'frontal_to_temporal')
        seg = _gate(streamlines, priorBool)
        #Claude note: the "inside the CC" exclusion: thalamus top plane between the CC's borders
        thalTop = planeFromROI_bsc(_THALAMUS[iSide], 'superior', atlas)
        ccPostLimit = planeFromROI_bsc(251, 'posterior', atlas)
        ccAntLimit = planeFromROI_bsc(254, 'anterior', atlas)
        ccInterior2 = modifyROI_bsc(modifyROI_bsc(thalTop, ccPostLimit, 'anterior'), ccAntLimit, 'posterior')
        #Claude note: the inclusion and exclusion parts of the CC 252 anterior plane
        periCSupLim = planeFromROI_bsc(156 + sideNum, 'superior', atlas)
        periCLatLim = planeFromROI_bsc(167 + sideNum, 'lateral', atlas)
        ccMidAnt = planeFromROI_bsc(252, 'anterior', atlas)
        cingIncPlane = modifyROI_bsc(ccMidAnt, periCLatLim, 'medial')
        cingSupExcPlane = modifyROI_bsc(ccMidAnt, periCSupLim, 'superior')
        cingSegBool = seg([cingIncPlane, cingSupExcPlane, ccInterior2], ['and', 'not', 'not'])
        classification = addTractToClassification(classification, side + 'cingulum', cingSegBool, priorBool)
    return classification


def segmentAntPostTracts(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: anterior-posterior association tracts, ported from MATLAB's
    bsc_segmentAntPostTracts_v3: cingulum (segmentCingulum, first), then per
    hemisphere Uncinate, IFOF, Arc, SLF1And2 and SLF3 (later tracts win).

    - Uncinate: frontal_to_temporal(_ufiber) prior; avoids the thalamus' top
      and posterior planes and the amygdala's posterior plane; both
      endpoints below the thalamus' top and anterior of the amygdala.
    - IFOF: frontal_to_occipital prior; crosses the 173-anterior plane below
      the CC (255) bottom; avoids a CC carve-out (thalamus top plane between
      the CC's borders, and the CC border planes above the thalamus).
    - Arc: frontal_to_temporal(_ufiber) prior; crosses the lateral fissure
      (141) bottom plane behind the CC and the insula (150) posterior plane
      above 133; avoids 108 (grown by 2); not cingulum.
    - SLF1And2: frontal_to_parietal prior; crosses that insula plane; avoids
      the CC carve-out behind CC 252's anterior border; not cingulum, not the
      IFOF ROI criterion, no endpoint on the medial wall.
    - SLF3: frontal_to_parietal prior; an endpoint in 112 and 150 (each
      grown by 9) overlapping; avoids the fissure plane and the pallidum's
      anterior plane above 155; not the IFOF ROI criterion, no medial-wall
      endpoint.

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
        12 tracts.
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    #Claude note: MATLAB starts from the cingulum classification
    classification = segmentCingulum(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas=False)
    cingulumBool = classificationBoolByName(classification, 'leftcingulum') | classificationBoolByName(classification, 'rightcingulum')
    #Claude note: endpoint labels for the medial-wall criterion (same as the prior's)
    first, last = _endpoints(streamlines)
    endpointLabels = (endpointAtlasLabels_bsc(inflatedAtlas, first), endpointAtlasLabels_bsc(inflatedAtlas, last))
    #Claude note: shared CC planes
    ccPostLimit = planeFromROI_bsc(251, 'posterior', atlas)
    ccAntLimit = planeFromROI_bsc(255, 'anterior', atlas)
    infCCLimit = planeFromROI_bsc(255, 'inferior', atlas)
    ccMidLimit = planeFromROI_bsc(252, 'anterior', atlas)
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        #Claude note: the priors, and the gate (their union)
        frontoTemporalBool = _priorBool(categoryPrior, side + 'frontal_to_temporal', side + 'frontal_to_temporal_ufiber')
        frontoOccipitalBool = _priorBool(categoryPrior, side + 'frontal_to_occipital')
        parietoFrontalBool = _priorBool(categoryPrior, side + 'frontal_to_parietal')
        seg = _gate(streamlines, frontoTemporalBool | frontoOccipitalBool | parietoFrontalBool)
        #Claude note: uncinate
        thalTop = planeFromROI_bsc(_THALAMUS[iSide], 'superior', atlas)
        thalPost = planeFromROI_bsc(_THALAMUS[iSide], 'posterior', atlas)
        amigPost = planeFromROI_bsc(_AMYGDALA[iSide], 'posterior', atlas)
        uncEndpointBool = endpointCriteria_bsc(streamlines, thalTop, 'inferior', 'both') & endpointCriteria_bsc(streamlines, amigPost, 'anterior', 'both')
        uncSegBool = seg([thalTop, amigPost, thalPost], ['not', 'not', 'not'])
        classification = addTractToClassification(classification, side + 'Uncinate', frontoTemporalBool, uncSegBool, uncEndpointBool)
        #Claude note: IFOF, with the CC carve-out
        postCCtopThal = modifyROI_bsc(ccPostLimit, thalTop, 'superior')
        ccInterior2 = modifyROI_bsc(modifyROI_bsc(thalTop, ccPostLimit, 'anterior'), ccAntLimit, 'posterior')
        antCCtopThal = modifyROI_bsc(ccAntLimit, thalTop, 'superior')
        ccCarveOut = mergeROIs(postCCtopThal, ccInterior2, antCCtopThal)
        infTempROI = modifyROI_bsc(planeFromROI_bsc(173 + sideNum, 'anterior', atlas), infCCLimit, 'inferior')
        ifofBool = seg([infTempROI, ccCarveOut], ['and', 'not'])
        classification = addTractToClassification(classification, side + 'IFOF', frontoOccipitalBool, ifofBool)
        #Claude note: arcuate
        postCingNot = roiFromAtlasLabels_bsc(atlas, 108 + sideNum, 5)
        postLatFisInf = modifyROI_bsc(planeFromROI_bsc(141 + sideNum, 'inferior', atlas), ccPostLimit, 'posterior')
        topArcAnd = modifyROI_bsc(planeFromROI_bsc(150 + sideNum, 'posterior', atlas), planeFromROI_bsc(133 + sideNum, 'superior', atlas), 'superior')
        arcBool = seg([postLatFisInf, topArcAnd, postCingNot], ['and', 'and', 'not'])
        classification = addTractToClassification(classification, side + 'Arc', frontoTemporalBool, arcBool, ~cingulumBool)
        #Claude note: SLF
        slf12exclude = modifyROI_bsc(ccInterior2, ccMidLimit, 'posterior')
        interiorWallBool = endpointAtlasCriteria_bsc(streamlines, inflatedAtlas, list(np.add([116, 109, 108, 107, 167, 147, 172, 130, 110, 106], sideNum)), 'either', endpointLabels)
        slf3Intersection = intersectROIs(roiFromAtlasLabels_bsc(atlas, 112 + sideNum, 19), roiFromAtlasLabels_bsc(atlas, 150 + sideNum, 19))
        slf3Fix = modifyROI_bsc(planeFromROI_bsc(_PALLIDUM[iSide], 'anterior', atlas), planeFromROI_bsc(155 + sideNum, 'inferior', atlas), 'superior')
        slf12SegBool = seg([slf12exclude, topArcAnd], ['not', 'and'])
        slf3SegBool = seg([slf3Intersection, postLatFisInf, slf3Fix], ['endpoints', 'not', 'not'])
        slf12Bool = slf12SegBool & parietoFrontalBool & ~cingulumBool & ~ifofBool & ~interiorWallBool
        slf3Bool = slf3SegBool & parietoFrontalBool & ~ifofBool & ~interiorWallBool
        classification = addTractToClassification(classification, side + 'SLF1And2', slf12Bool)
        classification = addTractToClassification(classification, side + 'SLF3', slf3Bool)
    return classification


def segmentCorpusCallosum(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True, experimental=True):
    """
    Claude note: callosal tracts, ported from MATLAB's
    bsc_segmentCorpusCallosum_v3 (experimental=1, as wma_segMajTracks_BL_v2
    runs it): forcepsMinor, forcepsMajor, parietalCC, middleFrontalCC and
    (experimental) anterioFrontalCC; bilateral.  Streamlines ending in the
    CC (grown by 6) or ventricles (grown by 3) are excluded from
    forcepsMinor.  Priors: frontal_ / occipital_ / parietal_to_*_interHemi.

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.
    experimental : bool, optional
        Include anterioFrontalCC.  The default is True.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    frontalCCBool = _priorBool(categoryPrior, 'frontal_to_frontal_interHemi')
    occipitalCCBool = _priorBool(categoryPrior, 'occipital_to_occipital_interHemi')
    parietalInterhemiBool = _priorBool(categoryPrior, 'parietal_to_parietal_interHemi')
    seg = _gate(streamlines, frontalCCBool | occipitalCCBool | parietalInterhemiBool)
    both = lambda base: [base + 11000, base + 12000]
    #Claude note: exclusions: ending in the CC or the ventricles
    excludeCCBool = seg([roiFromAtlasLabels_bsc(atlas, [255, 254, 253, 252, 251], 13)], ['endpoints'])
    excludeVentriBool = seg([roiFromAtlasLabels_bsc(atlas, list(_LATERAL_VENTRICLE), 7)], ['endpoints'])
    #Claude note: forceps minor
    anteriorCCROI = roiFromAtlasLabels_bsc(atlas, 255, 0)
    antMidCCPlaneAnt = planeFromROI_bsc(254, 'anterior', atlas)
    periCROI = planeFromROI_bsc(both(167), 'anterior', atlas)
    frontMiddleTopPlane = planeFromROI_bsc(both(154), 'superior', atlas)
    olfTop = planeFromROI_bsc(both(164), 'superior', atlas)
    forcepsMinorEndBool = (endpointCriteria_bsc(streamlines, periCROI, 'anterior', 'both')
                           & endpointCriteria_bsc(streamlines, olfTop, 'superior', 'both')
                           & endpointCriteria_bsc(streamlines, frontMiddleTopPlane, 'inferior', 'both'))
    anteriorCCBool = seg([anteriorCCROI, antMidCCPlaneAnt, periCROI], ['and', 'not', 'and'])
    forcepsMinorBool = forcepsMinorEndBool & anteriorCCBool & ~excludeCCBool & ~excludeVentriBool & frontalCCBool
    classification = addTractToClassification(classification, 'forcepsMinor', forcepsMinorBool)
    #Claude note: forceps major: midpoint anterior of the 161 posterior border
    latTempPostLimit = planeFromROI_bsc(both(161), 'posterior', atlas)
    forcepsMajorBool = midpointCriteria_bsc(streamlines, latTempPostLimit, 'anterior') & occipitalCCBool
    classification = addTractToClassification(classification, 'forcepsMajor', forcepsMajorBool)
    #Claude note: parietal CC
    latPutLeft = planeFromROI_bsc(12, 'lateral', atlas)
    latPutRight = planeFromROI_bsc(51, 'lateral', atlas)
    parietalEndpointsBool = endpointCriteria_bsc(streamlines, planeFromROI_bsc(252, 'superior', atlas), 'superior', 'both')
    parietalSegBool = seg([planeFromROI_bsc(253, 'posterior', atlas), planeFromROI_bsc(251, 'inferior', atlas), latPutLeft, latPutRight],
                          ['not', 'not', 'not', 'not'])
    parietalCCBool = parietalInterhemiBool & ~forcepsMinorBool & ~forcepsMajorBool & parietalSegBool & parietalEndpointsBool
    classification = addTractToClassification(classification, 'parietalCC', parietalCCBool)
    #Claude note: middle / anterior frontal CC
    frontoSeparate = planeFromROI_bsc(both(118), 'anterior', atlas)
    posteriorFrontoCCLimit = planeFromROI_bsc(255, 'posterior', atlas)
    middleFrontalNot = planeFromROI_bsc(254, 'inferior', atlas)
    anterioFrontalNot = modifyROI_bsc(middleFrontalNot, planeFromROI_bsc(254, 'anterior', atlas), 'posterior')
    anterioFrontalSegBool = seg([latPutLeft, latPutRight, frontoSeparate, anterioFrontalNot, posteriorFrontoCCLimit], ['not', 'not', 'and', 'not', 'not'])
    middleFrontalSegBool = seg([latPutLeft, latPutRight, frontoSeparate, middleFrontalNot], ['not', 'not', 'not', 'not'])
    middleFrontalEndpointBool = endpointCriteria_bsc(streamlines, frontoSeparate, 'anterior', 'both')
    notOtherCC = frontalCCBool & ~forcepsMinorBool & ~forcepsMajorBool & ~parietalCCBool
    middleFrontalBool = notOtherCC & middleFrontalSegBool & ~middleFrontalEndpointBool
    anterioFrontalBool = notOtherCC & ~middleFrontalBool & anterioFrontalSegBool & middleFrontalEndpointBool
    classification = addTractToClassification(classification, 'middleFrontalCC', middleFrontalBool)
    if experimental:
        classification = addTractToClassification(classification, 'anterioFrontalCC', anterioFrontalBool)
    return classification


def segmentAslant(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: frontal aslant, ported from MATLAB's bsc_segmentAslant.
    Per hemisphere: one endpoint in 116 (grown by 3), the other in 114/112
    lateral of 104's medial border; avoids the pericallosal (167) and 103
    anterior planes and the midline; midpoint posterior of 167's anterior
    border.  Prior: frontal_to_frontal(_ufiber).

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    notHemi = midlinePlane_bsc(atlas)
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        frontoFrontalBool = _priorBool(categoryPrior, side + 'frontal_to_frontal', side + 'frontal_to_frontal_ufiber')
        seg = _gate(streamlines, frontoFrontalBool)
        superiorROI = roiFromAtlasLabels_bsc(atlas, 116 + sideNum, 7)
        lateralROI = modifyROI_bsc(roiFromAtlasLabels_bsc(atlas, [114 + sideNum, 112 + sideNum], 1),
                                   planeFromROI_bsc(104 + sideNum, 'medial', atlas), 'lateral')
        keepBool = tractByEndpointROIs_bsc(streamlines, superiorROI, lateralROI)
        antPeriCall = planeFromROI_bsc(167 + sideNum, 'anterior', atlas)
        antParSup = planeFromROI_bsc(103 + sideNum, 'anterior', atlas)
        preventAnteriorBool = midpointCriteria_bsc(streamlines, antPeriCall, 'posterior')
        boundedBool = seg([antPeriCall, antParSup, notHemi], ['not', 'not', 'not'])
        classification = addTractToClassification(classification, side + 'Aslant', keepBool, boundedBool, frontoFrontalBool, preventAnteriorBool)
    return classification


def segmentMdLF_ILF(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: ILF and middle longitudinal fasciculus, ported from MATLAB's
    bsc_segmentMdLF_ILF_v4.  Per hemisphere:

    - ILF: occipital_to_temporal prior; exactly one endpoint anterior of the
      collateral / lingual sulcus (162; Destrieux 2010 Table 1: medial
      occipito-temporal sulcus and lingual sulcus, not the lingual gyrus,
      which is 122) and one posterior of 110; avoids the insula / 133
      plane and 109/110 (inflated, grown by 2).
    - MDLFang / MDLFspl: one endpoint in parietal {157 127 168 136 126 125}
      (inflated, grown by 2), the other in lateral temporal {134 144 174 135}
      (inflated, grown by 4) anterior of the amygdala's posterior border;
      parietal_to_temporal prior; avoids subcortical structures and the
      172-anterior plane above the insula; midpoint not anterior / above /
      behind those limits; split by passing (spl) or not (ang) the overlap
      of 130 and 157 grown by 13.

    The three tracts are reconciled in the order ILF, MDLFang, MDLFspl.

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    subCortIDs = ([5, 12, 13, 17, 18], [44, 51, 52, 53, 54])
    nStreams = len(streamlines)
    classificationILF, classificationAng, classificationSpl = emptyClassification(nStreams), emptyClassification(nStreams), emptyClassification(nStreams)
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        occipitoTemporalBool = _priorBool(categoryPrior, side + 'occipital_to_temporal')
        temporoParietalBool = _priorBool(categoryPrior, side + 'parietal_to_temporal')
        seg = _gate(streamlines, occipitoTemporalBool | temporoParietalBool)
        add = lambda base: list(np.add(base, sideNum))
        #Claude note: endpoint ROIs
        mergedParietalROI = roiFromAtlasLabels_bsc(inflatedAtlas, add([157, 127, 168, 136, 126, 125]), 5)
        amygdalaPost = planeFromROI_bsc(_AMYGDALA[iSide], 'posterior', atlas)
        mergedLatTempROI = modifyROI_bsc(roiFromAtlasLabels_bsc(inflatedAtlas, add([134, 144, 174, 135]), 9), amygdalaPost, 'anterior')
        #Claude note: ILF
        ilfAntEndpointBool = endpointCriteria_bsc(streamlines, planeFromROI_bsc(162 + sideNum, 'anterior', atlas), 'anterior', 'one')
        ilfPostEndpointBool = endpointCriteria_bsc(streamlines, planeFromROI_bsc(110 + sideNum, 'posterior', atlas), 'posterior', 'one')
        topArcAnd = modifyROI_bsc(planeFromROI_bsc(150 + sideNum, 'posterior', atlas), planeFromROI_bsc(133 + sideNum, 'superior', atlas), 'superior')
        cingRemoveROI = roiFromAtlasLabels_bsc(inflatedAtlas, add([109, 110]), 5)
        ilfSegBool = seg([topArcAnd, cingRemoveROI], ['not', 'not'])
        classificationILF = addTractToClassification(classificationILF, side + 'ILF', occipitoTemporalBool, ilfAntEndpointBool, ilfPostEndpointBool, ilfSegBool)
        #Claude note: MdLF
        mdlfEndpointBool = tractByEndpointROIs_bsc(streamlines, mergedParietalROI, mergedLatTempROI)
        subCortROI = roiFromAtlasLabels_bsc(atlas, subCortIDs[iSide], 1)
        anteriorMDLFLimit = planeFromROI_bsc(172 + sideNum, 'anterior', atlas)
        mdlfFrontRm = modifyROI_bsc(anteriorMDLFLimit, planeFromROI_bsc(150 + sideNum, 'superior', atlas), 'superior')
        removeMDLFBool = seg([subCortROI, mdlfFrontRm], ['not', 'not'])
        splWM = intersectROIs(roiFromAtlasLabels_bsc(atlas, 130 + sideNum, 27), roiFromAtlasLabels_bsc(atlas, 157 + sideNum, 27))
        splBool = seg([splWM], ['and'])
        subcortTop = planeFromROI_bsc(subCortROI, 'superior', atlas)
        badMidpointsA = midpointCriteria_bsc(streamlines, anteriorMDLFLimit, 'anterior', subcortTop, 'superior', amygdalaPost, 'posterior')
        badMidpointsB = midpointCriteria_bsc(streamlines, amygdalaPost, 'anterior')
        common = mdlfEndpointBool & removeMDLFBool & ~badMidpointsA & ~badMidpointsB & temporoParietalBool
        classificationAng = addTractToClassification(classificationAng, side + 'MDLFang', common, ~splBool)
        classificationSpl = addTractToClassification(classificationSpl, side + 'MDLFspl', common, splBool)
    classification = reconcileClassifications(classificationILF, classificationAng)
    return reconcileClassifications(classification, classificationSpl)


def segmentpArcTPC(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: posterior arcuate and temporo-parietal connection, ported
    from MATLAB's bsc_segpArcTPC (no category prior).  Per hemisphere: one
    endpoint in parietal {157 127 168 136 126 125}, the other in temporal
    {121 161 137 162 138 173} posterior of the putamen; crossing the planes
    at the 156 and 136 inferior borders; avoiding the 116 posterior plane
    above 156.  TPC passes the overlap of 130 and 157 (each grown by 13);
    pArc does not, and avoids the collateral / lingual sulcus (162; not the
    fusiform, which is 121) top plane behind the
    ventricle.  The tracts are reconciled in the order pArc, TPC.

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations (categoryPrior and inflatedAtlas are
        not used; accepted for a uniform signature).

    Returns
    -------
    classification : dict
    """
    if deislandAtlas:
        atlas = prepareAtlas_bsc(atlas)
    nStreams = len(streamlines)
    classificationpArc, classificationTPC = emptyClassification(nStreams), emptyClassification(nStreams)
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        add = lambda base: list(np.add(base, sideNum))
        mergedParietalROI = roiFromAtlasLabels_bsc(atlas, add([157, 127, 168, 136, 126, 125]), 1)
        mergedTemporalROI = modifyROI_bsc(roiFromAtlasLabels_bsc(atlas, add([121, 161, 137, 162, 138, 173]), 1),
                                          planeFromROI_bsc(_PUTAMEN[iSide], 'posterior', atlas), 'posterior')
        #Claude note: every tract needs the endpoint pair, so gate on it
        endpointBool = tractByEndpointROIs_bsc(streamlines, mergedParietalROI, mergedTemporalROI)
        seg = _gate(streamlines, endpointBool)
        supPlane = planeFromROI_bsc(156 + sideNum, 'inferior', atlas)
        infPlane = planeFromROI_bsc(136 + sideNum, 'inferior', atlas)
        frontPost = modifyROI_bsc(planeFromROI_bsc(116 + sideNum, 'posterior', atlas), supPlane, 'superior')
        vertBool = seg([supPlane, infPlane, frontPost], ['and', 'and', 'not'])
        tpcWM = intersectROIs(roiFromAtlasLabels_bsc(atlas, 130 + sideNum, 27), roiFromAtlasLabels_bsc(atlas, 157 + sideNum, 27))
        fusTopPost = modifyROI_bsc(planeFromROI_bsc(162 + sideNum, 'superior', atlas), planeFromROI_bsc(_LATERAL_VENTRICLE[iSide], 'posterior', atlas), 'posterior')
        splBool = seg([tpcWM], ['and'])
        postRemoveBool = seg([fusTopPost], ['not'])
        classificationpArc = addTractToClassification(classificationpArc, side + 'pArc', endpointBool, vertBool, ~splBool, postRemoveBool)
        classificationTPC = addTractToClassification(classificationTPC, side + 'TPC', endpointBool, vertBool, splBool)
    return reconcileClassifications(classificationpArc, classificationTPC)


def segmentOpticRadiation(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: optic radiation, ported from MATLAB's
    bsc_opticRadiationSeg_V7 (expandSegBool=0, as wma_segMajTracks_BL_v2
    runs it).  Per hemisphere, streamlines ending in the thalamus (grown by
    3) with an occipital_to_subcortical prior that cross the thalamus'
    lateral plane behind its posterior border:

    - Inferior_OR (MATLAB 'meyer', Meyer's loop): avoids the thalamus'
      anterior plane and its posterior plane above the hippocampus;
    - Superior_OR (MATLAB 'baum', Baum's loop): the rest, avoiding the
      pallidum's posterior plane.

    Names as in Bullock et al. 2022 (doi:10.1093/cercor/bhab500).

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    for iSide in range(2):
        side = SIDE_NAMES[iSide]
        occipSubBool = _priorBool(categoryPrior, side + 'occipital_to_subcortical')
        seg = _gate(streamlines, occipSubBool)
        thalamicROI = roiFromAtlasLabels_bsc(atlas, _THALAMUS[iSide], 7)
        thalAnt = planeFromROI_bsc(_THALAMUS[iSide], 'anterior', atlas)
        thalLat = planeFromROI_bsc(_THALAMUS[iSide], 'lateral', atlas)
        thalPost = planeFromROI_bsc(_THALAMUS[iSide], 'posterior', atlas)
        hippTop = planeFromROI_bsc(_HIPPOCAMPUS[iSide], 'superior', atlas)
        hippClip = modifyROI_bsc(thalPost, hippTop, 'superior')
        thalLatPost = modifyROI_bsc(thalLat, thalPost, 'anterior')
        putPostPlane = planeFromROI_bsc(_PALLIDUM[iSide], 'posterior', atlas)
        antThalExcludeBool = seg([putPostPlane], ['not'])
        extremeAntExcludeBool = seg([thalAnt], ['not'])
        thalBool = seg([thalamicROI], ['endpoints'])
        clipBool = seg([hippClip], ['not'])
        latThalBool = seg([thalLatPost], ['and'])
        inferiorBool = thalBool & clipBool & occipSubBool & latThalBool & extremeAntExcludeBool
        classification = addTractToClassification(classification, side + 'Inferior_OR', inferiorBool)
        classification = addTractToClassification(classification, side + 'Superior_OR', thalBool, ~inferiorBool, occipSubBool, latThalBool, antThalExcludeBool)
    return classification


def segmentCerebellarTracts(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True, experimental=False, includeSpinoCerebellar=False):
    """
    Claude note: cerebellar tracts, ported from MATLAB's
    bsc_segmentCerebellarTracts_v2 (experimentalBool=0, as
    wma_segMajTracks_BL_v2 runs it).  All end in the (inflated) cerebellum.
    Per hemisphere: MotorCerebellar (other end in motor cortex),
    AnterioFrontoCerebellar (crossing the thalamus' anterior plane),
    ThalamicoCerebellar, OccipitoCerebellar, ParietoCerebellar, and the
    contralateral ContraMotorCerebellar / ContraAnterioFrontoCerebellar
    (other side's cerebellum, *_interHemi priors, not touching the other
    side's white matter), plus, only on request, leftSpinoCerebellar /
    rightSpinoCerebellar (cerebellar endpoint, prior
    <side>cerebellum_to_spinal, not crossing the brainstem's top plane).

    Spino-cerebellar differs from MATLAB on purpose: MATLAB asked for the
    prior 'cerebellum_to_spinal(_interHemi)', which the prior never produces
    (a cerebellar endpoint always gets a 'left' / 'right' prefix), so those
    tracts were always empty and absent from its output.  Here the
    side-prefixed name is used (fix agreed with Dan Bullock, 2026-11).  The
    fixed recipe is too broad: on HCP sub-100206 (3M streamlines) it gives
    ~9k / ~13k streamlines, mostly cerebellum <-> brainstem / ventral DC
    through the peduncles, with implausible wanderers.  It needs work, so it
    is off by default (includeSpinoCerebellar=False) and should not be
    enabled in any default or production setting (Dan Bullock, 2026-11).

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.
    experimental : bool, optional
        Also the MATLAB experimental tracts.  The default is False.
    includeSpinoCerebellar : bool, optional
        Also the (fixed, but not yet reworked) spino-cerebellar tracts, see
        above.  For development only.  The default is False.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    #Claude note: gate on any cerebellar prior
    cerebellarPriorBool = np.zeros(len(streamlines), dtype=bool)
    for iNumber, iName in enumerate(categoryPrior['names']):
        if 'cerebellum' in iName:
            cerebellarPriorBool |= np.asarray(categoryPrior['index']) == iNumber + 1
    seg = _gate(streamlines, cerebellarPriorBool)
    cbROIs = [roiFromAtlasLabels_bsc(inflatedAtlas, list(_CEREBELLUM[iSide]), 1) for iSide in range(2)]
    #Claude note: spino-cerebellar, with the side-prefixed prior name (see
    #docstring)
    for iSide in range(2 if includeSpinoCerebellar else 0):
        spineTopBool = seg([planeFromROI_bsc(16, 'superior', atlas)], ['not'])
        spinoCebBool = _priorBool(categoryPrior, SIDE_NAMES[iSide] + 'cerebellum_to_spinal')
        classification = addTractToClassification(classification, SIDE_NAMES[iSide] + 'SpinoCerebellar',
                                                  seg([cbROIs[iSide]], ['endpoints']), spinoCebBool, spineTopBool)
    spineLimit = planeFromROI_bsc(16, 'anterior', atlas)
    posteriorStreamsBool = seg([spineLimit], ['not'])
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        cbROI, otherCBROI = cbROIs[iSide], cbROIs[1 - iSide]
        thalROI = roiFromAtlasLabels_bsc(inflatedAtlas, _THALAMUS[iSide], 1)
        antCebSplit = planeFromROI_bsc(_THALAMUS[iSide], 'anterior', atlas)
        thalTop = planeFromROI_bsc(_THALAMUS[iSide], 'superior', atlas)
        #Claude note: motor ROI as in the thalamic radiations
        frontoMotorLimit = planeFromROI_bsc(170 + sideNum, 'anterior', inflatedAtlas)
        frontoMedMotorROI = modifyROI_bsc(roiFromAtlasLabels_bsc(inflatedAtlas, 116 + sideNum), frontoMotorLimit, 'posterior')
        motorROI = mergeROIs(frontoMedMotorROI, roiFromAtlasLabels_bsc(inflatedAtlas, np.add(_MOTOR_BASE, sideNum)))
        anterioFrontoBool = seg([cbROI, antCebSplit], ['endpoints', 'and'])
        middleFrontoBool = seg([cbROI, antCebSplit], ['endpoints', 'not'])
        thalCebBool = seg([cbROI, thalROI, spineLimit, thalTop], ['endpoints', 'endpoints', 'not', 'not'])
        motorCebBool = seg([cbROI, motorROI], ['endpoints', 'endpoints'])
        thisCebBool = seg([cbROI], ['endpoints'])
        cebFrontal = _priorBool(categoryPrior, side + 'cerebellum_to_frontal')
        cebParietal = _priorBool(categoryPrior, side + 'cerebellum_to_parietal')
        motorCebBool = motorCebBool & (cebFrontal | cebParietal)
        anterioFrontoBool = anterioFrontoBool & cebFrontal
        middleFrontoBool = ~motorCebBool & middleFrontoBool & cebFrontal
        thalCebBool = thalCebBool & _priorBool(categoryPrior, side + 'cerebellum_to_subcortical')
        occipitoCebBool = posteriorStreamsBool & thisCebBool & _priorBool(categoryPrior, side + 'cerebellum_to_occipital')
        parietoCebBool = posteriorStreamsBool & thisCebBool & ~motorCebBool & cebParietal
        classification = addTractToClassification(classification, side + 'MotorCerebellar', motorCebBool)
        classification = addTractToClassification(classification, side + 'AnterioFrontoCerebellar', anterioFrontoBool)
        classification = addTractToClassification(classification, side + 'ThalamicoCerebellar', thalCebBool)
        classification = addTractToClassification(classification, side + 'OccipitoCerebellar', occipitoCebBool)
        classification = addTractToClassification(classification, side + 'ParietoCerebellar', parietoCebBool)
        if experimental:
            classification = addTractToClassification(classification, side + 'MiddleFrontoCerebellar', middleFrontoBool)
            classification = addTractToClassification(classification, side + 'MiddleFrontoBoolCerebellar', middleFrontoBool)
        #Claude note: contralateral tracts
        otherWM = roiFromAtlasLabels_bsc(atlas, WM_LABELS[1 - iSide], 1)
        notTheseBool = seg([otherWM], ['and'])
        contraAnterioFrontoBool = seg([otherCBROI, antCebSplit], ['endpoints', 'and'])
        contraMotorCebBool = seg([otherCBROI, motorROI], ['endpoints', 'endpoints'])
        contraMotorCebBool = ~notTheseBool & contraMotorCebBool & _priorBool(categoryPrior, 'cerebellum_to_frontal_interHemi', 'cerebellum_to_parietal_interHemi')
        contraAnterioFrontoBool = ~notTheseBool & contraAnterioFrontoBool & _priorBool(categoryPrior, 'cerebellum_to_frontal_interHemi')
        classification = addTractToClassification(classification, side + 'ContraMotorCerebellar', contraMotorCebBool)
        classification = addTractToClassification(classification, side + 'ContraAnterioFrontoCerebellar', contraAnterioFrontoBool)
        if experimental:
            contraMiddleFrontoBool = seg([otherCBROI, antCebSplit], ['endpoints', 'not'])
            contraThalCebBool = seg([otherCBROI, thalROI, spineLimit], ['endpoints', 'endpoints', 'not'])
            contraThisCebBool = seg([otherCBROI], ['endpoints'])
            contraMiddleFrontoBool = ~notTheseBool & ~contraMotorCebBool & contraMiddleFrontoBool & _priorBool(categoryPrior, 'cerebellum_to_frontal_interHemi')
            contraThalCebBool = ~notTheseBool & contraThalCebBool & _priorBool(categoryPrior, 'cerebellum_to_subcortical_interHemi')
            contraOccipitoCebBool = ~notTheseBool & posteriorStreamsBool & contraThisCebBool & _priorBool(categoryPrior, 'cerebellum_to_occipital_interHemi')
            contraParietoCebBool = ~notTheseBool & posteriorStreamsBool & contraThisCebBool & ~contraMotorCebBool & _priorBool(categoryPrior, 'cerebellum_to_parietal_interHemi')
            classification = addTractToClassification(classification, side + 'ContraOccipitoCerebellar', contraOccipitoCebBool)
            classification = addTractToClassification(classification, side + 'ContraParietoCerebellar', contraParietoCebBool)
            classification = addTractToClassification(classification, side + 'ContraMiddleFrontoBoolCerebellar', contraMiddleFrontoBool)
            classification = addTractToClassification(classification, side + 'ContraThalamicoCerebellar', contraThalCebBool)
    return classification


def segmentVOF(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: vertical occipital fasciculus, ported from MATLAB's
    bsc_segmentVOF_v4.  Per hemisphere: one endpoint in inferior occipital
    {143 102 122 152 161 162}, the other in superior occipital
    {120 121 158 111 166 159} (not both in the same group); avoids the 111
    anterior plane; exactly one endpoint above z=3 mm and one below
    z=-3 mm.  Prior: occipital_to_occipital(_ufiber).

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    supEndpointsBool = endpointCriteria_bsc(streamlines, [0, 0, 3], 'superior', 'one')
    infEndpointsBool = endpointCriteria_bsc(streamlines, [0, 0, -3], 'inferior', 'one')
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        occOccBool = _priorBool(categoryPrior, side + 'occipital_to_occipital', side + 'occipital_to_occipital_ufiber')
        seg = _gate(streamlines, occOccBool)
        inferiorROI = roiFromAtlasLabels_bsc(atlas, list(np.add([143, 102, 122, 152, 161, 162], sideNum)), 1)
        superiorROI = roiFromAtlasLabels_bsc(atlas, list(np.add([120, 121, 158, 111, 166, 159], sideNum)), 1)
        inferiorBool = tractByEndpointROIs_bsc(streamlines, inferiorROI, inferiorROI)
        superiorBool = tractByEndpointROIs_bsc(streamlines, superiorROI, superiorROI)
        criteriaBool = tractByEndpointROIs_bsc(streamlines, inferiorROI, superiorROI)
        vofCandidateBool = seg([planeFromROI_bsc(111 + sideNum, 'anterior', atlas)], ['not'])
        vofBool = ~inferiorBool & ~superiorBool & occOccBool & criteriaBool & vofCandidateBool & supEndpointsBool & infEndpointsBool
        classification = addTractToClassification(classification, side + 'VOF', vofBool)
    return classification


def segmentCST(streamlines, atlas, categoryPrior=None, inflatedAtlas=None, deislandAtlas=True):
    """
    Claude note: corticospinal tract, ported from MATLAB's bsc_segmentCST.
    Per hemisphere: an endpoint in motor cortex {168 128 146 129 170 103}
    (inflated), passing the brainstem (16, inflated); avoiding the other
    hemisphere's white matter, the cerebellum (inflated) and the thalamus'
    anterior plane; prior name containing neither 'subcortical' nor
    'cerebellum'.

    Parameters
    ----------
    streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas
        As segmentThalamicRadiations.

    Returns
    -------
    classification : dict
    """
    atlas, categoryPrior, inflatedAtlas = _prepareInputs(streamlines, atlas, categoryPrior, inflatedAtlas, deislandAtlas)
    classification = emptyClassification(len(streamlines))
    #Claude note: prior names without 'subcortical' or 'cerebellum'
    allowedNumbers = [iNumber + 1 for iNumber, iName in enumerate(categoryPrior['names'])
                      if 'subcortical' not in iName and 'cerebellum' not in iName]
    priorBool = np.isin(categoryPrior['index'], allowedNumbers)
    cbROI = roiFromAtlasLabels_bsc(inflatedAtlas, [7, 8, 46, 47], 1)
    spineROI = roiFromAtlasLabels_bsc(inflatedAtlas, 16, 1)
    for iSide in range(2):
        side, sideNum = SIDE_NAMES[iSide], SIDE_OFFSETS[iSide]
        motorROI = roiFromAtlasLabels_bsc(inflatedAtlas, np.add(_MOTOR_BASE, sideNum), 1)
        otherWM = roiFromAtlasLabels_bsc(atlas, WM_LABELS[1 - iSide], 1)
        thalAntPlane = planeFromROI_bsc(_THALAMUS[iSide], 'anterior', atlas)
        #Claude note: motor endpoint first: it is the most selective criterion
        motorIndexes = np.where(priorBool & segmentByROIs(streamlines, [motorROI], ['endpoints']))[0]
        cstBool = _segmentSubset(streamlines, motorIndexes, [spineROI, otherWM, cbROI, thalAntPlane], ['and', 'not', 'not', 'not'])
        classification = addTractToClassification(classification, side + 'CST', priorBool, cstBool)
    return classification


def segmentMajorTracts(streamlines, atlas, deislandAtlas=True, includeCerebellar=False):
    """
    Claude note: the whole major-tract segmentation, as MATLAB's
    wma_segMajTracks_BL_v2 / brainlife app-wmaSeg: island-clean the atlas
    (prepareAtlas_bsc), build the inflated atlas and category prior once,
    then run the families in MATLAB's order, a later family winning any
    streamline claimed twice: AntPostTracts (incl. cingulum),
    CorpusCallosum, ThalamicRadiations, Aslant, MdLF_ILF, pArcTPC,
    OpticRadiation, CerebellarTracts (only if includeCerebellar), VOF, CST,
    Cingulum (run again, as in MATLAB).  Tracts with no streamlines are
    dropped at the end (as wma_resortClassificationStruc does).

    The cerebellar tracts are left out by default (Dan Bullock, 2026-11):
    they were experimental, and a whole-brain tractogram rarely models them
    well; they generally need dedicated, targeted tracking, and are long
    enough that tracking must be set up to allow streamlines of that
    length.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines
        Whole brain tractogram, subject (RAS mm) space.
    atlas : nibabel.Nifti1Image
        FreeSurfer aparc.a2009s+aseg atlas.
    deislandAtlas : bool, optional
        Remove label islands first.  The default is True.
    includeCerebellar : bool, optional
        Also run segmentCerebellarTracts (as MATLAB did).  The default is
        False.

    Returns
    -------
    classification : dict
        wmc-style classification: 47 tracts for a typical subject; with
        includeCerebellar, MATLAB's 61 (the spino-cerebellar tracts are
        never produced here, see segmentCerebellarTracts).
    """
    if deislandAtlas:
        atlas = prepareAtlas_bsc(atlas)
    inflatedAtlas = inflateAtlasLabels_bsc(atlas, 2)
    categoryPrior = streamlineCategoryPriors(streamlines, atlas, inflatedAtlas)
    families = [segmentAntPostTracts, segmentCorpusCallosum, segmentThalamicRadiations, segmentAslant,
                segmentMdLF_ILF, segmentpArcTPC, segmentOpticRadiation, segmentCerebellarTracts,
                segmentVOF, segmentCST, segmentCingulum]
    if not includeCerebellar:
        families.remove(segmentCerebellarTracts)
    classification = emptyClassification(len(streamlines))
    for iFamily in families:
        familyClassification = iFamily(streamlines, atlas, categoryPrior=categoryPrior, inflatedAtlas=inflatedAtlas, deislandAtlas=False)
        classification = reconcileClassifications(classification, familyClassification)
    return dropEmptyTracts(classification)
