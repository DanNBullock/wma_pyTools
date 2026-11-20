# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:03:51 2021

@author: Daniel
"""
#lets do the importing at the top I guess
import numpy as np
import pandas as pd
import nibabel as nib
import itertools
from dipy.tracking.utils import near_roi
import dipy.tracking.utils as ut
from nilearn import masking 
import scipy
from scipy.spatial.distance import cdist
from dipy.core.geometry import dist_to_corner
from dipy.tracking.streamline import Streamlines

import time
import wmaPyTools.roiTools
import wmaPyTools.genUtils
import dipy.segment.mask as mask
from dipy.tracking.utils import streamline_near_roi
from dipy.tracking.utils import density_map
from scipy.ndimage import binary_dilation
from warnings import warn

import dipy.tracking.utils as ut
from dipy.tracking.vox2track import _streamlines_in_mask
import copy
#Claude note: used by the vectorised / KD-tree helpers below
from scipy.spatial import cKDTree
from nibabel.streamlines import ArraySequence


# Claude note: --------------------------------------------------------------
# Private speedup helpers (added 2026-11).
#
# Most of the segmentation time in this module was spent in three places:
#   1. per-streamline Python loops (each paying numpy call overhead),
#   2. all-pairs distance computations (scipy cdist of every streamline node
#      against every ROI voxel center, one streamline at a time, inside
#      dipy's near_roi / streamline_near_roi), and
#   3. np.unique over whole 256^3 mask volumes, just to check that the mask
#      only holds 0s and 1s.
# The helpers below replace these with operations over the concatenated node
# array of all streamlines, and with KD-tree lookups.  They are written to
# return results IDENTICAL to the code they replace (including its edge case
# behavior), not merely "close": see tests/test_speedupEquivalence.py.
# ---------------------------------------------------------------------------

def _arraySequenceRowIndexes(streamlines):
    """
    Claude note: the rows of streamlines._data that make up an ArraySequence,
    in streamline order.  An indexed ArraySequence (e.g. streamlines[[3,7,9]])
    shares its parent's _data and only carries its own offsets, so its rows
    need to be gathered rather than just read off.

    Returns
    -------
    rowIndexes : np.ndarray of intp, (sum(lengths),)
    lengths : np.ndarray of intp, (len(streamlines),)
    """
    lengths = np.asarray(streamlines._lengths, dtype=np.intp)
    offsets = np.asarray(streamlines._offsets, dtype=np.intp)
    totalNodes = int(np.sum(lengths))
    starts = np.cumsum(lengths) - lengths
    rowIndexes = np.repeat(offsets - starts, lengths) + np.arange(totalNodes, dtype=np.intp)
    return rowIndexes, lengths


#Claude note: streamlines are processed in chunks of at most this many nodes,
#so that the flattened (float64) node arrays stay small (~50MB) however large
#the tractogram.  Every per-streamline decision is independent of the other
#streamlines, so chunking cannot change any result.
_CHUNK_NODES = 2 ** 21


def _streamlineChunks(streamlines, maxNodes=None):
    """
    Claude note: split a streamline collection into consecutive chunks of at
    most maxNodes nodes (or a single streamline, if one is longer).

    Yields
    ------
    firstIndex : int
        index of the chunk's first streamline in the input
    chunk : same type as the input (an ArraySequence slice is a view)
    """
    if maxNodes is None:
        maxNodes = _CHUNK_NODES
    lengths = _streamlineLengths(streamlines)
    nStreams = len(lengths)
    if nStreams == 0:
        yield 0, streamlines[0:0]
        return
    cumulative = np.cumsum(lengths)
    firstIndex = 0
    while firstIndex < nStreams:
        alreadyCounted = cumulative[firstIndex - 1] if firstIndex > 0 else 0
        lastIndex = int(np.searchsorted(cumulative, alreadyCounted + maxNodes, side='right'))
        lastIndex = max(lastIndex, firstIndex + 1)
        yield firstIndex, streamlines[firstIndex:lastIndex]
        firstIndex = lastIndex


def _flattenStreamlines(streamlines):
    """
    Claude note: concatenate all nodes of a streamline collection into one
    array, along with the per-streamline node counts.

    Parameters
    ----------
    streamlines : nibabel ArraySequence / dipy Streamlines, or a list (or
        other iterable) of (N,3) arrays

    Returns
    -------
    nodes : np.ndarray, (sum(lengths),3), float64
        All streamline nodes, in streamline order.  Converted to float64,
        which is exact for float32 input, and which is also what scipy's
        cdist converts its input to.
    lengths : np.ndarray, (len(streamlines),), intp
        The number of nodes in each streamline.
    """
    if isinstance(streamlines, ArraySequence):
        rowIndexes, lengths = _arraySequenceRowIndexes(streamlines)
        nodes = np.asarray(streamlines._data)[rowIndexes].astype(np.float64)
    else:
        asArrays = [np.asarray(iStream) for iStream in streamlines]
        lengths = np.asarray([len(iStream) for iStream in asArrays], dtype=np.intp)
        if len(asArrays) == 0 or np.sum(lengths) == 0:
            nodes = np.zeros((0, 3), dtype=np.float64)
        else:
            nodes = np.concatenate([np.reshape(iStream, (-1, iStream.shape[-1])) for iStream in asArrays if len(iStream) > 0]).astype(np.float64)
    return nodes, lengths


def _nodesAtPositions(streamlines, positionFunc):
    """
    Claude note: one node per streamline, at a position computed from the
    streamline's length (e.g. 0 for the first node, length-1 for the last,
    length//2 for the midpoint), as float64, without gathering every node of
    every streamline first.

    Parameters
    ----------
    streamlines : ArraySequence / Streamlines, or list of (N,3) arrays
    positionFunc : function
        maps an array of streamline lengths to an array of within-streamline
        node indexes

    Returns
    -------
    nodes : np.ndarray, (len(streamlines),3) float64
    """
    if isinstance(streamlines, ArraySequence):
        lengths = np.asarray(streamlines._lengths, dtype=np.intp)
        offsets = np.asarray(streamlines._offsets, dtype=np.intp)
        return np.asarray(streamlines._data)[offsets + positionFunc(lengths)].astype(np.float64)
    allNodes, allLengths = _flattenStreamlines(streamlines)
    starts = np.cumsum(allLengths) - allLengths
    return allNodes[starts + positionFunc(allLengths)]


def _streamlineLengths(streamlines):
    """
    Claude note: the number of nodes in each streamline.
    """
    if isinstance(streamlines, ArraySequence):
        return np.asarray(streamlines._lengths, dtype=np.intp)
    return np.asarray([len(iStream) for iStream in streamlines], dtype=np.intp)


def _uniqueMaskValues(data):
    """
    Claude note: returns exactly what np.unique(data) returns, but without
    sorting the whole volume in the (overwhelmingly common) case that the
    data only contains 0s and 1s.  np.unique on a 256^3 float volume takes
    on the order of a second, and was run twice per segmentation criterion
    just as an input check.

    Parameters
    ----------
    data : np.ndarray
        Presumed mask data

    Returns
    -------
    uniqueVals : np.ndarray
        identical to np.unique(data)
    """
    data = np.asanyarray(data)
    isZero = data == 0
    isOne = data == 1
    zeroCount = np.count_nonzero(isZero)
    oneCount = np.count_nonzero(isOne)
    #Claude note: anything else present (other values, NaN): just use np.unique
    if zeroCount + oneCount != data.size or data.size == 0:
        return np.unique(data)
    present = []
    if zeroCount > 0:
        #Claude note: take the value from the data itself (rather than 0) so
        #that e.g. -0.0 is reproduced as np.unique would report it
        present.append(data[isZero].min())
    if oneCount > 0:
        present.append(1)
    return np.asarray(present, dtype=data.dtype)


def _nodesNearCoords(nodes, roiCoords, tol, tree=None):
    """
    Claude note: for each node, whether ANY of the roiCoords lies within tol of
    it, where "within" is decided by exactly the arithmetic dipy's
    streamline_near_roi uses: scipy's cdist (euclidean) followed by <= tol.

    A KD-tree over the ROI coordinates answers the question for nearly all
    nodes directly.  Only for nodes whose nearest ROI coordinate is within a
    tiny margin of tol (where the KD-tree's own floating point arithmetic
    could in principle disagree with cdist's) is the answer re-derived with
    cdist itself, so the result is identical to the all-pairs computation.

    Parameters
    ----------
    nodes : np.ndarray, (N,3) float64
    roiCoords : np.ndarray, (M,3) float64, M>0, all finite
    tol : float
    tree : scipy.spatial.cKDTree, optional
        a KD-tree already built over roiCoords (to reuse across chunks)

    Returns
    -------
    nodeNear : np.ndarray, (N,) bool
        False for non-finite nodes (NaN / inf), which cdist could never place
        within a finite tol either.
    """
    nodeNear = np.zeros(len(nodes), dtype=bool)
    finiteNodes = np.all(np.isfinite(nodes), axis=1)
    if not np.any(finiteNodes) or len(roiCoords) == 0:
        return nodeNear
    queryNodes = nodes[finiteNodes]
    if tree is None:
        tree = cKDTree(roiCoords)
    #Claude note: far larger than any floating point disagreement between the
    #KD-tree and cdist (which are of order 1e-15 relative), far smaller than
    #any meaningful distance
    margin = 1e-6 * max(abs(tol), 1.0)
    nearestDist, _ = tree.query(queryNodes, k=1, distance_upper_bound=tol + margin)
    finiteNear = nearestDist < tol - margin
    ambiguous = np.where(np.logical_and(~finiteNear, nearestDist <= tol + margin))[0]
    if len(ambiguous) > 0:
        candidateLists = tree.query_ball_point(queryNodes[ambiguous], r=tol + margin)
        for iAmbiguous, iCandidates in zip(ambiguous, candidateLists):
            if len(iCandidates) > 0:
                exactDists = cdist(queryNodes[iAmbiguous:iAmbiguous + 1], roiCoords[np.asarray(iCandidates)], 'euclidean')
                finiteNear[iAmbiguous] = np.min(exactDists) <= tol
    nodeNear[finiteNodes] = finiteNear
    return nodeNear


def _streamlinesNearCoords(nodes, lengths, roiCoords, tol, mode, tree=None):
    """
    Claude note: vectorised equivalent of applying a per-streamline
    "near ROI" test to every streamline in a flattened collection, with
    dipy.tracking.utils.streamline_near_roi's semantics: an all-zero (or
    empty) streamline is False, an empty ROI gives False, and the end modes
    test only the first and last node.

    Parameters
    ----------
    nodes, lengths : as returned by _flattenStreamlines
    roiCoords : np.ndarray, (M,3)
        ROI voxel center coordinates, in the streamlines' space
    tol : float
        distance tolerance
    mode : str
        one of 'any', 'all', 'either_end', 'both_end'

    Returns
    -------
    out : np.ndarray, (len(lengths),) bool
    """
    out = np.zeros(len(lengths), dtype=bool)
    if len(lengths) == 0:
        return out
    if len(roiCoords) == 0:
        return out
    nonEmpty = lengths > 0
    starts = np.cumsum(lengths) - lengths
    #Claude note: streamline-wise reductions are done with reduceat over the
    #non-empty streamlines only (reduceat misbehaves for empty segments)
    neStarts = starts[nonEmpty]
    neLengths = lengths[nonEmpty]

    if mode in ('either_end', 'both_end'):
        #Claude note: only the first and last node are tested, as dipy does
        endIndexes = np.column_stack([neStarts, neStarts + neLengths - 1]).ravel()
        endNodes = nodes[endIndexes]
        endNear = _nodesNearCoords(endNodes, roiCoords, tol, tree).reshape(-1, 2)
        #Claude note: a NaN in either end node makes cdist's min NaN, and
        #NaN <= tol is False
        endHasNaN = np.any(np.isnan(endNodes), axis=1).reshape(-1, 2)
        if mode == 'either_end':
            neOut = np.any(endNear, axis=1) & ~np.any(endHasNaN, axis=1)
        else:
            neOut = np.all(endNear, axis=1)
    else:
        nodeNear = _nodesNearCoords(nodes, roiCoords, tol, tree)
        if mode in ('any', 'either_end'):
            neOut = np.logical_or.reduceat(nodeNear, neStarts)
            #Claude note: np.min over a distance matrix holding any NaN is
            #NaN, so a single NaN node fails the whole streamline
            nodeNaN = np.any(np.isnan(nodes), axis=1)
            if np.any(nodeNaN):
                neOut = neOut & ~np.logical_or.reduceat(nodeNaN, neStarts)
        else:
            neOut = np.logical_and.reduceat(nodeNear, neStarts)
    out[nonEmpty] = neOut

    #Claude note: dipy returns False for any streamline where
    #`not np.any(streamline)`, i.e. empty or all-zero
    nodeNonZero = np.any(nodes != 0, axis=1)
    streamNonZero = np.zeros(len(lengths), dtype=bool)
    streamNonZero[nonEmpty] = np.logical_or.reduceat(nodeNonZero, neStarts)
    out &= streamNonZero
    return out


def _nearROISetup(affine, region_of_interest, tol=None, mode='any'):
    """
    Claude note: the part of dipy.tracking.utils.near_roi that does not depend
    on the streamlines: the tolerance (with dipy's warning when it is raised
    to dist_to_corner), the ROI voxel center coordinates, and a KD-tree over
    them.

    Returns
    -------
    useDipy : bool
        True when the fast path can't be used (an unknown mode, which dipy
        should be left to raise on, or non-finite ROI coordinates); callers
        then use dipy's near_roi itself.  No warning is issued in that case,
        so that dipy's own is not duplicated.
    tol : float
    x_roi_coords : np.ndarray, (M,3)
    tree : cKDTree, or None for an empty ROI
    """
    roi_coords = np.array(np.where(region_of_interest)).T
    x_roi_coords = ut.apply_affine(affine, roi_coords)
    if mode not in ('any', 'all', 'either_end', 'both_end') or not np.all(np.isfinite(x_roi_coords)):
        return True, tol, x_roi_coords, None
    dtc = dist_to_corner(affine)
    if tol is None:
        tol = dtc
    elif tol < dtc:
        w_s = "Tolerance input provided would create gaps in your"
        w_s += " inclusion ROI. Setting to: %s" % dtc
        warn(w_s, stacklevel=3)
        tol = dtc
    tree = cKDTree(x_roi_coords) if len(x_roi_coords) > 0 else None
    return False, tol, x_roi_coords, tree


def _nearROIExact(streamlines, affine, region_of_interest, tol=None, mode='any'):
    """
    Claude note: returns exactly what dipy.tracking.utils.near_roi(streamlines,
    affine, region_of_interest, tol=tol, mode=mode) returns, computed with a
    KD-tree over the ROI voxel centers, a chunk of whole streamlines at a time
    (see _streamlineChunks; streamlines are never split).

    Streamlines that cannot possibly meet the criterion are skipped with a
    bounding box test that is exact, not heuristic: a node within tol of a
    voxel center is within tol of it along every axis, so it must lie inside
    the ROI voxel centers' bounding box expanded by tol.  'any' needs at least
    one node in that box, 'all' needs every node, 'either_end' / 'both_end'
    need one / both endpoints.  The criterion itself is then evaluated on the
    candidate streamlines' *complete* node sets.

    Returns
    -------
    out : np.ndarray, (len(streamlines),) bool
    """
    useDipy, tolUsed, x_roi_coords, tree = _nearROISetup(affine, region_of_interest, tol, mode)
    if useDipy:
        return near_roi(streamlines, affine, region_of_interest, tol=tol, mode=mode)
    out = np.zeros(len(_streamlineLengths(streamlines)), dtype=bool)
    if len(x_roi_coords) == 0:
        return out
    #Claude note: the cushion only makes the box (harmlessly) larger, so that
    #floating point rounding can never exclude a node that cdist would accept
    cushion = 1e-6 * max(abs(tolUsed), 1.0)
    bounds = np.vstack([np.min(x_roi_coords, axis=0) - tolUsed - cushion,
                        np.max(x_roi_coords, axis=0) + tolUsed + cushion])
    for firstIndex, chunk in _streamlineChunks(streamlines):
        nodes, lengths = _flattenStreamlines(chunk)
        if len(lengths) == 0:
            continue
        nonEmpty = lengths > 0
        starts = np.cumsum(lengths) - lengths
        #Claude note: inclusive bounds here (the box is already padded)
        nodeInBox = np.all(np.logical_and(nodes >= bounds[0], nodes <= bounds[1]), axis=1)
        candidate = np.zeros(len(lengths), dtype=bool)
        neStarts = starts[nonEmpty]
        if mode == 'any':
            candidate[nonEmpty] = np.logical_or.reduceat(nodeInBox, neStarts)
        elif mode == 'all':
            candidate[nonEmpty] = np.logical_and.reduceat(nodeInBox, neStarts)
        else:
            firstIn = nodeInBox[neStarts]
            lastIn = nodeInBox[neStarts + lengths[nonEmpty] - 1]
            candidate[nonEmpty] = (firstIn | lastIn) if mode == 'either_end' else (firstIn & lastIn)
        if not np.any(candidate):
            continue
        streamIDs = np.repeat(np.arange(len(lengths)), lengths)
        candidateNodes = nodes[candidate[streamIDs]]
        candidateLengths = lengths[candidate]
        candidateIndexes = np.where(candidate)[0]
        out[firstIndex + candidateIndexes] = _streamlinesNearCoords(candidateNodes, candidateLengths, x_roi_coords, tolUsed, mode, tree=tree)
    return out


# Claude note: --------------------------------------------------------------
# Exact plane crossing ('cross' operation, added 2026-11).
#
# A planar ROI here is a nifti mask one voxel thick.  The distance-based modes
# ('any' etc.) treat it as a slab of voxel centres plus a tolerance, so they
# count streamlines that come near the plane and turn back ("kisses") or end
# near it, and, with widely spaced nodes, can miss streamlines that do cross.
# No tolerance fixes both, and counting the nodes within tolerance doesn't
# either (that count depends on the crossing angle, not on crossing vs.
# kissing).  What distinguishes a crossing is SIDE: a streamline crosses a
# plane exactly when two of its consecutive nodes lie on opposite sides of it.
#
# 'cross' therefore reads
#   * the streamline as a polyline (the straight path between consecutive
#     nodes; the nodes are never moved, added or removed), and
#   * the planar ROI as the infinitely thin plane through its voxel centres
#     (the slab's mid-plane), bounded by the mask's in-plane cells: the
#     square that each mask voxel cuts out of that plane.
# A streamline crosses the ROI if one of its segments passes from one side of
# the plane to the other through a mask cell.  The test is done in the mask's
# voxel coordinates, where the plane is axis-aligned even if the image is
# oblique in mm space; an affine map carries straight segments to straight
# segments and preserves where along a segment a point lies, so the crossing
# point found there is exact.  No tolerance is involved, and the result does
# not depend on how densely the nodes sample the path (adding nodes on a
# segment, or removing nodes that lie exactly on one, changes nothing).
# ---------------------------------------------------------------------------

def _planarROIGeometry(maskData):
    """
    Claude note: identify the plane of a one-voxel-thick planar mask.

    Parameters
    ----------
    maskData : np.ndarray, 3D
        The planar mask (non-zero = in the ROI).

    Returns
    -------
    planeAxis : int
        The voxel axis the plane is perpendicular to.
    planeIndex : int
        The voxel index of the plane along that axis.
    inPlane : np.ndarray of bool, 2D
        The mask's cells within the plane (the slice maskData[..., planeIndex,
        ...] along planeAxis).

    Raises
    ------
    ValueError
        If the mask is empty, more than one voxel thick along every axis, or
        one voxel thick along more than one axis (a line or a single voxel,
        whose plane is ambiguous).
    """
    nonZero = np.nonzero(maskData)
    if len(nonZero[0]) == 0:
        raise ValueError("'cross' requires a planar ROI, but the mask is empty.")
    #Claude note: axes along which every mask voxel has the same index
    singleAxes = [iAxis for iAxis in range(3) if np.all(nonZero[iAxis] == nonZero[iAxis][0])]
    if len(singleAxes) != 1:
        raise ValueError("'cross' requires a planar ROI one voxel thick along exactly one axis; this mask is one voxel thick along " + str(len(singleAxes)) + " axes.")
    planeAxis = singleAxes[0]
    planeIndex = int(nonZero[planeAxis][0])
    inPlane = np.take(np.asarray(maskData) != 0, planeIndex, axis=planeAxis)
    return planeAxis, planeIndex, inPlane


def _planeCrossingCounts(streamlines, affine, maskData):
    """
    Claude note: number of times each streamline crosses a planar ROI, read as
    a polyline crossing the plane through the mask's voxel centres within the
    mask's in-plane cells (see the section note above).

    A crossing is a change of side between consecutive nodes that are not on
    the plane.  Nodes exactly on the plane are skipped when deciding sides:
    a streamline that touches the plane and returns to the side it came from
    does not cross, one that passes through it does (the crossing point is
    then the first on-plane node).  A streamline that never leaves one side,
    or lies entirely in the plane, does not cross.

    Parameters
    ----------
    streamlines : streamline collection (in mm space)
    affine : np.ndarray, 4x4
        The planar mask's voxel-to-mm affine.
    maskData : np.ndarray, 3D
        The planar mask.

    Returns
    -------
    counts : np.ndarray of int, (len(streamlines),)
        Crossings through mask cells, per streamline.
    """
    planeAxis, planeIndex, inPlane = _planarROIGeometry(maskData)
    #Claude note: the two in-plane voxel axes, in order
    inPlaneAxes = [iAxis for iAxis in range(3) if iAxis != planeAxis]
    inverseAffine = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    counts = np.zeros(len(_streamlineLengths(streamlines)), dtype=np.intp)
    for firstIndex, chunk in _streamlineChunks(streamlines):
        nodes, lengths = _flattenStreamlines(chunk)
        if len(nodes) == 0:
            continue
        #Claude note: signed offset of every node from the plane, in voxels: only
        #the plane-normal voxel coordinate is needed for all nodes (one row of
        #the inverse affine); full voxel coordinates are computed below only
        #for the few nodes around crossings
        offset = nodes @ inverseAffine[planeAxis, :3] + (inverseAffine[planeAxis, 3] - planeIndex)
        side = np.sign(offset)
        streamIDs = np.repeat(np.arange(len(lengths)), lengths)
        #Claude note: consecutive OFF-plane nodes of the same streamline whose
        #sides differ; any nodes between them are exactly on the plane
        offPlane = np.flatnonzero(side != 0)
        if len(offPlane) < 2:
            continue
        before = offPlane[:-1]
        after = offPlane[1:]
        isCrossing = (streamIDs[before] == streamIDs[after]) & (side[before] != side[after])
        before = before[isCrossing]
        after = after[isCrossing]
        if len(before) == 0:
            continue
        #Claude note: the crossing point.  Adjacent nodes: interpolate along the
        #segment to offset 0.  Otherwise: the first on-plane node between them.
        toVoxel = lambda points: points @ inverseAffine[:3, :3].T + inverseAffine[:3, 3]
        adjacent = (after == before + 1)
        crossingPoints = toVoxel(nodes[before + 1])
        fraction = offset[before[adjacent]] / (offset[before[adjacent]] - offset[after[adjacent]])
        startPoints = toVoxel(nodes[before[adjacent]])
        crossingPoints[adjacent] = startPoints + fraction[:, None] * (toVoxel(nodes[after[adjacent]]) - startPoints)
        #Claude note: the in-plane cell holding each crossing point (nearest
        #voxel centre, i.e. floor(x+0.5) as dipy maps points to voxels)
        cellIndexes = np.floor(crossingPoints[:, inPlaneAxes] + 0.5).astype(np.intp)
        inBounds = np.all((cellIndexes >= 0) & (cellIndexes < np.asarray(inPlane.shape)), axis=1)
        throughMask = np.zeros(len(cellIndexes), dtype=bool)
        throughMask[inBounds] = inPlane[cellIndexes[inBounds, 0], cellIndexes[inBounds, 1]]
        #Claude note: count the crossings through mask cells per streamline
        counts[firstIndex:firstIndex + len(lengths)] += np.bincount(streamIDs[before[throughMask]], minlength=len(lengths))
    return counts


#Claude note: the plane crossing operation family, as predicates on the
#per-streamline crossing count (see planarROICrossings).  Each not_ name is
#the exact complement of its partner, i.e. the same streamlines as the
#positive operation with includeBool=False; they exist so criteria read
#naturally.  A streamline that crosses 3+ times (e.g. loops through the plane
#twice) is a 'cross' but neither a 'single_cross' nor a 'double_cross'.
CROSSING_OPERATIONS = {
    'cross': lambda counts: counts >= 1,
    'not_cross': lambda counts: counts == 0,
    'single_cross': lambda counts: counts == 1,
    'not_single_cross': lambda counts: counts != 1,
    'double_cross': lambda counts: counts == 2,
    'not_double_cross': lambda counts: counts != 2,
}


def _criterionStreamsBool(streamlines, affine, maskData, mode):
    """
    Claude note: the single place the per-ROI criteria are dispatched from:
    the plane crossing family (CROSSING_OPERATIONS) to the exact plane
    crossing test, everything else to the exact dipy near_roi equivalent.

    Returns
    -------
    out : np.ndarray, (len(streamlines),) bool
    """
    if mode in CROSSING_OPERATIONS:
        return CROSSING_OPERATIONS[mode](_planeCrossingCounts(streamlines, affine, maskData))
    return np.asarray(_nearROIExact(streamlines, affine, maskData, mode=mode))


def planarROICrossings(streamlines, planarROI):
    """
    Claude note: how many times each streamline crosses a planar ROI.

    Exact and tolerance-free: the streamline is read as a polyline, the ROI as
    the plane through its voxel centres within the mask's in-plane cells, and
    a crossing is a segment passing from one side of the plane to the other
    through a mask cell.  Streamlines that approach the plane and turn back,
    or end near it, are not counted.  Streamlines are not modified.  This is
    the test behind the 'cross' operation of segmentTractMultiROI and
    applyNiftiCriteriaToTract_DIPY(_Test).

    Parameters
    ----------
    streamlines : streamline collection (in mm space)
    planarROI : nifti
        A mask one voxel thick along exactly one voxel axis (e.g. from
        roiTools.makePlanarROI, planarROIFromAtlasLabelBorder, or such a
        plane cut down with sliceROIwithPlane / an intersection).

    Returns
    -------
    counts : np.ndarray of int, (len(streamlines),)
        0 = never crosses; 1 = crosses once; 2+ = crosses repeatedly (e.g. a
        streamline that loops back through the plane).  For a plane that
        covers the whole image slice, the parity says whether a streamline
        ends on the other side from where it started.
    """
    maskData = _checkMaskNifti(planarROI)
    return _planeCrossingCounts(streamlines, planarROI.affine, maskData)


def _checkMaskNifti(maskNifti):
    """
    Claude note: shared input check for the nifti-criterion functions.

    Previously each function compared np.unique(mask).astype(int) with
    [0, 1] elementwise, which (1) raised numpy's broadcasting ValueError
    instead of the intended Exception for masks with more than two values,
    (2) rejected a mask that is all 1s, and (3) reported masks such as
    {0, 0.5} or {0, 2} as "empty" while then using them anyway.

    Now: non-finite values raise; an all-zero mask warns that it is empty;
    values other than 0 and 1 warn that every non-zero voxel will be treated
    as part of the ROI (which is what the criterion does, as dipy's near_roi
    does).

    Returns
    -------
    maskData : np.ndarray
        the mask's data, read once
    """
    maskData=np.asanyarray(maskNifti.dataobj)
    maskUniqueVals=_uniqueMaskValues(maskData)
    if not np.all(np.isfinite(maskUniqueVals.astype(float))):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti contains non-finite values.  Likely not a mask.")
    if not np.any(maskUniqueVals!=0):
        warn("input mask nifti empty.")
    elif not np.all(np.isin(maskUniqueVals,[0,1])):
        warn("input mask nifti contains values other than 0 and 1 ("+str(maskUniqueVals[:10])+"); all non-zero voxels are treated as the ROI.")
    return maskData


def _unflattenStreamlines(nodes, lengths):
    """
    Claude note: inverse of _flattenStreamlines, as a list of (N,3) arrays.
    """
    return np.split(nodes, np.cumsum(lengths)[:-1]) if len(lengths) > 0 else []


def _expandedMaskBounds(maskNifti):
    """
    Claude note: the subject space bounding box of maskNifti's non-zero voxels,
    expanded by dist_to_corner, as used by subsetStreamsNodesByROIboundingBox.
    """
    #use distance to corner to set tolerance
    dtc = dist_to_corner(maskNifti.affine)
    #convert them to subject space
    subjectSpaceBounds=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(maskNifti)
    #expand to accomidate tolerance
    subjectSpaceBounds[0,:]=subjectSpaceBounds[0,:]-dtc
    subjectSpaceBounds[1,:]=subjectSpaceBounds[1,:]+dtc
    return subjectSpaceBounds


def _subsetNodesWithinBounds(nodes, lengths, bounds):
    """
    Claude note: for each (flattened) streamline, keep only its nodes within
    bounds, dropping streamlines left with no nodes, exactly as building a
    Streamlines object from per-streamline streamlineNodesWithinBounds
    subsets does.

    Returns
    -------
    keptIndexes : np.ndarray of int
        indexes (into the input) of the streamlines with any node in bounds
    outNodes, outLengths : flattened form of those streamlines' in-bounds
        nodes
    """
    nodeInBounds=_boundedNodeMask(nodes, bounds)
    streamIDs=np.repeat(np.arange(len(lengths)), lengths)
    keptPerStream=np.bincount(streamIDs[nodeInBounds], minlength=len(lengths))
    keptIndexes=np.where(keptPerStream>0)[0]
    return keptIndexes, nodes[nodeInBounds], keptPerStream[keptIndexes].astype(np.intp)


def _subsetNodesByROIboundingBox(streamlines, maskNifti, endpointsOnly=False):
    """
    Claude note: chunked, vectorised core of subsetStreamsNodesByROIboundingBox
    (see _subsetNodesWithinBounds), with its timing report.

    Parameters
    ----------
    streamlines : streamline collection
    maskNifti : nifti mask
    endpointsOnly : bool
        if True, each streamline is first reduced to its first and last node
        (as applyNiftiCriteriaToTract_DIPY_Test does for the 'end' modes)

    Returns
    -------
    outIndexes : np.ndarray of int
    outNodes, outLengths : flattened in-bounds nodes of those streamlines
    """
    #Claude note: timed with genUtils.StageTimer (wall-clock; silenced by
    #genUtils.setTimingVerbose(False)) instead of time.process_time
    stageTimer=wmaPyTools.genUtils.StageTimer('_subsetNodesByROIboundingBox')
    subjectSpaceBounds=_expandedMaskBounds(maskNifti)
    indexChunks, nodeChunks, lengthChunks = [], [], []
    nStreams=0
    for firstIndex, chunk in _streamlineChunks(streamlines):
        nodes, lengths = _flattenStreamlines(chunk)
        if endpointsOnly:
            starts=np.cumsum(lengths)-lengths
            nodes=nodes[np.column_stack([starts,starts+lengths-1]).ravel()]
            lengths=np.full(len(lengths),2,dtype=np.intp)
        keptIndexes, keptNodes, keptLengths = _subsetNodesWithinBounds(nodes, lengths, subjectSpaceBounds)
        indexChunks.append(keptIndexes+firstIndex)
        nodeChunks.append(keptNodes)
        lengthChunks.append(keptLengths)
        nStreams+=len(lengths)
    outIndexes=np.concatenate(indexChunks)
    outNodes=np.concatenate(nodeChunks) if len(nodeChunks)>0 else np.zeros((0,3))
    outLengths=np.concatenate(lengthChunks).astype(np.intp)
    #stop timing
    stageTimer.finish(lambda modifiedTime: 'Tractogram subseting complete in ' +str(modifiedTime) +', '+str(len(outIndexes)) + ' of ' + str(nStreams) + ' within mask boundaries')
    return outIndexes, outNodes, outLengths


def _boundedNodeMask(nodes, bounds):
    """
    Claude note: vectorised form of the within-bounds node test used by
    streamlineWithinBounds / streamlineNodesWithinBounds / near_roi_fast:
    strictly greater than bounds[0] and strictly less than bounds[1], in
    every dimension.
    """
    return np.all(np.logical_and(nodes > bounds[0, :], nodes < bounds[1, :]), axis=1)



def speedSortSegCriteria(criteriaROIs,inclusionCriteria,operations):
    """
    Sorts segmentation criteria in order to minimize segmentation time.
    Acheives this by sorting criteria by how fast/harsh they are.  Endpoint
    criteria applied first, because they are only computed on endpoints.  Then
    for standard traversal ROIS, smallest are placed before larger.  This is 
    because (1) fewer streamlines are likely to pass through smaller ROIs,
    and so this is expected to be fast and (2), when paired with a bounding
    box implementation, greatly reduces the numbr of vreticies that are
    computed against.

    The ordering of the input vectors is presumed to be locked to one another.

    Parameters
    ----------
    criteriaROIs : TYPE
        A list of nifti ROI criteria
    inclusionCriteria : TYPE
        The boolean vector indicating whether these rois will operate as
    exclusion or inclusion ROIs
    operations : TYPE
        The vector of strings indicating which operations will be performed

    Returns
    -------
    sortedCriteriaROIs : TYPE
        A sorted list of nifti ROI criteria
    sortedInclusionCriteria : TYPE
        The sorted boolean vector indicating whether these rois will operate as
    exclusion or inclusion ROIs
    sortedOperations : TYPE
        The sorted vector of strings indicating which operations will be 
    performed

    """
    #also remember all = [some criteria] == any != [some criteria]
    #we could auto convert those as well to speed things up further.
    
    
    #import numpy as np
    #import pandas as pd
    
    #initialize an output order vector
    outOrder=list(range(len(criteriaROIs)))
    
    #for each of the input ROIs compute the number of voxels that will be checked
    volumesToCheck=np.zeros(len(criteriaROIs))
    for iROIs in range(len(criteriaROIs)):
        #compute the volume that needs to be checked
        volumesToCheck[iROIs]=np.sum(np.asanyarray(criteriaROIs[iROIs].dataobj))

    print("volumes of input ROIs to check " + str(volumesToCheck))
    #find the criteria that only involve endpoints (you don't have to check all the nodes
    #for these, only the endpoints, so it's way faster to do cdist)
    #Claude note: the order is built as one concatenation:
    #endpoint criteria (smallest ROI first), then 'cross' plane criteria (a
    #single cheap pass over the nodes), then the other all-node criteria
    #(smallest ROI first), then anything else in its original position order,
    #so no criterion is duplicated or dropped.
    endpointCriteriaIndexes=np.where(['end' in x for x in operations])[0]
    crossCriteriaIndexes=np.where([x in CROSSING_OPERATIONS for x in operations])[0]
    allNodeCriteriaIndexes=np.where([(('any' in x) or ('all' in x)) and not ('end' in x) for x in operations])[0]
    grouped=set(endpointCriteriaIndexes)|set(crossCriteriaIndexes)|set(allNodeCriteriaIndexes)
    otherIndexes=[iIndex for iIndex in range(len(operations)) if iIndex not in grouped]
    #stable sorts, so equal volumes keep their input order.  The order only
    #affects speed: the criteria are ANDed, so the result is the same in any order.
    outOrder=(list(endpointCriteriaIndexes[np.argsort(volumesToCheck[endpointCriteriaIndexes], kind='stable')])
              +list(crossCriteriaIndexes[np.argsort(volumesToCheck[crossCriteriaIndexes], kind='stable')])
              +list(allNodeCriteriaIndexes[np.argsort(volumesToCheck[allNodeCriteriaIndexes], kind='stable')])
              +otherIndexes)
    #initialize output objects
    sortedCriteriaROIs=[]
    sortedInclusionCriteria=[]
    sortedOperations=[]
    #resort the inputs in accordance with predicted stringency
    volumesToCheckOut=np.zeros(len(outOrder)).astype(int)
    for iOutputs in range(len(outOrder)):
        sortedCriteriaROIs.append(criteriaROIs[outOrder[iOutputs]])
        sortedInclusionCriteria.append(inclusionCriteria[outOrder[iOutputs]])
        sortedOperations.append(operations[outOrder[iOutputs]])
        
        volumesToCheckOut[iOutputs]=np.sum(np.asanyarray(sortedCriteriaROIs[iOutputs].dataobj)).astype(int)
        #if it is an exclusion, get the number of streamlines that are left out
    print("volumes of output ROIs to check " + str(volumesToCheckOut))
    
    reportData=[[volumesToCheckOut[iOut],sortedInclusionCriteria[iOut],sortedOperations[iOut]] for iOut in range(len(criteriaROIs))]
    reportFrame=pd.DataFrame(data=reportData, columns=['roiVolumes','inclusionCriteria','operations'])
    print(reportFrame)
    
    return sortedCriteriaROIs,sortedInclusionCriteria, sortedOperations
    
def segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -roisvec: a list of nifti objects that will serve as your ROIs
    #
    # -includeVec: a boolean list indicating whether you want the associated roi to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # operationsVec: a list with any of the following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    # -outBoolVec: boolean vec indicating streamlines that survived ALL operations
    #
    # NOTE: roisvec, includeVec, and operationsVec should all be the same lenth
    # ADVICE: apply the harshest (fewest survivors) criteria first.  Will result 
    # in signifigant speed ups.
    # ADVICE: starting with specifying endpoints has the additional benefit of reducting
    # the number of nodes considered per streamline to 2.  This would be an effective way
    # of implementing a fast and harsh first criteria.
    """
    #import numpy as np
    #import itertools 
    
    #Claude note: `not` is the logical negation; ~ is *bitwise* not on an int (~1 == -2)
    #and would never fire this guard.
    if not len(np.unique([len(roisvec), len(includeVec), len(operationsVec)]))==1:
        raise ValueError('mismatch between lengths of roi, inclusion, and operation vectors')

    #ok, but, if you were clever, you would sort the criterion now in the following way:
    #First do the endpoint criteria, and of those, do those in the order of smallest ROI to largest
    #Also be sure to account for the negation logic as well
    #what you're aiming for is to do the criterion in order of harshest to least harsh
    [roisvec,includeVec,operationsVec]=speedSortSegCriteria(roisvec,includeVec,operationsVec)
    
    #ok, lets do this battle royale style
    #here's our starting "class"
    remainingStreamIndexes=list(range(len(streamlines)))
    
    for iOperations in list(range(len(roisvec))):
        
        #perform this segmentation operation
        curBoolVec=applyNiftiCriteriaToTract_DIPY_Test(streamlines[remainingStreamIndexes], roisvec[iOperations], includeVec[iOperations], operationsVec[iOperations])
        #after the cull
        remainingStreamIndexes=list(itertools.compress(remainingStreamIndexes,curBoolVec))
        
    #create blank output structure
    outBoolVec=np.zeros(len(streamlines),dtype=bool)
    #these are the winners
    outBoolVec[remainingStreamIndexes]=True
    return outBoolVec

def streamIndexesFromConnGroupDict(grouping,indexes1,indexes2,undirected=True):
    """
    

    Parameters
    ----------
    grouping : collections.defaultdict, dictionary
        The output grouping object dictionary from
        dipy.tracking.utils.connectivity_matrix
    indexes1 : list of int
        A list of integers corresponding to the labels that the desired
        streamlines are to be connected to.
    indexes2 : list of int
        A list of integers corresponding to the labels that the desired
        streamlines are to be connected to.
    undirected : Bool, optional
        Flag that indicates whether the requested lables are to be treated
        as undirected (i.e. indexes1 to indexes2 OR indexes2 to indexes1).
        The default is True.

    Returns
    -------
    streamIndexes : list of int
        A list of integers corresponding to the streamline indexes of the
        streamlines that meet the desired connectivity criteria.

    """
    #create a function that checks if the input indexes are in the specified cols in the key array
    def keyCheck(grouping,indexesIn1,indexesIn2):
        np.asarray(list(grouping.keys()))
        outBoolVec=np.zeros(len(grouping.keys()),dtype=bool)
        
        for iIndexes,iKeys in enumerate(list(grouping.keys())):
            criteria1=iKeys[0] in indexes1
            criteria2=iKeys[1] in indexes2
            outBoolVec[iIndexes]=np.all([criteria1,criteria2])
            
        return outBoolVec
    
    #if its undirected
    if undirected:
        #do it with both orderings
        outBoolVec1=keyCheck(grouping,indexes1,indexes2)
        outBoolVec2=keyCheck(grouping,indexes2,indexes1)
        #and combine the results
        keysToCheck=np.any([outBoolVec1,outBoolVec2],axis=0)
    else:
        outBoolVec1=keyCheck(grouping,indexes1,indexes2)
        #and combine the results
        keysToCheck=outBoolVec1
    
    #create a list to hold the streamline indexes
    streamsList=[]
    #now iterate across the keys and pull out the indexes
    for iIndexes,iBool in enumerate(keysToCheck):
        if iBool:
            streamsList.append(grouping[list(grouping.keys())[iIndexes]])
    
    #unpack the lists
    streamIndexes=[streamIndxs for sublist in streamsList for streamIndxs in sublist]
    
    return streamIndexes
    

def segmentTractMultiROI_fast(streamlines, roisVec, includeVec, operationsVec):
    """
    Iteratively applies ROI-based streamline criteria using a number of speedup
    heuristics.

    Parameters
    ----------
    streamlines : TYPE
        treamlines to segment from, e.g. a candidate tractome.
    roisVec : TYPE
        nifti or list there of that will serve as your criteria ROIs.
        Streamlines will be assesed for their relation to these objects
    includeVec : bool or list thereof
        A boolean  or list thereof indicating whether you want the
        corresponding entry in roisVec to act as an INCLUSION or EXCLUSION ROI 
        (True=inclusion)
    operationsVec : string or list thereof
        A string or list thereof with any of the following instructions
        regarding which streamline nodes to asses (and how).
            >  "any" : any point is within tol from ROI. The default.
            >  "all" : all points are within tol from ROI.
            >  "either_end" : either of the end-points is within tol from ROI
            >  "both_end" : both end points are within tol from ROI.

    Raises
    ------
    ValueError
        Throws error if input vectors are of different lengths. 
    TypeError
        Throws error if inputs are of incorrect types

    Returns
    -------
    outBoolVec : boolean vector
        A boolean vector indicating which streamlines survived ALL operations.

    """
    # import numpy as np
    # import nibabel as nib
    # import itertools
  
    if isinstance(roisVec, nib.nifti1.Nifti1Image):
        #make it a list
        roisVec=[roisVec]
    if isinstance(roisVec, list):
        #check to make sure the input is correct
        if isinstance(roisVec[0], np.ndarray):
            raise TypeError ('Input mask format expected to be Nifti, not ndarray')
        
    if isinstance(includeVec, bool):
        #make it a list
        includeVec=[includeVec]
    if isinstance(includeVec, list):
        #seems sensible
        #check to make sure the input is correct
        if not np.all( [isinstance(iInclude, bool) for iInclude in includeVec]):
            raise TypeError ('Input includeVec not boolean vector')

    #Claude note: the plane crossing family (planar ROIs only)
    acceptableOperations=['any', 'all', 'either_end', 'both_end']+list(CROSSING_OPERATIONS)
    if isinstance(operationsVec, str):
        #make it a list
        operationsVec=[operationsVec]
    if isinstance(operationsVec, list):
        #check to make sure the input is correct
        if not np.all( [iOperations in acceptableOperations for iOperations in operationsVec]):
            raise TypeError ('Input operationsVec operations not understood')

    if not len(np.unique([len(roisVec), len(includeVec), len(operationsVec)]))==1:
        # print( str(np.unique([len(roisVec), len(includeVec), len(operationsVec)])))
        # print( str(len(roisVec)))
        # print( str(len(includeVec)))
        # print( str(len(operationsVec)))
        raise ValueError('Mismatch between lengths of ROI/Mask, inclusion, and operation vectors ')
        
    #sort the criterion for speed, harshest criteria first
    sortedCriteriaROIs,sortedInclusionCriteria, sortedOperations=speedSortSegCriteria(roisVec,includeVec,operationsVec)
        
    #initialize a list of surviving streams, battle royale style
    remainingStreamIndexes=list(range(len(streamlines)))
    
    #iterate across segmentation operations
    #Claude note: this loop uses the sorted criteria from the speed sort
    #above.  The order of AND-ed criteria doesn't change the result, only the speed.
    for iOperations in range(len(sortedCriteriaROIs)):
        #perform this segmentation operation
        curBoolVec=near_roi_fast(streamlines[remainingStreamIndexes], sortedCriteriaROIs[iOperations].affine, np.asanyarray(sortedCriteriaROIs[iOperations].dataobj),mode=sortedOperations[iOperations])
        #if the inclusion operation requires it, negate the output
        if not sortedInclusionCriteria[iOperations]:
            curBoolVec=np.logical_not(curBoolVec)
            
        #after the cull
        remainingStreamIndexes=list(itertools.compress(remainingStreamIndexes,curBoolVec))
        
    #create blank output structure
    outBoolVec=np.zeros(len(streamlines),dtype=bool)
    #these are the winners
    outBoolVec[remainingStreamIndexes]=True
    
    return outBoolVec
                              

def applyNiftiCriteriaToTract_DIPY(streamlines, maskNifti, includeBool, operationSpec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria, uses a range of dipy functions
    #and custom made functions to expedite the typically slow segmentation process
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -maskNifti: a nifti Mask containing only 1s and 0s
    #
    # -includeBool: a boolean indicator of whether you want the associated ROI to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # -operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    #
    # Claude note: this returns exactly what dipy's near_roi returns for the
    # full streamlines and the full mask (see _nearROIExact).
    """
    #still learning how to import from modules
    # from dipy.tracking.utils import near_roi
    # import numpy as np
    # import dipy.tracking.utils as ut
    # import nibabel as nib
    # from nilearn import masking 
    # import scipy
    
    #perform some input checks
    #Claude note: the plane crossing family (planar ROIs only)
    validOperations=["any","all","either_end","both_end"]+list(CROSSING_OPERATIONS)
    if np.logical_not(np.isin(operationSpec, validOperations)):
         raise Exception("applyNiftiCriteriaToTract Error: input operationSpec not understood.")
    
    if np.logical_not(type(maskNifti).__name__=='Nifti1Image'):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not a nifti.")
    
    #Claude note: see _checkMaskNifti for what is checked (and what changed)
    maskData=_checkMaskNifti(maskNifti)
        
    if np.logical_not(isinstance(includeBool, bool )):
        raise Exception("applyNiftiCriteriaToTract Error: input includeBool not a bool.  See input description for usage")
    
    #Claude note: the tract-mask intersection and bounding box subsetting that
    #used to precede this were speed heuristics that changed the result; the
    #exact criterion is now fast enough on its own
    #Claude note: 'cross' goes to the exact plane crossing test, the other
    #modes to the exact near_roi equivalent (see _criterionStreamsBool)
    criteriaStreamsBool=_criterionStreamsBool(streamlines, maskNifti.affine, maskData, operationSpec)
    
    #find the indexes of the original streamlines that the survivors correspond to
    originalIndexes=np.where(criteriaStreamsBool)[0]
    
    if includeBool==True:
        #initalize an out bool vec
        outBoolVec=np.zeros(len(streamlines))
        #set the relevant entries to true
        outBoolVec[originalIndexes]=True
    elif includeBool==False:          
        #initalize an out bool vec
        outBoolVec=np.ones(len(streamlines))
        #set the relevant entries to true
        outBoolVec[originalIndexes]=False
    
    return outBoolVec

def preFiltered_Streamline_near_roi(streamline, roi_coords, tol, mode='any'):
    """
    An alternate version of dipy's streamline_near_roi with fewer checks.
    This should only be run if the input streamline collection has been
    prefiltered in near_roi_fast.
    
    This was done because streamline_near_roi was operating at about 2-3
    iterations per second for some reason.  Possibly a bad version.

    Parameters
    ----------
    streamline : array, shape (N, 3)
        A single streamline.
    roi_coords : array, shape (M, 3)
        ROI coordinates transformed to the streamline coordinate frame..
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, this function returns True..
    mode : string, optional
        One of {"any", "all", "either_end", "both_end"}, where return True
        if:
        "any" : any point is within tol from ROI.
        "all" : all points are within tol from ROI.
        "either_end" : either of the end-points is within tol from ROI
        "both_end" : both end points are within tol from ROI.
        
        The default is 'any'.

    Returns
    -------
    boolOut : boolean
        An indicator of whether the streamline meets the specified criterion

    """
    import numpy as np
    
    dist = cdist(streamline, roi_coords, 'euclidean')
    
    if mode == "any" or mode == "either_end":
        return np.min(dist) <= tol
    else:
        return np.all(np.min(dist, -1) <= tol)
    

def near_roi_fast(streamlines, affine, region_of_interest, tol=None,
             mode="any"):
    """
    Fast version of dipy's "near_roi" function.  Acheives speedup implementing 
    a number of heuristics:
        
        1. Subsets input ROI by intersecting with tract mask.  Reduces number
        of coordinates from ROI involved in cdist computation
        2. Pre-culls candidate streamlines to those within ROI bounding-box.
        Reduces number of streamline-based coordinates in cdist computation.
        3. Within candidate streamlines, eliminates those streamlines
        
        
    Moves much of the functionality of the prototype 
    wmaPyTools.segmentationTools.applyNiftiCriteriaToTract_DIPY_Test into
    a revision of dipy's dipy.tracking.utils.near_roi 
    
    Provide filtering criteria for a set of streamlines based on whether
    they fall within a tolerance distance from an ROI.
    Parameters
    ----------
    streamlines : list or generator
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    region_of_interest : ndarray
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    mode : string, optional
        One of {"any", "all", "either_end", "both_end"}, where return True
        if:
        "any" : any point is within tol from ROI. Default.
        "all" : all points are within tol from ROI.
        "either_end" : either of the end-points is within tol from ROI
        "both_end" : both end points are within tol from ROI.
        "cross" etc. : (Claude note) the plane crossing
            family (cross, not_cross, single_cross, not_single_cross,
            double_cross, not_double_cross): exact, tolerance-free tests on
            how often the streamline crosses the planar ROI (see
            planarROICrossings and CROSSING_OPERATIONS).
    Returns
    -------
    1D array of boolean dtype, shape (len(streamlines), )
    This contains `True` for indices corresponding to each streamline
    that passes within a tolerance distance from the target ROI, `False`
    otherwise.
    
    Claude note: returns exactly what dipy's near_roi returns (as a list),
    via _nearROIExact.  The tol argument is honored as in near_roi.
    """
    print('Input ROI with volume ' + str(np.sum(region_of_interest)) + ' being assessed with mode ' + str(mode))
    #Claude note: 'cross' is the exact, tolerance-free plane
    #crossing test (tol is ignored for it); see planarROICrossings
    if mode in CROSSING_OPERATIONS:
        out=CROSSING_OPERATIONS[mode](_planeCrossingCounts(streamlines, affine, np.asanyarray(region_of_interest)))
    else:
        out=_nearROIExact(streamlines, affine, region_of_interest, tol=tol, mode=mode)
    print(str(int(np.sum(out))) + ' of ' + str(len(out)) + ' streamlines meet the criterion')
    return list(out)

def applyNiftiCriteriaToTract_DIPY_Test(streamlines, maskNifti, includeBool, operationSpec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria, uses a range of dipy functions
    #and custom made functions to expedite the typically slow segmentation process
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -maskNifti: a nifti Mask containing only 1s and 0s
    #
    # -includeBool: a boolean indicator of whether you want the associated ROI to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # -operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    #
    # Claude note: this returns exactly what dipy's near_roi returns for the
    # full streamlines and the full mask (see _nearROIExact).
    """
    #still learning how to import from modules
    # from dipy.tracking.utils import near_roi
    # import numpy as np
    # import dipy.tracking.utils as ut
    # import nibabel as nib
    # from nilearn import masking 
    # import scipy
    # import time
    
    #perform some input checks
    #Claude note: the plane crossing family (planar ROIs only)
    validOperations=["any","all","either_end","both_end"]+list(CROSSING_OPERATIONS)
    if np.logical_not(np.isin(operationSpec, validOperations)):
         raise Exception("applyNiftiCriteriaToTract Error: input operationSpec not understood.")
    
    if np.logical_not(type(maskNifti).__name__=='Nifti1Image'):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not a nifti.")
    
    #Claude note: see _checkMaskNifti for what is checked (and what changed)
    maskData=_checkMaskNifti(maskNifti)
        
    if np.logical_not(isinstance(includeBool, bool )):
        raise Exception("applyNiftiCriteriaToTract Error: input includeBool not a bool.  See input description for usage")
    
    #Claude note: the tract-mask intersection and bounding box node subsetting
    #that used to precede this were speed heuristics that changed the result
    #(see the docstring); the exact criterion is now fast enough on its own
    #start timing
    #Claude note: timed with genUtils.StageTimer (wall-clock; silenced by
    #genUtils.setTimingVerbose(False)) instead of time.process_time
    stageTimer=wmaPyTools.genUtils.StageTimer('applyNiftiCriteriaToTract_DIPY_Test')
    #Claude note: 'cross' goes to the exact plane crossing test, the other
    #modes to the exact near_roi equivalent (see _criterionStreamsBool)
    criteriaStreamsBool=_criterionStreamsBool(streamlines, maskNifti.affine, maskData, operationSpec)
    #stop timing
    criterionName='INCLUSION' if includeBool else 'EXCLUSION'
    stageTimer.finish(lambda modifiedTime: 'Tractogram segmentation complete in ' +str(modifiedTime) +', '+str(np.sum(criteriaStreamsBool)) + ' of ' + str(len(streamlines)) + ' met ' + criterionName + ' criterion')
   

    originalIndexes=np.where(criteriaStreamsBool)[0]
    
    if includeBool==True:
        #initalize an out bool vec
        outBoolVec=np.zeros(len(streamlines)).astype(bool)
        #set the relevant entries to true
        outBoolVec[originalIndexes]=True
    elif includeBool==False:          
        #initalize an out bool vec
        outBoolVec=np.ones(len(streamlines)).astype(bool)
        #set the relevant entries to true
        outBoolVec[originalIndexes]=False
    
    return outBoolVec.astype(bool)

def subsetStreamsByROIboundingBox(streamlines, maskNifti):
    """subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, which will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    """
    #compute distance tolerance
    # from dipy.core.geometry import dist_to_corner
    # import time
    # import wmaPyTools.roiTools 
    
    #begin timing
    #Claude note: timed with genUtils.StageTimer (wall-clock; silenced by
    #genUtils.setTimingVerbose(False)) instead of time.process_time
    stageTimer=wmaPyTools.genUtils.StageTimer('subsetStreamsByROIboundingBox')
    
    #use distance to corner to set tolerance
    dtc = dist_to_corner(maskNifti.affine)
    
    #convert them to subject space
    subjectSpaceBounds=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(maskNifti)
    #expand to accomidate tolerance
    subjectSpaceBounds[0,:]=subjectSpaceBounds[0,:]-dtc
    subjectSpaceBounds[1,:]=subjectSpaceBounds[1,:]+dtc
    
    #map and lambda function to determine whether each streamline is within the bounds
    #Claude note: computed over all nodes at once; equivalent to
    #list(map(lambda streamline: streamlineWithinBounds(streamline,subjectSpaceBounds), streamlines))
    criteriaVec=[]
    for firstIndex, chunk in _streamlineChunks(streamlines):
        chunkNodes, chunkLengths = _flattenStreamlines(chunk)
        nodeInBounds=_boundedNodeMask(chunkNodes, subjectSpaceBounds)
        streamIDs=np.repeat(np.arange(len(chunkLengths)), chunkLengths)
        criteriaVec+=list(np.bincount(streamIDs[nodeInBounds], minlength=len(chunkLengths))>0)
    
    #stop timing
    stageTimer.finish(lambda modifiedTime: 'Tractogram subseting complete in ' +str(modifiedTime) +', '+str(sum(criteriaVec)) + ' of ' + str(len(streamlines)) + ' within mask boundaries')
    return criteriaVec

def subsetStreamsNodesByROIboundingBox(streamlines, maskNifti):
    """subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, which will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    """
    #compute distance tolerance
    # from dipy.core.geometry import dist_to_corner
    # from dipy.tracking.streamline import Streamlines
    # import numpy as np
    # import time
    # import wmaPyTools.roiTools  
    
    #Claude note: the work is done over all nodes at once, in
    #_subsetNodesByROIboundingBox, which also does the timing and reporting.
    #The result is the same as the per-streamline
    #list(map(lambda streamline: streamlineNodesWithinBounds(streamline,subjectSpaceBounds), streamlines))
    outIndexes, outNodes, outLengths = _subsetNodesByROIboundingBox(streamlines, maskNifti)
    #Claude note: keep the input's node dtype (e.g. float32), as building
    #Streamlines from the per-streamline subsets did
    if isinstance(streamlines, ArraySequence):
        nodeDtype=streamlines._data.dtype
    else:
        nonEmpty=[iStream for iStream in streamlines if len(iStream)>0]
        nodeDtype=np.asarray(nonEmpty[0]).dtype if len(nonEmpty)>0 else np.float64
    outStreams=Streamlines()
    if len(outLengths)>0:
        outStreams._data=outNodes.astype(nodeDtype)
        outStreams._lengths=outLengths
        outStreams._offsets=(np.cumsum(outLengths)-outLengths).astype(np.intp)
    return outIndexes, outStreams

def streamlineWithinBounds(streamline,bounds):
    """ determine whether **any** node of the input streamline is within the specified bounds.
    Returns a single bool
    Args:
        -streamline: an n by d shaped array where n=the number of nodes and d = the dimensionality of the streamline
        
        -bounds: a 2 x d array specifying the coordinate boundaries (in the pertinant space of the streamline) for assesment

    Output:
        withinBoundsBool:  a boolean value indicating whether the input streamline satisfies the within-bounds criteria
    
    """
    # import numpy as np

    #see which nodes are between the bounds
    nodeCriteria=np.asarray([np.logical_and(streamline[:,iDems]>bounds[0,iDems],streamline[:,iDems]<bounds[1,iDems]) for iDems in list(range(bounds.shape[1])) ])
    
    #return true if any single node is between all three sets of bounds
    return np.any(np.all(nodeCriteria,axis=0))

def streamlineNodesWithinBounds(streamline,bounds):
    """ determine whether **any** node of the input streamline is within the specified bounds.
    Returns the nodes within the bounds.
    Args:
        streamline: an n by d shaped array where n=the number of nodes and d = the dimensionality of the streamline
        bounds: a 2 x d array specifying the coordinate boundaries (in the pertinant space of the streamline) for assesment

    Output:
        withinBoundsNodes:  an array of the nodes 
    
    """
    # import numpy as np
    
    #see which nodes are between the bounds
    nodeCriteria=np.asarray([np.logical_and(streamline[:,iDems]>bounds[0,iDems],streamline[:,iDems]<bounds[1,iDems]) for iDems in list(range(bounds.shape[1])) ])
    
    #return return the viable nodes, could and often will be empty
    return streamline[np.all(nodeCriteria,axis=0)]

def applyEndpointCriteria(streamlines,planarROI,requirement,whichEndpoints):
    """ apply a relative location criteria to the endpoints of all streamlines in a collection of streamlines
    Args:
        streamlines: streamlines to be segmented from
        planarROI: the planar ROI relative to which the streamlines' endpoints' locations should be assessed
        requirement:  A relative anatomical positional term
        whichEndpoints:  whether this criteria should apply to 'both', 'one', or 'neither' endpoint
    
    Output:
        streamBool:  a boolean vector indicating which streamlines meet the specified criteria.
    """
    # import numpy as np
    # import nibabel as nib
    # import wmaPyTools.roiTools 
    
    #Claude note: the image's extent as its outer voxel faces, i.e. half a
    #voxel beyond the outermost voxel centres.
    convertedBoundCoords=wmaPyTools.roiTools._imageFaceBounds(planarROI.affine, planarROI.shape)

    #implement test to determine if input planar roi is indeed planar
    #get coordinates of mask voxels in image space
    
    from dipy.tracking.utils import apply_affine
    planeSubjCoords=apply_affine(planarROI.affine, np.array(np.where(np.asanyarray(planarROI.dataobj))).T)
    #find the unique values of img space coordinates for each dimension
    uniqueCoordCounts=[len(np.unique(planeSubjCoords[:,iCoords])) for iCoords in list(range(planeSubjCoords.shape[1]))]
    #one of them should be singular in the case of a planar roi, throw an error if not
    if not np.any(np.isin(uniqueCoordCounts,1)):
        raise ValueError('input ROI not planar (i.e. single voxel thick for True values)')
    
    planeCoord=planeSubjCoords[0,np.where(np.equal(uniqueCoordCounts,1))[0][0]]

    #set up the dictionary for boundaries
    positionTermsDict={'superior': np.max(convertedBoundCoords[:,2]),
                      'inferior': np.min(convertedBoundCoords[:,2]),
                      'medial':   np.min(convertedBoundCoords[np.min(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
                      'lateral': np.max(convertedBoundCoords[np.max(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
                      'anterior': np.max(convertedBoundCoords[:,1]),
                      'posterior': np.min(convertedBoundCoords[:,1]),
                      'rostral': np.max(convertedBoundCoords[:,1]),
                      'caudal': np.min(convertedBoundCoords[:,1]),
                      'left': np.min(convertedBoundCoords[:,0]),
                      'right': np.max(convertedBoundCoords[:,0])}
    
    dimensionDict={'superior': 2,
                   'inferior': 2,
                   'medial':   0,
                   'lateral': 0,
                   'anterior': 1,
                   'posterior': 1,
                   'rostral': 1,
                   'caudal': 1,
                   'left': 0,
                   'right': 0}        
    
    #throw an error if there's a mismatch 
    if  np.logical_not(np.where(np.equal(uniqueCoordCounts,1))[0][0]==dimensionDict[requirement]):
        raise Exception("applyEndpointCriteria Error: input relative position " + requirement + " not valid for input plane.")
    
    #create blank structure for endpoints
    endpoints=np.zeros((len(streamlines),6))
    #get the endpoints, taken from
    #https://github.com/dipy/dipy/blob/f149c756e09f172c3b77a9e6c5b5390cc08af6ea/dipy/tracking/utils.py#L708
    #Claude note: gathered from the concatenated node array in one step,
    #rather than streamline by streamline (same values: both are float64
    #copies of the nodes).  Empty streamlines (only possible in a list) make
    #the original loop raise, so it is still used in that case.
    allLengths = _streamlineLengths(streamlines)
    if len(allLengths)>0 and np.all(allLengths>0):
        endpoints[:,0:3]=_nodesAtPositions(streamlines, lambda lengths: np.zeros(len(lengths),dtype=np.intp))
        endpoints[:,3:6]=_nodesAtPositions(streamlines, lambda lengths: lengths-1)
    else:
        for iStreamline in range(len(streamlines)):
            #remember, first 3 = endpoint 1, last 3 = endpoint 2    
            endpoints[iStreamline,:]= np.concatenate([streamlines[iStreamline][0,:], streamlines[iStreamline][-1,:]])
    
    Endpoints1=endpoints[:,0:3]
    Endpoints2=endpoints[:,3:7]
    
    #sort the bounds
    sortedBounds=np.sort([planeCoord,positionTermsDict[requirement]])
    #get the relevant image dimension
    spaceDim=np.where(np.equal(uniqueCoordCounts,1))[0][0]
    
    #apply the criteria to both endpoints
    endpoints1Criteria=np.logical_and(np.greater(Endpoints1[:,spaceDim],sortedBounds[0]),np.less(Endpoints1[:,spaceDim],sortedBounds[1]))
    endpoints2Criteria=np.logical_and(np.greater(Endpoints2[:,spaceDim],sortedBounds[0]),np.less(Endpoints2[:,spaceDim],sortedBounds[1]))
    
    whichEndpointsDict={'neither': 0,
                        'one': 1,
                        'both':   2}
    
    
    #sum the two endpoint criterion vectors
    sumVec=np.add(endpoints1Criteria,endpoints2Criteria,dtype=int)
    
    #see where the target value is met
    targetMetVec=sumVec==whichEndpointsDict[whichEndpoints]
    
    return targetMetVec
    
def applyMidpointCriteria(streamlines,planarROI,requirement):
    """ apply a relative location criteria to the midpoints of all streamlines in a collection of streamlines
    Args:
        streamlines: streamlines to be segmented from
        planarROI: the planar ROI relative to which the streamlines' midpoints' locations should be assessed
        requirement:  A relative anatomical positional term

    Output:
        streamBool:  a boolean vector indicating which streamlines meet the specified criteria.
    """
    # import numpy as np
    # import nibabel as nib
    # import wmaPyTools.roiTools  
    
    #Claude note: the image's extent as its outer voxel faces, i.e. half a
    #voxel beyond the outermost voxel centres.
    convertedBoundCoords=wmaPyTools.roiTools._imageFaceBounds(planarROI.affine, planarROI.shape)

    #implement test to determine if input planar roi is indeed planar
    #get coordinates of mask voxels in image space
    
    from dipy.tracking.utils import apply_affine
    planeSubjCoords=apply_affine(planarROI.affine, np.array(np.where(np.asanyarray(planarROI.dataobj))).T)
    #find the unique values of img space coordinates for each dimension
    uniqueCoordCounts=[len(np.unique(planeSubjCoords[:,iCoords])) for iCoords in list(range(planeSubjCoords.shape[1]))]
    #one of them should be singular in the case of a planar roi, throw an error if not
    if not np.any(np.isin(uniqueCoordCounts,1)):
        raise ValueError('input ROI not planar (i.e. single voxel thick for True values)')
    
    planeCoord=planeSubjCoords[0,np.where(np.equal(uniqueCoordCounts,1))[0][0]]

    #set up the dictionary for boundaries
    positionTermsDict={'superior': np.max(convertedBoundCoords[:,2]),
                      'inferior': np.min(convertedBoundCoords[:,2]),
                      'medial':   np.min(convertedBoundCoords[np.min(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
                      'lateral': np.max(convertedBoundCoords[np.max(np.abs(convertedBoundCoords[:,0]))==np.abs(convertedBoundCoords[:,0]),0]),
                      'anterior': np.max(convertedBoundCoords[:,1]),
                      'posterior': np.min(convertedBoundCoords[:,1]),
                      'rostral': np.max(convertedBoundCoords[:,1]),
                      'caudal': np.min(convertedBoundCoords[:,1]),
                      'left': np.min(convertedBoundCoords[:,0]),
                      'right': np.max(convertedBoundCoords[:,0])}
    
    dimensionDict={'superior': 2,
                   'inferior': 2,
                   'medial':   0,
                   'lateral': 0,
                   'anterior': 1,
                   'posterior': 1,
                   'rostral': 1,
                   'caudal': 1,
                   'left': 0,
                   'right': 0}        
    
    #throw an error if there's a mismatch 
    if  np.logical_not(np.where(np.equal(uniqueCoordCounts,1))[0][0]==dimensionDict[requirement]):
        raise Exception("applyEndpointCriteria Error: input relative position " + requirement + " not valid for input plane.")

    #use dipy to get the midpoints    
    from dipy.segment.featurespeed import MidpointFeature
    feature = MidpointFeature()
    #Claude note: MidpointFeature.extract returns node len//2 as float32.  That
    #is gathered here from the concatenated node array in one step instead;
    #the original per-streamline map is kept for the degenerate inputs
    #(fewer than 2 streamlines, empty streamlines) where its squeeze/indexing
    #behaves specially
    allLengths = _streamlineLengths(streamlines)
    if len(allLengths)>1 and np.all(allLengths>0):
        #Claude note: these are exact float64 copies of the nodes, so this
        #cast rounds exactly as extract's own cast to float32 does
        midpoints=_nodesAtPositions(streamlines, lambda lengths: lengths//2).astype(np.float32)
    else:
        midpoints = np.squeeze(np.asarray(list(map(feature.extract, streamlines))))
    
    
    #sort the bounds
    sortedBounds=np.sort([planeCoord,positionTermsDict[requirement]])
    #get the relevant image dimension
    spaceDim=np.where(np.equal(uniqueCoordCounts,1))[0][0]
    
    #apply the criteria to both endpoints
    midpointsCriteria=np.logical_and(np.greater(midpoints[:,spaceDim],sortedBounds[0]),np.less(midpoints[:,spaceDim],sortedBounds[1]))

    return midpointsCriteria

def maskMatrixByBoolVec(dipyGrouping,boolVec):
    """ recompute a connectivity matrix for a specified subset of the streamlines
    Args:
        dipyGrouping: the grouping output of the dipy utils.connectivity_matrix [WITH symmetric=False]
        boolVec: a boolean vector indicating which streamlines to consider in this computation
        

    Output:
        matrixSubset:  a matrix whose entries have been altered in accordance with the input
    """

    # import numpy as np 

    #get the number of unique labels
    uniqueLabels=np.unique(np.asarray(list(dipyGrouping.keys())))
    #get the indexes of the valid streamlines from the input boolVec
    #concatenate to force to 1d array
    validStreams=np.concatenate(np.where(boolVec))
    #inatialize blank matrix object
    matrixSubset=np.zeros((len(uniqueLabels),len(uniqueLabels)))
    
    #iterate over the dictionary keys/ pairings
    for iPairings in range(len(list(dipyGrouping.keys()))):
        #get the current dictionary key entry
        currKey=list(dipyGrouping.keys())[iPairings]
        #get the length of the intersction of the boolVec streamlines and the current streamlines
        #and place it in the matrix location
        matrixSubset[currKey[0],currKey[1]]=len(np.intersect1d(dipyGrouping[currKey],validStreams))
    
    return matrixSubset.astype(int)

def subsetStreamsNodesByROIboundingBox_test(streamlines, maskNifti):
    """subsetStreamsByROIboundingBox(streamlines, maskNifti):
    #subsets the input set of streamlines to only those that have nodes within the box
    #
    # INPUTS
    #
    # -streamlines: streamlines to be subset
    #
    # -maskNifti:  the mask nifti from which a bounding box is to be extracted, which will be used to subset the streamlines
    #
    # OUTPUTS
    #
    # -criteriaVec:  a boolean vector indicating which streamlines contain nodes within the bounding box.
    #
    """
    #compute distance tolerance
    # from dipy.core.geometry import dist_to_corner
    # from dipy.tracking.streamline import Streamlines
    # import numpy as np
    # import wmaPyTools.roiTools  
    
    import time
    
    #begin timing
    #Claude note: timed with genUtils.StageTimer (wall-clock; silenced by
    #genUtils.setTimingVerbose(False)) instead of time.process_time
    stageTimer=wmaPyTools.genUtils.StageTimer('subsetStreamsNodesByROIboundingBox_test')
    
    #use distance to corner to set tolerance
    dtc = dist_to_corner(maskNifti.affine)
    
    #convert them to subject space
    subjectSpaceBounds=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(maskNifti)
    #expand to accomidate tolerance
    subjectSpaceBounds[0,:]=subjectSpaceBounds[0,:]-dtc
    subjectSpaceBounds[1,:]=subjectSpaceBounds[1,:]+dtc
    
    nodeArray=[]
    for iStreamlines in streamlines:
        nodeArray.append(iStreamlines[np.all(np.asarray([np.logical_and(iStreamlines[:,iDems]>subjectSpaceBounds[0,iDems],iStreamlines[:,iDems]<subjectSpaceBounds[1,iDems]) for iDems in list(range(subjectSpaceBounds.shape[1])) ]),axis=0)])
 
    
    #map and lambda function to extract the nodes within the bounds
    outIndexes=np.where(list(map(lambda x: x.size>0, nodeArray)))[0]
    outStreams=Streamlines(nodeArray)
    
        
    #map and lambda function to extract the nodes within the bounds
    #criteriaVec=list(map(lambda streamline: streamlineNodesWithinBounds(streamline,subjectSpaceBounds), streamlines))
    #outIndexes=np.where(list(map(lambda x: x.size>0, criteriaVec)))[0]
    #outStreams=Streamlines(criteriaVec)
    
    
    #stop timing
    stageTimer.finish(lambda modifiedTime: 'Tractogram subseting complete in ' +str(modifiedTime) +', '+str(len(outIndexes)) + ' of ' + str(len(streamlines)) + ' within mask boundaries')
    return outIndexes, outStreams   
    

def applyNiftiCriteriaToTract_DIPY_Cython(streamlines, maskNifti, includeBool, operationSpec):
    """segmentTractMultiROI(streamlines, roisvec, includeVec, operationsVec):
    #Iteratively applies ROI-based criteria, uses a range of dipy functions
    #and custom made functions to expedite the typically slow segmentation process
    #
    #adapted from https://github.com/dipy/dipy/blob/master/dipy/tracking/streamline.py#L200
    #basically a variant of
    #https://github.com/DanNBullock/wma/blob/33a02c0373d6742ddf07fd8ac3c8481662577743/utilities/wma_SegmentFascicleFromConnectome.m
    #
    #INPUTS
    #
    # -streamlines: appropriately formatted list of candidate streamlines, e.g. a candidate tractome
    #
    # -maskNifti: a nifti Mask containing only 1s and 0s
    #
    # -includeBool: a boolean indicator of whether you want the associated ROI to act as an INCLUSION or EXCLUSION ROI (True=inclusion)
    #
    # -operationSpec: operation specification, one following instructions on which streamline nodes to asses (and how)
    #    "any" : any point is within tol from ROI. Default.
    #    "all" : all points are within tol from ROI.
    #    "either_end" : either of the end-points is within tol from ROI
    #    "both_end" : both end points are within tol from ROI.
    #
    # OUTPUTS
    #
    # - outBoolVec: boolean vec indicating streamlines that survived operation
    """
    #still learning how to import from modules
    # from dipy.tracking.utils import near_roi
    # import numpy as np
    # import dipy.tracking.utils as ut
    # import nibabel as nib
    # from nilearn import masking 
    # import scipy
    # from dipy.tracking.vox2track import _streamlines_in_mask
    
    #perform some input checks
    validOperations=["any","all","either_end","both_end"]
    if np.logical_not(np.isin(operationSpec, validOperations)):
         raise Exception("applyNiftiCriteriaToTract Error: input operationSpec not understood.")
    
    if np.logical_not(type(maskNifti).__name__=='Nifti1Image'):
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not a nifti.")
    
    #the conversion to int may cause problems if the input isn't convertable to int.  Then again, the point of this is to raise an error, so...
    elif np.logical_not(np.all(np.unique(np.asanyarray(maskNifti.dataobj)).astype(int)==[0, 1])): 
        raise Exception("applyNiftiCriteriaToTract Error: input maskNifti not convertable to 0,1 int mask.  Likely not a mask.")
        
    if np.logical_not(isinstance(includeBool, bool )):
        raise Exception("applyNiftiCriteriaToTract Error: input includeBool not a bool.  See input description for usage")
        
    lin_T, offset =ut._mapping_to_voxel(maskNifti.affine)
    criteriaStreamsBool=_streamlines_in_mask( list(streamlines), np.asanyarray(maskNifti.dataobj).astype(np.uint8), lin_T, offset)
    
    if includeBool==True:
        #initalize an out bool vec
        outBoolVec=np.zeros(len(streamlines), dtype=bool)
        #set the relevant entries to true
        outBoolVec[criteriaStreamsBool.astype(bool)]=True
    elif includeBool==False:          
        #initalize an out bool vec
        outBoolVec=np.ones(len(streamlines), dtype=bool)
        #set the relevant entries to true
        outBoolVec[criteriaStreamsBool.astype(bool)]=False
    
    return outBoolVec

def segmentTractUsingTractMask(streamlines,singleTractMask):
    """
    

    Parameters
    ----------
    streamlines : TYPE
        DESCRIPTION.
    singleTractMask : TYPE
        DESCRIPTION.

    Returns
    -------
    outBoolVec : TYPE
        DESCRIPTION.

    """
    # import wmaPyTools.roiTools
    
    tractProbabilityMap2SegCriteria(singleTractMask)
    
    outBoolVec='incomplete'
    
    return outBoolVec

def tractProbabilityMap2SegCriteria(singleTractProbMap):
    """
    This function converts a nifti-based probablity map (presumably of an atlas
    maping of a tract) to a series of segmentation critersegmentedTract = StatefulTractogram(testStreamlines.streamlines[comboROIBool], testAnatomy, Space.RASMM)
    save_tractogram( segmentedTract,'testSegmentedTract.trk')ia
    
    You know what would make this process a lot easier?
    If atlases included endpoint masks, and not just allNode masks

    Parameters
    ----------
    singleTractProbMap : TYPE
        DESCRIPTION.

    Returns
    -------
    criteriaDict : dictionary
        DESCRIPTION.

    """
    import nibabel as nib
    import numpy as np
    import wmaPyTools.roiTools  
    
    if isinstance(singleTractProbMap,str):
        singleTractProbMap=nib.load(singleTractProbMap)
        
    #if it turns out we need to inflate the input mask, do it here using scipy dilate
    
    #create a bounding box of planarROIS.  These will be exclusion criteria as
    #nothing from the tract can traverse them.  Theoretically we could require
    #that all streamlines contain all nodes within the mask, but this is (1)
    #probably overly constrictive and (2) computationally intensive    
    boundaryPlanes=wmaPyTools.roiTools.boundaryROIPlanesFromMask(singleTractProbMap)
  
    
    #go ahead and get the boundaries in subject space as well
    #we'll be using these to find the primary axis along which we'll be making 
    #planar ROIs.  Alternatively, you could make ~3 spherical ROIS that span the
    #entrity of the mask, one at each end of the thresheld density, and one in the middle
    # but this presumes a bit about the morphology of the tract
    # the arcuate is going to be a real test case for this method
    maskBoundaries=wmaPyTools.roiTools.subjectSpaceMaskBoundaryCoords(singleTractProbMap)
    #compute the span of values for each of these
    #Claude note: maskBoundaries is 2 x 3 (row 0 the minima, row 1 the maxima),
    #so the span of each dimension is the difference of the rows
    dimSpans=np.abs(maskBoundaries[1,:]-maskBoundaries[0,:])
    primaryDim=np.where(dimSpans==np.max(dimSpans))[0][0]
    #kind of an inference here, but ok
    dimDict={0:'x',1:'y',2:'z'}
    letterDim=dimDict[primaryDim]
    
    #here we're arbitraily going to set a span frequency which will serve as 
    #the spacing between ROIs
    planarROISpacing=7
    #establish the coordinates of the planar ROIs
    #Claude note: the planes run from the mask's minimum to its maximum along the
    #primary dimension (the bounds, not the span, give their positions)
    planeCoords=np.arange(np.min(maskBoundaries[:,primaryDim]),np.max(maskBoundaries[:,primaryDim]),planarROISpacing)
    #maybe throw an error here if you end up with less than three values.  This method
    #is probably ill advised for tracts with length of 21 or less (i.e. ufibers)
    if len(planeCoords)<3:
        raise ValueError('dimensional span of ' + str(dimSpans[primaryDim]) + ' for dimension ' + letterDim + ' too small to create reasonable planar ROIs')
    #This is going to cause problems for curved tracts like the arcuate
    #you could imagine the plane hitting the extreme rear of the arc, and imposing
    #an inclusion criterion that ends up shaving everything that curved ahead 
    #of that point.  That's a problem
    #So here's what's going to happen in that case:  There will be a plane
    #which is outside the bounds formed by BOTH endpoint clusters
    
    
    #ACTUALLY, MAYBE IT DOESN'T MATTER IF WE ARE PASSED A DENSITY/PROBABILITY MASK
    #if we're doing this cleverly, we can assume that the input is density/probabilty
    #based, so here we can appply some sort of thesholding to extract the
    # "core" of the tract.
    
    #here we'll arbitraily set a threshold value
    #thresholdProportion=.7
    #next we find where this is in the unique values of the datablock
    #and set it as our threshold value
    #threshVal=np.unique(np.asanyarray(singleTractProbMap.dataobj))[np.floor(len(np.unique(np.asanyarray(singleTractProbMap.dataobj)))*thresholdProportion).astype(int)]
    
    #lets get to making the planar coordinates
    planarInclusionROIs=[]
    for iROIcords in planeCoords[1:-1]:
        planarInclusionROIs.append(wmaPyTools.roiTools.makePlanarROI(singleTractProbMap, iROIcords, letterDim))
        
    #Claude note: the 'any' entry has to exist before its keys can be filled
    criteriaDict={'any':{}}
    criteriaDict['any']['include']=planarInclusionROIs
    #there's abit of weirdness here, in that the exclusion rois have names.  This will
    #undoubtedly cause standardization errors later
    criteriaDict['any']['exclude']=boundaryPlanes

    return criteriaDict

def densityMaskToSegCriteria(densityMask):
    
    # import numpy as np
    # import nibabel as nib
    # import scipy
    # import copy
    
    #MAJOR INSIGHT:  it is never the case that endpoints can be in the "core" of a
    #tract.  Ergo, you can erode the mask a bit and use it as an endpoint exclusion criterion, 
    # but also as a traversal criterion
    coreMask=scipy.ndimage.binary_erosion(np.asanyarray(densityMask.dataobj), iterations=2)
    inflatedMask=scipy.ndimage.binary_dilation(np.asanyarray(densityMask.dataobj), iterations=1)
    endpointsMask=copy.deepcopy(inflatedMask)
    endpointsMask[coreMask]=False
    
    endpointCriteria=nib.Nifti1Image(endpointsMask, densityMask.affine, densityMask.header)
    coreIntersectCriteria=nib.Nifti1Image(coreMask, densityMask.affine, densityMask.header)
    
    criteriaDict={}
    
    
    
    return criteriaDict

