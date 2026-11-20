# -*- coding: utf-8 -*-
"""
Claude note: synthetic streamline generators with known ground truth.

Two families:

* generateBundle: canonical bundle shapes (straight, arc / C, hook, U-fibre,
  fan, commissural, loop) with tunable count, spread, smooth noise, length,
  node spacing (including uneven spacing), random reversal and outliers,
  placed anywhere by a rigid transform.  Returns the streamlines and a truth
  dictionary: the centreline, neck, endpoint regions, which streamlines were
  reversed and which are outliers.

* generatePlaneCrossingCases: streamlines around a plane, in classes chosen to
  probe crossing / kissing edge cases (shallow and steep crossings, kisses at
  many distances, streamlines ending near the plane, U-turns, repeated
  crossings, long runs parallel to the plane), with each streamline's true
  crossing count, plus a planar ROI nifti on a voxel grid aligned with the
  plane (oblique in mm space if the transform rotates it).

Everything is deterministic for a given seed.  Generators never depend on the
segmentation code they are meant to test: crossing counts come from the
construction (and are checked against the sampled polyline in the plane's own
frame).

All coordinates are in mm.
"""

import numpy as np
import nibabel as nib
from dipy.tracking.streamline import Streamlines


# Claude note: -------------------------------------------------------------
# shared geometry
# --------------------------------------------------------------------------

def _rotationMatrix(rotation):
    """
    Claude note: a 3x3 rotation from None (identity), a 3x3 matrix, or three
    Euler angles in degrees (applied about x, then y, then z).
    """
    if rotation is None:
        return np.eye(3)
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape == (3, 3):
        return rotation
    ax, ay, az = np.deg2rad(rotation)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _arcLength(points):
    """Claude note: cumulative arc length along a polyline (starts at 0)."""
    return np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _resampleByArcLength(densePoints, spacing, spacingJitter, rng):
    """
    Claude note: sample a densely given curve at arc-length steps of
    spacing * (1 + U(-spacingJitter, spacingJitter)), always keeping both ends.
    spacingJitter in [0, 1) gives uneven node spacing.
    """
    arc = _arcLength(densePoints)
    total = arc[-1]
    positions = [0.]
    while True:
        step = spacing * (1. + rng.uniform(-spacingJitter, spacingJitter)) if spacingJitter > 0 else spacing
        if positions[-1] + step >= total - 1e-9:
            break
        positions.append(positions[-1] + step)
    positions.append(total)
    positions = np.asarray(positions)
    return np.column_stack([np.interp(positions, arc, densePoints[:, iDim]) for iDim in range(3)])


def _frames(centreline):
    """
    Claude note: tangent and two normals along a centreline, by parallel
    transport (so the frame doesn't twist on planar curves).
    """
    tangents = np.gradient(centreline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    #Claude note: a starting normal not parallel to the first tangent
    helper = np.array([0., 0., 1.]) if abs(tangents[0, 2]) < 0.9 else np.array([1., 0., 0.])
    normal1 = np.cross(tangents[0], helper)
    normal1 /= np.linalg.norm(normal1)
    normals1 = [normal1]
    for iPoint in range(1, len(centreline)):
        #Claude note: remove the new tangent's component and renormalise
        candidate = normals1[-1] - np.dot(normals1[-1], tangents[iPoint]) * tangents[iPoint]
        normals1.append(candidate / np.linalg.norm(candidate))
    normals1 = np.asarray(normals1)
    normals2 = np.cross(tangents, normals1)
    return tangents, normals1, normals2


# Claude note: -------------------------------------------------------------
# canonical bundle shapes
# --------------------------------------------------------------------------

#Claude note: the shapes generateBundle knows
BUNDLE_SHAPES = ('straight', 'arc', 'hook', 'ufiber', 'fan', 'commissural', 'loop')


def _centreline(shape, length, nDense=2000):
    """
    Claude note: a dense centreline of the given shape, before scaling to
    the requested arc length.  Shapes (in the canonical, untransformed frame):
      straight    : along +y
      arc         : a half circle in the x-y plane (a C; the dominant travel
                    axis changes along the tract)
      hook        : straight along +y, then a tight 180-degree turn (uncinate-like)
      ufiber      : down, across, up again (a short U in the y-z plane)
      fan         : straight along +y (the fanning is in the spread profile)
      commissural : along +x through the midline x=0, ends curving up in z
      loop        : one and a quarter turns of a circle, rising slowly in z
    """
    t = np.linspace(0., 1., nDense)
    if shape in ('straight', 'fan'):
        points = np.column_stack([np.zeros_like(t), t, np.zeros_like(t)])
    elif shape == 'arc':
        angle = np.pi * t
        points = np.column_stack([-np.cos(angle), np.sin(angle), np.zeros_like(t)])
    elif shape == 'hook':
        straightPart = 0.65
        points = np.zeros((nDense, 3))
        isStraight = t <= straightPart
        points[isStraight, 1] = t[isStraight] / straightPart
        angle = np.pi * (t[~isStraight] - straightPart) / (1. - straightPart)
        radius = 0.15
        points[~isStraight, 0] = radius - radius * np.cos(angle)
        points[~isStraight, 1] = 1. + radius * np.sin(angle)
    elif shape == 'ufiber':
        angle = np.pi * t
        points = np.column_stack([np.zeros_like(t), -np.cos(angle), -0.6 * np.sin(angle)])
    elif shape == 'commissural':
        x = 2. * t - 1.
        points = np.column_stack([x, np.zeros_like(t), 0.35 * x ** 2])
    elif shape == 'loop':
        angle = 2.5 * np.pi * t
        points = np.column_stack([np.cos(angle), np.sin(angle), 0.15 * t])
    else:
        raise ValueError('unknown bundle shape ' + str(shape) + '; expected one of ' + str(BUNDLE_SHAPES))
    #Claude note: scale to the requested arc length, centred on its midpoint
    points = points * (length / _arcLength(points)[-1])
    return points - points[nDense // 2]


def generateBundle(shape='straight', nStreamlines=100, length=80., spread=2., noise=0.5,
                   spacing=1., spacingJitter=0., reverseFraction=0., outlierFraction=0.,
                   fanRatio=0.2, rotation=None, translation=(0., 0., 0.), seed=0):
    """
    Claude note: a synthetic bundle of one canonical shape, with ground truth.

    Parameters
    ----------
    shape : str
        One of BUNDLE_SHAPES (see _centreline for what each looks like).
    nStreamlines : int
        Number of streamlines, outliers included.
    length : float
        Centreline arc length in mm.
    spread : float
        Standard deviation (mm) of each streamline's constant offset from the
        centreline, across the bundle.
    noise : float
        Amplitude (mm) of each streamline's smooth wobble (a few low-frequency
        sinusoids) on top of its offset.
    spacing : float
        Mean node spacing in mm, measured along each streamline.
    spacingJitter : float in [0, 1)
        Each step is spacing * (1 + U(-spacingJitter, spacingJitter)): 0 gives
        evenly spaced nodes, larger values unevenly spaced ones.
    reverseFraction : float in [0, 1]
        Fraction of streamlines whose node order is reversed.
    outlierFraction : float in [0, 1]
        Fraction of streamlines replaced by outliers: smooth random curves of
        similar length starting near the bundle but going elsewhere.
    fanRatio : float
        For shape='fan': the bundle's spread at its start relative to its end
        (it converges at the start).
    rotation : None, 3x3 array, or three Euler angles in degrees
        Rigid rotation applied to everything, about the origin.
    translation : 3 floats
        Applied after the rotation.
    seed : int
        Random seed; the same arguments and seed give the same output.

    Returns
    -------
    streamlines : dipy Streamlines
    truth : dict
        'shape', 'centreline' (dense (N,3)), 'neck' (centreline midpoint),
        'neckTangent', 'startRegion' / 'endRegion' (centreline ends),
        'reversed' (bool per streamline: node order runs end -> start),
        'outlier' (bool per streamline), 'spacing', 'spacingJitter'.
    """
    rng = np.random.default_rng(seed)
    centreline = _centreline(shape, length)
    tangents, normals1, normals2 = _frames(centreline)
    nDense = len(centreline)
    t = np.linspace(0., 1., nDense)
    #Claude note: the fan converges at the start
    spreadProfile = (fanRatio + (1. - fanRatio) * t) if shape == 'fan' else np.ones(nDense)
    nOutliers = int(round(outlierFraction * nStreamlines))
    isOutlier = np.zeros(nStreamlines, dtype=bool)
    isOutlier[rng.choice(nStreamlines, nOutliers, replace=False)] = True
    isReversed = rng.random(nStreamlines) < reverseFraction
    streams = []
    for iStream in range(nStreamlines):
        if isOutlier[iStream]:
            #Claude note: an outlier: a smooth random curve from near the start
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            bend = rng.normal(size=3) * 0.3
            path = np.outer(t, direction) + np.outer(t ** 2, bend)
            path = path * (length / _arcLength(path)[-1]) + centreline[0] + rng.normal(0, spread, size=3)
        else:
            #Claude note: a constant offset in the normal plane, scaled by the
            #spread profile, plus a smooth wobble in both normal directions
            offset1, offset2 = rng.normal(0., spread, size=2)
            wobble1 = np.zeros(nDense)
            wobble2 = np.zeros(nDense)
            for frequency in (1, 2, 3):
                wobble1 += noise / frequency * np.sin(np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
                wobble2 += noise / frequency * np.sin(np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
            path = (centreline
                    + (offset1 * spreadProfile + wobble1)[:, None] * normals1
                    + (offset2 * spreadProfile + wobble2)[:, None] * normals2)
        nodes = _resampleByArcLength(path, spacing, spacingJitter, rng)
        if isReversed[iStream]:
            nodes = nodes[::-1]
        streams.append(nodes)
    #Claude note: place everything with the rigid transform
    rotationMatrix = _rotationMatrix(rotation)
    translation = np.asarray(translation, dtype=float)
    place = lambda points: np.asarray(points) @ rotationMatrix.T + translation
    streams = [place(s).astype(np.float32) for s in streams]
    truth = {'shape': shape,
             'centreline': place(centreline),
             'neck': place(centreline[nDense // 2]),
             'neckTangent': tangents[nDense // 2] @ rotationMatrix.T,
             'startRegion': place(centreline[0]),
             'endRegion': place(centreline[-1]),
             'reversed': isReversed,
             'outlier': isOutlier,
             'spacing': spacing,
             'spacingJitter': spacingJitter}
    return Streamlines(streams), truth


def referenceForStreamlines(streamlines, voxelSize=1., padding=5., rotation=None):
    """
    Claude note: a blank reference nifti covering the streamlines, with
    isotropic voxels of the given size; if rotation is given, the voxel grid
    is rotated (an oblique image in mm space).

    Returns
    -------
    reference : nibabel.Nifti1Image (uint8 zeros)
    """
    rotationMatrix = _rotationMatrix(rotation)
    nodes = np.concatenate([np.asarray(s, dtype=float) for s in streamlines])
    #Claude note: extent in the grid's own (rotated) axes
    local = nodes @ rotationMatrix
    low = local.min(axis=0) - padding
    high = local.max(axis=0) + padding
    shape = np.ceil((high - low) / voxelSize).astype(int) + 1
    affine = np.eye(4)
    affine[:3, :3] = voxelSize * rotationMatrix
    affine[:3, 3] = rotationMatrix @ low
    return nib.Nifti1Image(np.zeros(tuple(shape), dtype=np.uint8), affine)


# Claude note: -------------------------------------------------------------
# plane crossing edge cases
# --------------------------------------------------------------------------

#Claude note: the classes generatePlaneCrossingCases builds, with the
#crossing count each one has by construction
PLANE_CASE_CLASSES = {
    'cross1': 1,          # crosses once, at a random angle (down to very shallow)
    'cross2': 2,          # dips through the plane and comes back (U-turn)
    'cross3': 3,          # crosses three times (snakes through)
    'cross4': 4,
    'kiss': 0,            # approaches to within a small distance and turns back
    'endsNear': 0,        # ends a small distance short of the plane
    'parallelRun': 0,     # runs parallel to the plane, close to it, then leaves on the same side
    'parallelThenCross': 1,  # runs parallel close to the plane, then crosses
    'far': 0,             # never comes near the plane
}


def _planeCaseLocalPath(caseClass, rng, halfWidth, amplitude, nDense=4000):
    """
    Claude note: one case as a dense curve in the plane's local frame, where
    the plane is y = 0 and x, z are in-plane.  Returns the dense points.
    """
    u = np.linspace(0., 1., nDense)
    #Claude note: an in-plane path: x across the plane, z gently wandering,
    #kept well inside +-halfWidth so the crossings fall on the ROI
    x = (2. * u - 1.) * halfWidth * rng.uniform(0.3, 0.7) + rng.uniform(-0.2, 0.2) * halfWidth
    z = rng.uniform(-0.3, 0.3) * halfWidth + rng.uniform(-0.1, 0.1) * halfWidth * np.sin(np.pi * u + rng.uniform(0, 2 * np.pi))
    side = rng.choice([-1., 1.])
    #Claude note: the smallest distance from the plane of near-miss cases:
    #log-uniform from 1e-3 mm to 2 mm
    nearDistance = 10 ** rng.uniform(-3, np.log10(2.))
    if caseClass == 'cross1':
        #Claude note: y goes linearly through 0; the slope sets the crossing
        #angle, from steep to very shallow
        slope = 10 ** rng.uniform(-1.5, 1.)
        y = slope * (x - x[nDense // 2 + rng.integers(-nDense // 4, nDense // 4)])
        y = np.clip(y, -amplitude * 3, amplitude * 3)
    elif caseClass in ('cross2', 'cross3', 'cross4'):
        #Claude note: cos(pi*m*u) has exactly m zeros in (0,1) and is +-1 at the ends
        m = PLANE_CASE_CLASSES[caseClass]
        y = amplitude * np.cos(np.pi * m * u)
    elif caseClass == 'kiss':
        y = nearDistance + amplitude * (2. * u - 1.) ** 2
    elif caseClass == 'endsNear':
        y = nearDistance + amplitude * (1. - u) ** 2
    elif caseClass == 'parallelRun':
        #Claude note: flat at nearDistance over the middle, rising at both ends
        y = nearDistance + amplitude * np.clip(np.abs(2. * u - 1.) - 0.5, 0, None) ** 2 * 4
    elif caseClass == 'parallelThenCross':
        #Claude note: flat at nearDistance, then dives through the plane
        y = nearDistance - amplitude * (np.clip(u - 0.6, 0, None) / 0.4) ** 1.5 * 2
    elif caseClass == 'far':
        y = amplitude * (2. + rng.random()) + 0.5 * amplitude * np.sin(np.pi * u)
    else:
        raise ValueError('unknown case class ' + str(caseClass))
    return np.column_stack([x, side * y, z])


def _polylineCrossings(localNodes):
    """
    Claude note: crossings of the local plane y = 0 by a polyline, counted as
    side changes between consecutive nodes not exactly on the plane.  Done in
    the plane's own frame, where the plane is exactly y = 0.
    """
    side = np.sign(localNodes[:, 1])
    side = side[side != 0]
    return int(np.count_nonzero(side[1:] != side[:-1]))


def generatePlaneCrossingCases(nPerClass=25, classes=None, spacing=1., spacingJitter=0.,
                               halfWidth=30., amplitude=6., voxelSize=1., rotation=None,
                               translation=(0., 0., 0.), seed=0):
    """
    Claude note: streamlines around a plane, in classes that probe the
    crossing / kissing edge cases, with their true crossing counts and a
    planar ROI to test them against.

    The plane is built in a local frame (plane y = 0, through the origin) and
    everything is then placed with a rigid rotation and translation; the
    planar ROI's voxel grid is rotated with it, so its plane is exactly the
    local y = 0 plane (an oblique image in mm space if rotated).

    Parameters
    ----------
    nPerClass : int
        Streamlines per class.
    classes : list of str, optional
        Which of PLANE_CASE_CLASSES to build (default: all).
    spacing, spacingJitter : float
        Node spacing along each streamline (see generateBundle).
    halfWidth : float
        Half the in-plane extent (mm) the streamlines cross within; the ROI
        covers more than this, so every crossing falls on it.
    amplitude : float
        Typical distance (mm) streamlines travel away from the plane.
    voxelSize : float
        Voxel size of the planar ROI's grid.
    rotation, translation : rigid placement (see generateBundle)
    seed : int

    Returns
    -------
    streamlines : dipy Streamlines
    truth : dict
        'class' (str per streamline), 'crossings' (int per streamline: the
        sampled polyline's crossing count, which equals the class's count by
        construction unless sampling is too coarse to resolve the curve; such
        streamlines are listed in 'unresolved'), 'curveCrossings' (the class
        count).
    planarROI : nibabel.Nifti1Image
        The plane as a one-voxel-thick mask on its own grid.
    """
    rng = np.random.default_rng(seed)
    if classes is None:
        classes = list(PLANE_CASE_CLASSES)
    rotationMatrix = _rotationMatrix(rotation)
    translation = np.asarray(translation, dtype=float)
    streams, classLabels, polylineCounts, curveCounts = [], [], [], []
    for caseClass in classes:
        for iCase in range(nPerClass):
            localDense = _planeCaseLocalPath(caseClass, rng, halfWidth, amplitude)
            localNodes = _resampleByArcLength(localDense, spacing, spacingJitter, rng)
            #Claude note: reverse half of them, so direction is not a cue
            if rng.random() < 0.5:
                localNodes = localNodes[::-1]
            polylineCounts.append(_polylineCrossings(localNodes))
            curveCounts.append(PLANE_CASE_CLASSES[caseClass])
            classLabels.append(caseClass)
            streams.append((localNodes @ rotationMatrix.T + translation).astype(np.float64))
    #Claude note: the planar ROI: a grid whose second axis is the plane normal,
    #with the plane on one voxel slice through the local origin
    extent = halfWidth * 1.5
    nInPlane = int(np.ceil(2 * extent / voxelSize)) + 1
    nNormal = int(np.ceil(2 * (amplitude * 4) / voxelSize)) + 1
    planeIndex = nNormal // 2
    shape = (nInPlane, nNormal, nInPlane)
    affine = np.eye(4)
    affine[:3, :3] = voxelSize * rotationMatrix
    #Claude note: voxel (centre, planeIndex, centre) sits at the local origin
    centreVoxel = np.array([(nInPlane - 1) / 2., planeIndex, (nInPlane - 1) / 2.])
    affine[:3, 3] = translation - voxelSize * rotationMatrix @ centreVoxel
    data = np.zeros(shape, dtype=np.uint8)
    data[:, planeIndex, :] = 1
    polylineCounts = np.asarray(polylineCounts)
    curveCounts = np.asarray(curveCounts)
    truth = {'class': np.asarray(classLabels),
             'crossings': polylineCounts,
             'curveCrossings': curveCounts,
             'unresolved': np.flatnonzero(polylineCounts != curveCounts)}
    return Streamlines(streams), truth, nib.Nifti1Image(data, affine)
