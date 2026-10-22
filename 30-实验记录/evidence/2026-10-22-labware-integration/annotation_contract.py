"""RoboTwin-compatible annotations for independently staged Z-up lab assets.

No copy of the official Actor or create_actor implementation is used.  The
``load_with_official_create_actor`` function imports and calls the checkout's
actual loader; the only adapter is an absolute asset directory and explicit
mass assignment (Actor otherwise overwrites mass with its 0.01-kg default).

All dimensions supplied to generators are in metres *after* mesh scaling.
JSON point translations are saved in the mesh's unscaled coordinates, matching
ObjectViewer.update_config and Actor.get_point.  Rotations are never scaled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROBOTWIN = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
KINDS = ('beaker', 'erlenmeyer', 'graduated_cylinder', 'shallow_dish', 'bottle_body', 'bottle_cap')
RIGID_POINTS = {
    'contact': 'contact_points_pose', 'target': 'target_pose',
    'functional': 'functional_matrix', 'orientation': 'orientation_point',
}
ARTICULATION_POINTS = {
    'contact': 'contact_points', 'target': 'target_points',
    'functional': 'functional_points', 'orientation': 'orientation_point',
}
RIGID_DESCRIPTIONS = {
    'contact': 'contact_points_description', 'target': 'target_point_description',
    'functional': 'functional_point_description',
    'orientation': 'orientation_point_description',
}
ARTICULATION_DESCRIPTIONS = {
    'contact': 'contact_points_description', 'target': 'target_points_description',
    'functional': 'functional_points_description',
    'orientation': 'orientation_point_description',
}


def _scale3(scale):
    a = np.asarray(scale, dtype=float)
    if a.ndim == 0:
        a = np.repeat(a, 3)
    if a.shape != (3,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('scale must contain three finite positive numbers')
    return a


def _bounds(bounds_m):
    bounds = np.asarray(bounds_m, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError('bounds_m must be a finite 2x3 array')
    if np.any(bounds[1] <= bounds[0]):
        raise ValueError('each maximum must exceed its corresponding minimum')
    return bounds


def point_matrix(position_m, scale=(1., 1., 1.), rotation=None):
    """Encode a physical point in official unscaled-mesh JSON coordinates."""
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = np.asarray(position_m, dtype=float) / _scale3(scale)
    if rotation is not None:
        mat[:3, :3] = np.asarray(rotation, dtype=float)
    return mat.tolist()


def contact_frame(outward, upright=(0., 0., 1.)):
    """The official pregrasp offset points along contact +Y (approach is -Y).

    get_grasp_pose multiplies contact R by C=[[0,0,1],[-1,0,0],[0,-1,0]]
    then offsets the wrist by [-0.12-pre_dis,0,0] in that gripper frame.
    This function chooses a right-handed frame; it does not claim a validated
    RM65-B dexterous-hand grasp or use the official 0.12-m wrist offset.
    """
    yy = np.asarray(outward, dtype=float)
    yy /= np.linalg.norm(yy)
    zz = np.asarray(upright, dtype=float)
    zz = zz - yy * np.dot(zz, yy)
    if np.linalg.norm(zz) < 1e-8:
        raise ValueError('outward and upright must not be parallel')
    zz /= np.linalg.norm(zz)
    xx = np.cross(yy, zz)
    return np.column_stack([xx, yy, zz])


def generate_annotations(kind, bounds_m, *, scale=(1., 1., 1.), geometry=None,
                         source=None, mass_kg=.10):
    """Generate the actual rigid-Actor schema, plus explicit provenance.

    Input meshes must already be Z-up.  Annotation points lie in their actual
    bounding box.  ``geometry`` may refine ``inner_radius_m``, ``floor_z_m``,
    ``opening_z_m``, ``grasp_z_m`` and ``pour_rim_m``.  An AABB alone cannot
    establish that an object is hollow, or that a rim is a pouring spout;
    missing geometry is recorded as unvalidated instead of being invented.

    Contact markers are *grasp centres*, consistent with official pill-bottle
    annotations.  They are not physical finger contacts; the separate
    ``surface_anchor_candidates`` records actual mesh-surface checks.
    """
    if kind not in KINDS:
        raise ValueError(f'unsupported kind {kind!r}; choose one of {KINDS}')
    bounds = _bounds(bounds_m)
    scale = _scale3(scale)
    geometry = dict(geometry or {})
    center = bounds.mean(axis=0)
    axis_xy = np.asarray(geometry.get('axis_center_xy_m', center[:2]), dtype=float)
    if axis_xy.shape != (2,) or not np.isfinite(axis_xy).all():
        raise ValueError('axis_center_xy_m must contain two finite numbers')
    if np.any(axis_xy < bounds[0,:2]) or np.any(axis_xy > bounds[1,:2]):
        raise ValueError('axis centre must lie within the actual mesh bounds')
    lower, upper = bounds
    height = upper[2] - lower[2]
    if not np.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError('mass_kg must be finite and positive')
    opening_z = float(geometry.get('opening_z_m', upper[2]))
    floor_z = float(geometry.get('floor_z_m', lower[2]))
    grasp_z = float(geometry.get('grasp_z_m', lower[2] + .50 * height))
    for label, z in [('opening', opening_z), ('floor', floor_z), ('grasp', grasp_z)]:
        if not lower[2] - 1e-6 <= z <= upper[2] + 1e-6:
            raise ValueError(f'{label} z={z} is outside the actual mesh bounds')
    inner_radius = geometry.get('inner_radius_m')
    if inner_radius is not None and not 0 < float(inner_radius) < max(upper[:2] - lower[:2]):
        raise ValueError('inner_radius_m is not plausible for these bounds')

    config = {
        'center': (center / scale).tolist(),
        'extents': ((upper - lower) / scale).tolist(),
        'scale': scale.tolist(),
        'transform_matrix': np.eye(4).tolist(),
        'target_pose': [], 'contact_points_pose': [], 'functional_matrix': [],
        'orientation_point': [], 'contact_points_group': [],
        'contact_points_mask': [], 'target_point_description': [],
        'contact_points_description': [], 'functional_point_description': [],
        'orientation_point_description': [],
    }
    def add(which, position, description, rotation=None):
        config[RIGID_POINTS[which]].append(point_matrix(position, scale, rotation))
        config[RIGID_DESCRIPTIONS[which]].append(description)

    grasp_center = center.copy()
    grasp_center[:2] = axis_xy
    grasp_center[2] = grasp_z
    # Different side approaches share the same grasp centre and form one group.
    for outward, label in [([1,0,0], '+X'), ([0,1,0], '+Y'),
                            ([-1,0,0], '-X'), ([0,-1,0], '-Y')]:
        add('contact', grasp_center,
            f'{kind}: side grasp centre; outward/pregrasp +Y points toward {label}',
            contact_frame(outward))
    config['contact_points_group'] = [[0, 1, 2, 3]]
    config['contact_points_mask'] = [True]

    bottom = [*axis_xy, lower[2]]
    opening = [*axis_xy, opening_z]
    interior_floor = [*axis_xy, floor_z]
    opening_rotation = np.diag([1.,-1.,-1.]) if kind == 'bottle_cap' else np.eye(3)
    add('target', bottom, 'Support-plane alignment: bottom centre, not a grasp point')
    add('target', opening,
        'Cap opening / neck alignment centre (-Z outward)' if kind == 'bottle_cap'
        else 'Opening centre on the Z-up axis', opening_rotation)
    if kind == 'bottle_cap':
        add('functional', opening,
            'Cap underside opening centre (-Z outward) for neck alignment; no thread dynamics implied',
            opening_rotation)
        add('functional', [*axis_xy, upper[2]],
            'Cap twist-axis marker (+Z); removal must be supplied by physical mechanism')
    else:
        add('functional', opening, 'Receive/pour opening centre (+Z outward)')
        add('functional', interior_floor,
            'Interior floor centre; a valid receiving region requires audited hollow collision')
        pour_rim = np.asarray(geometry.get('pour_rim_m',
                              [upper[0], axis_xy[1], opening_z]), dtype=float)
        if pour_rim.shape != (3,) or np.any(pour_rim < lower-1e-6) or np.any(pour_rim > upper+1e-6):
            raise ValueError('pour_rim_m must be within mesh bounds')
        add('functional', pour_rim,
            'Candidate +X lip/spout target; nearest-surface evidence is stored separately')
    add('orientation', center, 'Object Z-up frame (+Z along vessel/cap axis)')
    config['annotation_provenance'] = {
        'generator': 'annotation_contract.py', 'date': '2026-10-22',
        'source': dict(source or {}), 'kind': kind,
        'method': 'human_specified_geometry_derived_not_official_original_annotation',
        'bounds_m': bounds.tolist(), 'geometry': geometry,
        'mass_kg': float(mass_kg), 'mass_source': 'engineering_assumption_not_measured',
        'hollow_collision_verified': False,
        'grasp_execution_verified': False,
        'scope': 'candidate keypoints; official loader/physics/robot validations are separate',
        'contact_semantics': 'grasp centre, not a finger-surface contact',
    }
    validate_annotations(config)
    return config


def generate_model_data(mesh_path, kind, *, scale=(1., 1., 1.), geometry=None,
                        source=None, mass_kg=.10):
    """Generate annotations from a real mesh, with nearest-surface evidence.

    The mesh must already have correct Z-up orientation.  No automatic AABB
    rotation, re-centring or change of scale occurs.  Scene node transforms
    are baked by dump(concatenate=True), just as the import converter does.
    """
    import trimesh
    path = Path(mesh_path).resolve(strict=True)
    mesh = trimesh.load(path, force='scene').dump(concatenate=True)
    mesh.apply_scale(_scale3(scale))
    # Some scanned source meshes contain zero-area triangles.  Exclude them
    # ONLY from surface queries (not from the imported visual or its hash),
    # otherwise trimesh's barycentric nearest-point calculation can emit NaN.
    face_areas = np.asarray(mesh.area_faces)
    usable_faces = np.isfinite(face_areas) & (face_areas > 1e-16)
    original_face_ids = np.flatnonzero(usable_faces)
    if not len(original_face_ids):
        raise ValueError('mesh has no nondegenerate surface triangles')
    query_mesh = mesh.copy()
    query_mesh.update_faces(usable_faces)
    query_mesh.remove_unreferenced_vertices()
    source = dict(source or {})
    source.update(mesh_path=str(path), mesh_sha256=sha256(path.read_bytes()).hexdigest())
    result = generate_annotations(kind, mesh.bounds, scale=scale, geometry=geometry,
                                  source=source, mass_kg=mass_kg)
    center = mesh.bounds.mean(axis=0)
    z = float((geometry or {}).get('grasp_z_m', center[2]))
    lo, hi = mesh.bounds
    queries = np.asarray([[hi[0],center[1],z], [center[0],hi[1],z],
                          [lo[0],center[1],z], [center[0],lo[1],z]])
    # Four calls avoid allocating a point_count x face_count array for dense meshes.
    anchors = []
    for query in queries:
        pts, distances, triangle = trimesh.proximity.closest_point_naive(query_mesh, query[None])
        anchors.append({'query_m': query.tolist(), 'point_m': pts[0].tolist(),
                        'distance_m': float(distances[0]),
                        'triangle_id': int(original_face_ids[triangle[0]])})
    result['annotation_provenance']['surface_anchor_candidates'] = anchors
    result['annotation_provenance']['surface_anchor_method'] = 'nearest point on actual triangulated mesh'
    result['annotation_provenance']['surface_query_excluded_degenerate_faces'] = int((~usable_faces).sum())
    # Replace the default AABB lip by its nearest *real* mesh surface.  It may
    # still not be a true pouring spout, which remains explicitly unvalidated.
    if kind != 'bottle_cap':
        p = np.asarray(result['functional_matrix'][2], dtype=float)[:3,3] * _scale3(scale)
        pts, distances, triangle = trimesh.proximity.closest_point_naive(query_mesh, p[None])
        result['functional_matrix'][2] = point_matrix(pts[0], scale)
        result['annotation_provenance']['pour_rim_surface_check'] = {
            'point_m': pts[0].tolist(), 'distance_from_candidate_m': float(distances[0]),
            'triangle_id': int(original_face_ids[triangle[0]]), 'actual_spout_verified': False,
        }
    validate_annotations(result)
    return result


def validate_annotations(data, *, articulation=False):
    """Strictly check our generated subset of either official schema."""
    keys = ARTICULATION_POINTS if articulation else RIGID_POINTS
    descriptions = ARTICULATION_DESCRIPTIONS if articulation else RIGID_DESCRIPTIONS
    scale = _scale3(data['scale'])
    if articulation and not np.allclose(scale, scale[0]):
        raise ValueError('official articulated loader requires a scalar uniform scale')
    for semantic, key in keys.items():
        points = data[key]
        if not isinstance(points, list):
            raise ValueError(f'{key} must be a list')
        if semantic == 'orientation' and len(points) > 1:
            raise ValueError('orientation_point holds zero or one matrix')
        if len(data[descriptions[semantic]]) != len(points):
            raise ValueError(f'{key} description count mismatch')
        for point in points:
            if articulation:
                if not isinstance(point.get('base'), str) or not point['base']:
                    raise ValueError('articulated points must name an actual base link')
                point = point['matrix']
            mat = np.asarray(point, dtype=float)
            if mat.shape != (4,4) or not np.isfinite(mat).all():
                raise ValueError(f'{key} must contain finite 4x4 matrices')
            if not np.allclose(mat[3], [0,0,0,1], atol=1e-7):
                raise ValueError(f'{key} last matrix row must be [0,0,0,1]')
            if not np.allclose(mat[:3,:3].T @ mat[:3,:3], np.eye(3), atol=1e-5):
                raise ValueError(f'{key} rotation is not orthonormal')
            if not np.isclose(np.linalg.det(mat[:3,:3]), 1., atol=1e-5):
                raise ValueError(f'{key} rotation must be right handed')
    groups = data['contact_points_group']
    if len(groups) != len(data['contact_points_mask']):
        raise ValueError('contact group and mask counts differ')
    count = len(data[keys['contact']])
    seen = []
    for group in groups:
        if not group or any(type(i) is not int or i < 0 or i >= count for i in group):
            raise ValueError('contact group contains an invalid point index')
        origins = []
        for i in group:
            point = data[keys['contact']][i]
            origins.append(np.asarray(point['matrix'] if articulation else point)[:3,3])
        if not np.allclose(origins, origins[0], atol=1e-7):
            raise ValueError('members of a contact group must share their grasp centre')
        seen += group
    if sorted(seen) != list(range(count)):
        raise ValueError('every contact point must occur exactly once in groups')
    if any(type(mask) is not bool for mask in data['contact_points_mask']):
        raise ValueError('contact_points_mask entries must be Boolean')
    return {'schema': 'articulation' if articulation else 'rigid',
            'counts': {semantic: len(data[key]) for semantic, key in keys.items()},
            'scale': scale.tolist(), 'passed': True}


def to_articulation_annotations(rigid_data, link_name):
    """Rebase ONE already-link-local rigid config to official URDF point format.

    This does not transform a body's points onto a different link.  A bottle
    and cap each need their own true link-local config before combining.
    """
    validate_annotations(rigid_data)
    scale = _scale3(rigid_data['scale'])
    if not np.allclose(scale, scale[0]):
        raise ValueError('articulations require uniform scale')
    result = {
        'scale': float(scale[0]), 'transform_matrix': deepcopy(rigid_data['transform_matrix']),
        'init_qpos': [], 'contact_points_group': deepcopy(rigid_data['contact_points_group']),
        'contact_points_mask': deepcopy(rigid_data['contact_points_mask']),
        'annotation_provenance': deepcopy(rigid_data.get('annotation_provenance', {})),
    }
    for semantic, rigid_key in RIGID_POINTS.items():
        result[ARTICULATION_POINTS[semantic]] = [
            {'base': link_name, 'matrix': deepcopy(matrix)} for matrix in rigid_data[rigid_key]]
        result[ARTICULATION_DESCRIPTIONS[semantic]] = deepcopy(rigid_data[RIGID_DESCRIPTIONS[semantic]])
    validate_annotations(result, articulation=True)
    return result


def world_point(data, semantic, idx, actor_matrix, *, link_matrices=None):
    """Reference transform to compare against real official Actor.get_point."""
    if link_matrices is None:
        local = np.asarray(data[RIGID_POINTS[semantic]][idx], dtype=float).copy()
        base = np.asarray(actor_matrix, dtype=float)
    else:
        point = data[ARTICULATION_POINTS[semantic]][idx]
        local = np.asarray(point['matrix'], dtype=float).copy()
        base = np.asarray(link_matrices[point['base']], dtype=float)
    local[:3,3] *= _scale3(data['scale'])
    return base @ local


@contextmanager
def _temporary_cwd(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def official_loader_module(robotwin=ROBOTWIN):
    """Import official package in its required asset-aware cwd, without edits."""
    robotwin = Path(robotwin).resolve(strict=True)
    if str(robotwin) not in sys.path:
        sys.path.insert(0, str(robotwin))
    with _temporary_cwd(robotwin):
        module = importlib.import_module('envs.utils.create_actor')
    expected = (robotwin / 'envs/utils/create_actor.py').resolve()
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(f'wrong create_actor module was cached: {module.__file__}')
    return module


def load_with_official_create_actor(scene, asset_dir, pose, *, model_id=0,
                                    convex=True, mass_kg=None, robotwin=ROBOTWIN):
    """Actually call the original rigid-body loader on an external asset folder.

    ``asset_dir`` contains visual/base0.glb, collision/base0.glb, model_data0.json
    (OBJ textured0.obj is also supported by the official loader).  Hollow
    collision must be represented by convex pieces; a single convex hull seals
    the opening.  This helper does NOT certify collision geometry or grasps.
    """
    directory = Path(asset_dir).resolve(strict=True)
    suffix = '' if model_id is None else str(model_id)
    data_path = directory / f'model_data{suffix}.json'
    data = json.loads(data_path.read_text(encoding='utf-8'))
    validate_annotations(data)
    loader = official_loader_module(robotwin)
    result = loader.create_actor(scene, pose, str(directory), model_id=model_id, convex=convex)
    if result is None:
        raise RuntimeError(f'official create_actor returned None for {directory}')
    if result.config != data:
        raise RuntimeError('official loader did not retain the requested annotation file')
    if mass_kg is None:
        mass_kg = data.get('annotation_provenance', {}).get('mass_kg')
    if mass_kg is not None:
        if not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('explicit mass_kg must be finite and positive')
        result.set_mass(float(mass_kg))
    for semantic, key in RIGID_POINTS.items():
        for idx in range(len(data[key])):
            actual = result.get_point(semantic, idx, 'matrix')
            expected = world_point(data, semantic, idx, result.get_pose().to_transformation_matrix())
            if actual is None or not np.allclose(actual, expected, atol=2e-6):
                raise AssertionError(f'official world transform failed: {semantic}[{idx}]')
    return result


def self_check():
    """Nontrivial scale/frame/schema checks; no claim of GPU/physics testing."""
    cases = []
    bounds = np.array([[-.03,-.03,0.], [.03,.03,.08]])
    angle = .7
    actor = np.eye(4)
    actor[:3,:3] = [[np.cos(angle),-np.sin(angle),0],
                    [np.sin(angle),np.cos(angle),0], [0,0,1]]
    actor[:3,3] = [.12,-.22,.74]
    for kind in KINDS:
        data = generate_annotations(kind, bounds, scale=[.001,.002,.003])
        check = validate_annotations(data)
        bottom = world_point(data, 'target', 0, actor)
        np.testing.assert_allclose(bottom[:3,3], actor[:3,3], atol=1e-12)
        contact = world_point(data, 'contact', 0, actor)
        np.testing.assert_allclose(contact[:3,3], actor[:3,3]+[0,0,.04], atol=1e-12)
        correction = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=float)
        approach_offset = contact[:3,:3] @ correction @ np.array([-.22,0,0])
        np.testing.assert_allclose(approach_offset, .22*contact[:3,1], atol=1e-12)
        # Improper transforms and nonuniform articulated scales must fail.
        broken = deepcopy(data)
        broken['functional_matrix'][0][0][0] = 3.
        try:
            validate_annotations(broken)
        except ValueError:
            pass
        else:
            raise AssertionError('nonrigid point rotation was accepted')
        uniform = generate_annotations(kind, bounds, scale=.001)
        articulated = to_articulation_annotations(uniform, 'body_link')
        validate_annotations(articulated, articulation=True)
        observed = world_point(articulated, 'target', 0, np.eye(4), link_matrices={'body_link': actor})
        np.testing.assert_allclose(observed, world_point(uniform,'target',0,actor), atol=1e-12)
        cases.append({'kind': kind, 'rigid': check, 'articulated_conversion': True})
    return {'passed': True, 'cases': cases, 'scope': 'CPU schema/scale/frame checks only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--kind', choices=KINDS)
    parser.add_argument('--scale', type=float, nargs=3, default=[1.,1.,1.])
    parser.add_argument('--geometry-json', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, indent=2))
    elif args.mesh and args.kind and args.output:
        geometry = json.loads(args.geometry_json.read_text()) if args.geometry_json else {}
        result = generate_model_data(args.mesh, args.kind, scale=args.scale, geometry=geometry)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(validate_annotations(result), ensure_ascii=False))
    else:
        parser.error('use --self-check or specify --mesh, --kind and --output')


if __name__ == '__main__':
    main()
