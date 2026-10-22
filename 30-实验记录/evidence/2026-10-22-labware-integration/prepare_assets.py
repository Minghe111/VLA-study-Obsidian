"""Prepare externally sourced labware meshes and explicit hollow collision proxies.

Run with the existing robotwin Python. Gmsh 4.14.1 is isolated in /tmp.
No original RoboTwin model, environment, calibration or task is changed.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '2026-10-21-realman-lab-specific'
SOURCES = ROOT / 'sources'
PREPARED = ROOT / 'prepared'
HF_REVISION = 'e622f0ec8123c9098d913ea56a4116fd83a29662'
SAMPLE_REVISION = 'e7217fb87daf42adc2e61233a4a1bba61adef946'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def save_mesh(mesh, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return str(path)


def step_mesh(path, volume=1):
    """Tessellate the actual pinned STEP B-rep, not a replacement primitive."""
    sys.path.insert(0, '/tmp/labware-cad-runtime')
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.7)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 2.0)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 64)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        vertices = np.asarray(coords).reshape(-1, 3) * 0.001
        mapping = {int(tag): i for i, tag in enumerate(tags)}
        triangles = []
        surfaces = gmsh.model.getBoundary([(3, volume)], oriented=False)
        for dim, tag in surfaces:
            types, _, nodes = gmsh.model.mesh.getElements(dim, abs(tag))
            for typ, group in zip(types, nodes):
                if typ != 2:
                    raise ValueError('Expected triangular STEP surface mesh')
                triangles.extend([[mapping[int(t)] for t in row]
                                  for row in np.asarray(group).reshape(-1, 3)])
        mesh = trimesh.Trimesh(vertices, np.asarray(triangles), process=True)
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh, multibody=True)
        return mesh
    finally:
        gmsh.finalize()


def revolved(profile, sections=128):
    """Z-axis closed radial profile in meters for declared derived geometry."""
    mesh = trimesh.creation.revolve(np.asarray(profile), sections=sections)
    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def collision_shell(asset_id, profile, floor_z, bottom_radius, n=48):
    """Each wall sector is a separate convex wedge. Never convexify the whole cup.

    profile = [(z, inner_radius, outer_radius), ...]. Each adjacent axial
    interval and angular sector is convexified separately; the cavity remains.
    These are declared geometric proxies, not transferred physics calibration.
    """
    dest = PREPARED / asset_id / 'collision'
    parts = []
    for j, (p0, p1) in enumerate(zip(profile[:-1], profile[1:])):
        for i in range(n):
            theta = np.array([i, i + 1]) * (2 * np.pi / n)
            vertices = []
            for z, ri, ro in (p0, p1):
                for r in (ri, ro):
                    vertices.extend([[r * np.cos(t), r * np.sin(t), z] for t in theta])
            mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=[]).convex_hull
            path = dest / f'wall_{j:02d}_{i:02d}.obj'
            parts.append({'type': 'convex_mesh', 'filename': save_mesh(mesh, path),
                          'sha256': sha(path), 'pose': [0, 0, 0, 1, 0, 0, 0]})
    if floor_z > 0:
        mesh = trimesh.creation.cylinder(bottom_radius, floor_z, sections=64)
        mesh.apply_translation([0, 0, floor_z / 2])
        path = dest / 'bottom.obj'
        parts.append({'type': 'convex_mesh', 'filename': save_mesh(mesh, path),
                      'sha256': sha(path), 'pose': [0, 0, 0, 1, 0, 0, 0]})
    return parts


def verify_collision_geometry(asset):
    scene = trimesh.load(asset['collision_glb'], force='scene', process=False)
    meshes = list(scene.geometry.values())
    def contains(point, mesh):
        # Convex half-space check avoids an optional rtree installation.
        products = (np.asarray(point) - mesh.triangles_center) * mesh.face_normals
        return bool(np.all(products.sum(axis=1) < 1e-7))
    if asset['id'] == 'reagent_bottle_cap':
        empty, sealed = [0,0,-.006], [0,0,0]
    else:
        empty = [0,0,(asset['inner_floor_z_m'] + asset['opening_z_m']) / 2]
        sealed = [0,0,asset['inner_floor_z_m'] / 2]
    result = {'id':asset['id'], 'node_count':len(meshes),
              'expected_parts':len(asset['collision_parts']),
              'all_convex':all(m.is_convex for m in meshes),
              'all_closed':all(m.is_watertight for m in meshes),
              'cavity_axis_probe_m':empty,
              'cavity_probe_clear':not any(contains(empty,m) for m in meshes),
              'floor_or_disc_probe_m':sealed,
              'floor_or_disc_probe_solid':any(contains(sealed,m) for m in meshes)}
    assert result['node_count'] == result['expected_parts']
    assert all(result[k] for k in ['all_convex','all_closed','cavity_probe_clear',
                                  'floor_or_disc_probe_solid']), result
    return result


GLASS = {'base_color': [0.91, 0.98, 1.0, 1.0], 'transmission': 0.94,
         'ior': 1.49, 'roughness': 0.045, 'transmission_roughness': 0.015,
         'metallic': 0, 'material_status': 'derived SAPIEN approximation of transparent glass'}


def record(asset_id, mesh, source, inner_radius, floor_z, outer_radius, profile,
           notes, material=None, grasp_z=None):
    path = PREPARED / asset_id / 'visual.obj'
    save_mesh(mesh, path)
    lo, hi = mesh.bounds
    z_height = float(hi[2] - lo[2])
    out = {'id': asset_id, 'name': asset_id, 'visual_file': str(path),
           'visual_groups': [{'filename': str(path), 'material': material or GLASS}],
           'scale': [1, 1, 1], 'local_axes': 'Z-up, X/Y horizontal', 'unit': 'meter',
           'origin': 'axis at x=y=0; bottom support plane z=0',
           'bounds_m': mesh.bounds.tolist(), 'height_m': z_height,
           'outer_radius_m': outer_radius, 'inner_radius_m': inner_radius,
           'inner_floor_z_m': floor_z, 'opening_z_m': float(hi[2]),
           'grasp_z_m': grasp_z if grasp_z is not None else z_height * 0.5,
           'visual_sha256': sha(path), 'source': source, 'notes': notes,
           'mesh_check': {'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
                          'watertight': bool(mesh.is_watertight),
                          'volume_m3': float(mesh.volume)},
           'collision_status': 'derived segmented convex wall and solid bottom proxies; not material calibrated',
           'collision_parts': collision_shell(asset_id, profile, floor_z, profile[0][2])}
    return out


def main():
    PREPARED.mkdir(exist_ok=True)
    old_models = json.loads((OLD / 'converted/models.json').read_text())
    old_manifest = json.loads((OLD / 'source/source_manifest.json').read_text())
    source_records = []
    assets = []
    for name in ['beaker', 'erlenmeyer']:
        m = next(a for a in old_models['models'] if a['name'] == name)
        src = OLD / 'source' / m['source_file']
        expected = next(f for f in old_manifest['files'] if f['path'] == m['source_file'])
        assert sha(src) == expected['sha256'], 'Pinned USD source changed: ' + str(src)
        dest = SOURCES / 'LabUtopia' / m['source_file']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        old_obj = OLD / 'converted' / m['parts'][0]['file']
        inp = SOURCES / 'LabUtopia' / old_obj.name
        shutil.copy2(old_obj, inp)
        mesh = trimesh.load(inp, force='mesh', process=True)
        # Beaker's vessel axis is displaced from the pouring spout's AABB centre.
        axis_x = 0.002335155 if name == 'beaker' else 0.0
        translation = [-axis_x, 0., -float(mesh.bounds[0, 2])]
        mesh.apply_translation(translation)
        source = {'project': 'LabUtopia', 'repository': 'https://github.com/Rui-li023/LabUtopia',
                  'dataset': 'https://huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset',
                  'revision': HF_REVISION, 'path': m['source_file'],
                  'source_file': str(dest), 'source_sha256': sha(dest),
                  'converted_source_obj': str(inp), 'converted_obj_sha256': sha(inp),
                  'source_up_axis': m['source_up_axis'], 'source_meters_per_unit': m['meters_per_unit'],
                  'usd_extraction': 'existing converted OBJ; USD source geometry and original stage transforms preserved',
                  'post_extraction_translation_m': translation,
                  'license': 'CC-BY-NC-4.0', 'attribution': 'Rui Li et al., LabUtopia, 2025'}
        source_records.append(source)
        if name == 'beaker':
            height = float(mesh.bounds[1, 2])
            asset = record('beaker', mesh, source, 0.0265, 0.0015, 0.0348,
                           [(0.0015, .0265, .02792), (height, .0265, .0290)],
                           ['Actual LabUtopia Beaker_01 source mesh, at its source scale.',
                            'Collision approximates the cylindrical cavity; scanned lip/spout details are visual only.',
                            'No volumetric liquid simulation or certified capacity markings.',
                            'Source visual mesh is not watertight; sealed collision proxies are independent.'], grasp_z=.036)
        else:
            height = float(mesh.bounds[1, 2])
            asset = record('erlenmeyer', mesh, source, .0164, .0022, .04008,
                           [(.0022, .033, .0354), (.0086, .0379, .0401),
                            (.1012, .0170, .0192), (.1272, .0168, .0190),
                            (height, .0181, .0208)],
                           ['Actual LabUtopia ErlenmeyerFlask_01 source mesh.',
                            'Conical collision wall is a segmented geometric approximation.',
                            'Additional background glassware; not a graduated cylinder.'], grasp_z=.07)
        assets.append(asset)
        if name == 'beaker':
            asset['body_outer_radius_m'] = .02792

    step_root = SOURCES / 'sample-data/packs/lab/assets/cad'
    for asset_id, filename, volume in [
        ('graduated_cylinder', 'demo-lab-4001-measuring-cylinder-100ml.step', 1),
        ('petri_dish', 'demo-lab-3003-petri-dish-90.step', 1)]:
        path = step_root / filename
        mesh = step_mesh(path, volume)
        shift = [0, 0, -float(mesh.bounds[0, 2])]
        mesh.apply_translation(shift)
        source = {'project': 'revenexx sample data', 'repository': 'https://github.com/revenexx/sample-data',
                  'revision': SAMPLE_REVISION, 'path': 'packs/lab/assets/cad/' + filename,
                  'url': f'https://github.com/revenexx/sample-data/blob/{SAMPLE_REVISION}/packs/lab/assets/cad/{filename}',
                  'source_file': str(path), 'source_sha256': sha(path),
                  'license': 'CC-BY-4.0', 'attribution': 'revenexx sample data, Copyright 2026 revenexx',
                  'source_unit': 'millimeter', 'conversion_scale_to_m': .001,
                  'conversion': 'Gmsh 4.14.1 B-rep surface tessellation',
                  'selected_volume': volume, 'post_conversion_translation_m': shift,
                  'source_status': 'fictional catalog model, not manufacturer or metrology certified'}
        source_records.append(source)
        if asset_id == 'graduated_cylinder':
            asset = record(asset_id, mesh, source, .0137, .0018, .029,
                           [(.0018, .0272, .029), (.003, .0272, .029),
                            (.012, .0167, .0185), (.016, .0137, .0155),
                            (.25, .0137, .0155)],
                           ['Actual imported 100 mL graduated-cylinder CAD, source diameter31 mm/foot58 mm/height250 mm.',
                            'CAD has no printed scale; graduation line overlay added from the same repository visual-model idea.',
                            'Added lines are decorative at 15 mm spacing, not a calibrated mL measurement.',
                            'Cavity proxy is segmented; shell offsets near the foot are approximated.'], grasp_z=.105)
            marks = []
            # Surface-only open arc marks, never white disks obstructing the cavity.
            for i, z in enumerate(np.arange(.031, .237, .015)):
                angles = np.linspace(-np.pi / 2 - (.6 if i % 2 == 0 else .35),
                                     -np.pi / 2 + (.6 if i % 2 == 0 else .35), 25)
                v = [[.01558 * np.cos(a), .01558 * np.sin(a), zz]
                     for zz in (z - .00022, z + .00022) for a in angles]
                f = [[j, j + 1, j + 26] for j in range(24)]
                f += [[j, j + 26, j + 25] for j in range(24)]
                marks.append(trimesh.Trimesh(v, f, process=False))
            mark_path = PREPARED / asset_id / 'graduation_marks.obj'
            save_mesh(trimesh.util.concatenate(marks), mark_path)
            asset['visual_groups'].append({'filename': str(mark_path),
                 'material': {'base_color': [.92, .94, .96, 1], 'transmission': 0,
                              'metallic': 0, 'roughness': .4}})
        else:
            asset = record(asset_id, mesh, source, .0425, .001, .0435,
                           [(.001, .0425, .0435), (.014, .0425, .0435)],
                           ['Actual 90 mm petri-dish STEP: volume 1 is the open 87 mm lower dish.',
                            'Volume 2 is the 90 mm lid and is intentionally omitted to keep a usable receiver.',
                            'Lid is preserved in the original STEP source, not placed over the receiver.'], grasp_z=.007)
        assets.append(asset)

    # These are new declared task-specific meshes, not native functionality of source assets.
    spec = json.loads((ROOT / 'physics_spec.json').read_text())
    b, c = spec['bottle'], spec['cap']
    rb, ri, h, fz = (b[k] for k in ['outer_radius_m','inner_radius_m','height_m','base_thickness_m'])
    body = revolved([(0,0),(rb,0),(rb,h),(ri,h),(ri,fz),(0,fz),(0,0)])
    source = {'project': 'local derived functional reagent bottle', 'license': 'local research prototype',
              'source_file': str(ROOT / 'physics_spec.json'), 'source_sha256': sha(ROOT / 'physics_spec.json'),
              'source_status': 'parametric derived body/cap, not downloaded native screw-cap mechanism',
              'attribution': 'local task adaptation; dimensions declared in physics_spec.json'}
    amber = dict(GLASS, base_color=[.57, .24, .045, 1], transmission=.76, roughness=.08)
    assets.append(record('reagent_bottle_body', body, source, ri, fz, rb,
                         [(fz,ri,rb),(h,ri,rb)],
                         ['Derived hollow reagent/pill bottle body matched to lab_physics.py.',
                          'Separate cap and independent tablets are implemented by the task physics module.',
                          'Thread behavior is a declared helical constraint approximation, not physical thread mesh.'],
                         material=amber, grasp_z=.045))
    ro, rc, thick, depth = (c[k] for k in ['outer_radius_m','inner_radius_m','top_thickness_m','skirt_depth_m'])
    cap = revolved([(0,thick/2),(ro,thick/2),(ro,-depth),
                    (rc,-depth),(rc,-thick/2),(0,-thick/2),(0,thick/2)])
    # A distinct cap origin is needed by the physics interface; do not move it to z=0.
    cap_path = PREPARED / 'reagent_bottle_cap' / 'visual.obj'
    save_mesh(cap, cap_path)
    assets.append({'id':'reagent_bottle_cap','name':'reagent_bottle_cap','visual_file':str(cap_path),
                   'visual_groups':[{'filename':str(cap_path),'material':
                        {'base_color':[.93,.94,.96,1],'transmission':0,'metallic':0,'roughness':.38}}],
                   'scale':[1,1,1],'unit':'meter','local_axes':'Z-up',
                   'origin':'top-disc center, matching LabBottle cap pose; not bottom plane',
                   'bounds_m':cap.bounds.tolist(),'height_m':float(cap.extents[2]),
                   'outer_radius_m':ro,'inner_radius_m':rc,'inner_floor_z_m':-thick/2,
                   'opening_z_m':-depth,'grasp_z_m':-depth/2,
                   'closed_center_z_m':c['closed_center_z_m'],'visual_sha256':sha(cap_path),
                   'source':source,'collision_parts':[],
                   'collision_status':'see LabBottle cap ring and top-disc primitives; no duplicate collision',
                   'notes':['Local cap frame intentionally differs from bottom-origin static labware.',
                            'Use visual pose identity when attaching to LabBottle.cap.']})

    # Native RoboTwin loader paths. Each collision wedge stays a separate scene
    # geometry/node, so add_multiple_convex_collisions_from_file preserves holes.
    for asset in assets:
        d = PREPARED / asset['id']
        visual_scene = trimesh.Scene()
        for i, group in enumerate(asset['visual_groups']):
            mesh = trimesh.load(group['filename'], force='mesh', process=False)
            material = group['material']
            color = np.clip(np.asarray(material.get('base_color', [1,1,1,1])) * 255, 0, 255).astype(np.uint8)
            # Material data in the interface is authoritative. GLB fallback uses
            # alpha blending because generic RoboTwin loading does not transfer
            # Omni MDL transmission parameters.
            if material.get('transmission', 0) > 0:
                color[3] = 70
            pbr = trimesh.visual.material.PBRMaterial(baseColorFactor=color,
                       metallicFactor=material.get('metallic',0),
                       roughnessFactor=material.get('roughness',.4),
                       alphaMode='BLEND' if color[3] < 255 else 'OPAQUE', doubleSided=True)
            mesh.visual = trimesh.visual.TextureVisuals(material=pbr)
            visual_scene.add_geometry(mesh, geom_name=f'visual_{i:02d}', node_name=f'visual_{i:02d}')
        vg = d / 'visual/base0.glb'
        vg.parent.mkdir(parents=True, exist_ok=True)
        visual_scene.export(vg)
        asset['visual_glb'] = str(vg)
        asset['visual_glb_sha256'] = sha(vg)
        if asset['id'] == 'reagent_bottle_cap':
            # Cap collision is exported for standalone official loader checks;
            # LabBottle itself uses its exact primitive ring/top from the spec.
            parts = collision_shell('reagent_bottle_cap',
                         [(-depth,rc,ro),(0,rc,ro)], 0, ro, n=32)
            disk = trimesh.creation.cylinder(ro, thick, sections=64)
            disk_path = d / 'collision/top_disc.obj'
            save_mesh(disk, disk_path)
            parts.append({'type':'convex_mesh','filename':str(disk_path),'sha256':sha(disk_path),
                          'pose':[0,0,0,1,0,0,0]})
            asset['collision_parts'] = parts
        collision_scene = trimesh.Scene()
        for i, part in enumerate(asset['collision_parts']):
            mesh = trimesh.load(part['filename'], force='mesh', process=False)
            collision_scene.add_geometry(mesh, geom_name=f'convex_{i:03d}', node_name=f'convex_{i:03d}')
        cg = d / 'collision/base0.glb'
        cg.parent.mkdir(parents=True, exist_ok=True)
        collision_scene.export(cg)
        asset['collision_glb'] = str(cg)
        asset['collision_glb_sha256'] = sha(cg)
        asset['collision_geometry_check'] = verify_collision_geometry(asset)

    interface = {'schema_version':'labware-assets-v1','date':'2026-10-22',
                 'root':str(ROOT),'scope':'source meshes, material translation and explicit hollow collision proxies',
                 'default_unit':'meter','default_axes':'Z-up','assets':assets,
                 'boundary':['No liquid or chemical process simulation.',
                             'Transparent visual material does not make any depth-sensor model realistic.',
                             'Collision walls, friction, density and screw guide are derived and uncalibrated.',
                             'No source CAD model is claimed to have native unscrewing behavior.',
                             'Functional points do not certify a RealMan grasp or collision-free motion.']}
    write_json(ROOT/'asset_interface.json', interface)
    write_json(ROOT/'source_manifest.json', {'date':'2026-10-22','sources':source_records,
                'prepared_meshes':[{k:a[k] for k in ['id','visual_file','visual_sha256','bounds_m','source']} for a in assets]})
    write_json(PREPARED/'geometry_audit.json',
               {'scope':'CPU mesh geometry integrity only; not dynamic or grasp proof',
                'checks':[a['collision_geometry_check'] for a in assets]})
    for a in assets:
        print(a['id'], 'bounds_m', a['bounds_m'], 'collision_parts', len(a['collision_parts']))


if __name__ == '__main__':
    main()
