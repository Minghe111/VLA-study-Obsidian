"""Independent free-only variant: retain actual obstacles and add 2 mm proxies.

Every complete native bottle convex shape gets the exact convex hull Minkowski
sum with a planning-base 2 mm cube. Each clutter component retains its containing
2 mm expanded box. These are planning-only obstacles; original numeric windows,
TOPP, physical geometry, full FMA and acceptance remain mandatory.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib, json
from pathlib import Path
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
PADDING_M = .002
_INSTALLED = None
_SESSIONS_BY_TASK = {}

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()

def _matrix(pose):
    value = pose.to_transformation_matrix() if hasattr(pose, 'to_transformation_matrix') else pose
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Finite 4x4 rigid pose required')
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
        raise ValueError('Malformed homogeneous pose')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5) or abs(np.linalg.det(rotation)-1) > 1e-5:
        raise ValueError('Only native rigid coordinate transforms supported')
    return matrix

def _positive(value, size, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError('Positive finite ' + label + ' required')
    return array

def native_component_vertices(component):
    """Read all shapes without modifying the component; fail unsupported geometry."""
    signs = np.asarray([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    chunks, receipts = [], []
    shapes = list(component.collision_shapes)
    if not shapes:
        raise ValueError('Native collision component cannot be empty')
    for index, shape in enumerate(shapes):
        name = type(shape).__name__
        if hasattr(shape, 'vertices'):
            raw = np.asarray(shape.vertices, dtype=np.float64)
            if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or not np.isfinite(raw).all():
                raise ValueError('Finite nonempty native mesh vertices required')
            vertices = raw * _positive(shape.scale, 3, 'native mesh scale')
        elif hasattr(shape, 'half_size'):
            vertices = signs * _positive(shape.half_size, 3, 'native half size')
        elif 'Capsule' in name or 'Cylinder' in name:
            radius = _positive([shape.radius], 1, 'radius')[0]
            half = float(shape.half_length)
            if not np.isfinite(half) or half < 0 or ('Cylinder' in name and half == 0):
                raise ValueError('Finite valid native half length required')
            vertices = signs * [half + (radius if 'Capsule' in name else 0), radius, radius]
        elif 'Sphere' in name:
            vertices = signs * _positive([shape.radius], 1, 'radius')[0]
        else:
            raise TypeError('Unsupported native proxy geometry ' + name)
        local = _matrix(shape.local_pose)
        vertices = vertices @ local[:3,:3].T + local[:3,3]
        if not np.isfinite(vertices).all():
            raise ValueError('Native transformed vertices overflowed')
        chunks.append(vertices)
        receipts.append({'shape_index':index, 'geometry_type':name,
                         'vertex_count':len(vertices), 'component_vertices_sha256':_digest(vertices),
                         'native_shape_local_pose_matrix':local.tolist()})
    return np.vstack(chunks), receipts

def expanded_base_aabb(local_vertices, component_world, inverse_root, *, padding_m=PADDING_M):
    if padding_m != PADDING_M:
        raise ValueError('Reviewed proxy padding is exactly 2 mm')
    vertices = np.asarray(local_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Finite nonempty native component vertices required')
    world = _matrix(component_world)
    inverse = _matrix(inverse_root)
    transform = inverse @ world
    base_vertices = vertices @ transform[:3,:3].T + transform[:3,3]
    # Independent two-stage readback detects frame/order mistakes, including translation.
    world_vertices = vertices @ world[:3,:3].T + world[:3,3]
    two_stage = world_vertices @ inverse[:3,:3].T + inverse[:3,3]
    if not np.isfinite(base_vertices).all() or not np.allclose(base_vertices, two_stage, rtol=0, atol=1e-12):
        raise ValueError('Native to world to planning-base transform failed')
    bounds = np.stack([base_vertices.min(0)-padding_m, base_vertices.max(0)+padding_m])
    sides = bounds[1]-bounds[0]
    if not np.isfinite(bounds).all() or not np.isfinite(sides).all() or np.any(sides <= 0):
        raise ValueError('Positive finite expanded AABB dimensions required')
    margin = np.minimum(base_vertices-bounds[0], bounds[1]-base_vertices)
    if np.min(margin) < padding_m-1e-12:
        raise RuntimeError('Expanded proxy failed native vertex containment')
    return bounds, base_vertices, {'bounds_base_m':bounds.tolist(), 'side_lengths_m':sides.tolist(),
        'native_vertex_count':len(vertices), 'native_vertices_base_sha256':_digest(base_vertices),
        'minimum_containment_margin_m':float(margin.min()),
        'world_to_base_transform_max_error_m':float(np.max(abs(base_vertices-two_stage)))}

def expanded_base_convex(local_vertices, component_world, inverse_root, fcl, *, padding_m=PADDING_M):
    """Exact native convex hull Minkowski sum with a planning-base 2 mm cube.

    The cube contains the radius-2 mm ball. Its furthest corner is sqrt(3)*2 mm.
    Original convex shape vertices are complete, not a sampled surface.
    """
    from scipy.spatial import ConvexHull
    bounds, base_vertices, details = expanded_base_aabb(local_vertices,component_world,inverse_root,padding_m=padding_m)
    corners = np.asarray([[x,y,z] for x in (-padding_m,padding_m)
                         for y in (-padding_m,padding_m) for z in (-padding_m,padding_m)])
    expanded = np.unique((base_vertices[:,None,:]+corners[None,:,:]).reshape(-1,3),axis=0)
    if not np.isfinite(expanded).all():raise ValueError('Native convex Minkowski vertices overflowed')
    hull = ConvexHull(expanded)
    vertices = expanded[hull.vertices]
    remap = {int(index):ordinal for ordinal,index in enumerate(hull.vertices)}
    faces = np.asarray([[remap[int(index)] for index in face] for face in hull.simplices],dtype=np.int32)
    for index,face in enumerate(faces):
        a,b,c = vertices[face]
        if np.dot(np.cross(b-a,c-a),hull.equations[index,:3]) < 0:
            faces[index,[1,2]] = faces[index,[2,1]]
    norms = np.linalg.norm(hull.equations[:,:3],axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError('Nonfinite or degenerate conservative hull plane')
    planes = hull.equations/norms[:,None]
    inside_distances = -(base_vertices@planes[:,:3].T+planes[:,3])
    minimum = float(inside_distances.min())
    if not np.isfinite(inside_distances).all() or minimum < padding_m-1e-10:
        raise RuntimeError('Complete native vertices lack full 2 mm hull-plane containment')
    if np.max(expanded@planes[:,:3].T+planes[:,3]) > 1e-10:
        raise RuntimeError('Minkowski corner omitted from conservative hull')
    geometry = fcl.Convex(vertices=vertices,faces=faces,throw_if_invalid=True)
    observed_vertices = np.asarray(geometry.get_vertices())
    expected_faces = np.column_stack([np.full(len(faces),3),faces]).reshape(-1)
    if not np.array_equal(observed_vertices,vertices) or not np.array_equal(geometry.get_faces(),expected_faces):
        raise RuntimeError('Solid FCL Convex vertices/faces readback differs')
    volume = float(geometry.compute_volume())
    if not np.isfinite(volume) or volume <= 0 or abs(volume-hull.volume) > max(1e-12,hull.volume*1e-8):
        raise RuntimeError('Solid conservative FCL Convex volume readback differs')
    details.update(proxy_geometry_type='Convex',proxy_construction='Exact full native convex hull Minkowski sum with 8 planning-base cube corners',
        minimum_containment_margin_m=minimum,normalized_hull_planes=planes.tolist(),
        native_vertex_plane_distance_count=int(inside_distances.size),minkowski_cube_offsets_m=corners.tolist(),
        full_minkowski_vertex_count=len(expanded),full_minkowski_vertices_sha256=_digest(expanded),
        hull_vertices_sha256=_digest(vertices),hull_face_count=len(faces),hull_vertex_count=len(vertices),
        FCL_solid_convex_volume_m3=volume,FCL_geometry_readback_verified=True,
        maximum_cube_corner_expansion_m=float(np.sqrt(3)*padding_m),native_surface_sampling_used=False)
    return geometry,base_vertices,details

def _actual_components(audit):
    from sapien.physx import PhysxRigidBaseComponent
    import randomized_environment_guard as environment
    task = audit.task
    actor = task.bottle.actor
    bottle = [c for c in actor.get_components() if isinstance(c, PhysxRigidBaseComponent) and len(c.collision_shapes)]
    if len(bottle) != 1 or bottle[0] is not actor.find_component_by_type(PhysxRigidBaseComponent):
        raise RuntimeError('Exactly the audited native bottle component is required')
    groups = {'bottle':bottle}
    actual = []
    for ordinal, item, components in environment._components(task):
        groups['clutter_%02d' % ordinal] = list(components)
        actual.extend((ordinal, ci, id(component)) for ci, component in enumerate(components))
    recorded = getattr(audit, '_rerecord_clutter_records', None)
    if recorded is None:
        raise RuntimeError('Verified complete original clutter audit must precede proxies')
    cached = [(r['clutter_index'], r['component_index'], id(r['component'])) for r in recorded]
    if sorted(actual) != sorted(cached) or len(set(cached)) != len(cached):
        raise RuntimeError('Native clutter component identities differ from original complete audit')
    if set(audit.static) != set(groups) | {'table'}:
        raise RuntimeError('Every original native static category must be accounted for')
    return groups

def _static_snapshot(audit):
    return {name:{'pose':_matrix(obj.pose).tolist(),
                  'shape_poses':[_matrix(p).tolist() for p in obj.shape_poses],
                  'shape_types':[type(s.get_collision_geometry()).__name__ for s in obj.shapes]}
            for name,obj in audit.static.items()}

def _register_actual_bounds(task,receipt):
    """Bind actual per-call proxy geometry proof bytes to the existing session."""
    session=_SESSIONS_BY_TASK.get(id(task))
    if session is None:return receipt
    if session.task is not task:raise RuntimeError('Clearance proxy session/task identity differs')
    records=session.report.setdefault('free_planning_clearance_proxy_bounds_receipts',[])
    path=Path(session.out)/('free_clearance_proxy_actual_bounds_%06d.json'%len(records))
    if path.exists():raise RuntimeError('Actual proxy bounds receipt must never overwrite prior evidence')
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    digest=_sha(path);session.sources[str(path.resolve())]=digest
    binding={'bounds_receipt_source':str(path.resolve()),'bounds_receipt_sha256':digest}
    records.append(binding)
    session.report['source_hashes']=session.sources
    return {**receipt,**binding}

def build_free_clearance_proxies(audit, inverse_root, original_obstacles):
    """Add separate planning objects; never replace actual native obstacles."""
    import full_mesh_path_audit as fma
    inverse = _matrix(inverse_root)
    root = _matrix(audit.task.robot.left_entity.root_pose)
    if not np.allclose(inverse @ root, np.eye(4), rtol=0, atol=5e-7):
        raise ValueError('Planning inverse must match the actual robot root')
    before = _static_snapshot(audit)
    original = list(original_obstacles)
    lookup = {o.name:o for o in original}
    if len(lookup) != len(original) or set(lookup) != {'realman_'+n for n in audit.static}:
        raise RuntimeError('All original actual FCL planning obstacles must remain present')
    for name, source in audit.static.items():
        obj = lookup['realman_'+name]
        if len(obj.shapes) != len(source.shapes) or len(obj.shape_poses) != len(source.shape_poses):
            raise RuntimeError('Original full native planning shape coverage differs')
        for s,p,cs,cp in zip(source.shapes, source.shape_poses, obj.shapes, obj.shape_poses):
            if type(s.get_collision_geometry()) is not type(cs.get_collision_geometry()):
                raise RuntimeError('Original native collision geometry type changed')
            if not np.allclose(_matrix(obj.pose) @ _matrix(cp), inverse @ _matrix(source.pose) @ _matrix(p), rtol=0, atol=1e-6):
                raise RuntimeError('Original native planning obstacle pose changed')
    groups = _actual_components(audit)
    proxies, records = [], []
    for category, components in groups.items():
        native_count = sum(len(c.collision_shapes) for c in components)
        if native_count != len(audit.static[category].shapes):
            raise RuntimeError('Native/FCL component shape coverage mismatch')
        # Clutter retains a box per component. Bottle uses every complete native
        # convex shape separately, avoiding the component-wide AABB's empty corners.
        for ci, component in enumerate(components):
            local, shape_receipts = native_component_vertices(component)
            audit_vertices, audit_types = fma._link_vertices(component)
            if local.shape != audit_vertices.shape or not np.allclose(local, audit_vertices, rtol=0, atol=1e-12):
                raise RuntimeError('Proxy vertices differ from original full native audit extraction')
            if category == 'bottle':
                units = []
                for si,shape in enumerate(component.collision_shapes):
                    if type(shape).__name__ != 'PhysxCollisionShapeConvexMesh':
                        raise TypeError('Bottle clearance proxy requires complete native ConvexMesh shapes')
                    vertices,rows = native_component_vertices(type('SingleShapeComponent',(),{'collision_shapes':[shape]})())
                    rows[0]['shape_index'] = si
                    units.append((si,vertices,rows,[audit_types[si]]))
            else:units=[(None,local,shape_receipts,audit_types)]
            for si,vertices,rows,types in units:
                name = '__free_clearance_proxy_'+category+'_component_%03d' % ci
                if si is not None:name += '_shape_%03d' % si
                if name in lookup:raise RuntimeError('Planning-only proxy name collision')
                if si is None:
                    bounds,base_vertices,details=expanded_base_aabb(vertices,component.entity.pose,inverse)
                    geometry=audit.fcl.Box(side=bounds[1]-bounds[0])
                    if not np.allclose(np.asarray(geometry.side),bounds[1]-bounds[0],rtol=0,atol=1e-12):
                        raise RuntimeError('FCL Box full side-length readback differs')
                    pose=np.eye(4);pose[:3,3]=(bounds[0]+bounds[1])/2
                    details['proxy_geometry_type']='Box'
                else:
                    geometry,base_vertices,details=expanded_base_convex(vertices,component.entity.pose,inverse,audit.fcl)
                    bounds=np.asarray(details['bounds_base_m']);pose=np.eye(4)
                obj=audit.fcl.FCLObject(name,audit.Pose(pose),[audit.fcl.CollisionObject(geometry)],[audit.Pose()])
                observed_pose=_matrix(obj.pose)@_matrix(obj.shape_poses[0])
                if si is None:
                    sides=np.asarray(obj.shapes[0].get_collision_geometry().side)
                    observed=np.stack([observed_pose[:3,3]-sides/2,observed_pose[:3,3]+sides/2])
                else:
                    final_vertices=np.asarray(obj.shapes[0].get_collision_geometry().get_vertices())
                    final_vertices=final_vertices@observed_pose[:3,:3].T+observed_pose[:3,3]
                    observed=np.stack([final_vertices.min(0),final_vertices.max(0)])
                if not np.allclose(observed,bounds,rtol=0,atol=1e-12):
                    raise RuntimeError('Native FCL proxy bounds readback differs')
                proxies.append(obj)
                records.append({'category':category,'component_index':ci,'component_name':component.name,
                    'native_shape_index':si,'proxy_name':name,'native_collision_shape_count':len(rows),
                    'shape_receipts':rows,'native_component_world_matrix':_matrix(component.entity.pose).tolist(),
                    'original_geometry_types':types,'FCL_bounds_readback_m':observed.tolist(),**details})
    if _static_snapshot(audit) != before:
        raise RuntimeError('Actual original audit obstacle geometry/poses were modified')
    receipt={'schema':'free_native_clearance_proxy_v2', 'padding_each_side_m':PADDING_M,
        'scope':'Only free planning; exact bottle convex Minkowski proxies and clutter boxes are not actual physical or FMA geometry',
        'original_actual_obstacles_retained':sorted(lookup), 'proxy_count':len(proxies),
        'covered_categories':sorted(groups), 'covered_native_component_count':sum(len(v) for v in groups.values()),
        'covered_native_shape_count':sum(len(r['shape_receipts']) for r in records),
        'actual_clutter_count':len(groups)-1, 'world_to_planning_base_matrix':inverse.tolist(),
        'original_static_snapshot_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest(),
        'original_static_geometry_and_poses_unchanged':True, 'table_proxy_added':False,
        'carry_support_intentional_contacts_modified':False, 'conservative_false_rejection_allowed':True,
        'proxy_collision_free_is_not_final_acceptance':True, 'records':records}
    return proxies,_register_actual_bounds(audit.task,receipt)

def derive_proxy_ast(parent):
    tree = copy.deepcopy(parent)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
             and ast.unparse(n.value.func)=='PlanningWorld']
    if len(calls)!=1 or ast.unparse(calls[0].value)!='PlanningWorld([model], obstacles)':
        raise RuntimeError('Exactly the reviewed original free planning world is required')
    call = calls[0]
    owners = [n for n in ast.walk(tree) if hasattr(n,'body') and isinstance(n.body,list) and call in n.body]
    if len(owners)!=1:
        raise RuntimeError('Unique free world construction owner required')
    prefix = ast.parse("_free_clearance_proxies, _free_clearance_proxy_receipt = _build_free_clearance_proxies(audit, inverse, obstacles)\nobstacles.extend(_free_clearance_proxies)\nmetadata['free_planning_clearance_proxies'] = _free_clearance_proxy_receipt").body
    body=owners[0].body; index=body.index(call); body[index:index]=prefix
    ast.fix_missing_locations(tree)
    return tree, {'parent_AST_sha256':hashlib.sha256(ast.dump(parent,include_attributes=False).encode()).hexdigest(),
        'derived_AST_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
        'only_change':'Insert native component clearance proxies before original free PlanningWorld construction',
        'original_obstacle_construction_and_model_untouched':True,
        'native_OMPL_TOPP_full_FMA_numeric_window_quality_gates_untouched':True,
        'carry_AST_changed':False, 'returned_paths_postprocessed':False}

def install_free_clearance_proxy_overlay():
    """Install in a fresh V3 worker after conservative/window, before observers."""
    global _INSTALLED
    if _INSTALLED is not None:
        raise RuntimeError('Free clearance proxy variant installs once per fresh worker')
    receipt={'overlay_source':str(Path(__file__).resolve()), 'overlay_sha256':_sha(__file__),
        'padding_each_side_m':PADDING_M, 'source_original_bytes_modified':False,
        'GPU_scene_or_physics_started_during_install':False, 'functions':{}, 'carry_unchanged':True}
    carry=importlib.import_module('full_native_carry_planner_v31').plan_carry_to_task_pose
    for name in ('full_native_approach_equivalent_goals_v6', 'full_native_approach_equivalent_goals_dense_v7'):
        module=importlib.import_module(name); previous=module.plan_free_approach
        if not getattr(previous,'__rerecord_window_overlay__',None) or not getattr(previous,'__rerecord_conservative_ompl_overlay__',None):
            raise RuntimeError('Verified numeric window and conservative native OMPL must precede proxies')
        parent=getattr(previous,'__rerecord_derivation_ast__',None)
        if parent is None:
            raise RuntimeError('Install before metadata observers, from the complete proven free AST')
        tree,details=derive_proxy_ast(parent)
        file=DIRECTORY/(name+'_plan_free_approach_clearance_proxy_generated.py')
        file.write_text(ast.unparse(tree)+'\n')
        namespace=dict(previous.__globals__); namespace['_build_free_clearance_proxies']=build_free_clearance_proxies
        exec(compile(tree,str(file),'exec'),namespace); derived=namespace['plan_free_approach']
        derived.__rerecord_derivation_ast__=copy.deepcopy(tree)
        for attribute in ('__rerecord_window_overlay__','__rerecord_conservative_ompl_overlay__'):
            setattr(derived,attribute,copy.deepcopy(getattr(previous,attribute)))
        derived.__rerecord_clearance_proxy_overlay__=copy.deepcopy(details)
        module.plan_free_approach=derived
        details.update(generated_source=str(file), generated_source_sha256=_sha(file))
        receipt['functions'][name]=details
    if importlib.import_module('full_native_carry_planner_v31').plan_carry_to_task_pose is not carry:
        raise RuntimeError('Free-only overlay unexpectedly modified carry')
    install_path=DIRECTORY/'free_clearance_proxy_install_receipt.json'
    install_path.write_text(json.dumps(receipt,indent=2)+'\n')
    receipt['install_receipt_source']=str(install_path.resolve())
    receipt['install_receipt_sha256']=_sha(install_path)
    expert=importlib.import_module('expert_generation').ExpertSession; previous_init=expert.__init__
    def init(session,*args,**kwargs):
        previous_init(session,*args,**kwargs)
        session.sources[str(Path(__file__).resolve())]=receipt['overlay_sha256']
        for item in receipt['functions'].values():
            session.sources[item['generated_source']]=item['generated_source_sha256']
        if _sha(install_path)!=receipt['install_receipt_sha256']:
            raise RuntimeError('Proxy install/AST receipt changed before session registration')
        session.sources[str(install_path.resolve())]=receipt['install_receipt_sha256']
        if hasattr(session,'task') and hasattr(session,'out'):
            _SESSIONS_BY_TASK[id(session.task)]=session
        session.report['free_planning_clearance_proxy_overlay']=copy.deepcopy(receipt)
        session.report['source_hashes']=session.sources
    expert.__init__=init; _INSTALLED=copy.deepcopy(receipt)
    return copy.deepcopy(receipt)
