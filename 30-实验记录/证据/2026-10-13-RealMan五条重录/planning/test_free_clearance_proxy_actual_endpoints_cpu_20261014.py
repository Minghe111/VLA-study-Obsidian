"""CPU endpoint feasibility from actual V2 records; no scene, GPU or physics.

Reload native convex components from the exact recorded asset IDs and scales,
at recorded settled poses. Use the frozen 93-link/36-DOF robot and original
measured start and declared free goals. This is an offline snapshot, not a new
physical episode or a native live-scene identity proof.
"""
from pathlib import Path
from types import SimpleNamespace
import copy, hashlib, json, sys, xml.etree.ElementTree as ET
import numpy as np
import sapien
from sapien import physx
import mplib
from mplib import PlanningWorld, Pose
from mplib.collision_detection import fcl
from mplib.sapien_utils.conversion import SapienPlanningWorld

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'planning'), str(ROOT/'source/adjust_bottle_20261002')]
import free_clearance_proxy_overlay as proxy
import full_mesh_path_audit as fma
import randomized_environment_guard as environment
from auxiliary_joint_window import create_model

V2 = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v2_conservative')
ASSETS = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/objects')
URDF = ROOT/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF = ASSETS.parent/'embodiments/realman-rm65b-inspire-gen4/robot.srdf'
MATERIAL = physx.PhysxMaterial(.5,.5,0.)
INPUT_HASHES = {}

def pinned(path):
    path = Path(path)
    INPUT_HASHES[str(path)] = proxy._sha(path)
    return path

def get_collision_asset(directory, model_id):
    stem = 'base' if model_id is None else 'base'+str(model_id)
    alt = 'textured' if model_id is None else 'textured'+str(model_id)
    for folder in [directory/'collision', directory]:
        if not folder.is_dir():
            continue
        for name in [stem+'.glb', alt+'.obj']:
            path = folder/name
            if path.is_file():
                return pinned(path)
    raise FileNotFoundError('Original native collision asset not found: '+str(directory))

def native_entity(name, shapes, matrix):
    entity = sapien.Entity(); entity.name = name
    component = physx.PhysxRigidDynamicComponent(); entity.add_component(component)
    for shape in shapes:
        component.attach(shape)
    entity.pose = sapien.Pose(proxy._matrix(matrix))
    assert np.max(abs(proxy._matrix(entity.pose)-matrix)) < 5e-7
    return entity, component

def native_asset(name, model_id, matrix):
    directory = ASSETS/name
    if not directory.is_dir():
        # Official objaverse loader has load_multiple_collisions_from_file=False.
        # These four recorded clutter actors each have one baseLink mesh.
        directory = ASSETS/'objaverse'/name/str(model_id)
        description = pinned(directory/'model.urdf')
        tree = ET.parse(description).getroot()
        assert len(tree.findall('link')) == 1 and not tree.findall('joint')
        collisions = tree.findall('link/collision'); shapes = []
        for collision in collisions:
            assert collision.find('origin') is None
            mesh = collision.find('geometry/mesh'); assert mesh is not None
            asset = pinned(directory/mesh.get('filename'))
            scale = np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            shapes.append(physx.PhysxCollisionShapeConvexMesh(str(asset),scale,MATERIAL))
        return native_entity(name,shapes,matrix)
    metadata = pinned(directory/('model_data.json' if model_id is None else 'model_data'+str(model_id)+'.json'))
    # Original create_actor retains its default scale when metadata has no scale.
    scale = np.asarray(json.loads(metadata.read_text()).get('scale',[1.,1.,1.]))
    asset = get_collision_asset(directory,model_id)
    shapes = physx.PhysxCollisionShapeConvexMesh.load_multiple(str(asset),scale,MATERIAL)
    return native_entity(name,shapes,matrix)

def table_component(matrix):
    # Exact original create_table collision construction: 1.2 x .7, .05 top.
    height = float(matrix[2,3]); thickness = .05
    shapes = []
    top = physx.PhysxCollisionShapeBox(np.array([.6,.35,thickness/2]),MATERIAL)
    top.local_pose = sapien.Pose([0,0,-thickness/2]); shapes.append(top)
    for i in [-1,1]:
        for j in [-1,1]:
            shape = physx.PhysxCollisionShapeBox(np.array([thickness/2,thickness/2,height/2-.002]),MATERIAL)
            shape.local_pose = sapien.Pose([i*(.6-.05),j*(.35-.05),-height/2-.002]); shapes.append(shape)
    return native_entity('table',shapes,matrix)

def make_actual_fixture(trial):
    episode = V2/'new_physical_episodes'/trial
    meta = json.loads(pinned(episode/'free_approach_0.json').read_text())['planning_metadata']
    report = json.loads(pinned(episode/'expert_report.json').read_text())
    scene = json.loads(pinned(episode/'randomized_scene_receipt.json').read_text())
    assert report['physical_steps'] == 0 and meta['rerecord_planning_sim_time_s'] == 0.
    poses = {r['name']:np.asarray(r['pose_matrix']) for r in scene['all_actor_initial_poses']}
    coverage = meta['randomized_obstacle_coverage']['native_geometry_coverage_receipt']
    bottle, bottle_component = native_asset('001_bottle',report['sample_spec']['model_id'],np.asarray(meta['initial_native_bottle_matrix']))
    clutter = []; components = []
    for row, recorded in zip(scene['official_clutter_records'],coverage['records']):
        name = row['object_type']; model_id = row['object_index']
        entity, component = native_asset(name,model_id,poses[name])
        assert recorded['name'] == name and recorded['shape_count'] == len(component.collision_shapes)
        assert recorded['geometry_types'] == [type(s).__name__ for s in component.collision_shapes]
        clutter.append(entity); components.append(component)
    assert len(clutter) == coverage['expected_clutter_count'] == 10
    assert sum(len(c.collision_shapes) for c in components) == coverage['covered_shape_count']
    base_pose = np.asarray(meta['initial_model_base_pose_matrix'])
    task = SimpleNamespace(bottle=SimpleNamespace(actor=bottle),
        robot=SimpleNamespace(left_entity=SimpleNamespace(root_pose=sapien.Pose(base_pose))),
        _rerecord_clutter_entities=clutter,record_cluttered_objects=scene['official_clutter_records'])
    audit = SimpleNamespace(task=task,fcl=fcl,Pose=Pose,static={})
    audit._rerecord_clutter_records = environment._snapshot_clutter(task)
    convert = lambda c:fma._convert_component_scaled(c,SapienPlanningWorld.convert_physx_component,fcl,Pose)
    audit.static['bottle'] = convert(bottle_component)
    table, comp = table_component(poses['table']); audit.static['table'] = convert(comp)
    for index, component in enumerate(components):
        audit.static['clutter_%02d'%index] = convert(component)
    # Keep the table entity alive without creating a scene.
    audit._cpu_table_entity = table
    inverse = np.linalg.inv(base_pose)
    originals = [fcl.FCLObject('realman_'+name,Pose(inverse@proxy._matrix(obj.pose)),
        [fcl.CollisionObject(s.get_collision_geometry()) for s in obj.shapes],
        [Pose(p) for p in obj.shape_poses]) for name,obj in audit.static.items()]
    extra,receipt = proxy.build_free_clearance_proxies(audit,inverse,originals)
    assert len(originals) == 12 and receipt['proxy_count'] == 10+len(bottle_component.collision_shapes)
    assert receipt['covered_native_shape_count'] == len(bottle_component.collision_shapes)+coverage['covered_shape_count']
    names = report['full_qpos_joint_names']; base = np.asarray(meta['initial_actual_full_qpos_rad'])
    tree = ET.parse(pinned(URDF)).getroot(); links = [n.get('name') for n in tree.findall('link')]
    assert len(names) == 36 and base.shape == (36,) and len(links) == 93
    source = mplib.Planner(urdf=URDF,srdf=pinned(SRDF),move_group='l_inspire_base_link',
        user_joint_names=names,user_link_names=links,use_convex=False)
    model = source.robot; model.set_base_pose(Pose()); model.set_qpos(base,full=True)
    bare = copy.deepcopy(tree)
    for link in bare.findall('link'):
        for child in list(link):
            if child.tag in ('visual','collision'):link.remove(child)
    allowed = {frozenset([n.get('link1'),n.get('link2')]) for n in ET.parse(SRDF).getroot().findall('disable_collisions')}
    factory = SimpleNamespace(model=model,pk=model.get_pinocchio_model(),urdf=ET.tostring(bare,encoding='unicode'),
        urdf_tree=bare,initial_sim_q=base.copy(),sim_joints=names,links={n:None for n in links},allowed=allowed,
        task=SimpleNamespace(robot=SimpleNamespace(left_arm_joints_name=meta['selected_arm_joint_names'])))
    bounded,window = create_model(factory,'left')
    return meta, base, bounded, originals, extra, receipt, window, model

def collision_rows(results):
    return [{'type':r.collision_type,'object1':r.object_name1,'object2':r.object_name2,
        'link1':r.link_name1,'link2':r.link_name2} for r in results]

def check_state(model, full_q, originals, extra):
    model.set_qpos(full_q,full=True)
    original_world = PlanningWorld([model],originals)
    actual = collision_rows(original_world.check_robot_collision())
    proxy_world = PlanningWorld([model],extra)
    expanded = collision_rows(proxy_world.check_robot_collision())
    all_world = PlanningWorld([model],originals+extra)
    all_results = collision_rows(all_world.check_collision())
    return {'original_native_environment_collisions':actual,'extra_proxy_environment_collisions':expanded,
        'all_world_collisions_including_self':all_results,
        'proxy_only_false_rejection':not actual and bool(expanded),
        'extra_proxy_collision_free':not expanded,'full_world_collision_free':not all_results}

def initial_bottle_box_diagnostic(model, full_q, originals, extra):
    model.set_qpos(full_q,full=True)
    native = next(o for o in originals if o.name == 'realman_bottle')
    # Keep the failed AABB geometry diagnosis in a separate historical receipt.
    bottle_extra = [o for o in extra if 'bottle' in o.name]
    if all(isinstance(o.shapes[0].get_collision_geometry(),fcl.Convex) for o in bottle_extra):
        return {'scope':'Tight complete native convex Minkowski proxies; failed AABB receipt preserved separately',
            'solid_convex_shape_count':len(bottle_extra),'minimum_original_native_distance_m':min(float(fcl.distance(o,native,fcl.DistanceRequest()).min_distance)
                for o in model.get_fcl_model().get_collision_objects()),'AABB_padding_sweep_not_used_as_new_proxy':True}
    box = bottle_extra[0]
    distance_rows = []
    for obj in model.get_fcl_model().get_collision_objects():
        if fcl.collide(obj,box,fcl.CollisionRequest()).is_collision():
            gap = float(fcl.distance(obj,native,fcl.DistanceRequest()).min_distance)
            assert np.isfinite(gap) and gap > 0
            distance_rows.append({'link':obj.name,'original_native_bottle_distance_m':gap})
    padding_rows = []
    for padding in [0.,.0005,.001,.002]:
        sides = np.asarray(box.shapes[0].get_collision_geometry().side)-2*.002+2*padding
        diagnostic = fcl.FCLObject('diagnostic_only_bottle_box',Pose(box.pose),
            [fcl.CollisionObject(fcl.Box(side=sides))],[Pose()])
        world = PlanningWorld([model],[diagnostic])
        padding_rows.append({'diagnostic_padding_m':padding,
            'collision_links':[r.link_name1 for r in world.check_robot_collision()]})
    return {'scope':'Diagnostic CPU boxes only; reviewed production padding remains 2 mm',
        'original_native_surface_gap_at_proxy_collision_links':distance_rows,
        'diagnostic_padding_sweep':padding_rows,
        'zero_padding_AABB_still_false_rejects':bool(padding_rows[0]['collision_links'])}

def main():
    frozen_helpers = {
        ROOT/'planning/auxiliary_joint_window.py':'19151529355d50bdb7baf35f185bf4abe0854adda1d30cb2978be9cb5b89130e',
        ROOT/'planning/conservative_ompl_overlay.py':'4e8e100a2c8ad63614e3d0912f3e0d5d6821719a10ffab441bebc4054a4e042c'}
    for path,expected in frozen_helpers.items():
        assert proxy._sha(pinned(path)) == expected
    for path in [Path(proxy.__file__),Path(fma.__file__),Path(environment.__file__)]:pinned(path)
    for path in [ASSETS.parent.parent/'envs/utils/create_actor.py',
                 ASSETS.parent.parent/'envs/utils/rand_create_cluttered_actor.py',
                 ROOT.parent/'2026-10-11-RealMan专家回放/shared_frozen_runtime/remote_base_task_snapshot.py']:pinned(path)
    for mesh in ET.parse(URDF).getroot().findall('.//mesh'):
        pinned((URDF.parent/mesh.get('filename')).resolve())
    rows = []
    for trial in ['slot17_trials/slot_0000017_randomized_attempt_003','slot18_trials/slot_0000018_randomized_attempt_001']:
        meta,base,model,originals,extra,receipt,window,source_model = make_actual_fixture(trial)
        source_q = np.asarray(source_model.get_qpos()).copy()
        source_limits = np.concatenate(source_model.get_pinocchio_model().get_joint_limits()).copy()
        source_poses = {o.name:proxy._matrix(o.pose).copy() for o in source_model.get_fcl_model().get_collision_objects()}
        index = meta['selected_arm_indices']; goals = []
        for attempt in meta['RRT_attempts']:
            for ordinal,goal in enumerate(attempt['predeclared_native_free_goals_full_rad']):
                key = tuple(goal)
                if any(tuple(g['full_qpos_rad']) == key for g in goals):continue
                q = np.asarray(goal)
                frozen = [i for i in range(36) if i not in index]
                assert np.array_equal(q[frozen],base[frozen])
                pk = model.get_pinocchio_model(); pk.compute_forward_kinematics(q)
                reference = source_model.get_pinocchio_model();reference.compute_forward_kinematics(q)
                full_fk_error = max(float(abs(pk.get_link_pose(i).to_transformation_matrix()-reference.get_link_pose(i).to_transformation_matrix()).max()) for i in range(93))
                assert full_fk_error <= 1e-8
                palm = np.asarray(meta['initial_model_base_pose_matrix'])@pk.get_link_pose(list(pk.get_link_names()).index('l_inspire_base_link')).to_transformation_matrix()
                error = float(abs(palm-np.asarray(meta['goal_palm_world_matrix'])).max())
                assert error < 5e-5
                goals.append({'branch':attempt['branch'],'IK_attempt':attempt['IK_attempt'],'goal_index':ordinal,
                    'full_qpos_rad':goal,'full_goal_FK_max_matrix_error':error,**check_state(model,q,originals,extra)})
                goals[-1]['all_93_source_vs_bounded_goal_FK_max_matrix_error']=full_fk_error
        assert np.array_equal(source_model.get_qpos(),source_q)
        assert np.array_equal(np.concatenate(source_model.get_pinocchio_model().get_joint_limits()),source_limits)
        assert all(np.array_equal(proxy._matrix(o.pose),source_poses[o.name]) for o in source_model.get_fcl_model().get_collision_objects())
        rows.append({'trial':trial,'full_DOF':36,'FK_link_count':len(model.get_pinocchio_model().get_link_names()),
            'initial_actual_full_qpos_rad':base.tolist(),'initial':check_state(model,base,originals,extra),
            'initial_bottle_box_diagnostic':initial_bottle_box_diagnostic(model,base,originals,extra),
            'source_model_q_limits_and_collision_poses_bitwise_unchanged':True,
            'source_goals':goals,'free_proxy_receipt':receipt,'numeric_window_receipt':window})
        print(json.dumps({'trial':trial,'initial_proxy_collisions':len(rows[-1]['initial']['extra_proxy_environment_collisions']),
            'goals':[{'branch':g['branch'],'original_collisions':len(g['original_native_environment_collisions']),
                'proxy_collisions':len(g['extra_proxy_environment_collisions']),'false_rejection':g['proxy_only_false_rejection']} for g in goals]}),flush=True)
    assert all(proxy._sha(path) == expected for path,expected in INPUT_HASHES.items())
    result = {'date':'2026-10-14','probe_completed':True,'scope':__doc__,
        'GPU_started':False,'SAPIEN_scene_created':False,'physics_steps':0,'new_OMPL_planning':False,
        'actual_source_goal_arrays_unmodified':True,'original_physics_FMA_source_or_production_modified':False,
        'source_hashes':INPUT_HASHES,'probe_sha256':proxy._sha(__file__),
        'proxy_sha256':proxy._sha(proxy.__file__),'rows':rows,
        'all_initials_and_source_goals_proxy_collision_free':all(r['initial']['extra_proxy_collision_free'] and all(g['extra_proxy_collision_free'] for g in r['source_goals']) for r in rows)}
    (ROOT/'planning/free_clearance_proxy_actual_endpoints_CPU_receipt_20261014.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__ == '__main__':main()
