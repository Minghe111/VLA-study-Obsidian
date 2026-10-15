"""CPU native PhysX/FCL, coverage, transform and unchanged AST counterexamples."""
from pathlib import Path
import ast, copy, hashlib, json, sys, types, unittest
from unittest.mock import patch
import numpy as np
import sapien
from sapien import physx
from mplib import Pose
from mplib.collision_detection import fcl
import free_clearance_proxy_overlay as overlay

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'source/adjust_bottle_20261002'), str(ROOT/'planning')]
import full_mesh_path_audit as fma
import randomized_environment_guard as environment
from mplib.sapien_utils.conversion import SapienPlanningWorld
ASSETS=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/objects/001_bottle')
MATERIAL=physx.PhysxMaterial(.5,.5,0.)
PROOFS=[]

def native_entity(name, shapes, pose=None):
    entity=sapien.Entity(); entity.name=name
    component=physx.PhysxRigidDynamicComponent(); entity.add_component(component)
    for shape in shapes: component.attach(shape)
    entity.pose=sapien.Pose() if pose is None else sapien.Pose(np.asarray(pose))
    return entity, component

def make_fixture():
    data=json.loads((ASSETS/'model_data13.json').read_text())
    shapes=physx.PhysxCollisionShapeConvexMesh.load_multiple(str(ASSETS/'collision/base13.glb'), np.asarray(data['scale']), MATERIAL)
    bottle,bottle_component=native_entity('bottle',shapes,sapien.Pose([-.2,-.1,.75], [.9238795,.3826834,0,0]).to_transformation_matrix())
    root=sapien.Pose([.04,-.65,.0954],[0.,0.,0.,1.]).to_transformation_matrix()
    refs=[]; components=[]
    # One actual actor has two independently posed collision components.
    for ordinal in range(2):
        members=[]
        for ci in range(2 if ordinal==0 else 1):
            shape=physx.PhysxCollisionShapeBox(np.asarray([.01,.02,.03]),MATERIAL)
            shape.local_pose=sapien.Pose([.005,-.004,.003],[.9238795,0,0,.3826834])
            entity, component=native_entity('clutter_%d_%d'%(ordinal,ci),[shape],sapien.Pose([.4+ci*.12,.2+ordinal*.15,.8],[.9238795,0,.3826834,0]).to_transformation_matrix())
            members.append(types.SimpleNamespace(entity=entity));components.append(component)
        actor=types.SimpleNamespace(get_links=lambda members=members:members)
        refs.append(actor)
    task=types.SimpleNamespace(bottle=types.SimpleNamespace(actor=bottle), robot=types.SimpleNamespace(left_entity=types.SimpleNamespace(root_pose=sapien.Pose(root))),
        _rerecord_clutter_entities=refs, record_cluttered_objects=[{},{}])
    audit=types.SimpleNamespace(task=task,fcl=fcl,Pose=Pose,static={})
    audit._rerecord_clutter_records=environment._snapshot_clutter(task)
    convert=lambda comp:fma._convert_component_scaled(comp,SapienPlanningWorld.convert_physx_component,fcl,Pose)
    audit.static['bottle']=convert(bottle_component)
    table, table_comp=native_entity('table',[physx.PhysxCollisionShapeBox(np.asarray([.7,.7,.01]),MATERIAL)],sapien.Pose([0,0,.73]).to_transformation_matrix())
    audit.static['table']=convert(table_comp)
    groups={}
    for r in audit._rerecord_clutter_records:
        obj=convert(r['component']);category='clutter_%02d'%r['clutter_index']
        g=groups.setdefault(category,{'shapes':[],'poses':[]})
        for shape,local in zip(obj.shapes,obj.shape_poses):
            g['shapes'].append(fcl.CollisionObject(shape.get_collision_geometry()))
            g['poses'].append(Pose(overlay._matrix(obj.pose)@overlay._matrix(local)))
    for name,g in groups.items(): audit.static[name]=fcl.FCLObject(name,Pose(),g['shapes'],g['poses'])
    inverse=np.linalg.inv(root)
    originals=[]
    for name,obj in audit.static.items():
        originals.append(fcl.FCLObject('realman_'+name,Pose(inverse@overlay._matrix(obj.pose)),
            [fcl.CollisionObject(s.get_collision_geometry()) for s in obj.shapes], [Pose(p) for p in obj.shape_poses]))
    return audit,inverse,originals

def parent(name):
    path=ROOT/'planning/conservative_CPU_component_20261014'/(name+'_plan_free_approach_conservative_native_ompl_generated.py')
    return ast.parse(path.read_text()),path

class ClearanceProxyCounterexamples(unittest.TestCase):
    def test_full_native_bottle_and_multicomponent_clutter_coverage_and_FCL_readback(self):
        audit,inverse,originals=make_fixture(); before=overlay._static_snapshot(audit)
        proxies,receipt=overlay.build_free_clearance_proxies(audit,inverse,originals)
        self.assertEqual(len(audit.static['bottle'].shapes),7)
        self.assertEqual(receipt['proxy_count'],4);self.assertEqual(receipt['covered_native_shape_count'],10)
        self.assertEqual(receipt['actual_clutter_count'],2)
        self.assertEqual(receipt['covered_categories'],['bottle','clutter_00','clutter_01'])
        self.assertEqual(len(originals),4);self.assertEqual(before,overlay._static_snapshot(audit))
        self.assertFalse(any('table' in o.name for o in proxies))
        for row in receipt['records']:
            self.assertGreaterEqual(row['minimum_containment_margin_m'],.002-1e-12)
            self.assertLess(row['world_to_base_transform_max_error_m'],1e-12)
        PROOFS.append(receipt)
    def test_scaled_native_mesh_all_vertices_match_original_full_audit_extractor(self):
        audit,inverse,originals=make_fixture()
        component=audit.task.bottle.actor.find_component_by_type(physx.PhysxRigidBaseComponent)
        local,rows=overlay.native_component_vertices(component); expected,types_=fma._link_vertices(component)
        self.assertEqual(len(rows),7);np.testing.assert_allclose(local,expected,rtol=0,atol=1e-12)
        self.assertTrue(any(not np.allclose(shape.scale,1.) for shape in component.collision_shapes))
    def test_rotations_local_offsets_world_base_order_and_full_side_dimensions(self):
        audit,inverse,originals=make_fixture();proxies,receipt=overlay.build_free_clearance_proxies(audit,inverse,originals)
        for proxy,row in zip(proxies,receipt['records']):
            bounds=np.asarray(row['bounds_base_m']);geometry=proxy.shapes[0].get_collision_geometry()
            np.testing.assert_allclose(geometry.side,bounds[1]-bounds[0],rtol=0,atol=1e-12)
            np.testing.assert_allclose(proxy.pose.p,(bounds[0]+bounds[1])/2,rtol=0,atol=1e-12)
        wrong=inverse.copy();wrong[:3,3]+=[.1,0,0]
        with self.assertRaises(ValueError):overlay.build_free_clearance_proxies(audit,wrong,originals)
    def test_nonfinite_vertices_invalid_rigid_pose_dimensions_and_unreviewed_padding_fail(self):
        vertices=np.array([[0.,0.,0.],[1.,1.,1.]])
        for value in [np.array([[float('nan'),0.,0.]]),np.empty((0,3)),np.array([[float('inf'),0,0]])]:
            with self.assertRaises(ValueError):overlay.expanded_base_aabb(value,np.eye(4),np.eye(4))
        bad=np.eye(4);bad[0,0]=2
        with self.assertRaises(ValueError):overlay.expanded_base_aabb(vertices,bad,np.eye(4))
        bad=np.eye(4);bad[0,3]=float('nan')
        with self.assertRaises(ValueError):overlay.expanded_base_aabb(vertices,np.eye(4),bad)
        with self.assertRaises(ValueError):overlay.expanded_base_aabb(vertices,np.eye(4),np.eye(4),padding_m=.0005)
    def test_invalid_native_shapes_and_unsupported_geometry_fail(self):
        for shape in [types.SimpleNamespace(half_size=np.array([1.,-1.,1.]),local_pose=np.eye(4)),
                      types.SimpleNamespace(vertices=np.array([[np.nan,0,0]]),scale=np.ones(3),local_pose=np.eye(4)),
                      types.SimpleNamespace(vertices=np.zeros((1,3)),scale=np.array([1,0,1]),local_pose=np.eye(4))]:
            with self.assertRaises(ValueError):overlay.native_component_vertices(types.SimpleNamespace(collision_shapes=[shape]))
        with self.assertRaises(TypeError):overlay.native_component_vertices(types.SimpleNamespace(collision_shapes=[object()]))
        with self.assertRaises(ValueError):overlay.native_component_vertices(types.SimpleNamespace(collision_shapes=[]))
    def test_every_actual_actor_component_and_shape_required(self):
        for alteration in ['record','shape','category','actor_count']:
            audit,inverse,originals=make_fixture()
            if alteration=='record':audit._rerecord_clutter_records.pop()
            if alteration=='shape':audit.static['bottle']=fcl.FCLObject('bottle',Pose(),audit.static['bottle'].shapes[:-1],audit.static['bottle'].shape_poses[:-1])
            if alteration=='category':audit.static['uncovered']=audit.static['bottle']
            if alteration=='actor_count':audit.task.record_cluttered_objects.pop()
            with self.assertRaises(RuntimeError):overlay.build_free_clearance_proxies(audit,inverse,originals)
    def test_missing_duplicate_or_moved_original_actual_obstacle_rejected(self):
        audit,inverse,originals=make_fixture()
        for altered in [originals[:-1], originals+[originals[0]]]:
            with self.assertRaises(RuntimeError):overlay.build_free_clearance_proxies(audit,inverse,altered)
        obj=originals[0];shift=overlay._matrix(obj.pose);shift[:3,3]+=.01
        originals[0]=fcl.FCLObject(obj.name,Pose(shift),obj.shapes,obj.shape_poses)
        with self.assertRaises(RuntimeError):overlay.build_free_clearance_proxies(audit,inverse,originals)
    def test_native_primitives_include_all_corners_and_match_original_extractor(self):
        shapes=[physx.PhysxCollisionShapeBox(np.array([.01,.02,.03]),MATERIAL),
                physx.PhysxCollisionShapeSphere(.025,MATERIAL),
                physx.PhysxCollisionShapeCapsule(.02,.05,MATERIAL),
                physx.PhysxCollisionShapeCylinder(.02,.05,MATERIAL)]
        for index,shape in enumerate(shapes):shape.local_pose=sapien.Pose([index*.1,.01,-.02],[.9238795,0,0,.3826834])
        entity,component=native_entity('primitives',shapes)
        local,rows=overlay.native_component_vertices(component);expected,_=fma._link_vertices(component)
        np.testing.assert_allclose(local,expected,rtol=0,atol=1e-12)
        self.assertEqual(len(local),32);self.assertEqual(len(rows),4)
    def test_proxy_AST_only_inserts_three_free_statements_and_keeps_all_prior_gates(self):
        for name in ['full_native_approach_equivalent_goals_v6','full_native_approach_equivalent_goals_dense_v7']:
            before,path=parent(name);saved=ast.dump(before,include_attributes=False)
            child,receipt=overlay.derive_proxy_ast(before)
            owner=next(n for n in ast.walk(child) if isinstance(n,ast.Try) and any(isinstance(s,ast.Assign) and isinstance(s.value,ast.Call) and ast.unparse(s.value.func)=='PlanningWorld' for s in n.body))
            index=next(i for i,s in enumerate(owner.body) if isinstance(s,ast.Assign) and isinstance(s.value,ast.Call) and ast.unparse(s.value.func)=='PlanningWorld')
            del owner.body[index-3:index]
            self.assertEqual(ast.dump(child,include_attributes=False),saved)
            self.assertEqual(ast.dump(before,include_attributes=False),saved)
            self.assertFalse(receipt['carry_AST_changed'])
    def test_controlled_install_source_registration_no_carry_scene_or_physics_change(self):
        modules={};hashes={}
        for name in ['full_native_approach_equivalent_goals_v6','full_native_approach_equivalent_goals_dense_v7']:
            tree,path=parent(name);hashes[str(path)]=overlay._sha(path);namespace={}
            exec(compile(tree,str(path),'exec'),namespace);function=namespace['plan_free_approach']
            function.__rerecord_derivation_ast__=tree
            function.__rerecord_window_overlay__={'controlled_parent_window':True}
            function.__rerecord_conservative_ompl_overlay__={'range':.01,'simplify':False}
            module=types.ModuleType(name);module.plan_free_approach=function;modules[name]=module
        carry=types.ModuleType('full_native_carry_planner_v31');carry.plan_carry_to_task_pose=lambda:None
        oldcarry=carry.plan_carry_to_task_pose;modules[carry.__name__]=carry
        class Expert:
            def __init__(self):self.sources={};self.report={}
        expert=types.ModuleType('expert_generation');expert.ExpertSession=Expert;modules[expert.__name__]=expert
        output=ROOT/'planning/free_clearance_proxy_CPU_component_20261014';output.mkdir(exist_ok=True)
        with patch.dict(sys.modules,modules),patch.object(overlay,'DIRECTORY',output),patch.object(overlay,'_INSTALLED',None):
            receipt=overlay.install_free_clearance_proxy_overlay();session=Expert()
            self.assertEqual(len(receipt['functions']),2);self.assertIs(carry.plan_carry_to_task_pose,oldcarry)
            self.assertEqual(session.sources[str(Path(overlay.__file__).resolve())],overlay._sha(overlay.__file__))
            self.assertEqual(len(session.sources),3);self.assertIn('free_planning_clearance_proxy_overlay',session.report)
        self.assertTrue(all(overlay._sha(path)==sha for path,sha in hashes.items()))
    def test_ambiguous_nonfree_or_already_extended_world_AST_rejected(self):
        for expression in ['PlanningWorld([model], [table_object,bottle_object])','PlanningWorld([model], obstacles+others)']:
            tree=ast.parse('def plan_free_approach():\n    world = '+expression)
            with self.assertRaises(RuntimeError):overlay.derive_proxy_ast(tree)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClearanceProxyCounterexamples))
    receipt={'date':'2026-10-14','passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'scope':'CPU real native PhysX shapes/full bottle asset scale and FCL boxes, transform/coverage/invalid-geometry counterexamples, unchanged full free AST and source registration; no actual randomized whole scene or physical success claimed',
        'native_geometry_proofs':PROOFS,'sources_sha256':{str(p):overlay._sha(p) for p in [Path(__file__),Path(overlay.__file__),ASSETS/'collision/base13.glb',ASSETS/'model_data13.json',ROOT/'source/adjust_bottle_20261002/full_mesh_path_audit.py',ROOT/'planning/randomized_environment_guard.py']},
        'originals_or_frozen_production_modules_overwritten':False,'GPU_started':False,'scene_created':False,'physical_steps':0}
    (ROOT/'planning/free_clearance_proxy_CPU_counterexamples_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
