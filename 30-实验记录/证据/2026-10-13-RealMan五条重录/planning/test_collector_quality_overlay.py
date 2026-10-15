"""Full original-planner call-path counterexamples with controlled CPU doubles.

These are integration/control-flow tests, not native model or physical evidence.
"""
import ast, hashlib, json, math, sys, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import collector_quality_overlay as overlay
import equivalent_arm_goals_v5 as equivalent

SOURCE = Path(__file__).resolve().parent.parent / 'source/rgb50_20261007_v1'
BASE = np.zeros(36); BASE[5] = .93
RAW = BASE.copy(); RAW[5] = -5.25
NEAR = RAW.copy(); NEAR[5] += 2*math.pi

class Pose:
    def __init__(self, matrix=None): self.matrix = np.eye(4) if matrix is None else np.asarray(matrix).copy()
    def to_transformation_matrix(self): return self.matrix.copy()

class PK:
    def __init__(self): self.q = BASE.copy()
    def get_link_names(self): return ['l_inspire_base_link'] + ['link'+str(i) for i in range(92)]
    def compute_forward_kinematics(self, q): self.q = np.asarray(q).copy()
    def get_link_pose(self, i):
        t = np.eye(4); a = self.q[5]; t[:2,:2] = [[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]
        return Pose(t)
    def compute_IK_CLIK(self, *args, **kw): return RAW.copy(), True, 0.

class Model:
    def __init__(self): self.q=BASE.copy(); self.pose=Pose(); self.group=['original']
    def get_qpos(self): return self.q.copy()
    def set_qpos(self,q,full=True): self.q=np.asarray(q).copy()
    def get_base_pose(self): return self.pose
    def set_base_pose(self,pose): self.pose=pose
    def set_move_group(self,group): self.group=group if isinstance(group,list) else [group]
    def get_move_group_end_effectors(self): return list(self.group)
    def get_move_group_joint_indices(self): return list(range(6)) if self.group!=['original'] else list(range(36))

class Auditor:
    def __init__(self,task):
        self.task=task; self.model=Model(); self.pk=PK(); self.initial_sim_q=BASE.copy()
        self.sim_joints=['j'+str(i) for i in range(36)]; self.Pose=Pose
        self.art=types.SimpleNamespace(active_joints=[types.SimpleNamespace(limits=np.array([[-3.1,3.1] if i!=5 else [-6.28,6.28]])) for i in range(36)])
        self.urdf_tree=types.SimpleNamespace(findall=lambda tag:[{'name':n,'type':'revolute'} for n in self.sim_joints])
        self.static={}; self.static_boxes={}; self.allowed=[]; self.metadata={}
    def audit(self,*a,**kw):
        self.task.audit_calls+=1
        return {'passed':True}

class World:
    def __init__(self,*a):pass
    def get_allowed_collision_matrix(self):return types.SimpleNamespace(set_entry=lambda *a:None)
    def set_qpos_all(self,q):pass
    def check_collision(self):return []

def fixture(mode='near'):
    task=types.SimpleNamespace(audit_calls=0,topp_calls=0,plan_goals=[],unchanged_calls=0,cluttered_table=False)
    pk=PK(); pk.compute_forward_kinematics(RAW); goal=pk.get_link_pose(0)
    task.robot=types.SimpleNamespace(left_arm_joints_name=['j'+str(i) for i in range(6)],
        left_entity=types.SimpleNamespace(root_pose=Pose()),_trans_from_gripper_to_endlink=lambda *a,**kw:goal)
    task.scene=types.SimpleNamespace(get_timestep=lambda:.004)
    class Planner:
        def __init__(self,world):pass
        def plan(self,start,goals,**kw):
            task.plan_goals.append(np.asarray(goals).copy())
            end=RAW[:6] if mode=='raw_far' else np.asarray(goals[0])
            path=np.stack([start,end])
            if mode=='raw_detour':
                mid=end.copy();mid[5]=4.;path=np.stack([start,mid,end])
            return 'Exact solution',path
    def topp(task,path,side,**kwargs):
        task.topp_calls+=1
        end=RAW[:6] if mode=='retimed_far' else path[-1]
        p=path[0]+np.linspace(0,1,101)[:,None]*(end-path[0]);t=np.arange(101)*.004
        vertices=np.stack([path[0],RAW[:6] if mode=='audit_far' else path[-1]])
        return p,np.zeros_like(p),t,{'official_joint_velocity_limits_rad_s':[1.]*6,'audit_vertices':vertices.tolist()}
    fma=types.ModuleType('full_mesh_path_audit');fma.FullMeshPathAuditor=Auditor
    fma._pose_matrix=lambda pose:pose.to_transformation_matrix()
    fma.joint_limit_audit=lambda *a,**kw:{'passed':True}
    carry=types.ModuleType('full_native_carry_planner_v31');carry.official_topp_polyline=topp
    mplib=types.ModuleType('mplib');mplib.PlanningWorld=World
    ompl=types.ModuleType('mplib.pymp.planning.ompl');ompl.OMPLPlanner=Planner
    modules={m.__name__:m for m in [fma,carry,mplib,ompl]}
    source,function,receipt=overlay.derive_planner_source((SOURCE/'full_native_approach_equivalent_goals_dense_v7.py').read_text(),'full_native_approach_equivalent_goals_dense_v7')
    namespace=dict(overlay.__dict__)
    namespace.update(equivalent=equivalent,np=np,time=__import__('time'),
        dense_policy=types.SimpleNamespace(refine_certificate=lambda audit,vertices,side,geometry:(geometry,{'original_refine_called':True})),
        _physical_snapshot=lambda task:{'q':BASE.copy(),'bottle':np.eye(4)},
        _assert_unchanged=lambda task,before:setattr(task,'unchanged_calls',task.unchanged_calls+1),
        _OVERLAY_RECEIPT=receipt)
    tree=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(tree)
    exec(compile(tree,'<controlled original dense_v7 call path>','exec'),namespace)
    return task,namespace['plan_free_approach'],modules

class IntegrationCounterexamples(unittest.TestCase):
    def run_planner(self,mode):
        task,function,modules=fixture(mode)
        with patch.dict(sys.modules,modules):result=function(task,np.eye(4),'left',ik_attempts=1)
        return task,result
    def test_nearest_original_FK_legal_family_only_one_goal(self):
        task,result=self.run_planner('near');self.assertEqual(result['status'],'Success')
        self.assertEqual(task.plan_goals[0].shape,(1,6));self.assertAlmostEqual(task.plan_goals[0][0,5],NEAR[5])
        self.assertEqual(task.audit_calls,1);self.assertEqual(task.topp_calls,1);self.assertEqual(task.unchanged_calls,1)
        proof=result['planning_metadata']['RRT_attempts'][0]['chosen_goal_target_FK_proof']
        self.assertEqual(proof['native_FK_checked_links'],93);self.assertTrue(proof['passed'])
    def test_raw_far_equivalent_endpoint_rejected_before_TOPP_or_FMA(self):
        task,result=self.run_planner('raw_far');self.assertEqual(result['status'],'Fail')
        self.assertEqual(task.topp_calls,0);self.assertEqual(task.audit_calls,0);self.assertEqual(task.unchanged_calls,1)
    def test_retimed_far_equivalent_rejected_before_FMA(self):
        task,result=self.run_planner('retimed_far');self.assertEqual(result['status'],'Fail')
        self.assertEqual(task.topp_calls,1);self.assertEqual(task.audit_calls,0)
    def test_wrong_audited_endpoint_rejected_even_when_dense_correct(self):
        task,result=self.run_planner('audit_far');self.assertEqual(result['status'],'Fail');self.assertEqual(task.audit_calls,0)
    def test_raw_detour_rejected_before_TOPP_or_FMA(self):
        task,result=self.run_planner('raw_detour');self.assertEqual(result['status'],'Fail')
        self.assertEqual(task.topp_calls,0);self.assertEqual(task.audit_calls,0)
    def test_cumulative_quality_rejects_long_previous_stage(self):
        task,function,modules=fixture()
        previous=BASE[:6].copy();previous[5]=-2.2
        p=np.stack([previous,BASE[:6]])
        task._rerecord_precontact_plans={'side':'left','position':p,'xyz':np.zeros((2,3))}
        with patch.dict(sys.modules,modules):result=function(task,np.eye(4),'left',ik_attempts=1)
        self.assertEqual(result['status'],'Fail');self.assertEqual(task.audit_calls,0)
    def test_missing_randomized_actor_reference_fails_closed(self):
        task,function,modules=fixture();task.cluttered_table=True
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(RuntimeError,'lost actual clutter'):
            function(task,np.eye(4),'left',ik_attempts=1)
    def test_uncovered_randomized_actor_fails_closed(self):
        task,function,modules=fixture();task.cluttered_table=True;task._rerecord_clutter_entities=[object()]
        with patch.dict(sys.modules,modules),self.assertRaisesRegex(RuntimeError,'missing from current'):
            function(task,np.eye(4),'left',ik_attempts=1)
    def test_explicit_original_random_clean_draw_is_legal(self):
        task,function,modules=fixture();task.cluttered_table=True
        task._rerecord_clutter_sampling_outcome='native_clean_background_draw'
        with patch.dict(sys.modules,modules):result=function(task,np.eye(4),'left',ik_attempts=1)
        self.assertEqual(result['status'],'Success')
        self.assertTrue(result['planning_metadata']['randomized_obstacle_coverage']['native_random_clean_background_draw'])
    def test_exact_source_sites_cover_both_v6_v7(self):
        for name,digest in overlay.SOURCE_PINS.items():
            path=SOURCE/(name+'.py');self.assertEqual(overlay._sha(path),digest)
            source,function,receipt=overlay.derive_planner_source(path.read_text(),name)
            self.assertEqual(source.count('official_topp_polyline(task, path, side, time_step=time_step)'),1)
            self.assertEqual(source.count('declared, near_selection = choose_nearest_family'),1)
            self.assertIn('_assert_unchanged(task, before)',source)
    def test_both_precontact_stages_measured_before_closure(self):
        p=np.zeros((1001,6));p[:,5]=np.linspace(0,3.5,len(p))
        session=types.SimpleNamespace(_rerecord_actual_motion={'q':p,'xyz':np.zeros((len(p),3)),
            'time':np.arange(len(p))*.004,'velocity_limits':np.ones(6),'hardware_limits':np.tile([-6.28,6.28],(6,1)),
            'finalized':False},report={},task=types.SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError,'actual precontact demonstration quality'):
            overlay._finish_actual(session)
        self.assertFalse(session.report['motion_quality_actual_precontact']['passed'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationCounterexamples))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'scope':'CPU integration counterexamples using controlled native/planner/physics doubles; not physical or native geometric success',
        'source_sha256':{p.name:overlay._sha(p) for p in [Path(__file__),Path(overlay.__file__),Path(overlay.__file__).parent/'near_branch_quality.py']}}
    (Path(__file__).parent/'collector_overlay_counterexample_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
