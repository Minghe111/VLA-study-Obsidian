"""CPU-only official TOPP/AST/negative fixtures; no scene or physical success."""
import ast
import copy
import hashlib
import importlib
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import mplib
import native_topp_request_margin_v1 as overlay

WORK=pathlib.Path(__file__).resolve().parent
LEGACY=WORK.parent/'adjust_bottle_20261002'
sys.path.insert(0,str(LEGACY))
from expert_generation_v6 import _NativeCarryReadback  # Same production pre-dependency vector bootstrap.
import full_native_carry_planner_v31 as original
DT=.004000000189989805


def holder_task():
    planner=types.SimpleNamespace(joint_vel_limits=np.ones(6),joint_acc_limits=np.ones(6))
    planner.TOPP=types.MethodType(mplib.Planner.TOPP,planner)
    wrapper=types.SimpleNamespace(planner=planner,TOPP=planner.TOPP)
    task=types.SimpleNamespace(scene=types.SimpleNamespace(get_timestep=lambda:DT),
        robot=types.SimpleNamespace(left_mplib_planner=wrapper))
    return task,planner


def historical_knots():
    root=WORK/'batch_candidates/v4_gpu4_fanta_left_diagonal_post_setup_wake8_attempt260'
    rows=[]
    for t in sorted(root.iterdir()):
        if not t.is_dir() or not t.name.startswith('slot0000026'):continue
        d=json.loads((t/'expert_report.json').read_text())
        m=d['raw_native_carry_results'][0]['planning_metadata']
        inds=[d['full_qpos_joint_names'].index(n) for n in m['move_group_joint_names']]
        rows.append((t,np.asarray(m['cartesian_full_native_knots_rad'])[:,inds]))
    if len(rows)!=2:raise RuntimeError('Exactly two retained failed26 knot proofs required')
    return rows


class Contracts(unittest.TestCase):
    def test_margin_validation(self):
        self.assertEqual(overlay.request_margin({}),1.)
        self.assertEqual(overlay.request_margin({'native_topp_request_margin':.99}),.99)
        for value in (True,False,None,'0.99',.98,.995,0,-1,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):overlay.request_margin({'native_topp_request_margin':value})

    def test_original_default_function_and_module_identity(self):
        private,binding=overlay.private_carry_module(original,1.)
        self.assertIs(private,original)
        self.assertIs(private.official_topp_command_polyline,original.official_topp_command_polyline)
        self.assertTrue(binding['original_module_and_function_returned_unchanged'])

    def test_private_AST_only_native_call_changed(self):
        payload=pathlib.Path(original.__file__).read_text()
        before=next(n for n in ast.parse(payload).body if isinstance(n,ast.FunctionDef) and n.name=='official_topp_command_polyline')
        after=overlay.private_official_tree(payload).body[0]
        calls=[n for n in ast.walk(after) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_safe_native_request']
        self.assertEqual(len(calls),1)
        node=calls[0];prior=ast.parse('native(p, step=time_step)').body[0].value
        node.func,node.args,node.keywords=prior.func,prior.args,prior.keywords
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))

    def test_changed_native_site_or_carry_source_rejected(self):
        payload=pathlib.Path(original.__file__).read_text()
        with self.assertRaises(RuntimeError):overlay.private_official_tree(payload.replace('native(p,step=time_step)','native(p,step=.004)'))
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'fake_carry.py';p.write_text(payload+'\n# intentional source mismatch negative fixture\n')
            fake=types.SimpleNamespace(__file__=str(p))
            with self.assertRaises(RuntimeError):overlay.private_carry_module(fake,.99)

    def test_real_retained_failed_knots_and_request_restoration(self):
        oldfunction=original.official_topp_command_polyline
        private,binding=overlay.private_carry_module(original,.99)
        for trial,path in historical_knots():
            task,planner=holder_task();v,a=planner.joint_vel_limits,planner.joint_acc_limits
            with self.assertRaisesRegex(ValueError,'Native velocity/acceleration limits exceeded'):
                original.official_topp_command_polyline(task,path,time_step=DT)
            pos,vel,ts,meta=private.official_topp_command_polyline(task,path,time_step=DT)
            self.assertIs(planner.joint_vel_limits,v);self.assertIs(planner.joint_acc_limits,a)
            np.testing.assert_array_equal(v,np.ones(6));np.testing.assert_array_equal(a,np.ones(6))
            self.assertEqual(meta['official_joint_velocity_limits_rad_s'],[1.]*6)
            self.assertEqual(meta['official_joint_acceleration_limits_rad_s2'],[1.]*6)
            self.assertLessEqual(np.max(np.abs(np.diff(pos,axis=0)/DT)),1.+1e-5)
            self.assertLessEqual(np.max(np.abs(np.diff(vel,axis=0)/DT)),1.+1e-5)
            proof=meta['conservative_native_TOPP_request']
            self.assertTrue(proof['restoration_verified'] and proof['additional_strict_hardware_clock_audit_passed'])
            self.assertEqual(proof['native_requested_velocity_limits_rad_s'],[.99]*6)
            self.assertEqual(meta['audit_vertices'],pos.tolist())
            self.assertEqual(meta['returned_commands_sha256'],hashlib.sha256(pos.tobytes()).hexdigest())
            # Actual installed native binding on same requests returns exact arrays.
            direct=types.SimpleNamespace(joint_vel_limits=np.full(6,.99),joint_acc_limits=np.full(6,.99))
            native_ts,native_pos,native_vel,native_acc,T=mplib.Planner.TOPP(direct,path,step=DT)
            np.testing.assert_array_equal(ts,native_ts);np.testing.assert_array_equal(pos,native_pos);np.testing.assert_array_equal(vel,native_vel)
        self.assertIs(original.official_topp_command_polyline,oldfunction)
        self.assertIs(private.plan_carry_to_task_pose.__code__,original.plan_carry_to_task_pose.__code__)
        self.assertIs(private.plan_carry_to_task_pose.__globals__['official_topp_command_polyline'],private.official_topp_command_polyline)

    def test_native_exception_restores_limit_objects_and_values(self):
        import mplib.planner as installed
        task,planner=holder_task();v,a=planner.joint_vel_limits,planner.joint_acc_limits
        p=np.linspace(np.zeros(6),np.ones(6)*.1,3)
        with patch.object(installed.algo,'TOPPRA',side_effect=RuntimeError('deliberate CPU-only native dependency exception')):
            with self.assertRaisesRegex(RuntimeError,'deliberate CPU-only'):
                overlay._native_request(task,planner,planner.TOPP,p,step=DT,margin=.99)
        self.assertIs(planner.joint_vel_limits,v);self.assertIs(planner.joint_acc_limits,a)
        np.testing.assert_array_equal(v,np.ones(6));np.testing.assert_array_equal(a,np.ones(6))
        self.assertFalse(task._rgb50_native_topp_request_active)
        self.assertTrue(task._rgb50_native_topp_request_receipts[-1]['restoration_verified'])

    def test_unknown_or_wrong_bound_native_rejected_before_request(self):
        task,planner=holder_task();other,_=holder_task()
        p=np.linspace(np.zeros(6),np.ones(6)*.1,3)
        for native in (lambda *a,**kw:None,other.robot.left_mplib_planner.TOPP):
            with self.assertRaises(ValueError):overlay._native_request(task,planner,native,p,step=DT,margin=.99)
        self.assertFalse(hasattr(task,'_rgb50_native_topp_request_receipts'))

    def test_source_drift_and_nested_request_rejected(self):
        task,planner=holder_task();p=np.linspace(np.zeros(6),np.ones(6)*.1,3)
        with patch.object(overlay.inspect,'getsource',return_value='changed native implementation'):
            with self.assertRaises(ValueError):overlay._native_request(task,planner,planner.TOPP,p,step=DT,margin=.99)
        task._rgb50_native_topp_request_active=True
        with self.assertRaises(RuntimeError):overlay._native_request(task,planner,planner.TOPP,p,step=DT,margin=.99)

    def test_strict_clock_negative_restores_before_reject(self):
        import mplib.planner as installed
        class FakeTrajectory:
            duration=.020
            def __call__(self,t,order=0):
                if order==0:return np.tile(t[:,None]*2,(1,6))
                return np.zeros((len(t),6))
        task,planner=holder_task();v,a=planner.joint_vel_limits,planner.joint_acc_limits;p=np.linspace(np.zeros(6),np.ones(6)*.1,3)
        with patch.object(installed.algo,'TOPPRA',return_value=types.SimpleNamespace(compute_trajectory=lambda:FakeTrajectory())):
            with self.assertRaisesRegex(ValueError,'strict hardware 4ms FD'):
                overlay._native_request(task,planner,planner.TOPP,p,step=DT,margin=.99)
        self.assertIs(planner.joint_vel_limits,v);self.assertIs(planner.joint_acc_limits,a)
        self.assertEqual(task._rgb50_native_topp_request_receipts[-1]['status'],'strict_clock_rejected')

    def test_no_scene_or_physical_setters_in_overlay(self):
        tree=ast.parse(pathlib.Path(overlay.__file__).read_text())
        banned={'step','wake_up','set_pose','set_qpos','set_qvel','set_drive_target','set_sleep_threshold','set_linear_velocity','set_angular_velocity','update_render'}
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,banned)

    def test_expert_only_init_and_original_v5_motion_inheritance(self):
        p=WORK/'expert_v9_wake_native_topp_request_v3.py';tree=ast.parse(p.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));self.assertEqual([n.name for n in cls.body if isinstance(n,ast.FunctionDef)],['__init__'])
        self.assertIn('expert_v9_settle_wake_v5',[n.module for n in tree.body if isinstance(n,ast.ImportFrom)])
        class FakeBase:
            def run(self):pass
            def step(self):pass
            def grip(self):pass
            def dense(self):pass
            def plan_path(self):pass
            def get_obs(self):pass
        env={'__file__':str(p),'_BaseSession':FakeBase,'install_initializer':object(),'command24':object(),'measured24':object(),'_sha':overlay.sha}
        filtered=ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='expert_v9_settle_wake_v5')],type_ignores=[])
        exec(compile(filtered,str(p)+':mock_inheritance','exec'),env)
        for name in ['run','step','grip','dense','plan_path','get_obs']:self.assertIs(getattr(env['ExpertSession'],name),getattr(FakeBase,name))

    def test_worker_only_expert_selection_diff_from_frozen_v5(self):
        p=WORK/'worker_wake_native_topp_request_v3.py'
        inverse=p.read_text().replace('expert_v9_wake_native_topp_request_v3','expert_v9_settle_wake_v5')
        self.assertEqual(ast.dump(ast.parse(inverse),include_attributes=False),ast.dump(ast.parse((WORK/'worker_settle_wake_v5.py').read_text()),include_attributes=False))


def production_thin_fixture(mode='okay'):
    from expert_generation_v6 import _NativeCarryReadback
    class FakeBase:
        def __init__(self,task,spec,out,legacy):
            self.sources={};self.report={};self.out=pathlib.Path(out)
            self.dense_calls=2;self.native_carry_candidate=0;self.write_count=0
            self.ncp=_NativeCarryReadback(self,original)
            self.initial_delegate=self.ncp;self.initial_source=self.ncp.source
            if mode=='wrong_type':self.ncp=types.SimpleNamespace(session=self,source=original)
            elif mode=='wrong_owner':self.ncp.session=object()
            elif mode=='wrong_source':self.ncp.source=types.SimpleNamespace(plan_carry_to_task_pose=original.plan_carry_to_task_pose)
            elif mode=='shadow_method':self.ncp.plan_carry_to_task_pose=lambda *a,**kw:None
            elif mode=='bound_source_method':
                module=types.ModuleType('bad_bound_module')
                module.plan_carry_to_task_pose=types.MethodType(lambda *a,**kw:None,object())
                self.ncp.source=module
        def write(self):self.write_count+=1
    p=WORK/'expert_v9_wake_native_topp_request_v3.py';tree=ast.parse(p.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='expert_v9_settle_wake_v5')]
    env={'__file__':str(p),'_BaseSession':FakeBase,'install_initializer':object(),'command24':object(),'measured24':object(),'_sha':overlay.sha}
    exec(compile(tree,str(p)+':CPU_production_delegate_fixture','exec'),env)
    return env['ExpertSession']


class DelegateContracts(unittest.TestCase):
    def test_real_frozen_delegate_to_real_rebound_module_entry_task_identity(self):
        Expert=production_thin_fixture()
        class SampleScene:
            pass
        raw,planner=holder_task();task=SampleScene();task.scene=raw.scene;task.robot=raw.robot
        self.assertFalse(hasattr(task,'source'))  # Production interface: task is not the delegate.
        with tempfile.TemporaryDirectory() as d:
            e=Expert(task,{'native_topp_request_margin':.99},d,LEGACY)
            self.assertIs(e.ncp,e.initial_delegate);self.assertIs(e.ncp.session,e)
            self.assertIs(e.ncp.source.plan_carry_to_task_pose.__code__,original.plan_carry_to_task_pose.__code__)
            entered=[]
            def stop_at_real_carry_entry(frame,event,arg):
                if event=='call' and frame.f_code is original.plan_carry_to_task_pose.__code__:
                    entered.append(frame.f_locals['task'])
                    raise RuntimeError('CPU zero-scene-stop at actual original carry function entry')
                return stop_at_real_carry_entry
            previous=sys.gettrace()
            try:
                sys.settrace(stop_at_real_carry_entry)
                with self.assertRaisesRegex(RuntimeError,'CPU zero-scene-stop'):
                    e.ncp.plan_carry_to_task_pose(task,[0.]*7,'left')
            finally:sys.settrace(previous)
            self.assertEqual(len(entered),1);self.assertIs(entered[0],task)
            self.assertFalse(task._rgb50_native_topp_request_receipts)

    def test_default_one_retains_real_delegate_and_original_source_identity(self):
        Expert=production_thin_fixture();task,planner=holder_task()
        with tempfile.TemporaryDirectory() as d:
            e=Expert(task,{},d,LEGACY)
            self.assertIs(e.ncp,e.initial_delegate);self.assertIs(e.ncp.source,e.initial_source)
            self.assertIs(e.ncp.source,original)
            self.assertTrue(e.report['native_topp_request_binding']['default1_original_delegate_and_source_identity_preserved'])

    def test_thin_private_source_real_TOPP_and_original_metadata_limits(self):
        Expert=production_thin_fixture();task,planner=holder_task();v,a=planner.joint_vel_limits,planner.joint_acc_limits
        with tempfile.TemporaryDirectory() as d:
            e=Expert(task,{'native_topp_request_margin':.99},d,LEGACY)
            _,path=historical_knots()[1]
            pos,vel,ts,meta=e.ncp.source.official_topp_command_polyline(task,path,time_step=DT)
            self.assertIs(planner.joint_vel_limits,v);self.assertIs(planner.joint_acc_limits,a)
            self.assertEqual(meta['official_joint_velocity_limits_rad_s'],[1.]*6)
            self.assertIs(e.report['native_topp_request_receipts'],task._rgb50_native_topp_request_receipts)
            self.assertEqual(len(e.report['native_topp_request_receipts']),1)
            for p in ['native_topp_request_margin_v1.py','expert_v9_wake_native_topp_request_v3.py','worker_wake_native_topp_request_v3.py','expert_generation_v6.py']:
                self.assertEqual(e.sources[str((WORK/p).resolve())],overlay.sha(WORK/p))
            self.assertNotIn('accepted',e.report)

    def test_actual_v6_readback_method_keeps_raw_result_recording(self):
        Expert=production_thin_fixture();task,planner=holder_task()
        with tempfile.TemporaryDirectory() as d:
            e=Expert(task,{'native_topp_request_margin':.99},d,LEGACY)
            calls=[]
            def cpu_raw_protocol_fixture(received,*args,**kwargs):
                calls.append(received)
                return {'status':'CPU_ONLY_DELEGATE_PROTOCOL_FIXTURE','planning_metadata':{'physical_success':False}}
            e.ncp.source.plan_carry_to_task_pose=cpu_raw_protocol_fixture  # Private mock namespace only.
            result=e.ncp.plan_carry_to_task_pose(task,[0.]*7,'left')
            self.assertEqual(calls,[task]);self.assertIsNotNone(result)
            self.assertEqual(len(e.report['raw_native_carry_results']),1)
            evidence=json.loads((pathlib.Path(d)/'native_carry_2_00_result.json').read_text())
            self.assertFalse(evidence['physical_success_checked'])
            self.assertIs(original.plan_carry_to_task_pose.__globals__['official_topp_command_polyline'],original.official_topp_command_polyline)

    def test_wrong_delegate_owner_type_source_or_shadow_rejects(self):
        for mode in ['wrong_type','wrong_owner','wrong_source','shadow_method','bound_source_method']:
            Expert=production_thin_fixture(mode);task,planner=holder_task()
            with tempfile.TemporaryDirectory() as d,self.subTest(mode=mode),self.assertRaises(RuntimeError):
                Expert(task,{'native_topp_request_margin':.99},d,LEGACY)
            self.assertFalse(hasattr(task,'_rgb50_native_topp_request_receipts'))

    def test_frozen_readback_sha_drift_rejected_before_binding(self):
        Expert=production_thin_fixture();task,planner=holder_task()
        env=Expert.__init__.__globals__
        with tempfile.TemporaryDirectory() as d,patch.dict(env,{'_sha':lambda p:'0'*64}),self.assertRaises(RuntimeError):
            Expert(task,{'native_topp_request_margin':.99},d,LEGACY)
        self.assertFalse(hasattr(task,'_rgb50_native_topp_request_receipts'))

    def test_production_binding_never_recompiles_readback_bound_method(self):
        p=WORK/'expert_v9_wake_native_topp_request_v3.py';tree=ast.parse(p.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='private_carry_module']
        self.assertEqual(len(calls),1);self.assertEqual(ast.unparse(calls[0].args[0]),'source')
        self.assertEqual(len([n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='source' for t in n.targets)]),1)


class CombinationContracts(unittest.TestCase):
    def test_only_reviewed_base_and_producer_binding_changed(self):
        new=(WORK/'expert_v9_wake_native_topp_request_v3.py').read_text()
        inverse=(new.replace('wake-v5/v7','wake-v4/v7')
            .replace('expert_v9_settle_wake_v5','expert_v9_settle_wake_v4')
            .replace('worker_wake_native_topp_request_v3.py','worker_wake_native_topp_request_v2.py')
            .replace('v9_dense_v7_wake_v5_explicit_native_topp_request_v3','v9_dense_v7_wake_v4_explicit_native_topp_request_v2'))
        self.assertEqual(ast.dump(ast.parse(inverse),include_attributes=False),ast.dump(ast.parse((WORK/'expert_v9_wake_native_topp_request_v2.py').read_text()),include_attributes=False))
        self.assertEqual(overlay.sha(WORK/'native_topp_request_margin_v1.py'),'ceae0ed22ffb0a98db6db20f39ca5ae584fac40c46dc2155c4a96b0bd785bd5d')

    def test_reviewed_v5_round_plan_and_point99_request_are_independent(self):
        import collector_settle_wake_v5 as collector
        spec={'initial_wake_before_extra_settle':True,'initial_extra_settle_s':8,'initial_wake_rounds':2,'native_topp_request_margin':.99}
        plan=collector.settle_plan(spec,DT)
        self.assertEqual(plan['initial_wake_rounds'],2)
        self.assertEqual(plan['extra_physics_steps_per_round'],2000)
        self.assertEqual(plan['extra_physics_steps'],4000)
        self.assertEqual(overlay.request_margin(spec),.99)
        for bad in [True,0,3,1.5,'2']:
            with self.subTest(rounds=bad),self.assertRaises((ValueError,RuntimeError)):
                collector.settle_plan({**spec,'initial_wake_rounds':bad},DT)
        self.assertEqual(collector.settle_plan({'native_topp_request_margin':.99},DT)['extra_physics_steps'],0)

    def test_v5_motion_and_initialization_validation_not_overridden(self):
        base=ast.parse((WORK/'expert_v9_settle_wake_v5.py').read_text())
        self.assertIn('expert_generation_v9_equivalent_goals_dense_v7',[n.module for n in base.body if isinstance(n,ast.ImportFrom)])
        cls=next(n for n in base.body if isinstance(n,ast.ClassDef))
        self.assertEqual([n.name for n in cls.body if isinstance(n,ast.FunctionDef)],['__init__'])
        new=ast.parse((WORK/'expert_v9_wake_native_topp_request_v3.py').read_text())
        calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='__init__']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.unparse(v) for v in calls[0].args],['task','sample_spec','output_dir','legacy_dir'])
        for name in ['run_episode','install_expert']:
            n1=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            old=ast.parse((WORK/'expert_v9_wake_native_topp_request_v2.py').read_text())
            n2=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(n1,include_attributes=False),ast.dump(n2,include_attributes=False))

    def test_all_actual_v5_and_new_producer_sources_bound(self):
        import collector_settle_wake_v5 as collector
        pins=collector.producer_pins()
        for name in ['generation_runner_v2.py','generation_worker_v2.py','collector_settle_wake_v5.py','worker_settle_wake_v5.py','expert_v9_settle_wake_v5.py']:
            self.assertEqual(pins[str((WORK/name).resolve())],overlay.sha(WORK/name))
        Expert=production_thin_fixture();task,planner=holder_task()
        with tempfile.TemporaryDirectory() as d:
            e=Expert(task,{'native_topp_request_margin':.99},d,LEGACY)
            for name in ['expert_v9_wake_native_topp_request_v3.py','worker_wake_native_topp_request_v3.py','native_topp_request_margin_v1.py','expert_generation_v6.py']:
                self.assertEqual(e.sources[str((WORK/name).resolve())],overlay.sha(WORK/name))
            self.assertNotIn('accepted',e.report)

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Contracts),unittest.defaultTestLoader.loadTestsFromTestCase(DelegateContracts),unittest.defaultTestLoader.loadTestsFromTestCase(CombinationContracts)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'passed':result.wasSuccessful(),'cases':result.testsRun,
        'scope':'Independent CPU/AST/actual installed stateless TOPP on retained failed knots; no scene/renderer/GPU/physical-success fixture. Original failures unchanged.',
        'scene_steps':0,'renders':0,'gpu_runs':0,
        'source_sha256':{str((WORK/n).resolve()):overlay.sha(WORK/n) for n in ['native_topp_request_margin_v1.py','expert_v9_wake_native_topp_request_v3.py','worker_wake_native_topp_request_v3.py','test_native_topp_request_margin_v3.py']}},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
