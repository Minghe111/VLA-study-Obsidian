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

    def test_expert_only_init_and_original_v4_motion_inheritance(self):
        p=WORK/'expert_v9_wake_native_topp_request_v1.py';tree=ast.parse(p.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));self.assertEqual([n.name for n in cls.body if isinstance(n,ast.FunctionDef)],['__init__'])
        self.assertIn('expert_v9_settle_wake_v4',[n.module for n in tree.body if isinstance(n,ast.ImportFrom)])
        class FakeBase:
            def run(self):pass
            def step(self):pass
            def grip(self):pass
            def dense(self):pass
            def plan_path(self):pass
            def get_obs(self):pass
        env={'__file__':str(p),'_BaseSession':FakeBase,'install_initializer':object(),'command24':object(),'measured24':object(),'_sha':overlay.sha}
        filtered=ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='expert_v9_settle_wake_v4')],type_ignores=[])
        exec(compile(filtered,str(p)+':mock_inheritance','exec'),env)
        for name in ['run','step','grip','dense','plan_path','get_obs']:self.assertIs(getattr(env['ExpertSession'],name),getattr(FakeBase,name))

    def test_worker_only_expert_selection_diff_from_frozen_v4(self):
        p=WORK/'worker_wake_native_topp_request_v1.py'
        inverse=p.read_text().replace('expert_v9_wake_native_topp_request_v1','expert_v9_settle_wake_v4')
        self.assertEqual(ast.dump(ast.parse(inverse),include_attributes=False),ast.dump(ast.parse((WORK/'worker_settle_wake_v4.py').read_text()),include_attributes=False))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    print(json.dumps({'passed':result.wasSuccessful(),'cases':result.testsRun,
        'scope':'Independent CPU/AST/actual installed stateless TOPP on retained failed knots; no scene/renderer/GPU/physical-success fixture. Original failures unchanged.',
        'scene_steps':0,'renders':0,'gpu_runs':0,
        'source_sha256':{str((WORK/n).resolve()):overlay.sha(WORK/n) for n in ['native_topp_request_margin_v1.py','expert_v9_wake_native_topp_request_v1.py','worker_wake_native_topp_request_v1.py','test_native_topp_request_margin_v1.py']}},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
