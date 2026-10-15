"""CPU/mock only: no SAPIEN, GPU, real physics, episode or success fixtures."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import collector_settle_wake_v5 as collector
import worker_settle_wake_v5 as worker

WORK = Path(__file__).resolve().parent
DT = .004000000189989805


class SceneCounter:
    def __init__(self, dt=DT):
        self.dt, self.mock_calls = dt, 0
    def get_timestep(self):
        return self.dt
    def step(self):
        self.mock_calls += 1


def task(dt=DT):
    return SimpleNamespace(scene=SceneCounter(dt), robot=SimpleNamespace(communication_flag=False))


def thin_module():
    class FakeBase:
        calls = 0
        def __init__(self, *args):
            FakeBase.calls += 1
            self.sources, self.report, self.write_count = {}, {}, 0
        def write(self):
            self.write_count += 1
    p = WORK / 'expert_v9_settle_wake_v5.py'
    tree = ast.parse(p.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'expert_generation_v9_equivalent_goals_dense_v7')]
    env = {'__file__': str(p), '__name__': 'independent_thin_expert_mock', '_BaseSession': FakeBase,
           'install_initializer': object(), 'command24': object(), 'measured24': object(), '_sha': collector.sha}
    exec(compile(tree, str(p) + ':CPU_mock_only', 'exec'), env)
    return env['ExpertSession'], FakeBase


class Contracts(unittest.TestCase):
    def test_default_zero_has_no_extra_calls(self):
        t = task(); r = collector._extra_initial_settle(t, {})
        self.assertEqual((r['requested_extra_seconds'], r['extra_physics_steps'], t.scene.mock_calls), (0, 0, 0))
        self.assertFalse(r['recording_or_episode_pause'])
        self.assertTrue(r['stability_not_certified'])

    def test_mock_eight_seconds_actual_dt(self):
        t = task(); r = collector._extra_initial_settle(t, {'initial_extra_settle_s': 8})
        self.assertEqual((r['extra_physics_steps'], r['completed_extra_physics_steps'], t.scene.mock_calls), (2000, 2000, 2000))
        self.assertEqual(r['actual_extra_simulated_seconds'], 2000 * DT)
        self.assertNotEqual(r['actual_extra_simulated_seconds'], 8.0)

    def test_mock_twelve_seconds(self):
        t = task(); r = collector._extra_initial_settle(t, {'initial_extra_settle_s': 12})
        self.assertEqual((r['extra_physics_steps'], t.scene.mock_calls), (3000, 3000))

    def test_unknown_duration_reject(self):
        for value in (True, False, '8', None, -8, 4, 1, 8.00001, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collector.settle_plan({'initial_extra_settle_s': value}, DT)

    def test_bad_dt_reject(self):
        for value in (True, 0, -.004, .008, .0041, float('nan'), float('inf'), '.004'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collector.settle_plan({}, value)

    def test_reentrant_reject(self):
        t = task(); collector._extra_initial_settle(t, {})
        with self.assertRaises(RuntimeError): collector._extra_initial_settle(t, {})
        self.assertEqual(t.scene.mock_calls, 0)

    def test_hardware_io_reject(self):
        t = task(); t.robot.communication_flag = True
        with self.assertRaises(RuntimeError): collector._extra_initial_settle(t, {'initial_extra_settle_s': 8})
        self.assertEqual(t.scene.mock_calls, 0)

    def test_changed_source_sha_reject(self):
        with tempfile.TemporaryDirectory(prefix='CPU_only_settle_contract_') as d:
            p = Path(d) / 'fake_source.txt'; p.write_text('only source-pin negative fixture')
            with self.assertRaises(RuntimeError): collector.require_source(p, '0' * 64)
            self.assertEqual(collector.require_source(p, collector.sha(p)), str(p.resolve()))

    def test_producer_source_change_during_mock_reject(self):
        t = task()
        with patch.object(collector, 'producer_pins', side_effect=[{'same_source': 'before'}, {'same_source': 'after'}]):
            with self.assertRaises(RuntimeError): collector._extra_initial_settle(t, {})
        self.assertFalse(hasattr(t, '_rgb50_initial_extra_settle_receipt'))
        self.assertEqual(t.scene.mock_calls, 0)

    def test_prepare_AST_only_one_insertion_original_exact(self):
        payload = (WORK / 'generation_runner_v2.py').read_text(encoding='utf-8-sig')
        original = next(n for n in ast.parse(payload).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_task')
        changed, receipt = collector.private_prepare_tree(payload)
        function = changed.body[0]
        injected = [i for i, node in enumerate(function.body) if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == '_extra_initial_settle']
        self.assertEqual(len(injected), 1)
        index = injected[0]
        self.assertEqual(ast.unparse(function.body[index]), '_extra_initial_settle(task, spec)')
        self.assertEqual(ast.unparse(function.body[index-1]), 'task.setup_demo(**args)')
        self.assertEqual(ast.unparse(function.body[index+1]), "comp = next((c for c in task.bottle.actor.get_components() if hasattr(c, 'collision_shapes')))")
        function.body.pop(index)
        self.assertEqual(ast.dump(function, include_attributes=False), ast.dump(original, include_attributes=False))
        self.assertFalse(receipt['original_delay_statement_changed'])
        self.assertTrue(receipt['original_official_stability_chain_preserved_and_not_repeated'])
        self.assertIn('complete original task.setup_demo', receipt['insertion_scope'])

    def test_post_setup_missing_duplicate_or_signature_reject(self):
        payload = (WORK / 'generation_runner_v2.py').read_text(encoding='utf-8-sig')
        for text in [payload.replace('task.setup_demo(**args)', 'task.not_setup(**args)'),
                     payload.replace('task.setup_demo(**args)', 'task.setup_demo(**args)\n    task.setup_demo(**args)'),
                     payload.replace('task.setup_demo(**args)', 'task.setup_demo(args)'),
                     payload.replace('task.setup_demo(**args)', 'task.setup_demo(**other)')]:
            with self.assertRaises(RuntimeError): collector.private_prepare_tree(text)

    def test_prepare_changed_delay_or_duplicate_reject(self):
        payload = (WORK / 'generation_runner_v2.py').read_text()
        for text in [payload.replace('self.delay(4)', 'self.delay(8)'), payload.replace('self.delay(4)', 'self.delay(4)\n            self.delay(4)')]:
            with self.assertRaises(RuntimeError): collector.private_prepare_tree(text)

    def test_worker_AST_only_import_changed(self):
        payload = (WORK / 'generation_worker_v2.py').read_text()
        changed = worker.private_worker_tree(payload)
        for node in changed.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'collector_settle_wake_v5': alias.name = 'generation_runner_v2'
        self.assertEqual(ast.dump(changed, include_attributes=False), ast.dump(ast.parse(payload), include_attributes=False))

    def test_worker_expert_selection(self):
        self.assertEqual(worker.worker_argv(['w', '--jobs', 'j'])[-2:], ['--expert-module', 'expert_v9_settle_wake_v5'])
        self.assertEqual(worker.worker_argv(['w', '--expert-module=expert_v9_settle_wake_v5']), ['w', '--expert-module=expert_v9_settle_wake_v5'])
        for argv in [['w', '--expert-module', 'expert_generation'], ['w', '--expert-module'], ['w', '--expert-module=expert_v9_settle_wake_v5', '--expert-module', 'expert_v9_settle_wake_v5']]:
            with self.assertRaises(ValueError): worker.worker_argv(argv)

    def test_extra_helper_has_only_step_no_physical_setters(self):
        tree = ast.parse((WORK / 'collector_settle_wake_v5.py').read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_extra_initial_settle')
        methods = [n.func.attr for n in ast.walk(helper) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(methods.count('step'), 2)  # Wake and no-wake branches.
        self.assertFalse(set(methods) & {'delay', 'set_qpos', 'set_qvel', 'set_pose', 'set_velocity', 'set_drive_target', 'update_render', 'take_picture'})
        self.assertIs(collector.export_recording, collector.original.export_recording)

    def test_thin_expert_no_control_methods(self):
        tree = ast.parse((WORK / 'expert_v9_settle_wake_v5.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual([n.name for n in cls.body if isinstance(n, ast.FunctionDef)], ['__init__'])

    def test_thin_registry_before_run(self):
        Expert, Base = thin_module(); t = task(); collector._extra_initial_settle(t, {'initial_extra_settle_s': 8})
        e = Expert(t, {'initial_extra_settle_s': 8}, Path('/unused'), Path('/unused'))
        self.assertEqual(e.sources, collector.producer_pins())
        self.assertIs(e.report['source_hashes'], e.sources)
        self.assertEqual((Base.calls, e.write_count), (1, 1))
        self.assertNotIn('accepted', e.report)  # Never fabricate a successful episode.
        self.assertTrue(e.report['producer_provenance_registered_before_run_and_all_final_audits'])

    def test_thin_missing_or_changed_receipt_reject_before_base(self):
        Expert, Base = thin_module(); t = task()
        with self.assertRaises(RuntimeError): Expert(t, {}, Path('/unused'), Path('/unused'))
        collector._extra_initial_settle(t, {})
        t._rgb50_initial_extra_settle_receipt['source_hashes']['changed_source'] = '0' * 64
        with self.assertRaises(RuntimeError): Expert(t, {}, Path('/unused'), Path('/unused'))
        self.assertEqual(Base.calls, 0)

    def test_thin_clock_and_steps_reject(self):
        Expert, Base = thin_module(); t = task(); collector._extra_initial_settle(t, {})
        with self.assertRaises(RuntimeError): Expert(t, {'initial_extra_settle_s': 8}, Path('/unused'), Path('/unused'))
        t._rgb50_initial_extra_settle_receipt['completed_extra_physics_steps'] = 1
        with self.assertRaises(RuntimeError): Expert(t, {}, Path('/unused'), Path('/unused'))
        self.assertEqual(Base.calls, 0)



class NativeBodyMock:
    """Fake counter/state only; deliberately no SAPIEN/native scene."""
    def __init__(self, actor, q, mode='okay'):
        self.entity, self.q, self.mode = actor, q, mode
        self.collision_shapes = []
        self.sleeping, self.threshold, self.wake_calls = True, .0002, 0
        self.linear, self.angular = [0., 0., 0.], [0., 0., 0.]
    @property
    def is_sleeping(self):
        return self.sleeping
    def get_sleep_threshold(self): return self.threshold
    def get_linear_velocity(self): return self.linear
    def get_angular_velocity(self): return self.angular
    def wake_up(self):
        self.wake_calls += 1
        if self.mode != 'still_sleeping': self.sleeping = False
        if self.mode == 'pose_changed': self.entity.matrix[0][3] += .001
        if self.mode == 'linear_changed': self.linear[0] = 1e-12
        if self.mode == 'angular_changed': self.angular[0] = 1e-12
        if self.mode == 'threshold_changed': self.threshold += 1e-12
        if self.mode == 'q_changed': self.q[0] = 1e-12


def wake_task(mode='okay'):
    t = task()
    actor = SimpleNamespace(matrix=[[1., 0., 0., 0.], [0., 1., 0., 0.],
                                     [0., 0., 1., .75], [0., 0., 0., 1.]])
    q = [0.] * 36
    t.robot.left_entity = SimpleNamespace(get_qpos=lambda: q.copy())
    component = NativeBodyMock(actor, q, mode)
    actor.get_components = lambda: [component]
    t.bottle = SimpleNamespace(actor=actor, get_pose=lambda: SimpleNamespace(
        to_transformation_matrix=lambda: copy.deepcopy(actor.matrix)))
    return t, component


class WakeContracts(unittest.TestCase):
    def test_true_wake_before_exact_eight_second_mock_steps(self):
        t, body = wake_task()
        original_step = t.scene.step
        def counted():
            self.assertEqual(body.wake_calls, 1)
            self.assertFalse(body.sleeping)
            original_step()
        t.scene.step = counted
        r = collector._extra_initial_settle(t, {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True})
        self.assertEqual((body.wake_calls, t.scene.mock_calls), (1, 2000))
        wake = r['native_wake']
        self.assertTrue(wake['native_state_before']['native_is_sleeping'])
        self.assertFalse(wake['native_state_immediately_after']['native_is_sleeping'])
        self.assertTrue(wake['immediate_verification']['pose_velocity_threshold_q_before_after_equal'])
        self.assertTrue(r['stability_not_certified'] and r['no_fix_or_grasp_success_claim'])
        self.assertFalse(wake['native_state_after_extra']['native_is_sleeping'])
        self.assertTrue(r['original_setup_demo_completed_before_extra'])
        self.assertTrue(r['official_check_stable_not_repeated'])

    def test_true_wake_twelve_seconds(self):
        t, body = wake_task()
        collector._extra_initial_settle(t, {'initial_extra_settle_s': 12, 'initial_wake_before_extra_settle': True})
        self.assertEqual((body.wake_calls, t.scene.mock_calls), (1, 3000))

    def test_unrequested_wake_never_reads_body(self):
        t = task()  # No bottle on this mock: false must never read it.
        r = collector._extra_initial_settle(t, {'initial_extra_settle_s': 8})
        self.assertEqual((r['native_wake']['wake_calls'], t.scene.mock_calls), (0, 2000))
        self.assertIsNone(r['native_wake']['native_state_before'])

    def test_wake_requires_explicit_bool(self):
        for value in (1, 0, None, 'true', [], 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collector.settle_plan({'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': value}, DT)

    def test_wake_zero_or_missing_duration_rejected_before_body_write(self):
        for spec in ({'initial_wake_before_extra_settle': True}, {'initial_wake_before_extra_settle': True, 'initial_extra_settle_s': 0}):
            t, body = wake_task()
            with self.assertRaises(ValueError): collector._extra_initial_settle(t, spec)
            self.assertEqual((body.wake_calls, t.scene.mock_calls), (0, 0))

    def test_numeric_mutations_or_failed_wake_reject_before_mock_steps(self):
        for mode in ('pose_changed', 'linear_changed', 'angular_changed', 'threshold_changed', 'q_changed', 'still_sleeping'):
            t, body = wake_task(mode)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                collector._extra_initial_settle(t, {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True})
            self.assertEqual((body.wake_calls, t.scene.mock_calls), (1, 0))
            self.assertFalse(hasattr(t, '_rgb50_initial_extra_settle_receipt'))

    def test_missing_ambiguous_or_wrong_actor_body_reject(self):
        for mode in ('missing', 'duplicate', 'wrong_entity'):
            t, body = wake_task()
            if mode == 'missing': t.bottle.actor.get_components = lambda: []
            if mode == 'duplicate': t.bottle.actor.get_components = lambda: [body, body]
            if mode == 'wrong_entity': body.entity = object()
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                collector._extra_initial_settle(t, {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True})
            self.assertEqual((body.wake_calls, t.scene.mock_calls), (0, 0))

    def test_nonfinite_state_reject_before_wake(self):
        t, body = wake_task(); body.linear[1] = float('nan')
        with self.assertRaises(RuntimeError): collector._extra_initial_settle(t, {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True})
        self.assertEqual((body.wake_calls, t.scene.mock_calls), (0, 0))

    def test_already_awake_explicit_one_wake_allowed(self):
        t, body = wake_task(); body.sleeping = False
        r = collector._extra_initial_settle(t, {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True})
        self.assertEqual(body.wake_calls, 1)
        self.assertFalse(r['native_wake']['native_state_before']['native_is_sleeping'])

    def test_default_zero_physical_operations_match_frozen_v3(self):
        import collector_settle_v3 as old
        a, b = task(), task()
        original = old._extra_initial_settle(a, {})
        current = collector._extra_initial_settle(b, {})
        self.assertEqual(a.scene.mock_calls, b.scene.mock_calls)
        for key in ['requested_extra_seconds', 'actual_physics_dt_s', 'extra_physics_steps', 'actual_extra_simulated_seconds', 'completed_extra_physics_steps']:
            self.assertEqual(original[key], current[key])
        self.assertEqual(current['native_wake']['wake_calls'], 0)

    def test_new_helper_only_allowed_native_wake_or_scene_step_writes(self):
        tree = ast.parse((WORK / 'collector_settle_wake_v5.py').read_text())
        banned = {'set_sleep_threshold', 'put_to_sleep', 'set_pose', 'set_entity_pose', 'set_qpos', 'set_qvel',
                  'set_linear_velocity', 'set_angular_velocity', 'set_velocity', 'set_drive_target', 'set_drive_velocity_target', 'update_render'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, banned)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute):
                        self.assertNotIn(target.attr, {'pose', 'qpos', 'qvel', 'sleep_threshold', 'linear_velocity', 'angular_velocity'})
        wake_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'wake_up']
        self.assertEqual(len(wake_calls), 1)

    def test_true_receipt_checked_before_thin_base(self):
        Expert, Base = thin_module(); t, body = wake_task()
        spec = {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True}
        collector._extra_initial_settle(t, spec)
        e = Expert(t, spec, Path('/unused'), Path('/unused'))
        self.assertEqual((Base.calls, e.write_count), (1, 1))
        self.assertEqual(e.sources, collector.producer_pins())
        self.assertEqual(len(e.sources), 5)

    def test_forged_wake_numeric_receipt_or_count_reject_before_base(self):
        for mode in ('count', 'threshold', 'sleep'):
            Expert, Base = thin_module(); t, body = wake_task()
            spec = {'initial_extra_settle_s': 8, 'initial_wake_before_extra_settle': True}
            collector._extra_initial_settle(t, spec)
            wake = t._rgb50_initial_extra_settle_receipt['native_wake']
            if mode == 'count': wake['wake_calls'] = 2
            if mode == 'threshold': wake['native_state_immediately_after']['native_sleep_threshold'] += 1e-12
            if mode == 'sleep': wake['native_state_immediately_after']['native_is_sleeping'] = True
            with self.subTest(mode=mode), self.assertRaises(RuntimeError): Expert(t, spec, Path('/unused'), Path('/unused'))
            self.assertEqual(Base.calls, 0)

    def test_false_receipt_cannot_claim_unrequested_wake(self):
        Expert, Base = thin_module(); t = task(); collector._extra_initial_settle(t, {})
        t._rgb50_initial_extra_settle_receipt['native_wake']['wake_calls'] = 1
        with self.assertRaises(RuntimeError): Expert(t, {}, Path('/unused'), Path('/unused'))
        self.assertEqual(Base.calls, 0)

    def test_v7_base_motion_methods_inherited_exactly(self):
        p = WORK / 'expert_v9_settle_wake_v5.py'; tree = ast.parse(p.read_text())
        imports = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
        self.assertIn('expert_generation_v9_equivalent_goals_dense_v7', imports)
        class FakeBase:
            def __init__(self, *args): pass
            def run(self): return 'not executed'
            def step(self): return 'not executed'
            def grip(self): return 'not executed'
            def plan_path(self): return 'not executed'
            def dense(self): return 'not executed'
            def get_obs(self): return 'not executed'
        filtered = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'expert_generation_v9_equivalent_goals_dense_v7')]
        env = {'__file__':str(p), '__name__':'independent_v7_inheritance_mock', '_BaseSession':FakeBase,
               'install_initializer':object(), 'command24':object(), 'measured24':object(), '_sha':collector.sha}
        exec(compile(ast.Module(body=filtered, type_ignores=[]), str(p)+':CPU_inheritance_mock', 'exec'), env)
        for name in ['run','step','grip','plan_path','dense','get_obs']:
            self.assertIs(getattr(env['ExpertSession'], name), getattr(FakeBase, name))


class RoundContracts(unittest.TestCase):
    def test_default_one_round_physical_operations_match_frozen_v4(self):
        import collector_settle_wake_v4 as old
        for spec in ({}, {'initial_extra_settle_s':8},
                     {'initial_extra_settle_s':8, 'initial_wake_before_extra_settle':True},
                     {'initial_extra_settle_s':12, 'initial_wake_before_extra_settle':True}):
            a, abody = wake_task(); b, bbody = wake_task()
            prior=old._extra_initial_settle(a, spec); current=collector._extra_initial_settle(b, spec)
            self.assertEqual((a.scene.mock_calls,abody.wake_calls),(b.scene.mock_calls,bbody.wake_calls))
            for key in ['requested_extra_seconds','actual_physics_dt_s','extra_physics_steps','actual_extra_simulated_seconds','completed_extra_physics_steps']:
                self.assertEqual(prior[key],current[key])
            for key,value in prior['native_wake'].items():
                self.assertEqual(value,current['native_wake'][key])

    def test_two_rounds_with_native_sleep_between_rounds(self):
        t, body=wake_task(); original=t.scene.step
        def counted():
            expected=1 if t.scene.mock_calls<2000 else 2
            self.assertEqual(body.wake_calls, expected)
            original()
            if t.scene.mock_calls in (2000,4000): body.sleeping=True
        t.scene.step=counted
        spec={'initial_extra_settle_s':8,'initial_wake_before_extra_settle':True,'initial_wake_rounds':2}
        r=collector._extra_initial_settle(t,spec)
        self.assertEqual((body.wake_calls,t.scene.mock_calls),(2,4000))
        self.assertEqual(r['actual_extra_simulated_seconds'],4000*DT)
        self.assertEqual([x['completed_extra_physics_steps'] for x in r['native_wake']['rounds']],[2000,2000])
        self.assertEqual([x['round_index'] for x in r['native_wake']['rounds']],[1,2])
        self.assertTrue(r['native_wake']['rounds'][1]['native_state_before']['native_is_sleeping'])
        self.assertFalse(r['native_wake']['rounds'][1]['native_state_immediately_after']['native_is_sleeping'])
        self.assertTrue(r['native_wake']['rounds'][1]['native_state_after_extra']['native_is_sleeping'])
        Expert,Base=thin_module(); e=Expert(t,spec,Path('/unused'),Path('/unused'))
        self.assertEqual(Base.calls,1); self.assertEqual(len(e.sources),5)

    def test_invalid_round_count_and_combination_rejected_before_wake(self):
        for n in (0,3,-1,True,False,2.,None,'2',float('nan')):
            t,body=wake_task()
            with self.subTest(n=n),self.assertRaises(ValueError):
                collector._extra_initial_settle(t,{'initial_extra_settle_s':8,'initial_wake_before_extra_settle':True,'initial_wake_rounds':n})
            self.assertEqual((body.wake_calls,t.scene.mock_calls),(0,0))
        for spec in ({'initial_wake_rounds':2},
                     {'initial_wake_rounds':2,'initial_extra_settle_s':8},
                     {'initial_wake_rounds':2,'initial_extra_settle_s':12,'initial_wake_before_extra_settle':True}):
            t,body=wake_task()
            with self.assertRaises(ValueError):collector._extra_initial_settle(t,spec)
            self.assertEqual((body.wake_calls,t.scene.mock_calls),(0,0))

    def test_second_wake_mutation_rejects_before_second_round_steps(self):
        for mode in ('pose_changed','linear_changed','angular_changed','threshold_changed','q_changed','still_sleeping'):
            t,body=wake_task(); original=body.wake_up
            def second_bad():
                if body.wake_calls==1:body.mode=mode;body.sleeping=True
                original()
            body.wake_up=second_bad
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):
                collector._extra_initial_settle(t,{'initial_extra_settle_s':8,'initial_wake_before_extra_settle':True,'initial_wake_rounds':2})
            self.assertEqual((body.wake_calls,t.scene.mock_calls),(2,2000))
            self.assertFalse(hasattr(t,'_rgb50_initial_extra_settle_receipt'))

    def test_forged_second_round_or_aggregate_rejects_before_base(self):
        for mode in ('missing','clock','count','step','order','threshold','sleep','continuity','aggregate'):
            Expert,Base=thin_module(); t,body=wake_task()
            spec={'initial_extra_settle_s':8,'initial_wake_before_extra_settle':True,'initial_wake_rounds':2}
            collector._extra_initial_settle(t,spec); w=t._rgb50_initial_extra_settle_receipt['native_wake']; x=w['rounds'][1]
            if mode=='missing':w['rounds'].pop()
            elif mode=='clock':x['actual_extra_simulated_seconds']+=1e-12
            elif mode=='count':x['wake_calls']=2
            elif mode=='step':x['completed_extra_physics_steps']-=1
            elif mode=='order':x['round_index']=1
            elif mode=='threshold':x['native_state_immediately_after']['native_sleep_threshold']+=1e-12
            elif mode=='sleep':x['native_state_immediately_after']['native_is_sleeping']=True
            elif mode=='continuity':
                x['native_state_before']['native_sleep_threshold']+=1e-12
                x['native_state_immediately_after']['native_sleep_threshold']+=1e-12
            elif mode=='aggregate':w['native_state_after_extra']=copy.deepcopy(w['native_state_after_extra']);w['native_state_after_extra']['native_sleep_threshold']+=1e-12
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):Expert(t,spec,Path('/unused'),Path('/unused'))
            self.assertEqual(Base.calls,0)

    def test_no_physical_buffers_are_modified_by_state_reader(self):
        import numpy as np
        t,body=wake_task(); body.linear=np.zeros(3);body.angular=np.zeros(3)
        component=collector._native_bottle_component(t);before=collector._read_native_bottle_state(t,component)
        body.linear[0]=1.;body.angular[0]=1.
        self.assertEqual(before['linear_velocity_m_s'],[0.,0.,0.])
        self.assertEqual(before['angular_velocity_rad_s'],[0.,0.,0.])

    def test_thin_changed_round_spec_rejects_before_base(self):
        Expert,Base=thin_module();t,body=wake_task()
        collector._extra_initial_settle(t,{'initial_extra_settle_s':8,'initial_wake_before_extra_settle':True})
        with self.assertRaises(RuntimeError):Expert(t,{'initial_extra_settle_s':8,'initial_wake_before_extra_settle':True,'initial_wake_rounds':2},Path('/unused'),Path('/unused'))
        self.assertEqual(Base.calls,0)
if __name__ == '__main__':
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Contracts), unittest.defaultTestLoader.loadTestsFromTestCase(WakeContracts), unittest.defaultTestLoader.loadTestsFromTestCase(RoundContracts)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'passed': result.wasSuccessful(), 'cases': result.testsRun,
        'scope': 'CPU AST and mock fixtures only; scene counters are fake, no GPU/SAPIEN/physics/episode success',
        'new_physics_steps': 0, 'gpu_runs': 0, 'source_sha256': {str((WORK / n).resolve()): collector.sha(WORK / n)
        for n in ('collector_settle_wake_v5.py', 'worker_settle_wake_v5.py', 'expert_v9_settle_wake_v5.py', 'test_collector_settle_wake_v5.py')}}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
