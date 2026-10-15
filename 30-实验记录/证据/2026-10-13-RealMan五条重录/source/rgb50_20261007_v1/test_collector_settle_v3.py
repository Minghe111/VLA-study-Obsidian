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
import collector_settle_v3 as collector
import worker_settle_v3 as worker

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
    p = WORK / 'expert_v9_settle_v3.py'
    tree = ast.parse(p.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == 'expert_generation_v9_pathdiag')]
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
        removed = 0
        for node in ast.walk(changed):
            if isinstance(node, ast.FunctionDef) and node.name == 'load_actors':
                before = len(node.body)
                node.body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == '_extra_initial_settle')]
                removed += before - len(node.body)
        self.assertEqual(removed, 1)
        self.assertEqual(ast.dump(changed.body[0], include_attributes=False), ast.dump(original, include_attributes=False))
        self.assertFalse(receipt['original_delay_statement_changed'])

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
                    if alias.name == 'collector_settle_v3': alias.name = 'generation_runner_v2'
        self.assertEqual(ast.dump(changed, include_attributes=False), ast.dump(ast.parse(payload), include_attributes=False))

    def test_worker_expert_selection(self):
        self.assertEqual(worker.worker_argv(['w', '--jobs', 'j'])[-2:], ['--expert-module', 'expert_v9_settle_v3'])
        self.assertEqual(worker.worker_argv(['w', '--expert-module=expert_v9_settle_v3']), ['w', '--expert-module=expert_v9_settle_v3'])
        for argv in [['w', '--expert-module', 'expert_generation'], ['w', '--expert-module'], ['w', '--expert-module=expert_v9_settle_v3', '--expert-module', 'expert_v9_settle_v3']]:
            with self.assertRaises(ValueError): worker.worker_argv(argv)

    def test_extra_helper_has_only_step_no_physical_setters(self):
        tree = ast.parse((WORK / 'collector_settle_v3.py').read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_extra_initial_settle')
        methods = [n.func.attr for n in ast.walk(helper) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(methods.count('step'), 1)
        self.assertFalse(set(methods) & {'delay', 'set_qpos', 'set_qvel', 'set_pose', 'set_velocity', 'set_drive_target', 'update_render', 'take_picture'})
        self.assertIs(collector.export_recording, collector.original.export_recording)

    def test_thin_expert_no_control_methods(self):
        tree = ast.parse((WORK / 'expert_v9_settle_v3.py').read_text())
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


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'passed': result.wasSuccessful(), 'cases': result.testsRun,
        'scope': 'CPU AST and mock fixtures only; scene counters are fake, no GPU/SAPIEN/physics/episode success',
        'new_physics_steps': 0, 'gpu_runs': 0, 'source_sha256': {str((WORK / n).resolve()): collector.sha(WORK / n)
        for n in ('collector_settle_v3.py', 'worker_settle_v3.py', 'expert_v9_settle_v3.py', 'test_collector_settle_v3.py')}}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)