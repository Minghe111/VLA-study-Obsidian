"""CPU fixture checks for stage boundaries; these are not physical episodes."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import collected_phase_seal as seal
import run_staged_collector as staged


class Joint:
    def __init__(self, name): self.name = name
    def get_name(self): return self.name


class FakeScene:
    def get_timestep(self): return .004000000189989805
    def step(self): raise AssertionError('Stage seal must not advance physics')


class SealFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='staged-seal-cpu-')
        self.out = Path(self.temp.name)
        self.names = ['native_joint_' + str(i) for i in range(36)]
        self.hands = {side: {'limits': {side + '_hand_' + str(i): {} for i in range(12)}}
                      for side in ('left', 'right')}
        self.events = []
        self.producer = self.out / 'registered_producer.py'
        self.producer.write_text('# CPU fixture only\n')
        self.pins = {str(self.producer): seal.sha(self.producer),
                     str(Path(seal.__file__).resolve()): seal.sha(seal.__file__)}
        self.task = types.SimpleNamespace(scene=FakeScene(), info={'fixture_only': True},
            save_dir=str(self.out), robot=types.SimpleNamespace(communication_flag=False,
                _manifest={'hands': self.hands}, left_entity=types.SimpleNamespace(
                    get_active_joints=lambda: [Joint(n) for n in self.names])))
        def save(idx):
            self.events.append('save_traj')
            self.assertEqual(idx, 0)
            (self.out / '_traj_data' / 'episode0.pkl').write_bytes(b'CPU fixture original saver output')
        self.task.save_traj_data = save
        self.frames = [0, 15, 30]
        self.samples = [{'physics_step': n, 'sim_time_s': n * self.task.scene.get_timestep(),
            'measured_state24': [0.] * 24, 'command24': [0.] * 24} for n in self.frames]
        cache = self.out / '.cache' / 'episode0'
        cache.mkdir(parents=True)
        # These bytes deliberately cannot be unpickled; sealing only hashes them.
        for i in range(3): (cache / (str(i) + '.pkl')).write_bytes(b'not a pickle ' + bytes([i]))
        for name in ('physics_trace.json', 'physics_telemetry.npz', 'run_config.yml',
            'scene_sample.json', 'randomized_scene_receipt.json',
            'actual_hardware_camera_clock_receipt.json',
            'pre_recording_clutter_settle_receipt.json', 'runtime_binding_receipt.json',
            'actual36_geometry_audit.json', 'actual36_geometry_audit_v13_original.json'):
            (self.out / name).write_text('{}\n')
        (self.out / 'sample_spec.json').write_text(json.dumps({'index': 34, 'seed': 41400034}))
        (self.out / 'recording_samples.json').write_text(json.dumps(self.samples))
        retained = self.out / 'expert_report_pre_actual_audit.json'
        retained.write_text('{"CPU_fixture_retained_original": true}\n')
        self.report = {'accepted': True, 'result': 'physics_success',
            'canonical_sources_unchanged': True, 'hold_passed': True,
            'hold_validation_s': 1., 'minimum_hold_bottle_gap_m': .02,
            'physical_steps': 35, 'recorded_frames': 3, 'full_qpos_joint_names': self.names,
            'source_hashes': self.pins,
            'actual36_geometry_audit_sha256': seal.sha(self.out / 'actual36_geometry_audit.json'),
            'actual36_geometry_audit': {'passed': True, 'physical_scene_unchanged': True,
                'retained_pre_actual_audit_report': {'original_bytes_unmodified': True,
                    'matches_original_report_sha256': True, 'sha256': seal.sha(retained)}}}

    def tearDown(self): self.temp.cleanup()

    def run_seal(self):
        def environment(task, out, report):
            self.assertIs(task, self.task)
            self.events.append('environment_before_close')
            (out / 'randomized_environment_execution.json').write_text('{"passed":true}\n')
            return {'passed': True}
        def hardware(task):
            self.assertIs(task, self.task)
            self.events.append('hardware_before_close')
            return {'passed': True, 'actual_link_count': 93,
                    'actual_physics_timestep_s': self.task.scene.get_timestep()}
        def branch(out, spec):
            self.events.append('branch')
            path = out / 'nearest_periodic_branch_receipt.json'
            path.write_text('{"CPU_fixture_only":true}\n')
            return path
        return seal.finalize_collected_phase(self.task, self.out, self.report, self.frames,
            finalize_environment=environment, read_hardware_contract=hardware,
            export_branch_receipt=branch, producer_pins=self.pins)

    def test_seal_never_decodes_RGB_or_accepts_dataset(self):
        path = Path(self.run_seal())
        data = json.loads(path.read_text())
        self.assertTrue(data['passed'])
        self.assertTrue(data['collected_physical'])
        self.assertFalse(data['accepted'])
        self.assertFalse(data['data_accepted'])
        self.assertFalse(data['normal_worker_exit_verified'])
        self.assertNotIn('recorded_actual36_geometry_v13_and_v14', data['pending_release_gates'])
        self.assertEqual(self.events, ['environment_before_close', 'hardware_before_close', 'save_traj', 'branch'])
        self.assertEqual(set(data['cache_sha256']), {'0.pkl', '1.pkl', '2.pkl'})
        self.assertEqual(data['inventory_snapshot_sha256'], seal.sha(self.out / 'inventory_snapshot.json'))
        recording = json.loads((self.out / 'recording_report.json').read_text())
        self.assertFalse(recording['accepted'])
        self.assertIsNone(recording['hdf5'])
        self.assertFalse((self.out / 'data').exists())
        self.assertFalse((self.out / 'video').exists())

    def test_missing_actual_geometry_rejects(self):
        self.report.pop('actual36_geometry_audit')
        with self.assertRaisesRegex(RuntimeError, 'actual36'): self.run_seal()

    def test_forged_geometry_file_identity_rejects(self):
        self.report['actual36_geometry_audit_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'actual36'): self.run_seal()

    def test_retained_original_report_changed_rejects(self):
        (self.out / 'expert_report_pre_actual_audit.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'retained'): self.run_seal()

    def test_unfinished_physical_hold_rejects(self):
        self.report['hold_validation_s'] = .996
        with self.assertRaisesRegex(RuntimeError, 'airborne'): self.run_seal()

    def test_physical_failed_report_rejects(self):
        self.report['accepted'] = False
        with self.assertRaisesRegex(RuntimeError, 'physical'): self.run_seal()

    def test_changed_source_rejects(self):
        self.producer.write_text('changed\n')
        with self.assertRaisesRegex(RuntimeError, 'source'): self.run_seal()

    def test_missing_registered_producer_rejects(self):
        self.report['source_hashes'] = {str(self.producer): seal.sha(self.producer)}
        with self.assertRaisesRegex(RuntimeError, 'register'): self.run_seal()

    def test_extraneous_cache_file_rejects(self):
        (self.out / '.cache' / 'episode0' / 'junk.pkl').write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError, 'extraneous'): self.run_seal()

    def test_missing_cache_file_rejects(self):
        (self.out / '.cache' / 'episode0' / '1.pkl').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing'): self.run_seal()

    def test_nonfinite_SDK_metadata_rejects(self):
        self.samples[1]['command24'][10] = float('nan')
        (self.out / 'recording_samples.json').write_text(json.dumps(self.samples))
        with self.assertRaisesRegex(RuntimeError, 'SDK24'): self.run_seal()

    def test_native_joint_order_mismatch_rejects(self):
        self.report['full_qpos_joint_names'] = list(reversed(self.names))
        with self.assertRaisesRegex(RuntimeError, 'joint order'): self.run_seal()

    def test_wrong_original_trajectory_target_rejects(self):
        self.task.save_dir = str(self.out / 'unrelated')
        with self.assertRaisesRegex(RuntimeError, 'another candidate'): self.run_seal()

    def test_capture_stride_and_endpoint_rejects(self):
        for frames, steps in (([0, 14, 30], 35), ([0, 15, 30], 45), ([0, 15, 30], 29)):
            with self.subTest(frames=frames, steps=steps):
                report = dict(self.report, physical_steps=steps)
                with self.assertRaises(RuntimeError):
                    seal.recording_clock(self.samples, frames, report, self.task.scene.get_timestep())


class WorkerAST(unittest.TestCase):
    def test_only_success_receipt_and_public_label_change(self):
        path = Path(staged.__file__).parent.parent / 'source/rgb50_20261007_v1/generation_worker_v2.py'
        source = ast.parse(path.read_text())
        tree, receipt = staged.staged_worker_tree(source)
        self.assertFalse(receipt['collection_success_is_dataset_acceptance'])
        changed = copy.deepcopy(tree)
        for node in ast.walk(changed):
            if isinstance(node, ast.Constant) and node.value == 'WORKER_COLLECTED_PENDING_VALIDATION':
                node.value = 'WORKER_ACCEPTED'
            if isinstance(node, ast.Dict):
                keys = [key.value if isinstance(key, ast.Constant) else None for key in node.keys]
                if 'collected_physical' in keys:
                    for key in ('collected_physical', 'published_accepted', 'collection_artifact'):
                        index = keys.index(key); del node.keys[index]; del node.values[index]; del keys[index]
                    node.values[keys.index('accepted')] = ast.Constant(value=True)
        self.assertEqual(ast.dump(source, include_attributes=False), ast.dump(changed, include_attributes=False))

    def test_changed_original_success_fields_fail_closed(self):
        path = Path(staged.__file__).parent.parent / 'source/rgb50_20261007_v1/generation_worker_v2.py'
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict) and any(isinstance(key, ast.Constant) and key.value == 'physics_steps' for key in node.keys):
                node.keys.append(ast.Constant(value='extra')); node.values.append(ast.Constant(value=True))
        with self.assertRaisesRegex(RuntimeError, 'fields changed'): staged.staged_worker_tree(tree)

    def test_collector_retains_original_complete_geometry_default(self):
        tree = ast.parse(Path(staged.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(key.arg == 'validate_actual_geometry' and isinstance(key.value, ast.Constant)
            and key.value.value is False for node in calls for key in node.keywords))
        exporters = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'export']
        self.assertEqual(len(exporters), 1)
        self.assertEqual(len(exporters[0].body), 1)
        self.assertEqual(ast.unparse(exporters[0].body[0].value.func), 'finalize_collected_phase')


if __name__ == '__main__':
    unittest.main(verbosity=2)
