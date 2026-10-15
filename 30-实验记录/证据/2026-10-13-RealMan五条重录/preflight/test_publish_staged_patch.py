"""Small CPU admission/copy tests; never publish an actual physical dataset."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('publish_staged_patch', HERE / 'publish_staged_patch.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.evidence = self.root / 'evidence'
        self.preflight = self.evidence / 'preflight'
        self.preflight.mkdir(parents=True)
        self.v5 = self.root / 'v5'
        self.v5.mkdir()
        self.guard = self.preflight / 'run_guarded_stage.py'
        self.guard.write_text('# Explicit synthetic producer fixture\n')
        self.entry = self.preflight / 'streaming_export_staged.py'
        self.entry.write_text('# Explicit synthetic export fixture\n')
        self.candidate = self.v5 / 'jobs/slot17_runtime/new_physical_episodes/slot_0000017_randomized_attempt_001'
        self.candidate.mkdir(parents=True)
        self.output = self.evidence / 'export17'
        self.output.mkdir()
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'byte-exact fixture\x00\xff\x18')
        self.rows = [{'slot': slot, 'candidate': str(self.v5 / f'jobs/slot{slot}_runtime/new_physical_episodes/c'),
            'export_output': str(self.evidence / f'x{slot}'), 'quality_output': str(self.evidence / f'q{slot}'),
            'render_output': str(self.evidence / f'r{slot}'), 'resource_receipts': {}} for slot in publisher.SLOTS]
        self.scope = mock.patch.multiple(publisher, E=self.evidence, HERE=self.preflight, V5=self.v5,
            TOOLS={**publisher.TOOLS, 'export': self.entry})
        self.scope.start()

    def tearDown(self):
        self.scope.stop()
        self.tmp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def resource(self):
        path = self.preflight / 'resource_runs/export_1/receipt.json'
        worker = {'stage_returncode': 0, 'source_unchanged': True,
            'resource': {'memory.events': {key: 0 for key in ('max', 'oom', 'oom_kill', 'oom_group_kill')}}}
        receipt = {'schema': 'realman_guarded_stage_v1', 'stage': 'export', 'passed': True,
            'check_only': False, 'launcher_returncode': 0, 'monitor_stop_reason': None,
            'cgroup_readback_verified': True,
            'command': ['/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python', str(self.entry),
                '--candidate', str(self.candidate), '--output', str(self.output)],
            'source_binding': {'entry_sha256': publisher.sha(self.entry), 'guard_sha256': publisher.sha(self.guard)},
            'limits': {'MemoryHigh': str(3 * publisher.GIB), 'MemoryMax': str(4 * publisher.GIB),
                'MemorySwapMax': '0', 'CPUQuota': '200%', 'KillMode': 'control-group', 'OOMPolicy': 'stop'},
            'worker_terminal': worker, 'unit_terminal': {'Result': 'success', 'ExecMainStatus': '0', 'ActiveState': 'inactive'}}
        self.write(path, receipt)
        self.write(path.parent / 'worker_terminal.json', worker)
        self.write(path.parent / 'cgroup_readback.json', {'limits_verified': True,
            'actual': {'memory.high': str(3 * publisher.GIB), 'memory.max': str(4 * publisher.GIB),
                'memory.swap.max': '0', 'cpu.max': '200000 100000', 'pids.max': '128'}})
        return path, receipt

    def check_resource(self, path):
        return publisher.resource_proof(path, 'export', self.candidate, {'export': self.output}, {'index': 17})

    def make_hdf(self, dim=24, rgb=(480, 640, 3)):
        path = self.root / 'tiny.hdf5'
        with h5py.File(path, 'w') as f:
            f.attrs['source_format'] = 'RoboTwin'
            f.attrs['recording_clock'] = '60ms observations,4ms physical steps'
            f.attrs['physical_success_report_sha256'] = 'a' * 64
            for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                camera = f.require_group('vision/' + name)
                camera.create_dataset('colors', shape=(3,), dtype='S10')
                camera.create_dataset('shape', data=np.asarray(rgb, np.int32))
            for name in ('measured_state24', 'next_command24'):
                f.create_dataset('realman_dexterous/' + name, shape=(3, dim), dtype=np.float64)
            for name in ('sample_time_s', 'next_sample_time_s'):
                f.create_dataset('realman_dexterous/' + name, shape=(3,), dtype=np.float64)
            f.create_dataset('realman_dexterous_physics/physics_step', shape=(45,), dtype=np.int64)
            f.create_dataset('realman_dexterous_physics/full36_qpos_after_rad', shape=(45, 36), dtype=np.float32)
            f.create_dataset('realman_dexterous_physics/hand24_qpos_after_rad', shape=(45, 24), dtype=np.float32)
        return path

    def test_requires_exact_five_slots(self):
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            publisher.request_rows({'schema': 'staged-five-release-request-v1', 'candidates': self.rows[:-1]})

    def early_admission(self, **override):
        quality, render = self.evidence / 'quality17', self.evidence / 'render17'
        quality.mkdir()
        render.mkdir()
        row = {'slot': 17, 'candidate': str(self.candidate), 'export_output': str(self.output),
            'quality_output': str(quality), 'render_output': str(render), 'resource_receipts': {}}
        old = {'seed': 1400017, 'model_id': 4}
        self.write(self.candidate / 'sample_spec.json', {'index': 17, 'source_slot_index': 17,
            'brand': 'Fanta', 'arm': 'left', 'pose_class': 'upright', 'source_seed': 1400017, 'model_id': 4})
        receipt = {'schema': 'staged-independent-quality-v1', 'passed': True, 'gates_passed': True,
            'component_only': False, 'accepted': False, 'data_accepted': False,
            'native_dataset_readback_passed': True, 'independent_motion_quality_passed': True,
            'candidate': str(self.candidate), 'export_output': str(self.output)}
        receipt.update(override)
        self.write(quality / 'quality_receipt.json', receipt)
        return row, old

    def test_component_quality_can_never_publish(self):
        row, old = self.early_admission(component_only=True)
        with self.assertRaisesRegex(ValueError, 'quality stage failed or is a component'):
            publisher.verify_candidate(row, old)

    def test_partial_quality_gate_can_never_publish(self):
        row, old = self.early_admission(gates_passed=False)
        with self.assertRaisesRegex(ValueError, 'quality stage failed or is a component'):
            publisher.verify_candidate(row, old)

    def test_quality_failure_can_never_publish(self):
        row, old = self.early_admission(passed=False)
        with self.assertRaisesRegex(ValueError, 'quality stage failed or is a component'):
            publisher.verify_candidate(row, old)

    def test_wrong_quality_capture_identity_refused(self):
        row, old = self.early_admission(candidate=str(self.root / 'another-capture'))
        with self.assertRaisesRegex(ValueError, 'quality stage failed or is a component'):
            publisher.verify_candidate(row, old)

    def test_duplicate_slot_refused(self):
        self.rows[-1]['slot'] = 17
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            publisher.request_rows({'schema': 'staged-five-release-request-v1', 'candidates': self.rows})

    def test_duplicate_physical_candidate_refused(self):
        self.rows[-1]['candidate'] = self.rows[0]['candidate']
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'):
            publisher.request_rows({'schema': 'staged-five-release-request-v1', 'candidates': self.rows})

    def test_known_slot_order_is_preserved(self):
        value = {'schema': 'staged-five-release-request-v1', 'candidates': self.rows[::-1]}
        self.assertEqual([r['slot'] for r in publisher.request_rows(value)], list(publisher.SLOTS))

    def test_byte_exact_copy_and_no_overwrite(self):
        path = self.root / 'copy.bin'
        publisher.copy_exact(self.source, path, publisher.sha(self.source), self.source.stat().st_size)
        self.assertEqual(path.read_bytes(), self.source.read_bytes())
        with self.assertRaises(FileExistsError):
            publisher.copy_exact(self.source, path, publisher.sha(self.source), self.source.stat().st_size)

    def test_changed_source_refused_before_copy(self):
        digest = publisher.sha(self.source)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA/size changed'):
            publisher.copy_exact(self.source, self.root / 'copy.bin', digest, 19)
        self.assertFalse((self.root / 'copy.bin').exists())

    def test_native_header_no_full_episode_arrays_loaded(self):
        result = publisher.header_contract(self.make_hdf(), 3, 45, 'a' * 64)
        self.assertTrue(result['passed'])
        self.assertFalse(result['numeric_or_RGB_episode_arrays_loaded'])

    def test_bad_sdk_dimension_refused(self):
        with self.assertRaisesRegex(ValueError, 'SDK24'):
            publisher.header_contract(self.make_hdf(dim=14), 3, 45, 'a' * 64)

    def test_bad_camera_resolution_refused(self):
        with self.assertRaisesRegex(ValueError, 'resolutions'):
            publisher.header_contract(self.make_hdf(rgb=(240, 320, 3)), 3, 45, 'a' * 64)

    def test_all_successful_actual_kernel_resource_proofs_required(self):
        path, _ = self.resource()
        self.assertTrue(self.check_resource(path)['passed'])

    def test_component_resource_command_refused(self):
        path, receipt = self.resource()
        receipt['command'].append('--benchmark-only')
        self.write(path, receipt)
        with self.assertRaisesRegex(ValueError, 'component benchmark'):
            self.check_resource(path)

    def test_failed_stage_refused(self):
        path, receipt = self.resource()
        receipt['passed'] = False
        self.write(path, receipt)
        with self.assertRaisesRegex(ValueError, 'incomplete, failed'):
            self.check_resource(path)

    def test_oom_and_limit_violation_refused(self):
        path, receipt = self.resource()
        receipt['worker_terminal']['resource']['memory.events']['max'] = 1
        self.write(path, receipt)
        self.write(path.parent / 'worker_terminal.json', receipt['worker_terminal'])
        with self.assertRaisesRegex(ValueError, 'hard memory cap'):
            self.check_resource(path)

    def test_actual_kernel_limit_mismatch_refused(self):
        path, _ = self.resource()
        readback = json.loads((path.parent / 'cgroup_readback.json').read_text())
        readback['actual']['memory.max'] = str(12 * publisher.GIB)
        self.write(path.parent / 'cgroup_readback.json', readback)
        with self.assertRaisesRegex(ValueError, 'Actual kernel'):
            self.check_resource(path)

    def test_other_capture_resource_receipt_refused(self):
        path, receipt = self.resource()
        receipt['command'][3] = str(self.root / 'another-candidate')
        self.write(path, receipt)
        with self.assertRaisesRegex(ValueError, 'different candidate'):
            self.check_resource(path)

    def test_changed_resource_guard_refused(self):
        path, _ = self.resource()
        self.guard.write_text('# changed guard\n')
        with self.assertRaisesRegex(ValueError, 'producer changed'):
            self.check_resource(path)

    def test_actual_worker_receipt_mismatch_refused(self):
        path, _ = self.resource()
        self.write(path.parent / 'worker_terminal.json', {'stage_returncode': 1})
        with self.assertRaisesRegex(ValueError, 'Embedded worker'):
            self.check_resource(path)

    def test_no_original_dataset_or_source_destination(self):
        request = self.root / 'request.json'
        self.write(request, {'schema': 'staged-five-release-request-v1', 'candidates': self.rows})
        source = self.evidence / 'source'
        source.mkdir()
        with self.assertRaisesRegex(ValueError, 'dedicated patch destination'):
            publisher.publish_request(request, source / 'newpatch', publish=True)

    def test_small_atomic_copy_unit_with_mock_admission_preserves_original(self):
        # File packaging only. Admission is deliberately mocked; temporary
        # synthetic artifacts are destroyed and never delivered as real data.
        original = self.root / 'original50.json'
        self.write(original, {'complete': True, 'count': 50, 'episodes': [{'index': i} for i in range(50)]})
        digest = publisher.sha(original)
        request = self.root / 'request.json'
        self.write(request, {'schema': 'staged-five-release-request-v1', 'candidates': self.rows})
        def admitted(row, old):
            return {'slot': row['slot'], 'synthetic_unit_fixture_only': True,
                'verified_raw_and_source_file_stat_before_copy': {},
                'copy_files': [{'source': str(self.source), 'destination_relative': f'data/episode_{row["slot"]:07d}.hdf5',
                    'sha256': publisher.sha(self.source), 'bytes': self.source.stat().st_size}]}
        with mock.patch.multiple(publisher, ORIGINAL50=original, ORIGINAL50_SHA=digest), \
             mock.patch.object(publisher, 'verify_candidate', admitted):
            preview = publisher.publish_request(request, self.evidence / 'patch', publish=False)
            self.assertFalse(preview['complete'])
            self.assertFalse((self.evidence / 'patch').exists())
            result = publisher.publish_request(request, self.evidence / 'patch', publish=True)
        self.assertTrue(result['complete'])
        self.assertFalse(result['full50_randomized'])
        self.assertEqual(publisher.sha(original), digest)
        self.assertEqual(len(list((self.evidence / 'patch/data').iterdir())), 5)
        self.assertFalse(list(self.evidence.glob('patch.partial.*')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
