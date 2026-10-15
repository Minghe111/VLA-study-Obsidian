"""CPU metadata/counterexample fixtures, never physical demonstration evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

import h5py
import staged_quality as quality


class ResourceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='staged-quality-resource-cpu-')
        self.root = Path(self.temp.name)
        self.entry = self.root / 'preflight/run_staged_collector.py'
        self.entry.parent.mkdir()
        self.entry.write_text("parser.add_argument('--output', default=TASK_ROOT / 'new_physical_episodes')\n")
        self.output = self.root / 'new_physical_episodes'
        self.candidate = self.output / 'slot_0000034_randomized_attempt_005'
        self.candidate.mkdir(parents=True)
        self.spec = {'index': 34, 'rerecord_attempt': 5, 'source_seed': 1400034, 'seed': 41400034}
        self.command = ['/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python', str(self.entry),
            '--index', '34', '--attempt', '5', '--domain-seed', '41400034', '--run']
        self.receipt = {'schema': 'realman_guarded_stage_v1', 'stage': 'collect', 'passed': True,
            'check_only': False, 'cgroup_readback_verified': True, 'launcher_returncode': 0,
            'monitor_stop_reason': None, 'command': self.command,
            'source_binding': {'entry_sha256': quality.sha(self.entry),
                               'guard_sha256': quality.sha(quality.HERE / 'run_guarded_stage.py')},
            'limits': {'MemoryMax': str(12 * 1024**3), 'MemoryHigh': str(10 * 1024**3),
                'MemorySwapMax': '0', 'CPUQuota': '200%', 'KillMode': 'control-group', 'OOMPolicy': 'stop'},
            'worker_terminal': {'stage_returncode': 0, 'source_unchanged': True,
                'resource': {'memory.events': {'max': 0, 'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0}}},
            'unit_terminal': {'Result': 'success', 'ExecMainStatus': '0', 'ActiveState': 'inactive'}}
        self.path = self.root / 'resource_runs/collect_fixture/receipt.json'
        self.path.parent.mkdir(parents=True)

    def tearDown(self): self.temp.cleanup()

    def save(self): self.path.write_text(json.dumps(self.receipt))
    def check(self):
        self.save()
        return quality.validate_collect_resource_receipt(self.path, self.candidate, self.spec)

    def test_exact_source_parser_default_output_allowed(self):
        self.assertTrue(self.check()['passed'])
        self.assertEqual(quality.collected_output(self.command), self.output)

    def test_explicit_output_allowed(self):
        self.command += ['--output', str(self.output)]
        self.assertTrue(self.check()['passed'])

    def test_another_candidate_output_rejects(self):
        self.command += ['--output', str(self.root / 'another')]
        with self.assertRaises(ValueError): self.check()

    def test_nonunique_or_unreviewed_parser_default_rejects(self):
        for text in ("parser.add_argument('--output',default=Path('/unrelated'))\n",
            "parser.add_argument('--output',default=TASK_ROOT/'new_physical_episodes')\n" * 2):
            with self.subTest(text=text):
                self.entry.write_text(text)
                with self.assertRaises(ValueError): quality.collected_output(self.command)

    def test_unfinished_external_worker_rejects(self):
        self.receipt['worker_terminal']['stage_returncode'] = 9
        with self.assertRaisesRegex(ValueError, 'worker'): self.check()

    def test_active_unit_rejects(self):
        self.receipt['unit_terminal']['ActiveState'] = 'active'
        with self.assertRaisesRegex(ValueError, 'cleanup'): self.check()

    def test_resource_hard_cap_or_oom_failure_rejects(self):
        for key in ('max', 'oom', 'oom_kill', 'oom_group_kill'):
            with self.subTest(key=key):
                self.receipt['worker_terminal']['resource']['memory.events'][key] = 1
                with self.assertRaisesRegex(ValueError, 'memory'): self.check()
                self.receipt['worker_terminal']['resource']['memory.events'][key] = 0

    def test_missing_cgroup_verification_rejects(self):
        self.receipt['cgroup_readback_verified'] = False
        with self.assertRaisesRegex(ValueError, 'bounded'): self.check()

    def test_changed_entry_or_guard_rejects(self):
        for name in ('entry_sha256', 'guard_sha256'):
            old = self.receipt['source_binding'][name]
            self.receipt['source_binding'][name] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'changed'): self.check()
            self.receipt['source_binding'][name] = old

    def test_wrong_caps_or_seed_rejects(self):
        self.receipt['limits']['MemoryMax'] = str(24 * 1024**3)
        with self.assertRaisesRegex(ValueError, 'budget'): self.check()
        self.receipt['limits']['MemoryMax'] = str(12 * 1024**3)
        self.spec['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'seed'): self.check()

    def test_exactly_one_external_success_required(self):
        self.save()
        self.assertTrue(quality.collection_exit_proof(self.candidate, self.spec, self.path.parent.parent)['passed'])
        other = self.path.parent.parent / 'collect_duplicate/receipt.json'
        other.parent.mkdir(); other.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            quality.collection_exit_proof(self.candidate, self.spec, self.path.parent.parent)


class BoundedIO(unittest.TestCase):
    def test_all_stage_directories_must_be_disjoint_in_both_directions(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-paths-cpu-') as temp:
            root = Path(temp)
            siblings = [root / name for name in ('candidate', 'export', 'quality')]
            self.assertEqual(quality.separate_stage_paths(*siblings), siblings)
            for first in range(3):
                for second in range(3):
                    if first == second: continue
                    with self.subTest(parent=first, child=second):
                        overlap = list(siblings); overlap[second] = overlap[first] / 'nested'
                        with self.assertRaisesRegex(ValueError, 'either direction'):
                            quality.separate_stage_paths(*overlap)
            with self.assertRaisesRegex(ValueError, 'either direction'):
                quality.separate_stage_paths(siblings[0], siblings[0], siblings[2])

    def test_numeric_budget_refuses_large_virtual_column_without_loading_it(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-budget-cpu-') as temp:
            path = Path(temp) / 'sparse_metadata_only.hdf5'
            with h5py.File(path, 'w') as handle:
                handle.create_dataset('state/left_arm_joint_states', shape=(1, 6), dtype='f4')
                handle.create_dataset('huge_virtual_column', shape=(100_000_000,), dtype='f8')
            self.assertLess(path.stat().st_size, 65536)
            with self.assertRaisesRegex(ValueError, 'budget'): quality.numeric_budget(path)

    def test_exact_alias_preserves_bytes_and_refuses_another_file(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-alias-cpu-') as temp:
            root = Path(temp); source = root / 'source'; target = root / 'candidate/data/episode.hdf5'
            source.write_bytes(b'CPU fixture artifact')
            digest = quality.sha(source)
            self.assertEqual(quality.copy_exact(source, target, digest)['sha256'], digest)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            target.write_bytes(b'other')
            with self.assertRaisesRegex(ValueError, 'overwrite'): quality.copy_exact(source, target, digest)

    def test_existing_different_quality_report_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-put-cpu-') as temp:
            path = Path(temp) / 'proof.json'
            quality.put_json(path, {'passed': False})
            quality.put_json(path, {'passed': False})
            with self.assertRaisesRegex(ValueError, 'overwrite'): quality.put_json(path, {'passed': True})


class ComponentBoundary(unittest.TestCase):
    def test_quality_receipt_binds_output_motion_copy_and_keeps_acceptance_pending(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-motion-receipt-cpu-') as temp:
            root = Path(temp); candidate = root / 'new_physical_episodes/slot_fixture'
            exported = root / 'export'; out = root / 'quality_result'
            candidate.mkdir(parents=True); (exported / 'data').mkdir(parents=True)
            (exported / 'instruction').mkdir()
            report = candidate / 'expert_report.json'; report.write_text('{"CPU_fixture_only":true}\n')
            (candidate / 'recording_report.json').write_text('{"accepted":false}\n')
            (candidate / 'sample_spec.json').write_text(json.dumps({'index': 34, 'arm': 'left',
                'pose_class': 'upright', 'brand': 'Coca-Cola', 'seed': 41400034}))
            (candidate / 'nearest_periodic_branch_receipt.json').write_text('{}\n')
            hdf = exported / 'data/episode_0000000.hdf5'
            with h5py.File(hdf, 'w') as handle:
                handle.create_dataset('state/left_arm_joint_states', shape=(1, 6), dtype='f4')
            instruction = exported / 'instruction/episode_0000000.json'
            instruction.write_text('{"instructions":["CPU fixture"]}\n')
            receipt = {'export_complete': True, 'accepted': False, 'component_only': False,
                'candidate': str(candidate), 'output': str(exported), 'mode': 'hdf_only',
                'protected_raw_inputs_sha256': {str(report): quality.sha(report)},
                'outputs': {'hdf5': {'path': str(hdf), 'sha256': quality.sha(hdf)},
                    'instruction': {'path': str(instruction), 'sha256': quality.sha(instruction)}}}
            (exported / 'export_receipt.json').write_text(json.dumps(receipt))
            relative = 'source/baseline_rgb20/rgb20_final_20261004/validate_rgb_three_camera.py'
            (root / relative).parent.mkdir(parents=True)
            shutil.copyfile(quality.TASK / relative, root / relative)
            (root / 'quality').mkdir()
            for name in ('motion_gate.py', 'thresholds_confirmed.json'):
                shutil.copyfile(quality.TASK / 'quality' / name, root / 'quality' / name)
            motion_result = {'overall_publish_gate_passed': True, 'CPU_fixture_only': True,
                'numeric_source': str(candidate / 'data/episode_0000000.hdf5'),
                'numeric_source_sha256': quality.sha(hdf)}
            fake_reader = types.SimpleNamespace(validate=lambda *args: {'passed': True, 'CPU_fixture_only': True})
            fake_motion = types.SimpleNamespace(evaluate_candidate=lambda *args, **kwargs: motion_result)
            def fake_module(name, path):
                return fake_reader if name == '_staged_original_rgb_validator' else fake_motion
            before = (candidate / 'recording_report.json').read_bytes()
            with mock.patch.object(quality, 'load_module', side_effect=fake_module), \
                 mock.patch.object(quality.exporter.base, 'original_physics_admission', return_value={}), \
                 mock.patch.object(quality, 'collection_exit_proof', return_value={'passed': True, 'CPU_fixture_only': True}), \
                 mock.patch.object(quality, 'upstream_proofs', return_value={'CPU_fixture_only': True}):
                result = quality.validate_quality(candidate, exported, out)
            self.assertTrue(result['passed'], result.get('error'))
            self.assertFalse(result['accepted'])
            self.assertTrue(result['pending_native_video_full_decode'])
            self.assertEqual(result['independent_motion_quality_path'], str(out / 'independent_motion_quality.json'))
            self.assertEqual(result['independent_motion_quality_sha256'], quality.sha(out / 'independent_motion_quality.json'))
            self.assertEqual((out / 'independent_motion_quality.json').read_bytes(),
                             (candidate / 'independent_motion_quality.json').read_bytes())
            self.assertEqual(before, (candidate / 'recording_report.json').read_bytes())

    def test_benchmark_never_aliases_or_accepts_interrupted_candidate(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-component-cpu-') as temp:
            root = Path(temp); candidate = root / 'interrupted'; exported = root / 'export'; out = root / 'quality'
            candidate.mkdir(); (exported / 'data').mkdir(parents=True)
            report = candidate / 'expert_report.json'; report.write_text('{"accepted":false}\n')
            (candidate / 'sample_spec.json').write_text('{"index":34}\n')
            hdf = exported / 'data/episode_0000000.hdf5'
            with h5py.File(hdf, 'w') as handle:
                handle.create_dataset('state/left_arm_joint_states', shape=(1, 6), dtype='f4')
            receipt = {'export_complete': True, 'accepted': False, 'component_only': True,
                'candidate': str(candidate), 'output': str(exported), 'mode': 'hdf_only',
                'protected_raw_inputs_sha256': {str(report): quality.sha(report)},
                'outputs': {'hdf5': {'path': str(hdf), 'sha256': quality.sha(hdf)}}}
            (exported / 'export_receipt.json').write_text(json.dumps(receipt))
            before = {path.name: path.read_bytes() for path in candidate.iterdir()}
            fake_reader = types.SimpleNamespace(validate=lambda *args: {'passed': True,
                'CPU_fixture_only': True, 'physical_success_not_inferred_from_images': True})
            with mock.patch.object(quality, 'load_module', return_value=fake_reader), \
                 mock.patch.object(quality.exporter.base, 'original_physics_admission',
                                   side_effect=AssertionError('No physical certification in benchmark')):
                result = quality.validate_quality(candidate, exported, out, benchmark_only=True)
            self.assertTrue(result['passed'])
            self.assertTrue(result['component_only'])
            self.assertFalse(result['accepted'])
            self.assertFalse(result['data_accepted'])
            self.assertFalse(result['gates_passed'])
            self.assertTrue(result['pending_native_video_full_decode'])
            self.assertIsNone(result['independent_motion_quality_path'])
            self.assertIsNone(result['independent_motion_quality_sha256'])
            self.assertEqual(before, {path.name: path.read_bytes() for path in candidate.iterdir()})

    def test_component_export_cannot_enter_strict_quality_stage(self):
        with tempfile.TemporaryDirectory(prefix='staged-quality-scope-cpu-') as temp:
            root = Path(temp); candidate = root / 'candidate'; exported = root / 'export'; out = root / 'quality'
            candidate.mkdir(); exported.mkdir()
            (exported / 'export_receipt.json').write_text(json.dumps({'export_complete': True,
                'accepted': False, 'component_only': True, 'candidate': str(candidate),
                'output': str(exported), 'mode': 'hdf_only'}))
            result = quality.validate_quality(candidate, exported, out)
            self.assertFalse(result['passed'])
            self.assertFalse(result['accepted'])
            self.assertIn('scope', result['error'])


class PublisherDispatch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='staged-publish-dispatch-cpu-')
        self.root = Path(self.temp.name)
        self.helper = self.root / 'publish_staged_patch.py'
        self.helper.write_text('# CPU fixture; never imported\n')
        self.digest = quality.sha(self.helper)
        self.result = {'complete': True, 'accepted': True, 'count': 5,
            'full50_randomized': False, 'original50_replaced_or_merged': False}

    def tearDown(self): self.temp.cleanup()

    def test_dispatch_calls_only_pinned_publisher_and_exact_publish_API(self):
        callback = mock.Mock(return_value=self.result)
        publisher = types.SimpleNamespace(publish_request=callback)
        with mock.patch.object(quality, 'HERE', self.root), \
             mock.patch.object(quality, 'PUBLISH_HELPER_SHA256', self.digest), \
             mock.patch.object(quality, 'load_module', return_value=publisher):
            self.assertEqual(quality.dispatch_publish_five(self.root / 'request.json', self.root / 'output'), self.result)
        callback.assert_called_once_with(self.root / 'request.json', self.root / 'output', publish=True)
        self.assertFalse((self.root / 'output').exists())  # No actual file publication in this fixture.

    def test_missing_or_wrong_pin_refuses_import(self):
        for digest in (None, '0' * 64):
            with self.subTest(digest=digest), mock.patch.object(quality, 'HERE', self.root), \
                 mock.patch.object(quality, 'PUBLISH_HELPER_SHA256', digest), \
                 mock.patch.object(quality, 'load_module') as importer:
                with self.assertRaisesRegex(ValueError, 'refuse import'):
                    quality.dispatch_publish_five(self.root / 'request', self.root / 'output')
                importer.assert_not_called()

    def test_publisher_mutation_during_dispatch_rejects(self):
        def changed(*args, **kwargs):
            self.helper.write_text('# changed producer\n')
            return self.result
        with mock.patch.object(quality, 'HERE', self.root), \
             mock.patch.object(quality, 'PUBLISH_HELPER_SHA256', self.digest), \
             mock.patch.object(quality, 'load_module', return_value=types.SimpleNamespace(publish_request=changed)):
            with self.assertRaisesRegex(ValueError, 'changed during dispatch'):
                quality.dispatch_publish_five(self.root / 'request', self.root / 'output')

    def test_incomplete_or_full50_replacement_contract_rejects(self):
        for key, value in (('complete', False), ('accepted', False), ('count', 4),
                           ('full50_randomized', True), ('original50_replaced_or_merged', True)):
            with self.subTest(key=key):
                wrong = dict(self.result, **{key: value})
                with mock.patch.object(quality, 'HERE', self.root), \
                     mock.patch.object(quality, 'PUBLISH_HELPER_SHA256', self.digest), \
                     mock.patch.object(quality, 'load_module', return_value=types.SimpleNamespace(
                        publish_request=lambda *args, **kwargs: wrong)):
                    with self.assertRaisesRegex(ValueError, 'patch contract'):
                        quality.dispatch_publish_five(self.root / 'request', self.root / 'output')


if __name__ == '__main__': unittest.main(verbosity=2)
