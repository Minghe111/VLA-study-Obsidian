"""CPU-only compatibility/counterexamples. No fixture is a real accepted episode."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v3_jobs_publisher_tests', HERE / 'publish_randomized5_patch_v3_jobs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def runtime_fixture(self, root):
        for slot in module.base.SLOTS:
            (root / 'jobs' / f'slot{slot}_runtime').mkdir(parents=True)

    def prepared_fixture(self, root):
        runtime = root / 'jobs/slot17_runtime'
        directory = runtime / 'new_physical_episodes/slot_0000017_randomized_attempt_004'
        directory.mkdir(parents=True)
        pins = {str(root / 'NOT-A-REAL-PRODUCER'): 'a' * 64}
        stable = {'version': 'pre_recording_clutter_settle_v1', 'passed': True, 'hardware_io': False,
            'recording_starts_after_preparation_at_true_new_t0': True, 'source_hashes_before': pins,
            'source_hashes_after': pins, 'bottle_after': {'initial_pose': [[1, 0], [0, 1]], 'actual_axis_yaw_deg': 0},
            'extra_random_draws': 0, 'wake_calls': 0, 'native_actor_robot_drive_velocity_geometry_material_writes': 0,
            'execution_vertex_drift_budget_m_unchanged': .001,
            'historical_original_extra_settle_receipt_preserved_exactly': True,
            'all_native_collision_component_identities_unchanged': True}
        module.base.write(directory / 'sample_spec.json', {'index': 17})
        module.base.write(directory / 'scene_sample.json', {**stable['bottle_after'], 'pre_recording_clutter_settle': stable})
        module.base.write(directory / 'pre_recording_clutter_settle_receipt.json', stable)
        module.base.write(directory / 'pre_recording_clutter_settle_initial.json', {'synthetic_CPU_fixture_only': True})
        (directory / 'pre_recording_clutter_settle_steps.jsonl').write_text('{"synthetic_CPU_fixture_only":true}\n')
        registry = dict(pins)
        for p in directory.glob('pre_recording_clutter_settle_*'):
            registry[str(p)] = module.base.sha(p)
        module.base.write(directory / 'expert_report.json', {'source_hashes': registry})
        return runtime, directory, pins

    def mutate(self, directory, name, change):
        data = module.base.read(directory / name)
        change(data)
        module.base.write(directory / name, data)

    def test_five_scattered_actual_runtime_areas_discovered_without_copying(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            root = Path(p)
            self.runtime_fixture(root)
            for slot in module.base.SLOTS:
                (root / 'jobs' / f'slot{slot}_runtime/new_physical_episodes/nested' / f'slot_{slot:07d}_randomized_attempt_004').mkdir(parents=True)
            rows = module.discover_candidates(root)
            self.assertEqual([slot for slot, _, _ in rows], list(module.base.SLOTS))
            self.assertTrue(all(path.is_relative_to(runtime) for _, runtime, path in rows))

    def test_candidate_symlink_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            root = Path(p)
            self.runtime_fixture(root)
            area = root / 'jobs/slot17_runtime/new_physical_episodes'
            area.mkdir()
            (area / 'slot_0000017_randomized_attempt_004').symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): module.discover_candidates(root)

    def test_runtime_symlink_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            root = Path(p)
            self.runtime_fixture(root)
            (root / 'jobs/slot17_runtime').rmdir()
            (root / 'jobs/slot17_runtime').symlink_to(root / 'jobs/slot18_runtime', target_is_directory=True)
            with self.assertRaises(ValueError): module.runtime_roots(root)

    def test_producer_component_fixture_does_not_claim_physics_acceptance(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            runtime, directory, pins = self.prepared_fixture(Path(p))
            result = module.prepared_producer_gate(directory, runtime, pins)
            self.assertTrue(result['passed'])
            self.assertNotIn('physical_success', result)

    def test_wrong_runtime_slot_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            runtime, directory, pins = self.prepared_fixture(Path(p))
            self.mutate(directory, 'sample_spec.json', lambda x: x.update(index=18))
            with self.assertRaises(ValueError): module.prepared_producer_gate(directory, runtime, pins)

    def test_registered_fixed_receipt_mutation_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            runtime, directory, pins = self.prepared_fixture(Path(p))
            (directory / 'pre_recording_clutter_settle_steps.jsonl').write_text('changed')
            with self.assertRaises(ValueError): module.prepared_producer_gate(directory, runtime, pins)

    def test_scene_t0_field_mutation_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            runtime, directory, pins = self.prepared_fixture(Path(p))
            self.mutate(directory, 'scene_sample.json', lambda x: x.update(actual_axis_yaw_deg=1))
            with self.assertRaises(ValueError): module.prepared_producer_gate(directory, runtime, pins)

    def test_missing_original_producer_pin_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            runtime, directory, pins = self.prepared_fixture(Path(p))
            self.mutate(directory, 'expert_report.json', lambda x: x['source_hashes'].pop(next(iter(pins))))
            with self.assertRaises(ValueError): module.prepared_producer_gate(directory, runtime, pins)

    def test_unapproved_collection_root_refused(self):
        with tempfile.TemporaryDirectory(prefix='v3_jobs_CPU_fixture_') as p:
            with self.assertRaises(ValueError): module.collection_gate(Path(p))

    def test_original_publisher_source_frozen(self):
        self.assertEqual(module.base.sha(HERE / 'publish_randomized5_patch.py'), module.BASE_SHA)

    def native_reader(self):
        if hasattr(self.__class__, '_native_fixture'):
            return self.__class__._native_fixture
        module.base.TASK_ROOT = module.COLLECTION
        module.base.SOURCE = module.COLLECTION / 'source'
        module.base.PACKAGE = module.base.SOURCE / 'rgb50_20261007_v1'
        module.base.LEGACY = module.base.SOURCE / 'adjust_bottle_20261002'
        bindings = []
        self.__class__._native_fixture = module.base.install_original_reader(bindings), bindings
        return self.__class__._native_fixture

    def actual_prepared_scene(self):
        return module.base.read(module.COLLECTION / 'jobs/slot17_runtime/new_physical_episodes/slot_0000017_randomized_attempt_004/scene_sample.json')

    def test_exact_float64_prepared_arithmetic_and_original_float32_branch(self):
        reader, bindings = self.native_reader()
        original = reader.original_readback.audit_trial.__globals__['pose_receipt']
        scene = self.actual_prepared_scene()
        with self.assertRaises(ValueError): original(scene)
        prepared = module.prepared_heading_reader(original, bindings)
        result = prepared(scene)
        self.assertEqual(result['prepared_native_float64_heading_deg'], scene['actual_axis_yaw_deg'])
        self.assertEqual(result['stored_heading_circular_error_deg'], 0)
        old_semantics = copy.deepcopy(scene)
        import numpy as np
        m = np.asarray(scene['initial_pose'], dtype=np.float32)
        old_semantics['actual_axis_yaw_deg'] = float(np.degrees(np.arctan2(m[1, 1], m[0, 1])))
        self.assertEqual(original(old_semantics)['stored_heading_circular_error_deg'], 0)

    def test_heading_tolerance_is_not_relaxed(self):
        reader, bindings = self.native_reader()
        original = reader.original_readback.audit_trial.__globals__['pose_receipt']
        prepared = module.prepared_heading_reader(original, bindings)
        scene = self.actual_prepared_scene()
        scene['actual_axis_yaw_deg'] += 1e-7
        with self.assertRaises(ValueError): prepared(scene)


if __name__ == '__main__':
    unittest.main(verbosity=2)
