"""CPU counterexamples for patch admission; these fixtures are NOT real episodes.

Does not run original physical admission or publish any dataset. Adversarial
metadata tests ensure no partial/wrong-slot/failed candidate becomes a patch.
"""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('publish5_guard_tests', HERE / 'publish_randomized5_patch.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class Tests(unittest.TestCase):
    def setUp(self):
        self.baseline = publisher.original50(publisher.SOURCE50)
        self.sample = {'index': 17, 'brand': 'Fanta', 'arm': 'left', 'pose_class': 'upright',
            'source_slot_index': 17, 'source_seed': self.baseline[17]['seed'], 'seed': self.baseline[17]['seed'] + 10_000_000,
            'actual_domain_seed': self.baseline[17]['seed'] + 10_000_000, 'rerecord_attempt': 1,
            'domain_seed_policy': 'source_seed + 10000000 * attempt'}
        self.rows = [{'original_slot_index': index, 'passed': True, 'attempt': 1,
            'actual_domain_seed': 11_400_000 + index, 'validated_source_candidate': '/NOT-A-REAL-EPISODE/slot' + str(index)}
            for index in publisher.SLOTS]

    def test_exact_original_slot_provenance(self):
        self.assertEqual(publisher.slot_gate(self.sample, self.baseline), 17)

    def test_wrong_source_seed(self):
        self.sample['source_seed'] += 1
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_wrong_arm(self):
        self.sample['arm'] = 'right'
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_wrong_brand(self):
        self.sample['brand'] = 'Sprite'
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_wrong_pose(self):
        self.sample['pose_class'] = 'horizontal'
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_outside_five_scope(self):
        self.sample['index'] = 31
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_default_seed_requires_fresh_attempt_seed(self):
        self.sample['seed'] = self.sample['source_seed']
        self.sample['actual_domain_seed'] = self.sample['seed']
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_explicit_seed_keeps_old_provenance(self):
        self.sample.update(domain_seed_policy='explicit --domain-seed', seed=2718281, actual_domain_seed=2718281)
        self.assertEqual(publisher.slot_gate(self.sample, self.baseline), 17)

    def test_bool_is_not_actual_seed(self):
        self.sample.update(domain_seed_policy='explicit --domain-seed', seed=True, actual_domain_seed=True)
        with self.assertRaises(ValueError): publisher.slot_gate(self.sample, self.baseline)

    def test_partial_four_are_not_a_complete_five_patch(self):
        self.assertEqual(publisher.choose_five(self.rows[:-1]), [])

    def test_failed_fifth_is_not_selected(self):
        self.rows[-1]['passed'] = False
        self.assertEqual(publisher.choose_five(self.rows), [])

    def test_duplicate_random_seeds_rejected(self):
        self.rows[-1]['actual_domain_seed'] = self.rows[0]['actual_domain_seed']
        with self.assertRaises(ValueError): publisher.choose_five(self.rows)

    def test_selection_retains_original_slot_numbers(self):
        self.assertEqual([row['original_slot_index'] for row in publisher.choose_five(list(reversed(self.rows)))], list(publisher.SLOTS))

    def test_source_constant_relocation_leaves_recorded_proof_path_untouched(self):
        import ast
        statement = "ROOT='/bigdata2/liminghe/VLA-benchmark/RoboTwin'\nproof='/actual/new_physical_episodes/slot17/physics_trace.json'\n"
        transform = publisher.BindSourcePaths()
        tree = transform.visit(ast.parse(statement))
        values = [row.value.value for row in tree.body]
        self.assertEqual(values, [str(publisher.REPOSITORY), '/actual/new_physical_episodes/slot17/physics_trace.json'])
        self.assertEqual(len(transform.sites), 1)

    def test_unvalidated_rows_cannot_start_staging_even_if_all_indices_present(self):
        # No publisher staging exists or real data read; every row lacks the
        # independent source physics/motion audits required by the entry gate.
        with self.assertRaises(ValueError):
            publisher.publish_patch(self.rows, publisher.TASK_ROOT / 'published/SHOULD_NOT_BE_CREATED', {}, publisher.SOURCE50, None)
        self.assertFalse((publisher.TASK_ROOT / 'published/SHOULD_NOT_BE_CREATED').exists())

    def random_fixture(self, directory):
        spec = self.sample
        scene = {'passed': True, 'hardware_io': False, 'official_domain_randomization': publisher.DR,
            'actual_domain_randomization': publisher.DR, 'source_slot_index': 17, 'source_seed': spec['source_seed'],
            'actual_seed': spec['seed'], 'original_source_pose_restored': False,
            'objects_settled_by_original_gravity_and_stability_chain': True, 'table_z_bias_m': -.015,
            'clutter_count': 0, 'official_clutter_records': [], 'clutter_sampling_outcome': 'native_clean_background_draw'}
        coverage = {'passed': True, 'all_collidable_clutter_covered': True, 'expected_clutter_count': 0,
            'covered_clutter_count': 0, 'covered_collision_component_count': 0}
        execution = {'passed': True, 'hardware_io': False, 'every_actual_physics_step_checked': True,
            'complete_physics_telemetry_coverage_verified': True, 'steps_checked': 1, 'original_report_physical_steps': 1,
            'robot_clutter_active_contacts': 0, 'bottle_clutter_active_contacts': 0, 'max_clutter_vertex_displacement_m': 0.}
        trace = [{'step': 1, 'randomized_environment': {'native_clutter_components_checked': 0,
            'active_target_or_robot_clutter_contacts': [], 'max_vertex_displacement_from_initial_m': 0.}}]
        for name, value in [('randomized_scene_receipt.json', scene), ('randomized_environment_coverage.json', coverage),
                            ('randomized_environment_execution.json', execution), ('physics_trace.json', trace)]:
            publisher.write(directory / name, value)
        (directory / 'run_config.yml').write_text(json.dumps({'domain_randomization': publisher.DR}))
        return trace, scene

    def test_official_two_percent_clean_draw_not_relabelled_as_disabled_randomization(self):
        with tempfile.TemporaryDirectory(prefix='publish5-unit-fixture-') as temporary:
            directory = Path(temporary)
            self.random_fixture(directory)
            result = publisher.randomization_gate(directory, self.sample, {'physical_steps': 1})
            self.assertTrue(result['passed'])
            self.assertEqual(result['clutter_count'], 0)

    def test_nonfinite_actual_clutter_drift_rejected_despite_passed_summary(self):
        with tempfile.TemporaryDirectory(prefix='publish5-unit-fixture-') as temporary:
            directory = Path(temporary)
            trace, _ = self.random_fixture(directory)
            trace[0]['randomized_environment']['max_vertex_displacement_from_initial_m'] = float('nan')
            publisher.write(directory / 'physics_trace.json', trace)
            with self.assertRaises(ValueError): publisher.randomization_gate(directory, self.sample, {'physical_steps': 1})

    def test_hidden_actual_clutter_contact_rejected_despite_passed_summary(self):
        with tempfile.TemporaryDirectory(prefix='publish5-unit-fixture-') as temporary:
            directory = Path(temporary)
            trace, _ = self.random_fixture(directory)
            trace[0]['randomized_environment']['active_target_or_robot_clutter_contacts'] = [{'normal_impulse_Ns': .1}]
            publisher.write(directory / 'physics_trace.json', trace)
            with self.assertRaises(ValueError): publisher.randomization_gate(directory, self.sample, {'physical_steps': 1})

    def test_missing_actual_clutter_step_rejected(self):
        with tempfile.TemporaryDirectory(prefix='publish5-unit-fixture-') as temporary:
            directory = Path(temporary)
            self.random_fixture(directory)
            publisher.write(directory / 'physics_trace.json', [])
            with self.assertRaises(ValueError): publisher.randomization_gate(directory, self.sample, {'physical_steps': 1})

    def test_fake_full_randomization_flag_does_not_override_actual_fields(self):
        with tempfile.TemporaryDirectory(prefix='publish5-unit-fixture-') as temporary:
            directory = Path(temporary)
            _, scene = self.random_fixture(directory)
            scene['actual_domain_randomization'] = dict(publisher.DR, random_background=False)
            publisher.write(directory / 'randomized_scene_receipt.json', scene)
            with self.assertRaises(ValueError): publisher.randomization_gate(directory, self.sample, {'physical_steps': 1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
