"""CPU counterexamples for strict randomized support; no scene/physics/GPU."""
from pathlib import Path
import importlib.util,json,math,types,unittest,copy
import numpy as np
import randomized_support_guard as guard

SOURCE=Path(__file__).resolve().parent.parent/'source/adjust_bottle_20261002/support_departure_certificate_v20.py'
spec=importlib.util.spec_from_file_location('original_support_for_tests',SOURCE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
geometry,certificate,details=guard.build_geometry_callable(original,SOURCE.read_text())

class Pose:
    def __init__(self,x=0.,y=0.,z=0.):self.position=[x,y,z]
    def to_transformation_matrix(self):m=np.eye(4);m[:3,3]=self.position;return m
class PhysxCollisionShapeBox:
    def __init__(self,half_size,pose):self.half_size=np.asarray(half_size);self.local_pose=pose
class PhysxCollisionShapeSphere:
    def __init__(self):self.radius=.02;self.local_pose=Pose()
def table(top=.72):
    shapes=[PhysxCollisionShapeBox([.7,.6,.015],Pose(z=top-.015))]
    shapes += [PhysxCollisionShapeBox([.02,.02,.35],Pose(x=x,y=y,z=.35)) for x in [-.6,.6] for y in [-.5,.5]]
    return types.SimpleNamespace(entity=types.SimpleNamespace(pose=Pose()),collision_shapes=shapes)
def raw_audit():
    names=['bottle_robot','self','robot_table','bottle_table','clutter_00_bottle','clutter_00_robot']
    return {'joint_limit_audit':{'passed':True},'unique_collision_pairs':1,
        'collisions_first_64':[{'category':'bottle_table'}],
        'thresholds_m':{n:.001 if n.startswith('clutter') else 0. for n in names},
        'continuous_gap_lower_bound':{n:{'lower_bound_m':-.0001 if n=='bottle_table' else .002} for n in names}}
def task(bias=-.02):
    settings={'random_table_height':.03,'random_background':True,'cluttered_table':True}
    receipt={'passed':True,'table_z_bias_m':bias,'actual_domain_randomization':settings.copy(),
        'official_domain_randomization':settings.copy()}
    return types.SimpleNamespace(table_z_bias=bias,random_table_height=.03,_rerecord_randomization_receipt=receipt)

class SupportCounterexamples(unittest.TestCase):
    def test_only_bottle_table_may_fail_its_original_bound(self):
        self.assertTrue(guard.strict_other_categories_pass(raw_audit()))
    def test_extra_continuous_margin_failure_never_replaced_by_support(self):
        raw=raw_audit();raw['continuous_gap_lower_bound']['clutter_00_bottle']['lower_bound_m']=.0002
        self.assertTrue(original._other_categories_pass(raw))
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_any_extra_category_is_required_even_when_name_unrecognized(self):
        raw=raw_audit();raw['thresholds_m']['new_obstacle']=.001
        raw['continuous_gap_lower_bound']['new_obstacle']={'lower_bound_m':0.}
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_missing_core_category_fails_closed(self):
        raw=raw_audit();raw['thresholds_m'].pop('self');raw['continuous_gap_lower_bound'].pop('self')
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_missing_extra_threshold_fails_closed(self):
        raw=raw_audit();raw['thresholds_m'].pop('clutter_00_robot')
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_NaN_extra_bound_fails_closed(self):
        raw=raw_audit();raw['continuous_gap_lower_bound']['clutter_00_robot']['lower_bound_m']=math.nan
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_real_clutter_collision_never_replaced_by_signed_table_certificate(self):
        raw=raw_audit();raw['unique_collision_pairs']=2;raw['collisions_first_64'].append({'category':'clutter_00_robot'})
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_original_joint_limit_gate_stays_required(self):
        raw=raw_audit();raw['joint_limit_audit']['passed']=False
        self.assertFalse(guard.strict_other_categories_pass(raw))
    def test_actual_shifted_native_table_retains_all_five_shapes(self):
        native=table();before=[s.local_pose.to_transformation_matrix().copy() for s in native.collision_shapes]
        with self.assertRaisesRegex(ValueError,'z=.74m'):original.native_tabletop_geometry(native)
        result=geometry(native,expected_table_top_z_m=.72)
        self.assertAlmostEqual(result['table_top_z'],.72);self.assertEqual(result['table_shape_count'],5)
        self.assertEqual(result['collision_shapes_removed'],0)
        self.assertTrue(all(np.array_equal(a,s.local_pose.to_transformation_matrix()) for a,s in zip(before,native.collision_shapes)))
    def test_actual_top_disagrees_with_recorded_height_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'recorded randomized'):geometry(table(.72),expected_table_top_z_m=.73)
    def test_table_height_cannot_be_implicitly_guessed(self):
        with self.assertRaises(TypeError):geometry(table())
    def test_nonfinite_or_out_of_official_range_expected_height_rejected(self):
        for value in [math.nan,.70,.75,True]:
            with self.assertRaises(ValueError):geometry(table(),expected_table_top_z_m=value)
    def test_unique_horizontal_top_check_retained(self):
        native=table();native.collision_shapes.append(copy.deepcopy(native.collision_shapes[0]))
        with self.assertRaisesRegex(ValueError,'unique real horizontal'):geometry(native,expected_table_top_z_m=.72)
    def test_outside_support_footprint_leg_rejected(self):
        native=table();native.collision_shapes[1].local_pose=Pose(x=2.,z=.35)
        with self.assertRaisesRegex(ValueError,'footprint extends outside'):geometry(native,expected_table_top_z_m=.72)
    def test_unsupported_shape_rejected(self):
        native=table();native.collision_shapes.append(PhysxCollisionShapeSphere())
        with self.assertRaisesRegex(ValueError,'verified native Box'):geometry(native,expected_table_top_z_m=.72)
    def test_actual_bias_matches_verified_official_receipt(self):
        evidence=guard.recorded_table_expectation(task())
        self.assertTrue(evidence['passed']);self.assertAlmostEqual(evidence['expected_table_top_z_m'],.72)
    def test_actual_bias_differs_from_record_fails_closed(self):
        t=task();t.table_z_bias=-.01
        with self.assertRaisesRegex(ValueError,'differs from actual'):guard.recorded_table_expectation(t)
    def test_outside_native_official_sampling_range_rejected(self):
        with self.assertRaisesRegex(ValueError,'sampling range'):guard.recorded_table_expectation(task(.01))
    def test_missing_setup_receipt_cannot_infer_height(self):
        with self.assertRaisesRegex(ValueError,'Missing finite'):guard.recorded_table_expectation(types.SimpleNamespace(table_z_bias=-.02))
    def test_unrequested_DR_setting_cannot_be_silently_accepted(self):
        t=task();t._rerecord_randomization_receipt['actual_domain_randomization']['cluttered_table']=False
        with self.assertRaisesRegex(ValueError,'settings differ'):guard.recorded_table_expectation(t)
    def test_actual_table_range_differs_from_record_rejected(self):
        t=task();t.random_table_height=0.
        with self.assertRaisesRegex(ValueError,'range differs'):guard.recorded_table_expectation(t)
    def test_original_source_and_single_certified_call_site_preserved(self):
        self.assertEqual(guard._sha(SOURCE),guard.ORIGINAL_SUPPORT_SHA256)
        self.assertEqual(details['fixed_native_height_guard_sites_changed'],1)
        self.assertEqual(details['explicit_actual_recorded_task_table_height_call_sites_changed'],1)
        import ast
        code=ast.unparse(certificate)
        self.assertIn('expected_table_top_z_m=_expected_table_top_for_task(task)',code)
        self.assertIn('certify_signed_support_polyline(',code)
        self.assertIn('_other_categories_pass(raw_mesh)',code)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SupportCounterexamples))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'scope':'CPU counterexamples using controlled table boxes and original source functions; no actual randomized episode or physical execution',
        'source_sha256':{str(p):guard._sha(p) for p in [Path(__file__),Path(guard.__file__),SOURCE]},'derivation_receipt':details}
    (Path(__file__).parent/'randomized_support_counterexample_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
