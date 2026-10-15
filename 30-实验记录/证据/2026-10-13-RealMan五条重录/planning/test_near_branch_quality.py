"""CPU counterexamples. Synthetic fixtures, never robot/physics evidence."""
import unittest,math,json,hashlib
from pathlib import Path
import numpy as np
from near_branch_quality import (enumerate_equivalent_goals,merge_equivalent_goals,
                                 validate_equivalent_goals,path_quality,PrecontactQualityPolicy)

LIMITS=np.array([[-3.1,3.1]]*5+[[-6.28,6.28]])
START=np.array([0.,0.,0.,0.,0.,.93])
RAW=np.array([0.,0.,0.,0.,0.,-5.25])
def proof(**changes):
    v={'passed':True,'link_count':93,'frozen_joint_max_change_rad':0.,'full_link_max_position_error_m':0.,
       'full_link_max_rotation_error_rad':0.,'task_endpoint_position_error_m':0.,'task_endpoint_rotation_error_rad':0.}
    v.update(changes);return v
def simple_path(n=101):
    start=np.zeros(6);goal=np.zeros(6);goal[5]=.2
    p=start[None]+np.linspace(0,1,n)[:,None]*(goal-start)[None]
    return p,start,goal

class Counterexamples(unittest.TestCase):
    def test_nearest_legal_endpoint_order(self):
        before=RAW.copy();rows=enumerate_equivalent_goals(RAW,START,LIMITS,periodic_indices=list(range(6)))
        self.assertAlmostEqual(rows[0]['goal_arm_q_rad'][5],RAW[5]+2*math.pi)
        self.assertTrue(np.array_equal(before,RAW));self.assertFalse(rows[0]['full_model_FK_validated'])
    def test_nonperiodic_cannot_be_wrapped(self):
        raw=RAW.copy();raw[0]=8
        self.assertEqual(enumerate_equivalent_goals(raw,START,LIMITS,periodic_indices=[5]),[])
    def test_legal_outside_raw_requires_predeclared_FK(self):
        raw=RAW.copy();raw[5]=7.3
        rows=enumerate_equivalent_goals(raw,START,LIMITS,periodic_indices=[5])
        self.assertTrue(rows);self.assertTrue(all(abs(r['goal_arm_q_rad'][5])<=6.28 for r in rows))
        self.assertTrue(all(r['full_model_FK_validated'] is False for r in rows))
    def test_unknown_dimension_rejected(self):
        with self.assertRaises(ValueError):enumerate_equivalent_goals(np.zeros(24),np.zeros(24),np.tile([-6,6],(24,1)),periodic_indices=[5])
    def test_candidate_budget_not_silently_truncated(self):
        with self.assertRaises(ValueError):enumerate_equivalent_goals(np.zeros(6),np.zeros(6),np.tile([-100,100],(6,1)),periodic_indices=list(range(6)))
    def test_IK_attempts_globally_sorted_and_deduplicated(self):
        rows=merge_equivalent_goals([RAW,RAW+np.array([0,0,0,0,0,2*math.pi])],START,LIMITS,periodic_indices=[5])
        self.assertEqual(len(rows),2);self.assertLess(rows[0]['distance_from_actual_start_L2_rad'],rows[1]['distance_from_actual_start_L2_rad'])
    def test_FK_callback_mandatory(self):
        with self.assertRaises(ValueError):validate_equivalent_goals([],fk_validator=None)
    def test_full_link_coverage_cannot_be_one_palm(self):
        rows=enumerate_equivalent_goals(RAW,START,LIMITS,periodic_indices=[5])
        self.assertEqual(validate_equivalent_goals(rows,fk_validator=lambda q:proof(link_count=1))['accepted_candidates'],[])
    def test_fixed_joint_motion_rejected(self):
        rows=enumerate_equivalent_goals(RAW,START,LIMITS,periodic_indices=[5])
        self.assertEqual(validate_equivalent_goals(rows,fk_validator=lambda q:proof(frozen_joint_max_change_rad=1e-3))['accepted_candidates'],[])
    def test_task_endpoint_mismatch_rejected(self):
        rows=enumerate_equivalent_goals(RAW,START,LIMITS,periodic_indices=[5])
        self.assertEqual(validate_equivalent_goals(rows,fk_validator=lambda q:proof(task_endpoint_position_error_m=.001))['accepted_candidates'],[])
    def test_FK_nan_rejected(self):
        rows=enumerate_equivalent_goals(RAW,START,LIMITS,periodic_indices=[5])
        self.assertEqual(validate_equivalent_goals(rows,fk_validator=lambda q:proof(full_link_max_position_error_m=float('nan')))['accepted_candidates'],[])
    def test_far_goal_fails_precontact_travel_even_with_same_FK(self):
        rows=enumerate_equivalent_goals(RAW,START,LIMITS,periodic_indices=[5])
        report=validate_equivalent_goals(rows,fk_validator=lambda q:proof())
        self.assertEqual(len(report['accepted_candidates']),1)
        self.assertLess(abs(report['accepted_candidates'][0]['J6_delta_from_actual_start_rad']),math.pi)
    def test_selected_goal_identity_not_any_equivalent(self):
        p=np.stack([START,RAW]);near=RAW.copy();near[5]+=2*math.pi;before=p.copy()
        row=path_quality(p,START,near)
        self.assertFalse(row['passed']);self.assertTrue(any('specifically selected' in s for s in row['rejection_reasons']))
        self.assertTrue(np.array_equal(before,p))
    def test_path_start_must_be_current_actual(self):
        p,s,g=simple_path();p[0,0]=.04
        self.assertFalse(path_quality(p,s,g)['passed'])
    def test_sparse_RRT_edge_not_physical_step(self):
        p,s,g=simple_path(2)
        self.assertTrue(path_quality(p,s,g)['passed'])
    def test_dense_step_uses_original_velocity_and_dt(self):
        p,s,g=simple_path(2)
        row=path_quality(p,s,g,times_s=[0,.004],velocity_limits_rad_s=np.ones(6))
        self.assertFalse(row['passed']);self.assertTrue(any('velocity-times-dt' in s for s in row['rejection_reasons']))
    def test_retimed_smooth_near_path_passes(self):
        p,s,g=simple_path();row=path_quality(p,s,g,times_s=np.arange(len(p))*.004,velocity_limits_rad_s=np.ones(6),hardware_limits=LIMITS)
        self.assertTrue(row['passed']);self.assertFalse(row['trajectory_modified'])
    def test_smooth_long_excursion_then_return_rejected(self):
        p,s,g=simple_path(3);p[1,5]=4
        self.assertFalse(path_quality(p,s,g)['passed'])
    def test_range_small_but_many_reversals_rejected(self):
        p=np.zeros((101,6));p[1:-1,5]=np.tile([.1,-.1],50)[:99]
        row=path_quality(p,np.zeros(6),np.zeros(6));self.assertLess(row['metrics']['each_joint_span_rad'][5],math.pi)
        self.assertFalse(row['passed']);self.assertGreater(row['metrics']['each_joint_total_variation_rad'][5],math.pi)
    def test_world_EE_detour_rejected_even_without_joint_revolution(self):
        p,s,g=simple_path(3);xyz=np.array([[0,0,0],[1,0,0],[.02,0,0]])
        self.assertFalse(path_quality(p,s,g,EE_positions_world_m=xyz)['passed'])
    def test_stationary_zero_denominator_not_false_failure(self):
        p=np.zeros((2,6));row=path_quality(p,p[0],p[1],EE_positions_world_m=np.zeros((2,3)))
        self.assertTrue(row['passed']);self.assertIsNone(row['metrics']['joint_detour_ratio']);self.assertIsNone(row['metrics']['EE_detour_ratio'])
    def test_nonmonotonic_time_rejected(self):
        p,s,g=simple_path(2)
        with self.assertRaises(ValueError):path_quality(p,s,g,times_s=[0,0],velocity_limits_rad_s=np.ones(6))
    def test_nan_policy_does_not_fail_open(self):
        with self.assertRaises(ValueError):PrecontactQualityPolicy(max_J6_span_rad=float('nan'))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Counterexamples)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    root=Path(__file__).resolve().parent
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'fixture_scope':'Synthetic CPU counterexamples; no simulator, physical robot, actual recollected demonstration or model query',
             'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'near_branch_quality.py',Path(__file__)]}}
    (root/'cpu_counterexample_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
