"""Pure CPU synthetic certificate regressions; no SAPIEN, planner or GPU."""
import unittest
import math
import numpy as np
from support_departure_certificate import certify_signed_support_polyline, _other_categories_pass


def straight(q,g0=-.00021):
    m=np.eye(4);m[2,3]=.1*float(q[0])
    g=g0+m[2,3]
    return {'signed_gap_m':g,'palm_pose_world':m,'bottle_table_collision':g<=0}


class SupportTests(unittest.TestCase):
    def run_case(self,fn=straight,path=None,derivative=None,**kw):
        p=np.array([[0.],[1.]]) if path is None else np.array(path,dtype=float)
        derivative=derivative or (lambda q,dq:.1*float(dq[0]))
        return certify_signed_support_polyline(p,fn,derivative,[.1],[.1],[True],is_first_world_lift=True,**kw)
    def test_vertical_support_exit(self):
        r=self.run_case();self.assertTrue(r['passed']);self.assertIsNotNone(r['release'])
        self.assertGreaterEqual(r['continuous_minimum_signed_gap_lower_bound_m'],-.000230000001)
        self.assertTrue(any(x['after_release'] for x in r['certified_intervals']))
    def test_duplicate_start_is_allowed(self):
        self.assertTrue(self.run_case(path=[[0],[0],[1]])['passed'])
    def test_too_deep_initial_rejected(self):
        self.assertFalse(self.run_case(fn=lambda q:straight(q,-.0005001))['passed'])
    def test_deeper_hidden_interval_rejected(self):
        def fn(q):
            r=straight(q);r['signed_gap_m']-=.08*math.sin(math.pi*float(q[0]));r['bottle_table_collision']=r['signed_gap_m']<=0;return r
        self.assertFalse(self.run_case(fn=fn)['passed'])
    def test_recontact_after_release_rejected(self):
        def fn(q):
            r=straight(q)
            if .44<float(q[0])<.56:r['signed_gap_m']=-.0001;r['bottle_table_collision']=True
            return r
        self.assertFalse(self.run_case(fn=fn)['passed'])
    def test_collision_flag_after_positive_gap_rejected(self):
        def fn(q):
            r=straight(q)
            if float(q[0])>.1:r['bottle_table_collision']=True
            return r
        self.assertFalse(self.run_case(fn=fn)['passed'])
    def test_wrong_endpoint_rejected(self):
        self.assertFalse(self.run_case(path=[[0],[.99]])['passed'])
    def test_lateral_motion_rejected(self):
        def fn(q):
            r=straight(q);r['palm_pose_world'][0,3]=.004*math.sin(math.pi*float(q[0]));return r
        self.assertFalse(self.run_case(fn=fn)['passed'])
    def test_backward_world_motion_rejected(self):
        self.assertFalse(self.run_case(path=[[0],[.6],[.5],[1]],require_monotonic_world_z=True)['passed'])
    def test_zero_progress_derivative_rejected(self):
        self.assertFalse(self.run_case(derivative=lambda q,dq:0.,require_monotonic_world_z=True)['passed'])
    def test_budget_exhaustion_rejected(self):
        self.assertFalse(self.run_case(max_evaluations=3)['passed'])
    def test_safe_curve_not_rejected_by_optional_monotonic_gate(self):
        self.assertTrue(self.run_case(path=[[0],[.6],[.5],[1]])['passed'])
    def test_measured_baseline_not_replaced_by_fk_error(self):
        self.assertTrue(self.run_case(measured_baseline_signed_gap_m=-.00022)['passed'])
        self.assertFalse(self.run_case(measured_baseline_signed_gap_m=-.00025)['passed'])
    def test_native_start_summary_names(self):
        import full_native_carry_planner_v18 as p
        self.assertTrue(p._support_start_pair_only([{'link_name1':'held_bottle','link_name2':'realman_table'}]))
        self.assertFalse(p._support_start_pair_only([]))
        self.assertFalse(p._support_start_pair_only([{'link_name1':'held_bottle','link_name2':'l_link4'}]))
        self.assertFalse(p._support_start_pair_only([{'link_name1':'held_bottle','link_name2':'realman_table'},{'link_name1':'l_link4','link_name2':'body_base_link'}]))
    def test_non_first_lift_rejected(self):
        with self.assertRaises(ValueError):
            certify_signed_support_polyline([[0],[1]],straight,lambda q,dq:float(dq[0]),[.1],[.1],[True],is_first_world_lift=False)
    def test_nonfinite_rejected(self):
        def fn(q):r=straight(q);r['signed_gap_m']=float('nan');return r
        with self.assertRaises(ValueError):self.run_case(fn=fn)
    def test_scope_budgets_cannot_be_widened(self):
        for opts in ({'max_extra_depth_m':.000021},{'max_initial_penetration_m':.000501},{'release_gap_m':.00099},{'lift_height_m':.11}):
            with self.assertRaises(ValueError):self.run_case(**opts)
    def test_only_table_category_replaced(self):
        raw={'joint_limit_audit':{'passed':True},'collisions_first_64':[{'category':'bottle_table'}],'unique_collision_pairs':1,
             'continuous_gap_lower_bound':{n:{'lower_bound_m':0.} for n in ['bottle_robot','self','robot_table']},
             'thresholds_m':{n:0. for n in ['bottle_robot','self','robot_table']}}
        self.assertTrue(_other_categories_pass(raw));raw['unique_collision_pairs']=2;self.assertFalse(_other_categories_pass(raw))
        raw['unique_collision_pairs']=1;raw['continuous_gap_lower_bound']['self']['lower_bound_m']=-1e-10;self.assertFalse(_other_categories_pass(raw))


if __name__=='__main__':unittest.main(verbosity=2)
