"""Same-region pure CPU regression, real v24 unknown cases plus fail-closed checks."""
from pathlib import Path
import copy,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog as native
import purecontact_lp_v26 as lp
B=Path(__file__).resolve().parent
FIXTURE=json.loads((B/'v24_failed_lp_cases_fixture_20261003.json').read_text())

def request(case):
    return dict(contacts=case['contacts'],com_world_m=case['com_world_m'],mass_kg=.01,
                previous_allocation_N=case['allocation']['target_normal_forces_N'],
                target_force_world_N=case['force_target_world_N'],target_torque_world_Nm=case['torque_target_world_Nm'],**FIXTURE['lp_options'])

class NumericalLPTests(unittest.TestCase):
    def test_both_real_unknowns_recover_same_original_constraints(self):
        for case in FIXTURE['failed_cases']:
            r=lp.allocate_contact_forces(**request(case));self.assertTrue(r['usable']);self.assertFalse(r['stale']);self.assertEqual(r['solver_status'],0)
            self.assertEqual(r['solver_attempts'][0]['status'],4);self.assertEqual(r['solver_method_used'],'highs-ipm')
            for key in ['force_residual_inf_N','cap_violation_N','minimum_violation_N','cone_violation_N','ray_nonnegative_violation_N','lp_all_inequality_violation_N','lp_all_variable_nonnegative_violation_N','lp_continuity_auxiliary_violation_N']:self.assertLessEqual(r[key],1e-7)
            self.assertLessEqual(r['torque_residual_inf_Nm'],1e-8);self.assertTrue(r['complete_original_lp_checked']);self.assertEqual(r['contact_count'],4)
    def test_first_solve_matrix_cost_rhs_exactly_original(self):
        ns={'__name__':'original_read_only_definition_copy'};exec(compile((B/'purecontact_lp.py').read_text(),str(B/'purecontact_lp.py'),'exec'),ns)
        original=[];new=[]
        def capture_orig(c,**kw):original.append((c.copy(),copy.deepcopy(kw)));return native(c,**kw)
        def capture_new(c,**kw):new.append((c.copy(),copy.deepcopy(kw)));return native(c,**kw)
        ns['linprog']=capture_orig;ns['allocate_contact_forces'](**request(FIXTURE['failed_cases'][0]))
        with patch.object(lp,'linprog',capture_new):lp.allocate_contact_forces(**request(FIXTURE['failed_cases'][0]))
        np.testing.assert_array_equal(original[0][0],new[0][0]);self.assertEqual(set(original[0][1]),set(new[0][1]))
        for k in original[0][1]:
            if isinstance(original[0][1][k],np.ndarray):np.testing.assert_array_equal(original[0][1][k],new[0][1][k])
            else:self.assertEqual(original[0][1][k],new[0][1][k])
    def test_unknown_never_usable_even_if_x_provided(self):
        fake=SimpleNamespace(status=4,success=False,message='Unknown',x=np.ones(67),fun=0.)
        with patch.object(lp,'linprog',return_value=fake):r=lp.allocate_contact_forces(**request(FIXTURE['failed_cases'][0]))
        self.assertFalse(r['usable']);self.assertTrue(r['stale']);self.assertEqual(len(r['solver_attempts']),3)
    def test_full_continuity_auxiliary_checked(self):
        def corrupt(c,**kw):
            result=native(c,**kw)
            if result.success:
                result.x[-3:]=0.
            return result
        with patch.object(lp,'linprog',corrupt):r=lp.allocate_contact_forces(**request(FIXTURE['failed_cases'][0]))
        self.assertFalse(r['usable']);self.assertEqual(r['reason'],'residual_check_failed');self.assertGreater(r['lp_continuity_auxiliary_violation_N'],1e-7)
    def test_scaled_fallback_retains_identical_feasible_region(self):
        orig=[];calls=[]
        def emulate_ipm_unknown(c,**kw):
            calls.append(copy.deepcopy(kw));orig.append(c.copy())
            if kw['method']=='highs-ipm':return SimpleNamespace(status=4,success=False,message='forced Unknown',x=None,fun=None)
            return native(c,**kw)
        with patch.object(lp,'linprog',emulate_ipm_unknown):r=lp.allocate_contact_forces(**request(FIXTURE['failed_cases'][0]))
        self.assertTrue(r['usable']);self.assertEqual(r['solver_method_used'],'highs-ds');meta=r['solver_attempts'][-1]
        for a,b,scale in [('A_eq','b_eq','equality_row_divisors'),('A_ub','b_ub','inequality_row_divisors')]:
            d=np.array(meta[scale]);self.assertTrue(np.all(d>0));np.testing.assert_allclose(calls[-1][a]*d[:,None],calls[0][a],rtol=1e-15,atol=0);np.testing.assert_allclose(calls[-1][b]*d,calls[0][b],rtol=1e-15,atol=0)
        for c in orig:np.testing.assert_array_equal(c,orig[0])
    def test_real_infeasible_stays_fail_closed_without_retry(self):
        q=request(FIXTURE['failed_cases'][0]);q['target_force_world_N']=[0,0,1000]
        r=lp.allocate_contact_forces(**q);self.assertFalse(r['usable']);self.assertEqual(r['solver_status'],2);self.assertEqual(len(r['solver_attempts']),1)
    def test_missing_main_contacts_stays_unusable(self):
        q=request(FIXTURE['failed_cases'][0]);q['contacts']=[x for x in q['contacts'] if x['channel']!='thumb'];r=lp.allocate_contact_forces(**q)
        self.assertFalse(r['usable']);self.assertEqual(r['reason'],'missing_main_contacts')

if __name__=='__main__':unittest.main()
