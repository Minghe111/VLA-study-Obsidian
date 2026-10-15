"""CPU-only validation; no simulation imports, GPU or robot IO."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from purecontact_lp import allocate_contact_forces, CHANNELS

FIXTURES = json.loads(Path(__file__).with_name("purecontact_lp_fixtures.json").read_text(encoding="utf-8"))


def synthetic_contacts():
    return [
        {"channel":"middle", "point_world_m":[-.02,.03,0.], "force_on_bottle_normal_world":[0.,-1.,0.]},
        {"channel":"index", "point_world_m":[.02,.03,0.], "force_on_bottle_normal_world":[0.,-1.,0.]},
        {"channel":"thumb", "point_world_m":[0.,-.03,0.], "force_on_bottle_normal_world":[0.,1.,0.]},
    ]


def fixture_cases():
    com = np.array(FIXTURES["com_world_m"])
    q0 = np.array(FIXTURES["initial_functional_quaternion_wxyz"])
    q1 = np.array(FIXTURES["goal_functional_quaternion_wxyz"])
    d = Rotation.from_quat(q1[[1,2,3,0]]).as_matrix() @ Rotation.from_quat(q0[[1,2,3,0]]).as_matrix().T
    upright = []
    for c in FIXTURES["actual_contacts"]:
        upright.append(dict(channel=c["channel"],
            point_world_m=(com + d @ (np.array(c["point_world_m"])-com)).tolist(),
            force_on_bottle_normal_world=(d @ np.array(c["force_on_bottle_normal_world"])).tolist()))
    return [("nominal",FIXTURES["nominal_contacts"]),("actual_v8",FIXTURES["actual_contacts"]),("official_goal_upright",upright)]


def verify_external_wrench(test, result, contacts, com, mass):
    test.assertTrue(result["feasible"], result)
    test.assertTrue(result["usable"])
    allocation = np.array(result["target_normal_forces_N"])
    test.assertEqual(allocation.shape,(5,))
    test.assertEqual(allocation[0],0.)
    test.assertEqual(allocation[1],0.)
    test.assertTrue(np.all(allocation[2:]>=.1-1e-8))
    test.assertTrue(np.all(allocation[2:]<=np.array([1.5,1.5,2.5])+1e-8))
    force = np.zeros(3)
    torque = np.zeros(3)
    grouped = np.zeros(5)
    active = [c for c in contacts if c["channel"] in ("middle","index","thumb")]
    for c, f in zip(active,result["theoretical_contact_forces_world_N"]):
        f = np.array(f)
        n = np.array(c["force_on_bottle_normal_world"])
        n /= np.linalg.norm(n)
        fn = float(n@f)
        test.assertGreaterEqual(fn,-1e-8)
        test.assertLessEqual(np.linalg.norm(f-fn*n),result["mu"]*fn+1e-7)
        grouped[CHANNELS.index(c["channel"])]+=fn
        force += f
        torque += np.cross(np.array(c["point_world_m"])-com,f)
    np.testing.assert_allclose(grouped,allocation,atol=1e-8)
    np.testing.assert_allclose(force,[0,0,mass*9.8],atol=1e-7)
    np.testing.assert_allclose(torque,[0,0,0],atol=1e-8)
    test.assertLess(result["force_residual_inf_N"],1e-7)
    test.assertLess(result["torque_residual_inf_Nm"],1e-8)
    json.dumps(result)


class PureContactLPTests(unittest.TestCase):
    def test_synthetic_force_and_moment_balance(self):
        contacts = synthetic_contacts()
        result = allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=.01)
        verify_external_wrench(self,result,contacts,np.zeros(3),.01)

    def test_three_known_geometries(self):
        for name,contacts in fixture_cases():
            with self.subTest(geometry=name):
                result = allocate_contact_forces(contacts,com_world_m=FIXTURES["com_world_m"],mass_kg=FIXTURES["mass_kg"])
                verify_external_wrench(self,result,contacts,np.array(FIXTURES["com_world_m"]),FIXTURES["mass_kg"])

    def test_feasible_previous_is_preferred_to_force_reduction(self):
        previous = [0,0,.3,.3,.6]
        result = allocate_contact_forces(synthetic_contacts(),com_world_m=[0,0,0],mass_kg=.01,previous_allocation_N=previous)
        self.assertTrue(result["feasible"],result)
        np.testing.assert_allclose(result["target_normal_forces_N"],previous,atol=1e-8)

    def test_previous_is_not_blindly_reused_when_wrench_changes(self):
        previous = [0,0,.1,.1,.1]
        result = allocate_contact_forces(synthetic_contacts(),com_world_m=[0,0,0],mass_kg=.01,previous_allocation_N=previous)
        verify_external_wrench(self,result,synthetic_contacts(),np.zeros(3),.01)
        self.assertGreater(result["target_normal_forces_N"][4],.19)

    def test_duplicate_points_bounds_are_per_channel(self):
        contacts = synthetic_contacts()
        duplicated = contacts+copy.deepcopy(contacts)
        result = allocate_contact_forces(duplicated,com_world_m=[0,0,0],mass_kg=.01)
        verify_external_wrench(self,result,duplicated,np.zeros(3),.01)
        expected = allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=.01)
        np.testing.assert_allclose(result["target_normal_forces_N"],expected["target_normal_forces_N"],atol=1e-8)

    def test_ring_little_disabled(self):
        contacts=synthetic_contacts()+[{"channel":"little"},{"channel":"ring"}]
        result=allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=.01)
        self.assertTrue(result["feasible"],result)
        self.assertEqual(result["ignored_ring_little_contacts"],2)
        self.assertEqual(result["target_normal_forces_N"][:2],[0.,0.])

    def test_missing_contact_is_explicit_unverified_fallback(self):
        previous=[0,0,.3,.3,.6]
        result=allocate_contact_forces(synthetic_contacts()[:2],com_world_m=[0,0,0],mass_kg=.01,previous_allocation_N=previous)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["usable"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["reason"],"missing_main_contacts")
        self.assertEqual(result["missing_channels"],["thumb"])
        self.assertEqual(result["target_normal_forces_N"],previous)

    def test_infeasible_wrench_is_not_returned_as_success(self):
        previous=[0,0,.3,.3,.6]
        result=allocate_contact_forces(synthetic_contacts(),com_world_m=[0,0,0],mass_kg=10.,previous_allocation_N=previous)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["reason"],"lp_infeasible")
        self.assertEqual(result["target_normal_forces_N"],previous)

    def test_coordinate_rotation_and_translation_equivariance(self):
        contacts=synthetic_contacts()
        previous=[0,0,.3,.3,.6]
        original=allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=.01,previous_allocation_N=previous)
        r=Rotation.from_rotvec([.2,-.3,.4]).as_matrix()
        center=np.array([1.,2.,3.])
        moved=[dict(channel=c["channel"],point_world_m=(center+r@c["point_world_m"]).tolist(),
                    force_on_bottle_normal_world=(r@c["force_on_bottle_normal_world"]).tolist()) for c in contacts]
        rotated=allocate_contact_forces(moved,com_world_m=center,mass_kg=.01,previous_allocation_N=previous,target_force_world_N=r@np.array([0,0,.098]))
        self.assertTrue(rotated["feasible"],rotated)
        np.testing.assert_allclose(rotated["target_normal_forces_N"],original["target_normal_forces_N"],atol=1e-7)
        np.testing.assert_allclose(rotated["theoretical_net_wrench_world"][:3],r@np.array([0,0,.098]),atol=1e-7)
        np.testing.assert_allclose(rotated["theoretical_net_wrench_world"][3:],[0,0,0],atol=1e-8)

    def test_invalid_geometry_rejected_without_input_mutation(self):
        contacts=synthetic_contacts()
        before=copy.deepcopy(contacts)
        allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=.01)
        self.assertEqual(contacts,before)
        for bad in ([0,0,0],[float("nan"),0,0]):
            invalid=copy.deepcopy(contacts);invalid[0]["force_on_bottle_normal_world"]=bad
            with self.assertRaises(ValueError):allocate_contact_forces(invalid,com_world_m=[0,0,0],mass_kg=.01)
        with self.assertRaises(ValueError):allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=True)
        with self.assertRaises(ValueError):allocate_contact_forces(contacts,com_world_m=[0,0,0],mass_kg=.01,cone_edges=3)


def audit_fixture_results():
    import scipy
    results=[]
    previous=None
    for name,contacts in fixture_cases():
        start=time.perf_counter()
        out=allocate_contact_forces(contacts,com_world_m=FIXTURES["com_world_m"],mass_kg=FIXTURES["mass_kg"],previous_allocation_N=previous)
        elapsed=time.perf_counter()-start
        results.append({"geometry":name,"elapsed_s":elapsed,**out})
        if out["feasible"]:previous=out["target_normal_forces_N"]
    return {"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
            "scope":"Pure CPU theoretical contact LP only; no physical episode acceptance.",
            "fixture_sequence_results":results}


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PureContactLPTests)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"tests_run":outcome.testsRun,"successful":outcome.wasSuccessful(),**audit_fixture_results()},indent=2))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)

