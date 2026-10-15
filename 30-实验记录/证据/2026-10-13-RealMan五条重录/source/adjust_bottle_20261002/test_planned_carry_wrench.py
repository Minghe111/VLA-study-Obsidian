"""CPU checks for rigid planned carry wrench; no simulation or hardware IO."""
import json
import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from planned_carry_wrench import planned_carry_wrench, preflight_contact_lp

MASS = .009999999776482582
INERTIA = np.array([.00287611479870975,.00034546677488833666,.0028816256672143936])
CMASS = np.array([
    [.9999178647994995,-.00037181106745265424,-.012811822816729546,.00003116555672022514],
    [.00037067433004267514,.9999999403953552,-.00009110154496738687,.11045115441083908],
    [.012811856344342232,.00008634504774818197,.9999179244041443,.00001299949872191064],
    [0,0,0,1]])
V9_BOTTLE = np.array([
    [-.16230738162994385,.9867256879806519,.005343926139175892,-.09982889890670776],
    [-.9865514039993286,-.16237986087799072,.01868004910647869,-.09661271423101425],
    [.019299831241369247,-.002240147441625595,.9998112320899963,.7738915681838989],
    [0,0,0,1]])
V9_CONTACTS = [
    dict(channel="middle", point_world_m=[.04140549153089523,-.08639375120401382,.7811410427093506],
         force_on_bottle_normal_world=[-.1542235165834427,-.9479069709777832,-.2787247598171234]),
    dict(channel="thumb", point_world_m=[.05687480419874191,-.1551530510187149,.7631456851959229],
         force_on_bottle_normal_world=[.1554470658302307,.9473802447319031,.27983367443084717]),
    dict(channel="index", point_world_m=[.06377478688955307,-.09112672507762909,.7848598957061768],
         force_on_bottle_normal_world=[-.15422353148460388,-.9479069709777832,-.2787247896194458])]
PREVIOUS = [0,0,.576306,.827664,1.398137]


def poses(n):
    return np.tile(np.eye(4), (n,1,1))


def schedule(palm, relative=np.eye(4), **kwargs):
    return planned_carry_wrench(palm, relative, cmass_local_matrix=kwargs.pop("cmass",CMASS),
        mass_kg=kwargs.pop("mass",MASS), inertia_kg_m2=kwargs.pop("inertia",INERTIA), **kwargs)


def synthetic_contacts():
    return [
        dict(channel="middle",point_world_m=[-.02,.03,0],force_on_bottle_normal_world=[0,-1,0]),
        dict(channel="index",point_world_m=[.02,.03,0],force_on_bottle_normal_world=[0,-1,0]),
        dict(channel="thumb",point_world_m=[0,-.03,0],force_on_bottle_normal_world=[0,1,0])]


def v9_quintic_reference(T=40.,dt=.01):
    # Frozen real-v9 three contacts; ideal rotation about physical COM.
    n=int(round(T/dt))+1; t=np.arange(n)*dt; s=t/T
    u=10*s**3-15*s**4+6*s**5
    axis=np.array([.513418,-.606042,.607548]);axis/=np.linalg.norm(axis)
    theta=math.radians(125.9427)
    D=Rotation.from_rotvec((theta*u)[:,None]*axis).as_matrix()
    com=(V9_BOTTLE@CMASS)[:3,3]
    bottle=poses(n)
    bottle[:,:3,:3]=D@V9_BOTTLE[:3,:3]
    bottle[:,:3,3]=com+np.einsum("nij,j->ni",D,V9_BOTTLE[:3,3]-com)
    out=schedule(bottle,dt=dt,endpoint_mode="hold")
    return out,axis,theta,s


class PlannedCarryTests(unittest.TestCase):
    def test_stationary_real_mass_com_rotation_inertia(self):
        palm=poses(101);palm[:,:3,:3]=Rotation.from_euler("xyz",[.2,-.5,.3]).as_matrix()
        palm[:,:3,3]=[1,2,3]
        relative=np.eye(4);relative[:3,3]=[.1,.2,-.3]
        out=schedule(palm,relative)
        expected=palm[0]@relative@CMASS
        np.testing.assert_allclose(out["planned_com_world_m"],np.tile(expected[:3,3],(101,1)),atol=1e-12)
        np.testing.assert_allclose(out["force_target_world_N"],np.tile([0,0,MASS*9.8],(101,1)),atol=1e-12)
        np.testing.assert_allclose(out["torque_target_world_Nm"],0,atol=1e-12)
        expected_I=expected[:3,:3]@np.diag(INERTIA)@expected[:3,:3].T
        np.testing.assert_allclose(out["inertia_world_kg_m2"][0],expected_I,atol=1e-12)

    def test_constant_linear_and_nonuniform_timestamps(self):
        t=np.cumsum(np.r_[0,np.linspace(.003,.007,100)])
        palm=poses(len(t));velocity=np.array([.1,-.2,.03]);palm[:,:3,3]=t[:,None]*velocity
        out=schedule(palm,timestamps_s=t)
        np.testing.assert_allclose(out["com_velocity_world_m_s"],np.tile(velocity,(len(t),1)),atol=1e-10)
        np.testing.assert_allclose(out["com_acceleration_world_m_s2"],0,atol=1e-9)
        np.testing.assert_allclose(out["force_target_world_N"],np.tile([0,0,MASS*9.8],(len(t),1)),atol=1e-10)

    def test_constant_world_omega_and_gyroscopic_torque(self):
        n=201;dt=.004;t=np.arange(n)*dt
        axis=np.array([.2,.3,.8]);axis/=np.linalg.norm(axis);omega=.7*axis
        base=Rotation.from_euler("xyz",[.5,.3,-.7]).as_matrix()
        palm=poses(n);palm[:,:3,:3]=Rotation.from_rotvec(t[:,None]*omega).as_matrix()@base
        out=schedule(palm,cmass=np.eye(4))
        np.testing.assert_allclose(out["omega_world_rad_s"],np.tile(omega,(n,1)),atol=1e-10)
        np.testing.assert_allclose(out["alpha_world_rad_s2"],0,atol=1e-8)
        expected=np.cross(np.tile(omega,(n,1)),np.einsum("nij,j->ni",out["inertia_world_kg_m2"],omega))
        np.testing.assert_allclose(out["torque_target_world_Nm"],expected,atol=1e-10)

    def test_quintic_rotation_analytic_inertia(self):
        out,axis,theta,s=v9_quintic_reference()
        wp=theta*(30*s**2-60*s**3+30*s**4)/40
        ap=theta*(60*s-180*s**2+120*s**3)/40**2
        expected_w=wp[:,None]*axis
        expected_a=ap[:,None]*axis
        # Central-difference truncation at dt=.01,T=40: omega<3.5e-8,
        # alpha<2.6e-9; allow a small margin for float32 fixture rotations.
        np.testing.assert_allclose(out["omega_world_rad_s"][2:-2],expected_w[2:-2],atol=4e-8)
        np.testing.assert_allclose(out["alpha_world_rad_s2"][2:-2],expected_a[2:-2],atol=3e-9)
        expected_tau=np.einsum("nij,nj->ni",out["inertia_world_kg_m2"],expected_a)+np.cross(
            expected_w,np.einsum("nij,nj->ni",out["inertia_world_kg_m2"],expected_w))
        np.testing.assert_allclose(out["torque_target_world_Nm"][2:-2],expected_tau[2:-2],atol=1e-11)
        np.testing.assert_allclose(out["com_acceleration_world_m_s2"],0,atol=1e-8)

    def test_discrete_translation_knot_is_retained(self):
        n=101;dt=.004;t=np.arange(n)*dt
        palm=poses(n);palm[:,0,3]=np.maximum(0,t-t[50])  # velocity 0 -> 1 m/s at sample50
        out=schedule(palm,cmass=np.eye(4),dt=dt)
        self.assertAlmostEqual(out["com_acceleration_world_m_s2"][50,0],1/dt,places=8)
        self.assertEqual(int(np.argmax(np.linalg.norm(out["com_acceleration_world_m_s2"],axis=1))),50)

    def test_discrete_angular_knot_is_retained(self):
        n=101;dt=.004;t=np.arange(n)*dt
        palm=poses(n);angle=np.maximum(0,t-t[50])
        palm[:,:3,:3]=Rotation.from_rotvec(np.c_[angle,np.zeros((n,2))]).as_matrix()
        out=schedule(palm,cmass=np.eye(4),dt=dt)
        self.assertAlmostEqual(out["alpha_world_rad_s2"][50,0],1/dt,places=7)

    def test_hold_endpoints_include_explicit_rest_transition(self):
        n=10;dt=.004;palm=poses(n);palm[:,0,3]=np.arange(n)*dt
        out=schedule(palm,cmass=np.eye(4),dt=dt,endpoint_mode="hold")
        self.assertAlmostEqual(out["com_acceleration_world_m_s2"][0,0],1/dt)
        self.assertAlmostEqual(out["com_acceleration_world_m_s2"][-1,0],-1/dt)
        np.testing.assert_allclose(out["com_acceleration_world_m_s2"][1:-1],0,atol=1e-9)

    def test_input_validation_and_no_mutation(self):
        p=poses(11);before=p.copy();schedule(p);np.testing.assert_array_equal(p,before)
        with self.assertRaises(ValueError):schedule(p,timestamps_s=np.zeros(11))
        with self.assertRaises(ValueError):schedule(p,inertia=[1,0,1])
        p[0,0,0]=2
        with self.assertRaises(ValueError):schedule(p)

    def test_actual_readback_file_matches_fixture(self):
        p=Path(__file__).with_name("bottle_mass_inertia_cmass_20261003.json")
        if not p.exists():self.skipTest("server readback evidence unavailable on this host")
        data=json.loads(p.read_text())
        self.assertEqual(data["mass"],MASS)
        np.testing.assert_array_equal(data["inertia"],INERTIA)
        np.testing.assert_allclose(data["cmass_local_pose"],CMASS,atol=1e-18,rtol=0)

    def test_preflight_frozen_contacts_and_all_samples(self):
        p=poses(9);out=schedule(p,cmass=np.eye(4))
        audit=preflight_contact_lp(out,synthetic_contacts(),sample_stride=1,include_cases=True)
        self.assertTrue(audit["passed"],audit)
        self.assertEqual(audit["coverage"],"all_discrete_samples")
        self.assertEqual(audit["checked_count"],9)
        np.testing.assert_allclose(audit["cases"][0]["contacts"][0]["point_world_m"],[-.02,.03,0])

    def test_sparse_preflight_adds_raw_knot_failure_evidence(self):
        n=101;dt=.004;t=np.arange(n)*dt
        p=poses(n);p[:,0,3]=100*np.maximum(0,t-t[23])
        out=schedule(p,cmass=np.eye(4),dt=dt)
        audit=preflight_contact_lp(out,synthetic_contacts(),sample_stride=50)
        self.assertFalse(audit["passed"])
        self.assertIn(23,audit["checked_indices"])
        self.assertTrue(any(x["sample_index"]==23 for x in audit["failed_cases"]))
        self.assertIn("contacts",audit["failed_cases"][0])
        self.assertEqual(audit["coverage"],"selected_discrete_samples_only")
        self.assertFalse(audit["continuous_feasibility_certified"])

    def test_v9_real_geometry_inertial_quintic_reference(self):
        out,_,_,_=v9_quintic_reference()
        audit=preflight_contact_lp(out,V9_CONTACTS,sample_stride=50,previous_allocation_N=PREVIOUS)
        self.assertTrue(audit["passed"],audit)
        self.assertLess(audit["peak_torque_target_Nm"]["norm"],2e-5)
        self.assertLess(audit["maximum_checked_force_residual_inf_N"],1e-7)
        self.assertLess(audit["maximum_checked_torque_residual_inf_Nm"],1e-8)

    def test_preflight_rejects_bad_timestamps_and_handles_zero_caps(self):
        out=schedule(poses(9),cmass=np.eye(4),gravity_world_m_s2=[0,0,0])
        audit=preflight_contact_lp(out,synthetic_contacts(),sample_stride=1,
            lp_options={"min_main_force_N":0,"max_channel_force_N":[0]*5})
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["maximum_cap_utilization"],0)
        json.dumps(audit,allow_nan=False)
        out["timestamps_s"][1]=out["timestamps_s"][0]
        with self.assertRaises(ValueError):preflight_contact_lp(out,synthetic_contacts())


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PlannedCarryTests))
    if not result.wasSuccessful():sys.exit(1)
    out,_,_,_=v9_quintic_reference(T=40.,dt=.004)
    audit=preflight_contact_lp(out,V9_CONTACTS,sample_stride=50,previous_allocation_N=PREVIOUS)
    print(json.dumps(dict(tests_run=result.testsRun,successful=result.wasSuccessful(),
        fixture_source="v9 final close_hold pose + pre_retreat true3contacts; official PhysX mass/inertia/cmass readback 20261003",
        fixture_scope="ideal fixed-axis rigid COM rotation; NOT v10 actual joint path or physical success",
        python=sys.version,numpy=np.__version__,mass_kg=MASS,inertia_kg_m2=INERTIA.tolist(),
        cmass_local_matrix=CMASS.tolist(),reference_40s_dt004=audit),indent=2))

