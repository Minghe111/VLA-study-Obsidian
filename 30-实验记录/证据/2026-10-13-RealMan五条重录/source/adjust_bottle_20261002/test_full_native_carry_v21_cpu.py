"""Pure CPU v21 geometry/TOPP changes; no SAPIEN scene, CLIK/RRT or GPU."""
import unittest
import sys
from types import SimpleNamespace,MethodType
import numpy as np
import mplib
import full_native_carry_planner_v21 as p


class V21Tests(unittest.TestCase):
    def task(self,dt=.004):
        planner=SimpleNamespace(joint_vel_limits=np.ones(6),joint_acc_limits=np.ones(6))
        planner.TOPP=MethodType(mplib.Planner.TOPP,planner)
        return SimpleNamespace(scene=SimpleNamespace(get_timestep=lambda:dt),
            robot=SimpleNamespace(left_mplib_planner=SimpleNamespace(planner=planner,TOPP=planner.TOPP)))
    def path(self):
        u=np.linspace(0,1,21);return np.stack([.1*u,.2*u+.01*np.sin(np.pi*u),-.15*u,.08*u,.02*u,-.04*u],axis=1)
    def test_actual_official_TOPP_called_once(self):
        calls=[];code=mplib.Planner.TOPP.__code__;prior=sys.getprofile()
        def observe(frame,event,arg):
            if frame.f_code is code and event=='call':calls.append(1)
        sys.setprofile(observe)
        try:pos,vel,ts,m=p.official_topp_command_polyline(self.task(),self.path())
        finally:sys.setprofile(prior)
        self.assertEqual(len(calls),1);self.assertEqual(m['input_knot_count'],21)
        np.testing.assert_allclose(pos[[0,-1]],self.path()[[0,-1]],atol=1e-8)
        np.testing.assert_allclose(m['audit_vertices'],pos,atol=0,rtol=0)
        self.assertFalse(m['internal_knot_rest_stops_forced']);self.assertEqual(m['additional_hold_seconds'],0)
        self.assertIn('does NOT certify internal',m['command_curve_scope'])
        self.assertTrue(np.isfinite(pos).all());self.assertTrue(np.isfinite(vel).all());self.assertGreater(m['duration_s'],0)
    def test_official_limits_and_physical_clock_recorded(self):
        pos,vel,ts,m=p.official_topp_command_polyline(self.task(),self.path())
        self.assertLessEqual(np.abs(vel).max(),1.00001)
        self.assertLessEqual(np.max(m['native_peak_acceleration_rad_s2']),1.00001)
        self.assertEqual(m['dense_rows'],len(pos));self.assertAlmostEqual(m['dense_duration_s'],len(pos)*.004)
        self.assertGreaterEqual(m['native_grid_over_physical_dt_ratio'],1.)
        self.assertLessEqual(np.abs(np.diff(pos,axis=0)/.004).max(),m['native_grid_over_physical_dt_ratio']+1e-5)
    def test_exact_21_world_knots(self):
        start=np.eye(4);start[:3,3]=[.12,-.21,.84];goal=start.copy();goal[2,3]+=.1
        k=p.cartesian_world_lift_knots(start,goal);self.assertEqual(k.shape,(21,4,4))
        np.testing.assert_allclose(k[:,:2,3],np.tile(start[:2,3],(21,1)))
        np.testing.assert_allclose(np.diff(k[:,2,3]),np.repeat(.005,20))
        np.testing.assert_allclose(k[:,:3,:3],np.tile(start[:3,:3],(21,1,1)))
    def test_wrong_lift_target_rejected(self):
        start=np.eye(4);goal=start.copy();goal[:3,3]=[.002,0,.1]
        with self.assertRaises(ValueError):p.cartesian_world_lift_knots(start,goal)
    def test_clock_mismatch_rejected(self):
        with self.assertRaises(ValueError):p.official_topp_command_polyline(self.task(.005),self.path())
    def test_nonfinite_and_nomotion_rejected(self):
        for path in [np.full((21,6),np.nan),np.zeros((21,6))]:
            with self.assertRaises(ValueError):p.official_topp_command_polyline(self.task(),path)
    def test_native_binding_required(self):
        task=self.task();task.robot.left_mplib_planner.TOPP=lambda *a,**k:None
        with self.assertRaises(ValueError):p.official_topp_command_polyline(task,self.path())


if __name__=='__main__':unittest.main(verbosity=2)
