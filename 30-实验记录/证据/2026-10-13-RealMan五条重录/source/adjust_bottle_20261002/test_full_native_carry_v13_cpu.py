"""Pure CPU actual official TOPP regression; no SAPIEN scene/IK/RRT/GPU."""
import ast,hashlib,json,pathlib,types,unittest,numpy as np
import full_native_carry_planner_v13 as m
from mplib import Planner
BASE=pathlib.Path(__file__).resolve().parent

class OfficialTOPP(unittest.TestCase):
    def setUp(self):
        self.planner=types.SimpleNamespace(joint_vel_limits=np.ones(6),joint_acc_limits=np.ones(6))
        self.planner.TOPP=Planner.TOPP.__get__(self.planner,Planner)
        self.wrapper=types.SimpleNamespace(planner=self.planner,TOPP=self.planner.TOPP)
        self.task=types.SimpleNamespace(robot=types.SimpleNamespace(left_mplib_planner=self.wrapper,right_mplib_planner=self.wrapper),scene=types.SimpleNamespace(get_timestep=lambda:.004000000189989805))
        self.path=np.array([[0.]*6,[.2,-.1,.06,0.,.12,-.08]])
    def test_real_installed_TOPP_and_limits_unchanged(self):
        before=self.planner.joint_vel_limits.copy();q,v,t,a=m.official_topp_polyline(self.task,self.path)
        self.assertTrue(np.allclose(q[0],self.path[0],atol=1e-10));self.assertTrue(np.allclose(q[-1],self.path[-1],atol=1e-10))
        self.assertLessEqual(abs(v).max(),1+1e-5);self.assertEqual(len(q),int(t[-1]/.004));self.assertTrue(np.array_equal(before,self.planner.joint_vel_limits))
        self.assertLess(a['segments'][0]['sample_line_error_rad'],1e-9);self.assertEqual(a['segments'][0]['two_point_spline_high_order_coeff_max'],0.)
        self.assertAlmostEqual(a['dense_duration_s'],len(q)*.004000000189989805)
    def test_multisegment_turn_exact_no_global_cubic_corner_cut(self):
        p=np.vstack([self.path,np.array([-.06,.14,-.12,.04,.01,.11])]);q,v,t,a=m.official_topp_polyline(self.task,p)
        self.assertEqual(len(a['segments']),2);self.assertTrue(np.allclose(a['audit_vertices'],p))
        index=a['segments'][0]['dense_end_index'];self.assertTrue(np.allclose(q[index],p[1]));self.assertTrue(np.allclose(v[index],np.zeros(6),atol=1e-6,rtol=0))
        self.assertTrue(np.all(np.diff(t)>0));self.assertTrue(np.allclose(q[-1],p[-1]))
    def test_native_unequal_actual_limits_read_without_override(self):
        self.planner.joint_vel_limits=np.array([.4,.8,1.2,.6,.7,.5]);self.planner.joint_acc_limits=np.array([.5,1.,.6,.7,.8,.9])
        q,v,t,a=m.official_topp_polyline(self.task,self.path)
        self.assertTrue(np.all(np.max(abs(v),axis=0)<=self.planner.joint_vel_limits+1e-5));self.assertEqual(a['official_joint_velocity_limits_rad_s'],self.planner.joint_vel_limits.tolist())
    def test_no_motion_has_no_artificial_seconds(self):
        p=np.array([[.1]*6]*3);q,v,t,a=m.official_topp_polyline(self.task,p)
        self.assertEqual(q.shape,(1,6));self.assertEqual(a['duration_s'],0);self.assertTrue(np.all(v==0))
    def test_reject_nonfinite_path(self):
        self.path[1,0]=float('nan')
        with self.assertRaises(ValueError):m.official_topp_polyline(self.task,self.path)
    def test_reject_dt_mismatch(self):
        with self.assertRaises(ValueError):m.official_topp_polyline(self.task,self.path,time_step=.01)
    def test_reject_wrong_side(self):
        with self.assertRaises(ValueError):m.official_topp_polyline(self.task,self.path,'middle')
    def test_reject_invalid_native_limits(self):
        self.planner.joint_vel_limits[0]=0
        with self.assertRaises(ValueError):m.official_topp_polyline(self.task,self.path)
    def test_reject_unreviewed_TOPP_binding(self):
        self.wrapper.TOPP=lambda p,step:None
        with self.assertRaises(ValueError):m.official_topp_polyline(self.task,self.path)
    def test_source_gates_and_actual_curve_audit(self):
        source=(BASE/'full_native_carry_planner_v13.py').read_text();tree=ast.parse(source)
        f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plan_carry_to_task_pose');body=ast.unparse(f)
        self.assertIn('auditor.audit(geometry, side',body);self.assertIn('Actual TOPP command joint limits rejected',body)
        self.assertIn('resolve_bottle_table_bound(task, geometry',body);self.assertIn('Official TOPP mode forbids artificial minimum duration',body)
        self.assertIn('ppg._assert_snapshot(task, before)',body);self.assertIn("retime.get('curve'",body)
        self.assertNotIn('scene.step(',source);self.assertNotIn('cuda',source.lower())

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OfficialTOPP))
    real=[]
    for number in (4,5,6):
        info=json.loads((BASE/f'coupled_side_entry_v12_B085_20261003_seed7/carry_planning_{number}.json').read_text());case=OfficialTOPP();case.setUp()
        q,v,t,a=m.official_topp_polyline(case.task,np.array(info['geometric_polyline']))
        real.append({'original_planner_dense_call':number,'source_original_duration_s':info['retiming']['duration_s'],
                     'native_TOPP_duration_s':a['duration_s'],'dense_rows':len(q),'actual_250Hz_duration_s':a['dense_duration_s'],
                     'native_peak_velocity_rad_s':float(abs(v).max()),'retiming':a})
    out={'passed':result.wasSuccessful(),'test_count':result.testsRun,'tests':[x.id() for x in unittest.defaultTestLoader.loadTestsFromTestCase(OfficialTOPP)],
         'scope':'Pure CPU actual installed official Planner.TOPP only; no scene, IK/RRT, physics, GPU or physical IO',
         'source_sha256':hashlib.sha256((BASE/'full_native_carry_planner_v13.py').read_bytes()).hexdigest(),
         'original_helper_sha256':hashlib.sha256((BASE/'full_native_carry_planner.py').read_bytes()).hexdigest(),
         'actual_v12_sparse_paths_retimed':real,'total_new_dense_carry_duration_s':sum(x['actual_250Hz_duration_s'] for x in real)}
    dest=BASE/'full_native_carry_v13_cpu_audit_20261003.json'
    if dest.exists():raise FileExistsError('Refuse evidence overwrite')
    dest.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'passed':out['passed'],'tests':out['test_count'],'total_new_dense_carry_duration_s':out['total_new_dense_carry_duration_s'],'output':str(dest)}))
    raise SystemExit(0 if out['passed'] else 2)
