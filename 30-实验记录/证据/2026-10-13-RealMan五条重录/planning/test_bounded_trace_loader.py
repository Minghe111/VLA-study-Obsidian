import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
P=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261014_v5_staged/jobs/slot34_runtime')
sys.path[:0]=[str(P/'source/adjust_bottle_20261002'),str(P/'source/rgb50_20261007_v1')]
from observed_trace_audit_v13 import _load_trace as original
from bounded_trace_overlay import derive
class LoaderTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.bounded,_=derive(original,'loader',self.root)
  m=np.eye(4).tolist();self.rows=[{'step':i,'phase':{'name':'close' if i<=3 else 'after_grasp'},'bottle':{'pose':m},'arm_qpos':{'left':[0]*6,'right':[0]*6},'grip':{'relative_bottle_palm':m}} for i in range(1,8)]
  self.report={'pre_retreat_grip':{'passed':True,'reference_step':3,'actual':{'relative_bottle_palm':copy.deepcopy(m)}}}
 def tearDown(self):self.temp.cleanup()
 def execute(self):
  t=self.root/'trace.json';r=self.root/'report.json';t.write_text(json.dumps(self.rows));r.write_text(json.dumps(self.report))
  return t,r
 def test_exact_pass_results(self):
  t,r=self.execute();a=original(t,r);b=self.bounded(t,r);self.assertEqual(a[0],list(b[0]));self.assertEqual(a[1:4],b[1:4]);np.testing.assert_array_equal(a[4],b[4])
 def reject_same(self):
  t,r=self.execute()
  for f in (original,self.bounded):
   with self.assertRaises(ValueError):f(t,r)
 def test_missingstep(self):self.rows[4]['step']=2;self.reject_same()
 def test_late_reset_reference(self):self.rows[0]['phase']['name']='after_grasp';self.reject_same()
 def test_missing_real_close(self):
  for row in self.rows:row['phase']['name']='after_grasp'
  self.reject_same()
 def test_wrong_reference_step(self):self.report['pre_retreat_grip']['reference_step']=2;self.reject_same()
 def test_nonfinite_arm(self):self.rows[0]['arm_qpos']['left'][1]=float('nan');self.reject_same()
 def test_no_endpoint_afterclose(self):self.rows=self.rows[:3];self.reject_same()
 def test_reference_matrix_error(self):self.report['pre_retreat_grip']['actual']['relative_bottle_palm'][0][3]=.001;self.reject_same()
if __name__=='__main__':unittest.main()
