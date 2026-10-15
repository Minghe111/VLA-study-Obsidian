"""CPU-only strict endpoint numerical counterexamples; no physics or GPU."""
from pathlib import Path
import copy,json,math,sys,unittest
import numpy as np
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import collector_quality_overlay as quality
import strict_goal_pose_roundtrip_overlay as fixed
ORIGINAL=quality._strict_palm_goal

def rotation_z(theta):
 c,s=math.cos(theta),math.sin(theta)
 return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def proof(goal, actual=np.eye(4), q=None):
 q=np.zeros(3) if q is None else q; base=np.zeros(3)
 return fixed.strict_native_goal_proof(ORIGINAL,q,base,[0],lambda supplied_q:np.array([np.eye(4),actual]),goal,1,2)

class Counterexamples(unittest.TestCase):
 def test_identical_true_SO3_and_original_literal_thresholds(self):
  g=np.eye(4);g[:3,:3]=rotation_z(.77)
  r=proof(g,g);self.assertTrue(r['passed']);self.assertEqual(r['new_quality_rotation_tolerance_rad'],2e-4)
  self.assertEqual(r['new_quality_position_tolerance_m'],5e-5)
 def test_float32_goal_roundoff_has_false_raw_angle_but_correct_native_proof(self):
  g=np.eye(4);g[:3,:3]=rotation_z(.54)*(1.-6e-8)
  canonical,_=fixed.validate_native_pose_target(g)
  old=ORIGINAL(np.zeros(3),np.zeros(3),[0],lambda q:np.array([np.eye(4),canonical]),g,1,2)
  self.assertFalse(old['passed']);self.assertTrue(proof(g,canonical)['passed'])
 def test_true_rotation_beyond_threshold_stays_rejected(self):
  g=np.eye(4);g[:3,:3]=rotation_z(2.01e-4)
  self.assertFalse(proof(g)['passed'])
 def test_true_rotation_inside_original_threshold_passes(self):
  g=np.eye(4);g[:3,:3]=rotation_z(1.99e-4)
  self.assertTrue(proof(g)['passed'])
 def test_actual_translation_beyond_threshold_stays_rejected(self):
  g=np.eye(4);g[0,3]=5.01e-5;self.assertFalse(proof(g)['passed'])
 def test_actual_translation_inside_original_threshold_passes(self):
  g=np.eye(4);g[0,3]=4.99e-5;self.assertTrue(proof(g)['passed'])
 def test_frozen_joint_change_stays_rejected(self):
  self.assertFalse(proof(np.eye(4),q=np.array([0.,2e-8,0.]))['passed'])
 def test_reflection_is_rejected(self):
  g=np.eye(4);g[0,0]=-1.
  with self.assertRaisesRegex(ValueError,'roundoff bound'):proof(g)
 def test_scale_and_shear_over_roundoff_rejected(self):
  for index in [(0,0),(0,1)]:
   g=np.eye(4);g[index]+=2e-6
   with self.assertRaisesRegex(ValueError,'roundoff bound'):proof(g)
 def test_large_actual_FK_nonSO3_is_rejected(self):
  a=np.eye(4);a[0,0]-=1e-7
  with self.assertRaisesRegex(ValueError,'genuine SO3'):proof(np.eye(4),a)
 def test_nonfinite_and_wrong_shape_rejected(self):
  for g in [np.full((4,4),np.nan),np.eye(3),np.full((4,4),np.inf)]:
   with self.assertRaisesRegex(ValueError,'Finite complete'):proof(g)
 def test_homogeneous_goal_and_FK_rows_rejected(self):
  g=np.eye(4);g[3,0]=1e-15
  with self.assertRaisesRegex(ValueError,'homogeneous goal'):proof(g)
  a=np.eye(4);a[3,3]=.9999
  with self.assertRaisesRegex(ValueError,'homogeneous rows'):proof(np.eye(4),a)
 def test_native_poses_do_not_mutate_original_goal(self):
  g=np.eye(4);g[:3,:3]=rotation_z(.6).astype('float32');before=g.copy()
  proof(g);np.testing.assert_array_equal(g,before)
 def test_original_threshold_change_fails_closed(self):
  def wrong(*args):
   r=ORIGINAL(*args);r['new_quality_rotation_tolerance_rad']=.01;return r
  with self.assertRaisesRegex(RuntimeError,'thresholds changed'):
   fixed.strict_native_goal_proof(wrong,np.zeros(3),np.zeros(3),[0],lambda q:np.array([np.eye(4),np.eye(4)]),np.eye(4),1,2)

if __name__=='__main__':unittest.main(verbosity=2)
