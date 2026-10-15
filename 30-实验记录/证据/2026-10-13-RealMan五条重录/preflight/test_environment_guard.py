"""Counterexamples for actual post-physics environmental rejection; no GPU."""
import sys,types,unittest
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'planning'))
import randomized_environment_guard as guard
class Pose:
 def __init__(self,x=0):self.x=x
 def to_transformation_matrix(self):m=np.eye(4);m[0,3]=self.x;return m
class Entity:
 def __init__(self,name,x=0):self.name=name;self.pose=Pose(x)
class Tests(unittest.TestCase):
 def setUp(self):
  self.old=sys.modules.get('full_mesh_path_audit');sys.modules['full_mesh_path_audit']=types.SimpleNamespace(_pose_matrix=lambda p:p.to_transformation_matrix())
  self.extra=Entity('clutter');self.arm=Entity('l_link6');self.bottle=Entity('001_bottle')
  self.contacts=[];self.task=types.SimpleNamespace(robot=types.SimpleNamespace(left_entity=types.SimpleNamespace(links=[types.SimpleNamespace(entity=self.arm)])),bottle=types.SimpleNamespace(actor=self.bottle),scene=types.SimpleNamespace(get_contacts=lambda:self.contacts))
  self.g={'records':[{'entity':self.extra,'vertices':np.array([[0,0,0],[.1,0,0]]),'initial_matrix':np.eye(4)}], 'steps_checked':0,'max_clutter_vertex_displacement_m':0.,'robot_clutter_active_contacts':0,'bottle_clutter_active_contacts':0}
 def tearDown(self):
  if self.old is None:sys.modules.pop('full_mesh_path_audit',None)
  else:sys.modules['full_mesh_path_audit']=self.old
 def contact(self,first,second,impulse,separation):
  return types.SimpleNamespace(bodies=[types.SimpleNamespace(entity=first),types.SimpleNamespace(entity=second)],points=[types.SimpleNamespace(impulse=np.array([impulse,0,0]),normal=np.array([1,0,0]),separation=separation)])
 def test_safe_stationary_records_one_real_step(self):
  row={'step':1};guard.monitor_environment_step(self.task,1,row,self.g);self.assertEqual(self.g['steps_checked'],1);self.assertEqual(row['randomized_environment']['native_clutter_components_checked'],1)
 def test_positive_robot_clutter_impact_rejected_and_not_misclassified(self):
  self.contacts=[self.contact(self.arm,self.extra,1e-6,.0001),self.contact(self.arm,self.bottle,1e-4,0)]
  with self.assertRaisesRegex(RuntimeError,'collision'):guard.monitor_environment_step(self.task,1,{'step':1},self.g)
  self.assertEqual(self.g['robot_clutter_active_contacts'],1);self.assertEqual(self.g['bottle_clutter_active_contacts'],0)
 def test_bottle_clutter_impact_rejected(self):
  self.contacts=[self.contact(self.bottle,self.extra,1e-6,0)]
  with self.assertRaisesRegex(RuntimeError,'collision'):guard.monitor_environment_step(self.task,1,{'step':1},self.g)
  self.assertEqual(self.g['bottle_clutter_active_contacts'],1)
 def test_zero_impulse_near_contact_does_not_equal_physical_impact(self):
  self.contacts=[self.contact(self.arm,self.extra,0,.001)]
  guard.monitor_environment_step(self.task,1,{'step':1},self.g)
 def test_deep_penetration_rejected_even_without_impulse(self):
  self.contacts=[self.contact(self.arm,self.extra,0,-.0001)]
  with self.assertRaisesRegex(RuntimeError,'collision'):guard.monitor_environment_step(self.task,1,{'step':1},self.g)
 def test_clutter_motion_exceeding_continuous_clearance_budget_rejected(self):
  self.extra.pose=Pose(.0011)
  with self.assertRaisesRegex(RuntimeError,'conservative'):guard.monitor_environment_step(self.task,1,{'step':1},self.g)
 def test_foreign_clutter_table_contact_is_not_robot_collision(self):
  self.contacts=[self.contact(self.extra,Entity('table'),1e-5,0)]
  guard.monitor_environment_step(self.task,1,{'step':1},self.g)
 def test_nonfinite_clutter_pose_rejected(self):
  self.extra.pose=Pose(float('nan'))
  with self.assertRaisesRegex(RuntimeError,'Nonfinite'):guard.monitor_environment_step(self.task,1,{'step':1},self.g)
 def test_nonfinite_contact_rejected(self):
  self.contacts=[self.contact(self.arm,self.extra,float('nan'),0)]
  with self.assertRaisesRegex(RuntimeError,'Nonfinite'):guard.monitor_environment_step(self.task,1,{'step':1},self.g)
 def test_missing_or_skipped_physics_step_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'continuity'):guard.monitor_environment_step(self.task,22,{'step':22},self.g)
 def test_trace_step_mismatch_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'continuity'):guard.monitor_environment_step(self.task,1,{'step':2},self.g)
if __name__=='__main__':unittest.main()
