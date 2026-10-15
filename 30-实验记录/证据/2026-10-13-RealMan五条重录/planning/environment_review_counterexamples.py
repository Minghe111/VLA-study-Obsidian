"""Independent CPU review witnesses; original source bytes stay untouched."""
from pathlib import Path
import importlib.util,json,sys,types
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
LEGACY=ROOT/'source/adjust_bottle_20261002'
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
support=load('review_original_support',LEGACY/'support_departure_certificate_v20.py')
environment=load('review_environment_overlay',ROOT/'planning/randomized_environment_guard.py')
cases={}
# New carry obstacles must not fail the original hardcoded two-object registry.
registry={'realman_table','held_bottle','realman_clutter_00'}
cases['carry_registry_guard_not_extended']={'actual_registry':sorted(registry),
    'original_guard_rejects':registry!={'realman_table','held_bottle'},
    'source':'full_native_carry_planner_v31.py:285'}
# A support certificate may replace only bottle_table, never extra categories.
raw={'joint_limit_audit':{'passed':True},'unique_collision_pairs':1,
     'collisions_first_64':[{'category':'bottle_table'}],
     'thresholds_m':{'bottle_robot':0.,'self':0.,'robot_table':0.,'clutter_00_bottle':.001,'clutter_00_robot':.001},
     'continuous_gap_lower_bound':{n:{'lower_bound_m':.0002 if n=='clutter_00_bottle' else .002}
        for n in ['bottle_robot','self','robot_table','clutter_00_bottle','clutter_00_robot']}}
cases['support_replacement_omits_clutter_bounds']={
    'original_other_categories_pass_returns':bool(support._other_categories_pass(raw)),
    'actual_clutter_bound_below_required_margin':raw['continuous_gap_lower_bound']['clutter_00_bottle']['lower_bound_m']<raw['thresholds_m']['clutter_00_bottle'],
    'source':'support_departure_certificate_v20.py:128'}
class Pose:
    def __init__(self,z=0.,x=0.):self.z=z;self.x=x
    def to_transformation_matrix(self):
        m=np.eye(4);m[0,3]=self.x;m[2,3]=self.z;return m
class PhysxCollisionShapeBox:
    def __init__(self):self.half_size=np.array([.7,.6,.015]);self.local_pose=Pose()
component=types.SimpleNamespace(entity=types.SimpleNamespace(pose=Pose(z=.705)),collision_shapes=[PhysxCollisionShapeBox()])
try:
    support.native_tabletop_geometry(component)
    table_error=None
except ValueError as error:table_error=str(error)
cases['official_randomized_table_height_rejected']={'actual_native_top_z_m':.72,
    'official_table_height_offset_m':-.02,'original_error':table_error,
    'source':'support_departure_certificate_v20.py:164'}
# Missing finiteness in the physical clutter monitor is a real fail-open input.
prior=sys.modules.get('full_mesh_path_audit')
sys.modules['full_mesh_path_audit']=types.SimpleNamespace(_pose_matrix=lambda p:p.to_transformation_matrix())
clutter=types.SimpleNamespace(name='clutter',pose=Pose(x=float('nan')))
arm=types.SimpleNamespace(name='arm');bottle=types.SimpleNamespace(name='001_bottle')
task=types.SimpleNamespace(robot=types.SimpleNamespace(left_entity=types.SimpleNamespace(links=[types.SimpleNamespace(entity=arm)])),
    bottle=types.SimpleNamespace(actor=bottle),scene=types.SimpleNamespace(get_contacts=lambda:[]))
guard={'records':[{'entity':clutter,'vertices':np.array([[0.,0.,0.],[.1,0.,0.]]),'initial_matrix':np.eye(4)}],
    'steps_checked':0,'max_clutter_vertex_displacement_m':0.,'robot_clutter_active_contacts':0,'bottle_clutter_active_contacts':0}
row={}
try:
    environment.monitor_environment_step(task,1,row,guard)
    monitor_error=None
except (ValueError,RuntimeError) as error:monitor_error=str(error)
finally:
    if prior is None:sys.modules.pop('full_mesh_path_audit',None)
    else:sys.modules['full_mesh_path_audit']=prior
cases['clutter_NaN_drift_input']={'rejected':monitor_error is not None,'error':monitor_error,
    'reported_displacement_m':row.get('randomized_environment',{}).get('max_vertex_displacement_from_initial_m'),
    'source':'randomized_environment_guard.py::monitor_environment_step'}
receipt={'scope':'Static original source and controlled CPU counterexamples, no scene/GPU/physical execution',
    'cases':cases,'root_overlay_was_not_modified':True}
(ROOT/'planning/environment_independent_review_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
