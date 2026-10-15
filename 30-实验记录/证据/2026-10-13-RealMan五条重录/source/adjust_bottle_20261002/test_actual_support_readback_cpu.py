"""Pure CPU schema/partial-evidence checks; no SAPIEN/GPU/scene step."""
from pathlib import Path
import copy,json,tempfile,unittest
import numpy as np
import verify_actual_support_departure as v

class SupportReadbackTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.d=Path(self.t.name)
        self.g=self.d/'mesh.json';self.r=self.d/'attempt_report.json'
        self.guard={'bottle_collision_vertices_local_m':[[0,0,0],[.01,.01,.1]],'table_world_bounds_m':[[-.5,-.5,.69],[.5,.5,.74]],'table_top_world_z_m':.74,'release_gap_m':.001}
        self.report={'accepted':True,'result':'physics_success','physical_steps':3,'full_qpos_joint_names':['joint'+str(i) for i in range(36)],'action_timing':[{'start_step':0,'end_step':3,'duration_s':.012}],
          'support_departure_initial':{'configured_at_step':1,'baseline_signed_gap_m':-5e-6,'floor_signed_gap_m':-25e-6},
          'support_departure_actual':{'release_step':2,'checked_steps':2}}
        self.rows=[]
        for i,gap in enumerate([-5e-6,.002,.01],1):
            p=np.eye(4);p[2,3]=.74+gap
            row={'step':i,'sim_time_s':i*.004,'phase':{'name':'after_grasp'},'bottle':{'pose':p.tolist()},'actual_self_contacts':[], 'full_qpos_actual_rad':[0]*36}
            if i>1:row['support_departure']={'actual_signed_gap_m':gap,'baseline_signed_gap_m':-5e-6,'floor_signed_gap_m':-25e-6}
            self.rows.append(row)
    def tearDown(self):self.t.cleanup()
    def run_case(self):
        self.g.write_text(json.dumps({'physical_signed_height_guard':self.guard}));self.r.write_text(json.dumps(self.report));(self.d/'physics_trace.json').write_text(json.dumps(self.rows),encoding='utf-8')
        return v.analyze(self.d,self.g,dt_s=.004)
    def test_accepted_geometry_missing_impulses_is_not_zero(self):
        r=self.run_case();self.assertTrue(r['passed']);self.assertFalse(r['contact_telemetry']['full_coverage']);self.assertIsNone(r['per_physics_step'][-1]['table_normal_force_N'])
    def test_failed_prefix_is_not_completed_carry(self):
        self.report.update(result='exception',accepted=False);self.report.pop('physical_steps');r=self.run_case()
        self.assertFalse(r['passed']);self.assertTrue(r['recorded_support_prefix_passed']);self.assertFalse(r['geometry_passed']);self.assertFalse(r['completed_episode_endpoint_covered'])
    def test_positive_gap_true_contact_offset_is_preserved(self):
        for row in self.rows:row['bottle_actual_contacts']=[]
        self.rows[-1]['bottle_actual_contacts']=[{'other_link':'table','normal_impulse_Ns':.004,'points':[{'normal_impulse_Ns':.004,'impulse_Ns':[0,0,.004],'normal_world':[0,0,1],'separation_m':.0004}]}]
        r=self.run_case();self.assertTrue(r['passed']);self.assertTrue(r['contact_telemetry']['full_coverage']);self.assertEqual(len(r['contact_telemetry']['active_table_impulse_after_release']),1);self.assertEqual(r['per_physics_step'][-1]['table_normal_force_N'],1)
    def test_depth_violation_is_fail_closed(self):
        self.rows[1]['bottle']['pose'][2][3]=.74-.0001;self.rows[1]['support_departure']['actual_signed_gap_m']=-.0001;self.report['support_departure_actual']['release_step']=3
        r=self.run_case();self.assertFalse(r['passed']);self.assertFalse(r['recorded_support_prefix_passed']);self.assertEqual(r['signed_gap_or_footprint_violations'][0]['step'],2)
    def test_nonfinite_or_nonrigid_reject(self):
        self.rows[-1]['bottle']['pose'][0][0]=2
        with self.assertRaisesRegex(ValueError,'Nonrigid'):self.run_case()
    def test_actual_self_active_not_shell(self):
        self.rows[1]['actual_self_contacts']=[{'names':['a','b'],'normal_impulse_Ns':0}]
        self.rows[2]['actual_self_contacts']=[{'links':['c','d'],'normal_impulse_Ns':2e-7}]
        r=self.run_case();self.assertEqual(len(r['recorded_actual_self_contacts']['active_normal_contacts']),1)
    def test_stream_chunk_boundaries(self):
        self.run_case();p=self.d/'physics_trace.json'
        for n in [1,7,31,1024]:self.assertEqual(list(v.iter_rows(p,n)),self.rows)
    def test_missing_endpoint_and_none_release_are_diagnostics(self):
        self.report.update(result='exception',accepted=False);self.report.pop('physical_steps');self.report['support_departure_actual']['release_step']=None
        r=self.run_case();self.assertFalse(r['passed']);self.assertFalse(r['recorded_support_prefix_passed'])

if __name__=='__main__':unittest.main()
