"""Small synthetic reference-loader regression, CPU only; no scene or planner."""
from pathlib import Path
import copy,hashlib,json,tempfile,unittest,numpy as np
import observed_trace_audit_v13 as loader
BASE=Path(__file__).resolve().parent

class TrueClosingReference(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='reference_v13_',dir=BASE);self.directory=Path(self.temp.name)
        self.trace=[]
        for step,phase in enumerate(('approach','close','close','after_grasp','after_grasp'),1):
            self.trace.append({'step':step,'phase':{'name':phase},'arm_qpos':{'left':[0]*6,'right':[0]*6},'bottle':{'pose':np.eye(4).tolist()},'grip':{'relative_bottle_palm':np.eye(4).tolist()}})
        self.report={'pre_retreat_grip':{'passed':True,'reference_step':3,'actual':{'relative_bottle_palm':np.eye(4).tolist()}}}
    def tearDown(self):self.temp.cleanup()
    def read(self):
        t=self.directory/'trace.json';r=self.directory/'report.json';t.write_text(json.dumps(self.trace));r.write_text(json.dumps(self.report))
        return loader._load_trace(t,r)
    def rejected(self):
        with self.assertRaises((ValueError,KeyError)):self.read()
    def test_last_close_with_no_extra_close_hold_accepted(self):
        before=copy.deepcopy(self.trace);rows,report,anchor,after,ref=self.read();self.assertEqual(anchor,2);self.assertEqual(after,[3,4]);self.assertEqual(rows,before);self.assertTrue(np.array_equal(ref,np.eye(4)))
    def test_reject_missing_reference_step(self):del self.report['pre_retreat_grip']['reference_step'];self.rejected()
    def test_reject_boolean_reference_step(self):self.report['pre_retreat_grip']['reference_step']=True;self.rejected()
    def test_reject_stale_earlier_close_reference(self):self.report['pre_retreat_grip']['reference_step']=2;self.rejected()
    def test_reject_reference_step_outside_trace(self):self.report['pre_retreat_grip']['reference_step']=30;self.rejected()
    def test_reject_gapped_first_after_grasp(self):self.trace[3]['step']=7;self.trace[4]['step']=8;self.rejected()
    def test_reject_synthetic_close_hold_instead_of_last_close(self):self.trace[2]['phase']['name']='close_hold';self.rejected()
    def test_reject_no_after_grasp(self):self.trace=self.trace[:3];self.rejected()
    def test_reject_midcarry_reference_reset(self):self.trace[0]['phase']['name']='after_grasp';self.rejected()
    def test_reject_failed_grip(self):self.report['pre_retreat_grip']['passed']=False;self.rejected()
    def test_reject_report_relative_pose_disagrees(self):self.report['pre_retreat_grip']['actual']['relative_bottle_palm'][0][3]=.00003;self.rejected()
    def test_reject_duplicate_step(self):self.trace[1]['step']=1;self.rejected()
    def test_reject_fractional_physics_step(self):self.trace[0]['step']=1.5;self.rejected()
    def test_reject_nonfinite_arm(self):self.trace[1]['arm_qpos']['left'][0]=float('nan');self.rejected()
    def test_reject_invalid_bottle_pose(self):self.trace[1]['bottle']['pose'][0][0]=0.;self.rejected()
    def test_missing_direct_grip_still_uses_later_full36_fk_validation(self):
        del self.trace[2]['grip'];self.assertEqual(self.read()[2],2)
    def test_complete36_auditor_only_import_diff_and_original_unchanged(self):
        old=(BASE/'observed_full_trace_audit_v11.py').read_text();new=(BASE/'observed_full_trace_audit_v13.py').read_text()
        self.assertEqual(new,old.replace('from observed_trace_audit import _matrix,_relative_metrics,_load_trace,_phase','from observed_trace_audit_v13 import _matrix,_relative_metrics,_load_trace,_phase'))
        self.assertNotIn('scene.step(',new)
        self.assertIn('Full audit requires every physical step beginning at1',new);self.assertIn('Full FK differs from historical direct grasp reference',new)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrueClosingReference))
    sha=lambda name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    report={'passed':result.wasSuccessful(),'tests':result.testsRun,'scope':'Synthetic CPU reference loader and exact source-copy parity only; no simulation/physics/GPU execution',
            'reference_loader_sha256':sha('observed_trace_audit_v13.py'),'full36_auditor_sha256':sha('observed_full_trace_audit_v13.py'),
            'original_v11_auditor_sha256':sha('observed_full_trace_audit_v11.py'),'original_v9_loader_sha256':sha('observed_trace_audit.py')}
    out=BASE/'observed_trace_v13_cpu_audit_20261003.json'
    if out.exists():raise FileExistsError('Refuse evidence overwrite')
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 2)
