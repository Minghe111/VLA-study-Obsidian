"""Additional private zero-step fixture with nonidentity original aux base/q."""
from pathlib import Path
import hashlib
work=Path(__file__).resolve().parent
base=work/'waypoint_contract_check_v4.py'
source=base.read_text(encoding='utf-8')
old="self.q=np.zeros(36);self.pose=Pose();self.group=['original_ee']"
new="self.q=np.full(36,.123);m=np.eye(4);m[:3,3]=[.7,-.4,.9];self.pose=Pose(m);self.group=['original_ee']"
assert source.count(old)==1;source=source.replace(old,new)
old="assert np.array_equal(model.q,np.zeros(36)) and np.array_equal(model.pose.m,np.eye(4))"
new="assert np.array_equal(model.q,np.full(36,.123));original=np.eye(4);original[:3,3]=[.7,-.4,.9];assert np.array_equal(model.pose.m,original)"
assert source.count(old)==1;source=source.replace(old,new)
source=source.replace('waypoint_contract_receipt_v4.json','auxiliary_setup_restoration_nontrivial_v1.json')
source=source.replace("'physical_validation_completed':False,", "'physical_validation_completed':False,\n    'original_auxiliary_q_nonzero_and_base_nonidentity':True,\n    'private_fixture_source_sha256':"+repr(hashlib.sha256(base.read_bytes()).hexdigest())+",")
exec(compile(source,str(Path(__file__).resolve())+':private_nontrivial_fixture','exec'),
     {'__file__':str(Path(__file__).resolve()),'__name__':'__main__'})
