"""Offline CPU endpoint IK only; no physics stepping or hardware IO."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
URDF=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot_sapien.urdf')
PRESET=Path(__file__).resolve().with_name('scene_preset.json')
preset=json.loads(PRESET.read_text());qmap=preset['joint_positions_rad']
assert hashlib.sha256(URDF.read_bytes()).hexdigest()==preset['robot_urdf_sha256']
r=ET.parse(URDF).getroot();joints={j.find('child').attrib['link']:j for j in r.findall('joint')}
root=np.eye(4);pose=preset['root_pose_xyz_quaternion_wxyz'];root[:3,3]=pose[:3];root[:3,:3]=Rot.from_quat([*pose[4:],pose[3]]).as_matrix()
def xyz(el,k,d='0 0 0'):return np.fromstring(el.attrib.get(k,d),sep=' ') if el is not None else np.zeros(3)
def prepare(side):
 names=[f'{side}_joint{i}' for i in range(1,7)];link=f'{side}_inspire_base_link';seq=[]
 while link in joints:
  j=joints[link];seq.append(j);link=j.find('parent').attrib['link']
 seq=seq[::-1];prepared=[]
 for j in seq:
  t=np.eye(4);o=j.find('origin');t[:3,3]=xyz(o,'xyz');t[:3,:3]=Rot.from_euler('xyz',xyz(o,'rpy')).as_matrix()
  prepared.append((j.attrib['name'],t,xyz(j.find('axis'),'xyz','1 0 0'),j.attrib['type']))
 bounds=np.array([[float(next(j for j in seq if j.attrib['name']==n).find('limit').attrib[k]) for n in names] for k in ['lower','upper']])
 def fk(q):
  d=dict(zip(names,q));t=root.copy()
  for n,o,a,typ in prepared:
   t=t@o
   if typ!='fixed':
    rr=np.eye(4);rr[:3,:3]=Rot.from_rotvec(a*d.get(n,qmap.get(n,0))).as_matrix();t=t@rr
  return t
 return names,bounds,fk
report={'date':'2026-10-22','scope':'Offline CPU nominal endpoint full-pose IK on actual URDF, no simulator stepping or hardware commands','source_preset':str(PRESET),'source_preset_sha256':hashlib.sha256(PRESET.read_bytes()).hexdigest(),'urdf':str(URDF),'urdf_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),'nominal_tcp_definition':'palm base link + 0.18m along palm local +Z; geometric assumption, uncalibrated','target_orientation':'retain full corresponding-arm palm-base rotation from saved default scene pose','position_tolerance_m':.003,'rotation_tolerance_rad':.03,'not_verified':['collision-free endpoint','continuous path','grasp/contact mechanics','real robot calibration','simultaneous dual-arm feasibility'],'targets':[]}
rng=np.random.default_rng(20261022)
for side,label,cat,mid,xyz_target in [('l','scale','072_electronicscale',0,[-.20,-.04,.82]),('r','plate','objaverse/plate',7,[.04,-.04,.79]),('r','bottle','080_pillbottle',4,[.25,-.04,.84])]:
 obj=next(o for o in preset['objects'] if o['category']==cat and o['model_id']==mid)
 assert np.max(np.abs(np.array(obj['center_xy_m'])-xyz_target[:2]))<1e-10,(label,obj)
 names,bnd,fk=prepare(side);q0=np.array([qmap[n] for n in names]);r0=fk(q0)[:3,:3];target=np.array(xyz_target)
 def residual(q):
  t=fk(q);return np.r_[t[:3,3]+.18*t[:3,2]-target,.2*Rot.from_matrix(r0.T@t[:3,:3]).as_rotvec()]
 best=None
 for i in range(16):
  seed=q0 if i==0 else np.clip(q0+rng.normal(0,.8,6),bnd[0]+1e-6,bnd[1]-1e-6)
  opt=least_squares(residual,seed,bounds=bnd,xtol=1e-11,ftol=1e-11,gtol=1e-11,max_nfev=250)
  t=fk(opt.x);achieved=t[:3,3]+.18*t[:3,2];pe=np.linalg.norm(achieved-target);re=np.linalg.norm(Rot.from_matrix(r0.T@t[:3,:3]).as_rotvec())
  quat=Rot.from_matrix(t[:3,:3]).as_quat();legal=bool(np.all(opt.x>=bnd[0]) and np.all(opt.x<=bnd[1]));success=bool(pe<.003 and re<.03 and legal)
  item={'object':label,'category':cat,'model_id':mid,'arm':side,'joint_names_in_solver_order':names,'tcp_target_world_xyz_m':target.tolist(),'target_rotation_matrix_world':r0.tolist(),'achieved_tcp_world_xyz_m':achieved.tolist(),'achieved_palm_quaternion_wxyz':[float(quat[3]),*quat[:3].tolist()],'palm_origin_world_m':t[:3,3].tolist(),'palm_normal_world':t[:3,1].tolist(),'finger_axis_world':t[:3,2].tolist(),'q_rad':dict(zip(names,opt.x.tolist())),'delta_from_default_deg':dict(zip(names,np.degrees(opt.x-q0).tolist())),'joint_limits_rad':{n:[float(lo),float(hi)] for n,lo,hi in zip(names,*bnd)},'joint_limits_pass':legal,'position_error_m':float(pe),'rotation_error_rad':float(re),'rotation_error_deg':float(np.degrees(re)),'endpoint_success':success,'seed_index':i}
  score=np.linalg.norm(residual(opt.x))
  if best is None or score<best[0]:best=(score,item)
  if success:break
 report['targets'].append(best[1]);print(label,best[1]['endpoint_success'],best[1]['position_error_m'],best[1]['rotation_error_deg'])
report['all_three_endpoint_full_pose_and_limits_pass']=all(x['endpoint_success'] for x in report['targets'])
Path(__file__).resolve().with_name('reach_check.json').write_text(json.dumps(report,indent=2))
