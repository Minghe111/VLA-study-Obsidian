from types import MethodType
from integrations.realman_rm65b_gen4.native_robot import RealManRobot

def install_hand_coupling(robot):
    entity=robot.left_entity
    records=[]
    for side in ('left','right'):
        for name,m in robot._manifest['hands'][side]['mimic'].items():
            child=entity.find_joint_by_name(name)
            master=entity.find_joint_by_name(m['joint'])
            assert child.parent_link == master.child_link
            a=float(m.get('multiplier',1)); b=float(m.get('offset',0))
            entity.create_fixed_tendon([child.parent_link.parent,child.parent_link,child.child_link],
                [0,-a,1],[0,-1/a,1],rest_length=b,stiffness=1e5,damping=0)
            child.set_drive_property(stiffness=0,damping=0,force_limit=0,mode='force')
            child.set_drive_velocity_target(0)
            records.append({'joint':name,'master':m['joint'],'multiplier':a,'offset':b})
    robot.hand_coupling_audit=records
    return records

def mimic_errors(robot):
    q=robot._measured_positions()
    return {n:float(q[n]-float(m.get('multiplier',1))*q[m['joint']]-float(m.get('offset',0)))
            for h in robot._manifest['hands'].values() for n,m in h['mimic'].items()}

OUT=BASE/'mimic_coupling_audit_v2_20261003';OUT.mkdir(exist_ok=False)
report={'result':'running','purpose':'Independent no-object hand coupling audit; not a task episode','hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected}}
task.scene.remove_entity(task.bottle.actor)
audit=[]
for mode in ['independent_pd','fixed_tendon_master_pd']:
    if mode=='fixed_tendon_master_pd':report['tendons']=install_hand_coupling(task.robot)
    for side in ('left','right'):
        for n in task.robot._manifest['hands'][side]['channels']:
            task.robot._hand_joints[n].set_drive_property(stiffness=.1,damping=.01,force_limit=10,mode='force')
    for pose in [[1000,1000,1000,1000,1000,0],[400,400,400,400,500,0],[1000,1000,1000,1000,1000,0]]:
        for side in ('left','right'):task.robot.set_hand_angle(side,pose)
        hold_current_targets(task,2.0)
        entry={'mode':mode,'sdk_command':pose,'sdk_measured':{s:task.robot.get_hand_sdk(s) for s in ['left','right']},'mimic_residual_rad':mimic_errors(task.robot)}
        audit.append(entry);print('MIMIC_AUDIT',json.dumps(entry),flush=True)
report.update(result='audit_complete',samples=audit,canonical_sources_unchanged=all(sha(p)==report['source_hashes'][str(p)] for p in protected));write()
original_initialize_hands=RealManRobot._initialize_hands

def initialize_coupled_hands(self):
    original_initialize_hands(self)
    install_hand_coupling(self)

RealManRobot._initialize_hands=initialize_coupled_hands