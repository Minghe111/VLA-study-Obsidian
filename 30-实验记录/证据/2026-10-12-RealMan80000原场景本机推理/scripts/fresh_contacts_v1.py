"""Read-only actual normal-contact records. No expert trace reader."""
import numpy as np

def point_record(point):
    normal=np.asarray(point.normal,dtype=float);impulse=np.asarray(point.impulse,dtype=float)
    if not np.isfinite(normal).all() or not np.isfinite(impulse).all():raise ValueError('Nonfinite actual contact')
    return {'normal_impulse_Ns':abs(float(normal@impulse)),'normal_world':normal.tolist(),'impulse_Ns':impulse.tolist(),'point_world_m':np.asarray(point.position,dtype=float).tolist(),'separation_m':float(point.separation) if getattr(point,'separation',None) is not None else None}

def collect_actual_self_contacts(task,robot_link_names,allowed,*,normal_impulse_threshold_Ns=1e-7):
    rows=[]
    for c in task.scene.get_contacts():
        names=[b.entity.name for b in c.bodies]
        if len(names)!=2 or not all(n in robot_link_names for n in names) or frozenset(names) in allowed:continue
        points=[point_record(p) for p in c.points];value=sum(p['normal_impulse_Ns'] for p in points)
        rows.append({'links':names,'normal_impulse_Ns':value,'active':value>normal_impulse_threshold_Ns,'points':points})
    return rows

def collect_actual_bottle_contacts(task):
    name=task.bottle.actor.name;rows=[]
    for c in task.scene.get_contacts():
        names=[b.entity.name for b in c.bodies]
        if name not in names:continue
        points=[point_record(p) for p in c.points]
        rows.append({'other_link':names[1] if names[0]==name else names[0],'normal_impulse_Ns':sum(p['normal_impulse_Ns'] for p in points),'points':points})
    return rows

def contacts_and_grip(task):
    dt=float(task.scene.get_timestep());prefix='l_' if task.qpose_tag==0 else 'r_';forces=np.zeros(5);norms={'thumb':np.zeros(3),'fingers':np.zeros(3)};raw=[];lp=[];table=0.
    for c in task.scene.get_contacts():
        names=[b.entity.name for b in c.bodies]
        if task.bottle.actor.name not in names:continue
        bottle0=names[0]==task.bottle.actor.name;other=names[1] if bottle0 else names[0]
        for p in c.points:
            n=np.asarray(p.normal,dtype=float)*(-1 if bottle0 else 1);imp=np.asarray(p.impulse,dtype=float)
            force=abs(float(imp@n))/dt
            if 'table' in other.lower():table+=force
            if not other.startswith(prefix+'inspire_'):continue
            channel=next((v for v in ('little','ring','middle','index','thumb') if v in other),None)
            if channel is None:continue
            forces[('little','ring','middle','index','thumb').index(channel)]+=force
            norms['thumb' if channel=='thumb' else 'fingers']+=n*force
            raw.append({'link':other,'channel':channel,'normal_force_N':force,'normal_bottle_to_hand':n.tolist(),'impulse_Ns':imp.tolist(),'point_world_m':np.asarray(p.position).tolist()})
            if channel in ('middle','index','thumb') and force>=1e-5:lp.append({'channel':channel,'point_world_m':np.asarray(p.position).tolist(),'force_on_bottle_normal_world':(-n).tolist()})
    a,b=norms['thumb'],norms['fingers'];scale=np.linalg.norm(a)*np.linalg.norm(b)
    palm=task.robot.left_entity.find_link_by_name(prefix+'inspire_base_link').get_pose().to_transformation_matrix();bottle=task.bottle.get_pose().to_transformation_matrix()
    return {'forces_N':forces.tolist(),'thumb_normal_N':float(forces[4]),'fingers_normal_N':float(forces[:4].sum()),'opposing_normals_dot':float(a@b/scale) if scale>1e-12 else 1.,'table_normal_N':table,'relative_bottle_palm':(np.linalg.inv(palm)@bottle).tolist(),'contacts':raw,'lp_contacts':lp}

def real_grip(grip):
    return bool(grip['thumb_normal_N']>.05 and grip['fingers_normal_N']>.05 and grip['opposing_normals_dot']<-.3)
