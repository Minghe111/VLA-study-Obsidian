"""Independent CPU RRT with explicit fixed-right holding-contact planning roles.

No physical collision filter or drive is edited. Allowances concern only the
frozen actual right-hand/body/cap contacts while planning the moving left arm.
"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from mplib import PlanningWorld
from mplib.pymp.planning.ompl import OMPLPlanner
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf

OUT=Path(__file__).resolve().parent;PIPE=OUT.parent;sys.path[:0]=[str(PIPE/'robot_cap_task')]
from cap_grasp_probe import Geometry,sha
from left_cap_held_probe import set_hand

folder=OUT/'left115_static_rrt_fast_v2';folder.mkdir(exist_ok=True)
snapshot=PIPE/'robot_cap_task/static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json'
rebind=OUT/'left115_static_rebind400/rebind.json';d=json.loads(rebind.read_text())
g=Geometry(folder,snapshot=snapshot,geometry_spec_path=PIPE/'robot_cap_task/profiles/bottle115/physics_spec115.json')
start=set_hand(g,g.q0,d['open_sdk6']);goal=np.asarray(d['open_q36']);arm=[g.index[f'l_joint{i}']for i in range(1,7)]
g.model.set_qpos(start,full=True);g.model.set_move_group('l_inspire_base_link');assert list(g.model.get_move_group_joint_indices())==arm
obstacles=[]
for obj in (g.body_obj,g.cap_obj,g.table_obj):
    shapes=[g.fcl.CollisionObject(x.get_collision_geometry())for x in obj.shapes]
    obstacles.append(g.fcl.FCLObject(obj.name,g.Pose(obj.pose),shapes,[g.Pose(x)for x in obj.shape_poses]))
world=PlanningWorld([g.model],obstacles);acm=world.get_allowed_collision_matrix()
for pair in g.allowed:
    a,b=sorted(pair);acm.set_entry(a,b,True)
observed=[]
for name in g.right_names:
    for label,obj in [('body',g.body_obj),('cap',g.cap_obj)]:
        if g.distance(g.objects[name],obj)[1]:
            acm.set_entry(name,obj.name,True);observed.append({'right_link':name,'held_object':obj.name,'role':'fixed existing actual holding contact, planning only'})
world.set_qpos_all(start[arm])
def events():
    return [{key:str(getattr(e,key))for key in ('link_name1','link_name2','object_name1','object_name2')if hasattr(e,key)}for e in world.check_collision()]
start_collisions=events();world.set_qpos_all(goal[arm]);goal_collisions=events();world.set_qpos_all(start[arm])
result={'scope':'Independent CPU geometric RRT candidate only','source_sha256':sha(__file__),'snapshot_sha256':sha(snapshot),'rebind_sha256':sha(rebind),
    'planned_arm':'left','fixed_hand_sdk6':d['open_sdk6'],'physical_filter_or_drive_changes':False,'planning_only_existing_right_contact_pairs':observed,
    'start_collisions':start_collisions,'goal_collisions':goal_collisions,'continuous_swept_certified':False,'physical_execution':False}
t0=time.time()
if start_collisions or goal_collisions:result['failure']='Planning world has remaining start/goal collisions; no relaxation'
else:
    status,path=OMPLPlanner(world).plan(start[arm],[goal[arm]],time=10.,range=.08,simplify=False,verbose=False)
    result['status']=str(status)
    if str(status).rstrip('.')!='Exact solution':result['failure']='Finite10s RRT failed, not global unreachability'
    else:
        path=np.asarray(path,dtype=float)
        # Preserve every returned waypoint; add exact connectors rather than
        # silently snapping/normalizing an endpoint or changing the curve.
        path=np.vstack([start[arm],path,goal[arm]])
        result['raw_arm6_waypoints']=path.tolist()
        (folder/'candidate_plan.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'stage':'raw_polyline_saved','rows':len(path),'elapsed_s':time.time()-t0}),flush=True)
        # Independently test <=1mm conservative vertex displacement samples.
        proxy=SimpleNamespace(urdf_tree=ET.fromstring(export_kinematic_chain_urdf(g.art)),vertices=g.vertices,sim_joints=g.names)
        coeff=g.fma.FullMeshPathAuditor._motion_coefficients(proxy)
        rows=[];failed=None;minimum={'body':10.,'right':10.,'table':10.,'cap':10.}
        for segment,(a,b)in enumerate(zip(path[:-1],path[1:])):
            delta=np.zeros(len(start));delta[arm]=np.abs(b-a);bound=max(float(c@delta)for c in coeff.values())
            count=max(1,int(np.ceil(bound/.001)))
            for fraction in np.linspace(0,1,count+1):
                if time.time()-t0>50:
                    failed={'reason':'Finite50s plan/resample budget exhausted','incomplete':True};break
                q=start.copy();q[arm]=a+(b-a)*fraction;world.set_qpos_all(q[arm]);hits=events();clear=not hits
                if not clear:
                    failed={'segment':segment,'fraction':float(fraction),'collisions':hits};break
                rows.append({'segment':segment,'fraction':float(fraction)})
            if failed:break
        result.update(dense_sample_count=len(rows),dense_audit_failure=failed,dense_discrete_clear=failed is None,minimum_sampled_gaps_m=minimum,
            interval_vertex_displacement_bound_m=.001,dense_audit_scope='Fresh dense evaluations of the native planning-world collision predicate; same FCL geometry/declared ACM. No left contacts allowed. This is not a separate implementation or continuous gap certificate.')
result['elapsed_s']=time.time()-t0
(folder/'candidate_plan.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k]for k in ['status','failure','dense_sample_count','dense_discrete_clear','minimum_sampled_gaps_m','elapsed_s']if k in result},indent=2))
