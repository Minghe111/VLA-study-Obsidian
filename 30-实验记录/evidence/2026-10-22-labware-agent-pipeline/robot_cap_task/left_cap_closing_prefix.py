"""Finite independent thumb/index closing prefixes using full collision links.

Sensor-only and horizontal-normal restrictions are NOT acceptance criteria.
The first actual FCL parent/sensor contact stops each geometric finger sweep.
No dynamics, attachment, external wrench, or robot success is claimed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cap_grasp_probe import Geometry,sha,matrix
from left_cap_held_probe import set_hand,solve

ROOT=Path(__file__).resolve().parents[1]


def first_contact_bracket(g,base,before,after,channel,group):
    """Bound first intersection by actual mesh motion, not EPA depth output.

    The bound relies on the FCL collision predicate. It is not a contact-force
    prediction or an independent proof that FCL's narrow-phase is exact.
    """
    free=float(before[channel]);hit=float(after[channel])
    for _ in range(10):
        value=.5*(free+hit);sdk=before.copy();sdk[channel]=value
        g.model.set_qpos(set_hand(g,base,sdk),full=True);detail=g.cap_contact_depths()
        if detail['groups'][group]:hit=value
        else:free=value
    qs=[];positions=[]
    for value in (free,hit):
        sdk=before.copy();sdk[channel]=value;q=set_hand(g,base,sdk);qs.append((sdk,q));g.model.set_qpos(q,full=True)
        points={}
        for name in g.left_names:
            if group not in name:continue
            T=matrix(g.objects[name].pose);points[name]=g.vertices[name]@T[:3,:3].T+T[:3,3]
        positions.append(points)
    upper=max(float(np.linalg.norm(positions[1][n]-positions[0][n],axis=1).max())for n in positions[0])
    detail=g.cap_contact_depths();audit=g.screen(qs[1][1])
    return qs[1][0],qs[1][1],audit,detail,{'free_sdk':free,'contact_sdk':hit,'channel_index':channel,
        'maximum_moving_group_vertex_displacement_m':upper,'basis':'previous bracket endpoint has no contact for this finger; cap held fixed in geometric clone',
        'FCL_EPA_depth_not_used_as_penetration_measure':True,'predicate_boundary_not_PhysX_validation':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoints',type=Path,default=ROOT/'cap_geometry_probes/left115_held_v2_wide_roll/endpoints.json');p.add_argument('--max-palms',type=int,default=6)
    p.add_argument('--bracket-contact',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);t0=time.time()
    g=Geometry(args.output,snapshot=args.snapshot,geometry_spec_path=ROOT/'robot_cap_task/profiles/bottle115/physics_spec115.json')
    endpoints=json.loads(args.endpoints.read_text());pool=[r for r in endpoints if r.get('baseline_collision',{}).get('collision_free_except_intended_cap_contact')]
    pool.sort(key=lambda r:-r['baseline_collision']['minimum_gaps_m']['left_right']);chosen=[]
    for row in pool:
        if any(x['intrinsic_id']==row['intrinsic_id']and x['cap_yaw_deg']==row['cap_yaw_deg']for x in chosen):continue
        chosen.append(row)
        if len(chosen)>=args.max_palms:break
    mask=np.ones(len(g.names),dtype=bool);mask[[g.index[f'l_joint{i}']for i in range(1,7)]]=False
    palm_index=list(g.pk.get_link_names()).index('l_inspire_base_link');trials=[];valid=[]
    for row in chosen:
        for start_index in (1000.,900.,800.):
            for order in ('index_then_thumb_opposition','thumb_opposition_then_index'):
                sdk=np.array([0.,0.,0.,start_index,1000.,1000.]);base=set_hand(g,g.q0,sdk)
                q,ik=solve(g,np.asarray(row['palm_target_matrix']),base,mask,palm_index,row['id']+3300)
                trial={'id':len(trials),'endpoint_id':row['id'],'order':order,'open_sdk6':sdk.tolist(),'palm_target_matrix':row['palm_target_matrix'],'ik':ik,'samples':[]}
                if q is None:trial['failure']='IK failed';trials.append(trial);continue
                current=q.copy();audit=g.screen(current);detail=g.cap_contact_depths()
                trial['open_collision']=audit;trial['open_contacts']=detail
                if not audit['collision_free_except_intended_cap_contact']or any(detail['groups'].values()):
                    trial['failure']='Open/prebent pose already has prohibited collision or cap contact';trials.append(trial);continue
                sequence=[('index',3,np.arange(start_index-10,449,-10.)),('thumb',5,np.arange(980,-1,-20.))]
                if order.startswith('thumb'):sequence.reverse()
                stopped={};failure=None
                for group,channel,values in sequence:
                    first=False
                    for value in values:
                        previous_sdk=sdk.copy()
                        sdk[channel]=value;current=set_hand(g,q,sdk);audit=g.screen(current);detail=g.cap_contact_depths()
                        sample={'moved_group':group,'sdk6':sdk.tolist(),'collision':audit,'cap_contacts':detail}
                        trial['samples'].append(sample)
                        if not audit['collision_free_except_intended_cap_contact']:
                            failure='Prohibited collision during independent closure';break
                        if not args.bracket_contact and detail['max_depth_m']>.0015:
                            failure='First-contact geometry exceeds1.5mm penetration bound';break
                        if detail['groups'][group]:
                            bracket=None
                            if args.bracket_contact:
                                sdk,current,audit,detail,bracket=first_contact_bracket(g,q,previous_sdk,sdk,channel,group)
                                if not audit['collision_free_except_intended_cap_contact']or bracket['maximum_moving_group_vertex_displacement_m']>.00005:
                                    failure='Refined first-contact bracket violates forbidden geometry or50um motion bound';break
                            stopped[group]={'sample_index':len(trial['samples'])-1,'sdk6':sdk.tolist(),'contacts':detail['groups'][group],'first_contact_bracket':bracket};first=True;break
                    if failure:break
                    if not first:failure=f'No {group} contact in finite actuator range';break
                trial['first_contacts']=stopped
                if failure:trial['failure']=failure
                elif set(stopped)=={'thumb','index'}:
                    detail=g.cap_contact_depths();dot=min((float(np.dot(a['world_normal'],b['world_normal']))for a in detail['groups']['thumb']for b in detail['groups']['index']),default=1.)
                    trial['most_opposed_full_link_normal_dot']=dot
                    if dot<-.5 or args.bracket_contact:
                        trial['first_contact_probe_proposal']=True;trial['q36']=current.tolist();trial['sdk24']=g.sdk24(current);trial['closed_sdk6']=sdk.tolist();valid.append(trial['id'])
                        trial['FCL_normal_antipodality_certified']=bool(dot<-.5)
                        trial['scope']='First-contact proposal for low-speed force-controlled actual PhysX probe; no force-closure claim'
                    else:trial['failure']='Both first contacts exist but normals do not oppose'
                trials.append(trial)
    summary={**g.receipt,'source_endpoints':str(args.endpoints),'source_endpoints_sha256':sha(args.endpoints),'source_sha256':sha(__file__),
        'selected_endpoint_ids':[x['id']for x in chosen],'trial_count':len(trials),'candidate_trial_ids':valid,'elapsed_s':time.time()-t0,
        'criterion':('Full actual thumb/index parents and sensors; first contact binary bracket <=50um mesh motion, no prohibited collision; EPA depth and normal antipodality uncalibrated'if args.bracket_contact else 'Full actual thumb/index parents and sensors; first contact stop, <=1.5mm penetration, opposed normals dot<-.5, no prohibited collision; no horizontal normal constraint'),
        'physical_execution':False,'arc_tested':False,'boundary':'Geometry-only first-contact prefixes do not certify contact force, frictional force closure, approach, grasp stability or opening.'}
    (args.output/'closing_trials.json').write_text(json.dumps(trials,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
