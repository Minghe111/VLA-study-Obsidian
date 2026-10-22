"""Bind intrinsic left cap pinches to an ACTUAL held-state snapshot.

CPU FK/IK/FCL only. No physics step or robot execution. A failed-run snapshot
remains a failed-run snapshot; it supplies a measured geometric configuration.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from cap_grasp_probe import Geometry,matrix,sha
from left_cap_intrinsic import canonical_transform

ROOT=Path(__file__).resolve().parents[1]


def set_hand(g,q,sdk):
    q=np.asarray(q).copy()
    for n,v in g.maps['left'].state(sdk)[0].items():q[g.index[n]]=v
    return q


def shallow_contact(g):
    detail=g.cap_contact_depths();side={}
    for key,contacts in detail['groups'].items():
        side[key]=[x for x in contacts if abs(float(np.asarray(x['world_normal'])@g.cap_matrix[:3,2]))<.5]
    dot=min((float(np.dot(a['world_normal'],b['world_normal']))for a in side['thumb']for b in side['index']),default=1.)
    detail['side_contacts']=side;detail['most_opposed_side_normal_dot']=dot
    detail['shallow_opposed_side_contacts']=bool(side['thumb']and side['index']and dot<-.5 and detail['max_depth_m']<=.0015)
    return detail


def solve(g,target,base,mask,palm_index,seed):
    inverse=np.linalg.inv(matrix(g.art.root_pose));limits=np.asarray(g.art.get_qlimits())
    arm=np.flatnonzero(~mask);rng=np.random.default_rng(seed)
    starts=[base.copy()]
    for _ in range(3):
        s=base.copy();s[arm]=rng.uniform(-2.,2.,6);starts.append(s)
    last=None
    for start in starts:
        q,ok,err=g.pk.compute_IK_CLIK(palm_index,g.Pose(inverse@target),start,mask.tolist(),eps=1e-6,max_iter=1500,dt=.1,damp=1e-6)
        q=np.asarray(q);legal=bool(np.all(q[arm]>=limits[arm,0]-1e-6)and np.all(q[arm]<=limits[arm,1]+1e-6))
        last={'solver_success':bool(ok),'arm_limits_valid':legal,'error_norm':float(np.linalg.norm(err))}
        if ok and legal:return q,last
    return None,last


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--intrinsic',type=Path,default=ROOT/'cap_geometry_probes/left_intrinsic_exposed_final/intrinsic_candidates.json')
    p.add_argument('--spec',type=Path,default=ROOT/'robot_cap_task/profiles/bottle115/physics_spec115.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wide-roll',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);t0=time.time()
    g=Geometry(args.output,snapshot=args.snapshot,geometry_spec_path=args.spec)
    original_cap=g.cap_matrix.copy();intrinsic=json.loads(args.intrinsic.read_text());records=[];shallow=[]
    mask=np.ones(len(g.names),dtype=bool);mask[[g.index[f'l_joint{i}']for i in range(1,7)]]=False
    palm_index=list(g.pk.get_link_names()).index('l_inspire_base_link')
    poses=[]
    for pattern in intrinsic:
        rolls=pattern['preferred_roll_family']
        if args.wide_roll:
            rolls=[{'jaw_line_roll_deg':r,'T_cap_from_palm':canonical_transform(pattern['thumb']['surface_in_palm_m'],pattern['index']['surface_in_palm_m'],r).tolist()}for r in range(-60,61,15)]
        for yaw,ref in itertools.product(range(0,360,30),rolls):poses.append((pattern,yaw,ref))
    for pattern,yaw,ref in poses:
        yaw_T=np.eye(4);yaw_T[:3,:3]=Rotation.from_rotvec([0,0,math.radians(yaw)]).as_matrix()
        cap_to_palm=yaw_T@np.asarray(ref['T_cap_from_palm']);target=original_cap@cap_to_palm
        base=set_hand(g,g.q0,pattern['left_sdk6']);q,ik=solve(g,target,base,mask,palm_index,len(records)+20261022)
        row={'id':len(records),'intrinsic_id':pattern['candidate_id'],'cap_yaw_deg':yaw,'jaw_line_roll_deg':ref['jaw_line_roll_deg'],
            'left_sdk6':pattern['left_sdk6'],'palm_target_matrix':target.tolist(),'T_cap_from_palm':cap_to_palm.tolist(),'ik':ik,'ik_solved':q is not None}
        if q is not None:
            row['baseline_collision']=g.screen(q)
            if row['baseline_collision']['collision_free_except_intended_cap_contact']:
                row['pinch_refinements']=[]
                for index_delta,thumb_delta in itertools.product((-10.,-5.,0.,5.,10.),(-10.,-5.,0.,5.,10.)):
                    sdk=np.asarray(pattern['left_sdk6'],dtype=float);sdk[3]+=index_delta;sdk[4]+=thumb_delta
                    if np.any(sdk<0)or np.any(sdk>1000):continue
                    v=set_hand(g,q,sdk);audit=g.screen(v);candidate={'sdk6':sdk.tolist(),'collision':audit}
                    if audit['collision_free_except_intended_cap_contact']:
                        candidate['contact']=shallow_contact(g)
                        if candidate['contact']['shallow_opposed_side_contacts']:
                            candidate['q36']=v.tolist();candidate['sdk24']=g.sdk24(v);shallow.append({'endpoint_id':row['id'],'candidate':candidate,'T_cap_from_palm':cap_to_palm.tolist()})
                    row['pinch_refinements'].append(candidate)
        records.append(row)
    # If, and only if, a genuine shallow static contact candidate exists, check
    # one small geometric screw arc. This is a reference trajectory under the
    # explicit maintained-contact hypothesis, not a permanent attachment.
    arcs=[]
    for item in shallow[:1]:
        base=np.asarray(item['candidate']['q36']);cap_to_palm=np.asarray(item['T_cap_from_palm'])
        relative=np.linalg.inv(g.bottle_matrix)@original_cap;arc=[]
        for angle in (0.,3.,6.,9.,12.,15.):
            d=np.eye(4);d[:3,:3]=Rotation.from_rotvec([0,0,math.radians(angle)]).as_matrix()
            d[2,3]=g.spec['thread_guide']['pitch_m_per_turn']*angle/360.
            cap=g.bottle_matrix@d@relative;target=cap@cap_to_palm
            q,ik=solve(g,target,base,mask,palm_index,4400+int(angle));row={'angle_deg':angle,'ik':ik,'ik_solved':q is not None}
            if q is not None:
                g.cap_matrix=cap;g.cap_obj=g.fcl.FCLObject(g.cap_obj.name,g.Pose(cap),g.cap_obj.shapes,g.cap_obj.shape_poses)
                row['collision']=g.screen(q);row['contact']=shallow_contact(g);row['sdk24_geometric_reference']=g.sdk24(q);base=q
            arc.append(row)
        arcs.append({'source_endpoint':item['endpoint_id'],'scope':'15degree static sampled screw-reference arc assuming contact maintained; not dynamics, no success claim','samples':arc})
    summary={**g.receipt,'intrinsic_candidates_sha256':sha(args.intrinsic),'probe_source_sha256':sha(__file__),
        'snapshot_scope':json.loads(args.snapshot.read_text()).get('scope'),'endpoint_count':len(records),'ik_solutions':sum(x['ik_solved']for x in records),
        'baseline_collision_free_count':sum(x.get('baseline_collision',{}).get('collision_free_except_intended_cap_contact',False)for x in records),
        'shallow_opposed_contact_count':len(shallow),'shallow_endpoint_ids':sorted(set(x['endpoint_id']for x in shallow)),
        'arc_probes':len(arcs),'elapsed_s':time.time()-t0,'physics_steps':0,'executed':False,
        'boundary':'Endpoint/static sampled path only; actual cap pinch, multi-regrasp unscrewing, body hold stability and pouring remain unverified.'}
    (args.output/'endpoints.json').write_text(json.dumps(records,indent=2));(args.output/'shallow_candidates.json').write_text(json.dumps(shallow,indent=2))
    (args.output/'arc_probes.json').write_text(json.dumps(arcs,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
