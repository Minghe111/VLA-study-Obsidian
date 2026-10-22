"""Rebind the single left first-touch proposal to current actual held geometry.

No physical stepping. Right cap contact already present in the snapshot is
recorded, not reclassified as a new left-hand prohibition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np

from cap_grasp_probe import Geometry,sha
from left_cap_held_probe import set_hand,solve
from left_cap_closing_prefix import first_contact_bracket

ROOT=Path(__file__).resolve().parents[1]


def evaluate(g,q):
    audit=g.screen(q);contacts=g.cap_contact_depths()
    return {'collision':audit,'cap_contacts':contacts,'cap_free':not any(contacts['groups'].values()),
        'clear':audit['collision_free_except_intended_cap_contact']and not any(contacts['groups'].values())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--proposal',type=Path,default=ROOT/'cap_geometry_probes/left_cap_first_contact_probe_candidate.json')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);t0=time.time();proposal=json.loads(args.proposal.read_text())
    g=Geometry(args.output,snapshot=args.snapshot,geometry_spec_path=ROOT/'robot_cap_task/profiles/bottle115/physics_spec115.json')
    target=g.cap_matrix@np.asarray(proposal['T_cap_from_palm']);mask=np.ones(len(g.names),dtype=bool);arm=[g.index[f'l_joint{i}']for i in range(1,7)];mask[arm]=False
    palm_index=list(g.pk.get_link_names()).index('l_inspire_base_link');sdk=np.array([0.,0.,0.,1000.,1000.,400.]);base=set_hand(g,g.q0,sdk)
    q,ik=solve(g,target,base,mask,palm_index,61022)
    result={**g.receipt,'scope':'Static CPU rebind plus discrete approach/first-contact audit; no physical execution','snapshot_scope':json.loads(args.snapshot.read_text()).get('scope'),
        'proposal_sha256':sha(args.proposal),'source_sha256':sha(__file__),'palm_target_world_matrix':target.tolist(),'T_cap_from_palm':proposal['T_cap_from_palm'],'open_sdk6':sdk.tolist(),'ik':ik}
    if q is None:result['failure']='Rebound endpoint IK failed'
    else:
        result['open_sdk24_geometric_target']=g.sdk24(q);result['open_q36']=q.tolist();result['open_audit']=evaluate(g,q)
        if result['open_audit']['clear']:
            # Check folding/setting opposition while the left arm stays at its
            # actual starting pose, then a finite straight joint-line approach.
            home_prepared=set_hand(g,g.q0,sdk);pre=[]
            for fraction in np.linspace(0,1,41):
                v=g.q0+(home_prepared-g.q0)*fraction;check=evaluate(g,v)
                pre.append({'fraction':float(fraction),'clear':check['clear'],'collisions':check['collision']['collisions'],'cap_free':check['cap_free']})
                if not check['clear']:break
            result['prepare_at_home_samples']=pre
            count=max(2,int(np.ceil(np.max(np.abs(q[arm]-home_prepared[arm]))/.008)))
            path=[]
            for fraction in np.linspace(0,1,count+1):
                v=home_prepared+(q-home_prepared)*fraction;check=evaluate(g,v)
                path.append({'fraction':float(fraction),'clear':check['clear'],'collisions':check['collision']['collisions'],'cap_free':check['cap_free']})
                if not check['clear']:break
            result['joint_line_approach_samples']=path;result['joint_line_approach_discrete_clear']=bool(path[-1]['fraction']==1 and all(x['clear']for x in path))
            result['approach_continuously_certified']=False
            current=q.copy();closing=[];first={};failure=None
            for group,channel,values in [('index',3,np.arange(990,449,-10.)),('thumb',5,np.arange(399,-1,-1.))]:
                contacted=False
                for value in values:
                    previous=sdk.copy();sdk[channel]=value;current=set_hand(g,q,sdk);check=evaluate(g,current)
                    # Refine a cap first-touch bracket before evaluating a
                    # potential forbidden collision farther into that small step.
                    if check['cap_contacts']['groups'][group]:
                        sdk,current,audit,detail,bracket=first_contact_bracket(g,q,previous,sdk,channel,group)
                        if not audit['collision_free_except_intended_cap_contact']:
                            failure='Refined cap first touch still has prohibited geometry';break
                        first[group]={'sdk6':sdk.tolist(),'first_contact_bracket':bracket,'contacts':detail['groups'][group],'forbidden_geometry_gaps_m':audit['minimum_gaps_m']};contacted=True;break
                    closing.append({'moved_group':group,'sdk6':sdk.tolist(),'clear':check['collision']['collision_free_except_intended_cap_contact'],'collisions':check['collision']['collisions']})
                    if not check['collision']['collision_free_except_intended_cap_contact']:
                        failure='Prohibited geometry before cap first touch';break
                if failure:break
                if not contacted:failure='No finite first contact for '+group;break
            result['closing_trace']=closing;result['first_contacts']=first
            if failure:result['closure_failure']=failure
            elif len(first)==2:
                result['first_contact_probe_proposal']=True;result['first_contact_sdk6']=sdk.tolist();result['first_contact_sdk24']=g.sdk24(current);result['first_contact_q36']=current.tolist()
        else:result['failure']='Thumb opposition400 open pose is not clear'
    result['elapsed_s']=time.time()-t0;result['physical_grasp_or_force_closure_verified']=False
    (args.output/'rebind.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k in ['failure','closure_failure','open_sdk6','ik','joint_line_approach_discrete_clear','first_contact_sdk6','elapsed_s','first_contact_probe_proposal']},indent=2))


if __name__=='__main__':main()
