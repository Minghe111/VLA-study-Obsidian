"""Static small right-index opening diagnostic; no load-bearing claim."""
import json
from pathlib import Path
import sys
import numpy as np

OUT=Path(__file__).resolve().parent;PIPE=OUT.parent;sys.path[:0]=[str(PIPE/'robot_cap_task')]
from cap_grasp_probe import Geometry,sha

snapshot=PIPE/'robot_cap_task/static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json'
folder=OUT/'right_index_clearance_static';folder.mkdir(exist_ok=True)
g=Geometry(folder,snapshot=snapshot,geometry_spec_path=PIPE/'robot_cap_task/profiles/bottle115/physics_spec115.json')
mp=g.maps['right'];primary=mp.channels[3];width=mp.limits[primary]['upper']-mp.limits[primary]['lower'];records=[]
for sdk_delta in (0,5,10,15,20):
    q=g.q0.copy();dq=-width*sdk_delta/1000.;q[g.index[primary]]+=dq
    # Preserve all observed hand deflections, applying only the small index
    # primary/follower increment. Do not replace the whole actual hand state by
    # an ideal mimic command or modify middle/thumb to manufacture clearance.
    increments={primary:dq}
    for n,parent,a,b in mp.mimic_order:
        if parent in increments:increments[n]=a*increments[parent];q[g.index[n]]+=increments[n]
    g.model.set_qpos(q,full=True);by_object={}
    for category,obj in [('cap',g.cap_obj),('body',g.body_obj),('table',g.table_obj)]:
        rows=[{'link':n,'gap_m':g.distance(o,obj)[0],'collision':g.distance(o,obj)[1]}for n,o in g.objects.items()if n.startswith('r_')and'index'in n]
        by_object[category]=rows
    records.append({'index_opening_delta_sdk':sdk_delta,'changed_joint_deltas_rad':increments,'geometry':by_object,
        'cap_collision':any(x['collision']for x in by_object['cap']),'minimum_cap_gap_m':min(x['gap_m']for x in by_object['cap'])})
result={'scope':'Actual static q36 plus small measured-deflection-preserving index increments; no physical execution','snapshot_sha256':sha(snapshot),'source_sha256':sha(__file__),
    'records':records,'right_middle_thumb_unchanged':True,'body_cap_poses_held_fixed_in_clone':True,
    'physical_load_bearing_or_hold_stability_verified':False,'boundary':'Kinematic index release may unload or perturb body/cap in real dynamics; root must decide any actual probe.'}
(folder/'index_scan.json').write_text(json.dumps(result,indent=2));print([(x['index_opening_delta_sdk'],x['cap_collision'],x['minimum_cap_gap_m'])for x in records])
