from pathlib import Path
import json,math,collections,hashlib
p=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task/left_contact_probes/left115_first_contact_v4_axisymmetric')
rows=json.loads((p/'independent_v4_raw_left_cap_manifolds.json').read_text());counts=collections.Counter();points=collections.Counter();maxraw=0.;maxtan=0.;normalerr=0.
for row in rows:
 link=next(x for x in row['body_names'] if x.startswith('l_inspire_'));counts[link]+=1;points[link]+=len(row['points'])
 for x in row['points']:
  maxraw=max(maxraw,math.sqrt(sum(z*z for z in x['impulse_raw_world_Ns'])));maxtan=max(maxtan,math.sqrt(sum(z*z for z in x['tangent_impulse_world_Ns'])));normalerr=max(normalerr,abs(math.sqrt(sum(z*z for z in x['normal_raw_world']))-1))
r=json.loads((p/'independent_v4_audit.json').read_text());r['raw_cap_left_contact'].update(manifold_count_per_link=dict(counts),point_count_per_link=dict(points),max_raw_impulse_vector_norm_Ns=maxraw,max_raw_tangent_impulse_vector_norm_Ns=maxtan,max_raw_normal_unit_error=normalerr);r['raw_contact_summary_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(p/'independent_v4_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('Saved reproducible raw contact per-link/vector summary.')
