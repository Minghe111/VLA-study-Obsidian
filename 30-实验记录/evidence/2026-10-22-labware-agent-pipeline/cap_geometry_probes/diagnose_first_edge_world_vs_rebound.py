"""Bounded actual-q36 diagnostic, no search: original vs cap-rebound edge.

Only the existing path is queried. The physical scene is never stepped or
pose-written. Native create_table supplies all five exact collision boxes.
"""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import sapien
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf

ROOT=Path(__file__).resolve().parent;PIPE=ROOT.parent;TASK=PIPE/'robot_cap_task'
sys.path[:0]=[str(TASK),str(ROOT)]
from cap_grasp_probe import Geometry,sha
from left_cap_held_probe import set_hand
from audit_pour_full_table import SOURCE,SceneProxy

RUN=TASK/'left_contact_probes/left115_first_contact_v2'
SNAPSHOT=RUN/'actual_preapproach_snapshot.json'
REBINDINGS=RUN/'actual_cap_rebindings.json'
PLAN=ROOT/'left115_static_shortcut_v1/candidate_plan.json'
OUT=ROOT/'first_edge_world_vs_rebound_v1'

def main():
    OUT.mkdir(exist_ok=False)
    g=Geometry(OUT,snapshot=SNAPSHOT,geometry_spec_path=TASK/'profiles/bottle115/physics_spec115.json')
    prepared=g.q0.copy();old=json.loads(PLAN.read_text());path=np.asarray(old['raw_arm6_waypoints'])
    rebound=json.loads(REBINDINGS.read_text())[0]
    arm=[g.index[f'l_joint{i}']for i in range(1,7)]
    nominal=set_hand(g,prepared,old['fixed_left_hand_sdk6'])
    function=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='create_table')
    ns={'sapien':sapien,'preprocess':lambda scene,p:(scene,p)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(SOURCE),'exec'),ns)
    table=ns['create_table'](SceneProxy(g.scene),sapien.Pose([0,0,.74]),length=1.2,width=.7,height=.74,thickness=.05,is_static=True,texture_id=None)
    component=table.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    fulltable=g.convert(component)
    proxy=SimpleNamespace(urdf_tree=ET.fromstring(export_kinematic_chain_urdf(g.art)),vertices=g.vertices,sim_joints=g.names)
    coeff=g.fma.FullMeshPathAuditor._motion_coefficients(proxy)
    left=[n for n in g.objects if n.startswith('l_')]
    def check(label,a,b):
        fulla=prepared.copy();fulla[arm]=a
        fullb=prepared.copy();fullb[arm]=b
        delta=np.abs(fullb-fulla)
        count=max(1,int(np.ceil(max(float(c@delta)for c in coeff.values())/.001)))
        first=None;minimum=10.;nearest=None
        for k in range(count+1):
            fraction=k/count;q=fulla+(fullb-fulla)*fraction
            g.model.set_qpos(q,full=True)
            for name in left:
                gap,hit=g.distance(g.objects[name],fulltable)
                if gap<minimum:minimum=gap;nearest={'fraction':fraction,'link':name,'gap_m':gap}
                if hit:
                    # Also compare the earlier one-box proxy at this same q.
                    proxygap,proxyhit=g.distance(g.objects[name],g.table_obj)
                    first={'sample':k,'sample_total':count,'fraction':fraction,'link':name,
                           'fulltable_collision':True,'original_proxy_collision':proxyhit,
                           'proxy_gap_m':proxygap,'q36':q.tolist()};break
            if first:break
        result={'label':label,'clear_all_left_vs_fulltable':first is None,
            'sample_total':count,'first_collision':first,'minimum_sampled_gap_m':minimum,
            'nearest':nearest,'interval_vertex_bound_m':.001,
            'frozen_hand':'Actual measured prepared q36, no SDK/mimic reconstruction',
            'continuous_clearance_certified':False}
        print(json.dumps({k:result[k]for k in['label','clear_all_left_vs_fulltable','sample_total','minimum_sampled_gap_m','first_collision']}),flush=True)
        return result
    first=check('original world waypoint53, actual measured prepared start',prepared[arm],path[1])
    second=check('actual cap-follow rebound of waypoint53',prepared[arm],np.asarray(rebound['left_result6']))
    existing=[]
    # Conditional finite existing-edge checks only: no new endpoints/search.
    if first['clear_all_left_vs_fulltable']:
        for i in range(1,len(path)-1):
            existing.append(check(f'existing original world segment{i}',path[i],path[i+1]))
            if not existing[-1]['clear_all_left_vs_fulltable']:break
    snapshot=json.loads(SNAPSHOT.read_text())
    original_snapshot=json.loads(Path(old['snapshot']).read_text())
    result={'scope':'Existing world path vs whole-cap-follow rebound diagnostic; actual prepared q36 and source-faithful full five-piece table. No new search or physical execution.',
        'source_sha256':sha(__file__),'snapshot_sha256':sha(SNAPSHOT),'plan_sha256':sha(PLAN),
        'rebindings_sha256':sha(REBINDINGS),'original_native_table_source_sha256':sha(SOURCE),
        'fulltable_parts':len(component.collision_shapes),'scene_steps':0,
        'nominal_vs_actual_prepared_left_hand_max_delta_rad':float(np.max(np.abs(nominal-prepared)[[i for i,n in enumerate(g.names)if n.startswith('l_inspire')]])),
        'original_world_start_vs_actual_left_arm_max_delta_rad':float(np.max(np.abs(path[0]-prepared[arm]))),
        'rebound_vs_original_via0_arm_delta_rad':(np.asarray(rebound['left_result6'])-path[1]).tolist(),
        'old_cap_world':original_snapshot['cap_matrix'],'current_cap_world':snapshot['cap_matrix'],
        'first_edge_original_world':first,'first_edge_cap_follow_rebound':second,
        'remaining_existing_world_edges_table_queries':existing,
        'whole_original_world_path_table_clear':bool(first['clear_all_left_vs_fulltable']and len(existing)==len(path)-2 and all(x['clear_all_left_vs_fulltable']for x in existing)),
        'planned_left_hand_original_was_nominal_open_sdk':True,
        'original_arm_and_right_reference_actual_q36':True,
        'original_table_proxy_scope':'one tabletop box half[1,.35,.02] at[0,0,.72]; full native table top half[.6,.35,.025] at[0,0,.715] plus four legs',
        'limitation':'This diagnostic queries all moving left links vs full table only. It is not a renewed full body/cap/pills/self/path certificate. The task actual-scene full auditor must validate world transit and the last actual-cap approach separately. Actual cap drift and maintained grasp still require physical guards.',
    }
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2))
    (OUT/'diagnosis.md').write_text('# 原world途径与整体随盖重绑的首边对照\n\n'
        f'原world首边桌面通过：{first["clear_all_left_vs_fulltable"]}；整体随盖重绑首边桌面通过：{second["clear_all_left_vs_fulltable"]}。原完整7节点world曲线桌面查询通过：{result["whole_original_world_path_table_clear"]}。\n\n'
        '全程冻结真实prepare后实测q36手指；未用SDK重建。原CPU计划的arm/right确来自actual q36，但left手在规划时改为SDK400名义mimic目标。此处另记录名义与实测差异，不凭此先验归因。\n\n'
        '使用原create_table函数5碰撞件，未搜索新path、未物理执行；这里只查桌面。原world transit若通过，仍交actualscene完成body/cap/pills/self审计，只有接近cap末段用实际cap重绑。\n')

if __name__=='__main__':main()
