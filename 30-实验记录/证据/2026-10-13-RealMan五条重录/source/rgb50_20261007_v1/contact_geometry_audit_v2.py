"""Independent contact geometry audit; auxiliary FK only, no post-setup physics."""
from pathlib import Path
from types import SimpleNamespace
import argparse, copy, json, os, sys, time
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--indices", default="12,24,3,7,17,22,9,36,0,4,2,8")
opt=ap.parse_args()
WORK=Path(__file__).resolve().parent
ROOT=Path("/bigdata2/liminghe/VLA-benchmark/RoboTwin")
LEGACY=ROOT.parent/"deploy/realman_gen4/adjust_bottle_20261002"
sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)]
os.chdir(ROOT)
os.environ.setdefault("RGB50_EXPERT_MODULE", "expert_generation_v3")
os.environ.setdefault("ROBOTWIN_RT_DENOISER","none")
from generation_runner import prepare_task,json_write
from expert_generation import DEFAULT_SETTING
from expert_generation_v2 import ExpertSession, SPRITE_SETTING
from geometry_overlay import install_native_shape_count_overlay
install_native_shape_count_overlay(LEGACY)
import full_mesh_path_audit as fma

opt.out.mkdir(parents=True,exist_ok=False)
plan=json.loads((WORK/"sample_plan.json").read_text())
specs={v["index"]:v for v in plan["specs"]}
for source_index in (24,8,7,22,36,4):
    altered=copy.deepcopy(specs[source_index]);altered["index"]=source_index+1000
    altered["yaw_deg"]=float(altered["yaw_deg"])+180.
    altered["probe_variant"]="opposite_neck_direction_only_geometry"
    specs[altered["index"]]=altered
summary=[]
for index in [int(v) for v in opt.indices.split(",")]:
    spec=copy.deepcopy(specs[index]); side=spec["arm"]
    scene_out=opt.out/f"spec_{index:07d}"; scene_out.mkdir()
    print("START",index,spec["brand"],side,spec["pose_class"],flush=True)
    task=auditor=original=None
    try:
        task,receipt,original=prepare_task(spec,scene_out/"scene",need_planner=False)
        actualq=task.robot.left_entity.get_qpos().copy()
        actualposes={l.name:l.get_pose().to_transformation_matrix().copy() for l in task.robot.left_entity.get_links()}
        bottlepose=task.bottle.get_pose().to_transformation_matrix().copy()
        auditor=fma.FullMeshPathAuditor(task)
        prefix="l_" if side=="left" else "r_"
        names=[n for n in auditor.objects if n.startswith(prefix+"inspire_")]
        mapping=task.robot._hand_maps[side]
        indices={n:i for i,n in enumerate(auditor.sim_joints)}
        cache={}
        def hand_local(sdk):
            key=tuple(sdk)
            if key in cache:return cache[key]
            q=auditor.initial_sim_q.copy()
            positions=dict(zip(auditor.sim_joints,q))
            for name,value in mapping.from_sdk(sdk,positions).items():q[indices[name]]=value
            auditor.model.set_qpos(q,full=True)
            palm=fma._pose_matrix(auditor.objects[prefix+"inspire_base_link"].pose)
            local={name:np.linalg.inv(palm)@fma._pose_matrix(auditor.objects[name].pose) for name in names}
            cache[key]=local
            return local
        clones={name:auditor.fcl.FCLObject(name, auditor.Pose(np.eye(4)),
                   [auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in auditor.objects[name].shapes],
                   [auditor.Pose(p) for p in auditor.objects[name].shape_poses]) for name in names}
        def evaluate(local,palm_pose):
            groups={t:{"gap_m":float("inf"),"collision":False,"closest_link":None} for t in ("thumb","index","middle","ring","little","palm")}
            table={"gap_m":float("inf"),"collision":False,"closest_link":None}
            for name,m in local.items():
                obj=auditor.fcl.FCLObject(name,auditor.Pose(palm_pose@m),
                    [auditor.fcl.CollisionObject(s.get_collision_geometry()) for s in auditor.objects[name].shapes],
                    [auditor.Pose(p) for p in auditor.objects[name].shape_poses])
                token=next((t for t in groups if t!="palm" and t in name),"palm")
                gap,hit=auditor._distance(obj,auditor.static["bottle"])
                if gap<groups[token]["gap_m"]:groups[token].update(gap_m=gap,closest_link=name)
                groups[token]["collision"]|=bool(hit)
                gap,hit=auditor._distance(obj,auditor.static["table"])
                if gap<table["gap_m"]:table.update(gap_m=gap,closest_link=name)
                table["collision"]|=bool(hit)
            return {"finger_groups":groups,"table":table}
        base=copy.deepcopy(SPRITE_SETTING if int(spec["model_id"])==16 else DEFAULT_SETTING)
        base.update(copy.deepcopy(spec.get("grasp_setting",{})))
        rows=[]
        # Retain default family plus explicit geometric alternatives; not physics changes.
        families=[("current",{}),("y_plus5mm",{"hand_y":base["hand_y"]+.005}),
                  ("y_minus5mm",{"hand_y":base["hand_y"]-.005}),
                  ("z_plus5mm",{"hand_z":base["hand_z"]+.005})]
        thumbs=sorted(set([int(base["close_sdk"][4]),900,887,875,850,825,800,750,650,610]))
        open_local=hand_local(base["open_sdk"])
        for family,delta in families:
            for x in [-.010,-.005,0.,.005,.010]:
                setting=copy.deepcopy(base);setting.update(delta);setting["hand_x"]=x
                session=SimpleNamespace(side=side,setting=setting,report={})
                pre,goal=ExpertSession.grasp_pose(session,task.bottle,side)
                palm_goal=task.robot._trans_from_gripper_to_endlink(goal,arm_tag=side).to_transformation_matrix()
                palm_pre=task.robot._trans_from_gripper_to_endlink(pre,arm_tag=side).to_transformation_matrix()
                open_goal=evaluate(open_local,palm_goal)
                open_pre=evaluate(open_local,palm_pre)
                for thumb in thumbs:
                    sdk=copy.deepcopy(base["close_sdk"]);sdk[4]=thumb
                    closed=evaluate(hand_local(sdk),palm_goal)
                    g=closed["finger_groups"]
                    main_near=all(g[k]["gap_m"]<=.001 for k in ("thumb","index","middle"))
                    passive_clear=all(g[k]["gap_m"]>=.002 and not g[k]["collision"] for k in ("ring","little"))
                    open_goal_clear=not any(v["collision"] for v in open_goal["finger_groups"].values())
                    open_pre_clear=all(v["gap_m"]>=.003 and not v["collision"] for v in open_pre["finger_groups"].values())
                    table_clear=not any(e["table"]["collision"] for e in (closed,open_goal,open_pre))
                    row={"family":family,"setting":{k:setting[k] for k in ("hand_x","hand_y","hand_z","pitch_deg")},
                         "close_sdk6":sdk,"closed":closed,"open_goal":open_goal,"open_pre":open_pre,
                         "geometry_candidate":bool(main_near and passive_clear and open_goal_clear and open_pre_clear and table_clear),
                         "criteria":{"all_main3_within1mm":main_near,"little_ring_no_hit_min2mm":passive_clear,
                                     "open_goal_no_mesh_hit":open_goal_clear,"open_pre_min3mm":open_pre_clear,
                                     "all_hand_no_table_mesh_hit":table_clear}}
                    rows.append(row)
                    if family=="current" and x==0 and thumb in [int(base["close_sdk"][4]),887,850]:
                        print("CURRENT",index,thumb,{k:round(v["gap_m"]*1000,3) for k,v in g.items()},
                              "open",round(min(v["gap_m"] for v in open_goal["finger_groups"].values())*1000,3),
                              "table",round(closed["table"]["gap_m"]*1000,3),"candidate",row["geometry_candidate"],flush=True)
        actual_unchanged=bool(np.array_equal(actualq,task.robot.left_entity.get_qpos()) and
                    np.array_equal(bottlepose,task.bottle.get_pose().to_transformation_matrix()) and
                    all(np.array_equal(m,task.robot.left_entity.find_link_by_name(n).get_pose().to_transformation_matrix()) for n,m in actualposes.items()))
        candidates=[r for r in rows if r["geometry_candidate"]]
        ranked=sorted(rows,key=lambda r:(not r["criteria"]["all_hand_no_table_mesh_hit"],
                         not r["criteria"]["open_goal_no_mesh_hit"],not r["criteria"]["open_pre_min3mm"],
                         not r["criteria"]["little_ring_no_hit_min2mm"],
                         sum(r["closed"]["finger_groups"][k]["gap_m"] for k in ("thumb","index","middle")),
                         abs(r["setting"]["hand_x"])+abs(r["setting"]["hand_y"]-base["hand_y"])+abs(r["setting"]["hand_z"]-base["hand_z"])))
        report={"spec":spec,"scope":"Actual collision meshes copied into auxiliary FK; sampled palm placements only; no IK/reachability/contact force/physics success claim",
                "hardware_io":False,"new_motion_physics":False,"post_setup_scene_steps":0,
                "physical_state_unchanged":actual_unchanged,"source_scene":receipt,"fk_audit":auditor.fk_audit,
                "default_setting":base,"rows":rows,"candidate_count":len(candidates),"top_candidates":(candidates or ranked)[:12]}
        json_write(scene_out/"contact_geometry.json",report)
        short={"spec_index":index,"brand":spec["brand"],"arm":side,"pose_class":spec["pose_class"],
               "candidate_count":len(candidates),"physical_state_unchanged":actual_unchanged,"report":str(scene_out/"contact_geometry.json"),
               "top3":(candidates or ranked)[:3]}
        summary.append(short);json_write(opt.out/"summary.json",summary)
        print("DONE",index,"candidates",len(candidates),"rows",len(rows),"unchanged",actual_unchanged,flush=True)
    except BaseException as e:
        import traceback
        json_write(scene_out/"probe_error.json",{"error":str(e),"traceback":traceback.format_exc(),"geometry_only":True})
        print("ERROR",index,str(e),flush=True)
    finally:
        if auditor is not None:auditor.model.set_qpos(auditor.old_model_q,full=True)
        if original is not None:
            from integrations.realman_rm65b_gen4.native_robot import RealManRobot
            RealManRobot._initialize_hands=original
        if task is not None:task.close_env()
