"""Actual CPU rigid-body integration; the holding fixture is not a robot policy."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import sapien
from lab_physics import ROOT, LabBottle, add_receiver, create_cpu_scene, qaxis, read_spec, rigid


def run(seed=0, out=None):
    s=read_spec()
    dt=s["physics"]["dt_s"]
    scene=create_cpu_scene(s)
    b=scene.create_actor_builder()
    b.add_box_collision(half_size=[1,1,.01])
    b.initial_pose=sapien.Pose([0,0,-.011])
    b.build_static("floor")
    receiver=add_receiver(scene,s,pose=sapien.Pose(s["fixture"]["receiver_position_m"]))
    neck=np.array(s["fixture"]["bottle_mouth_position_m"])
    h=s["bottle"]["height_m"]
    bottle=LabBottle(scene,s,pose=sapien.Pose(neck-[0,0,h]),seed=seed)
    events=[];trace=[];step=0;contact_steps={"tablet_cap":0,"tablet_body":0,"tablet_receiver":0,"tablet_floor":0}
    phase=""
    all_closed_inside=[]
    def simulate(seconds,label,control=None):
        nonlocal step,phase
        phase=label
        for j in range(round(seconds/dt)):
            t=(j+1)/round(seconds/dt)
            if control is not None:control(t)
            scene.step();bottle.after_step();step+=1
            if label in ("closed_invert","closed_inverted_hold"):
                all_closed_inside.append(inside())
            for contact in scene.get_contacts():
                if not any(np.linalg.norm(point.impulse)>1e-10 for point in contact.points):
                    continue
                names=[comp.entity.name for comp in contact.bodies]
                if any(name.startswith("tablet_") for name in names):
                    for contact_label,entity_name in (("tablet_cap",bottle.cap.name),("tablet_body",bottle.body.name),("tablet_receiver",receiver.name),("tablet_floor","floor")):
                        if entity_name in names:contact_steps[contact_label]+=1
            if step%20==0:
                trace.append({"step":step,"time_s":step*dt,"phase":label,
                    "body_pose":np.r_[bottle.body.pose.p,bottle.body.pose.q].tolist(),
                    "cap_pose":np.r_[bottle.cap.pose.p,bottle.cap.pose.q].tolist(),
                    "cap_state":bottle.cap_state(),"tablets":bottle.tablet_states()})
    def tilt(a):
        q=qaxis([0,1,0],a)
        R=sapien.Pose(q=q).to_transformation_matrix()[:3,:3]
        bottle.set_fixture_pose(sapien.Pose(neck-R@np.array([0,0,h]),q))
    def inside():
        v=bottle.tablet_local_positions()
        return int(np.sum((np.linalg.norm(v[:,:2],axis=1)<s["bottle"]["inner_radius_m"]+.002)&(v[:,2]>-.003)&(v[:,2]<h+.007)))
    simulate(.8,"upright_settle")
    initial_inside=inside()
    try:
        bottle.release_cap();premature_refused=False
    except RuntimeError:
        premature_refused=True
    simulate(1.5,"closed_invert",lambda t:tilt(math.pi*(3*t*t-2*t*t*t)))
    closed_counts=[]
    for _ in range(20):
        simulate(.05,"closed_inverted_hold")
        closed_counts.append(inside())
    events.append({"event":"closed_inverted_no_leak","step":step,"minimum_inside":min(closed_counts)})
    simulate(1.5,"return_upright",lambda t:tilt(math.pi*(1-(3*t*t-2*t*t*t))))
    simulate(.8,"upright_resettle")
    simulate(3.0,"unscrew_guide",lambda t:bottle.command_unscrew(s["thread_guide"]["opening_turns"]*t))
    simulate(.6,"unscrew_hold")
    release=bottle.release_cap()
    events.append({"event":"cap_released","step":step,"measured":release})
    cap0=bottle.cap.pose
    simulate(.8,"cap_fixture_remove",lambda t:bottle.command_cap_holder(sapien.Pose(cap0.p+np.array([.14*t,0,.04*t]),cap0.q)))
    simulate(1.6,"pour_fixture_tilt",lambda t:tilt(math.radians(s["fixture"]["pour_tilt_deg"])*(3*t*t-2*t*t*t)))
    simulate(3.,"pour_settle")
    pos=np.asarray([(receiver.pose.inv()*a.pose).p for a in bottle.tablets])
    receiver_inside=(np.linalg.norm(pos[:,:2],axis=1)<s["receiver"]["inner_radius_m"]-s["tablets"]["radius_m"]*.5)&(pos[:,2]>0)&(pos[:,2]<s["receiver"]["height_m"])
    final_count=int(receiver_inside.sum())
    events.append({"event":"receiver_count","step":step,"count":final_count})
    allfinite=all(np.isfinite(row["pose"]).all() for row in bottle.tablet_states())
    report={"spec_version":s["version"],"seed":seed,"engine":"SAPIEN 3.0.0b1 PhysX CPU",
        "physics_steps":step,"dt_s":dt,"elapsed_sim_s":step*dt,"initial_tablets":s["tablets"]["count"],
        "initial_inside":initial_inside,"closed_inverted_min_inside":min(all_closed_inside),
        "closed_inverted_checked_physics_steps":len(all_closed_inside),
        "premature_release_refused":premature_refused,"release_measured":release,
        "final_receiver_count":final_count,"remaining_bottle_count":inside(),"tablets_finite":allfinite,
        "receiver_collision_geometry":s["receiver"],"fixture_trajectory_spec":s["fixture"],
        "contact_pair_step_counts":contact_steps,"final_tablet_actor_count":len(bottle.tablets),
        "tablets_pose_writes_after_initialization":0,"tablet_pose_write_boundary":"Only ActorBuilder.initial_pose during construction; subsequent integration by PhysX",
        "bottle_motion":"declared kinematic fixture, not RM65-B grasp/policy", "cap_motion":"force/torque limited D6 guide then separate cap-holding fixture",
        "true_thread_mesh_contact":False,"hardware_io":False,"events":events,
        "passed":initial_inside==s["tablets"]["count"] and min(all_closed_inside)==s["tablets"]["count"] and premature_refused and final_count==s["tablets"]["count"] and allfinite,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"lab_physics.py",ROOT/"test_lab_physics.py",ROOT/"physics_spec.json"]}}
    target=Path(out or ROOT/f"physics_result_seed{seed}.json")
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    target.with_name(target.stem+"_trace.json").write_text(json.dumps(trace,separators=(",",":"))+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def blocked_cap(out=None):
    """Negative control: a fixture blocks cap twist; a requested turn cannot lift it."""
    s=read_spec();scene=create_cpu_scene(s);bottle=LabBottle(scene,s)
    rigid(bottle.cap).set_locked_motion_axes([False,False,False,False,False,True])
    steps=3000
    for i in range(steps):
        bottle.command_unscrew(s["thread_guide"]["opening_turns"]*(i+1)/steps)
        scene.step();bottle.after_step()
    state=bottle.cap_state()
    try:
        bottle.release_cap();refused=False
    except RuntimeError:
        refused=True
    report={"test":"blocked_cap_negative_control","physics_steps":steps,"dt_s":s["physics"]["dt_s"],
        "rotation_target_rad":s["thread_guide"]["opening_turns"]*2*math.pi,
        "measured":state,"guide_axial_target_lift_m":float(bottle.guide.drive_target.p[2]-bottle.closed_z),
        "release_refused":refused,"blocking_fixture":"world-Z angular motion locked on dynamic cap; body upright",
        "passed":refused and abs(state["rotation_rad"])<.001 and abs(state["lift_m"])<.001,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"lab_physics.py",ROOT/"test_lab_physics.py",ROOT/"physics_spec.json"]}}
    Path(out or ROOT/"physics_blocked_cap.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=0);p.add_argument("--out");p.add_argument("--blocked-cap",action="store_true")
    args=p.parse_args()
    result=blocked_cap(args.out) if args.blocked_cap else run(args.seed,args.out)
    raise SystemExit(0 if result["passed"] else 1)
