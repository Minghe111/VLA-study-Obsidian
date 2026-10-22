"""CPU-only gravity drop into dynamic petri: never a robot demonstration."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import numpy as np
import sapien
from petri_receiver import load_dynamic_petri, SPEC
from capsule_geometry import read_capsule_entities

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "receiver_probes" / "dynamic_petri_drop12_v1"
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(SPEC.read_text())
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.004)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.5, .5, .01],
        material=sapien.physx.PhysxMaterial(.5, .5, 0))
    builder.initial_pose = sapien.Pose([0, 0, -.01])
    table = builder.build_static("receiver_component_bench_table")
    receiver, contract = load_dynamic_petri(scene, sapien.Pose([.04, -.03, .0005]))
    tablets = []
    initial = []
    for i in range(12):
        cfg = spec["tablets"]
        builder = scene.create_actor_builder()
        builder.add_capsule_collision(radius=cfg["radius_m"], half_length=cfg["half_length_m"],
            material=sapien.physx.PhysxMaterial(cfg["friction"], cfg["friction"], cfg["restitution"]))
        angle = 2*math.pi*(i % 3)/3
        yaw = .371*i
        local = sapien.Pose([.009*math.cos(angle), .009*math.sin(angle), .030+.012*(i//3)],
                            [math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
        builder.initial_pose = receiver.pose*local
        entity = builder.build("receiver_component_tablet_"+str(i).zfill(2))
        rigid = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        rigid.mass = cfg["mass_kg"]
        rigid.sleep_threshold = 0.
        rigid.solver_position_iterations = spec["physics"]["solver_position_iterations"]
        rigid.solver_velocity_iterations = spec["physics"]["solver_velocity_iterations"]
        for shape in rigid.collision_shapes:
            shape.contact_offset = spec["physics"]["contact_offset_m"]
            shape.rest_offset = spec["physics"]["rest_offset_m"]
        tablets.append(entity)
        initial.append({"name":entity.name,"pose":np.r_[entity.pose.p,entity.pose.q].tolist(),
                        "dynamic":not rigid.kinematic,"mass_kg":float(rigid.mass)})
    tablet_ids = {int(t.per_scene_id) for t in tablets}
    trace = []
    hold = 0
    maximum_hold = 0
    spill_contacts = 0
    receiver_component = receiver.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    for tick in range(1, 2501):
        scene.step()
        geometric = read_capsule_entities(tablets, receiver.pose,
            half_length_m=cfg["half_length_m"], capsule_radius_m=cfg["radius_m"],
            inner_radius_m=spec["receiver"]["inner_radius_m"],
            floor_z_m=spec["receiver"]["base_thickness_m"],
            rim_z_m=spec["receiver"]["height_m"],
            tolerance_m=spec["physics"]["contact_offset_m"])
        spilled_this_tick = False
        for c in scene.get_contacts():
            ids = {int(b.entity.per_scene_id) for b in c.bodies}
            if int(table.per_scene_id) in ids and ids & tablet_ids:
                if any(np.linalg.norm(p.impulse) > 1e-9 for p in c.points):
                    spill_contacts += 1
                    spilled_this_tick = True
        inside = geometric["full_geometry_inside_count"] == 12 and not spilled_this_tick
        hold = hold+1 if inside else 0
        maximum_hold = max(maximum_hold, hold)
        trace.append({"physics_step":tick,"time_s":tick*.004,
                      "receiver_pose":np.r_[receiver.pose.p, receiver.pose.q].tolist(),
                      "tablet_poses":[np.r_[t.pose.p,t.pose.q].tolist() for t in tablets],
                      "full_geometry_inside_count":geometric["full_geometry_inside_count"],
                      "minimum_margin_m":min(min(row[k] for k in
                          ("radial_margin_m","floor_margin_m","rim_margin_m")) for row in geometric["tablets"]),
                      "direct_table_spill_contact":spilled_this_tick})
    trace_path = output / "trace.json.gz"
    with gzip.open(trace_path, "wt") as stream:
        json.dump(trace, stream, allow_nan=False)
    report = {"scope":"Dynamic receiver gravity-drop component only; no robot or bottle pour",
              "robot_present":False,"robot_success":False,"usable_robot_demonstration":False,
              "receiver_contract":contract,"initial_tablets":initial,
              "steps":2500,"physics_dt_s_actual":float(scene.get_timestep()),
              "final_geometry":geometric,"last_inside_contiguous_seconds":hold*.004,
              "max_inside_contiguous_seconds":maximum_hold*.004,
              "tablet_table_spill_contact_count":spill_contacts,
              "receiver_final_pose":np.r_[receiver.pose.p,receiver.pose.q].tolist(),
              "receiver_final_velocity_m_s":np.asarray(receiver_component.linear_velocity).tolist(),
              "pose_or_velocity_writes_after_initialization":False,
              "source_sha256":{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__),ROOT/'petri_receiver.py',ROOT/'capsule_geometry.py']},
              "trace_sha256":hashlib.sha256(trace_path.read_bytes()).hexdigest(),
              "frozen_geometry_sources_unchanged":all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                  for p,h in contract['source_sha256'].items()),
              "passed_component":hold*.004 >= .5 and spill_contacts == 0 and contract['dynamic']}
    (output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['passed_component','last_inside_contiguous_seconds',
                    'tablet_table_spill_contact_count','robot_success']}),flush=True)


if __name__ == '__main__':
    main()
