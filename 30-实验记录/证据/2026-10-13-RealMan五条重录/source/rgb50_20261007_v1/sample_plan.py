"""CPU-only sampling contract. Does not import SAPIEN, run physics, or alter the old20."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
OLD20 = ROOT.parent / "realman_rgb20_final_20261004" / "dataset" / "manifest.json"
OLD20_SHA256 = "97b33109a740ae2d9b99155dbf228e5d2a19761b323e60fc622c8998aec0e577"
SHOULDERS_WORLD_M = {
    "left": [-0.26380305357951817, -0.6165399525745515, 1.1258990198444143],
    "right": [0.26388325787272415, -0.6167138122090484, 1.125897418553584],
}
TABLE_TOP_Z_M = 0.740
# Counts are real new demonstrations to collect, not relabeled copies of old20.
CELLS = {
    13: {"left": {"upright": 4, "horizontal": 3, "diagonal": 3},
         "right": {"upright": 4, "horizontal": 3, "diagonal": 3}},
    16: {"left": {"upright": 4, "horizontal": 3, "diagonal": 3},
         "right": {"upright": 4, "horizontal": 3, "diagonal": 3}},
    4: {"left": {"upright": 2, "horizontal": 1, "diagonal": 2},
        "right": {"upright": 2, "horizontal": 2, "diagonal": 1}},
}

def matvec(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(3)) for row in matrix]

def quat_matrix(q):
    w, x, y, z = q
    return [[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]]

def qz_qx(z_angle, x_angle):
    # qz * qx, WXYZ. Upright maps asset local+Y to world+Z.
    cz, sz = math.cos(z_angle/2), math.sin(z_angle/2)
    cx, sx = math.cos(x_angle/2), math.sin(x_angle/2)
    return [cz*cx, cz*sx, sz*sx, sz*cx]

def nearest_arm(center):
    ds = {arm: math.dist(center, origin) for arm, origin in SHOULDERS_WORLD_M.items()}
    arm = min(ds, key=ds.get)
    return arm, ds, abs(ds["left"]-ds["right"])

def legacy_audit(assets):
    raw = OLD20.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == OLD20_SHA256
    old = json.loads(raw)
    rows = []
    for ep in old["episodes"]:
        asset = assets["brands"][str(ep["model_id"])]
        matrix = ep["source_identity"]["initial_bottle_pose"]
        rotated = matvec([row[:3] for row in matrix[:3]], asset["collision_scene_bbox_center_m"])
        center = [rotated[j] + matrix[j][3] for j in range(3)]
        arm, ds, margin = nearest_arm(center)
        pca_axis = matvec([row[:3] for row in matrix[:3]], asset["collision_pca_long_axis_local"])
        vertical_angle = math.degrees(math.acos(min(1., abs(pca_axis[2]))))
        rows.append({"episode_index": ep["index"], "source_id": ep["source_id"],
                     "model_id": ep["model_id"], "recorded_arm": "left",
                     "geometry_center_world_m": center, "nearest_arm": arm,
                     "distances_m": ds, "distance_margin_m": margin,
                     "nearest_arm_matches_recorded": arm == "left",
                     "pca_long_axis_world": pca_axis,
                     "axis_angle_from_vertical_deg": vertical_angle})
    return {"status": "read_only_historical_geometry_audit", "new_physics_runs": 0,
            "original_manifest": str(OLD20), "original_manifest_sha256": OLD20_SHA256,
            "position_reference": "Actual collision GLB scene bbox center after asset scale, transformed by historical actor pose",
            "arm_reference": "Fixed world origins of l_joint1/r_joint1 from historical robot root pose and URDF chain",
            "shoulders_world_m": SHOULDERS_WORLD_M,
            "count": len(rows), "nearest_arm_pass_count": sum(r["nearest_arm_matches_recorded"] for r in rows),
            "episodes": rows,
            "decision": "Preserve old20 unchanged. Do not inherit or relabel them into a strict nearest-arm50 dataset if they fail this rule."}

def make_plan(assets):
    cells = [(model, arm, pose) for model, arms in CELLS.items()
             for arm, poses in arms.items() for pose, count in poses.items() for _ in range(count)]
    random.Random(20261007).shuffle(cells)
    specs = []
    for index, (model, requested_arm, pose) in enumerate(cells):
        seed = 100000 + index
        rng = random.Random(seed)
        x = rng.uniform(.085, .160) * (-1 if requested_arm == "left" else 1)
        y = rng.uniform(-.140, -.070)
        if pose == "upright":
            yaw = rng.uniform(-180., 180.)
            q = qz_qx(math.radians(yaw), math.pi/2)
        else:
            reverse = rng.choice([0., 180.])
            deviation = (rng.uniform(-12., 12.) if pose == "horizontal"
                         else rng.choice([-1, 1])*rng.uniform(35., 60.))
            yaw = reverse + deviation
            # Asset local+Y is bottle long axis. yaw describes its XY azimuth.
            q = qz_qx(math.radians(yaw-90.), 0.)
        asset = assets["brands"][str(model)]
        rot = quat_matrix(q)
        local_center = asset["collision_scene_bbox_center_m"]
        lo, hi = asset["collision_scene_bbox_min_m"], asset["collision_scene_bbox_max_m"]
        points = [matvec(rot, corner) for corner in itertools.product(*zip(lo, hi))]
        rotated_center = matvec(rot, local_center)
        # Put the true collision bbox bottom 2mm above the physical table then settle.
        center_z = TABLE_TOP_Z_M + .002 + rotated_center[2] - min(point[2] for point in points)
        center = [x, y, center_z]
        origin = [center[j]-rotated_center[j] for j in range(3)]
        arm, distances, margin = nearest_arm(center)
        assert arm == requested_arm and margin > .020
        specs.append({"index": index, "seed": seed, "model_id": model,
                      "brand": asset["brand"], "arm": arm, "pose_class": pose,
                      "xy": [x, y], "yaw_deg": yaw,
                      "yaw_semantics": "world XY azimuth of bottle long axis for lying poses; vertical-axis label rotation for upright",
                      "position_reference": "collision_mesh_bbox_center_world",
                      "initial_geometry_center_world_m": center,
                      "actor_origin_world_m": origin, "actor_q_wxyz": q,
                      "nearest_arm_distances_m": distances, "nearest_arm_margin_m": margin,
                      "acceptance_requires_settled_pose_check": True})
    counts = lambda key: dict(Counter(str(spec[key]) for spec in specs))
    assert counts("arm") == {"left": 25, "right": 25} or counts("arm") == {"right": 25, "left": 25}
    assert counts("model_id") == {"13": 20, "16": 20, "4": 10} or all(counts("model_id")[k] == v for k,v in {"13":20,"16":20,"4":10}.items())
    assert all(counts("pose_class")[k] == v for k,v in {"upright":20,"horizontal":15,"diagonal":15}.items())
    return {"status": "candidate_sampling_plan_only_not_generated_dataset", "new_physics_runs": 0,
            "dataset_goal": 50, "old20_policy": "Kept unchanged as historical dataset, strict nearest-arm version collects 50 fresh real trajectories",
            "model_counts": counts("model_id"), "arm_counts": counts("arm"), "pose_counts": counts("pose_class"),
            "cells": CELLS, "shoulders_world_m": SHOULDERS_WORLD_M,
            "position_reference": "Actual collision mesh geometry center, not actor origin",
            "arm_selection_rule": "Choose Euclidean nearest fixed shoulder before planning; after settling recompute with actual pose. Reject ambiguous margin<20mm or unexpected arm, do not change label.",
            "post_settle_pose_checks": {"upright": "PCA long axis <=15 degrees from world vertical",
                "horizontal": "PCA long axis <=15 degrees from table plane and XY azimuth within20 degrees of0/180",
                "diagonal": "PCA long axis <=15 degrees from table plane and XY azimuth35..65 degrees from0/180"},
            "formal_success_requirements": ["Native unassisted bottle lift and physical support verified",
                "No fixed constraint, no bottle pose writes during grasp", "Validate no table support when airborne",
                "Both hands remain in physical traces", "Three D435 RGB streams and state/action timing must follow recording contract"],
            "specs": specs}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write only independent JSON planning/audit files in this folder")
    args = parser.parse_args()
    assets_path = ROOT / "assets_identity.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    audit = legacy_audit(assets)
    plan = make_plan(assets)
    plan["assets_identity_sha256"] = hashlib.sha256(assets_path.read_bytes()).hexdigest()
    plan["legacy20_nearest_arm_pass_count"] = audit["nearest_arm_pass_count"]
    if args.write:
        for name, obj in [("legacy20_nearest_arm_audit.json", audit), ("sample_plan.json", plan)]:
            (ROOT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "legacy20_nearest_pass": audit["nearest_arm_pass_count"],
                      "new50_counts": {k: plan[k] for k in ["model_counts", "arm_counts", "pose_counts"]},
                      "minimum_new_nearest_margin_m": min(s["nearest_arm_margin_m"] for s in plan["specs"]),
                      "new_physics_runs": 0}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
