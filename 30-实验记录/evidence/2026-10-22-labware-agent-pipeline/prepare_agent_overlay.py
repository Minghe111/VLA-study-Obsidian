#!/usr/bin/env python3
"""Create an independent RoboTwin object/agent metadata overlay, then verify it.

The 2026-10-22 labware integration bundle is read-only input. Geometry is not
rewritten: only new model_data and points_info files are authored here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "2026-10-22-labware-integration"
ROBOTWIN = Path("/home/admin123/liminghe/vla-benchmark/RoboTwin")
META = ROOT / "metadata"
OBJECTS = META / "assets" / "objects"
DESCRIPTIONS = META / "description" / "objects_description"
ASSETS = {
    "beaker": ("laboratory beaker", "open laboratory beaker", "short cylindrical laboratory beaker"),
    "erlenmeyer": ("Erlenmeyer flask", "open Erlenmeyer flask", "conical laboratory flask"),
    "graduated_cylinder": ("measuring cylinder", "tall laboratory measuring cylinder", "narrow cylindrical laboratory measure"),
    "petri_dish": ("Petri dish", "open shallow Petri dish", "round shallow laboratory dish"),
    "reagent_bottle_body": ("reagent bottle body", "open reagent bottle body", "cylindrical laboratory bottle body"),
    "reagent_bottle_cap": ("reagent bottle cap", "separate reagent bottle cap", "cylindrical laboratory bottle cap"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() == data:
        return
    path.write_text(data, encoding="utf-8")


def source_fingerprints():
    paths = [INPUT / "asset_interface.json", INPUT / "annotation_specs.json",
             INPUT / "source_manifest.json", INPUT / "physics_spec.json"]
    for asset_id in ASSETS:
        paths += [INPUT / "prepared" / asset_id / "model_data0.json",
                  INPUT / "prepared" / asset_id / "visual" / "base0.glb",
                  INPUT / "prepared" / asset_id / "collision" / "base0.glb"]
    return {str(p): sha(p) for p in paths}


def link_readonly_geometry(source, target):
    """No write is ever performed through these geometry links."""
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting symlink: {target}")
    elif target.exists():
        raise RuntimeError(f"Geometry destination must be a new symlink: {target}")
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


def add_bottom_point(asset_id, data):
    # Append only; the original functional IDs keep their precise old semantics.
    before = copy.deepcopy(data["functional_matrix"])
    bottom = copy.deepcopy(data["target_pose"][0])
    bottom[0][:3] = [1.0, 0.0, 0.0]
    bottom[1][:3] = [0.0, -1.0, 0.0]
    bottom[2][:3] = [0.0, 0.0, -1.0]
    bottom_idx = len(before)
    data["functional_matrix"].append(bottom)
    descriptions = list(data["functional_point_description"])
    descriptions.append("External lower support-plane centre (-Z outward); align to a support surface; this is not the interior floor or opening.")
    # Keep original key and add the spelling used in downstream annotation tools.
    data["functional_point_description"] = descriptions
    data["functional_points_description"] = descriptions.copy()
    data["target_points_description"] = list(data["target_point_description"])
    data["agent_overlay_provenance"] = {
        "date": "2026-10-22", "authoring": "local Codex-authored metadata / geometry derived from frozen specification",
        "original_annotation": str(INPUT / "prepared" / asset_id / "model_data0.json"),
        "original_annotation_sha256": sha(INPUT / "prepared" / asset_id / "model_data0.json"),
        "preserved_functional_ids": list(range(len(before))),
        "appended_functional_bottom_id": bottom_idx,
        "not_official_RoboTwin_authored_metadata": True,
    }
    if data["functional_matrix"][:len(before)] != before:
        raise AssertionError("Original functional IDs changed")
    return bottom_idx


def point_semantics(asset_id, data, bottom_idx):
    cap = asset_id == "reagent_bottle_cap"
    functions = [
        {"id": 0, "description": "Underside cap opening centre; local -Z points out of the opening." if cap else "Upper opening centre; local +Z points out of the vessel.",
         "usage": "Align the cap opening to the separate bottle neck. Physical release requires the task's screw mechanism." if cap else "Receiving/pouring mouth alignment. Never use this point to place the bottom on a table."},
        {"id": 1, "description": "Upper twist-axis marker." if cap else "Interior floor centre, inside the hollow receiving volume.",
         "usage": "Define the cap twist axis; this point alone does not actuate or unscrew the cap." if cap else "Define the receiving region for dry pellets. It is not the external support plane."},
    ]
    if not cap:
        functions.append({"id": 2, "description": "Selected rim surface point stored in the local mesh frame.",
                          "usage": "Candidate pouring-edge alignment. It is not a calibrated spout or guaranteed pouring direction."})
    functions.append({"id": bottom_idx, "description": "External lower support-plane centre, with local -Z outward.",
                      "usage": "Use this functional point for upright support placement. Do not substitute the opening or inner floor."})
    contacts = []
    for idx, direction in enumerate(("+X", "+Y", "-X", "-Y")):
        contacts.append({"id": idx, "description": f"Side grasp centre, candidate {idx}; contact-frame +Y points toward object-frame {direction}.",
                         "usage": "Candidate grasp frame; pregrasp offset is along contact +Y and approach along -Y. Must pass this robot's actual grasp/clearance validation."})
    return {
        "contact_points": contacts,
        "functional_points": functions,
        "target_points": [{"id": 0, "description": "External lower support-plane centre.", "usage": "Support alignment; not a side grasp."},
                          {"id": 1, "description": "Cap underside opening alignment." if cap else "Upper opening centre.", "usage": "Opening alignment."}],
        "orientation_points": [{"id": 0, "description": "Object Z-up frame, +Z along vessel/cap axis.", "usage": "Upright orientation reference."}],
        "metadata_provenance": {"date": "2026-10-22", "source": "locally authored from frozen model_data0 transforms and geometric specification", "not_official_original": True},
    }


def object_description(asset_id):
    noun, seen_name, unseen_name = ASSETS[asset_id]
    # Official schema; variations are locally authored, not LLM/API output.
    seen = [noun, seen_name, "open-topped " + noun if asset_id != "reagent_bottle_cap" else "detached " + noun,
            "laboratory " + noun, "round " + noun, "small " + noun]
    unseen = [unseen_name, "laboratory vessel shaped as a " + noun if asset_id != "reagent_bottle_cap" else "detachable cover for the reagent bottle"]
    return {"raw_description": noun, "seen": list(dict.fromkeys(seen)), "unseen": unseen,
            "provenance": {"authoring": "locally authored by Codex from asset names; no separate remote LLM API call", "date": "2026-10-22",
                           "split_meaning": "phrase split only; one model, not an unseen geometry benchmark", "not_official_RoboTwin_authored": True}}


def task_instructions():
    seen = [
        "Use {a} to pick up {A}, keep it upright in the air for at least 0.5 seconds, and put it back at its initial position before releasing and withdrawing the hand.",
        "With {a}, grasp {A}, lift it upright, hold it for at least half a second, then return it upright to its original spot, release it, and move the hand away.",
        "Pick up {A} using {a}; maintain an upright airborne hold for 0.5 seconds or longer, replace it at the starting location, and release and retreat.",
        "Grasp {A} with {a}, raise it without tipping, hold it above the table for at least 0.5 seconds, then set it back on the same spot and withdraw after releasing.",
        "Use {a} to lift {A}, hold it upright for half a second or more, place it upright where it started, open the hand, and retreat.",
        "Take hold of {A} with {a}, keep it upright off the table for at least 0.5 seconds, and return it stably to its original position before letting go and moving away.",
        "Using {a}, pick up {A} upright and pause in the air for at least half a second; put it back at the initial position, release, and move clear.",
        "Lift {A} with {a}, maintain its upright pose for no less than 0.5 seconds, then replace it at its starting spot and leave it standing after releasing and retreating.",
        "With {a}, raise {A} upright from the table, hold for at least half a second, and set it back at the same position; release it and withdraw the hand.",
        "Use {a} to grasp and lift {A}; after an upright hold of at least 0.5 seconds, return it to the original location, release it stably, and retreat.",
        "Pick {A} up with {a}, keep it upright in the air for a minimum of 0.5 seconds, then put it upright back where it was and move away after letting go.",
        "Using {a}, grasp {A}, lift and hold it upright for at least half a second, and place it stably at its initial position before releasing and withdrawing.",
    ]
    unseen = [
        "Employ {a} to raise {A}, sustain an upright airborne grasp for at least 0.5 seconds, then restore its initial placement, release it, and withdraw.",
        "With {a}, lift {A} vertically, hold it upright for no less than half a second, and leave it standing at its original spot after releasing and moving clear.",
        "Have {a} pick up {A} and retain its upright orientation for at least 0.5 seconds off the table, before returning it stably to the start and retreating after release.",
        "Secure {A} in {a}, elevate it while upright, pause for at least half a second, then replace it at the initial location and withdraw once released.",
        "Using {a}, take {A} off its support, preserve an upright hold for a minimum of 0.5 seconds, then return and release it on its original spot and move away.",
        "Raise {A} upright with {a}; after holding it airborne for half a second or longer, set it upright at the same location, release, and pull the hand clear.",
        "Let {a} grasp {A} and keep it upright above the table for at least 0.5 seconds, then settle it back at the starting position and retreat after letting go.",
        "With {a}, lift and retain {A} upright for at least half a second, then restore its stable original placement, release the grasp, and move away.",
        "Use {a} for an upright pickup of {A}, an airborne pause lasting at least 0.5 seconds, and a stable return to the original position followed by release and retreat.",
        "Take {A} in {a}, hold it upright clear of the table for no less than half a second, then place it back where it began and withdraw after releasing.",
        "Using {a}, elevate {A} upright and maintain the grasp for at least 0.5 seconds in the air; return it to the initial spot, release, and retreat.",
        "Pick up {A} by means of {a}, hold it upright off the table for a minimum of half a second, then replace it stably at its original location, let go, and withdraw.",
    ]
    return {"seen": seen, "unseen": unseen,
            "provenance": {"date": "2026-10-22", "authoring": "locally authored by Codex from the user task and verified runtime contract; no separate remote LLM API call",
                           "task": "empty open bottle contact-driven pickup / upright hold >=0.5s / stable return / release / hand retreat",
                           "split_meaning": "instruction phrasing only; no unseen geometry or task success claim"}}


def concrete_task_info(registry):
    body = registry["assets"]["reagent_bottle_body"]
    description_value = body["episode_description_value"]
    return {
        "task_name": "labware_pick_place",
        "task_description": (
            "Operate only self.bottle: it is the empty, open reagent_bottle_body, without a cap or tablets. "
            "Use the right arm to make a real contact-driven grasp, lift it at least 0.02 m clear of the table, "
            "hold it upright continuously for at least 0.5 seconds, return it upright to its initial position, "
            "release it, withdraw the hand, and allow stable placement. This first baseline deliberately uses "
            "the explicitly provided LOCAL REALMAN SDK24 EXTENSION self.grasp_lift_place(self.bottle, ArmTag('right')). "
            "This helper is not an original RoboTwin API: it executes the native hand/arm control and physical checks. "
            "Do not invoke unlisted APIs, teleport/set an actor pose, attach a fixture, change physics or success labels, "
            "open a cap, pour pills, operate other actors, or replace the helper with a hand-authored joint trajectory. "
            "The actor's functional point 0 is the opening, 1 the interior floor, 2 a rim candidate, and 3 its external bottom. "
            "Use the listed helper for the entire pickup/hold/place/release/retreat sequence. "
            f"Retain/record self.info['info'] = {{'{{A}}': {description_value!r}, '{{a}}': 'right'}} "
            "and return self.info at the end. The setup and physical check_success are already implemented by the task environment; "
            "only implement play_once."
        ),
        "current_code": "class gpt_labware_pick_place(labware_pick_place):\n    def play_once(self):\n        pass\n",
        "actor_list": {"self.bottle": {
            "name": "empty open reagent bottle",
            "description": "The only manipulated actor: an empty open cylindrical reagent bottle body. The right-hand native contact helper acts on this actor. No cap or pellets are loaded in this minimum task.",
            "modelname": body["absolute_modelname"],
        }},
    }


def prepare():
    before = source_fingerprints()
    registry = {"schema_version": 1, "date": "2026-10-22", "robotwin_root": str(ROBOTWIN),
                "OBJECTS_PATH": str(OBJECTS), "objects_description_root": str(DESCRIPTIONS), "assets": {},
                "binding": "Official code_gen/test_gen_code.py uses a fixed OBJECTS_PATH. Bind that module variable to this root, or use absolute modelname paths from this registry; no official asset files are changed.",
                "frozen_input_fingerprints": before}
    actors, absolute_actors = {}, {}
    for asset_id, (noun, _, _) in ASSETS.items():
        source = INPUT / "prepared" / asset_id
        dest = OBJECTS / asset_id
        dest.mkdir(parents=True, exist_ok=True)
        for dirname in ("visual", "collision"):
            link_readonly_geometry(source / dirname, dest / dirname)
        data = json.loads((source / "model_data0.json").read_text())
        bottom_idx = add_bottom_point(asset_id, data)
        points = point_semantics(asset_id, data, bottom_idx)
        write_json(dest / "model_data0.json", data)
        write_json(dest / "points_info.json", points)
        write_json(DESCRIPTIONS / asset_id / "base0.json", object_description(asset_id))
        registry["assets"][asset_id] = {
            "modelname": asset_id, "absolute_modelname": str(dest), "model_id": 0,
            "model_data": str(dest / "model_data0.json"), "points_info": str(dest / "points_info.json"),
            "object_description": str(DESCRIPTIONS / asset_id / "base0.json"),
            "episode_description_value": str(DESCRIPTIONS / asset_id / "base0"),
            "convex": True, "mass_kg": data["annotation_provenance"]["mass_kg"],
            "functional_ids": {"opening": 0, "twist_axis" if asset_id.endswith("_cap") else "inner_floor": 1,
                               **({"rim": 2} if not asset_id.endswith("_cap") else {}), "bottom": bottom_idx},
            "grasp_execution_verified": False,
        }
        info = {"name": noun, "description": f"Locally imported/derived {noun}; functional bottom ID {bottom_idx}. Follow exact point semantics. Candidate side grasps require robot verification.", "modelname": asset_id}
        actors["self." + asset_id] = info
        absolute_actors["self." + asset_id] = dict(info, modelname=str(dest))
    assert source_fingerprints() == before, "Frozen inputs changed"
    write_json(META / "registry.json", registry)
    write_json(META / "actor_list_template.json", actors)
    write_json(META / "actor_list_absolute.json", absolute_actors)
    write_json(META / "task_info_template.json", {
        "task_name": "labware_pick_place", "task_description": "Robot-driven pick/lift/place baseline. Use the named body's side grasp candidates and its explicit bottom functional point to place upright. No state teleporting during the episode. Implement load_actors and check_success in the task environment first.",
        "current_code": "class gpt_labware_pick_place(labware_pick_place):\n    def play_once(self):\n        pass\n", "actor_list": actors,
        "template_scope": "Actor template only; the actual accepted task must specify its active actors and physical success predicate."})
    write_json(META / "description/task_instruction/labware_pick_place.json", task_instructions())
    # The original helper opens 'utils/../task_instruction': the intermediate
    # directory must exist even though no local Python utility code is needed.
    (META / "description/utils").mkdir(parents=True, exist_ok=True)
    (META / "description/utils/.gitkeep").touch(exist_ok=True)
    write_json(META / "labware_pick_place_task_info.json", concrete_task_info(registry))
    return registry


def validate_original(registry):
    """Import the genuine upstream enrich_actors and Actor; no reimplementation."""
    import numpy as np
    import sapien
    import transforms3d as t3d
    oldcwd = os.getcwd()
    sys.path.insert(0, str(ROBOTWIN))
    sys.path.insert(0, str(ROBOTWIN / "code_gen"))
    try:
        os.chdir(ROBOTWIN)
        upstream = importlib.import_module("test_gen_code")
        Actor = importlib.import_module("envs.utils.actor_utils").Actor
        actors = json.loads((META / "actor_list_template.json").read_text())
        old_path = upstream.OBJECTS_PATH
        try:
            upstream.OBJECTS_PATH = str(OBJECTS)
            enriched = upstream.enrich_actors(actors)
        finally:
            upstream.OBJECTS_PATH = old_path
        absolute_enriched = upstream.enrich_actors(json.loads((META / "actor_list_absolute.json").read_text()))
        assert enriched == absolute_enriched, "Absolute/overlay bindings differ"
        write_json(META / "enriched_actor_list.json", enriched)
        spec = importlib.util.spec_from_file_location("labware_official_episode_instructions", ROBOTWIN / "description/utils/generate_episode_instructions.py")
        description_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(description_module)
        rows = []
        for i, (asset_id, entry) in enumerate(registry["assets"].items()):
            data = json.loads(Path(entry["model_data"]).read_text())
            source = json.loads((INPUT / "prepared" / asset_id / "model_data0.json").read_text())
            points = json.loads(Path(entry["points_info"]).read_text())
            assert enriched["self." + asset_id]["functional_points"] == points["functional_points"]
            assert enriched["self." + asset_id]["contact_points"] == points["contact_points"]
            assert "modelname" not in enriched["self." + asset_id]
            assert [p["id"] for p in points["functional_points"]] == list(range(len(data["functional_matrix"])))
            assert [p["id"] for p in points["contact_points"]] == list(range(len(data["contact_points_pose"])))
            assert data["functional_matrix"][:-1] == source["functional_matrix"]
            for key in ("contact_points_pose", "target_pose", "orientation_point", "scale"):
                assert data[key] == source[key], "Original key changed: " + key
            # Real SAPIEN Entity and official Actor transformation; no GPU/mesh needed.
            entity = sapien.Entity()
            entity.name = "metadata_fixture_" + asset_id
            pose = sapien.Pose([.25 + .01 * i, -.12, .77], t3d.euler.euler2quat(.07, -.11, .35))
            entity.set_pose(pose)
            wrapped = Actor(entity, data)
            checks = 0
            max_error = 0.0
            for point_type, key in Actor.POINTS.items():
                for idx, mat in enumerate(data[key]):
                    mat = np.array(mat, dtype=float)
                    assert mat.shape == (4, 4) and np.isfinite(mat).all()
                    assert np.allclose(mat[3], [0, 0, 0, 1])
                    assert np.allclose(mat[:3, :3].T @ mat[:3, :3], np.eye(3), atol=1e-7)
                    assert abs(np.linalg.det(mat[:3, :3]) - 1) < 1e-7
                    scaled = mat.copy()
                    scaled[:3, 3] *= data["scale"]
                    expected = pose.to_transformation_matrix() @ scaled
                    actual = wrapped.get_point(point_type, idx, "matrix")
                    error = float(np.max(np.abs(expected - actual)))
                    assert error < 1e-7
                    max_error = max(max_error, error)
                    checks += 1
            description = json.loads(Path(entry["object_description"]).read_text())
            params = {"{object}": entry["episode_description_value"], "{a}": "left"}
            template = "Pick up {object} with {a}."
            seen = description_module.replace_placeholders(template, params, rng=random.Random(42))
            unseen = description_module.replace_placeholders_unseen(template, params, rng=random.Random(42))
            assert "{" not in seen + unseen
            assert any("the " + phrase in seen for phrase in description["seen"])
            assert any("the " + phrase in unseen for phrase in description["unseen"])
            assert "the left arm" in seen and "the left arm" in unseen
            rows.append({"asset_id": asset_id, "passed": True, "functional_count": len(data["functional_matrix"]),
                         "contact_count": len(data["contact_points_pose"]), "new_bottom_id": entry["functional_ids"]["bottom"],
                         "original_IDs_preserved": True, "original_enrich_actors_semantics_read": True,
                         "original_Actor_world_matrix_checks": checks, "max_absolute_matrix_error": max_error,
                         "original_instruction_placeholder_check": {"seen": seen, "unseen": unseen, "passed": True}})
        assert source_fingerprints() == registry["frozen_input_fingerprints"]
        old_description_parent = description_module.parent_directory
        description_module.parent_directory = str(META / "description/utils")
        episode_params = {"{A}": registry["assets"]["reagent_bottle_body"]["episode_description_value"], "{a}": "right"}
        try:
            task_descriptions = description_module.generate_episode_descriptions("labware_pick_place", [episode_params], max_descriptions=12, seed=42)
        finally:
            description_module.parent_directory = old_description_parent
        assert len(task_descriptions) == 1
        assert len(task_descriptions[0]["seen"]) == len(task_descriptions[0]["unseen"]) == 12
        for split in ("seen", "unseen"):
            texts = task_descriptions[0][split]
            assert len(set(texts)) == 12
            assert all("{" not in x and "}" not in x and "the right arm" in x for x in texts)
        write_json(META / "task_instruction_validation.json", {
            "date": "2026-10-22", "passed": True, "actual_function": "generate_episode_instructions.generate_episode_descriptions",
            "parent_directory_binding": str(META / "description/utils"), "episode_params": episode_params,
            "max_descriptions": 12, "seed": 42, "seen_count": 12, "unseen_count": 12, "all_placeholders_resolved": True,
            "outputs": task_descriptions, "task_info": str(META / "labware_pick_place_task_info.json"),
            "source_file_sha256": sha(ROBOTWIN / "description/utils/generate_episode_instructions.py"),
        })
        receipt = {"date": "2026-10-22", "passed": True, "tests": rows,
                   "upstream": {"commit": subprocess.check_output(["git", "-C", str(ROBOTWIN), "rev-parse", "HEAD"], text=True).strip(),
                                "enrich_actors_file": str(ROBOTWIN / "code_gen/test_gen_code.py"),
                                "enrich_actors_file_sha256": sha(ROBOTWIN / "code_gen/test_gen_code.py"),
                                "Actor_file_sha256": sha(ROBOTWIN / "envs/utils/actor_utils.py"),
                                "episode_instructions_file_sha256": sha(ROBOTWIN / "description/utils/generate_episode_instructions.py")},
                   "relative_overlay_binding_and_absolute_modelname_match": True, "frozen_input_files_unchanged": True,
                   "concrete_task_instruction_seen_unseen_counts": [12, 12], "concrete_task_info_actor_count": 1,
                   "geometry_loader_verified_here": False, "robot_task_verified_here": False,
                   "scope": "Actual upstream enrich_actors reads all six points_info files; actual upstream Actor transforms every point on actual SAPIEN Entity. Mesh/grasp/task/export validation are separate gates."}
        write_json(ROOT / "metadata_validation.json", receipt)
        return receipt
    finally:
        os.chdir(oldcwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    registry = prepare()
    if not args.prepare_only:
        receipt = validate_original(registry)
        print(json.dumps({"passed": receipt["passed"], "asset_count": len(receipt["tests"]), "output": str(ROOT / "metadata_validation.json")}, ensure_ascii=False))
    else:
        print(META / "registry.json")


if __name__ == "__main__":
    main()
