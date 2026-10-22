"""Unanchored dynamic receiver using the frozen, imported petri geometry.

This loader has no fixture, kinematic target or post-initialization pose write.
The future robot task must use the receiver's current physical pose when
checking complete capsule containment, because the 30-g dish may move.
"""
from pathlib import Path
import hashlib
import json
import sapien

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "metadata" / "registry.json"
SPEC = ROOT.parent / "2026-10-22-labware-integration" / "physics_spec.json"
SPEC_SHA = "9d5ae8d89d60dd2d4e466a966c3862d7d160cc2e5ebe4d61bc8a1eac9a2d59ef"


def load_dynamic_petri(scene, pose, *, visual=False, name="labware_receiver_petri"):
    if hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA:
        raise RuntimeError("Frozen receiver dimensional specification changed")
    entry = json.loads(REGISTRY.read_text())["assets"]["petri_dish"]
    metadata = json.loads(Path(entry["model_data"]).read_text())
    if metadata["scale"] != [1., 1., 1.]:
        raise RuntimeError("Unexpected petri scale")
    model = Path(entry["absolute_modelname"])
    files = [model / "collision" / "bottom.obj", *sorted((model / "collision").glob("wall_*.obj"))]
    if len(files) != 49 or not all(p.is_file() for p in files):
        raise RuntimeError("Expected the frozen bottom and 48 open-ring convex pieces")
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source_hashes[str(Path(entry["model_data"]))] = hashlib.sha256(Path(entry["model_data"]).read_bytes()).hexdigest()
    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(.35, .30, .02)
    for path in files:
        builder.add_convex_collision_from_file(str(path), material=material, density=1100)
    visual_path = model / "visual" / "base0.glb"
    source_hashes[str(visual_path)] = hashlib.sha256(visual_path.read_bytes()).hexdigest()
    if visual:
        builder.add_visual_from_file(str(visual_path))
    builder.initial_pose = pose
    entity = builder.build(name)
    component = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    component.mass = float(entry["mass_kg"])
    spec = json.loads(SPEC.read_text())
    component.solver_position_iterations = spec["physics"]["solver_position_iterations"]
    component.solver_velocity_iterations = spec["physics"]["solver_velocity_iterations"]
    for shape in component.collision_shapes:
        shape.contact_offset = spec["physics"]["contact_offset_m"]
        shape.rest_offset = spec["physics"]["rest_offset_m"]
    contract = {"actor_name": entity.name, "actor_id": int(entity.per_scene_id),
                "dynamic": not component.kinematic,
                "mass_readback_kg": float(component.mass),
                "mass_status": "Retained 30-g engineering assignment, not a measurement",
                "collision_parts_actual": len(component.collision_shapes),
                "cavity_frame": "actual entity pose; bottom z=0, floor z=0.001m, rim z=0.014m",
                "cavity_dimensions": spec["receiver"],
                "contact_offset_m": spec["physics"]["contact_offset_m"],
                "rest_offset_m": spec["physics"]["rest_offset_m"],
                "physics_dt_s_actual": float(scene.get_timestep()),
                "world_fixture_or_holder_created": False,
                "states_written_after_initialization": False,
                "visual_loaded": bool(visual), "metadata": metadata,
                "source_sha256": source_hashes,
                "frozen_spec_sha256": SPEC_SHA}
    return entity, contract
