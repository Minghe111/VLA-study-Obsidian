"""Read actual contact impulse and its moment on an entity, without actuation.

SAPIEN 3.0.0b1 reports impulse on contact.bodies[0]. The installed-runtime
momentum check is in contact_wrench_component_check.py. These quantities do
not include gravity, joint constraints, or the passive thread surrogate.
"""
import numpy as np


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Expected finite 3-vector: "+name)
    return value


def read_contact_wrench(scene, actor, *, origin_world_m, axis_world):
    """Instantaneous post-step contacts, using actual dt and entity identities.

    Axis is a measured direction, normally body +Z; origin is normally the
    measured cap center. Positive axial moment follows the right-hand rule.
    No contact-force, opposition, grasp, or cap-opening success is inferred.
    """
    entity = getattr(actor, "actor", actor)
    actor_id = int(entity.per_scene_id)
    origin = _vector(origin_world_m, "origin")
    axis = _vector(axis_world, "axis")
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("Axis must be nonzero")
    axis = axis/norm
    dt = float(scene.get_timestep())
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Expected positive actual physical timestep")
    groups = {key: {"force_world_N": np.zeros(3),
                    "normal_force_world_N": np.zeros(3),
                    "tangent_force_world_N": np.zeros(3),
                    "moment_world_Nm": np.zeros(3),
                    "normal_moment_world_Nm": np.zeros(3),
                    "tangent_moment_world_Nm": np.zeros(3)}
              for key in ("left_hand", "right_hand", "other", "all")}
    rows = []
    for contact in scene.get_contacts():
        bodies = [c.entity for c in contact.bodies]
        ids = [int(b.per_scene_id) for b in bodies]
        if actor_id not in ids:
            continue
        target_index = ids.index(actor_id)
        other = bodies[1-target_index]
        sign = 1. if target_index == 0 else -1.
        group = ("left_hand" if other.name.startswith("l_inspire_") else
                 "right_hand" if other.name.startswith("r_inspire_") else "other")
        for point in contact.points:
            raw_impulse = _vector(point.impulse, "impulse")
            normal = _vector(point.normal, "normal")
            if abs(float(np.linalg.norm(normal))-1.) > 1e-5:
                raise ValueError("Nonunit actual contact normal")
            position = _vector(point.position, "position")
            separation = float(point.separation)
            if not np.isfinite(separation):
                raise ValueError("Nonfinite actual contact separation")
            force = sign*raw_impulse/dt
            normal_force = sign*float(raw_impulse@normal)*normal/dt
            tangent_force = force-normal_force
            arm = position-origin
            moments = [np.cross(arm, f) for f in (force, normal_force, tangent_force)]
            values = [force, normal_force, tangent_force, *moments]
            for category in (group, "all"):
                for key, value in zip(groups[category], values):
                    groups[category][key] += value
            # Zero-impulse manifolds are retained only as counts in upstream
            # geometry logs; never report them as a loaded contact here.
            if float(np.linalg.norm(raw_impulse)) > 0:
                rows.append({"other_actor_id": int(other.per_scene_id),
                             "other_actor_name": other.name,
                             "target_contact_body_index": target_index,
                             "position_world_m": position.tolist(),
                             "separation_m": separation,
                             "impulse_on_target_world_Ns": (sign*raw_impulse).tolist(),
                             "normal_force_world_N": normal_force.tolist(),
                             "tangent_force_world_N": tangent_force.tolist(),
                             "axis_moment_Nm": float(moments[0]@axis),
                             "tangent_axis_moment_Nm": float(moments[2]@axis)})
    output = {}
    for group, values in groups.items():
        output[group] = {key: value.tolist() for key, value in values.items()}
        output[group].update(axis_moment_Nm=float(values["moment_world_Nm"]@axis),
                             normal_axis_moment_Nm=float(values["normal_moment_world_Nm"]@axis),
                             tangent_axis_moment_Nm=float(values["tangent_moment_world_Nm"]@axis))
    return {"actor_id": actor_id, "actor_name": entity.name,
            "physics_dt_s": dt, "origin_world_m": origin.tolist(),
            "axis_world": axis.tolist(), "groups": output, "loaded_points": rows,
            "scope": "Contact impulses only; excludes passive thread, joint reactions and gravity",
            "state_or_drive_writes": False, "grasp_or_opening_success_inferred": False}
