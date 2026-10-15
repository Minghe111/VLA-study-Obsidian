"""CPU audit of native RealMan path using copies of actual PhysX geometry.

No SAPIEN qpos, drive, collision mask, or scene.step is modified. FK is rebuilt
from the current articulation, so the fixed head and all 36 active joints match
this scene. Only the already configured native rigid/adjacent exemptions apply.

Usage in the existing worker:
    from full_mesh_path_audit import audit_path, validate_scene_fk
    audit = audit_path(task, planned_path, "left")
A path is an N x 6 arm array, N x full-DOF array, or dict with "position".
Hands and the other arm stay at the measured starting state for an arm-only path.
The conservative subdivision bound applies to the audited linear interpolation;
actual drive dynamics still require a separate physics trace / object guard.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np


def _array(a):
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def _pose_matrix(pose):
    return _array(pose.to_transformation_matrix())


def _link_vertices(link):
    """Actual collision mesh vertices, or containing primitive box corners."""
    chunks = []
    types = []
    signs = np.asarray([[x, y, z] for x in (-1., 1.)
                        for y in (-1., 1.) for z in (-1., 1.)])
    for shape in link.collision_shapes:
        name = type(shape).__name__
        types.append(name)
        if hasattr(shape, "vertices"):
            vertices = _array(shape.vertices) * _array(shape.scale)
        elif hasattr(shape, "half_size"):
            vertices = signs * _array(shape.half_size)
        elif "Capsule" in name or "Cylinder" in name:
            # Native PhysX primitives have their long axis along x.
            extra = float(shape.radius) if "Capsule" in name else 0.
            vertices = signs * [float(shape.half_length) + extra,
                                float(shape.radius), float(shape.radius)]
        elif "Sphere" in name:
            vertices = signs * float(shape.radius)
        else:
            raise TypeError("Unsupported robot collision geometry: " + name)
        local = _pose_matrix(shape.local_pose)
        chunks.append(vertices @ local[:3, :3].T + local[:3, 3])
    return (np.concatenate(chunks) if chunks else np.zeros((0, 3))), types


def _aabb_gap(a, b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.)))


def _convert_component_scaled(component, native_convert, fcl, Pose):
    """Use the installed converter, explicitly handling its unsupported mesh scale.

    PhysX exposes raw mesh vertices and a separate scale. Scaling precedes the
    shape's local pose. No shape or entity in SAPIEN is modified.
    """
    if not any(hasattr(s, "vertices") and not np.allclose(_array(s.scale), 1., atol=1e-7)
               for s in component.collision_shapes):
        return native_convert(component)
    shapes, shape_poses = [], []
    for shape in component.collision_shapes:
        name = type(shape).__name__
        local = Pose(shape.local_pose)
        if "ConvexMesh" in name:
            geometry = fcl.Convex(vertices=_array(shape.vertices) * _array(shape.scale),
                                  faces=np.asarray(shape.triangles, dtype=np.int32))
        elif "TriangleMesh" in name:
            geometry = fcl.BVHModel(); geometry.begin_model()
            geometry.add_sub_model(vertices=_array(shape.vertices) * _array(shape.scale),
                                   faces=np.asarray(shape.triangles, dtype=np.int32))
            geometry.end_model()
        elif "Box" in name:
            geometry = fcl.Box(side=_array(shape.half_size) * 2.)
        elif "Capsule" in name or "Cylinder" in name:
            cls = fcl.Capsule if "Capsule" in name else fcl.Cylinder
            geometry = cls(radius=float(shape.radius), lz=float(shape.half_length) * 2.)
            local = local * Pose(q=[math.sqrt(.5), 0., math.sqrt(.5), 0.])
        elif "Sphere" in name:
            geometry = fcl.Sphere(radius=float(shape.radius))
        else:
            raise TypeError("Unsupported scaled collision component geometry " + name)
        shapes.append(fcl.CollisionObject(geometry)); shape_poses.append(local)
    if not shapes:return None
    return fcl.FCLObject(component.name, Pose(component.entity.pose), shapes, shape_poses)

def joint_limit_audit(full_path_with_start, limits, joint_names, commanded_indices,
                      commanded_tolerance_rad=1e-5, measured_tolerance_rad=1e-4):
    """Separate commanded path limits from unchanged measured/start roundoff.

    Row zero is the measured start, remaining rows are command path endpoints.
    Nx6 paths command six joints; Nx36 paths command all joints. Nothing is
    clipped. Since interpolation is linear, endpoint limits cover all substeps.
    """
    q = _array(full_path_with_start)
    lim = _array(limits)
    command = list(commanded_indices)
    frozen = [i for i in range(q.shape[1]) if i not in command]
    delta = np.maximum(np.maximum(lim[:, 0] - q, q - lim[:, 1]), 0.)
    def metric(rows, columns, tolerance):
        if not rows or not columns:
            return {"excess_rad": 0., "tolerance_rad": tolerance,
                    "passed": True, "worst": None}
        values = delta[np.ix_(rows, columns)]
        ri, ci = np.unravel_index(int(values.argmax()), values.shape)
        row, col = rows[ri], columns[ci]
        excess = float(values[ri, ci])
        return {"excess_rad": excess, "tolerance_rad": tolerance,
                "passed": bool(excess <= tolerance),
                "worst": {"joint": joint_names[col], "joint_index": col,
                          "source_row_with_start": row, "path_index": row - 1,
                          "sample_index": None, "position_rad": float(q[row, col]),
                          "lower_rad": float(lim[col, 0]), "upper_rad": float(lim[col, 1]),
                          "excess_rad": excess}}
    command_result = metric(list(range(1, len(q))), command, commanded_tolerance_rad)
    frozen_result = metric(list(range(1, len(q))), frozen, measured_tolerance_rad)
    start_result = metric([0], list(range(q.shape[1])), measured_tolerance_rad)
    overall = metric(list(range(len(q))), list(range(q.shape[1])), measured_tolerance_rad)
    return {"passed": bool(command_result["passed"] and frozen_result["passed"] and start_result["passed"]),
            "commanded": command_result, "frozen_measured": frozen_result,
            "start_measured": start_result, "overall": overall,
            "commanded_joints": [joint_names[i] for i in command],
            "frozen_joints": [joint_names[i] for i in frozen],
            "scope": "Command endpoints strictly checked; measured start and unchanged joints have separate numerical tolerance; no qpos clipping."}

class FullMeshPathAuditor:
    def __init__(self, task, *, fk_position_tolerance_m=2e-5,
                 fk_rotation_tolerance_rad=2e-4, check_table=True):
        from mplib import ArticulatedModel, Pose
        from mplib.collision_detection import fcl
        from mplib.sapien_utils.conversion import SapienPlanningWorld
        from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
        from sapien.physx import PhysxRigidBaseComponent
        self.task, self.art = task, task.robot.left_entity
        if self.art is not task.robot.right_entity:
            raise ValueError("Expected one native dual-arm articulation")
        self.fcl, self.Pose = fcl, Pose
        self.sim_joints = [j.name for j in self.art.active_joints]
        self.links = {link.name: link for link in self.art.links}
        self.initial_sim_q = _array(self.art.qpos).copy()
        self.initial_scene_poses = {n: _pose_matrix(l.entity.pose) for n, l in self.links.items()}
        self.urdf = export_kinematic_chain_urdf(self.art)
        self.urdf_tree = ET.fromstring(self.urdf)
        self.allowed = set()
        physics = getattr(task.robot, "physics_audit", None)
        if not physics or "fixed_components" not in physics or "exemptions" not in physics:
            raise ValueError("Missing native physics_audit; refusing inferred self-collision exemptions")
        for group in physics["fixed_components"]:
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    self.allowed.add(frozenset((a, b)))
        for exemption in physics["exemptions"]:
            a, b = exemption["components"]
            for first in a:
                for second in b:
                    if first != second:
                        self.allowed.add(frozenset((first, second)))
        self.native_exemption_count = len(self.allowed)
        self.engine_adjacent = []
        self.engine_adjacent_additional = []
        # PhysX reduced-coordinate articulations always filter direct parent-child
        # link collisions. This does not depend on custom shape collision groups.
        # The installed mplib.sapien_utils.srdf_exporter exports exactly these
        # pairs as reason="Adjacent" before checking shape groups (lines 53-61).
        for link in self.art.links:
            parent_link = link.parent
            if parent_link is None or not len(link.collision_shapes) or not len(parent_link.collision_shapes):
                continue
            pair = frozenset((link.name, parent_link.name))
            self.engine_adjacent.append(sorted(pair))
            if pair not in self.allowed:
                self.engine_adjacent_additional.append(sorted(pair))
            self.allowed.add(pair)
        srdf = ET.Element("robot", name=self.urdf_tree.get("name"))
        for pair in sorted((sorted(p) for p in self.allowed)):
            ET.SubElement(srdf, "disable_collisions", link1=pair[0], link2=pair[1], reason="Existing native physics configuration")
        native_convert = SapienPlanningWorld.convert_physx_component
        convert = lambda comp: _convert_component_scaled(comp, native_convert, fcl, Pose)
        colliders = [obj for link in self.art.links if (obj := convert(link)) is not None]
        self.model = ArticulatedModel.create_from_urdf_string(
            self.urdf, ET.tostring(srdf, encoding="unicode"), colliders,
            link_names=list(self.links), joint_names=self.sim_joints, verbose=False)
        self.model.set_base_pose(Pose(self.art.root_pose))
        self.model.set_qpos(self.initial_sim_q, full=True)
        self.old_model_q = _array(self.model.get_qpos()).copy()
        self.pk = self.model.get_pinocchio_model()
        if list(self.pk.get_joint_names()) != self.sim_joints:
            raise ValueError("Auxiliary FK joint ordering differs from supplied SAPIEN joint names")
        self.objects = {obj.name: obj for obj in self.model.get_fcl_model().get_collision_objects()}
        self.vertices = {}
        geometry_types = {}
        for n in self.objects:
            self.vertices[n], geometry_types[n] = _link_vertices(self.links[n])
        if set(self.objects) != {n for n, l in self.links.items() if len(l.collision_shapes)}:
            raise ValueError("Auxiliary FCL omitted a physical collision link")
        self.fk_audit = self.validate_fk(fk_position_tolerance_m, fk_rotation_tolerance_rad)
        self.static = {}
        self.static_boxes = {}
        bottle_component = task.bottle.actor.find_component_by_type(PhysxRigidBaseComponent)
        if bottle_component is None:
            raise ValueError("Missing native bottle rigid component")
        self.static["bottle"] = convert(bottle_component)
        if self.static["bottle"] is None or len(self.static["bottle"].shapes) != 7:
            raise ValueError("Expected the official bottle's seven real convex collision shapes")
        v, _ = _link_vertices(bottle_component)
        t = _pose_matrix(bottle_component.entity.pose)
        w = v @ t[:3, :3].T + t[:3, 3]
        self.static_boxes["bottle"] = np.stack([w.min(0), w.max(0)])
        if check_table:
            entities = [e for e in task.scene.get_all_actors() if e.name == "table"]
            if len(entities) != 1:
                raise ValueError("Expected exactly one table actor")
            component = entities[0].find_component_by_type(PhysxRigidBaseComponent)
            self.static["table"] = convert(component)
            v, _ = _link_vertices(component)
            t = _pose_matrix(component.entity.pose)
            w = v @ t[:3, :3].T + t[:3, 3]
            self.static_boxes["table"] = np.stack([w.min(0), w.max(0)])
        names = sorted(self.objects)
        self.self_pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                           if frozenset((a, b)) not in self.allowed]
        self.motion_coefficients = self._motion_coefficients()
        self.distance_request = fcl.DistanceRequest(enable_signed_distance=False)
        self.collision_request = fcl.CollisionRequest()
        self.metadata = {
            "geometry_source": "Current actual PhysX collision shapes copied through installed MPlib converter",
            "full_dof": len(self.sim_joints), "collision_links": len(self.objects),
            "collision_shapes": sum(len(o.shapes) for o in self.objects.values()),
            "geometry_types": geometry_types, "bottle_shapes": len(self.static["bottle"].shapes),
            "native_exemption_pairs": self.native_exemption_count,
            "engine_adjacent_pairs": self.engine_adjacent,
            "engine_adjacent_additional_pairs": self.engine_adjacent_additional,
            "engine_adjacent_source": "Installed mplib.sapien_utils.srdf_exporter.py:53-61 and NVIDIA PhysX articulation parent-child filtering",
            "engine_adjacent_source_url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/articulation_stability_guide.html",
            "effective_exemption_pairs": len(self.allowed), "checked_self_pairs": len(self.self_pairs),
            "excluded_environment": ["floor contact of wheels/base; planner table separately checked"],
            "urdf_kinematic_export_sha256": hashlib.sha256(self.urdf.encode()).hexdigest(),
            "fk_audit": self.fk_audit,
        }

    def validate_fk(self, position_tol=2e-5, rotation_tol=2e-4):
        self.pk.compute_forward_kinematics(self.initial_sim_q)
        root = _pose_matrix(self.art.root_pose)
        checks = []
        for index, name in enumerate(self.pk.get_link_names()):
            actual = self.initial_scene_poses[name]
            expected = root @ _pose_matrix(self.pk.get_link_pose(index))
            position = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
            r = expected[:3, :3] @ actual[:3, :3].T
            rotation = float(math.acos(np.clip((np.trace(r) - 1.) / 2., -1., 1.)))
            # Input SAPIEN matrices are float32; project rotation by SVD first.
            u, _, vt = np.linalg.svd(r)
            rotation = float(math.acos(np.clip((np.trace(u @ vt) - 1.) / 2., -1., 1.)))
            checks.append({"link": name, "position_error_m": position, "rotation_error_rad": rotation})
        worst_p = max(checks, key=lambda x: x["position_error_m"])
        worst_r = max(checks, key=lambda x: x["rotation_error_rad"])
        for name, obj in self.objects.items():
            expected = self.initial_scene_poses[name]
            delta = np.max(np.abs(_pose_matrix(obj.pose) - expected))
            if delta > 2e-4:
                raise ValueError(f"FCL global link pose differs for {name}: {delta}")
        if worst_p["position_error_m"] > position_tol or worst_r["rotation_error_rad"] > rotation_tol:
            raise ValueError(f"FK / scene mismatch: {worst_p}, {worst_r}")
        return {"passed": True, "checked_links": len(checks), "worst_position": worst_p,
                "worst_rotation": worst_r, "position_tolerance_m": position_tol,
                "rotation_tolerance_rad": rotation_tol}

    def _motion_coefficients(self):
        """Conservative vertex displacement bound for linear joint interpolation.

        For a revolute ancestor, radius <= sum of downstream translation lengths
        plus the furthest collider vertex norm in its link frame. Thus every
        vertex moves no more than sum(radius_j * abs(delta_q_j)).
        """
        parent = {}
        for joint in self.urdf_tree.findall("joint"):
            child = joint.find("child").get("link")
            origin = joint.find("origin")
            length = np.linalg.norm(np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")) if origin is not None else 0.
            parent[child] = (joint.find("parent").get("link"), joint.get("name"), joint.get("type"), float(length))
        result = {}
        for link, vertices in self.vertices.items():
            coefficients = np.zeros(len(self.sim_joints))
            radius = float(np.linalg.norm(vertices, axis=1).max())
            child = link
            while child in parent:
                next_link, joint, kind, length = parent[child]
                if joint in self.sim_joints:
                    if kind in ("revolute", "continuous"):
                        coefficients[self.sim_joints.index(joint)] = radius
                    elif kind == "prismatic":
                        raise ValueError("Prismatic downstream reach needs an explicit displacement bound")
                    else:
                        raise ValueError("Unsupported active joint type " + kind)
                radius += length
                child = next_link
            result[link] = coefficients
        return result

    def _distance(self, first, second):
        collided = bool(self.fcl.collide(first, second, self.collision_request).is_collision())
        value = float(self.fcl.distance(first, second, self.distance_request).min_distance)
        if not np.isfinite(value):
            raise RuntimeError("FCL returned a non-finite distance")
        return (0. if collided else max(value, 0.)), collided

    def audit(self, path, side="left", *, vertex_step_m=.001,
              min_bottle_clearance_m=.003, min_self_clearance_m=0.,
              min_table_clearance_m=0., max_samples=20000, include_samples=False):
        started = time.monotonic()
        if side not in ("left", "right") or not 0. < vertex_step_m <= .001:
            raise ValueError("side must be left/right and vertex step must be <= 1 mm")
        if isinstance(path, dict):
            path = path["position"]
        arm_path = _array(path)
        if arm_path.ndim != 2 or not len(arm_path) or not np.isfinite(arm_path).all():
            raise ValueError("Expected a finite nonempty N x joint array")
        base = _array(self.art.qpos).copy()
        if np.max(np.abs(base - self.initial_sim_q)) > 1e-8:
            raise ValueError("Articulation changed since this auditor was constructed; rebuild it")
        names = list(getattr(self.task.robot, side + "_arm_joints_name"))
        indices = [self.sim_joints.index(n) for n in names]
        if arm_path.shape[1] == len(indices):
            full = np.tile(base, (len(arm_path), 1));full[:, indices] = arm_path
            commanded_indices = indices
        elif arm_path.shape[1] == len(base):
            full = arm_path.copy()
            commanded_indices = list(range(len(base)))
        else:
            raise ValueError("Path must contain six arm joints or all native joints")
        # Include measured start, so the unrecorded jump to row zero is checked too.
        full = np.vstack([base, full])
        limits = np.concatenate([_array(j.limits).reshape(-1, 2) for j in self.art.active_joints])
        limit_audit = joint_limit_audit(full, limits, self.sim_joints, commanded_indices)
        excess = limit_audit["overall"]["excess_rad"]
        samples = [(full[0], -1, 0.)]
        interval_bounds = []
        max_vertex_bound = 0.
        for original_index, (q0, q1) in enumerate(zip(full[:-1], full[1:])):
            delta = np.abs(q1 - q0)
            bounds = {n: float(c @ delta) for n, c in self.motion_coefficients.items()}
            pieces = max(1, int(math.ceil(max(bounds.values(), default=0.) / vertex_step_m)))
            if len(samples) + pieces > max_samples:
                raise ValueError("Adaptive subdivision exceeded max_samples; refusing incomplete audit")
            split_bounds = {n: b / pieces for n, b in bounds.items()}
            max_vertex_bound = max(max_vertex_bound, max(split_bounds.values(), default=0.))
            for k in range(1, pieces + 1):
                interval_bounds.append(split_bounds)
                samples.append((q0 + (q1 - q0) * k / pieces, original_index, k / pieces))
        interval_bounds.append({n: 0. for n in self.objects})
        endpoint_sample_indices = {-1: 0}
        for index, (_, path_index, fraction) in enumerate(samples):
            if fraction == 1.:endpoint_sample_indices[path_index] = index
        for group in ("commanded", "frozen_measured", "start_measured", "overall"):
            if limit_audit[group]["worst"] is not None:
                worst = limit_audit[group]["worst"]
                worst["sample_index"] = endpoint_sample_indices[worst["path_index"]]
        categories = [*self.static, "self"]
        best = {n: {"distance_m": math.inf} for n in categories}
        certified = {n: {"lower_bound_m": math.inf} for n in categories}
        events, sample_records = [], []
        seen_collisions = set(); collision_occurrences = 0
        static_cache = {}
        moving = {n: bool(np.any(np.abs(full - full[0]) @ c > 1e-12)) for n, c in self.motion_coefficients.items()}
        try:
            for sample_index, ((q, path_index, fraction), outgoing) in enumerate(zip(samples, interval_bounds)):
                self.model.set_qpos(q, full=True)
                boxes = {}
                for n, obj in self.objects.items():
                    t = _pose_matrix(obj.pose)
                    v = self.vertices[n] @ t[:3, :3].T + t[:3, 3]
                    boxes[n] = np.stack([v.min(0), v.max(0)])
                stamp = {"sample_index": sample_index, "path_index": path_index, "interval_fraction": fraction}
                row_min = {n: math.inf for n in categories}
                for category, obstacle in self.static.items():
                    for link, obj in self.objects.items():
                        lower = _aabb_gap(boxes[link], self.static_boxes[category])
                        motion = outgoing[link]
                        if lower > 0. and lower > best[category]["distance_m"] and lower - motion > certified[category]["lower_bound_m"]:
                            continue
                        key = (category, link)
                        if not moving[link] and key in static_cache:
                            gap, collision = static_cache[key]
                        else:
                            gap, collision = self._distance(obj, obstacle)
                            if not moving[link]:static_cache[key] = gap, collision
                        row_min[category] = min(row_min[category], gap)
                        item = {**stamp, "link": link, "other": category}
                        if gap < best[category]["distance_m"]:best[category] = {**item, "distance_m": gap}
                        if gap - motion < certified[category]["lower_bound_m"]:
                            certified[category] = {**item, "lower_bound_m": gap - motion, "interval_vertex_bound_m": motion}
                        if collision:
                            collision_occurrences += 1; collision_key = (category, link)
                            if collision_key not in seen_collisions:
                                seen_collisions.add(collision_key)
                                if len(events) < 64:events.append({**item, "category": category})
                for first, second in self.self_pairs:
                    lower = _aabb_gap(boxes[first], boxes[second])
                    motion = outgoing[first] + outgoing[second]
                    if lower > 0. and lower > best["self"]["distance_m"] and lower - motion > certified["self"]["lower_bound_m"]:
                        continue
                    key = ("self", first, second)
                    if not moving[first] and not moving[second] and key in static_cache:
                        gap, collision = static_cache[key]
                    else:
                        gap, collision = self._distance(self.objects[first], self.objects[second])
                        if not moving[first] and not moving[second]:static_cache[key] = gap, collision
                    row_min["self"] = min(row_min["self"], gap)
                    item = {**stamp, "link": first, "other": second}
                    if gap < best["self"]["distance_m"]:best["self"] = {**item, "distance_m": gap}
                    if gap - motion < certified["self"]["lower_bound_m"]:
                        certified["self"] = {**item, "lower_bound_m": gap - motion, "interval_vertex_bound_m": motion}
                    if collision:
                        collision_occurrences += 1; collision_key = ("self", first, second)
                        if collision_key not in seen_collisions:
                            seen_collisions.add(collision_key)
                            if len(events) < 64:events.append({**item, "category": "self"})
                if include_samples:
                    sample_records.append({**stamp, "evaluated_minima_m": {n: (v if math.isfinite(v) else None) for n, v in row_min.items()}})
            thresholds = {"bottle": min_bottle_clearance_m, "self": min_self_clearance_m, "table": min_table_clearance_m}
            passed = limit_audit["passed"] and not events and all(certified[n]["lower_bound_m"] >= thresholds[n] for n in categories)
            result = {"passed": bool(passed), "side": side, "metadata": self.metadata,
                      "path_rows": len(arm_path), "audited_samples": len(samples),
                      "max_vertex_displacement_bound_m": max_vertex_bound,
                      "max_joint_limit_excess_rad": excess,
                      "joint_limit_audit": limit_audit,
                      "commanded_excess_rad": limit_audit["commanded"]["excess_rad"],
                      "frozen_excess_rad": limit_audit["frozen_measured"]["excess_rad"],
                      "start_excess_rad": limit_audit["start_measured"]["excess_rad"],
                      "sampled_minimum_gap": best, "continuous_gap_lower_bound": certified,
                      "thresholds_m": {n: thresholds[n] for n in categories},
                      "collisions_first_64": events, "unique_collision_pairs": len(seen_collisions),
                      "collision_pair_sample_occurrences": collision_occurrences,
                      "elapsed_s": time.monotonic() - started,
                      "scope": "Complete native collision geometry; linear interpolation and frozen uncommanded joints. Actual drive motion requires physics guards."}
            if include_samples:result["samples"] = sample_records
            return result
        finally:
            self.model.set_qpos(self.old_model_q, full=True)
            if not np.array_equal(_array(self.art.qpos), self.initial_sim_q):
                raise RuntimeError("Physical articulation qpos changed during CPU audit")
            if any(np.max(np.abs(_pose_matrix(l.entity.pose) - self.initial_scene_poses[n])) > 1e-8 for n, l in self.links.items()):
                raise RuntimeError("Physical link pose changed during CPU audit")


def audit_path(task, path, side="left", **kwargs):
    """Build a fresh CPU model and audit an arm/full-state path; no physics steps."""
    check_table = kwargs.pop("check_table", True)
    return FullMeshPathAuditor(task, check_table=check_table).audit(path, side, **kwargs)


def validate_scene_fk(task):
    """Validate every FK link plus FCL global poses against the current scene."""
    return FullMeshPathAuditor(task, check_table=False).fk_audit