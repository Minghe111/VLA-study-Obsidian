"""In-memory exact relative-motion distance cache for frozen FMA free paths.

The unchanged frozen auditor still visits every original sample and pair and
uses its original AABB pruning, gap/bound arithmetic, limits and thresholds.
Only FCL self-pair distance/collision calls are reused when the URDF proves both
links have the same ordered chain of joints that actually change in this path.
A cache is scoped to one audit call and never persists across paths or scenes.
Carry auditing already has its own invariant cache and is left untouched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changing_ancestor_chains(tree, sim_joints, links, full_path):
    """No epsilon: any nonidentical endpoint marks a joint as changing.

    Linear interpolation cannot change a column whose endpoints are identical.
    Identical ordered ancestor chains mean every variable transform is shared;
    everything below their last shared variable ancestor is constant, so the
    two complete collision objects undergo one common rigid transformation.
    """
    names = list(sim_joints)
    q = np.asarray(full_path, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(names) or not len(q) or not np.isfinite(q).all():
        raise ValueError("Expected finite full-state endpoints including measured start")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate full-state joint names")
    changing = {names[i] for i in np.flatnonzero(np.any(q != q[0], axis=0))}
    parents = {}
    active = set()
    for joint in tree.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            raise ValueError("Incomplete exported URDF edge")
        child_name = child.get("link")
        if child_name in parents:
            raise ValueError("Exported URDF is not a tree")
        parents[child_name] = (parent.get("link"), joint.get("name"))
        if joint.get("type") not in ("fixed", "floating"):
            active.add(joint.get("name"))
    if active != set(names):
        raise ValueError("Exported active URDF joints differ from full-state ordering")
    chains = {}
    for link in links:
        chain, seen, node = [], set(), link
        while node in parents:
            if node in seen:
                raise ValueError("Cycle in exported URDF")
            seen.add(node)
            node, joint = parents[node]
            if joint in changing:
                chain.append(joint)
        chains[link] = tuple(reversed(chain))
    return changing, chains


def prepare_cache(auditor, path, side):
    raw = np.asarray(path["position"] if isinstance(path, dict) else path, dtype=np.float64)
    base = np.asarray(auditor.art.qpos, dtype=np.float64).copy()
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Expected finite nonempty path")
    arm_names = list(getattr(auditor.task.robot, side + "_arm_joints_name"))
    indices = [auditor.sim_joints.index(n) for n in arm_names]
    if raw.shape[1] == len(indices):
        full = np.tile(base, (len(raw), 1)); full[:, indices] = raw
    elif raw.shape[1] == len(base):
        full = raw.copy()
    else:
        raise ValueError("Path width is neither arm-only nor full-state")
    full = np.vstack([base, full])
    changing, chains = changing_ancestor_chains(auditor.urdf_tree, auditor.sim_joints,
                                                auditor.objects, full)
    eligible = {(a, b) for a, b in auditor.self_pairs if chains[a] == chains[b]}
    ids = {id(obj): name for name, obj in auditor.objects.items()}
    groups = Counter(chains.values())
    return {"eligible_pairs": eligible, "object_ids": ids, "values": {},
            "distance_calls": 0, "native_calls": 0, "hits": 0,
            "changing_joints": sorted(changing), "chains": chains,
            "rigid_group_sizes": sorted(groups.values(), reverse=True),
            "original_self_pairs": len(auditor.self_pairs),
            "verification_checks": 0, "verification_max_abs_gap_delta_m": 0.,
            "verification_collision_disagreements": 0}


def _receipt(state):
    return {"mode": "same_ordered_changing_ancestor_chain_FCL_result_cache",
            "cache_scoped_to_single_audit": True,
            "changing_joints_exact_no_epsilon": state["changing_joints"],
            "original_checked_self_pairs": state["original_self_pairs"],
            "eligible_invariant_self_pairs": len(state["eligible_pairs"]),
            "rigid_group_sizes": state["rigid_group_sizes"],
            "distance_calls": state["distance_calls"], "native_fcl_calls": state["native_calls"],
            "cache_hits": state["hits"], "cached_pair_count": len(state["values"]),
            "verification_checks": state["verification_checks"],
            "verification_max_abs_gap_delta_m": state["verification_max_abs_gap_delta_m"],
            "verification_collision_disagreements": state["verification_collision_disagreements"],
            "samples_pairs_shapes_thresholds_bounds_changed": False,
            "physical_scene_changed": False, "carry_auditor_changed": False}


def install_exact_distance_cache_overlay(legacy_dir, *, verify_hits=False,
                                         verification_tolerance_m=1e-6):
    """Install before dependent audit/planner imports in a NEW worker only.

    In verify mode every hit is independently recomputed through the original
    FCL routine, so this mode measures agreement, not runtime improvement.
    Floating FCL distances may differ at roundoff under shared rigid motion;
    verification reports the difference and refuses collision disagreement.
    """
    from geometry_overlay import install_native_shape_count_overlay
    legacy = Path(legacy_dir)
    source = legacy / "full_mesh_path_audit.py"
    receipt = {"source": str(source.resolve()), "source_sha256": sha(source),
               "overlay_source": str(Path(__file__).resolve()),
               "overlay_source_sha256": sha(__file__), "verify_hits": bool(verify_hits),
               "verification_tolerance_m": float(verification_tolerance_m),
               "distance_cache_only": True, "bounds_changed": False}
    if not np.isfinite(verification_tolerance_m) or verification_tolerance_m < 0:
        raise ValueError("Invalid verification tolerance")
    existing = sys.modules.get("full_mesh_path_audit")
    previous = getattr(existing, "rgb50_distance_cache_overlay_receipt", None)
    if previous is not None:
        if {k: v for k, v in previous.items() if k != "shape_count_overlay"} == receipt:
            return receipt
        raise RuntimeError("Do not change an installed overlay in an active worker")
    stale = [n for n in ("carry_mesh_path_audit", "full_native_carry_planner_v31",
                         "observed_full_trace_audit_v13", "full_native_approach") if n in sys.modules]
    if stale:
        raise RuntimeError("Install exact cache before dependent imports: " + str(stale))
    shape_receipt = install_native_shape_count_overlay(legacy)
    module = sys.modules["full_mesh_path_audit"]
    cls = module.FullMeshPathAuditor
    original_audit, original_distance = cls.audit, cls._distance

    def distance(self, first, second):
        state = getattr(self, "_rgb50_exact_distance_cache", None)
        if state is None:
            return original_distance(self, first, second)
        state["distance_calls"] += 1
        a, b = state["object_ids"].get(id(first)), state["object_ids"].get(id(second))
        key = tuple(sorted((a, b))) if a is not None and b is not None else None
        eligible = key in state["eligible_pairs"] if key is not None else False
        if eligible and key in state["values"]:
            state["hits"] += 1
            cached = state["values"][key]
            if verify_hits:
                state["native_calls"] += 1
                actual = original_distance(self, first, second)
                delta = abs(actual[0] - cached[0])
                state["verification_checks"] += 1
                state["verification_max_abs_gap_delta_m"] = max(delta, state["verification_max_abs_gap_delta_m"])
                state["verification_collision_disagreements"] += int(actual[1] != cached[1])
                if actual[1] != cached[1] or delta > verification_tolerance_m:
                    raise RuntimeError("Invariant FCL pair disagrees beyond verification tolerance: " + str((key, cached, actual)))
            return cached
        state["native_calls"] += 1
        value = original_distance(self, first, second)
        if eligible:
            state["values"][key] = value
        return value

    def audit(self, path, side="left", **kwargs):
        if getattr(self, "_rgb50_exact_distance_cache", None) is not None:
            raise RuntimeError("Nested cache audit is not supported")
        if side not in ("left", "right"):
            return original_audit(self, path, side, **kwargs)
        started = time.monotonic()
        state = prepare_cache(self, path, side)
        self._rgb50_exact_distance_cache = state
        try:
            result = original_audit(self, path, side, **kwargs)
            result["distance_cache_audit"] = _receipt(state)
            result["distance_cache_audit"]["overlay_receipt"] = receipt
            result["distance_cache_audit"]["elapsed_including_chain_setup_s"] = time.monotonic() - started
            return result
        finally:
            self.rgb50_last_distance_cache_audit = _receipt(state)
            del self._rgb50_exact_distance_cache

    cls._distance, cls.audit = distance, audit
    cls.rgb50_original_distance, cls.rgb50_original_audit = original_distance, original_audit
    module.rgb50_distance_cache_overlay_receipt = receipt
    receipt["shape_count_overlay"] = shape_receipt
    return receipt

