"""Geometric readback predicates for complete tablets, not center-only labels.

A capsule is a line segment Minkowski-summed with a sphere. These margins
therefore test its whole geometry against a cylindrical cavity. They do not
prove physical support, stability, successful control, or calibrated CAD.
"""
from __future__ import annotations
import math
import numpy as np


def capsule_cylinder_margins(center, axis, *, half_length_m, capsule_radius_m,
                             inner_radius_m, floor_z_m, rim_z_m,
                             tolerance_m=0.):
    """Signed cavity clearances for a capsule, all in the cavity's frame.

    ``axis`` is the capsule local +X axis expressed in that frame. Positive
    margins mean the entire capsule lies inside. Contact tolerance is explicit
    and does not change the reported raw margins.
    """
    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    values = [half_length_m, capsule_radius_m, inner_radius_m,
              floor_z_m, rim_z_m, tolerance_m]
    if center.shape != (3,) or axis.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(axis).all():
        raise ValueError("Expected finite three-dimensional center/axis")
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Dimensions and tolerance must be finite")
    if half_length_m < 0 or capsule_radius_m <= 0 or inner_radius_m <= 0 or rim_z_m <= floor_z_m or tolerance_m < 0:
        raise ValueError("Invalid capsule/cavity dimensions")
    if abs(float(np.linalg.norm(axis))-1.) > 1e-5:
        raise ValueError("Capsule axis must be a unit vector")
    endpoints = center[None, :] + np.asarray([-1., 1.])[:, None]*half_length_m*axis[None, :]
    radial_extent = float(np.max(np.linalg.norm(endpoints[:, :2], axis=1))) + capsule_radius_m
    z_extent = half_length_m*abs(float(axis[2])) + capsule_radius_m
    margins = {"radial_margin_m": float(inner_radius_m-radial_extent),
               "floor_margin_m": float(center[2]-z_extent-floor_z_m),
               "rim_margin_m": float(rim_z_m-center[2]-z_extent)}
    return {**margins,
            "full_geometry_inside": all(x >= -tolerance_m for x in margins.values()),
            "center_inside": bool(np.linalg.norm(center[:2]) <= inner_radius_m
                                  and floor_z_m <= center[2] <= rim_z_m),
            "contact_tolerance_m": float(tolerance_m)}


def read_capsule_entities(tablets, container_pose, *, half_length_m,
                          capsule_radius_m, inner_radius_m, floor_z_m,
                          rim_z_m, tolerance_m=0.):
    """Read actual entity poses without writing simulator state.

    ``container_pose`` must be the actual collision cavity frame, not an
    unrelated visual origin. Callers separately check dynamic-body identity,
    support contacts, absence of floor spills, and a stable time interval.
    """
    inv = container_pose.inv()
    rows = []
    for tablet in tablets:
        local = inv * tablet.pose
        axis = np.asarray(local.to_transformation_matrix(), dtype=float)[:3, 0]
        rows.append({"name": tablet.name,
                     "center_in_container_m": np.asarray(local.p, dtype=float).tolist(),
                     "axis_in_container": axis.tolist(),
                     **capsule_cylinder_margins(local.p, axis,
                         half_length_m=half_length_m,
                         capsule_radius_m=capsule_radius_m,
                         inner_radius_m=inner_radius_m, floor_z_m=floor_z_m,
                         rim_z_m=rim_z_m, tolerance_m=tolerance_m)})
    return {"count": len(rows), "names_unique": len({r['name'] for r in rows}) == len(rows),
            "full_geometry_inside_count": sum(row["full_geometry_inside"] for row in rows),
            "center_inside_count": sum(row["center_inside"] for row in rows),
            "tablets": rows,
            "scope": "Actual pose geometry only; no control, support, or stability claim."}
