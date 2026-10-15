"""Stricter subdivision only for complete collision-free insufficient bounds."""
import numpy as np

VERTEX_STEP_M = .0001
MAX_SAMPLES = 250000
THRESHOLDS_M = {'bottle': .0005, 'table': 0., 'self': 0.}


def refine_certificate(auditor, actual_TOPP_vertices, side, original):
    """Preserve original evidence, audit unchanged actual TOPP affine vertices.

    A real collision, joint-limit failure, insufficient sampled gap or missing
    evidence never enters this path. Exhausted finite capacity rejects; no
    prefix or partial audit can be accepted. Physics is not advanced.
    """
    thresholds = original.get('thresholds_m', {})
    samples = original.get('sampled_minimum_gap', {})
    bounds = original.get('continuous_gap_lower_bound', {})
    eligible = (original.get('passed') is False
        and original.get('joint_limit_audit', {}).get('passed') is True
        and original.get('unique_collision_pairs') == 0
        and original.get('collisions_first_64') == []
        and thresholds == THRESHOLDS_M
        and set(samples) == set(THRESHOLDS_M) and set(bounds) == set(THRESHOLDS_M)
        and all(np.isfinite(samples[k]['distance_m']) and samples[k]['distance_m'] >= thresholds[k] for k in thresholds)
        and all(np.isfinite(bounds[k]['lower_bound_m']) for k in thresholds)
        and any(bounds[k]['lower_bound_m'] < thresholds[k] for k in thresholds))
    receipt = {'eligible_original_complete_collision_free_insufficient_certificate': bool(eligible),
               'original_complete_default_grid_audit': original,
               'all_geometry_thresholds_unchanged': True, 'paths_not_wrapped_clipped_or_modified': True,
               'actual_TOPP_vertices_used_without_retiming_or_new_planning': True,
               'refined_vertex_step_m': VERTEX_STEP_M, 'finite_max_samples': MAX_SAMPLES,
               'sample_truncation_allowed': False, 'CPU_certificate_is_not_physical_success': True}
    if not eligible:
        return original, receipt
    vertices = np.asarray(actual_TOPP_vertices, dtype=float).copy()
    saved = vertices.copy()
    if vertices.ndim != 2 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError('Dense certificate requires complete finite actual TOPP vertex matrix')
    receipt['exact_actual_TOPP_vertices'] = vertices.tolist()
    try:
        dense = auditor.audit(vertices, side, vertex_step_m=VERTEX_STEP_M,
            min_bottle_clearance_m=thresholds['bottle'], min_self_clearance_m=thresholds['self'],
            min_table_clearance_m=thresholds['table'], max_samples=MAX_SAMPLES)
    except ValueError as error:
        if 'Adaptive subdivision exceeded max_samples; refusing incomplete audit' not in str(error):
            raise
        receipt.update(refined_complete_audit_unavailable=True, finite_capacity_exhausted=True,
                       error=str(error), partial_or_original_failed_path_not_accepted=True)
        return original, receipt
    if not np.array_equal(vertices, saved):
        raise RuntimeError('Dense native audit changed original actual TOPP vertices')
    if dense.get('thresholds_m') != thresholds:
        raise RuntimeError('Dense native audit changed required geometry clearances')
    if dense.get('metadata') != original.get('metadata'):
        raise RuntimeError('Dense native audit changed complete native model metadata')
    receipt.update(refined_complete_native_audit=dense,
                   unchanged_actual_TOPP_vertices=True,
                   refined_certificate_passed=bool(dense['passed']))
    return dense, receipt
