"""Finite, explicit bounded-revolute goals and exact numeric path matching.

This module never changes a returned path. Equivalent configurations are
declared before RRT, bounded by the current native joint limits, and checked
against native FK for every supplied link. It does not certify physics.
"""
from itertools import product
import math
import numpy as np

TWO_PI = 2.0 * math.pi
FK_MATRIX_TOLERANCE = 1e-8
RAW_START_TOLERANCE_RAD = 1e-7
RAW_END_TOLERANCE_RAD = 1e-6


def _finite_vector(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(label + ' must be a finite vector')
    return result.copy()


def _fk_matrices(callback, q):
    matrices = np.asarray(callback(q.copy()), dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not len(matrices) or not np.isfinite(matrices).all():
        raise ValueError('Native FK must return finite 4x4 matrices for every link')
    return matrices.copy()


def declare_bounded_equivalent_goals(q, indices, limits, joint_types, native_fk):
    """Return finite legal goals, each native-FK checked against the input IK.

    ``joint_types`` comes from the actual native exported URDF, in full-q order.
    Only explicitly bounded revolute commanded joints are supported. Frozen
    joints retain the input IK values bit for bit. No goal or path is clipped.
    """
    original = _finite_vector(q, 'Original full IK configuration')
    columns = list(indices)
    if not columns or len(set(columns)) != len(columns) or any(type(i) is not int or i < 0 or i >= len(original) for i in columns):
        raise ValueError('Commanded joint indices must be unique valid integers')
    lim = np.asarray(limits, dtype=np.float64)
    if lim.shape != (len(original), 2) or len(joint_types) != len(original):
        raise ValueError('Native limits/types must follow the full-q ordering')
    if any(joint_types[i] != 'revolute' for i in columns):
        raise ValueError('Equivalence requires explicitly bounded revolute joints')
    if not np.isfinite(lim[columns]).all() or np.any(lim[columns, 0] > lim[columns, 1]):
        raise ValueError('Commanded manufacturer limits must be finite and ordered')
    reference = _fk_matrices(native_fk, original)
    windings = []
    for i in columns:
        first = math.ceil((float(lim[i, 0]) - float(original[i])) / TWO_PI)
        last = math.floor((float(lim[i, 1]) - float(original[i])) / TWO_PI)
        windings.append(range(first, last + 1))
    count = math.prod(len(choices) for choices in windings)
    if count > 4096:
        raise ValueError('Unexpected native bounded-goal product; refusing oversized declaration')
    goals, attempts = [], []
    frozen = np.ones(len(original), dtype=bool)
    frozen[columns] = False
    for turns in product(*windings):
        candidate = original.copy()
        candidate[columns] = original[columns] + TWO_PI * np.asarray(turns)
        row = {'integer_turns': list(turns), 'arm_qpos_rad': candidate[columns].tolist()}
        if not np.array_equal(candidate[frozen], original[frozen]):
            raise RuntimeError('Equivalent goal changed a frozen joint')
        row['inside_exact_native_commanded_limits'] = bool(np.all(candidate[columns] >= lim[columns, 0]) and np.all(candidate[columns] <= lim[columns, 1]))
        if not row['inside_exact_native_commanded_limits']:
            row['rejection_reason'] = 'Numeric candidate is outside exact native bounded limits'
            attempts.append(row)
            continue
        observed = _fk_matrices(native_fk, candidate)
        if observed.shape != reference.shape:
            raise ValueError('Native FK link inventory changed during goal declaration')
        delta = abs(observed - reference)
        row['native_FK_checked_links'] = len(reference)
        row['native_FK_max_matrix_error'] = float(delta.max())
        row['native_FK_max_position_error_m'] = float(np.max(np.linalg.norm(observed[:, :3, 3] - reference[:, :3, 3], axis=1)))
        row['native_FK_matrix_tolerance'] = FK_MATRIX_TOLERANCE
        row['native_FK_equivalent'] = bool(row['native_FK_max_matrix_error'] <= FK_MATRIX_TOLERANCE)
        if row['native_FK_equivalent']:
            goals.append(candidate)
        else:
            row['rejection_reason'] = 'Native FK does not prove the declared bounded configuration equivalent'
        attempts.append(row)
    return goals, {'original_full_IK_qpos_rad': original.tolist(),
                   'commanded_indices': columns,
                   'commanded_native_limits_rad': lim[columns].tolist(),
                   'commanded_native_URDF_types': [joint_types[i] for i in columns],
                   'finite_integer_goal_count': count,
                   'declared_native_FK_equivalent_goal_count': len(goals),
                   'native_FK_link_count': len(reference), 'goal_attempts': attempts,
                   'frozen_joints_bitwise_unchanged': True,
                   'returned_RRT_path_is_never_wrapped_or_clipped': True}


def match_declared_numeric_endpoint(path, start, declared_arm_goals):
    """Original raw start/end tolerances against a predeclared finite goal set."""
    initial = _finite_vector(start, 'Original numeric start')
    goals = np.asarray(declared_arm_goals, dtype=np.float64)
    raw = np.asarray(path, dtype=np.float64)
    row = {'passed': False, 'raw_start_tolerance_rad': RAW_START_TOLERANCE_RAD,
           'raw_end_tolerance_rad': RAW_END_TOLERANCE_RAD,
           'path_modified': False, 'declared_goal_count': len(goals)}
    if goals.ndim != 2 or goals.shape[1] != len(initial) or not len(goals) or not np.isfinite(goals).all():
        raise ValueError('Endpoint goals must be a finite predeclared nonempty matrix')
    if raw.ndim != 2 or raw.shape[1] != len(initial) or len(raw) < 2 or not np.isfinite(raw).all():
        row['rejection_reason'] = 'Original finite >=2-row arm-joint raw path requirement'
        return row
    errors = np.max(abs(goals - raw[-1]), axis=1)
    selected = int(np.argmin(errors))
    row.update(raw_path_rows=len(raw), raw_start_error_rad=float(np.max(abs(raw[0] - initial))),
               raw_end_error_rad=float(errors[selected]), matched_declared_goal_index=selected,
               matched_declared_goal_arm_qpos_rad=goals[selected].tolist(),
               raw_end_arm_qpos_rad=raw[-1].tolist())
    row['passed'] = bool(row['raw_start_error_rad'] <= RAW_START_TOLERANCE_RAD and row['raw_end_error_rad'] <= RAW_END_TOLERANCE_RAD)
    if not row['passed']:
        row['rejection_reason'] = 'Original numeric raw start/end predicate against explicit legal goals'
    return row
