"""Synthetic component fixtures, never actual planner/physics evidence."""
import copy
import json
import numpy as np

from motion_gate import (ARM_BASE, CONFIG, DENSE_AFTER, DENSE_BEFORE, DENSE_COMMAND,
                         LOW_COMMAND, LOW_STATE, LOW_TIME, NEXT_TIME, TIME_AFTER,
                         TIME_BEFORE, load_json, numerical_motion, validate_periodic_branch_receipt,
                         upstream_gate)


def main():
    cfg = load_json(CONFIG)
    raw = [0.1, 0.2, -0.3, 0.4, -0.5, -5.5]
    reference = raw[:5] + [0.9]
    selected = raw[:5] + [-5.5 + 2 * np.pi]
    receipt = {'schema': 'nearest-periodic-branch-v1', 'synthetic_component_fixture_only': True,
               'arm_joint_limits_rad': {'left': [[-np.pi, np.pi]] * 5 + [[-2 * np.pi, 2 * np.pi]]},
               'arm_joint_types': {'left': ['revolute'] * 6},
               'events': [{'arm': 'left', 'stage': 'component_fixture', 'sim_time_s': 0,
                           'reference_kind': 'measured', 'reference_q_rad': reference,
                           'raw_ik_q_rad': raw, 'selected_q_rad': selected,
                           'selection_policy': 'nearest_periodic_equivalent',
                           'candidates': [{'q_rad': raw}, {'q_rad': selected}]}]}
    assert validate_periodic_branch_receipt(receipt, cfg)['passed']
    wrong = copy.deepcopy(receipt);wrong['events'][0]['selected_q_rad'] = raw
    assert not validate_periodic_branch_receipt(wrong, cfg)['passed']
    omitted = copy.deepcopy(receipt);omitted['events'][0]['candidates'] = [{'q_rad': raw}]
    assert not validate_periodic_branch_receipt(omitted, cfg)['passed']
    assert not validate_periodic_branch_receipt(None, cfg)['passed']
    assert not upstream_gate(None)['passed']
    n = 30
    tb = np.arange(n) * .004;ta = tb + .004
    before = np.zeros((n, 24));before[:, 6:11] = 1000;before[:, 18:23] = 1000
    after = before.copy();command = before.copy();command[20:, 10] = 900
    low = before[[0, 15]].copy();nextcommand = command[[14, 29]].copy()
    wrist = np.repeat(np.eye(4)[None], 2, axis=0);wrist[1, 0, 3] = .12
    arrays = {DENSE_BEFORE: before, DENSE_AFTER: after, DENSE_COMMAND: command,
              TIME_BEFORE: tb, TIME_AFTER: ta, LOW_STATE: low, LOW_COMMAND: nextcommand,
              LOW_TIME: np.array([0., .06]), NEXT_TIME: np.array([.06, .12]),
              'vision/cam_left_wrist/extrinsics_matrix': wrist}
    metadata = {'index': 17, 'arm': 'left', 'brand': 'Fanta', 'pose_class': 'upright'}
    assert numerical_motion(arrays, metadata, cfg)['passed']
    jumped = {k: v.copy() for k, v in arrays.items()};jumped[DENSE_COMMAND][8:, 5] += 2 * np.pi
    assert not numerical_motion(jumped, metadata, cfg)['passed']
    stale = {k: v.copy() for k, v in arrays.items()};stale[LOW_STATE][1, 0] = .1
    assert not numerical_motion(stale, metadata, cfg)['passed']
    early = {k: v.copy() for k, v in arrays.items()};early[DENSE_COMMAND][0, 10] = 900
    assert not numerical_motion(early, metadata, cfg)['passed']
    extra = numerical_motion(arrays, metadata, cfg, actual_first_target_contact_time_s=.08)
    assert extra['passed'] and 'before_actual_first_target_contact' in extra['metrics']
    print(json.dumps({'passed': True, 'synthetic_component_only': True,
                      'actual_planner_or_physics_evidence_generated': False,
                      'checks': ['nearest legal periodic branch recomputed', 'non-nearest/omitted candidate rejected',
                                 'missing branch/upstream evidence blocked', 'raw2pi target jump rejected',
                                 'stale low state rejected', 'already-closed recording prefix rejected',
                                 'true contact timestamp kept separate from SDK proxy']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
