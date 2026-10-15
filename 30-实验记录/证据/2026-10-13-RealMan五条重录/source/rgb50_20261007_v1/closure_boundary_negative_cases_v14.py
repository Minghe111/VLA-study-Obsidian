"""Evidence-driven rejection tests using the saved real step1511/1512 contacts."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np

from closure_boundary_audit_v14 import (native_boundary_evidence, depth_within_budget,
                                       _trial_identity, _events, _assert_cpu_constructor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=Path, required=True)
    ap.add_argument('--certificate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise ValueError('Refusing to overwrite negative-case evidence')
    trace = json.loads((args.trial / 'physics_trace.json').read_text())
    report = json.loads((args.trial / 'expert_report.json').read_text())
    cert = json.loads(args.certificate.read_text())
    previous, current = trace[1510], trace[1511]
    link = 'l_inspire_left_thumb_4'
    results = []
    def check(name, value):
        results.append({'case': name, 'passed': bool(value)})
        if not value: raise AssertionError(name)
    def rejected(name, prev, row, requested_link=link, side='left'):
        result = native_boundary_evidence(prev, row, requested_link, side, 1e-10)
        check(name, result['eligible'] is False)
        results[-1]['rejection_reason'] = result['reason']
    check('real1512_native_boundary_positive', native_boundary_evidence(previous, current, link, 'left', 1e-10)['eligible'])
    check('only_one_original_event_resolved', cert['passed'] is True and cert['original_v13_passed'] is False
          and len(cert['resolved_closure_boundary_events']) == 1 and not cert['collision_events'])
    check('all3240_old_frames_retained', cert['recorded_samples'] == 3240 and cert['unchanged_sample_count'] == 3239
          and len(cert['sample_overlay']) == 1 and cert['sample_overlay'][0]['step'] == 1512)
    rejected('wrong_exact_link', previous, current, 'l_inspire_left_index_2')
    rejected('inactive_right_arm', previous, current, 'r_inspire_right_thumb_4')
    rejected('palm_excluded', previous, current, 'l_inspire_base_link')
    rejected('ring_not_primary', previous, current, 'l_inspire_left_ring_2')
    rejected('little_not_primary', previous, current, 'l_inspire_left_little_2')
    rejected('previous_is_step_minus2', trace[1509], current)
    rejected('future_touch_cannot_authorize', trace[1512], current)
    rejected('resolved_zero_frame_cannot_propagate', current, trace[1512])
    for phase in ('approach', 'close_hold', 'after_grasp'):
        row = copy.deepcopy(current); row['phase']['name'] = phase
        rejected('excluded_phase_' + phase, previous, row)
    row = copy.deepcopy(current); row['phase']['gripper'] = False
    rejected('gripper_false', previous, row)
    row = copy.deepcopy(current); row['phase']['dense_call'] += 1
    rejected('different_dense_call', previous, row)
    row = copy.deepcopy(current); row['bottle']['model_id'] = 16
    rejected('different_bottle_model', previous, row)
    row = copy.deepcopy(current); row.pop('bottle_actual_contacts')
    rejected('no_native_exact_contacts', previous, row)
    def exact(row): return next(c for c in row['bottle_actual_contacts'] if c['other_link'] == link)
    for value, label in [(-1e-12, 'negative'), (None, 'missing'), (float('nan'), 'nan')]:
        row = copy.deepcopy(current); exact(row)['points'][0]['separation_m'] = value
        rejected('native_separation_' + label, previous, row)
    row = copy.deepcopy(current); exact(row)['points'] = []
    rejected('native_points_empty', previous, row)
    row = copy.deepcopy(current); exact(row)['points'][0]['point_world_m'][0] = float('inf')
    rejected('native_point_nonfinite', previous, row)
    row = copy.deepcopy(current); point = exact(row)['points'][0]
    normal = np.asarray(point['normal_world']); impulse = normal * (1e-12 / float(normal @ normal))
    point['impulse_Ns'] = impulse.tolist(); point['normal_impulse_Ns'] = abs(float(impulse @ normal))
    exact(row)['normal_impulse_Ns'] = point['normal_impulse_Ns']
    rejected('nonzero_current_impulse_even_below_old_threshold', previous, row)
    depth = cert['resolved_closure_boundary_events'][0]['depth_certificate']
    check('real_event_specific_depth_certificate_positive', depth_within_budget(depth))
    for label, field, value in [('deep_penetration', 'measured_penetration_depth_m', depth['numerical_vertex_budget_m'] * 2),
                              ('nonfinite_depth', 'measured_penetration_depth_m', float('nan')),
                              ('unknown_depth', 'measured_penetration_depth_m', None),
                              ('unproved_budget', 'numerical_vertex_budget_m', None),
                              ('truncated_contacts', 'contact_lists_untruncated', False),
                              ('shape_pairs_missing', 'all_shape_pairs_evaluated', False)]:
        altered = copy.deepcopy(depth); altered[field] = value
        check('depth_reject_' + label, not depth_within_budget(altered))
    altered = copy.deepcopy(depth)
    altered['event_specific_vertex_certificate']['native_signed_gap_within_conversion_resolution'] = False
    check('native_fk_negative_gap_exceeds_conversion_budget', not depth_within_budget(altered))
    for field, value, name in [('active_arm', 'right', 'report_side_conflicts_trace'),
                             ('sample_spec', dict(report['sample_spec'], model_id=16), 'report_model_conflicts_scene')]:
        altered = copy.deepcopy(report); altered[field] = value
        try: _trial_identity(args.trial, trace, altered)
        except ValueError: check(name, True)
        else: check(name, False)
    row = {'step': 1512, 'trace_index': 1511, 'self_collisions': [['l_link2', 'r_link2']],
           'robot_table_collisions': ['l_link3'], 'bottle_non_touch_collisions': ['l_inspire_base_link'],
           'actual_self_contacts': [{'links': ['l_link4', 'r_link4'], 'normal_impulse_Ns': 1e-6}]}
    inherited = _events([row], 1e-7)
    check('other_self_table_palm_historical_events_remain_failed', {k[0] for k in inherited}
          == {'self', 'robot_table', 'bottle_robot_non_touch', 'historical_true_self_normal'})
    sys.path[:0] = [str(Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')),
                   str(Path('/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/adjust_bottle_20261002'))]
    _assert_cpu_constructor(); check('frozen_zero_step_constructor_identity', True)
    outcome = {'passed': all(r['passed'] for r in results), 'cases': results, 'case_count': len(results),
               'inputs_are_copies_only': True, 'new_physics_steps': 0,
               'trace_sha256': cert['trace_sha256'],
               'certificate_sha256': hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
               'boundary_helper_sha256': cert['boundary_helper_sha256']}
    args.out.write_text(json.dumps(outcome, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': outcome['passed'], 'cases': len(results), 'out': str(args.out)}), flush=True)


if __name__ == '__main__': main()
