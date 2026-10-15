"""Independent gate fixtures: original collisions/margins never enter refinement."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import dense_free_certificate_policy_v7 as policy

base = {'passed': False, 'joint_limit_audit': {'passed': True}, 'unique_collision_pairs': 0,
        'collisions_first_64': [], 'thresholds_m': {'bottle': .0005, 'table': 0., 'self': 0.},
        'sampled_minimum_gap': {'bottle': {'distance_m': .0008}, 'table': {'distance_m': .005}, 'self': {'distance_m': .002}},
        'continuous_gap_lower_bound': {'bottle': {'lower_bound_m': -.0001}, 'table': {'lower_bound_m': .004}, 'self': {'lower_bound_m': .001}},
        'metadata': {'collision_links': 93, 'native_model_fingerprint': 'unchanged_all_native_model'}}
vertices = np.asarray([[.1, .2, .3, .4, .5, .6], [.2, .3, .4, .5, .6, .7]])


class Auditor:
    def __init__(self, mode): self.calls = []; self.mode = mode
    def audit(self, path, side, **kwargs):
        self.calls.append({'path': path.copy(), 'side': side, **kwargs})
        if self.mode == 'capacity_exceeded': raise ValueError('Adaptive subdivision exceeded max_samples; refusing incomplete audit')
        result = copy.deepcopy(base)
        result['passed'] = self.mode != 'dense_still_insufficient'
        result['continuous_gap_lower_bound']['bottle']['lower_bound_m'] = .0007
        if self.mode == 'dense_still_insufficient': result['continuous_gap_lower_bound']['bottle']['lower_bound_m'] = .0004
        if self.mode == 'changed_threshold': result['thresholds_m']['bottle'] = .0004
        if self.mode == 'changed_model': result['metadata']['collision_links'] = 92
        if self.mode == 'changed_vertices': path[0, 0] += .001
        return result


results = []
for mode in ('eligible_success', 'original_passed', 'real_collision', 'collision_count_without_list',
             'joint_limit_failed', 'sampled_bottle_margin_failed', 'sampled_table_margin_failed',
             'missing_full_bounds', 'changed_original_threshold', 'all_bounds_sufficient',
             'nonfinite_gap', 'dense_still_insufficient', 'capacity_exceeded',
             'changed_threshold', 'changed_model', 'changed_vertices'):
    original = copy.deepcopy(base)
    if mode == 'original_passed': original['passed'] = True
    if mode == 'real_collision': original['collisions_first_64'] = [{'link': 'actual_collision'}]
    if mode == 'collision_count_without_list': original['unique_collision_pairs'] = 1
    if mode == 'joint_limit_failed': original['joint_limit_audit']['passed'] = False
    if mode == 'sampled_bottle_margin_failed': original['sampled_minimum_gap']['bottle']['distance_m'] = .00049
    if mode == 'sampled_table_margin_failed': original['sampled_minimum_gap']['table']['distance_m'] = -.00001
    if mode == 'missing_full_bounds': del original['continuous_gap_lower_bound']['self']
    if mode == 'changed_original_threshold': original['thresholds_m']['bottle'] = .0004
    if mode == 'all_bounds_sufficient': original['continuous_gap_lower_bound']['bottle']['lower_bound_m'] = .0007
    if mode == 'nonfinite_gap': original['sampled_minimum_gap']['bottle']['distance_m'] = np.nan
    auditor = Auditor(mode)
    before = json.dumps(original, sort_keys=True)
    saved_vertices = vertices.copy()
    try:
        result, receipt = policy.refine_certificate(auditor, vertices, 'right', original)
        if mode in ('changed_threshold', 'changed_model', 'changed_vertices'): raise AssertionError('Invalid dense result not rejected')
        if mode == 'eligible_success':
            assert result['passed'] and receipt['refined_certificate_passed']
        elif mode in ('dense_still_insufficient', 'capacity_exceeded'):
            assert not result['passed']
        else:
            assert result is original and not auditor.calls
        if auditor.calls:
            call = auditor.calls[0]
            assert call['vertex_step_m'] == .0001 and call['max_samples'] == 250000
            assert (call['min_bottle_clearance_m'], call['min_table_clearance_m'], call['min_self_clearance_m']) == (.0005, 0., 0.)
            assert np.array_equal(call['path'], vertices)
        results.append({'case': mode, 'passed': True, 'dense_audit_calls': len(auditor.calls)})
    except RuntimeError:
        assert mode in ('changed_threshold', 'changed_model', 'changed_vertices')
        results.append({'case': mode, 'passed': True, 'invalid_dense_result_rejected': True})
    assert json.dumps(original, sort_keys=True) == before and np.array_equal(vertices, saved_vertices)
output = {'passed': True, 'scope': 'Zero-step policy fixtures only, not full native FCL or physical success',
          'new_physics_steps': 0, 'all_original_thresholds_preserved': True,
          'original_rejected_evidence_never_rewritten': True, 'cases': results,
          'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), Path(policy.__file__))}}
with (Path(__file__).parent / 'dense_free_certificate_contract_receipt_v7.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps(output, indent=2), flush=True)
