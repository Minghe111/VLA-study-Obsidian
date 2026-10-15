"""Exact additive FMA self broadphase vectorization; no distance caching.

All original pairs, shapes, thresholds, samples and scalar/FCL calculations are
retained. At each sample the original AABB skip condition is evaluated in bulk
against the minima already known BEFORE this sample's self loop. Only pairs
certain to be skipped by the unchanged scalar loop are filtered. Remaining
pairs keep their original order and recompute the original scalar AABB/FCL.
As minima only decrease during the loop, a skip proved at loop entry remains
valid. A numerical guard prevents floating reduction roundoff pruning a pair
close to either scalar comparison boundary. This guard changes optimization
eligibility only, not collision/clearance thresholds or interval bounds.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import time

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def candidate_pairs(self, boxes, outgoing, best_gap, certified_gap):
    started=time.monotonic()
    context=getattr(self,'_rgb50_vector_context',None)
    if context is None:
        names=list(self.objects);ids={n:i for i,n in enumerate(names)}
        pairs=list(self.self_pairs)
        context={'names':names,'pairs':pairs,
                 'first':np.asarray([ids[a] for a,b in pairs],dtype=np.intp),
                 'second':np.asarray([ids[b] for a,b in pairs],dtype=np.intp),
                 'original_pair_count':len(pairs)}
        self._rgb50_vector_context=context
    # Immutable topology is checked on each audit, not inferred from labels.
    packed=np.asarray([boxes[n] for n in context['names']],dtype=np.float64)
    a,b=packed[context['first']],packed[context['second']]
    delta=np.maximum(np.maximum(a[:,0]-b[:,1],b[:,0]-a[:,1]),0.)
    lower=np.linalg.norm(delta,axis=1)
    motion=np.asarray([outgoing[n] for n in context['names']],dtype=np.float64)
    summed=motion[context['first']]+motion[context['second']]
    guard=1e-12+64*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(packed),initial=0.)))
    # For nonfinite broadphase inputs, defer every pair to the original code.
    if np.isfinite(lower).all() and np.isfinite(summed).all():
        proven_skip=(lower>guard)&(lower-guard>best_gap)&(lower-guard-summed>certified_gap)
        selected=np.flatnonzero(~proven_skip)
    else:selected=np.arange(len(context['pairs']))
    stats=self._rgb50_vector_stats
    stats['samples']+=1;stats['original_pair_visits']+=len(context['pairs'])
    stats['original_skip_proved_in_bulk']+=len(context['pairs'])-len(selected)
    stats['scalar_pair_candidates']+=len(selected)
    stats['broadphase_elapsed_s']+=time.monotonic()-started
    stats['maximum_roundoff_optimization_guard_m']=max(stats['maximum_roundoff_optimization_guard_m'],guard)
    return (context['pairs'][int(i)] for i in selected)


def install_vectorized_aabb_overlay(legacy_dir):
    """Call only at bootstrap of a NEW worker, before planner/auditor imports."""
    from geometry_overlay import install_native_shape_count_overlay
    source=Path(legacy_dir)/'full_mesh_path_audit.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cls_node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FullMeshPathAuditor')
    method=next(n for n in cls_node.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    sites=[n for n in ast.walk(method) if isinstance(n,ast.For) and ast.unparse(n.iter)=='self.self_pairs']
    if len(sites)!=1 or ast.unparse(sites[0].target)!='(first, second)':
        raise RuntimeError('Frozen FMA self loop signature differs')
    sites[0].iter=ast.parse("self._rgb50_self_candidate_pairs(boxes, outgoing, best['self']['distance_m'], certified['self']['lower_bound_m'])",mode='eval').body
    changed=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(changed)
    receipt={'source':str(source.resolve()),'source_sha256':sha(source),
             'overlay_source':str(Path(__file__).resolve()),'overlay_source_sha256':sha(__file__),
             'overlay_audit_ast_sha256':hashlib.sha256(ast.dump(changed,include_attributes=False).encode()).hexdigest(),
             'changed_iteration_sites':1,'original_loop_body_unchanged':True,
             'distance_calls_and_values_not_cached':True,'bounds_thresholds_pairs_shapes_unchanged':True,
             'production_scene_changed':False,'carry_auditor_changed':False}
    existing=sys.modules.get('full_mesh_path_audit')
    previous=getattr(existing,'rgb50_vector_aabb_overlay_receipt',None)
    if previous is not None:
        if {k:v for k,v in previous.items() if k!='shape_count_overlay'}==receipt:return receipt
        raise RuntimeError('Do not change an installed overlay in an active worker')
    if getattr(existing,'rgb50_distance_cache_overlay_receipt',None) is not None:
        raise RuntimeError('Do not combine vectorization with unaccepted distance caching')
    stale=[n for n in ('carry_mesh_path_audit','full_native_carry_planner_v31','observed_full_trace_audit_v13','full_native_approach') if n in sys.modules]
    if stale:raise RuntimeError('Install vector broadphase before geometry dependencies: '+str(stale))
    shape=install_native_shape_count_overlay(legacy_dir)
    module=sys.modules['full_mesh_path_audit'];cls=module.FullMeshPathAuditor
    namespace=vars(module).copy();exec(compile(changed,str(source)+':rgb50_vector_aabb','exec'),namespace)
    unchanged=cls.audit;vector_method=namespace['audit']

    def audit(self,path,side='left',**kwargs):
        if getattr(self,'_rgb50_vector_stats',None) is not None:raise RuntimeError('Nested vector audit unsupported')
        # Topology never persists across audit invocations, even on warm workers.
        self._rgb50_vector_context=None
        self._rgb50_vector_stats={'samples':0,'original_pair_visits':0,'original_skip_proved_in_bulk':0,
                                  'scalar_pair_candidates':0,'broadphase_elapsed_s':0.,
                                  'maximum_roundoff_optimization_guard_m':0.}
        started=time.monotonic()
        try:
            result=vector_method(self,path,side,**kwargs)
            result['vector_broadphase_audit']={**self._rgb50_vector_stats,'overlay_receipt':receipt,
                'elapsed_including_wrapper_s':time.monotonic()-started,
                'all_unfiltered_pairs_recompute_original_scalar_lower':True,
                'pair_order_samples_limits_thresholds_bounds_preserved':True}
            return result
        finally:
            self.rgb50_last_vector_stats=dict(self._rgb50_vector_stats)
            del self._rgb50_vector_stats;del self._rgb50_vector_context

    cls.audit=audit;cls._rgb50_self_candidate_pairs=candidate_pairs
    cls.rgb50_original_audit=unchanged
    receipt['shape_count_overlay']=shape
    module.rgb50_vector_aabb_overlay_receipt=receipt
    return receipt
