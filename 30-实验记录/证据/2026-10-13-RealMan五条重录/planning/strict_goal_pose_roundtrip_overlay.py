"""Use the exact original native Pose IK target for the strict FK proof.

Float32 SAPIEN pose matrices have ~1e-7 orthogonality residual. Raw trace-acos
can report 3e-4rad even for an identical orientation. This supplements, without
editing, the pinned checker: validate roundoff scale, round-trip through the
same MPLib Pose used by original IK, then retain original FK thresholds.
No IK, native collision, returned trajectory or physical state is changed.
"""
from __future__ import annotations
import hashlib,importlib,json,math
from pathlib import Path
import numpy as np
from mplib import Pose

ORIGINAL_QUALITY_SHA256='0489a134b6a99383c13f2a6ec60c24662aa89ef0df6115c911b9ba5c39b4a9e6'
GOAL_FLOAT32_ROTATION_ROUNDOFF_LIMIT=1e-6
FK_SO3_ROUNDOFF_LIMIT=1e-10

def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _rotation_residual(rotation):
    r=np.asarray(rotation,dtype=float)
    return float(np.max(abs(r.T@r-np.eye(3)))),float(np.linalg.det(r))

def validate_native_pose_target(goal):
    raw=np.asarray(goal,dtype=float)
    if raw.shape!=(4,4) or not np.isfinite(raw).all():
        raise ValueError('Finite complete original goal pose required')
    if not np.array_equal(raw[3],np.array([0.,0.,0.,1.])):
        raise ValueError('Original homogeneous goal row differs')
    orthogonal,determinant=_rotation_residual(raw[:3,:3])
    if orthogonal>GOAL_FLOAT32_ROTATION_ROUNDOFF_LIMIT or abs(determinant-1.)>GOAL_FLOAT32_ROTATION_ROUNDOFF_LIMIT:
        raise ValueError('Original goal matrix exceeds float32 rotation roundoff bound')
    # This is exactly the constructor supplied to original native IK_CLIK.
    canonical=np.asarray(Pose(raw).to_transformation_matrix(),dtype=float)
    if not np.array_equal(canonical[:3,3],raw[:3,3]):
        raise ValueError('Native Pose changed the original goal translation')
    canonical_orthogonal,canonical_det=_rotation_residual(canonical[:3,:3])
    conversion_max=float(np.max(abs(canonical[:3,:3]-raw[:3,:3])))
    if canonical_orthogonal>FK_SO3_ROUNDOFF_LIMIT or abs(canonical_det-1.)>FK_SO3_ROUNDOFF_LIMIT or conversion_max>GOAL_FLOAT32_ROTATION_ROUNDOFF_LIMIT:
        raise ValueError('Native goal Pose readback is not bounded rigid SO3')
    receipt={'target_uses_same_original_native_Pose_conversion_as_IK':True,
        'original_goal_rotation_orthogonality_max_error':orthogonal,
        'original_goal_rotation_determinant':determinant,
        'native_Pose_target_rotation_orthogonality_max_error':canonical_orthogonal,
        'native_Pose_target_rotation_determinant':canonical_det,
        'native_Pose_rotation_roundoff_max_matrix_change':conversion_max,
        'goal_rotation_roundoff_limit':GOAL_FLOAT32_ROTATION_ROUNDOFF_LIMIT,
        'original_translation_bitwise_retained':True,
        'actual_planner_goal_or_returned_paths_changed':False,
        'original_goal_matrix':raw.tolist(),'exact_original_native_Pose_goal_matrix':canonical.tolist()}
    return canonical,receipt

def strict_native_goal_proof(original,q,base,indices,native_fk,goal,palm_index,required_link_count):
    canonical,receipt=validate_native_pose_target(goal)
    matrices=np.asarray(native_fk(q),dtype=float)
    if matrices.shape!=(required_link_count,4,4) or not np.isfinite(matrices).all():
        raise ValueError('Original complete native FK inventory required')
    if not np.array_equal(matrices[:,3,:],np.tile([0.,0.,0.,1.],(required_link_count,1))):
        raise ValueError('Native FK homogeneous rows differ')
    residuals=[_rotation_residual(m[:3,:3]) for m in matrices]
    if any(o>FK_SO3_ROUNDOFF_LIMIT or abs(d-1.)>FK_SO3_ROUNDOFF_LIMIT for o,d in residuals):
        raise ValueError('Native FK matrices must be genuine SO3 rotations')
    row=original(q,base,indices,lambda supplied_q:matrices,canonical,palm_index,required_link_count)
    # Literal equality protects the original 5e-5m / 2e-4rad decision boundary.
    if row['new_quality_position_tolerance_m']!=5e-5 or row['new_quality_rotation_tolerance_rad']!=2e-4:
        raise RuntimeError('Original exact FK acceptance thresholds changed')
    row['native_Pose_goal_roundtrip_proof']=receipt
    row['all_native_FK_rotations_checked_SO3']=required_link_count
    row['original_FK_acceptance_thresholds_unchanged']=True
    return row

def install_strict_goal_pose_roundtrip_overlay(*,receipt_path=None):
    """Install after all free-planner derivations and BEFORE observer wrappers.

    The worker must SHA-protect this file before constructing its expert.
    Constructor hook also adds it to session.sources before first motion.
    """
    quality=importlib.import_module('collector_quality_overlay')
    if _sha(quality.__file__)!=ORIGINAL_QUALITY_SHA256:
        raise RuntimeError('Pinned original strict FK quality checker changed')
    original=quality._strict_palm_goal
    if getattr(original,'_rerecord_native_goal_roundtrip',False):
        raise RuntimeError('Native goal roundtrip overlay must install only once')
    def checked(q,base,indices,native_fk,goal,palm_index,required_link_count):
        return strict_native_goal_proof(original,q,base,indices,native_fk,goal,palm_index,required_link_count)
    checked._rerecord_native_goal_roundtrip=True
    modules=[]
    for name in ('full_native_approach_equivalent_goals_v6','full_native_approach_equivalent_goals_dense_v7'):
        module=importlib.import_module(name)
        environment=module.plan_free_approach.__globals__
        if environment.get('_strict_palm_goal') is not original:
            raise RuntimeError('Install after derivation and before observer; strict FK binding differs')
        modules.append((name,environment))
    source=Path(__file__).resolve();digest=_sha(source)
    receipt={'schema':'strict_original_native_Pose_goal_roundtrip_v1','installer':str(source),'installer_sha256':digest,
        'original_quality_source_sha256':ORIGINAL_QUALITY_SHA256,
        'original_goal_position_tolerance_m':5e-5,'original_goal_rotation_tolerance_rad':2e-4,
        'native_original_IK_Pose_target_used':True,'paths_IK_collision_or_physical_state_changed':False,
        'original_frozen_source_files_modified':False,'CPU_only_install_not_episode_acceptance':True,
        'bound_planner_modules':[name for name,_ in modules],
        'installed_before_observer_and_expert_construction':True}
    for _,environment in modules:environment['_strict_palm_goal']=checked
    quality._strict_palm_goal=checked
    expert=importlib.import_module('expert_generation');old_init=expert.ExpertSession.__init__
    def init(session,*args,**kwargs):
        old_init(session,*args,**kwargs)
        session.sources[str(source)]=digest
        session.report['source_hashes']=session.sources
        session.report['strict_native_goal_roundtrip_overlay']=receipt.copy()
    expert.ExpertSession.__init__=init
    if receipt_path is not None:
        with Path(receipt_path).open('x',encoding='utf-8') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    return receipt
