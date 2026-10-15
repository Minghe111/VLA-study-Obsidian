"""v9 experiment: LP refresh every 15 physics steps, same feasible region.

This tests within-segment tracking. v9 already refreshes each new segment's
first physical step. It is not a fix for a missing boundary refresh.
The .4 s infeasible-contact timeout is measured in actual control steps since
the first unusable solve, with at most one LP refresh interval of detection.
All old files stay unchanged; this version is only for fresh physics trials.
"""
from pathlib import Path
from types import MethodType
import ast,copy
import numpy as np
from expert_generation_v9 import (ExpertSession as _BaseSession,
                                  install_initializer,command24,measured24,_sha)
from expert_generation import _extract,_SideTransformer


class LPPeriodTransformer(ast.NodeTransformer):
    def __init__(self):
        self.period_sites=0;self.failure_clock_sites=0;self.recovery_sites=0

    def visit_BinOp(self,node):
        self.generic_visit(node)
        if isinstance(node.op,ast.Mod) and ast.unparse(node.left)=="state['control_steps']":
            if not isinstance(node.right,ast.Constant) or node.right.value!=50:
                raise RuntimeError('Unexpected original LP refresh interval')
            node.right=ast.Constant(value=15);self.period_sites+=1
        return node

    def visit_AugAssign(self,node):
        self.generic_visit(node)
        if ast.unparse(node.target)=="state['lp_failure_steps']":
            if not isinstance(node.op,ast.Add) or not isinstance(node.value,ast.Constant) or node.value.value!=50:
                raise RuntimeError('Unexpected original LP failure clock')
            replacement=ast.parse("""if state['lp_failure_first_control_step'] is None:
    state['lp_failure_first_control_step']=state['control_steps']
state['lp_failure_steps']=state['control_steps']-state['lp_failure_first_control_step']
""").body
            self.failure_clock_sites+=1
            return [ast.copy_location(n,node) for n in replacement]
        return node

    def visit_Assign(self,node):
        self.generic_visit(node)
        if len(node.targets)==1 and ast.unparse(node.targets[0])=="state['lp_failure_steps']":
            if not isinstance(node.value,ast.Constant) or node.value.value!=0:
                raise RuntimeError('Unexpected original LP recovery reset')
            reset=ast.parse("state['lp_failure_first_control_step']=None").body[0]
            self.recovery_sites+=1
            return [node,ast.copy_location(reset,node)]
        return node


class _CompositeTransformer:
    def __init__(self,side):
        self.side=side;self.period=LPPeriodTransformer()
    def visit(self,node):
        return self.period.visit(_SideTransformer(self.side).visit(node))


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        from continuous_servo import ContinuousHandMapping,SDKWaypointProfile
        from integrations.realman_rm65b_gen4.native_robot import RealManRobot
        robot=task.robot
        snapshot={'qpos':robot.left_entity.qpos.copy(),'qvel':robot.left_entity.qvel.copy(),
                  'bottle_pose':task.bottle.actor.pose.to_transformation_matrix().copy(),
                  'hand_targets':np.asarray([robot._hand_joints[n].get_drive_target()[0] for n in self.hand_names])}
        old=self.feedback;transformer=_CompositeTransformer(self.side)
        context={'np':np,'sapien':self.sapien,'MethodType':MethodType,'RealManRobot':RealManRobot,
                 'ContinuousHandMapping':ContinuousHandMapping,'SDKWaypointProfile':SDKWaypointProfile,'stage':self.stage}
        digest=_extract(self.legacy/'contact_capture_control_v26.py',
            {'_feedback_rate_limits','install_contact_capture'},context,transformer)
        counts=(transformer.period.period_sites,transformer.period.failure_clock_sites,transformer.period.recovery_sites)
        if counts!=(1,1,1):raise RuntimeError('Expected exactly one refresh, failure clock and recovery site')
        self.feedback,self.contact_update=context['install_contact_capture'](task,self.setting)
        self.feedback['lp_failure_first_control_step']=None
        self.feedback['lp_refresh_period_physics_steps']=15
        setattr(robot,self.side+'_plan_grippers',MethodType(self.feedback['plan'],robot))
        for key in ('index','latched','filtered_N','normal_N','force_target_N','profile_motion_steps','profile_padding_steps'):
            if not np.array_equal(np.asarray(old[key]),np.asarray(self.feedback[key])):
                raise RuntimeError('Fresh-controller initial state differs: '+key)
        actual={'qpos':robot.left_entity.qpos,'qvel':robot.left_entity.qvel,
                'bottle_pose':task.bottle.actor.pose.to_transformation_matrix(),
                'hand_targets':np.asarray([robot._hand_joints[n].get_drive_target()[0] for n in self.hand_names])}
        if not all(np.array_equal(snapshot[k],actual[k]) for k in snapshot):
            raise RuntimeError('Controller selection mutated the physical scene')
        source=Path(__file__).resolve();self.sources[str(source)]=_sha(source)
        self.report.update(expert_version='v11',source_hashes=self.sources,
            controller_overlay_ast_sha256=digest,
            LP_refresh_experiment={'baseline_period_physics_steps':50,'selected_period_physics_steps':15,
                'physics_dt_s':float(task.scene.get_timestep()),'selected_period_s':15*float(task.scene.get_timestep()),
                'new_segment_first_step_already_refreshed_in_v9':True,
                'LP_feasible_region_and_solver_unchanged':True,'force_gain_and_rate_caps_unchanged':True,
                'five_finger_profile_streak_and_passive_guard_unchanged':True,
                'infeasible_timeout_physical_seconds':.4,
                'failure_time_is_control_step_elapsed_since_first_unusable_solve':True,
                'timeout_detection_resolution_s':15*float(task.scene.get_timestep()),
                'private_AST_sites':{'period':counts[0],'failure_clock':counts[1],'recovery_reset':counts[2]},
                'controller_initial_state_same':True,'physical_state_and_targets_unchanged_at_install':True,
                'static_checks_are_not_physical_success':True})
        self.write()

    def save_evidence(self):
        self.report['LP_refresh_final_clock']={'period_physics_steps':self.feedback.get('lp_refresh_period_physics_steps'),
            'first_unusable_control_step':self.feedback.get('lp_failure_first_control_step'),
            'elapsed_unusable_control_steps':self.feedback['lp_failure_steps']}
        return super().save_evidence()


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,
                capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
