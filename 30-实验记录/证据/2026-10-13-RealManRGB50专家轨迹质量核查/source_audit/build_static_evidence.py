from pathlib import Path
import json,hashlib,shutil

OUT=Path('/tmp/realman-data-motion-audit-20261013/source_audit')
REPO=Path('/home/admin123/liminghe/vla-benchmark')
VAULT=Path('/home/admin123/liminghe/VLA-WAM-RL')
ARCHIVE=VAULT/'30-实验记录/证据/2026-10-11-RealMan专家回放'
COLLECTOR=ARCHIVE/'read_only_collector_sources'
files={
 'expert_generation.py':COLLECTOR/'expert_generation.py',
 'expert_generation_v3.py':COLLECTOR/'expert_generation_v3.py',
 'expert_generation_v7.py':COLLECTOR/'expert_generation_v7.py',
 'expert_generation_v9.py':COLLECTOR/'expert_generation_v9.py',
 'current_Robot.py':REPO/'RoboTwin/envs/robot/robot.py',
 'current_native_robot.py':REPO/'RoboTwin/integrations/realman_rm65b_gen4/native_robot.py',
 'current_pkl2hdf5.py':REPO/'RoboTwin/envs/utils/pkl2hdf5.py',
 'frozen_remote_base_task.py':ARCHIVE/'shared_frozen_runtime/remote_base_task_snapshot.py',
 'current_full_native_carry_planner_v31.py':REPO/'deploy/realman_gen4/adjust_bottle_20261002/full_native_carry_planner_v31.py',
 'current_full_mesh_path_audit.py':REPO/'deploy/realman_gen4/adjust_bottle_20261002/full_mesh_path_audit.py',
 'data_adapter_rgb50_v1.py':REPO/'deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/data_pipeline_v1/data_adapter_rgb50_v1.py',
 'source_run_config_ep31.yml':ARCHIVE/'episode_0000031/source_run_config.yml',
}
rows=[]
for name,path in files.items():
    target=OUT/name;shutil.copy2(path,target)
    rows.append({'local_source_path':str(path),'archived_copy':str(target),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
report={
 'scope':'Static read-only source audit; no server, simulation, model query or source dataset mutation',
 'codegraph_attempt':'unable to open database file; no index created',
 'label_binding':{
   'source':'data_adapter_rgb50_v1.py:236-248',
   'state_field':'realman_dexterous/measured_state24',
   'action_field':'realman_dexterous/next_command24',
   'arm_delta_indices':list(range(6))+list(range(12,18)),
   'arm_delta_transform':'ordinary subtraction from anchor measured state; inverse ordinary addition',
   'angle_wrap_or_unwrap_in_adapter':False,
   'shortest_path_or_motion_efficiency_filter_in_adapter':False,
   'EE_pose_fields_used_as_model_action_labels':False,
 },
 'EE_export_chain':{
   'source':'frozen BaseTask get_obs/get_arm_pose + archived collector get_obs + current Robot/pkl2hdf5',
   'getter':'get_obs -> get_arm_pose -> get_left/right_ee_pose -> _trans_endpose',
   'physical_pose_source':'SAPIEN actual fixed hand-mount joint global_pose',
   'collector_overlay_modifies_EE_fields':False,
   'recorded_controller_field':'absent; Robot default scalar_gripper',
   'legacy_world_rotation':'R_actual_joint_world @ global_trans_matrix @ delta_matrix',
   'legacy_world_translation':'p_actual_joint_world + R_reported @ [gripper_bias - 0.12,0,0]',
   'recorded_gripper_bias_m':0.18,'effective_offset_m':0.06,
   'quaternion_order':'wxyz',
   'root_frame_transformation_in_HDF_EE':False,
   'HDF_state_and_action':'state = original measured endpose values[:-1]; action = values[1:]',
   'real_machine_TCP_calibration_claimed':False,
   'historical_source_boundary':'Frozen BaseTask and collector are archived; current Robot/exporter helpers are separately named/current. Exact original complete Robot helper bytes not independently matched here.',
 },
 'path_acceptance':{
   'collector_source':'expert_generation.py:450-479',
   'verified_gates':'native task success; >=250 airborne hold steps; >=1cm gap; opposed contact; no table force; <2mm/<2deg drift; source unchanged; full geometry checks',
   'path_length_detour_ratio_or_cumulative_joint_efficiency_gate_seen':False,
   'global_shortest_path_proved':False,
   'legal_nonminimal_successful_paths_can_pass_these_gates':True,
   'carry_source':'current_full_native_carry_planner_v31.py:339-357,408-480',
   'carry_selection':'IK limits/FK/collision; sort by joint L2 +0.15 abs J6 delta; first RRT/retiming/geometry-valid branch',
   'RRT_simplify_does_not_prove_global_optimality':True,
   'approach_source_available':'expert_generation_v3.py delegates first two segments to full_native_approach; core implementation not in local 8file source archive',
 },
 'angle_branch_note_evidence':{
   'note':'2026-10-07-RM65B-RGB50双臂近臂抓瓶数据集.md:109,130-134',
   'J6_hardware_finite_limit_rad':[-6.28,6.28],
   'continuous_SO2_joint':False,
   'out_of_limit_2pi_endpoints_reported_rejected':True,
   'legal_FK_equivalent_goal_declarations_reported_in_v6':True,
   'exact_final_finite_goal_core_source_not_read_here':True,
 },
 'metric_guidance':{
   'raw_joint':'per named raw finite joint min/max/span/net, total variation and longest monotonic excursion; do not automatically np.unwrap source',
   'continuity':'check actual dense4ms and command60ms jumps separately; smooth >=2pi span is genuine large same-joint travel, not an encoding discontinuity',
   'EE_position':'world-frame length, endpoint displacement, maximum excursion, repeated reversals and stage-specific detour; whole pickup/lift/carry intentionally multi-stage',
   'EE_rotation':'quaternion q/-q equivalence; shortest adjacent SO3 increment via abs dot; cumulative SO3 angle is distinct from net and fixed-axis full-circle winding',
   'claim_boundary':'no-360 does not imply no spatial detour; one joint >=360 also does not automatically imply EE circles object once',
   'reference_metadata':'source physics_trace phase/dense_call/terminal_hold, planning_calls and free_approach_*.json/ planned_path_*.npz recommended for exact segmentation',
 },
 'source_inventory':rows,
}
(OUT/'static_source_findings.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print('STATIC_SOURCE_FINDINGS_WRITTEN',len(rows),'files')
