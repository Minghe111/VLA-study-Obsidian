# 完整开盖倒药采集入口接线

2026-10-22。本文件与 `full_cap_collection.py`、实际消费的 `full_cap_collection_config.json`、`full_collection_entry_checks.py` 独立新增。空瓶collector/exporter、task/controller、冻结资产与原RoboTwin源码未修改。当前没有完整cap-pour示范，不能宣称采集已验收。

## 当前范围

固定任务为 `labware_cap115_unscrew_pour`，明确源码为本目录 `labware_cap115_unscrew_pour_task.py`、同名类。源码 `FULL_WORKFLOW_READY=False` 时，纯CPU preflight报告未就绪，execute/factory拒绝；不把当前完整场景框架、body-hold、无接触的v4或组件试验当作完整专家。默认只要求1条示范、1次seed尝试；原Agent测试可用1次，不增加10seed门槛。

纯源码检查：`python full_cap_collection.py --preflight`；CPU拒收检查：`python full_collection_entry_checks.py`。两者不导入/启动任务物理场景，不生成HDF5或成功轨迹。

## 原collect调用链和重放所有权

实际执行仍调用未修改的 `scripts.collect_data.run(task,args)`，而非自建一个同名collector。其顺序保留planning setup→实际play_once→final success/关节限位→实际目标trajectory/seed；同seed reset→load/set_path→实际重放→原语言函数→close→merge/export→再次check_success。通用14维PKL导出不用于本任务，任务hooks直接接已验SDK24 exporter。

`FullCollectionTask.play_once`独占capture重放分支，调用 `central_replay`，每条保存的requested_action恰执行一次；不再调用多个stage helper重放整段。第二次调用重放直接拒绝。所有planning metrics/outcome flags均忽略，实际apply_sdk24重新观测。保存source SHA和不舍入的实际reset初态（36关节位置/速度、root、body/cap、12片完整pq及名称、动态皿、被动装配状态），reset不同或source不同即拒绝。

plan与capture均从clock读出actual_final_drive_command24，保留requested_action/actual action/完整drive record；state和next_state来自实际测量，禁止将next_state移位当action。capture要求每行15个实际4ms步、60ms、实际drive与planning目标float32一致，三路480×640×3 RGB由原codec写入。最后调用原 `sdk24_export.export_episode/validate_episode`，逐帧解码、时间链、状态链、动作drive读回全部核验；视频仅诊断。

## 最终验收与close后生命周期

配置消费当前full task已有 `receiver_complete_capsules`、`cap_physically_clear`、`receiver_stable_seconds`、实际tablet table/bottle force与速度字段；`success`与`full_unscrew_pour_success`必须都是真正final。异常、阶段成功、11片、仅中心计数、仍接触瓶/桌、移动接收皿均拒收。移盖直接依据 `cap_state.released` 与真实whole-cap保守几何清口，不额外要求人工 `cap_end_state` 标签。初稿多加的该标签已在独立接口审计后移除，不能把真实成功永久挡在未实现的派生标签上。

原run在merge前close。adapter在close前冻结final-only verdict、实际metrics与语言info，避免actor释放后把旧阶段结果当成功。只有重放final通过、error=None、完整{A,B,C,a,b}绑定与12条无占位符语言都满足才进入HDF5；失败证据和cache保留。setup失败也占seed预算，防止在缺接口时无限尝试。

异常/中断外层finally会仅清理尚未closed的task；正常close幂等，不重复释放。cleanup错误另存且不覆盖首个rollout异常；主收集异常status明确为failed，不能留作running。

## 原Agent parser/factory和来源边界

`create_task`可作为原Agent的setup_factory，task_module_path为上述真实full模块；objects_root和TaskInfo使用 `full_agent_metadata` 独立overlay。factory必须加载当前run_dir中的 `envs_gen/gpt_labware_cap115_unscrew_pour.py`，找不到就拒绝，不fallback到native参考。严格AST要求原四个import、正确full生成类/继承和单个play_once；parser后字节变化时必须与实际被测play_once AST一致并有原parser receipt。

提供专家但未绑定真实完整Agent测试，只记录 `provided_candidate_provenance_unverified`，不称Agent生成验收。可独立运行native完整参考并如实标注；任何方式仍须实际final-only gate与原collect计划/重放/HDF读回。Agent“已测试”声明须真实full任务session receipt，并带 `full_task_source_sha256` 完整来源闭包；原agent_runtime当前仅绑定旧空瓶task/expert两个文件的receipt不够，入口明确拒绝据此冒充full provenance。完整Agent物理测试后的来源闭包receipt仍待未来接线/实际验证，未伪造。

依赖闭包逐字节快照小文本源，SHA绑定115 profile/spec/metadata/语言、冻结被动装配、capsule predicate、receiver、SDK/clock/physics/camera/native源以及生成候选/真实receipts。大型原几何仅保留源路径+SHA。execute默认无replay-source捷径，首先要求本次原run规划，再reset重放；不偷用旧空瓶种子/轨迹。

## 初态布局不能沿用倒药端点

推荐pour endpoint皿底 `[.16,-.12,.74]` 只在“瓶已被持起、盖已移除、左手安全位”的静态端点下通过。瓶初始约 `[.11,-.12]`，瓶半径25mm加皿外半径43.5mm，不能直接把该endpoint当setup布局。当前full场景默认皿 `[.21,-.12,.7405]` 仅有瓶/皿bounding cylinder间隙证据，仍不是完整actual初态可用。

入口要求实际setup的 `initial_layout_receipt.full_world_runtime_verified=True` 才继续，并要求初态指纹相同；该标记必须来自真实body/cap/12pill/robot/table/dish完整几何和动态状态检查，不能由本collector赋值。后续携瓶路径还要把动态接收皿纳入actualscene，不能用开盖前不存在的皿覆盖旧corridor证明。

## 未完成范围

目前没有完整专家、真实开盖清口倒12片成功、完整Agent候选物理测试、原collect全流程成功或最终三RGB HDF5。此处仅接好已明确接口并做CPU失败拒收。真实末态gate、初始化完整几何、full来源receipt和至少1条完整示范仍须以后实际验证。

## 实际CPU复核证据

2026-10-22实际运行`full_collection_entry_checks.py`，结果如下；positive fixture只测试内存布尔判据，没有物理数据或HDF输出。真实CLI preflight另实测exit2，FULL_WORKFLOW_READY=False拒绝execute。

```json
{
  "scope": "CPU-only in-memory failure controls; no robot/scene step or success trajectory/HDF5 created",
  "passed": true,
  "checks": [
    {
      "check": "stage_only_success_rejected",
      "passed": true
    },
    {
      "check": "rollout_exception_rejected_even_with_final_flags",
      "passed": true
    },
    {
      "check": "eleven_complete_capsules_rejected",
      "passed": true
    },
    {
      "check": "center_only_counts_rejected",
      "passed": true
    },
    {
      "check": "retained_bottle_contact_rejected",
      "passed": true
    },
    {
      "check": "unknown_cap_final_state_rejected",
      "passed": true
    },
    {
      "check": "unsettled_dynamic_receiver_rejected",
      "passed": true
    },
    {
      "check": "active_physical_guard_rejected",
      "passed": true
    },
    {
      "check": "positive_boolean_fixture_only_no_export",
      "passed": true
    },
    {
      "check": "current_full_workflow_false_blocks_execution",
      "passed": true
    },
    {
      "check": "missing_task_has_no_stage_fallback",
      "passed": true
    },
    {
      "check": "actual_cli_missing_task_rejected_before_output_or_physics",
      "passed": true
    },
    {
      "check": "wrong_clock_config_refused",
      "passed": true
    },
    {
      "check": "declared_full_official_wrapper_ast",
      "passed": true
    },
    {
      "check": "empty_task_agent_wrapper_rejected",
      "passed": true
    },
    {
      "check": "parser_changed_tested_method_rejected",
      "passed": true
    },
    {
      "check": "central_replay_executes_three_rows_once_no_copied_success",
      "passed": true
    },
    {
      "check": "second_full_replay_call_rejected",
      "passed": true
    },
    {
      "check": "reset_initial_state_mismatch_rejected",
      "passed": true
    },
    {
      "check": "stores_actual_drive_not_next_measured_state",
      "passed": true
    },
    {
      "check": "wrong_clock_rejected",
      "passed": true
    },
    {
      "check": "unresolved_full_language_rejected",
      "passed": true
    },
    {
      "check": "empty_only_language_schema_rejected",
      "passed": true
    },
    {
      "check": "close_preserves_immutable_verdict_metrics_language_fixture",
      "passed": true
    },
    {
      "check": "setup_error_counts_as_one_attempt",
      "passed": true
    },
    {
      "check": "seed_budget_stops_repeated_setup_failures",
      "passed": true
    },
    {
      "check": "actual_source_manifest_binds_full_implementation_profile_geometry",
      "passed": true
    }
  ],
  "count": 27,
  "source_sha256": "563d495c74ea425060bd523f63d455a7aed2d0df998f8a77a2d1156b3d90cb43",
  "checks_sha256": "f1e7df99ba2924901e419d7e5b47e81a423dbd364c6c4bdbedc40d63781a4c3a",
  "physics_steps": 0,
  "hdf5_or_success_demo_created": false,
  "source_manifest_files": 505
}
```

## 独立源码审计后的修订复核

移除额外cap_end_state标签；修正异常status和未关闭任务清理。2026-10-22重新实际运行29项CPU检查全部通过，0physics/HDF。下方结果取代初稿27项作为当前版本证据，旧结果保留作历史。

```json
{
  "scope": "CPU-only in-memory failure controls; no robot/scene step or success trajectory/HDF5 created",
  "passed": true,
  "checks": [
    {
      "check": "stage_only_success_rejected",
      "passed": true
    },
    {
      "check": "rollout_exception_rejected_even_with_final_flags",
      "passed": true
    },
    {
      "check": "eleven_complete_capsules_rejected",
      "passed": true
    },
    {
      "check": "center_only_counts_rejected",
      "passed": true
    },
    {
      "check": "retained_bottle_contact_rejected",
      "passed": true
    },
    {
      "check": "actual_whole_cap_not_clear_rejected",
      "passed": true
    },
    {
      "check": "unsettled_dynamic_receiver_rejected",
      "passed": true
    },
    {
      "check": "active_physical_guard_rejected",
      "passed": true
    },
    {
      "check": "positive_boolean_fixture_only_no_export",
      "passed": true
    },
    {
      "check": "current_full_workflow_false_blocks_execution",
      "passed": true
    },
    {
      "check": "missing_task_has_no_stage_fallback",
      "passed": true
    },
    {
      "check": "actual_cli_missing_task_rejected_before_output_or_physics",
      "passed": true
    },
    {
      "check": "wrong_clock_config_refused",
      "passed": true
    },
    {
      "check": "declared_full_official_wrapper_ast",
      "passed": true
    },
    {
      "check": "empty_task_agent_wrapper_rejected",
      "passed": true
    },
    {
      "check": "parser_changed_tested_method_rejected",
      "passed": true
    },
    {
      "check": "central_replay_executes_three_rows_once_no_copied_success",
      "passed": true
    },
    {
      "check": "second_full_replay_call_rejected",
      "passed": true
    },
    {
      "check": "reset_initial_state_mismatch_rejected",
      "passed": true
    },
    {
      "check": "stores_actual_drive_not_next_measured_state",
      "passed": true
    },
    {
      "check": "wrong_clock_rejected",
      "passed": true
    },
    {
      "check": "unresolved_full_language_rejected",
      "passed": true
    },
    {
      "check": "empty_only_language_schema_rejected",
      "passed": true
    },
    {
      "check": "close_preserves_immutable_verdict_metrics_language_fixture",
      "passed": true
    },
    {
      "check": "setup_error_counts_as_one_attempt",
      "passed": true
    },
    {
      "check": "seed_budget_stops_repeated_setup_failures",
      "passed": true
    },
    {
      "check": "exceptional_run_cleanup_only_once",
      "passed": true
    },
    {
      "check": "cleanup_error_returned_without_overwriting_primary_exception",
      "passed": true
    },
    {
      "check": "actual_source_manifest_binds_full_implementation_profile_geometry",
      "passed": true
    }
  ],
  "count": 29,
  "source_sha256": "5192faadc745351ee07f4da8f2fbd35e6d7a3ecff3fd3fb7856d568477f99459",
  "checks_sha256": "ae10eed8d0da1617d03966870bdf9bd280d231fddd62392b82d542e12439e16a",
  "physics_steps": 0,
  "hdf5_or_success_demo_created": false,
  "source_manifest_files": 505
}
```
