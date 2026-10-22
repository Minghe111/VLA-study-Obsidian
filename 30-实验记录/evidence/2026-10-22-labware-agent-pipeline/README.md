# RoboTwin 实验室器皿 Agent 采集接入

日期：2026-10-22。本目录是独立适配层，上一阶段 `../2026-10-22-labware-integration` 为冻结输入。

目标是用原 RoboTwin `code_gen` 代码生成／物理反馈入口，以及原 `scripts/collect_data.py::run` 的“成功种子 → 保存轨迹 → 重新 reset → 重放采集 → 语言 → HDF5”主循环，执行本机 RM65-B 新器皿任务。RealMan 手部与数据出口通过明确的 SDK24 扩展适配。

## 当前入口

- `prepare_agent_overlay.py`：六件物体的 Agent 语义、底面功能点与语言 overlay。
- `metadata_validation.json`：原 `enrich_actors`、`Actor` 和描述替换函数的真实组件验证。
- `labware_task.py`：实际 SAPIEN / Base_Task，真实接触与稳定放置判据。
- `robot_expert.py`：SDK24 手指预姿态、避碰规划与接触控制；不将物体轨迹设姿态当作机器人动作。
- `agent_runtime.py`：原 Prompt、API 模式及经用户授权的本会话代码生成／本地反馈模式。
- `session_candidates/`：本会话实际生成的候选版本；不是虚构的外部 API 回复。
- `collect_official.py`：原采集主循环＋任务局部轨迹／SDK24 导出 hooks。
- `sdk24_export.py`：实测 state、实际下发 action、下一实测 state、三路 RGB、60ms 时钟分开存储并全量读回。

源码依据：本机 `RoboTwin-RealMan` HEAD `931ce21f4f0e63feab661f349634d5a25bd92681`；通过 GitHub connector 读取官方 `RoboTwin-Platform/RoboTwin/main/scripts/collect_data.py`（文件 blob SHA `467d805d1a11352c900cffc4b3b9cc6c313dda63`）核对采集协议。当前文件与冻结 native RealMan 执行器的实际 SHA 在各次运行 launch/receipt 中记录；不把用户 fork 的 commit 冒称官方仓库 commit。

## 运行边界

本会话没有额外 LLM API。用户明确要求使用当前会话生成代码。因此会保留原 Prompt、本会话候选、原官方物理测试反馈和后续迭代，记录 `actual_llm_request_sent=false`。这不等于已经运行原脚本的远程 API 自动调用。

最小任务先验收空开口药瓶的真实抓起、保持、放回、松手和撤离。瓶盖／药片的夹具组件验证另有冻结证据；它们不能自动作为机器人开盖／倒药示范，也不能用来补写成功标签。

## 成功标准

1. 真实机器人接触驱动任务通过，不改变物体 pose、不使用抓持附件或重力补偿。
2. 原官方任务测试函数返回实际物理反馈，失败记录保留。
3. 原采集主循环完成规划与独立 reset 后的实际下发轨迹重放，重放任务也通过。
4. HDF5 三路 RGB 全帧可解码，state/action 均为24维，动作是下发target而非下一帧状态，tick差15×4ms、周期60ms，语言无占位符。
5. 不将单例成功当作大范围随机场景稳定成功率。

实际进度以各次 `rollout_receipt.json`、`collection_receipt.json`、`*.validation.json` 和 `pipeline_status.json` 为准。编译、schema或渲染组件通过均不代表上述全流程已通过。

## 已通过的实际闭环

`pipeline_acceptance.json` 当前通过范围为**右臂空开口药瓶抓起—直立保持—放回—松手撤离**。原官方 `test_gen_code.run` 对声明的近位1cm×1cm XY随机区域实际运行10次，10/10通过；这是专家执行范围内的结果，不是VLA策略SR，也不代表任意桌面或其余器皿的成功率。

首轮原 `collect_data.run` 实际筛选seed0、保存SDK24轨迹、独立reset后重放，产生389个60ms窗口（23.34秒）。三路640×480 RGB共1167帧、24维state/action、真实驱动目标读回、12条seen和12条unseen语言均通过独立读回。实际空中持稳0.812秒、放回稳定0.54秒。第一版头视图下沿裁切已保留诊断；最终 **v2** 只为head增加名义俯视15°，保持37°FOV、两腕视图和物理执行不变，重新真实重放采集。所有389帧目标几何完整入画，最小边距142.04px；state/action/next_state与v1逐字相同。

最终结果：

- `collection_runs/official_session_pick_place_v2_head_framed/data/episode_0000000.hdf5`
- 同目录 `episode_0000000.validation.json`
- `collection_runs/official_session_pick_place_v2_head_framed/independent_dataset_audit.json`
- `collection_runs/official_session_pick_place_v2_head_framed/video/episode_0000000.mp4`
- `dataset_visualization/official_session_pick_place_v2_head_framed/episode_0000000_keyframes.png`

最终HDF5 SHA：`18ae2495bcb29049b67d59040444e10f3c3d15b6ddd6463cce47d702ab82e794`。详细时序、单位和字段见 `DATASET_CONTRACT.md`。相机预设未做真机标定，几何入画检查不等于手指永不遮挡。

## 新采一轮

下面会执行**全新规划＋重放采集**，输出目录必须尚不存在。`task_config.yml`记录任务契约，实际执行参数以CLI生成的 `launch.json` 为准；没有改写RoboTwin全局配置。

```bash
ROBOTWIN_PY=/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python
PIPELINE_ROOT=/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline
"$ROBOTWIN_PY" "$PIPELINE_ROOT/collect_official.py" \
  --output "$PIPELINE_ROOT/collection_runs/my_new_collection" \
  --episodes 1 --max-seed-attempts 3 \
  --expert-code "$PIPELINE_ROOT/agent_runs/20261022T052331_906084Z_session_file_bridge_postprocess/envs_gen/gpt_labware_pick_place.py" \
  --agent-receipt "$PIPELINE_ROOT/agent_runs/20261022T050753_025169Z_session_export/revision_01/session_receipt.json" \
  --parser-receipt "$PIPELINE_ROOT/agent_runs/20261022T052331_906084Z_session_file_bridge_postprocess/postprocess_receipt.json" \
  --camera-framing "$PIPELINE_ROOT/camera_framing_probes/selected_head_framing.json"
```

批量时同时增加 `--episodes` 和有限种子尝试数。`--replay-source <已通过的采集目录>` 走原 `use_seed=True`，复用已验证的实际规划命令并重新仿真采图；receipt明确记录未重新规划，不把旧规划冒充新规划。相机配置只更改当前任务私有光学变换，不改厂家URDF、原相机YAML或物体状态。

读回一个实际文件：

```bash
"$ROBOTWIN_PY" "$PIPELINE_ROOT/sdk24_export.py" \
  "$PIPELINE_ROOT/collection_runs/official_session_pick_place_v2_head_framed/data/episode_0000000.hdf5"
"$ROBOTWIN_PY" "$PIPELINE_ROOT/dataset_audit.py" \
  --collection "$PIPELINE_ROOT/collection_runs/official_session_pick_place_v2_head_framed"
```

原始HDF5与逐帧缓存按本库惯例保留本地，已做本任务范围的Git忽略；代码、配置、receipt和可视化保留可归档。此次没有提交或推送Git。每次run自动留存实际源文件快照，历史v1源码也已逐SHA精确保存。

## 旋盖组件进展与剩余工作

`robot_cap_assembly.py`已将瓶身、瓶盖和12颗药片改为动态刚体，移除旧夹具、旋转位置电机和释放后固定器。采用实测相对转角的被动螺旋约束、反作用力矩、单侧紧盖止挡和载荷相关摩擦；工程刚度／阻尼待标定。12个CPU闭盖负控以及小／大力矩台架检查通过，证据为 `cap_component_probes/passive_final_summary.json`。

**该组件尚未通过机器人真实旋盖／倒药，也没有此类成功数据集。** 上述台架力矩不是机器人动作，旧夹具倾倒视频也不是当前空瓶示范。其余五件资产目前通过元数据接入，尚无各自抓取示范验收。新增器皿操作仍需按官方框架提供对应reset、动作专家和物理成功判据。
