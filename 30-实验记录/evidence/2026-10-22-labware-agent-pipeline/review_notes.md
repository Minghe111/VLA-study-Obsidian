# Collector 独立审核

日期：2026-10-22。范围：只读审核 `sdk24_export.py`、`collect_official.py`、`labware_task.py`、`robot_expert.py` 以及它们实际调用的原收集入口和时钟。没有修改这四个被审文件。下列行号对应审核时快照；主任务后续修复后应以新 SHA 和回归结果为准。

**复核状态：下述首次审核发现的四项问题均已修复；12项CPU受控负控与close复核通过。** 这仍不代表物理采集成功。复核源 SHA：collector `d3c16ad380b8cf73c34d6d865af71191a752d34265f6c4b7c936cbb3584f20fc`；exporter `2741d6a763a3ba171e9cbfbb38ec3e83e9cf427d9b5ede766ff3186d048fe4a4`；task `22cd86c3d0836633ad7911e10f999fe366f99b66d9b883a626fe91d05711fa3e`。

## 已确认正确的链路

- `labware_task.capture()` 从 `robot.get_state24(True)` 读取实际关节状态；`apply_sdk24()` 单独保存传入目标。HDF5 state、action、next_state 没有以相邻测量状态替代实际动作目标。
- `ClockedTask24Adapter` 每目标执行 15 个 4 ms 物理步，本任务关闭成功时提前截断。导出前后均检查 tick 差15、跨帧 tick 连续和测量 state 链连续。视频三次重复控制帧、50 fps 对应同样的60 ms控制周期；视频只作诊断。
- 原 `collect_data.run` 先规划筛成功 seed，再以相同 seed 重置并读回 SDK24 命令序列。`CollectionTask.set_path_lst` 将完整24维动作行交给本任务重放，而不是只重放两个6轴路径。
- 原入口在 `close_env()` 后合并数据；任务 close 保留 metrics、collection_receipt、磁盘缓存，所以 merge 不依赖活着的 scene。这种时序与原入口一致。
- 捕获回放失败会在 merge 的实际成功检查和最终物理 metrics 检查处拒绝导出；空/断裂/无图像的缓存也被拒绝。
- 新元数据的语言绝对路径与 parent_directory 绑定已用原函数独立验证，正常 info 可以生成12 seen + 12 unseen。

## 首次发现的问题（已修复，保留原失败证据）

### 1. 异常之后仍能产生 success=true 的 rollout 收据

审核快照 `collect_official.py` SHA256：`0bd336a36285f204100c4912266b8d3216b2ce48f790448e3aedb422c95faa4a`。

`persist_run()` 原102–107行把 success 直接设为 `check_success()`，没有检查 `error`。如果已达到物理成功但后续生成代码/记录函数抛异常，收据可同时包含 `success=true` 和非空 error。`save_traj_data()` 原122–127行也只检查物理 success；`sdk24_export.export_episode()` 原72行只检查 receipt.success，没有拒绝 receipt.error。

最小修复：收据区分 `physical_success` 和整次执行 `success`；后者应为 `physical_success and error is None`。保存轨迹、合并与导出都要求这次执行无异常。后续某步骤异常不能抹掉已有失败证据。

CPU受控负控已真实调用从被审文件 AST 提取的原方法，拦截全部 dump：physical success=true + error非空时，原方法确实产生success=true，且save_traj_data接受。没有创建或标记任何实际物理轨迹。证据：`metadata/review_component_checks.json`。

### 2. 仅传入 --expert-code 不足以证明官方智能体生成

`collect_official.py` 原47–53行将任意指定候选文件自动标为 `official_agent_candidate`。这只能证明执行了指定文件，无法证明文件来自官方Agent、远程模型响应或当前Codex会话。

最小修复：默认记录为明确的 provided candidate，`agent_generated` 必须依据可检查的生成收据、原响应/会话来源及候选 SHA，而不是由参数存在推断。当前没有成功远程LLM调用时，不能把本地手工/当前会话编写的候选重新标为官方远程Agent生成。

### 3. 官方语言 fallback 会掩盖生成任务 info 错误

当前导出检查只要求语言非空且没有占位符。原 `collect_data.build_episode_instructions` 在任务 info 错误导致没有匹配模板时，会返回一条 `labware pick place`。这仍能通过当前 exporter 的语言条件，但不满足本任务12条具体任务指令的契约。

CPU受控负控真实调用原 `build_episode_instructions` 和原 `generate_episode_descriptions`，输入空 info 时得到seen/unseen各1条fallback，且符合当前exporter非空条件。证据同上。

最小修复：在具体任务采集层要求 `info['info']` 含正确 `{A}` 路径和 `{a}` 臂侧；merge要求语言条数等于12、无未替换占位符，并对比该episode对应的instruction sidecar。这样不需要修改官方fallback函数。

### 4. close 的重复调用应安全

`labware_task.close()` 原303–308行仅用hasattr检查 scene/_raw_step，然后访问 `self.scene.step`；第一次close将两者设None后，再次close会对None访问step。正常每episode一次close不触发，但异常清理或外部finally重复关闭可能覆盖原失败原因。

最小修复：只在scene和_raw_step均非None时恢复step；重复close保留metrics/receipt/cache，不抛新异常。

## 建议保留的后续诊断

`apply_sdk24()` 目前忽略时钟返回的实际执行 record。建议显式拒绝 record=None，并在每条transition记录 `executed_physics_steps`、`endpoint_command_written`、`actual_final_drive_command24`，核对最终写入目标与原输入一致。现有tick检查已经阻止零步记录成为合法HDF5；追加此诊断可更直接证明“实际目标”来源，不是新的物理成功证据。

本审核的CPU负控只验证错误路径和数据契约，不能代替真实机器人仿真示范、抓持成功或完整采集验收。

## 修复后的独立复核

运行 `metadata/run_review_regression.py`，从当前源文件提取未经修改的方法AST，执行受控错误路径。涉及真实export函数时使用临时合成npz，只验证在创建HDF5之前拒绝错误收据；这些合成fixture未作为物理数据保存或归档。结果为 `metadata/review_regression_checks.json`。

12项均通过：异常收据success=False；save拒绝异常；merge拒绝异常；merge拒绝官方单条fallback；merge拒绝错误臂侧；merge拒绝缺失对象描述；无生成收据的候选不被标为agent_generated；候选SHA不匹配被拒绝；未达官方阈值的候选被拒绝；export拒绝中断但success=True的收据；错误export不创建HDF5；close重复两次不报错且保留merge所需metrics/receipt/cache路径。

本次没有改动root拥有的collector、exporter、task或expert实现。已有正常状态/目标分流、60 ms时钟和close后merge设计保持；实际物理rollout、三路图像读回与完整官方采集应继续在主任务中验收。
