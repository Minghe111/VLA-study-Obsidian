# 官方 Agent 入口与实际验收边界

核查日期：2026-10-22。使用当前本机 RoboTwin commit `931ce21f4f0e63feab661f349634d5a25bd92681` 的 `code_gen/task_generation.py`、`test_gen_code.py` 和 `prompt.py`。通过 GitHub 插件读取 [官方 gpt_agent.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/code_gen/gpt_agent.py)，原版三个密钥字段仍为 `Your key` 占位；本机同样未配置。GitHub 文件 blob SHA `3a42e75f9dcc6b9ac62791b86361c09cb4922fd2`。

**后续授权修正**：用户明确表示没有 API，并授权直接使用当前 Codex 会话生成专家。本次采用“当前会话生成＋官方原始物理测试／错误反馈”的桥接方式；它不发送远程 API 请求，也不声称使用了原版 `gpt_agent.generate` 的远程自动循环。API 自动模式作为另一入口保留，缺 API 已不阻止会话桥接的实施。

## 已实际通过

- `robotwin` 环境有 OpenAI SDK，原始 code-generation 模块能够实际导入。
- `agent_runtime.py prompt` 调用原始 `generate_code`，由原始 `enrich_actors` 从本次 overlay 的 `points_info.json` 读取语义，构造原始 Prompt，运行到 LLM 调用边界后明确中止；没有伪造回复。
- 最新 Prompt、补全后的 Actor 字典与逐项检查保存于 `agent_runs/`。`latest_prompt.json` 指向最近一次。原始空代码骨架经过 Python AST 语法检查，**不是 Agent 生成结果**。
- 选择 RealMan task 的模式先导入 `labware_task.py`，再导入当前官方 `code_gen`，显式检查 `envs._base_task` 指向已冻结的 native RealMan 版本。实际模块文件路径和 SHA 记录在 `agent_preflight.json`，不凭工作目录推断运行版本。
- 原官方源码保持不变；运行时仅适配 LLM transport、对象语义路径、独立输出目录、任务 factory、测试次数及明确标为本地扩展的 RealMan SDK24 接触操作 API。

## 当前缺少的配置

当前 shell 的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 等常见 API 变量未配置，项目正常 `.env`／Agent config 文件不存在。未读取 Codex 私有 token、浏览器登录态或其它应用私有令牌。

本机仅 3000 端口的 Open WebUI 提供界面；公开 `/api/config` 显示启用登录验证，`/api/models` 和 `/ollama/api/tags` 只读请求均返回 401。常见本地模型端口 11434、8000、8001、8080、1234、5000 不开放。这不证明其它主机没有模型，只说明当前核查范围内没有可直接调用的已授权模型 API。

早期 API 预检的状态为 `missing_llm_configuration`，`actual_llm_request_sent=false`。当时没有宣称官方 LLM 已生成成功专家，也没有将人工 reference 专家冒充生成结果。若选择 API 自动方式，需要正常授权的兼容 API、模型名和凭据来源。

上述缺配置仅针对 **API 自动模式**。后续 `agent_preflight.json` 将两种入口分开记录：`api_auto_loop_status=missing_llm_configuration`，`session_bridge_available=true`。采用示例中的 `generation_transport=current_session` 时，导入与绑定通过的预检退出码为0。

## 当前会话桥接

`session-export` 实际运行原版 `generate_code` 并捕获完整 Prompt；可同时用 `--candidate` 将本会话输出的候选 SHA 与 Prompt SHA 绑定。当前单瓶任务为 `metadata/labware_pick_place_task_info.json`，仅包含 `self.bottle` 和右臂，避免把六资产模板误当作实际任务。

`session-test` 在独立 `revision_01` 至 `revision_05` 中按候选原字节冻结专家类，严格加载 `gpt_labware_pick_place`，实际调用官方 `test_gen_code.run`。它记录候选哈希、模块版本、真实种子测试记录和成功率；每轮失败后实际调用原版 `generate_code` 的错误分支构造反馈 Prompt，由当前会话生成下一轮候选。为让会话看到实际候选，桥接历史显式保留候选代码。全部种子成功时不调用原来带“unsuccessful”措辞的错误分支，避免将原测试器的默认错误标签当成真实失败。

```bash
$PY agent_runtime.py session-export --config agent_runtime_config.example.json \
  --task-info metadata/labware_pick_place_task_info.json \
  --candidate session_candidates/revision_01/gpt_labware_pick_place.py
$PY agent_runtime.py session-test --config agent_runtime_config.example.json \
  --session-dir /absolute/path/to/the/session_export_directory \
  --candidate session_candidates/revision_01/gpt_labware_pick_place.py --check-num 2
```

默认 `check_num=10`；指定2或3仅是对应种子数的冒烟测试，不能称为10种子评测。原版停止阈值0.5与所有试验都成功分别写入不同字段。测试结果只在真实物理执行完成后写入；准备 Prompt／检查候选语法不等于机器人或数据集采集成功。

## 可运行命令

以下均在本地仿真任务目录中操作，不连接真机。以 `PY=/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python` 和本目录绝对路径为前缀：

```bash
$PY agent_runtime.py preflight --config agent_runtime_config.example.json
$PY agent_runtime.py prompt --config agent_runtime_config.example.json --task-info task_info.json
$PY agent_runtime.py generate --config agent_runtime_config.example.json --task-info task_info.json
```

示例配置的 `api_key_env` 只保存环境变量名字，不存密钥。`base_url` 和 `model` 留空，优先从对应命名环境变量读取；也可填入普通 endpoint／模型名。不要把 key 写进 JSON、笔记或 Git。API 自动模式缺少配置的 preflight 退出码为2；当前会话桥接模式的成功预检为0，不能将它解释为 API 已配置。

`task_info.json` 必须由实际任务提供 Actor 成员名和任务目标；`metadata/task_info_template.json` 是六件资产的通用模板，不能把它当成当前单瓶任务的完成证明。

## 真实生成与反馈的执行链

配置就绪后，`generate` 实际调用原版 `task_generation.main`：原 Prompt → 真实 API 请求 → 官方 `generate_code` 提取 `play_once` 并写入新 `envs_gen/gpt_<task>.py` → 独立 factory 加载生成类 → 原版 `test_gen_code.run` 进行物理测试 → 原版错误反馈与下一轮生成。默认每轮 10 次、最多 5 轮、官方停止阈值成功率 0.5，均记录来源。

任务 factory 返回类必须严格为 `envs_gen.gpt_labware_pick_place.gpt_labware_pick_place`；若回退到人工 reference 专家，Agent 入口会拒绝执行。每轮原始 Prompt、真实回复和原版日志保存在独立新运行目录，不覆盖历史结果。API 异常仅记录异常类型及 HTTP 状态码，不打印供应商原始错误正文以避免意外回显凭据。

首轮 `grasp_lift_place` 是明确新增的 RealMan 接触控制扩展，只承担空瓶抓持、提起保持、放回释放；它不是官方原始 API。Agent 生成编排这一扩展是本体适配后的最小接通，不等于原始 Agent 已自主发现完整灵巧手抓持控制器，也不等于旋盖／倒药通过。

原 Agent 的 0.5 停止阈值与本项目数据集验收是两回事；生成通过后仍要以固定版本专家进行真实机器人仿真执行、成功判据、SDK24/三 RGB/60 毫秒导出与读回验收。

## 本会话候选的实际官方测试结果

当前会话候选 `session_candidates/revision_01/gpt_labware_pick_place.py` 的 SHA 为 `085fc69513076cdc64560cccac891c267157147f90041bcbc4a2affc2b699168`。原 `test_gen_code.run` 已完整执行默认 **10个实际物理试验，seed 0–9全部成功**，不是2/3种子冒烟、旧轨迹回放或改写成功标签。原测试器代码及本轮 task／robot_expert 源码SHA在整个测试中保持不变。

测试证据：`agent_runs/20261022T050753_025169Z_session_export/revision_01/session_receipt.json`；每seed独立 `simulation/seed_XXXX/` 保存结果、真实24维state/action/next_state、物理trace和运行日志。逐项离线读回检查见同目录 `session_physics_summary.json`：10条全部24维、有限值、相邻state连续，每action真实推进15个4毫秒物理步，最短空中对向持稳0.784秒、最短释放放回稳定0.54秒。

适用范围严格为：**右臂、空开口药瓶、1厘米×1厘米的近位XY随机区**。瓶盖和药片不在这一任务中；未证明任意桌面位置或所有器皿都能操作。这10次只验收Agent专家的物理计划，`capture_rgb=false`；含三RGB的数据集采集和导出由独立collector另外执行。

## 官方生成器后响应解析补验

使用用户授权的真实本会话候选文件作 file bridge 返回 fenced Python 文本，实际运行了原 `generate_code` 的后响应提取、模板组装、写 `envs_gen` 和 class factory 导入。没有伪造API响应或声称调用DeepSeek。原解析输出的 `play_once` AST与已物理测试候选完全相等；外层代码为原始模板的imports、单class和单play_once。

原解析输出：`agent_runs/20261022T052331_906084Z_session_file_bridge_postprocess/envs_gen/gpt_labware_pick_place.py`，SHA `e8cc7e693214d645fa5fed0dca91280e6c521f62457dbf12a8df14b2cb36224b`。独立 `postprocess_receipt.json` 明确保存 `original_generate_code_post_response_executed=true`、`method_ast_equal=true`、已测候选SHA、链接的真实10次物理测试receipt及相同task/helper SHA。

此解析补验没有追加物理测试，`new_physical_trials=0`、`score=null`、`API_request=false`。旧物理测试receipt未改。collector可以用此原解析文件作为输入，并通过receipt及独立AST检查接受其与已测方法的等价性；随后的plan与replay仍须实际执行，不能用AST等价替代新数据集验收。

复现补验命令：

```bash
$PY agent_runtime.py session-postprocess --config agent_runtime_config.example.json \
  --session-dir /absolute/path/to/the/tested/session_export_directory \
  --candidate session_candidates/revision_01/gpt_labware_pick_place.py
```
