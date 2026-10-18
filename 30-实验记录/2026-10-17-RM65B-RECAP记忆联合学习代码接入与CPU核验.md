---
title: RM65-B RECAP记忆联合学习代码接入与CPU核验
date: 2026-10-17
updated: 2026-10-17
status: 代码已接入-CPU组件通过-未训练未仿真验收
tags: [实验, RLinf, RECAP, π0.5, RM65-B, 记忆, 知识隔离]
sources:
  - "[[本机-RM65B-RECAP记忆联合学习实现计划]]"
  - "[[RECAP扩展训练框架-记忆读写与高层决策的回报优化]]"
deployment_root: /home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017
model: pi05_80000_openpi
evidence_kind: 本机CPU组件实测与静态检查
---

# RM65-B RECAP记忆联合学习代码接入与CPU核验

用户授权“开始执行修改计划，先做修改不训练”。本次在独立任务目录接入扩展基线；未安装依赖、转换或修改真实权重、加载真实模型、执行GPU计算、启动SAPIEN、进行优化器更新或训练。

## 实际交付

持久代码根：`/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017`。

由原本机PPO任务副本派生，**不是原RLinf仓库已提交版本**。原仓参考HEAD仍为RLinf `50f22ad9a7825eeaaf4180dca5300f2fa70758be`、RoboTwin `931ce21f4f0e63feab661f349634d5a25bd92681`；具体新副本以`reports/source_manifest.json`、`reports/changes.json`和`reports/implementation.patch`为准。本次30个代码／配置文件发生新增或修改，其中28个Python文件；原2137个常规源码基线SHA256保持不变。机器人及CuRobo资产经原路径复用，未逐资产重新验收；原入口、模型、venv和资产保留。没有Git提交或推送。

| 模块 | 已实现职责 | 当前证据 |
| --- | --- | --- |
| `openpi/task=recap_hier` | 同一个π0.5 VLM生成READ、WRITE和自由子任务文本，连续expert生成动作 | tiny native模型CPU前后向与生成逻辑；非真实权重语言能力验收 |
| 原生文本／FAST／LoRA | 共享decode、因果CE、分块大词表CE、EOS/cache、VLM attention专属LoRA、严格checkpoint | 16项新模型测试及23项原接口测试 |
| 记忆和schema | episode UUID、目录／已读正文／当前写入可见范围、revision、来源及工具审计 | 记忆重放、伪造上下文拒绝、时钟与动作窗测试 |
| 联合batch | 保留原Task/量化State、统一train/infer序列化、FAST无答案泄漏、短窗FM mask | CPU分支与prompt一致性测试 |
| 独立value／优势 | 三相机、pre-READ历史、action实际可见条件；物理时间return、AH/AA与IH/IA | CPU数据／数学和phase平衡门槛；未拟合真实value |
| 暖启动与round入口 | verified SFT、RECAP标签门槛、历史replay、独立示范锚定、串行阶段驱动、checkpoint接口 | CLI默认只审计、执行与训练双开关门禁 |
| 可选仿真任务 | 双样品按序核验、归位支撑和释放；物理事件只供reward/info | 事件逻辑CPU通过；未加载场景或验证可达性 |

完整接口、配置和后续检查说明见持久目录`README.md`。未从固定子任务库挑选；首轮能力范围通过任务／示范限定。

## 训练目标与梯度边界

| 目标 | 更新对象 | 不允许的流向 |
| --- | --- | --- |
| 高层READ/WRITE/SUBTASK CE | 同一VLM的可训练组，默认attention LoRA r16 | 不更新expert，不穿透工具 |
| FAST CE | 同一VLM可训练组 | 不把teacher-forcing FAST答案交给连续expert |
| 连续FM | expert及动作／时间投影 | 每层条件K/V detach，FM不更新VLM |
| 独立value CE | 独立value参数 | 不更新actor；优势仅为重标条件 |

KI在VLM允许训练的tiny模型条件下验证。CPU反向检查没有调用`optimizer.step`；并未验收真实大模型更新或学习效果。原Task/量化State格式保留，但新增记忆／子任务仍改变输入分布，必须另做同协议基线与回归评测。

## 数据与调用约束

保留SDK24、三路`480×640×3` RGB8、15×4ms名义控制周期、内部32维及H16/K1。提前终止记录真实物理步数；原始下发目标、最终drive命令、端点标志、实测状态分别保存。完整bank仅critic使用；value输入先重放验证记忆，防止伪造正文污染标签。

只有同decision／condition／policy版本的完整实际H16窗可编码FAST；未执行proposal尾部及短窗padding不作正确标签。verified SFT需显式可信来源；RECAP需拟合value版本与有效AH/AA分化，缺失／恒定优势不能冒称学习信号。value最小采样覆盖action及pre-READ两类，避免只训动作价值。

跨进程继续运行须明确actor/value checkpoint与`data.prior_episode_paths`；没有自动读取历史round manifest。示范锚定buffer独立，不把held-out评测当训练数据。trainer提供优化器状态恢复接口，但默认串行round每次重建优化器，并非自动完整trainer续训。

## 实测结果与收据

- RLinf四个定向测试文件：**57通过**，4.56s，其中16新模型、23原接口、8记忆／数据、10流水线测试。
- RoboTwin新任务及原环境合同：**30通过，1个spawn/vector相关用例未运行**，0.10s。
- 合计**87通过**；28个变更Python文件Ruff及格式检查通过，两个标签CLI帮助入口和shell语法通过。
- 持久副本代码／辅助文件与已核验staging逐文件一致；两处资产软链接存在且指向原资产路径。
- 在持久路径执行默认审计：`config_valid=true`，已知actor/norm/SentencePiece路径存在；`ready_for_execution=false`、`training_enabled=false`、`execution_performed=false`。

收据：[[30-实验记录/evidence/2026-10-17-recap-memory-cpu/validation.json|验证范围]]、[[30-实验记录/evidence/2026-10-17-recap-memory-cpu/changes.json|变更清单]]、[[30-实验记录/evidence/2026-10-17-recap-memory-cpu/audit_config.json|持久入口资产审计]]。完整JUnit、源码基线与补丁留在持久目录`reports/`。CPU测试覆盖不等于真实模型、仿真、训练或SR验收。

## 未完成与下一步门槛

1. FAST processor与独立value所需SigLIP/Gemma3本地资产未准备；本次没有下载。
2. 真实SFT权重在新task上的严格加载、原动作格式回归、文本生成与实际FAST调用未验证。
3. 新任务尚未运行物理仿真；固定场景几何下，train/eval seeds不等于留出布局泛化。
4. 原50条抓起持稳数据不提供完整核验归位技能或可靠READ/WRITE/子任务监督；需要可信新示范与SFT基线。
5. 未测联合训练显存／延迟，未更新真实actor/value，未评测SR或遗忘。
6. snapshot/restore、配对记忆信用及DPO后置未实现；本次不是原论文完整复现或研究创新成立证据。

## 关联笔记

- [[本机-RM65B-RECAP记忆联合学习实现计划]]
- [[RECAP扩展训练框架-记忆读写与高层决策的回报优化]]
- [[本机-RLinf-RM65B-5090独立环境]]
- [[本机-Linux-RTX5090]]
