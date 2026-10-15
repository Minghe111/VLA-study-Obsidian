---
title: 本机-VLA-Precision-RM65B最小ACoB闭环复现方案
date: 2026-10-14
updated: 2026-10-15
verified: 2026-10-15
status: 独立适配与无纠正计算闭环通过-完整ACoB纠正及学习效果待完成
tags: [环境, 部署, VLA-Precision, ACoB, RealMan, RoboTwin, π0.5, RTX5090]
precision_version: 2609.04355v3
precision_code_commit: 41edacde80d8c9d8146d610202299d71b0e79f7c
source_openpi_commit: 215abfb217dbac7d5f1273282331b9b1866c0479
target_openpi_commit: 2d70d966582e711128ad8358d8dbf23d2cc3d658
sources:
  - https://github.com/scy-v/VLA-Precision/tree/41edacde80d8c9d8146d610202299d71b0e79f7c
  - https://arxiv.org/html/2609.04355v3
  - 50-环境与部署/证据/2026-10-14-VLA-Precision-RM65B-ACoB静态核查/static_audit.json
---

# 本机 VLA-Precision：RM65-B 最小 ACoB 闭环复现方案

> [!info] 2026-10-15 执行进展
> 已创建独立运行根并完成依赖、SDK24仿真桥、严格SFT恢复和纯SFT完整回合。新增执行与失败证据见[[2026-10-15-VLA-Precision-RM65B仿真复现启动与验收]]；一次专家LoRA联合更新、保存及独立进程重载再执行已通过；完整ACoB与学习效果仍待验收。以下原静态方案保留为历史范围。

> [!abstract] 2026-10-14 静态结论（历史）
> 官方仓库已克隆到 RLinf、RoboTwin 同级目录，工作树干净。现有原生 JAX/Orbax `80000` SFT 是合理初始化源，原则上不用先转换为 PyTorch，也不要求重做 SFT；但跨 OpenPI 版本的严格恢复与零 LoRA 动作对齐还没有运行验收。官方 UR schema 不能直接接 SDK24，需要独立模型变换、NumPy 仿真桥、回放长度／终止和评测适配。本轮只克隆、读源码／元数据并整理方案，没有安装依赖、加载模型、转换权重、启动仿真或训练。

`verified` 的当前范围见2026-10-15实验记录；下列1–13节保留其原核查日期与历史表述，不把旧“未执行”条目当成当前执行状态。目标先定为 **ACoB 在自有 RM65-B 单瓶任务上的闭环验证**；九项原论文任务的同条件成绩复现需要额外材料。

关联：[[VLA-Precision精读-价值校准、动作专家更新与Q-VGM比较]] · [[RM65-B接入RoboTwin仿真]] · [[π0.5-RealMan强化学习研究路线-算法与训练闭环]] · [[2026-10-13-RLinf-RM65B-本机5090最小PPO闭环验证]] · [[本机-RLinf-RM65B-5090独立环境]] · [[本机-Linux-RTX5090]] · [[环境与部署索引]]。

## 1. 已完成与保留范围

| 项目 | 当前事实 |
| --- | --- |
| 新仓库 | `/home/admin123/liminghe/vla-benchmark/VLA-Precision` |
| 官方 origin | `https://github.com/scy-v/VLA-Precision.git` |
| 固定 HEAD | `41edacde80d8c9d8146d610202299d71b0e79f7c`，2026-09-18 |
| 许可证 | Apache-2.0 |
| 原工程 | 同级 `RLinf`、`RoboTwin`，原有未提交适配保留 |
| 本轮新写内容 | 本方案、静态核查 JSON、索引与精读笔记的增量链接 |
| 未实施 | RM65 adapter/profile/服务/配置、依赖准备、参数恢复、数据转换、GPU测试、训练 |

证据：[[50-环境与部署/证据/2026-10-14-VLA-Precision-RM65B-ACoB静态核查/static_audit.json|静态核查 JSON]]。JSON 记录三个仓库 HEAD、工作树状态、tracked diff SHA、参数元数据／统计文件 SHA；不包含凭据。克隆只新增新目录，不移动原仓库，不执行上游入口或安装脚本。

后续独立运行根建议为 `/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_local_5090_rm65b_v1`，**当前尚未创建**。新增入口放在该根下；训练输出不得指向原 SFT、PPO 或源资产目录，`overwrite=false`。需要代码实现时使用新 `codex/` 分支或独立副本，保留官方初始提交可比较。

## 2. 可复用的 SFT 权重与数据门槛

本机当前存在的原生 SFT 检查点：

```text
/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/checkpoints/pi05_realman_rgb50_all50_full_v1/full50_100k_b64_fallback32_save20k_20261008_v4_bs64_formal100k/80000
```

其 `params`、`assets`、`train_state` 均存在；该本机父目录目前只有 `80000`，没有核查远端是否已有 `99999`。`params/_METADATA` 有51个逻辑叶，NNX 保存带 `value` 包装，四卡 `write_shape` 不能当成完整逻辑 shape。

原工厂在同一部署根的 `training_v4_batch_save20k/full50_pi05_config_161_v3_batch.py`：`pi05=True`、`dtype=bfloat16`、`action_dim=32`、`action_horizon=16`、`gemma_2b`＋`gemma_300m`，无 LoRA，全参 SFT。工厂未显式传 `max_token_len`，后续应从原生模型配置／tokenizer解析；不能把转换副本里的200说成源工厂的显式设置。

| 权重类别 | 本次定位 |
| --- | --- |
| 原生 JAX/Orbax SFT 80000 | **优先初始化源**；固定 step，不自动选 latest |
| RLinf `models/pi05_80000_openpi/model.safetensors` | 同源 SFT 的 PyTorch 转换副本，ACoB 原生 JAX 路线不需要以它为桥 |
| RLinf `global_step_3/4` | 已经 PPO 更新过，不用作 ACoB 的原始 SFT 起点 |
| 原 SFT `train_state` | 保留，但不恢复成新的 ACoB optimizer；训练参数范围与目标已改变 |

原统计为 `80000/assets/realman_rgb50_full50/norm_stats.json`，state/actions 的 mean/std/q01/q99 均24维；本轮重新计算 SHA256：

```text
34cbdd09c831e04549d6c4a4e2eb321fb975bdea13bb816ec8aa3a2332d1ccc5
```

该统计针对手部 SDK 缩放、臂动作 delta 之后的表示。先锁定它，不根据在线回放重新计算统计，也不对数据做两次 delta。

**数据准备不是权重复用的同一条件。** 本机原 `source_training_files/data/*.hdf5` 当前为0，`data50_v1/native_index.json` 仍指向历史服务器目录；旧临时六条 HDF 路径也不存在。另一任务目前只定位到slot34 attempt006一条新候选的源／导出副本，export／quality／render回执尚未accepted，不自动等于合格五条或原 Full50；也没有ACoB自主提案／纠正标签，不能未经合同验收直接混用。后续可补回少量原示范用于 FM，或采集有真实物理转移的自主／纠正数据；本轮没有下载或访问服务器。

原 Full50 索引显示50条全部用于训练，validation split为空。原训练场景只能用于回归，不当作留出泛化测试。旧 HDF 也没有完整的同状态原提案／纠正、权限边界和可靠末态 RGB，不能不加核验就变成 ACoB TD 回放。

## 3. 初始化必须通过的门槛

VLA-Precision 的 checkpoint resolver 接受 manager 根、数字 step 或 `params`，后续显式设置 `initialization_checkpoint=<上述80000目录>`、`initialization_checkpoint_step=80000`。配套 assets 的自动搜索可能取首个排序后的统计文件，必须另外核对原 asset ID、24/24 shape与上述 SHA。

目标 OpenPI 固定为 `2d70d966…`，源 SFT 使用 `215abfb2…`；两者不相同。Stage II loader 没有 Stage I 那样的完整 tree equality 检查，可能将漏载基础叶留为初始化值。后续严格核查：

1. 参数逻辑键、完整 shape、dtype、NNX `value` 包装与 source metadata对应；**非 LoRA 基础叶必须全部命中**，不允许漏叶随机补齐掩盖错误。
2. 只初始化新增专家 LoRA：当前 A 随机／B为0，参考 LoRA为0；critic单独初始化。VLM、专家基座、动作投影与原 norm保持SFT来源。
3. 相同三图、实测SDK24、指令和显式噪声下，原生SFT与零LoRA ACoB reference动作对齐；解析精度和采样设置后预先定义容差，失败则停止闭环准备并定位。
4. 对同一输入，核验用于prefix编码的状态token／KV、normalized action与逆变换一致；learner已消费正确KV时，不要求其模板batch token逐位等同actor token。校验缓存来自正确的指令／实测状态和对应版本。

如果仅序列化包装不兼容，后续可做只改导入键包装、不改变数值的新兼容产物，再重复全叶核对。不能将它等同于本轮已经转换或恢复通过。**当前没有实际加载验证。**

源码：`integrations/openpi/checkpoints.py:27`、`configs.py:253`、`training_state.py:71`、`acob_stream/checkpoints.py:249`。官方参数筛选为专家LLM路径与LoRA路径的交集，见 `configs.py:32`；输入／输出投影没有自动一起训练。

## 4. 固定的 RM65-B 合同

| 层 | 保留的合同 |
| --- | --- |
| 原始物理状态／动作 | 左臂6 rad、左手6 SDK、右臂6 rad、右手6 SDK，共24维 |
| 状态来源 | 实测 qpos／手部状态；不能用 drive target或旧命令当 observation |
| 手维 | `6:12`、`18:24`；模型输入／动作目标先 `/1000`，输出恢复 `×1000` |
| 臂动作 delta | `0:6`、`12:18`，锚点为当前决策实测状态；手保持 absolute |
| norm与padding | 24维原 quantile stats；模型层 pad32，环境只执行24维 |
| 图像 | 原始480×640 uint8 RGB；头、左腕、右腕都保留 |
| 模型图像键 | `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb` |
| 控制时钟 | 正常一个目标15×4ms＝60ms；末态提前停止按实际tick/count记账 |
| 策略 | π0.5、H16、10步ODE；专家LoRA＋新critic，冻结VLM和原专家基座 |
| 生成／执行 | H16保持；执行K是共同评测协议，不能拿原K16成绩当K1基线 |

手维连续12维，不套UR的两个binary gripper索引 `(6,13)`。首先对24个物理坐标明确声明critic和参考约束范围；若改变原论文只约束body坐标的规则，应作为本体适配差异记录。FM当前默认对32维求均值，原SFT的8个padding维也参加损失；新增mask需要单独说明，不偷偷改变损失。

## 5. 架构与新增接口清单（尚未实施）

原生 JAX/OpenPI Actor/Learner通过独立客户端连接 **RoboTwin NumPy 原生后端** `RealManVectorEnv`，不把RLinf Torch wrapper或PPO runner接进ACoB。ACoB critic为独立视觉编码器＋proprio动作价值，不是沿用PPO的VLM后价值头。

建议分开策略与仿真进程环境：新JAX/OpenPI环境固定官方commit；仿真先复用已存在的隔离SAPIEN运行环境。跨进程只传有版本的NumPy/RGB/状态／动作／reward／terminal数据，避免Gymnasium、SciPy、JAX与Torch依赖互相覆盖。原仿真env中的spawn子进程要由仿真解释器启动，不从JAX learner的解释器误启动。

| 未来新增／增量位置 | 责任与原因 |
| --- | --- |
| `integrations/openpi/policies/realman_policy.py` | Full50 units、arm delta、三图example、24→32与输出24 |
| `integrations/openpi/realman_adapter.py` | actor／JIT learner统一变换；显式raw state路径，不依赖保值marker推断 |
| `data_configs.py`、`configs.py` 的新增RM65注册项 | 成对新增full-SFT初始化与ACoB LoRA profile，保留UR/Franka默认 |
| `robotics/environments/realman_sim.py`与独立service launcher | Gym/EnvironmentClient代理到现有 `RealManVectorEnv(B=1)`；不走UR TCP相对坐标包装 |
| SDK24 schema及actor选择分支 | 明确24维动作／手索引／delta mask；禁止沿用14维检查与UR夹爪规则 |
| 回放／trajectory／loss的长度字段 | 保存actual m、有效mask、真正末态和context；统一TD／MC／FM／排序 |
| 独立 `realman_acob_eval.py` | 物理预算、terminated/truncated与无接管评测；不沿用UR墙钟退出逻辑 |
| RM65 task/deployment YAML与外部启动器 | 新输出根／cache／日志／seed；原入口不改 |

现有后端可复用 `RoboTwin/integrations/realman_rm65b_gen4/rl_env.py` 的 `reset/get_obs/step([B,K,24])/close`。正常转移中映射 `executed_action_count→steps_executed`、`episode_success→succeed`；逐步奖励、终止、实际tick、有效观察、action_trace与episode ID保留。后端count／valid在`session.apply()`前递增，异常时可能包括动作尝试，必须再核对实际physics ticks／失败点与后继观察，不能仅据count断言执行了60ms。先提交有效末态再reset，拒绝将下一回合初态冒充terminal observation。

**仅换DataConfig不够。** `adapter.py:155`硬要求双臂38state/14action；`agent.py:1054`及`actor.py:63`有UR维度/夹爪索引。learner快路径在 `adapter.py:355,476,483` 只直接norm状态、识别DeltaActions／AbsoluteActions，不会自动调用任意RealMan SDK缩放。必须覆盖状态、FM目标、critic动作、执行decode四条路径；只解除shape检查会造成无报错的actor／learner不一致。

默认cached-prefix路径已按每条实际观察编码状态token，不能凭 `batch()` 的模板广播断言其一定丢失状态；应验证完整cache provenance和真正next-context。

## 6. 回放、信用与控制时钟

ACoB已有H>K：Q只看前K物理动作；FM短标签补到H16，但损失只计前K；reference也仅约束前K。未执行的H16尾部不当作环境动作标签。

RM65末态可能只执行 `m<K`。现代码TD按reward数组长度K折扣，`steps_executed`只用于统计，MC也使用固定K；固定形状本身没有表达完整的执行有效性。若真实成功／失败已关闭bootstrap，且逐步奖励位置正确，`γ^K`乘0不会造成该条TD目标错误；主要风险是非终止截断仍bootstrap、未执行尾部进入FM／排序，以及MC／next-context跨边界。未来transition至少需要：

```text
observation / next_observation / current_context_id / next_context_id
action_proposed / action_executed / correction_valid / controller_source
rewards[K] / valid_action_mask[K] / actual_m / physics_ticks
terminated / truncated / bootstrap_mask / observation_valid
episode_id / scene_seed / policy_version / norm_sha / contract_version
```

对于真实已执行前缀：

$$
R_t=\sum_{j=0}^{m_t-1}\gamma^j r_{t+j},\qquad
y_t=R_t+\gamma^{m_t}b_t\min_i Q_i^-(x_{t+m_t},a'_{\pi}).
$$

`m_t`与mask贯穿TD、MC、FM、参考／偏好排序和buffer，不能补零奖励后把未执行动作当监督。`action_executed`是环境实际采用的SDK24目标命令前缀，不能替换成测得qpos或early-stop的最终drive endpoint；这些执行后状态另存trace。正常非terminal每步60ms，早停的最终周期记录实际ticks；真正terminal关闭bootstrap。异常count可能含动作尝试，需结合真实ticks与有效后继确定`m_t`；无法确认控制转移、零执行或无有效后继的样本直接拒收。

第一版把24秒物理预算（最多400目标、6000tick）定义为有限任务时限：到时失败、bootstrap=0。回放记录剩余物理预算，critic可单独读取该信息；**策略本体输入仍为原SDK24，不向原state token追加维度**。若以后改成训练采样窗口截断，必须保留有效最后观察并允许bootstrap，不能复用当前KV。原代码 `masks=1-done` 和末行复用当前context的默认路径需专项审计。

ACoB实际执行轨迹进入Replay；只在同一决策观察下有被替换的原提案与**真实执行的有效纠正**时进入CorrectionBuffer。原提案没有后继，不做TD；普通SFT示范只做FM或经过真实回放重新验证的转移，不伪装局部纠正。默认learner等非零CorrectionBuffer门槛，禁用接管后没有纠正会等待。**2026-10-15源码修正：官方入口要求 `correction_to_replay_ratio>0` 且batch至少2，不能直接设比例0；此前“设0即可作为无纠正变体”的建议不可直接执行。当前无纠正计算探针是独立runner调用损失，未复现官方Replay＋Correction调度。**

## 7. 依赖与资源准备（仅计划，未执行）

策略使用新独立Python3.11环境及官方锁文件；不修改原Conda、SFT临时运行包或已有RLinf venv。未来先检查Python／uv／CUDA／JAX wheel与5090 sm120的兼容性，再在隔离副本准备：

```bash
# 未来依赖准备命令；本轮没有执行。
# 工作目录为固定版本VLA-Precision副本，环境及cache位置单独配置。
uv sync --frozen --group stage2
```

`stage2`包含Stage I所需依赖，不表示必须重训Stage I。仿真侧先核对现有环境清单与可用解释器，避免安装UR/Franka SDK。若新桥需要尚未存在的依赖，列入后续独立环境准备，不把依赖装到共享环境。

`jax_runtime.py`允许单设备，但 **5090上ACoB完整反传／多角色共存尚未测量**。原PPO约29.3GiB的历史结果不能推算JAX LoRA容量。关闭XLA预分配只避免预占，不构成显存硬上限；同时记录renderer、actor、learner、参考专家、critic与编译临时量。

先单独验证learner batch2（每源至少1样本；若实现最小batch需要4则记录调整），再测试Actor/Learner/仿真同时驻留。参考起始设置：B1环境、critic ensemble2、10步ODE、小回放、少量有效context，批次及cache按实测调整。不要直接照搬batch128／20万KV／百万回放；先量一条真实三相机KV大小再设磁盘／RAM预算。

四角色对应官方serve-robot、bridge、actor、learner。RM65使用新增sim service代替UR低层服务。若单卡共存不通过，可先采用分段采样→更新→重新加载→再采样的单卡闭环；保留同一ACoB目标，但它不是原ACoB-Stream并发吞吐复现。当前没有实现或运行此备选入口。

## 8. 最小闭环顺序与验收

| 阶段 | 未来工作 | 通过条件 |
| --- | --- | --- |
| A 合同与严格恢复 | SDK24往返、统计SHA、全叶恢复、零LoRA／原SFT固定输入对齐 | 基础参数无漏载；三图／state token／动作一致；无随机基础叶 |
| B 环境单步 | B1 reset、K1 step、末态、close；再测K>1提前停止 | 正常15tick/目标，真实末态有效，未执行尾部不入回放 |
| C SFT基线 | 同种子、10步ODE、固定K／物理时限／成功标准 | 记录SR、真实回报、失败类别；与原入口的协议差异清楚 |
| D 真实数据 | 少量自主与有效纠正；或恢复少量原示范并验合同 | 有可归因的物理转移、非伪造纠正、正确next context；真实正回报或可用纠正支持 |
| E 更新探针 | 小批critic warmup，再2～4次joint update | critic／专家LoRA梯度有限、参数变化；VLM／base／reference不变；loss与排序日志可追溯 |
| F 同步与恢复 | 发布新LoRA、再采样；保存／重载训练状态 | 新policy version确实执行；optimizer与critic恢复；不宣称环境轨迹逐位续跑 |
| G 独立评测 | 重载新权重、无接管，同协议SFT配对 | 完整物理回合退出、留出seed结果、资源与交互成本记录 |

**K的选择分两步。** K1仅作为最小接口／梯度探针，不能默认是已有SFT的可成功协议。效果验证先保持原H16/K16入口作回归，再评测K4等候选；选择有实际任务信号且资源可承受的共同K后固定。若仅验证K1，其SFT和ACoB都必须K1，不拿K16历史成绩作对照。K>1需先完成actual m与mask链。

SFT暂未自主成功不必否决所有off-policy学习：可通过真实有效纠正建立成功与恢复数据；但如果仍没有可用任务反馈或纠正支持，不启动长训，也不人为填成功标签。旧PPO448动作全零回报只是历史警示，不能成为ACoB已失败或已通过的证据。

建议更新探针从critic warmup 10～20步、joint 2～4步起测，这是容量／梯度验收，不是有效学习预算。不能把2～4步更新后行为变化称为SR提升。后续研究另定交互／梯度预算和多种子试验。

## 9. 奖励与独立评测协议

首轮任务沿现有 `adjust_bottle`，先固定一个瓶子的指令。若目标为转直并保持，选择现有 `upright_hold` 作主判据：升高、指定手有效接触、无桌面支撑、倾角与1秒物理持续保持满足约束；具体阈值以当前后端resolved config核对。`native_success`保留诊断，它只检查位置／高度，不能代表转直保持。

SFT和ACoB共用scene_spec、训练／留出seed池、norm、H16、K、10步ODE、噪声规则、24秒物理预算与成功判据。主SR评测禁止人工接管，接管率／恢复数据单独报告。原evaluator只在成功或墙钟timeout退出，RM65必须响应terminated/truncated，否则可能持续查询已停止的环境。

指标：完整任务SR、native诊断SR、回报、到成功的物理时间、掉落／错抓／超时、纠正次数、实际仿真tick与墙钟／GPU成本。少量配对样例只是冒烟；研究SR需明确留出布局、样本数、种子和不确定性。仿真成功不能写成实体机器人已验证，现有SDK24策略不复制论文UR力控条件。

## 10. 当前决定与未核验项

- 已完成：同级克隆、源码／元数据核查、原统计SHA、原生80000候选路径与数据缺口确认、具体接口方案。
- 可复用：原SFT基座、原统计、三相机和SDK24变换、NumPy后端及资产；不恢复原SFT optimizer。
- 必须先实现：独立SDK24 profile／actor-learner一致变换、sim IPC、actual m与末态／cache、纠正记录、独立评测。
- 待实际验收：跨版本Orbax加载、零LoRA行为一致性、GPU／编译／KV资源、有效反馈、更新同步恢复、留出SR。
- 原资产、RoboTwin／RLinf运行入口、SFT／PPO检查点和已有环境保留；本轮没有实施上述接口改动。

## 11. 证据类型与来源

| 结论 | 来源 | 日期／类型 |
| --- | --- | --- |
| 官方代码及初始化／loss机制 | 本地固定commit源码；[官方仓库](https://github.com/scy-v/VLA-Precision/tree/41edacde80d8c9d8146d610202299d71b0e79f7c) | 2026-10-14，静态核查 |
| 本机80000、51叶、norm24／SHA、HDF缺口 | `static_audit.json`及源工厂／native_index | 2026-10-14，元数据／文件清单核查，无模型加载 |
| 原生SFT逻辑shape／历史执行能力 | [[2026-10-12-RealManπ0.5八万步原场景本机推理与实时画面]]，`model_identity.json` | 2026-10-12历史实际运行；本轮没有重跑 |
| 原后端24维／三RGB／60ms及终止行为 | 当前 `rl_env.py/rl_factory.py/rl_contract.py/action_clock.py` | 2026-10-14，静态核查 |
| PPO计算闭环与全零任务奖励 | [[2026-10-13-RLinf-RM65B-本机5090最小PPO闭环验证]] | 2026-10-13历史运行，不等于ACoB验证 |
| 本文batch、K候选、更新探针与sim IPC | 本体合同＋源码分析后的方案 | 2026-10-14，计划，未实现未执行 |

完整实现、硬件控制与论文成绩需要后续分别验收；本笔记不是一键可运行配置。

## 12. 当前克隆仓库的资产清单（2026-10-15静态复核）

固定HEAD仍为`41edacde80d8c9d8146d610202299d71b0e79f7c`，源码工作树干净。本节只复核实际随仓库提供的文件与接口，不改变10月14日模型恢复／训练尚未验证的状态，也没有安装、启动环境或连接硬件。

| 类别 | 随仓库实际提供 | 未随仓库提供／边界 |
| --- | --- | --- |
| 机器人接口 | `robotics/robots/`的UR单臂、双UR、Franka客户端；UR RTDE低层服务 | 这些是实体机器人控制代码；没有RM65-B后端、URDF/MJCF/USD或机器人mesh |
| 夹爪／相机 | UR／Franka PGI、固定夹爪；RealSense及latest-frame异步采集 | 没有RM65-B灵巧手仿真本体、虚拟相机渲染器 |
| 遥操作 | 单／双键盘干预及纠正记录；README链接外部主从、VR、3D鼠标采集项目 | 外部采集工程没有因克隆本仓库而一并安装或下载 |
| 训练与评测 | OpenPI SFT、ACoB、critic、专家LoRA、Replay/Correction/Context Buffer、actor/learner同步、checkpoint及评测 | 基座／任务权重需另外准备；本机原生80000在旧部署根，不在新仓库内 |
| 任务配置 | 清洗试管与双瓶插斜架两项Stage I／II YAML；UR单／双臂部署示例 | YAML是运行参数，不包含瓶子、试管、桌面或九任务物理场景资产 |
| 仿真与假环境 | `environments/fake.py`包含FakeRobot/FakeCamera，用于空间构造和接口测试 | FakeCamera全黑图，动作主要原样返回，没有动力学、接触、物理渲染；没有RoboTwin/LIBERO/MuJoCo/SAPIEN任务后端 |
| `assets/` | `overview.pdf`、`overview.png`、`workflow.svg`、`workflow_zh-CN.svg`四份说明图 | 该目录不是机器人／物体仿真资产库 |
| 数据 | 数据索引、预处理与LeRobot配置代码 | 本次文件清单没有HDF5／Parquet示范数据或任务模型权重 |

本轮对`.git`外139个文件只读检查，未见URDF/MJCF/SDF/USD/STL/OBJ/DAE/STEP等资产文件，也未见safetensors/PT/PTH/PKL/NPZ任务权重或HDF5/Parquet示范文件。此为当前本地克隆清单，不表示所有外部项目都没有相关资产。

依据：`git ls-files`与本地文件清单、`robotics/robots/factory.py:59`、`robotics/cameras/factory.py:37`、`robotics/grippers/factory.py:63`、`robotics/environments/fake.py:41/51/79`、`README_zh.md`。因此RM65-B本体、灵巧手、三相机与瓶子／桌面物理场景仍从现有RoboTwin接入，VLA-Precision提供训练端；计划中的独立bridge尚未实现。

## 13. VLA-Precision适配仿真器的选型（2026-10-15）

> [!abstract] 当前项目建议
> **首选现有 RoboTwin 2.0 的 SAPIEN 仿真栈。**这是为了复用 RM65-B 资产、任务场景与24维控制合同；VLA-Precision 仓库本身没有内置 RoboTwin、SAPIEN、Isaac Lab 或 MuJoCo 后端，所以仍需实现环境桥接，不能视为开箱即用。

### 框架与环境接口

VLA-Precision 的 ACoB actor 通过 `env_reset`、`env_step` 读写观察、动作、奖励、终止和 `info`；环境服务接受 Gymnasium `Env` 工厂。当前默认工厂注册的是 UR、双 UR 和 Franka，扩展文档提供机器人工厂注册方式。由此可判断，ACoB算法层不锁定物理引擎，但仿真器必须经适配满足动作块、图像、本体状态、奖励、terminated/truncated、有效执行长度和人工纠正信息等契约。内置 `fake_env` 只返回零状态／黑图并原样回传动作，不是可用于任务学习的仿真器。

### 候选比较

| 仿真器 | 对当前 RM65-B ACoB 的匹配度 | 优点 | 主要适配成本与适用场景 |
| --- | --- | --- | --- |
| **RoboTwin 2.0 / SAPIEN** | **最高，建议主路线** | 当前 RoboTwin 已有 RM65-B 与因时四代手模型、`adjust_bottle` 场景和本地 Full50 `RealManVectorEnv` 工作树实现；该环境边界为 `[B,K,24]`，包含三路 480×640 RGB、实测SDK24、15×4ms=60ms控制及终止/执行记录，和本项目合同最接近。 | 仍需把该后端包装成 VLA-Precision 的 Gymnasium/`EnvironmentClient`，统一三相机键、RGB顺序与 OpenPI resize/pad、策略动作转换、奖励/终止、干预纠正和实际执行长度。相关 `rl_env.py` 等文件当前在 RoboTwin 本地工作树中未跟踪，实验前应固定代码快照；本轮未运行该后端或策略。 |
| **Isaac Lab / Isaac Sim** | 中；适合吞吐优先的后续路线 | 官方框架提供 PhysX、向量化渲染、域随机化并支持用户加入机器人，适合大规模并行采样和多任务机器人学习。 | 当前没有 RM65-B/四代手、对应场景和三相机任务资产；需要导入/校验机器人与手模型，重建 `adjust_bottle` 任务、相机、接触/成功奖励以及24维动作桥。若研究目标是先验证 ACoB 在现有本体上的闭环，迁移工作量高于复用 RoboTwin。 |
| **MuJoCo / MJX** | 中低；适合简化动力学或状态策略原型 | MuJoCo 支持 URDF 输入、接触动力学和离屏 RGB/深度渲染；MJX 与 MuJoCo Warp 提供并行模拟后端，适合轻量控制实验。 | 当前没有已对齐的 RM65-B/四代手 MuJoCo 模型、场景和观测定义；需要重建并重新校验多指接触、驱动、相机与任务判据。若核心问题是视觉闭环和沿用现有 RealMan 资产，迁移不划算。 |

### 研究结论的边界

VLA-Precision论文聚焦真机在线RL，论文明确将仿真到真机差距视为视觉观测与接触动力学迁移的限制。因此 RoboTwin/SAPIEN 适合先做**ACoB机制与RM65-B仿真闭环验证**，能提供固定种子、可重复重置和成本较低的交互；它不能单独复现论文的“真机在线学习”结论。仿真人工接管也应单列为模拟干预，不能等同于真实操作者在实体系统中的纠正代价与效果。

### 来源与核查范围

- VLA-Precision 本地代码：`41edacde80d8c9d8146d610202299d71b0e79f7c`；静态核查 `robotics/environments/factory.py`、`robotics/environments/server.py`、`acob_stream/interfaces.py`、`acob_stream/actor.py`、`robotics/environments/fake.py` 与 `docs/EXTENDING_zh-CN.md`。
- RoboTwin 本地代码：`931ce21f4f0e63feab661f349634d5a25bd92681`；静态核查仓库原生 RM65-B 文件及当前未跟踪的 `integrations/realman_rm65b_gen4/rl_env.py`、`rl_factory.py`、`rl_contract.py`、`RL_ENV.md`。未改动或提交这些文件。
- 外部官方资料：[VLA-Precision论文](https://arxiv.org/html/2609.04355v3)、[VLA-Precision仓库](https://github.com/scy-v/VLA-Precision)、[RoboTwin官方安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)、[RoboTwin官方仓库](https://github.com/RoboTwin-Platform/RoboTwin)、[Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/main/index.html)、[MuJoCo官方概览](https://mujoco.readthedocs.io/en/stable/overview.html)、[SAPIEN文档](https://sapien-sim.github.io/docs/)。

本节为当前代码、已有资产和官方文档的静态适配评估；没有安装依赖、启动任何仿真、加载权重或训练，也没有验证三种仿真器之间的物理数值等价性。


## 14. 2026-10-15 仿真复现已开始执行

独立运行根已创建，实际完成官方锁依赖、SDK24三相机60ms仿真桥、51基础叶严格恢复、源／目标OpenPI单输入逐位对齐、纯SFT完整400目标基线、8条真实转移、1次critic warmup＋1次专家LoRA joint update、动态检查点及独立新进程8步再执行。28项CPU检查通过。

本轮SFT seed2010 K1为0/1、回报0；无有效纠正对，因此只认无纠正计算闭环。full-SFT到零LoRA存在已归因的低精度图漂移，cached实现本身与原生零LoRA逐位一致。完整ACoB的成功FM、纠正偏好、正比例混合Buffer／Stream和留出SR尚未验收。

全状态JIT曾OOM，改为数值等价的分区输出后，更新阶段整卡峰值17.81GiB，独立恢复带仿真21.04GiB。原三仓库与关键资产SHA保持；未操作真机、远程、长训或Git提交。具体证据、失败修复、后续门槛和视频见[[2026-10-15-VLA-Precision-RM65B仿真复现启动与验收]]。
