---
title: 本机RM65-B RECAP记忆联合学习实现与代码修改计划
date: 2026-10-17
updated: 2026-10-18
status: 真实SFT与仿真及完整value前向通过-高层生成未通过-未训练
tags: [环境, 部署, RLinf, RECAP, 记忆, π0.5, RM65-B]
sources:
  - "[[RECAP扩展训练框架-记忆读写与高层决策的回报优化]]"
  - "[[记忆RL方案框架-冻结π0.5与显式进度记忆]]"
  - "[[本机-RLinf-RM65B-5090独立环境]]"
  - "[[本机-Linux-RTX5090]]"
source_rlinf_head: 50f22ad9a7825eeaaf4180dca5300f2fa70758be
source_robotwin_head: 931ce21f4f0e63feab661f349634d5a25bd92681
---

# 本机RM65-B RECAP记忆联合学习实现与代码修改计划

> [!info] 2026-10-17实施进展
> 用户随后授权修改、不训练，已完成独立副本代码接入及87项CPU组件测试，详见本文第10节与[[2026-10-17-RM65B-RECAP记忆联合学习代码接入与CPU核验]]。下面第1–9节保留最初只读方案及当时源码事实，不代表当前仍未改工程。

## 1. 本次确认的架构与范围

用户要求读取当前架构笔记，分析实现方案与代码修改计划，并进一步明确：**π0.5自身的VLM读取记忆、自回归生成下一子任务文本；主方案不从固定子任务库中选择，首轮只限制实际动作能力范围。** READ／WRITE的工具语法仍受约束，但不建立子任务技能ID分类头、候选句列表或子任务词汇白名单。

架构依据为 [[RECAP扩展训练框架-记忆读写与高层决策的回报优化]]；旧文件 [[记忆RL方案框架-冻结π0.5与显式进度记忆]] 的冻结PPO图保留为历史受控对照，不再作为主实现。

本轮仅阅读笔记、源码和权重文件头，形成计划并维护Markdown。没有改训练工程、安装依赖、转换权重、加载GPU模型、执行仿真或训练；本文中的新接口与文件名均为拟议修改，不能当已存在API。本轮未修改Word或既有图片。

## 2. 推荐选定的实现路线

沿用已有本机RLinf **native Torch `openpi`**，新增独立 `task=recap_hier`；复用现有RM65-B权重、图像变换、归一化与24→32维映射。不要把native PPO、外部`openpi_cfg`和ACoB冻结KV路径拼成一个actor。ACoB与PPO保留为各自范围内的对照。

现有可运行源码参考根：

`/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rlinf_local_5090_minppo_20261013`

下文`R`为该根下`code/RLinf`，`T`为`code/RoboTwin`。建议未来实施另建独立任务目录，例如`deploy/realman_gen4/recap_memory_20261017`；本轮未创建。原入口、原权重和机器人资产不覆盖，复用venv也不在原环境中直接升级依赖。

### 2.1 执行流程与状态所有权

1. 每次高层决策输入当前三路RGB、本体状态、总目标、有限可见历史和固定槽位目录。
2. π0.5自身VLM以READ模式生成少量槽位ID或不读取；此时看不到目录之外的记忆正文。
3. 确定性工具返回选中正文；同一VLM生成KEEP／WRITE／CORRECT／REVOKE等操作、槽位、值与确认状态。
4. 工具更新episode bank，形成“已读正文＋本轮写入”的可见视图。
5. 同一VLM根据当前观测、总目标与该视图，自回归生成完整子任务文本，可带对象、位置、顺序和关系。
6. 动作分支读取对应上下文的VLM隐藏条件，FM专家生成H16动作；首轮保持K1采集与执行。
7. RoboTwin真实步进并返回下一观察、奖励、终止与实际物理步数；达到固定高层窗口或终止时再决策。

记忆bank由rollout侧控制器拥有，按`env_id + episode_uuid`隔离；环境只负责物理状态、时钟、事件和观察，不替策略填入正确进度。episode重置清bank，模型切换与记忆revision均有记录。无效工具输出需显式报错／记录，不静默用专家或固定任务选择器替代。

首轮只覆盖少量可执行动作能力，例如取送、归位释放与重新抓取；这约束示范和任务分布，而不是把高层变成有限类别分类。现有抓起持稳权重并未证明已具备这些完整技能，需要另行补数据和验收。

### 2.2 初版模型接口

新task拟提供三个显式能力：

- `generate_text(mode, observation, visible_context)`：READ／WRITE结构化输出，以及SUBTASK自然语言生成；使用同一套π0.5参数。
- `sample_actions(observation, generated_subtask, selected_memory, action_condition)`：保留SDK24物理合同，不接受FAST答案作条件。
- `forward_joint(batch)`：分别返回高层CE、FAST CE、FM及各自指标；value由独立模型／worker处理。

这些是计划中的职责接口，不承诺直接兼容某个未核验的现成函数签名。生成子任务已经是文本token ID，不要求字符串往返重新分词；首版允许按角色分次调用同一VLM，但训练与执行必须使用一致的序列化和可见范围。KV只能在相同模型版本与相同上下文内复用。

## 3. 训练目标与梯度流向

令共享VLM参数为$\theta$，动作专家及动作侧投影为$\phi$，独立价值参数为$\psi$。

| 分支 | 训练标签／条件 | 允许更新 | 不允许的路径 |
| --- | --- | --- | --- |
| 高层生成$L_H$ | READ、WRITE和完整SUBTASK文本；暖启动监督，之后用$I_H$条件化CE | 共享VLM的可训练参数$\theta$ | 不通过离散token采样或工具反传；不更新expert |
| FAST $L_F$ | 记录动作$A$经FAST编码的标签；$I_A$条件化CE | 同一共享VLM的$\theta$ | 不把FAST teacher-forcing答案送给expert |
| 连续动作$L_A$ | 数据动作加噪形成FM速度场目标；$I_A$条件化FM／动作示范锚定 | $\phi$ | 每层条件K/V处stop-gradient；FM不更新$\theta$ |
| 价值$L_V$ | 分阶段实际轨迹return的分箱CE | 独立$\psi$ | value loss不更新actor；优势是数据标签，不是critic梯度穿透策略 |
| 可选配对记忆偏好 | 恢复验收后真实续跑产生的READ＋WRITE偏好 | $\theta$ | 首轮关闭；不把动作失败直接当记忆错误标签 |

同一动作记录的两条监督是$A\to\operatorname{FAST}(A)\to L_F$与$A+\text{noise}\to L_A$；当前FM预测不能自动充当自己的正确FAST标签。FAST的编码维度、归一化和padding策略必须版本化，真实物理动作仍只有24维。

质量条件顺序固定：

```text
当前观察／目标／固定目录／I_H
→ READ → 工具正文 → WRITE → 更新后的可见视图
→ 生成NEXT_SUBTASK → I_A
→ FAST预测（训练）／连续FM（训练和执行）
```

工具正文、原始输入、条件标记和padding不计生成CE；READ／WRITE／SUBTASK和FAST分别有loss mask。READ看不到未来WRITE或子任务标签，连续专家看不到FAST答案。高层采用完整序列CE，不增加固定子任务类别损失。高层任务回报通过$I_H$影响生成概率，不要求再套一套PPO；初版共用高层条件，不建立三个独立actor／critic。

### 3.1 KI的首版实现建议

文本／FAST分支保留VLM计算图；FM分支另建不含FAST标签的上下文，调用native prefix-cache接口，将**每层K与V全部detach**后交给动作suffix。这样更容易审计，代价是额外前缀计算。只detach最终hidden不够，梯度仍可能通过各层K/V绕回。

KI不冻结前向特征。高层／FAST更新VLM后，动作专家读到的条件会变化，所以每轮都要重新评测原技能，不能认为expert固定就不会退化。

## 4. 当前源码事实与关键缺口

以下均为2026-10-17定向静态核查，不代表新模型已运行。

| 现有位置（相对R／T） | 当前事实 | 对计划的影响 |
| --- | --- | --- |
| `R/rlinf/models/embodiment/openpi/pi0.py:339–408` | native训练仅FM；`:223–227`把语言位置AR mask统一设False | 补因果文本suffix、next-token shift、segment与loss mask，不可直接拿prefix逐位置argmax当生成 |
| `R/.../openpi/modules/model.py:107–110` | 数据类已有token AR／loss mask字段 | 字段存在不等于当前forward实际使用 |
| `R/.../openpi/modules/gemma.py:184–203` | embedding与词表decode共享权重 | 可复用文本投影，无需无依据新建LM head |
| `R/.../openpi/pi0.py:410–433` | 已有prefix cache基础 | 可实现增量文本生成及KI动作分支；EOS、位置、mask要单独验收 |
| `R/.../openpi_cfg/openpi_cfg_action_model.py:363–444` | 外部Torch CFG路径仍FM-only | 不能把CFG开关当完整FAST＋文本CE＋KI |
| `R/.../openpi/modules/gemma.py:228–231` | attention LoRA配置非空时抛NotImplementedError | 不能直接选native `gemma_2b_lora` |
| `R/rlinf/models/__init__.py:375–426` | 通用PEFT命名与native ModuleList分工需要适配 | 明确匹配VLM的`.0`与expert的`.1`，避免误冻expert或两边一起注入 |
| `R/rlinf/workers/sft/fsdp_cfg_worker.py:348–417` | 输入为observation/actions/advantage三元组；主动关闭gradient checkpointing | 新联合worker应使用扩展batch和可控checkpoint，不能原封不动套用 |
| `R/rlinf/data/datasets/recap/cfg_model.py:148–188` | 当前只构造prompt的positive／negative版本 | 需要高层token协议、双时间尺度条件及生成label mask |
| `R/rlinf/data/datasets/recap/value_dataset.py:61–89,309–329` | 无RealMan三相机映射 | 为RM新增transform，保留原LIBERO／Franka行为 |
| `R/examples/offline_rl/advantage_labeling/recap/process/compute_returns.py:54–85` | 默认每frame -1，末步成功0／失败惩罚 | 不能无说明覆盖RM原奖励；定义统一研究回报并保留raw env reward |
| `R/.../process/compute_advantages.py:89–134,810–814` | 无RM映射；按frame估计N步优势 | 增加高层边界与真实duration，两层标签不能直接复制同一个bool |
| `R/rlinf/runners/offline_runner.py:253–317` | 离线训练与评测，未包含完整“重新采集→拟合value→重标→联合更新”轮次 | 需新round runner或驱动；保留旧训练器 |
| `R/.../realman_env.py:336–425` | K>1返回最终观察，含真实执行count／mask；逐动作采集有限制 | 首轮K1收集每步观察，显式保存info元数据 |
| `R/.../realman_env.py:60–63`、`T/.../rl_env.py:39–56` | 不支持state restore；RPC没有snapshot／restore | 首个闭环不依赖物理配对信用 |

现存80000是训练步数，**不是词表大小**。本次只读权重头确认667条tensor、共享词嵌入形状`[257152,2048]`，未加载张量或验证语言能力。venv的`openpi/models/tokenizer.py:51–139`已有FAST codec及词表ID桥接实现，可参考其映射规则，不必扩词表；但FAST processor资产尚未确认能离线加载。现有RECAP value配置另需SigLIP2／Gemma3资产，不能由π0.5 actor权重自动补齐，也不能把配置中的`/path/to/...`当已可用。

## 5. 拟议代码修改清单

文件名可随实现小幅调整，职责和兼容门槛不变。全部在新任务副本实施，旧默认路径不改行为。

| 优先级 | 文件／模块（拟新增明确标注） | 修改职责与交付 |
| --- | --- | --- |
| P0 | `R/.../openpi/tasks/recap_hier.py`（新增）＋`tasks/__init__.py`／任务构建配置 | 独立联合task；高层生成、FAST CE、KI-FM；旧`task=rl`继续可用 |
| P0 | `R/.../openpi/pi0.py`、`modules/model.py`、`modules/gemma.py` | 文本suffix因果mask、共享decode、KV位置、EOS；暴露按角色生成与每层detach接口；默认关闭新行为 |
| P0 | `R/.../openpi/recap_tokenizer.py`（新增） | 分段拼接协议，避免旧prompt清理破坏换行／角色；FAST词表桥、token预算、三类loss mask与合法输出校验 |
| P0 | `R/rlinf/models/__init__.py`、`R/.../openpi/__init__.py`、native checkpoint模块 | native VLM专属LoRA目标、参数组与严格基座载入；保存adapter／schema／norm／tokenizer身份 |
| P1 | `R/rlinf/data/storage/episode_memory.py`（新增） | 固定目录、按ID读取、受限写入／更正／撤销、revision和来源；不持有仿真oracle状态 |
| P1 | `R/rlinf/data/schema/hierarchical_types.py`（新增） | episode、高层决策、记忆操作、实际执行与时长等schema；不能复用原流水线`stage_id`冒充语义阶段 |
| P1 | `R/rlinf/data/datasets/recap/hierarchical_dataset.py`（新增） | 保留rawRGB／实际动作，重建因果上下文和两层样本；不依赖旧VLM缓存 |
| P1 | `R/rlinf/workers/rollout/hf/hierarchical_recap_worker.py`（新增） | 同一π0.5的READ／WRITE／SUBTASK／动作调度；bank唯一所有者与固定高层窗；记录生成文本而非技能ID |
| P1 | `R/rlinf/workers/env/env_worker.py`及新轨迹导出适配 | 把真实physics duration、执行mask、episode和decision键显式传入schema，不仅藏在info里 |
| P2 | `R/rlinf/workers/sft/fsdp_hierarchical_recap_worker.py`（新增） | 多分支batch、CE／FM分项统计、梯度累积与checkpoint；独立value训练不改actor |
| P2 | `R/.../value_model/recap/modeling_critic.py`及`recap/value_dataset.py` | RM三图／状态／子任务／记忆输入；独立value参数，阶段条件化或H／A两头；value可读完整bank但不泄漏给actor |
| P2 | `R/examples/offline_rl/advantage_labeling/recap/process/compute_hierarchical_returns.py`、`compute_hierarchical_advantages.py`（新增） | 实际时钟回报、完整轨迹MC、分层优势与阈值；保留连续优势和标签来源 |
| P3 | `R/rlinf/runners/hierarchical_recap_runner.py`及独立collect／train／eval入口（新增） | 轮次快照→采集→value拟合与重标→actor更新→保存重载→独立评测；首版串行 |
| P3 | `R/examples/offline_rl/config/realman_hierarchical_recap.yaml`及测试 | 独立配置、参数范围、token预算、数据与资产版本；新旧入口验收 |
| P3 | `T/integrations/realman_rm65b_gen4/`中的独立任务／factory配置 | 新干式核验归位场景、物理奖励事件及必要技能示范；保留原`adjust_bottle/full50_sdk24`合同与原资产 |
| 后置 | `T/.../rl_env.py`、`rl_factory.py`及branch审计器 | 独立snapshot／restore阶段，通过可恢复性验收后才允许配对信用 |

共享VLM可更新不等于要全参Adam。32GB本机建议首轮冻结vision、词嵌入／共享decode及VLM基权重，给VLM attention加r8／r16 LoRA，动作expert及动作／时间投影保留可训练；这是预算建议，尚未实现或测显存。native投影的VLM候选匹配为`llm.layers.<n>.attn.(q_proj|k_proj|v_proj|o_proj).0`，必须在真实模块清单验证后注入，不盲用`all-linear`。冻结decode仍允许CE梯度经固定矩阵更新LoRA。

## 6. 数据、回报与信息边界

每条数据至少保留：原始三RGB、实测状态、总任务、实际生成token、READ集合与实际工具返回、WRITE操作、记忆前后revision及当时可见视图、子任务、模型proposal与实际执行动作、终止／截断原因、physics计数、回报来源、归一化／模型／schema／prompt版本。完整bank另存训练内部字段，actor输入构建器只暴露授权视图。

高层窗先固定长度，遇episode终止提前闭合；同一个窗不能跨episode。原schema中的`stage_id`是pipeline stage，并非本方案的语义子任务ID。动作层和高层分别保存$I_A$、$I_H$及其连续优势、阈值、value版本。

H16／K1下，未执行的proposal尾部不能标为已发生行为。动作标签优先从连续实际执行动作构建；初版只对完整、同子任务／记忆条件的H16窗构造FAST目标，边界短窗不把padding编码成正确标签。FM的有效时间／维度mask也需明确。若将proposal作为蒸馏数据，单列来源，不能冒称实际执行轨迹。

每条动作通常为15×4ms，但成功可提前停止。两层折扣按真实physics时间统一，例如以60ms为单位、经历$d$个4ms步时采用$\gamma^{d/15}$；窗口内部回报也按同一时钟累计。保留raw env reward与研究回报两个字段，不能直接照搬RECAP脚本的frame时间惩罚覆盖原奖励。

value以完整轨迹实际return做MC监督；优势中可有未来value bootstrap，两者不能混称TD训练。真终止、任务预算失败、普通时间截断和采集中断分别处理；未知结局不伪标成功或失败。旧策略混合数据的value只是经验分布近似，policy ID不能把它自动校正成新策略价值。

所有工具正文只作为输入，不作生成标签；后验$I_H/I_A$不写入当时的bank。重建训练上下文时不能把未来事实或事后真值塞进actor输入。高层变化后动作条件随之变化，需保留并重算原始输入，不永久复用ACoB式冻结KV。

## 7. 实施与验收顺序

| 阶段 | 交付 | 必须通过的门槛 |
| --- | --- | --- |
| A：接口与初始化 | 原权重严格加载、共享文本decode／生成、同一task的文本与动作入口 | 新功能关闭时原动作在固定输入／噪声下可复测；输出非空、EOS正常、因果位置正确；通顺文本不等于理解记忆 |
| B：真实batch联合训练组件 | 高层CE、FAST CE、KI-FM和参数组，先不连新任务长训 | 分loss梯度审计、优化器实际参数差异、保存重载；改变FAST未来标签不能改变同一动作上下文的FM预测 |
| C：工具和轨迹 | episode bank、READ／WRITE工具、固定窗与真实数据采集 | 不串env／episode，未读正文不可见；实际动作、图像与physics时钟对齐；重载能复建相同上下文 |
| D：技能与高层SFT基线 | 补取送／归位释放等必要动作及READ／WRITE／子任务监督 | 单技能有可复测非零成功率；同画面不同历史需不同正确决策；高层生成语义有效且能被专家执行 |
| E：最小联合RECAP闭环 | 固定版本采集、独立value拟合、两层重标、actor更新、重载与无接管新rollout | value与actor各有真实更新；rollout确实用新版本；奖励／优势有有效信号；与同架构同参数范围SFT对照 |
| F：研究效果与信用扩展 | 留出种子、扰动、记忆错误和后续策略变化评测 | 报整程SR、遗漏／重复、错误完成记忆、恢复、基础技能退化及总交互；配对仅在恢复验收后启用 |

梯度验收逐项进行：高层CE→VLM可训练组非零、expert为零；FAST CE→VLM非零、expert为零；FM→expert非零、VLM为零；value loss→actor为零。不能通过永久freeze主干来冒充KI；需在主干可训练的测试条件下确认隔离。LoRA零初始化时也不要求每个A／B矩阵第一步都非零，应检查有效adapter梯度与实际更新。

单batch和一轮保存重载只验收计算链，不证明学会记忆或任务。若所有rollout失败且没有有效优势分化，先修技能／奖励／价值，不能以多跑几次优化器更新替代学习信号。真实效果比较保持相同历史、容量、可训练范围、轨迹／标签／仿真与计算预算。

## 8. 本机资源安排与配对信用后置

资源入口为 [[本机-Linux-RTX5090]]，旧PPO／ACoB显存收据仅是历史容量证据，不证明共享VLM长文本反传、FAST大词表CE与独立value能同时驻留。

首版micro-batch=1、梯度累积、按监督位置分块词表CE，并保留可控gradient checkpointing；先串行运行采集、value拟合／推断和actor训练。高层三个角色共用一套参数，初版不同时驻留三个VLM副本。逐阶段记录GPU峰值、主存、推理时延及优化器状态；60ms合同是物理执行步长，不承诺高层推理墙钟小于60ms。

需要完整snapshot的不只是模型参数，还包括机器人／物体pose与速度、驱动目标、控制器、接触相关状态、睡眠／唤醒、RNG、时钟、hold／终止、未执行chunk和记忆revision。相同seed重置只能复建初态；重放前缀也需逐步误差验收，无法恢复的求解器状态要披露。未通过时停留在联合RECAP基线，不能先生成伪反事实偏好。配对额外数据、两侧续跑、重放和重分析成本全部计入预算。

## 9. 来源、版本与本轮核查边界

原RLinf仓库HEAD为本页frontmatter所列commit，并有已有RealMan适配改动；**任务副本没有`.git`，不能将原仓HEAD冒充其当前精确版本**。本轮通过父CodeGraph查询后确认目标deploy副本未被索引收录，随后才定向读取当前副本源码。

本轮读取的关键文件SHA-256：

| 相对R路径 | SHA-256 |
| --- | --- |
| `rlinf/models/embodiment/openpi/pi0.py` | `ad69c5fd5f5e65e5cc947d6be384e1354c8aa33f1d4db187d6f5da279f46523b` |
| `rlinf/models/embodiment/openpi/modules/gemma.py` | `33ffecc20b3655e1e99d39d3b20cf78c34ad529a8a8391f4c3a2a68f533ed6e6` |
| `rlinf/models/embodiment/openpi/__init__.py` | `c47a9a0cfb93368d2b701dc4ab09f87ab758fbe7a6b290ab6eb7c69586c9246e` |
| `rlinf/workers/sft/fsdp_cfg_worker.py` | `0f07c1a8066944f7295647ecd84e2f971529a8ec6db7260fd9ed51d828ee5d72` |
| `rlinf/data/datasets/recap/cfg_model.py` | `aff790db8800711aa1c98a0ea9bc8222729d058ad2fb8e40aaf1be8dc97d0aa2` |
| `rlinf/envs/sim/robotwin/realman_env.py` | `ff2be0d2ef089302e87216f2f3ad61a03aa0470960e6e13cffe3f764857dfe11` |

这些哈希标记只读分析所依据的文件，不是完整环境／权重一致性验收。本轮没有执行模型、仿真、单元测试或新梯度测量；文档检查也不等于实现通过。相关历史证据见 [[2026-10-13-RLinf-RM65B-本机5090最小PPO闭环验证]]、[[2026-10-15-RM65B完整ACoB仿真闭环补齐]]。

## 10. 代码实施与CPU核验（2026-10-17后续授权）

已在`/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017`接入自身VLM的READ/WRITE/自由子任务生成、FAST与逐层KI-FM、专属LoRA与严格checkpoint、episode工具和因果数据、实际时钟分层value/优势、verified SFT暖启动与串行RECAP阶段入口。新增可选双样品核验归位任务；原默认入口保留。

本次30个代码／配置文件新增或修改，2137个原常规源码基线哈希不变；28个变更Python文件静态检查通过。CPU定向测试RLinf57、RoboTwin30，共87通过，1个spawn/vector相关用例未运行。真实大模型、GPU、新场景与优化器未运行；没有训练，未证明学习收益。FAST/value资产与可信高层／完整技能示范仍缺，默认持久入口审计明确`ready_for_execution=false`。

细节、收据、边界与下一步门槛见[[2026-10-17-RM65B-RECAP记忆联合学习代码接入与CPU核验]]。配对信用、物理恢复及DPO仍后置；A–F验收表中的真实权重、参数更新、物理技能和完整闭环门槛没有因CPU通过而自动完成。

## 11. 资产与真实功能验证（2026-10-18，不训练）

FAST和SigLIP固定版本下载校验完成，真实SFT严格加载、原入口动作逐元素对照、BF16三loss纯前向、三RGB/SDK24新场景及60ms往返通过；92项CPU回归通过。修复了真实BF16 CE累加类型错误与value本地加载／预算检查。模型与单环境共驻实测整卡约11.26GiB；不是训练显存预算。

READ/WRITE/SUBTASK实际生成未通过，框架在动作执行前停止，不能称分层自主闭环通过。Gemma3首次官方访问失败的收据保留；同日用户完成CLI授权后，权重和自身tokenizer的9份文件下载校验完成，配置已填入独立资产路径。完整生产value使用真实三图／SDK24，pre_read与action两阶段201-bin无梯度前向通过，重复输出一致、参数不变；峰值allocated1556.75MiB，整卡当时3091MiB，未测训练显存。新增value投影／expert／head仍未拟合，checkpoint为空；审计ready仅表示资产／包路径齐备。原工程及真实权重保留。详见[[2026-10-18-RM65B-RECAP资产准备与真实功能验证]]。

同日继续完善：BF16／FP32各6种短语言提示仍未通过；原Orbax＋源Flax在一条纯文本路径的前两步复现相同异常续写，首步logits余弦约0.999999995，不能只归因于新增AR循环。原SFT是全参数FM目标，语言失配属于待隔离的原因假设，未随意更改共享底层数值或权重。参考高层文本＋真实动作模型＋新场景实际16步／240tick记录、17份观察和因果数据重载验收通过，奖励0、任务未成功，诊断数据被回放拒绝纳入训练；首轮包装器的防覆盖后处理错误单独保留。审计现明确标注资产路径／包发现范围，既有10项pipeline回归通过。详见同一实验记录新增节；仍未训练。

## 12. 高层SFT暖启动方案（2026-10-18，只制定，未实施）

本节沿用同一个π0.5 VLM生成READ／WRITE／自由子任务、工具执行记忆、动作专家读取逐层K/V、FAST／KI-FM及独立value的架构。物理合同仍为SDK24、三RGB、H16／K1、15×4ms＝60ms；原80000步checkpoint和入口保留。**本轮只读取代码、制定并归档方案，没有采集、改工程代码、训练或执行模型。** 下列数据量、学习率和阈值是工程起步建议，不是论文保证或已测结果。

本节的“冻结视觉塔”限于**当前单时刻三图＋文本记忆的高层暖启动**。当前尚未实现MEM短期视频编码；若接入该模块，需要另外定义视觉时序结构与可训练范围，不能把本节参数冻结表直接当作完整MEM训练方案，详见12.10。

### 12.1 目标和能力边界

暖启动先解决三件事：输出完整可执行的记忆协议；按当前可见证据维护事实与不确定性；生成能够指向当前动作能力的对象／角色条件子任务。它不负责用语言标签凭空补出归位释放、重抓等尚未验收的动作技能，也不训练独立value或生成IH／IA。

初始候选技能限于原SFT已有数据的抓取／持稳范围，仍需按当前场景逐项验收；失败状态中的正确高层标签可以训练高层，不能据此把失败动作一起作为优秀BC／FAST标签。先恢复基本读出与协议，再决定是否扩大共享表征的更新范围。

### 12.2 标签来源和生成流水线

1. **保留原始记录。** 候选来源是原50条示范的合格片段、后续真实执行的技能／新任务轨迹及当前可达的失败状态。原50条尚无可信READ／WRITE标签，不能直接视为现成高层数据；现有16步参考链路／保持探针仍为诊断材料，不重新标成示范。
2. **先分组划分，再切片。** 按原episode和派生样本族做70／15／15训练／验证／测试划分，再生成窗口、重标或语言改写。真实存在多个layout时再按layout／scene hash留出；固定几何下换seed只报告新episode泛化。评测seed、heldout、诊断和excluded_from_training标记统一拒绝进入训练。
3. **因果重放。** 首轮在实际固定高层边界0、32、64……形成样本；W32为1.92s名义物理时间。标签只用该时刻已提供的图像、实测状态、指令、目录、实际可见历史和本轮工具返回。事件发生在窗口内部时，只能影响其后允许的决策，不能回填成窗口开始时已知事实。
4. **规则标注器先产语义事实和动作意图。** 用任务规范、当前可见证据、合法记忆重放产生标签；仿真位姿／接触／事件只放教师审计侧带。每条事实绑定observation／step引用及observed_until_step。离线规则生成参考文本不等于部署时从固定子任务库选择；运行时仍由自身VLM自由生成完整文本。
5. **按角色构造三个独立因果样本。** READ标签应用工具取得真实selected快照；据此构造WRITE输入，应用已审核的WRITE标签，再构造SUBTASK输入。教师答案、工具正文和原始输入不计生成CE，回答token及EOS计CE。首轮统一短句与JSON规范形式，先查本地tokenizer确保回答≤96 token，不截断正确标签；后续确需扩大预算时同步改训练／推理配置。
6. **质量检查后另存标签。** 先用100—200个决策点全部人工审核；扩大后普通样本抽查约10%—20%，CORRECT／REVOKE、完成事实及对象绑定歧义全查。审核是看记录和文字，不是逐步人工遥操作。外部LLM最多用于后续文字改写，不能把猜测当事实，首轮不依赖它。

规则与工具校验可以自动化，但当前没有已经验收的三图语义标注器。首轮可由仿真真值提出候选事实／定位图像区域，再逐项审核图像中是否真能辨认；不可见候选不得写成确定标签。不能把自动产生JSON和完成多模态语义核验混为一谈。

| 角色 | 标签生成规则 | 主要防错约束 |
| --- | --- | --- |
| READ | 按目标、当前观测和可见历史选完成决策所需的最小槽位集合；固定排序；允许读到empty | 不能依据未读bank的隐藏内容／是否active决定标签；不默认读全库；相同actor可见上下文不应因隐藏bank不同而产生互相矛盾的目标 |
| WRITE | 空／revoked且有新证据用WRITE；active且有新证据需更新用CORRECT；明确失效才REVOKE；真正重新核实才KEEP；无新证据用operations空列表 | **任何待操作槽位先READ**；显式写confirmed，不能继承旧确认状态；遮挡不等于事实错误；original_location不可改成当前位置 |
| SUBTASK | 根据总要求、当前图像及有效最新记忆，生成对象／初始角色／动作／目标关系明确的可执行自然语言 | 不输出固定skill ID；不把不可见sample_a/b当视觉身份；不从未来成功或隐藏状态机直接取答案；不加入未验收动作技能 |

示例仅表示标签形式，须先确证当前READ返回current_progress为空且目标在画面可辨：

```json
{"read_ids":["current_progress","sample_identity"]}
{"operations":[{"op":"WRITE","slot_id":"current_progress","value":"Target visible; grasp completion not confirmed.","confirmed":false}]}
```

对应子任务可为`Pick up and hold the sample initially on the left.`；其对象／动作正确性仍要由可见目标和技能范围审核，不能只验英文通顺。若槽位已active，第二行须改为有依据的CORRECT或空操作，不能盲目重复WRITE。

### 12.3 必须先统一的信息边界

当前源码存在一个静态可达旁路：对未READ的active槽位执行KEEP，会把旧value复制到current_writes，进而暴露给actor。现有因果重放只验证同一工具规则，不能识别该旁路。**标注器、运行时和数据校验器须统一“操作槽位必须本轮已读”的前置条件**，不能只靠示范教模型守规则。本轮仅记录此待改项，没有修改代码。

当前selected保留写前快照，current_writes含写后值；同槽的CORRECT／REVOKE必须按新revision覆盖旧事实。先在标注／校验中定义同一优先级；若后续增加统一resolved view，需版本化并同步运行与训练prompt，不能只改单侧。

新场景的“核验／归位”真值使用物体位姿、接触、速度及连续30个4ms物理步。当前actor只有当前三RGB＋SDK24，history仅前两条子任务文本，并不包含这些完整测量或视觉历史。因此不能直接将隐藏task_events翻译为actor记忆的“已核验完成”。首轮只能标可见位置／当前尝试／尚待确认等与证据相称的内容；严格完成确认需要部署时同样可得的可见反馈或因果历史机制。保持三相机合同也可另行设计画面内设备指示，但这不在本轮实施，更不能拿事后成功当作当时已知。

source、provenance和审计侧带也要检查：演员可见记忆的source不得带隐藏阶段、精确真值或未来结果。原始完整bank与task_events仅供内部审计／critic，不能混入高层输入。当前collector尚未逐步保存task_events，需要新增独立privileged侧带后才可做相应教师审计。

固定“先初始左后初始右”只够测试流程记忆；目标／角色绑定需要同画面换要求、同要求换位置／历史的留出题。文本逻辑合成与真实可达历史分开标来源，不冒称物理反事实；若改变整程顺序，环境任务规范和奖励监视顺序必须同时匹配，不能只改指令。

### 12.4 数据规模与覆盖预算

计量单位为**一个高层决策点包含READ、WRITE、SUBTASK三个监督回答**，不是一帧等于一条独立轨迹。相邻帧、语言改写、错误记忆派生样本分别计数，但不能当作新增独立物理经验。

| 阶段 | 审核决策点 | 三角色回答数 | 用途 |
| --- | ---: | ---: | --- |
| 小试 | 100—200 | 300—600 | 验证文字读出可学习、协议／EOS／因果标签质量及动作保留机制 |
| 首轮完整暖启 | 1000—2000 | 3000—6000 | 覆盖对象／角色、可见历史、空记忆、进度和不确定／恢复状态；先看到留出收益再逐批扩充 |
| 条件扩充 | 3000—5000 | 9000—15000 | 仅在有新增真实状态覆盖、且验证仍受数据限制时考虑，不预先承诺必要 |

上述量指总体池，再按episode族划分。**不先采满2000个点**：先做100—200个、审核和可学习性验收，再决定是否继续。当前max_steps=160、W32，每条到时限的episode最多5个决策点；若完全依赖此长度的新rollout，1000—2000点至少约200—400条完整长度episode，提前结束还会更多，不能误说成几十条就够。

可复用旧长示范中有可见证据的片段，或增加真实录制的有效高层状态，但要单列来源和相关性。高层纯CE不要求每点后有H16动作，也无需每条轨迹最终成功；需要的是标签当时可判且下一步合理。完整双样品任务是否能在9.6s时限内完成尚未验收；若日后延长时限，需对所有对照统一，60ms控制合同不变。

首轮训练采样建议READ／WRITE／SUBTASK三角色均衡；决策场景约60%常规、20%无新事实／empty返回／不确定、20%有证据的纠正／恢复与目标角色变化，不必用伪造错误硬凑类别。错误记忆合成作为单独低比例逻辑集，不覆盖真实视觉证据集；验证集单独报告各类而不只看总CE。

### 12.5 训练哪些参数：两档暖启

| 参数／目标 | A：先恢复文字读出 | B：按需要开放共享条件 |
| --- | --- | --- |
| 语言侧LoRA | **只开最后一层17的q_proj.0和o_proj.0，r16／α16，共131,072参数** | 开现有18层q/k/v/o语言侧index0 LoRA，共3,686,400参数 |
| VLM基权重、视觉塔、共享词嵌入／decode | 全冻结 | 全冻结 |
| 动作专家、动作／状态／时间投影 | 全冻结 | 暖启期间仍全冻结 |
| 独立value | 不加载、不训练 | 不加载、不训练 |
| 损失 | 按角色归一化的高层CE；不填IH／IA | 高层CE＋逐层K/V锚定；有可信完整实际H16时才加辅助FAST CE |

源码推导依据：每层K/V在q的注意力结果及o输出投影之前形成；**最后层语言q/o改变最终文本隐藏读出，但不改变任何语言层的K/V**。当前动作分支读取逐层K/V而不读取最后层语言输出，所以A在相同图像、状态、完整prompt及动作噪声下理论上可保留原连续输出。它仍是同一VLM，不加第二个规划器／残差网络，也不改变自由文本输出目标；尚未经过训练后的实际等价测试。

这一保证只针对同一输入的底层映射。高层生成不同子任务后prompt会变，端到端动作和任务SR当然可能变化；A也不能用来证明动作专家利用了更新的物理表征。A若有格式改善但留出对象／角色不改善，先查数据可观测性与覆盖，再按需解冻少量上层或整套LoRA。语言侧final_norm的额外2048参数可另作诊断消融，不首轮同时开放。

A建议起点：BF16、microbatch1、累积8、学习率约1e-4、梯度裁剪1、1—3个epoch，根据留出指标早停；每50个optimizer update或每epoch评测一次。B可先用1e-5—3e-5，并采用现有checkpointing／分块词表CE。数据量和有效batch会决定实际更新数，不再用固定4次更新判断可学习性；真实反向／显存仍需后续获得授权才测。

### 12.6 动作保留机制与梯度

A的主目标是：

$$
L_H=\tfrac13(L_{READ}+L_{WRITE}+L_{SUBTASK}),\qquad
\nabla L_H\rightarrow\theta^{last}_{q/o\text{-LoRA}}.
$$

各角色内部对回答token归一化，包括EOS；动作专家及其他冻结参数无梯度、不得进入optimizer。启动前、单步后、保存重载后审计实际参数名／梯度／权重差异。

B开放整套LoRA后，**冻结动作专家仍不能阻止VLM条件变化导致动作漂移**。优先新增旧动作上下文的K/V保护：

$$
L_{KV}=\frac1{N}\sum_{\ell,\,X\in\{K,V\}}
\frac{\|M_\ell\odot(X^\ell_{\theta}-\operatorname{sg}[X^\ell_0])\|_2^2}
{\|M_\ell\odot X^\ell_0\|_2^2+\epsilon}.
$$

教师为原SFT、LoRA关闭时的冻结模型；学生为同一实例启用LoRA的前向，可串行计算，无需两个常驻VLM。M仅覆盖真实有效位置。教师／学生使用同一旧原生prompt、同一图像／状态，不能直接对齐不同长度的新旧prompt。每层先归一化再平均；系数按实际分项／梯度规模调，不能从损失数值大小生硬设定。

梯度为`高层CE／FAST CE／K/V锚 → 共享VLM LoRA`；KI-FM仍是`FM → 动作专家`并在VLM条件处截断。**冻结专家后，现有KI-FM没有任何保护梯度可以更新VLM**，不能只加FM系数就声称保护了VLM。若另做动作输出蒸馏并让梯度通过冻结专家输入传回VLM，这是新增的非KI保护路径，首轮不引入。

原示范保留独立motor anchor池。FAST只从可信、同条件、实际执行的完整H16经原归一化／padding／FAST编码产生目标；失败rollout、保持探针、未执行proposal尾部不能当作优秀动作标签。B的FAST最多先作为低权重辅助，不以其CE降低替代连续动作回归。

### 12.7 保留动作能力和高层能力的验收

先固定同一H16／K1／60ms、场景和时限协议，建立三种条件：原SFT＋原生prompt、原SFT＋新记忆／子任务prompt、暖启LoRA＋相同新prompt。否则prompt变化和模型更新的效应无法分开。

- A：同原生输入／噪声的逐层K/V及H16动作应与LoRA关闭一致；多状态实测后才宣称保留。同一结果还需保存重载后复验。任何差异都先排查选错层／专家索引、k/v误解冻、输入或随机性，而非直接容忍。
- 高层：留出集先看EOS／JSON完成率，初步门槛可设≥99%；合法槽位／操作校验必须通过，不能靠空操作刷正确率；分别报告READ集合、各WRITE操作和无依据事实率。小样本阈值只是先导门槛，不是统计可靠性保证。
- 语义：人工审核的对象／初始角色、顺序、目标关系准确率可先以≥95%为项目目标；同画面换要求／同要求换有效历史分别计分，多种等价自然语言按语义判分，不要求文字逐字相同。输入无法区分的状态不做强制唯一答案。
- B：在保留池监测K/V漂移与固定噪声动作差，再做相同seed配对的旧技能独立仿真SR；先用约30—50条筛查退化，报告配对结果与不确定性。不能据这一数量宣称排除了2%等小幅退化；需要更大预定样本时再扩。
- 新任务：有效高层＋动作执行的整程SR、遗漏／重复、错误完成记忆另列。高层暖启通过不等于归位／重抓技能已存在，更不等于RL收益成立。

### 12.8 实施前最小代码计划

以下均为待实施项，本轮没有修改：

| 位置／接口 | 待改内容 |
| --- | --- |
| `recap_lora.configure_recap_parameters`及model factory | 显式`high_readout_only`／`shared_vlm_warmstart`参数选择、train_action_expert=false；建立白名单后再建optimizer；保存名称清单 |
| `recap_hier.compute_joint_loss` | 先读启用分支再算loss；仅高层时直接使用已有text_loss；保留trainer指标兼容，不能先算FAST/FM再乘0 |
| `hierarchical_collator`／`prepare_actor_samples`／runner | 高层独立batch和采样；允许没有动作，禁用FAST时不调用codec；不强制高层／动作zip配对 |
| 高层标注数据入口 | 优先独立`verified_high_sft`样本或episode旁挂标签，逐分支保存审核状态；不把整个失败episode笼统标成verified_sft，防止其动作进入BC |
| 标签schema／工具／因果检查 | 记录observation_refs、decision_step、observed_until_step、证据、teacher版本、label_scope、family/layout/split；统一操作先READ、revision覆盖和actor source过滤 |
| 训练资格校验 | warm-start和RECAP复用同一heldout seed／split／diagnostic／excluded／派生族门槛；禁止把当前诊断记录改标后绕过 |
| 可选B保护 | 安全adapter开关、无梯度教师K/V、可微学生K/V、独立旧动作anchor sampler及分项日志 |
| checkpoint／评测 | 保存暖启档位、数据／教师／prompt／norm／tokenizer版本；逐层缓存和动作对照、对象角色评测、原技能回归 |

当前正式warm-start只能接收完整episode且高层／动作样本均非空，collator与joint loss即使系数0也计算三支；默认LoRA配置仍会开放动作专家。`verified_sft`只机械要求非空provenance，不证明语义可信，也不会在这一模式自动混入原技能anchor池。以上缺口必须实际改造并核查后，才执行本方案；不能只改YAML冒称已经具备。

完全无动作的图文样本另增高层样本格式，不伪造动作满足episode schema。对已有真实轨迹离线重标，应保留原生成／执行记录；新教师记忆上下文不能自动使原动作变成该上下文下的正确示范。

### 12.9 本节来源与版本

静态核查对象为独立副本`/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/code/RLinf`和对应RoboTwin；actor为原`pi05_80000_openpi`，权重指纹与前述实验记录相同。核查日期2026-10-18，当前无新的模型更新或训练结果。

- `recap_lora.py:64–105`：语言index0的18层LoRA和默认expert训练范围；131,072／3,686,400为按当前模块形状计算的参数量。
- `modules/gemma.py:319–342,382–395`及`pi0.py:424–447`：K/V产生顺序、专家独立输出、动作cache路径，支持最后层q/o方案的结构推导；不代替训练后实测。
- `recap_hier.py:124–185`、`hierarchical_collator.py:168–271`、runner的`prepare_actor_samples`／`fit_actor`：三分支强制计算及当前warm-start约束。
- `episode_memory.py:113–211`、`sample_verification.py:109–190`及rollout worker的`_decision/collect_episode`：可见视图、未读KEEP旁路、隐藏事件和固定W32调度。
- 真实前向、源Flax对照、参考仿真记录及未通过范围见[[2026-10-18-RM65B-RECAP资产准备与真实功能验证]]。本节全部训练安排均为方案，尚未验证能恢复语言能力或保留整程SR。

### 12.10 冻结视觉塔与MEM短期视频记忆的区别

用户随后要求将短期视频记忆纳入完整目标架构，已另列[[本机-RM65B-MEM视频记忆与高层SFT完整接入方案]]：三相机四帧／1.44s、MEM式视觉改造及视觉LoRA、历史状态投影、高层／FAST与KI-FM分路、动作保留及5090／数据预算。本节和第12节原单时刻方案保留为A阶段与历史对照；完整视频接入仍是方案，未改工程或训练。

2026-10-18复核[MEM v1 §III-C/D](https://arxiv.org/html/2603.03596v1#S3.SS3)：原方法确实改造ViT的attention模式以融合时间信息，并压缩传给后续主干的历史token；复用原视觉参数和固定时间位置编码，不等于原视觉计算结构不变，也不等于无需历史条件训练。详见[[MEM精读-短期视觉与长期语言记忆#4. 视频编码器内部的关键设计]]。

当前副本仅接收当前时刻三相机；文本bank、前两条子任务文本、NPZ保存和单次文本生成KV cache都不能充当短期视频记忆。后续若沿MEM路线，应增加三相机因果帧／状态缓冲、视频版编码与训练，显式选择可训练视觉参数／适配范围；也可以冻结逐帧视觉编码器另训时序适配器，但必须标为替代实现，不能称论文原结构已复现。

加入或训练视频前端会改变动作条件K/V；末层q/o暖启动的保留推导只适用于前端和输入相同的基线，不能继续保证与原单帧SFT相同。在当前KI路由中，若视觉／时序参数开放，需让高层／FAST等可微监督到达它们；FM仍在prefix处截断。视频记忆与动作保留需单独复验，本轮没有执行这一扩展。
