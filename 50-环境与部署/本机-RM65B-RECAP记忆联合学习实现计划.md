---
title: 本机RM65-B RECAP记忆联合学习实现与代码修改计划
date: 2026-10-17
updated: 2026-10-17
status: 笔记与本地源码只读核查-工程方案-未修改训练代码未运行
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
