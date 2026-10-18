---
title: 本机RM65-B：MEM视频记忆与高层SFT完整接入方案
date: 2026-10-18
updated: 2026-10-18
status: 已接入-协议SFT128更新完成-高层协议未通过
tags: [VLA, MEM, RECAP, SFT, π0.5, RM65-B, RTX5090]
sources:
  - https://arxiv.org/html/2603.03596v1
  - https://www.pi.website/research/memory
  - "[[MEM精读-短期视觉与长期语言记忆]]"
  - "[[本机-RM65B-RECAP记忆联合学习实现计划]]"
  - "[[2026-10-18-RM65B-RECAP资产准备与真实功能验证]]"
  - "[[本机-Linux-RTX5090]]"
source_rlinf_base: 50f22ad9a7825eeaaf4180dca5300f2fa70758be
source_robotwin_base: 931ce21f4f0e63feab661f349634d5a25bd92681
---

# 本机RM65-B：MEM视频记忆与高层SFT完整接入方案

> [!important] 2026-10-18协议SFT小试实测
> 后续已在独立实验目录准备480条高层标签（360／60／60）并执行32→128次protocol更新；四张末层q/o LoRA实际改变，冻结参数哈希保持，8例同原输入的K/V与动作差0。val16平均CE16.6408→11.7451，但EOS／协议验收仍0/16，未进入video／FAST／expert训练。数据可由纯文本规则480/480解答，因此不作为多模态或MEM学习收益。详见[[2026-10-18-RM65B高层协议SFT首轮暖启动与保留验证]]；下方“未正式训练”等为方案／早期代码阶段的历史快照。

> [!info] 2026-10-18后续执行
> 用户随后授权修改代码和小规模验证：34文件已同步到独立RECAP工程，175项CPU通过、1项未运行；实际26步／390tick历史、原入口三输入动作差0、四帧真实前向／反向和隔离一步更新／回退、跨进程紧凑重载通过。高层生成仍未通过，尚未开展审核数据SFT。详见[[2026-10-18-RM65B-MEM代码接入与小规模验证]]。下文保留执行前完整方案，历史“本轮未修改”指制定方案阶段，不表示当前工程仍未接入。

> [!abstract] 推荐路线与证据边界
> 把短期视频编码接入现有π0.5视觉前端，让同一个VLM和动作专家使用近期历史；继续由该VLM生成READ、WRITE和自由子任务文本。视觉基权重先冻结，训练少量视觉LoRA、历史状态投影及VLM LoRA；先稳定高层与视频表征，再开放专家LoRA。知识隔离保持：高层／FAST监督更新视觉与VLM，FM只更新动作专家。
>
> 这是针对π0.5、三相机和单5090的**MEM式迁移方案**，不是原π0.6-MEM训练规模的复现。当前代码仍无视频编码。本轮仅阅读源码、论文和既有收据并维护方案Markdown，没有改工程、安装、采集、执行模型、测新显存或训练。

## 1. 已有实现、目标架构和合同

### 1.1 已有事实

独立工程根为 /home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017。本文以其下code/RLinf为R，code/RoboTwin为T。YAML中的base commit是来源版本；副本含此前授权的未提交接入修改，不能只用base commit复现当前工作树。

当前输入为**当前三RGB＋实测SDK24＋文本bank＋前两条子任务文字**，没有历史帧接口、视频ViT或历史状态投影。已有严格SFT加载、单输入动作对照差0、三loss无梯度前向、16步参考高层驱动仿真及独立value前向收据；自主READ／WRITE／SUBTASK仍未通过，value新增层未拟合，未训练。详见[[2026-10-18-RM65B-RECAP资产准备与真实功能验证]]。

上次[[本机-RM65B-RECAP记忆联合学习实现计划#12. 高层SFT暖启动方案（2026-10-18，只制定，未实施）|单时刻高层暖启动方案]]保留为阶段A／对照；其中冻结视觉塔和末层q/o保留同输入K/V的推导，不能扩张为视频前端更新后的动作保留保证。

### 1.2 保持的接口

| 项目 | 约定 |
| --- | --- |
| 环境动作 | SDK24绝对目标：左臂6弧度、左手6 SDK、右臂6弧度、右手6 SDK |
| 环境观察 | 当前三RGB8，均480×640×3：full_image、left_wrist_image、right_wrist_image；同一物理边界实测SDK24 |
| 内部预处理 | 原resize/pad到224、归一化、手单位缩放、24→32 padding及动作反变换 |
| 连续动作 | 内部32维、H16；执行K_exec=1后重新观察 |
| 物理时钟 | 每完整控制步15×4ms＝60ms；终止时不足15tick的步显式记录 |
| 高层频率 | 第一版仍W32＝1.92s名义物理时间；三角色共享该决策时刻快照 |
| 原资产／入口 | 原SFT、归一化、原推理和RoboTwin入口不覆盖；MEM使用独立配置 |

视频帧数写作T_hist，避免与K_exec混淆。60ms为物理控制合同，**不代表视频和三轮文本推理可在60ms墙钟内完成**；实际耗时与仿真时间分别记录。首轮不引入RTC／异步控制。

### 1.3 完整执行图

~~~mermaid
flowchart TB
    O["每60ms的新观察<br/>三RGB＋实测SDK24＋physics_step"]
    B["episode独立因果环缓冲<br/>四帧：t-24、t-16、t-8、t"]
    V["MEM式SigLIP视频编码<br/>空间／因果时间attention<br/>压回每相机256个融合token"]
    Q["历史状态连续投影<br/>过去三时刻各一个token<br/>当前状态仍用原文本"]
    L["同一个π0.5 VLM<br/>READ→检索→WRITE→更新<br/>→自由生成子任务文本"]
    M[("episode文本bank<br/>READ门控／revision")]
    C["动作上下文的VLM逐层K/V<br/>视频／状态／目标／记忆／子任务"]
    A["连续动作专家<br/>Flow Matching／ODE采样"]
    E["H16内部32维→SDK24<br/>执行首动作60ms"]
    F["高层文本CE＋FAST CE<br/>更新视觉／状态／VLM适配参数"]
    X["KI-FM<br/>仅更新动作专家LoRA"]
    O --> B --> V
    B --> Q
    V --> L
    Q --> L
    M -->|"只显示已读正文"| L
    L -->|"工具执行写入"| M
    V --> C
    Q --> C
    L -->|"子任务＋有效记忆＋总目标"| C
    C -->|"stop-gradient：仅切反传"| A --> E --> O
    F -.-> V
    F -.-> Q
    F -.-> L
    X -.-> A
~~~

VLM是同一组权重的不同角色调用。高层输出文本token；动作专家读取对应上下文隐藏K/V，不直接读取字符串。第一版沿用文本输出→现有prompt/tokenizer→动作上下文的接口；不同prompt不能误共享整段VLM cache。

独立Gemma3-270M value保留为后续RECAP模块，SFT不加载／更新，不用未拟合value生成优势。数据预留history_refs；后续需单独验证value的历史条件，当前单时刻value前向通过不等于视频价值学习完成。value损失不反传actor。

## 2. 视频和状态接口

### 2.1 默认采样

主配置T_hist=4，间隔8个完整控制步＝480ms，取t−24、t−16、t−8、t，**包含当前，共1.44s历史**。继续每60ms保存原观察，离线按帧引用取窗。

| 配置 | 覆盖跨度 | 用途 |
| --- | ---: | --- |
| T_hist=1 | 0 | 原单帧对照／保留池 |
| T_hist=4，间隔3步 | 0.54s | 局部运动对照 |
| **T_hist=4，间隔8步** | **1.44s** | 首轮主配置 |
| T_hist=6，间隔5步 | 1.50s | 近似同跨度，增加密度 |
| T_hist=6，间隔8步 | 2.40s | 单独考察更长历史 |

使用真实physics timestamp／frame age。开局缺帧用valid mask，跨episode清空；复制首帧不能冒充多次真实观察。历史状态逐帧同步，不能把当前状态复制给历史腕相机。图像时间均不晚于样本决策时刻；实际末步不足60ms时按物理时间匹配，不伪造固定间隔。

### 2.2 视觉编码改造

[MEM v1 §III-C/D及附录C](https://arxiv.org/html/2603.03596v1#S3.SS3)提供时空分离attention、每第4层时间融合、复用原投影、固定时间编码和上层丢弃旧帧token，并保留FAST／FM知识隔离。原报告基于π0.6、448输入和大规模混合训练；本方案的π0.5／224／三相机／LoRA／小数据预算是工程适配。

当前actor SigLIP为27层、width1152、16heads、patch14，224输入每相机256patch。建议：

1. 中间特征按B×3×T_hist×256×1152组织，三相机共享视觉权重。逐帧保留原空间路径；第4、8、12、16、20、24层加入同一patch的跨帧因果attention，复用该层QKV／out。
2. 使用固定相对时间编码，当前e(0)=0；时间融合只在同一camera内进行，跨相机由VLM融合。实现须核对附录公式，不能凭“多加一次残差”宣称单帧等价。
3. 首版在第24层后丢弃历史帧token，后3层只处理当前融合表示。**第24层是工程选择，非论文指定值**；预算不足时，较早压缩作为另一个受控配置。
4. 每相机输出256×2048，共**768个视觉token**给VLM，不将12张图全部展开传给语言主干。历史状态和文字仍增加总前缀，不能说整个序列长度不变。
5. 首版整窗重算。推理时同一高层决策可复用视觉结果；训练各损失分支独立重算，不能detach可训练视觉特征，也不复用过期模型／时刻的深层cache。

单帧legacy路径完整旁路时序、历史状态和额外位置／压缩。仅当所有新adapter关闭，才要求与原SFT数值对照。e(0)=0或零状态向量本身不保证等价：额外token即使数值零，也会改变attention归一化和位置。

### 2.3 历史本体状态

保留当前状态的原量化文本；过去T_hist−1时刻各加一个连续状态token，**三个token只加入一次，不按相机复制**。历史SDK24先走原单位／归一化及32维padding，再由Linear(32,2048)投影，配固定时间编码和valid mask，含bias共**67,584参数**。

这是兼顾旧SFT的迁移设计，区别于原MEM全部K个状态连续化。状态token进入Gemma prefix，属于**VLM融合历史状态**，并非让1152维SigLIP时序attention直接以状态为条件。dtype／device／mask须统一，允许投影保FP32而输出显式cast；不能重现新增模块BF16／FP32不匹配。

## 3. 训练范围、阶段与梯度

### 3.1 参数范围

| 模块 | 首轮开放范围 | 冻结范围 |
| --- | --- | --- |
| 视觉编码器 | 六个时间改造层，共享视觉LoRA r8；packed-QKV共用A＋out独立LoRA，**331,776参数** | 原视觉base、patch、MLP、norm及视觉输出投影；时间编码固定 |
| 历史状态 | 32→2048投影，67,584参数 | 当前原状态文本及原归一化 |
| VLM语言主干 | A末层q/o r16：**131,072**；B按需扩至18层q/k/v/o r16：**3,686,400** | base、词嵌入及tied输出权重 |
| 动作专家 | A/B全冻；C新增专家attention LoRA r8 | 专家base、原动作／时间投影保持冻结；如以后要调投影，另加可关闭adapter |
| 独立value | SFT不加载／不更新 | 后续独立拟合／验收 |

视觉LoRA计数为6层×6×1152×8＝331776；若Q/K/V/O各有独立A则为442368，不能混称。时空共用同一组视觉adapter，不另建一套QKV。现有Gemma helper仅注入语言index0，**视觉packed MHA和专家index1 LoRA均待实现**。

B全语言档合计约**4,085,760可训练参数**，不是全量更新3.35B。参数少节省optimizer，仍需穿过冻结中间层反传，不能省掉全部激活。

### 3.2 暖启动三阶段

| 阶段 | 输入／数据 | 目标 | 更新与退出门槛 |
| --- | --- | --- | --- |
| A：高层协议 | 100–200个审核决策轮，先单时刻三图＋文本bank | READ／WRITE／SUBTASK CE | 只开末层q/o；先通过EOS／JSON／工具／绑定。若小读出档不足，开放更深语言LoRA，而非一直训小头 |
| B：视频语义／动作表征 | 同步视频／状态、高层标签、审核的实际H16窗；混入旧技能单帧池 | 高层CE＋FAST CE＋单帧表征保留 | 视觉LoRA、历史状态投影和VLM LoRA；expert冻。验证视频利用和两条路径的动作质量 |
| C：动作对齐联合SFT | B数据＋正确技能示范；失败状态的高层标签仍独立 | 高层CE＋FAST CE＋KI-FM＋保留项 | B范围再开专家LoRA；若B已过动作门槛，可暂不开放expert，报告实际训练范围 |

本表均为未来安排，未执行。microbatch1、累积8作为容量起点；各角色／分支按有效样本／token分别归一化。可用VLM LoRA约1e−4、视觉LoRA约1e−5至5e−5、状态投影约1e−4、专家LoRA约1e−5至5e−5起步，均为调参建议。先检查1–3遍数据的留出结果，协议稳定后转B，不用训练CE代替语义／动作验收。

C按启用分支独立构图：high forward/backward→FAST forward/backward→FM forward/backward，全部累积完再统一clip／step。各支重算视觉／prefix，不复用已释放的图；否则需retain_graph且失去省显存目的。三支之间不能step或更改adapter状态。高层与动作样本不强制一一zip。

### 3.3 两条监督路径

设视觉／状态／VLM参数为φ、ρ、θ，专家为ψ：

$$
L_H=-\sum_{j\in\{\mathrm{READ,WRITE,SUBTASK}\}}\log p_{\phi,\rho,\theta}(y_j^*\mid x_j),\qquad
L_F=-\log p_{\phi,\rho,\theta}(\operatorname{FAST}(A^*)\mid x_A).
$$

高层目标是审核文本及EOS；FAST目标是**实际记录的动作块**经codec得到的数字动作token，不是FM当前预测或子任务词语。prompt／工具返回正文不计回答CE。工具不可微，仍可用teacher forcing训练目标token。

$$
L_{\rm FM}=\|v_\psi(a_\tau,\tau;\operatorname{sg}(KV_{\phi,\rho,\theta}(x_A)))-u^*\|^2_{\rm valid}.
$$

| 损失 | 视觉／历史投影 | VLM | 动作专家 |
| --- | --- | --- | --- |
| 高层CE | B/C可达 | A/B/C开放LoRA | 无 |
| FAST CE | B/C可达 | B/C开放LoRA | 无 |
| KI-FM | **零** | **零** | C开放LoRA |
| 单帧教师K/V约束 | B/C共享视觉LoRA可达；历史投影被旁路 | 开放LoRA | 无 |
| 后续独立value loss | actor零 | actor零 | actor零 |

当前pi0.compute_loss的KI分支对整个prefix cache使用no_grad并detach。**MEM原论文也隔离expert到VLM梯度**，接入视频不要求关闭KI，视觉可由高层／FAST学习。离散生成的子任务和环境执行之间没有SFT可微梯度；高层靠其标签训练。

首轮不强制新时序loss。先检验正确历史是否有用；若持续走当前帧捷径，再单列可见运动事件CE／时序对齐监督，不把隐藏完成标志作为视觉事实。

## 4. 动作能力保留

冻结专家只能固定映射，不能固定其输入条件；视觉／VLM变了，输出仍可能变。建议：

1. **严格回退。** 原checkpoint／入口不覆盖。MEM关闭＋所有adapter关闭才是旧模型基线；多真实状态、同prompt／噪声比较K/V、H16及SDK24，保存重载复验。
2. **旧技能单帧回放。** B/C先将25%–40%微批分给原技能合格片段和原动作上下文；MEM历史旁路，**共享视觉／语言LoRA仍启用**。比例是起点，不是保留保证。
3. **可微表征锚。** 同实例adapter-off＋MEM-off、no_grad取得原SFT K/V；恢复全部开关后，学生在相同单帧输入／token位置／mask取得可微K/V，按层有效token和尺度归一化约束。若学生单帧也自动关LoRA，锚恒为零，无保护作用。
4. **教师成立条件。** 所有base始终冻结，MEM／历史状态在单帧旁路，才可用同实例关闭adapter表示原SFT。原action／time projection也不得悄悄解冻。教师开关必须在学生forward及checkpoint重算前恢复，不同时驻留第二个大模型。
5. **不压制有效视频。** 不对全部四帧输出强贴单帧教师；不同历史本应改变决策。新增token会改变RoPE位置，不能直接对位置不同的cache做逐元素MSE。单帧保护也不覆盖被旁路的历史投影及四帧交互，因此另测四帧退化。
6. **专家保护。** C加入旧技能KI-FM；可选同状态／噪声／流时间的旧专家速度场蒸馏，仍只训expert。旧FM被截断，不能保护视觉／VLM。
7. **行为验收。** 固定噪声动作差默认是指标，不能通过现有KI路径反传上游；如要动作蒸馏保护视觉，须另外定义非KI路径。原技能同协议配对SR、掉落、边界和角色混淆同时报告。

末层q/o的K/V不变推导仅适用于同前端／同token输入的A基线；视频／状态更新后不再覆盖全模型。旧分支能回退，与新分支已经保留能力是两项结论。

## 5. 数据、标签与采集预算

### 5.1 记录和因果边界

每60ms记录三图、实测状态、physics_step／时间、episode UUID、policy版本、实际动作／奖励／终止；窗口只存history_refs和mask。标签侧带包含decision_step、observed_until_step、图像证据、teacher／prompt／norm版本、对象／初始角色、layout／family／split和分支质量状态。

高层数据优先用独立verified_high_sft或原episode旁挂标签。没有动作的图文样本不伪造动作；失败episode的正确高层标签不能把坏动作一并标成优秀BC。原50条示范仅是候选保留池，需先确认数据齐备、真实时间及三相机可重建，不能补假视频。既有参考高层16步、保持探针、heldout诊断继续排除于训练。

动作数据优先复用经审核的已有本体示范／可执行技能片段；缺少新技能时，可先单独验收仿真路径专家，再收集真实物理步进成功轨迹。不能假设旧抓瓶专家已覆盖新场景的归位／释放／重抓。仿真真值可以规划专家和验收动作，但教师高层标签仍只写部署可见的事实；失败rollout可补高层失败判断，却不能自动补出成功动作示范。

| 标签 | 生成／审核规则 |
| --- | --- |
| READ | 根据当前及近期视频、总目标和固定目录，选择最小必要槽位；不能依据未读正文生成标签 |
| WRITE | 新证据才新增／修正；明确错误才撤销；无新证据空操作。操作槽位先READ、显式confirmed；新revision覆盖旧selected快照 |
| SUBTASK | 根据可见证据和有效已读记忆，自由生成对象／初始角色／动作／目的地明确的文本；首轮限制技能能力，不选固定skill ID |
| FAST／FM | 对应上下文下真实执行、审核正确的动作；预测H16未执行尾部、填零保持、错对象或坏动作不得充优秀标签 |

在固定高层边界逐条因果重放READ→检索→WRITE→更新→SUBTASK。规则可产候选，但目前无已验收自动三图语义标注器。首轮100–200轮全部审视频与文字；扩大后普通样本抽查10%–20%，CORRECT／REVOKE、完成事实和绑定歧义全查。这不是逐步人工遥操作；首轮不依赖外部LLM猜事实。

视频不会自动使隐藏判据可见：当前严格核验用到位姿、接触、速度和30个4ms tick，480ms采样不能证明连续120ms接触／驻留。应写“正在核验／已尝试／尚未确认”等证据支持内容；task_events只在独立privileged审计侧带，不能泄入actor source或高层答案。确认完成需要部署同样可得的可见反馈。

准备当前画面相近、过去不同的诊断：遮挡前身份、接近／离开、抓起／掉回、搬动后仍保留初始角色。先确认三路当前图确实不足，否则不能证明视频必要性；加入无历史状态／时间错位对照排除状态捷径。

当前任务monitor按spec.sample_ids判定顺序，与goal文本独立。后续“同画面换要求”必须从统一任务规范同步生成**指令、教师标签、期望对象／顺序与成功判据**，并版本化检查一致性；不能只改prompt。该接口完成前，固定左→右任务只能报告初始角色／流程绑定，不能称指令条件语义泛化。

### 5.2 数据量计数

在当前max_steps160、W32、H16下，满长度且无提前终止的episode：

- 高层在0／32／64／96／128决策，仅5轮，每轮3条文本CE。
- 每32步条件段17个完整H16窗，最多85个FAST窗。
- FM最多160个起点，其中75个为不足H16的有效后缀，按mask训练，不得补齐计为完整FAST。

| 训练预算 | 满长度episode参考 | 高层轮 | 文本条数 | 完整FAST上限 |
| --- | ---: | ---: | ---: | ---: |
| pilot | 20–40 | 100–200 | 300–600 | 1,700–3,400 |
| 正式范围 | 200–400 | 1,000–2,000 | 3,000–6,000 | 17,000–34,000 |
| **推荐正式起点** | **300** | **1,500** | **4,500** | **25,500** |

以上为**待采集预算，非已有资产**。过滤坏动作、缺帧／提前终止后更少，相邻重叠窗口不等于独立经验。若要求四帧全部有效，首段17个FAST窗排除，降为68窗／episode；建议保留开局mask样本学习初始绑定。

验证／测试另计，如300训练＋40验证＋60测试episode。先按episode、layout、派生族划分再切窗，重标／改写不跨集。旧技能配对30–50回合只适合初筛大退化，不足以排除2%等小SR变化。

当前160步仅9.6s，不能假设完整双对象取送／核验可完成。先测物理时间需求，再给所有比较组统一回合预算，并重新算数据量；60ms合同不变。也可先单对象／单阶段验证视频。

### 5.3 存储需求

按161观察／160步episode、原生三RGB uint8未压缩：每episode约424.51MiB，300训练episode约124.37GiB；400个含验证／测试共约165.82GiB。300episode的224处理副本约20.31GiB，但不能替代原图证据。

压缩比需真实样本测量，不预设倍率承诺。只存帧引用、按需解码，不把每个视频／H16窗复制一份；旧保留池、更长回合另计。

## 6. 5090显存和主存

### 6.1 已有实测收据

资源入口[[本机-Linux-RTX5090]]：总32607MiB≈31.84GiB显存；主存约31.10GiB。CPU／主存完整库存最近核验2026-10-13，GPU于10月18日单独复核。本轮仅读旧收据：

| 对象 | 数字／范围 |
| --- | --- |
| 单时刻actor无梯度 | allocated峰值约6.68GiB，整卡约8.25GiB |
| actor＋一个SAPIEN场景reset采样 | 整卡约11.26GiB，不是训练峰值 |
| actor加载CPU峰值 | RSS约24.18GiB，非稳态RSS |
| 完整独立value无梯度 | allocated约1.52GiB，整卡当时约3.02GiB；非联合共驻训练 |
| 旧PPO／ACoB更新 | 其他实现的收据不能保证新增MEM反传容量 |

### 6.2 新增结构的静态估计

四帧三相机实际选中原图10.55MiB；要每60ms精确滑窗取t、t−8、t−16、t−24，ring需保留25时刻，原生RGB约**65.92MiB CPU**。六帧stride5需26时刻，约68.55MiB；stride8需41时刻，约108.11MiB。不能把只选中的四帧叫全量环缓冲。

四帧patch隐藏张量的单份BF16约6.75MiB。单层独立帧空间attention一个BF16稠密矩阵约24MiB，逐patch时间attention约0.375MiB；六帧分别36／0.844MiB。它们只是形状预算，非反传峰值；当前MHA默认请求attention weights，不能直接按FlashAttention最低占用估。

权重不因T_hist乘4，但前24层视觉计算／激活会增加。768视觉token只控制VLM端增长，不免除视频编码。约4.09M adapter／投影按每参数12–16B保守计权重／梯度／Adam等约47–62MiB；主要不确定项为激活、logits和临时工作区。

第24层才裁帧并非大幅减少视觉前端工作：仅按空间block等价计，四帧约24×4＋3＝99个单帧层，相对单帧27层约3.67倍，另有时间attention开销；它主要避免VLM接收四倍视觉token。后续可测更早裁帧的性能／吞吐折中，不能从token数不变推断总推理速度不变。

### 6.3 分阶段预留

下表为**工程预算带，非实测／精确峰值预测**。前提：BF16、microbatch1、激活checkpoint、CE分块、各loss独立顺序反传；SFT与仿真／value串行，不驻留第二个教师。

| 阶段 | 整卡预留 | 判断 |
| --- | ---: | --- |
| A：单帧高层LoRA | 10–16GiB | 最稳起点，仍需反向量测 |
| B：四帧，视觉＋VLM LoRA | 14–24GiB | 单5090首选 |
| B：六帧，同样LoRA | 16–28GiB | 四帧过关后考虑 |
| C：四帧＋专家LoRA、顺序反传 | 18–28GiB | 验证optimizer创建／图释放／保存峰值，不保证必装下 |
| 视频＋约430M专家全量联合更新 | 暂无可靠整程预算 | 单expert额外梯度／FP32 moments／可选master约4.0–5.6GiB，首轮不推荐 |

区间不能套给现有同时构造三loss图的实现，也不能直接相加独立进程整卡读数。累积提高有效batch，不降低单microbatch峰值。

未来先争取整卡峰值约26–28GiB、余量3–5GiB；记录allocated／reserved／整卡峰值、CPU available和swap趋势，覆盖首次optimizer创建、反向、保存、重载，不只看forward。主存加载曾达24.18GiB，先单进程／少预取／按需解码／分阶段启停，改进低副本加载；不能默认CPU offload可用。64GiB RAM值得后续扩展，不作为本轮采购前提。

结论：**本机适合尝试四帧MEM＋LoRA分阶段小规模SFT；尚未确认视频训练容量，不推荐首轮全量微调或仿真／value／训练同时常驻。**

## 7. 待实施代码计划

下列全部待实施。本轮未修改工程。

| 顺序／位置（R下相对路径） | 改动与完成条件 |
| --- | --- |
| 配置／严格loader | 新MEM配置、T_hist／stride／drop层／参数阶段白名单；原base严格加载，新projection仅白名单初始化，保存全部新模块与版本 |
| rlinf/workers/rollout/hf/hierarchical_recap_worker.py | episode/env独立ring，每次真实step后append；高层三角色共用快照；reset清空 |
| rlinf/models/embodiment/openpi/modules/model.py及预处理 | optional history／mask／state／physics time，逐帧原变换，旧4D路径保留；dtype/device一致 |
| rlinf/models/embodiment/openpi/modules/siglip.py | video forward、共享时空attention／固定时间编码、压缩／mask、packed视觉LoRA、legacy旁路 |
| rlinf/models/embodiment/openpi/pi0.py | 视频prefix和历史状态token；high／FAST可微到视觉，FM仍no_grad prefix＋detach；相机顺序不变 |
| rlinf/models/embodiment/openpi/recap_lora.py | 视觉／语言index0／专家index1分别选择；A/B完全冻结expert；安全adapter-off教师上下文 |
| data schema／dataset／hierarchical_collator.py | 高层独立数据、history_refs／证据／分支质量；实际动作mask；FAST只用真实完整H16 |
| openpi/tasks/recap_hier.py及hierarchical_recap_runner.py | 启用分支才构图，顺序backward／统一step；旧技能sampler和教师KV；逐模块梯度／参数日志 |
| rlinf/data/storage/episode_memory.py及校验器 | 先READ才能操作；revision覆盖；source／privileged过滤；运行与标注一致 |
| 资格校验／checkpoint／评测 | heldout seed／split／diagnostic／excluded／派生族统一拒绝，旧入口保留，新模式独立评测，保存／恢复后复验 |

另需在T的新任务spec／monitor入口实现上述goal与期望顺序的一致性协议；先保留现有固定顺序任务为基线，新的指令变化实验独立配置。

当前参数helper默认开放完整expert，collator和joint loss系数0仍计算三支。只改YAML不能实现上述冻结或预算。当前loader只允许adapter缺键，新增非adapter投影需严格白名单，不能用全局strict=False掩盖漏载。

现200文本token预算须实测总目标／目录／记忆／状态／子任务长度，各回答及EOS首轮尽量≤96token；不截断正确标签。确需加长时显式版本化训练／推理配置并复算显存，不能悄悄改变比较组。

## 8. 验收和对照

### 8.1 代码接入后、训练前

- 多episode／env时间和状态对齐、缺帧mask、无未来信息；SDK24／三RGB／60ms不变。
- 原分支关闭adapter与原SFT多样本K/V／动作对照，严格保存／重载。
- 各分支单独反向审计：high／FAST可达B视觉／状态／VLM，KI-FM对它们零梯度；expert按阶段开关。LoRA零初始化时部分A矩阵首步零梯度属于结构现象，不要求每个张量首步都非零。
- 基权重不变和训练名称白名单；未来可做无optimizer提交的反传探针，本轮没有执行。

### 8.2 SFT验收

1. **协议**：EOS／JSON完整率先以≥99%为项目目标，工具合法、READ／WRITE各类准确分别看，不能全空操作刷分。
2. **绑定**：对象／初始角色／顺序／目标关系先以≥95%为项目目标；同画面换要求、同要求换有效历史分别测，按语义而非逐字匹配。
3. **历史利用**：真实历史、仅当前、重复当前、打乱历史、状态错位。记忆必要子集应受益；打乱引入分布变化，不能仅凭变差断言学成。
4. **动作**：旧技能单帧与新四帧分别测；固定噪声漂移与同seed配对SR。H16/K_exec1、物理时限、早停及成功标准一致。
5. **完整自主链**：真实VLM生成三角色→工具→专家→环境，不用参考文本／脚本替代；任务SR、错确认、重复尝试、角色错误及掉落并列。
6. **容量／复现**：反向／optimizer／保存／重载峰值和数据／模型版本。组件通过、资源通过、自主成功、泛化四项分开。

主对照：单帧＋文本记忆；视频无文本bank；视频＋文本记忆；同数据同更新预算去掉保留项。参考高层只作故障隔离，不作自主效果。

MEM视频结构、KI／FAST／FM及已有记忆工具的拼接本身不是已成立算法创新；后续仍需在强SFT和同预算对照上检验条件利用、角色保留及物理后果信用的具体贡献。

## 9. 来源与本轮产物

- 论文：[MEM v1](https://arxiv.org/html/2603.03596v1)，§III-C/D及附录C；[官方说明](https://www.pi.website/research/memory)。作者机制和本地工程选择已分别标识。
- 源码静态对象：SigLIP、pi0 prefix／KI分支、recap_lora、joint loss、prepare_actor_samples、collator、worker及memory；根路径见第1节，核查日期2026-10-18。
- 既有实测来源：[[2026-10-18-RM65B-RECAP资产准备与真实功能验证]]及[[本机-Linux-RTX5090]]；本轮未重测硬件／模型。
- 本轮仅新增本方案并补旧计划／主题／索引入口；原单帧方案、原入口及历史失败保留，未改工程或训练，未产生新SR。
