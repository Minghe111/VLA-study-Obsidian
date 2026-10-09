---
title: "GR00T N1.7精读：跨本体动作与人类视频预训练"
date: 2026-10-03
updated: 2026-10-03
tags:
  - VLA
  - 论文精读
  - GR00T
  - 基座选型
  - 跨本体
  - 灵巧手
aliases:
  - GR00T N1.7详细笔记
  - GROOT N1.7精读
status: 文献与官方实现核查完成-本机训练待验证
paper_type: methods
locator_mode: page-grounded
code_commit: 51d4c89f72fda44cbf77285c6a8114b52676b8a1
model_revision: 2fc962b973bccdd5d8ce4f67cc63b264d6886495
---
 
# GR00T N1.7 精读：跨本体动作与人类视频预训练

> Source coverage: Full paper（GR00T N1 v2，共 36 页）；N1.7 官方代码、文档和模型卡；EgoScale 方法与实验重点核查
> Extraction confidence: High；已视觉核查架构、公式、结果表与消融图
> Locator mode: page-grounded（以下未另行说明的 PDF 页码属于 GR00T N1 v2）
> Primary analytical lens: methods
> Secondary analytical lens: None
> Context verification: Targeted external check（官方版本、EgoScale、部署与自定义本体）
> Card completeness: Complete relative to supplied source；N1.7 缺少独立官方论文的部分明确标为文档证据

> [!important] 首先划清版本边界
> 截至 2026-10-03，当前 NVIDIA 官方仓库的 Paper / Citation 和 N1.7 模型卡都仍指向 **GR00T N1（arXiv:2503.14734）**。本次未找到 NVIDIA 发布的独立 N1.7 完整原论文。因此本笔记精读 N1 的继承架构与完整实验，再通过固定代码、发布权重配置及 EgoScale 原论文解释 N1.7。**N1 的成功率、参数量、训练预算不能自动改名成 N1.7。** 检索中出现的 *One-Step Drifting Action Heads for GR00T N1.7* 是第三方修改报告，不是 NVIDIA N1.7 原论文，不纳入官方结果。

证据标签：`[Paper]` 为 N1 论文直接报告；`[External]` 为 N1.7/N1.6 官方材料或 EgoScale 原文；`[Analysis]` 为本次分析；`[Hypothesis]` 为待实验研究候选；`[User]` 为项目已确认的信息。没有本机训练、真机闭环或 3090 时延实测。

## 01 基本信息

| 项目 | 核查结果 |
| --- | --- |
| 继承架构原论文 | *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*，NVIDIA；项目研究负责人 Linxi “Jim” Fan、Yuke Zhu；完整贡献者见 Appendix A |
| 论文版本 | arXiv:2503.14734v2；v1 2025-03-18，v2 2025-03-27；读取日期 2026-10-03；未以同行评审录用状态作为前提 |
| DOI | 10.48550/arXiv.2503.14734（arXiv DOI） |
| 当前要选择的基座 | `nvidia/GR00T-N1.7-3B`，官方约 3B 参数；不是原论文的 `GR00T-N1-2B` |
| 代码固定点 | NVIDIA/Isaac-GR00T main，`51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| 权重固定点 | Hugging Face `2fc962b973bccdd5d8ce4f67cc63b264d6886495`；API lastModified 为 2026-04-23；仓库 createdAt 不是模型公开发布日期 |
| 公开时间与状态 | NVIDIA 官方组织 HF 文章于 2026-04-17 宣布 Early Access；此次固定 GitHub README 已称 GA；模型卡仍残留 EA，不能由此确定确切 GA 发布日 |
| 专门支持人类视频迁移的研究 | *EgoScale: Scaling Dexterous Manipulation with Diverse Egocentric Human Data*，arXiv:2602.16710v1；2026-02-18 提交，项目页日期 2026-02-19 |
| 本地原文 | [[GR00T-N1.pdf]]；EgoScale PDF 与来源快照存放于工程 `research/2026-10-03-vla-foundation-reading/groot-n17` |
| 本项目定位 | 双臂灵巧手、跨本体动作和人类动作先验的对照候选；RM65-B 专用权重、真机适配与成功率均未验证 |

### 来源账本

- **P1**：[N1 v2 原论文](https://arxiv.org/abs/2503.14734v2)。下文 `[Paper: PDF p. N, ...]` 都指这一版本。
- **E1**：[EgoScale 原论文](https://arxiv.org/abs/2602.16710v1)；[NVIDIA 项目页](https://research.nvidia.com/labs/gear/egoscale/)。涉及 EgoScale 页码时显式写出论文名。
- **C1**：[固定 README](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/README.md)。
- **C2**：[固定模型配置](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/configs/model/gr00t_n1d7.py)；[动作头实现](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)；[Qwen3 主干](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/model/modules/qwen3_backbone.py)；[DiT 实现](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/model/modules/dit.py)。
- **C3**：[NEW_EMBODIMENT 指南](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/finetune_new_embodiment.md)；[模态配置与相对动作](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/data_config.md)。
- **C4**：[硬件建议](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/hardware_recommendation.md)；[微调配置](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/configs/finetune_config.py)；[微调入口](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/experiment/launch_finetune.py)。
- **C5**：[N1.7 LIBERO 流程和结果](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/LIBERO/README.md)；[SimplerEnv 结果与 N1.6 对照](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/SimplerEnv/README.md)。
- **M1**：[N1.7 模型卡](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/README.md)；[实际发布 checkpoint 配置](https://huggingface.co/nvidia/GR00T-N1.7-3B/blob/2fc962b973bccdd5d8ce4f67cc63b264d6886495/config.json)。
- **V1**：[N1.6 官方技术页](https://research.nvidia.com/labs/gear/gr00t-n1_6/)；**V2**：[NVIDIA 官方组织 N1.7 发布文章](https://huggingface.co/blog/nvidia/gr00t-n1-7)。

### 术语锁定

| 统一术语 | 本笔记含义 | 容易混淆的地方 |
| --- | --- | --- |
| GR00T N1 / N1.6 / N1.7 | 明确的代际版本 | 用户习惯写 GROOT；正式模型名有两个数字 0 |
| System 2 | 视觉语言主干输出条件表征 | 不自动等于逐步生成可观测思维链 |
| System 1 | flow-matching DiT 连续动作专家 | 文档称 diffusion，但主要训练目标是向量场匹配 |
| relative EEF | 相对当前腕/末端姿态的动作块 | 不是任意关节差分，也不是块内逐帧累加增量 |
| action horizon | 模型预测动作数 | 与 execution horizon、去噪步数、控制频率不同 |
| mid-training | 对齐人类与机器人感知/控制的中间训练 | 不等于只有一条目标任务机器人示范的 post-training |

## 02 一句话总结

[Analysis] GR00T 路线用视觉语言条件表征、共享 flow-matching 动作专家和本体专用接口吸收异构动作监督；N1.7 将主干切到 Cosmos-Reason2-2B，并强调 relative EEF 与 20K 小时 EgoScale 人类动作数据，从而成为值得测试的双臂灵巧手基座，但现有证据没有证明它能直接控制 RM65-B，也没有证明 4×3090 能按默认配置训练。

## 03 研究问题

[Paper] 核心困难是机器人数据分散在不同本体、传感器与控制接口上：多收机器人数据仍是多个“数据岛”，而大量人类视频没有真实机器人的动作标签。模型既要理解任务，又要输出可执行、连续且与硬件对应的动作；单纯将 VLM 的文本答案交给控制器不能解决这种对齐。[Paper: PDF p. 1–3, Introduction and Section 2]

可精确表述为：**能否让一个共享 VLA，在动作维度和本体差异很大的训练源之间学习可迁移的操作表征，并只用少量目标机器人数据适配？** 原 N1 的实验主要测试短程桌面操作和数据效率；N1.7/EgoScale 增加了一个更针对灵巧手的问题：**相对腕运动与细粒度手动作监督，能否把人类操作视频中的先验迁移到真实多指机器人？** [External: E1, EgoScale PDF p. 3–5, Sections 2.1–2.5]

[Analysis] 对 RM65-B，重点不是“3B 是否足够大”，而是数据中可学到的相对运动、手部协调和语言信息，经过本体接口后是否仍有有效监督。24 维接口能接通只证明张量映射成立；先验能否转移要通过任务与闭环证据回答。

## 04 背景与发展路线

[Paper-framed; external verification not performed] N1 原论文将已有路线区分为高层基础模型调用低层技能，以及机器人数据端到端微调 VLA；其选择是保留预训练视觉语言模块，用简单 cross-attention 接到可连续生成动作的专家，而不要求两者采用同一 Transformer 架构。这里是原论文的背景叙述，不将作者的优先性主张当作独立查新。[Paper: PDF p. 17–18, Related Work]

[External] 对本次需要选择的版本，官方演进可以明确区分：

| 版本 | 已核查的主要变化 | 证据边界 |
| --- | --- | --- |
| N1（2025） | Eagle-2 + DiT，16 步动作块；人类视频以潜动作标签参与数据金字塔 | 本笔记完整原论文的模型和实验主体 |
| N1.6（官方技术页日期 2025-12-15） | 更大 DiT（官方称 32 层，相对 N1.5 的 16 层）；灵活图像分辨率；大多数本体使用 state-relative 动作；增加 YAM、Genie1、G1 等遥操作数据 | 是 N1.6 的技术报告，不能当 N1.7 独立消融 |
| N1.7（2026） | Cosmos-Reason2-2B / Qwen3-VL；relative EEF 与人类机器人动作一致性；20K 小时 EgoScale 数据；发布接口 132 维上限、40 步动作块；扩展部署与多数据集工作流 | 当前固定代码、权重与文档；精确默认值和发布权重并不完全相同 |

来源：V1、C1、C2、M1。N1.7 的优势方向是动作空间和数据来源的对齐；“更好推理”“更强语言跟随”仍需用它自己的评测和受控对照检验。

## 05 原论文识别的核心痛点

| 痛点 | 表现 | 作者解释与应对 | 证据 |
| --- | --- | --- | --- |
| 动作维度异构 | 单臂、双臂、颈腰、多指手的 state/action 不能直接共用输出维度 | 每个本体使用专属编码器和解码器，共享 DiT | [Paper: PDF p. 3–5, Section 2.1] |
| 机器人示范昂贵 | 灵巧双臂遥操作组织、标注和扩充成本高 | 真实、物理仿真、神经生成、人类视频共同训练 | [Paper: PDF p. 5–10, Sections 2.2–3.2] |
| 视频没有机器人 action | 网络人类视频和生成视频只有观察变化 | VQ-VAE 潜动作，或用真实数据训练 IDM 预测伪动作 | [Paper: PDF p. 5 and 24, Latent Actions and Appendix F] |
| 多任务需要语言落到动作 | 干扰物、多个目标容器及左右手协调增加选择困难 | VLM 条件化、细粒度标注和目标定位辅助损失 | [Paper: PDF p. 9–10 and 24, dataset and auxiliary loss] |
| 有限微调容易丢泛化 | 单手微调后，原来会的左右手交接消失 | 原文直接展示能力退化，未宣称微调一定保留所有技能 | [Paper: PDF p. 16, Section 4.5] |

[Analysis] N1.7 的 relative EEF 和 EgoScale 针对的是第三行的“监督能否共享”，不是仅增加视频数量。手部关节重定向仍是本体相关的；相同腕运动也不消除不同臂长、可达范围和接触力学差异。

## 06 核心思想

**表面设计**：VLM 处理 RGB 与语言；DiT 处理状态与加噪动作块；本体 MLP 在输入输出处转换维度。动作专家通过 cross-attention 读取 VLM token，不先强制把动作离散成自然语言 token。[Paper: PDF p. 3–5, Figures 2–3]

**更深的思想**：共享的是动作相关条件表示和生成结构，保留的是本体接口。潜动作使没有机器人标签的视频成为一种可训练的“本体”；N1.7 的人类相对腕运动进一步把抽象动作监督转为与机器人可比较的几何动作。[Paper: PDF p. 5–6, Figure 4] [External: C1; E1, EgoScale PDF p. 3–5]

**[Analysis] 可迁移的经验**：跨本体学习不能只靠 padding。需要共享部分具有相同物理意义，专属部分又能表达真实控制接口。对于双手，腕姿态较容易共享；每手 6 主动控制通道与 Sharpa 22DoF 的关节目标差异必须留在数据和适配器中，不能把“六个控制量”当作“六个独立人类关节”。

## 07 方法总览

### 7.1 N1 的继承数据流

```text
当前多视角 RGB + 任务语言
  -> Eagle-2 视觉语言主干 -> 视觉语言特征 φt
当前本体 state qt -> 本体 State Encoder -> 状态 token
加噪未来动作块 Aτt + 扩散时间 τ -> 本体 Action Encoder -> 动作 token
  -> DiT：状态/动作 self-attention，读取 φt 的 cross-attention
  -> 本体 Action Decoder -> 向量场 -> 4 次积分 -> 连续动作块
  -> 执行部分动作 -> 获取新观察 -> 重规划
```

[Paper] N1 公开模型总参数 2.2B，其中 VLM 1.34B。图像为 224×224，通过 pixel shuffle 后每帧 64 个视觉 token；使用 Eagle-2 第 12 层特征；默认预测 $H=16$ 个动作，推理 $K=4$ 次去噪。论文报告 L40、bf16 下一个 16 步 chunk 为 63.9ms。这些是 N1 配置，不是 N1.7 的部署时延。[Paper: PDF p. 3–5, Section 2.1]

### 7.2 N1.7 当前实现路径

[External: C2] `Qwen3Backbone` 加载 Cosmos-Reason2-2B，保留选定深度之前的 language layers，输出条件 token、attention mask 和 image mask。动作头先对条件特征做可配置 LayerNorm / self-attention；`AlternateVLDiT` 在 self-attention 与 cross-attention 间交替，并可按 block 选择图像或非图像 token。状态和动作仍由 embodiment-conditioned MLP 编码，最终动作输出使用同一 embodiment id 的解码器。

[Analysis] 因而 System 2 在实际策略路径里首先是“提供视觉语言 latent 条件”的模块。本次检查的策略 forward 没有必须先生成一段文本规划再执行的阶段；名称里的 reasoning 不应被写成已观测到的可解释思维链、任务分解正确率或语言约束保证。

### 7.3 发布权重和代码默认必须分别记录

| 配置项 | 固定代码 `Gr00tN1d7Config` 默认 | 官方发布 `GR00T-N1.7-3B` config.json |
| --- | --- | --- |
| 主干 | Cosmos-Reason2-2B | Cosmos-Reason2-2B |
| max_state_dim / max_action_dim | 132 / 132 | 132 / 132 |
| action_horizon | 40 | 40 |
| DiT num_layers | 16 | 32 |
| select_layer | 12 | 16 |
| load_bf16 | false | true |
| state_dropout_prob | 0.8 | 0.2 |
| tune_llm / tune_visual | false / false | true / true（checkpoint 保存字段，不能代表下一次微调实际开关） |
| 默认推理迭代 | 4 | 4 |

[External: C2, C4, M1] 微调 CLI 另有自身默认值，例如 `state_dropout_prob=0.2`，入口会覆盖部分模型参数并强制 `load_bf16=False` 等。因此 README 的“默认从 32 改 16 层”不等于已下载权重只有 16 层。**实际复现实验必须导出加载后的 config 和 requires_grad 参数，而不是仅抄 README。** `132` 是 padding/接口容量，不是 RM65-B 的真实关节数；40 步也不是必须开环执行 40 步。

## 08 核心模块拆解

| 模块 | 功能与存在理由 | 输入输出 | 直接支持及移除边界 |
| --- | --- | --- | --- |
| VLM / System 2 | 继承视觉语言语义，将目标、物体和空间关系转为动作条件 | 图像/指令 → token 特征 | N1 Figure 3 与任务结果；未提供同数据同算力下独立移除 VLM 的完整因果消融 [Paper: PDF p. 4 and 15, Figure 3 and Tables 2–3] |
| State Encoder | 将不同 proprioception 投到相同宽度 | 本体 state + id → 状态 token | N1 架构；N1.7 padding 上限 132。尺寸兼容不证明新本体可零样本控制 [Paper: PDF p. 3–5, Section 2.1] [External: C2] |
| Action Encoder / Decoder | 可表达维度、单位、运动接口不同的动作 | 动作块与 τ → latent；latent → 本体向量场 | 架构明确；没有隔离证明“本体专属 MLP”一定优于所有统一接口 [Paper: PDF p. 4–5, Figure 3] |
| DiT / System 1 | 联合建模未来动作的连续分布和跨时间协调 | 当前状态、噪声块、VLM 条件 → 向量场 | N1 4 步采样；N1.7 当前 head loss 与 inference 源码 [Paper: PDF p. 5, Eq. 1] [External: C2] |
| LAPA 潜动作 | 从前后图像提取行为变化，使无动作视频可监督 | $(x_t,x_{t+H})$ → 连续 pre-quantized latent | Figure 4 是相似检索示例，不是所有潜动作具有一致物理语义的证明 [Paper: PDF p. 5–6, Figure 4] |
| IDM 伪动作 | 把生成视频映到更接近真实控制的动作标签 | 当前/未来图像 → 中间 action chunk | Figure 9 在较高数据下 IDM 优于 LAPA，训练预算 30K/60K [Paper: PDF p. 16 and 24, Figure 9 and Appendix F] |
| DexMimicGen | 分解对象中心子任务并变换到新物体姿态 | 源示范、对象姿态、仿真 → 成功轨迹 | 保留成功轨迹，节省采集；不自动保证 sim-to-real [Paper: PDF p. 6–7 and 10, Simulation Trajectories] |
| N1.7 relative EEF | 统一人类腕运动和机器人末端运动 | 绝对姿态 → 相对当前姿态的未来块 | 官方 README 与 EgoScale 几何表示；无 N1.7 全部更新逐项隔离消融 [External: C1; E1, EgoScale PDF p. 3–5] |
| RTC 衔接 | 用旧动作块的重叠段初始化，冻结已承诺段、渐进调整剩余段 | 旧块、延迟/重叠参数 → 新块 | N1.7 head 源码存在该路径；本项目未运行，其顺滑提升不能当作当前真机事实 [External: C2, get_action_with_features] |

### 8.0 N1 的实际训练范围与数据生成成本

[Paper] 原 N1 的 pre-training 与 post-training 均冻结视觉语言主干的语言组件，更新视觉编码器、动作接口和 DiT；“端到端联合训练”因此不等于全部 VLM 参数都被更新。Table 6 给出 AdamW、LR1e-4、beta1=0.95/beta2=0.999、weight decay1e-5、cosine schedule、warmup5%；预训练200K steps/global batch16384，post-training20K–60K/global batch128或1024。神经视频增强在 post-training 中按真实/生成轨迹1:1采样。[Paper: PDF p. 8, Section 2.3; PDF p. 27, Table 6]

[Paper] N1预训练本身约50,000 H100 GPU hours；神经视频生成另用约105,000 L40 GPU hours。780K仿真轨迹的“11小时”是特定自动生成系统的墙钟时间，不能省略资源背景后推算本项目3090的训练/采集时长。作者在单A6000测试仅adapter+DiT和解冻vision两种微调，仍是N1的资源记录，不是N1.7的24GB保证。[Paper: PDF p. 7–8 and 10, data generation and training infrastructure]
### 8.1 人类视频并不是天然动作标签

[Paper] N1 的七个人类视频源使用 VQ-VAE 前后帧重建训练，再取量化前连续 latent；它们作为 LAPA 本体使用同类 flow-matching 目标。机器人数据可同时提供真实动作和潜动作；神经生成视频可使用潜动作和 IDM 标签。推理不需要未来真实图像，未来帧只用于训练标签构造。[Paper: PDF p. 5 and 8, Latent Actions and Pre-training]

[External: E1, EgoScale PDF p. 3–5] EgoScale 的路线不同：由 SLAM 和手姿态估计提取 world wrist motion，将 21 个手关键点约束重定向为 Sharpa 22DoF 手动作；人类数据缺少机器人 proprioception 时用可学习 placeholder。共享的是相对腕运动预测、VLM 和 DiT；手动作与 proprioception 接口仍保留本体适配器。

### 8.2 EgoScale 的三阶段训练及与 N1.7 的联系

| 阶段 | 数据、预算与训练范围（EgoScale 原文） | 能回答什么 |
| --- | --- | --- |
| Human pretraining | 20,854 小时；包括大规模野外人类视频和 829 小时 EgoDex；100K steps，256×GB200，global batch 8192，LR 5e-5，VLA 参数全部解冻 | 大规模动作监督可形成灵巧操作先验 |
| Aligned mid-training | 344 个任务，约 50 小时人类 + 4 小时机器人；匹配头/双腕视角、相机与场景；50K steps，batch 2048，LR 3e-5；语言主干冻结，视觉编码器和 DiT 更新 | 小规模精确对齐能把先验落到目标传感和控制域 |
| Post-training | 具体机器人任务示范；10K steps，batch 512，LR 3e-5；有 mid-training 时冻结视觉编码器，否则解冻以适配 | 目标任务的微调效果与 few-shot 迁移 |

[Analysis] EgoScale 原文 Stage II 使用“冻结 vision-language backbone、更新 vision encoder 和 DiT”的表述，冻结范围在措辞上并不完全清晰；本表按语言主干/视觉编码器分开转述，复现仍应核对实际 requires_grad，而不能将整套 VLM 冻结当作已验证配置。

[External: E1, EgoScale PDF p. 4–5, Sections 2.2–2.4] 这张表描述 **EgoScale 研究模型**，不能将其训练次数和显卡预算不加说明地当作 N1.7 发布权重的完整训练日志。C1 和 V2 确认 N1.7 纳入 20K 小时 EgoScale 数据，但此次没有得到发布权重全部混合比例、全部预训练阶段与完整复现配方。

## 09 关键公式与符号

### 9.1 N1 的动作块和条件 flow matching

$$
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}],\qquad A_t^{\tau}=(1-\tau)\epsilon+\tau A_t,\quad \epsilon\sim\mathcal N(0,I).
$$

[Paper] $t$ 是真实机器人时间，$\tau$ 是生成流时间，$H$ 是动作块长度；两种时间不能混为机器人执行时间。原论文 Eq. 1 印为：

$$
\mathcal L_{\mathrm{fm}}=\mathbb E_\tau\|V_\theta(\varphi_t,A_t^\tau,q_t)-(\epsilon-A_t)\|^2.
$$

同页随后写从噪声出发的正向 Euler 更新 $A_t^{\tau+1/K}=A_t^\tau+V_\theta/K$。[Paper: PDF p. 5, Equation 1 and inference update]

> [!warning] 原文符号与当前代码不一致
> 已视觉确认，N1 v2 纸面插值的导数是 $A_t-\epsilon$，却把 loss target 写为 $\epsilon-A_t$，并保持正号 Euler 更新；这组纸面表达存在方向不一致，不能原封不动据此实现。N1.7 固定 head 源码实际采用 `velocity = actions - noise`，插值从 noise 到 actions，推理使用 `actions += dt * pred_velocity`，三者一致。本笔记保留原文问题并以当前源码说明实现，不擅自改写原论文。[External: C2, forward / get_action_with_features]

[External: C2] N1.7 在实际维度 mask 上计算 MSE：

$$
\mathcal L=\frac{\sum_{b,h,d}m_{bhd}\bigl(v_{\theta,bhd}-(A-\epsilon)_{bhd}\bigr)^2}{\sum_{b,h,d}m_{bhd}+10^{-6}}.
$$

[Analysis] mask 避免为了兼容最大 132 维而学习无效 padding；它不自动均衡“两个机械臂”和“两个手”的重要性。不同单位、归一化与有效维度数量会改变梯度贡献，因此必须检查各模态误差而非只有 total loss。

### 9.2 N1 辅助目标定位损失

$$
L_{\mathrm{det}}=\|x_{\mathrm{pred}}-x_{\mathrm{gt}}\|^2,\qquad L=L_{\mathrm{fm}}+L_{\mathrm{det}}.
$$

[Paper] OWL-v2 给出指令目标物框，中心坐标按图像宽高归一化，VLM 特征上加线性层预测二维中心。该监督作用于训练，不能说测试时提供真实目标框。[Paper: PDF p. 24, Appendix F, Auxiliary Object Detection Loss] [Analysis] N1.7 当前 head 源码报告的是 mask 后 action loss；没有核实到它复现全部 N1 辅助监督，不能把祖先训练项自动写成 N1.7 微调必备 loss。

### 9.3 相对腕姿态

[External: E1, EgoScale PDF p. 3, Section 2.1] 人类腕在世界坐标的位姿由相机运动与相机下腕姿态组合得到；chunk 内相对变换为：

$$
W_w^t=T_{w\leftarrow c}^tH_{c,1}^t,\qquad \Delta W^t=(W_w^0)^{-1}W_w^t.
$$

$W_w^0$ 是动作块起点腕姿态。它去除全局位置/相机运动的影响，保留该时刻之后的局部运动；不能将每个未来目标相对不同前一帧做差，再按“相对当前状态”解释。旋转是 SE(3) 组合，不是任意欧拉角逐元素相减。

## 10 实验设计与证据链

### 10.1 N1 完整原论文的评测协议

[Paper] 仿真使用 RoboCasa 24 个任务、DexMimicGen 9 个任务/3 类双臂本体、GR-1 tabletop 24 个任务；每任务分别用 30/100/300 条示范 post-train。RoboCasa 为三 RGB 视角、EEF 动作；GR-1 主要头部单 RGB、关节状态动作。不能把所有原论文结果理解为同一视角/动作接口。[Paper: PDF p. 11–12, Section 4.1]

基线是从头训练的 BC-Transformer 与 Diffusion Policy，而 GR00T 使用大规模预训练。通常 post-training batch 1024、60K steps，DexMG 部分 batch 128。仿真每任务 100 次试验，从最后 5 个 checkpoint 选最高得分，每 500 steps 保存。真机大多每任务 10 次，Pack Machinery 则 5 次、每次 5 个物体/30 秒；使用阶段部分分数。没有提供足以替代多种子置信区间的完整不确定性分析。[Paper: PDF p. 14–15, Section 4.3]

### 10.2 N1 中心主张与对应证据

| 实验 | 具体条件与作者报告结果 | 支持的结论 | 更强解释的边界 | 来源 |
| --- | --- | --- | --- | --- |
| 未 post-train 的 GR-1 两类任务 | 左右手交接场景 76.6%（11.5/15，含 0.5 部分分）；新物体/新容器 73.3%（11/15） | 该预训练模型存在迁移与组合行为 | 不是任意新机器人 zero-shot，也不是纯二元成功率 | [Paper: PDF p. 15, Pre-training Evaluations] |
| 仿真 100 demos/task | 表2 N1：RoboCasa 32.1%、DexMG 66.5%、GR-1 50.0%；DP：25.6%、56.1%、32.7% | 表中协议下 N1 有优势 | 与附录 DexMG 数值冲突，见下；不是 N1.7 结果 | [Paper: PDF p. 15, Table 2] |
| 真机全数据 | GR-1 N1 76.8%，DP 46.4%；按表差值 30.4 个百分点 | 该训练/评分设置下预训练 VLA 优于从头 DP | 不能隔离架构、数据规模与预算各自贡献 | [Paper: PDF p. 15 and 27, Tables 3 and 5] |
| 真机 10% 数据 | N1 42.6%，DP 10.2%；N1 比全数据 DP 46.4% 低 3.8 个百分点 | N1 数据效率较好 | 仍需已有跨本体大规模预训练，不能称总训练数据只有 10% | [Paper: PDF p. 15, Table 3] |
| 加神经生成轨迹 | RoboCasa 30/100/300 体制平均增益 4.2/8.8/6.8 个百分点；GR-1 8 个选定任务约 +5.8 | 合成增强在这些设定中有效 | 生成/过滤/IDM 的额外算力没有消失；不等于全量任务皆改善 | [Paper: PDF p. 16, Figure 9] |
| 单手 post-training 后交接 | 原 pretrain 可完成交接；只用右手数据 post-train 后失败 | 微调可能损伤原有组合技能 | 定性例子不能量化所有忘却程度 | [Paper: PDF p. 16 and 21, Figure 11] |

> [!warning] N1 原论文内部的数字冲突
> 同为 100 demos，表2 的 DexMG 为 DP **56.1%** / N1 **66.5%**；表4和图10为 DP **46.9%** / N1 **58.5%**。已核查渲染页，属于原文差异，不是提取错误。本笔记分别保留，不选择其中一个伪装成唯一精确成绩。GR-1 表4 N1 从 100 demos 的 50.0% 到 300 demos 的 49.3%，也不支持“每个 benchmark 都随数据单调提升”的强表述。[Paper: PDF p. 15, Table 2; PDF p. 20, Figure 10; PDF p. 26, Table 4]

### 10.3 全部主图表和附录证据库存

| 图表 | 在论文论证中的角色与读取重点 | 定位 |
| --- | --- | --- |
| Figure 1 | 数据金字塔：数据量与本体特异性不同，示意结构而非独立实验结果 | [Paper: PDF p. 2, Figure 1] |
| Figure 2 | System 2 到 System 1 的总体输入输出 | [Paper: PDF p. 3, Figure 2] |
| Figure 3 | 状态/动作专属 MLP、cross/self-attention 与迭代采样细节 | [Paper: PDF p. 4, Figure 3] |
| Figure 4 | 跨八种本体的相似潜动作检索，两类左右运动的定性例子 | [Paper: PDF p. 6, Figure 4] |
| Figure 5 | 更换语言、物体、目标位置的神经视频，展示生成多样性 | [Paper: PDF p. 7, Figure 5] |
| Figure 6 | 遥操作手腕/手指捕获与重定向，说明动作标签的来源 | [Paper: PDF p. 10, Figure 6] |
| Figure 7 | 三类仿真任务的样例；图像好看不等于策略成功率 | [Paper: PDF p. 11, Figure 7] |
| Figure 8 | 预训练泛化场景及真机四类任务覆盖 | [Paper: PDF p. 13, Figure 8] |
| Figure 9 | LAPA/IDM 合成增强消融，低/中/高数据条件区分 | [Paper: PDF p. 16, Figure 9] |
| Figure 10 | 三种 demos 体制的成功率，DexMG 与表2矛盾 | [Paper: PDF p. 20, Figure 10] |
| Figure 11、Figure 12 | 交接泛化、微调实例与 DP 失败样例，定性证据 | [Paper: PDF p. 21, Figures 11–12] |
| Figure 13 | 多视角 grid、续接长视频、液体与关节物体生成；作者未完成所有下游量化 | [Paper: PDF p. 22, Figure 13] |
| Figure 14 | 七个人类视频源样例，解释场景与语言覆盖 | [Paper: PDF p. 25, Figure 14] |
| Table 1 | 哪些源可用 latent、neural、simulation 三种生成方式 | [Paper: PDF p. 9, Table 1] |
| Tables 2–3 | 仿真和真机汇总，注意基线、部分评分与来源版本 | [Paper: PDF p. 15, Tables 2–3] |
| Table 4、Table 5 | 每任务结果，避免只看均值；100/300 demos 非单调和物体泛化差异 | [Paper: PDF p. 26–27, Tables 4–5] |
| Table 6 | AdamW、1e-4 LR、预训练 batch16384/200K steps、post batch128或1024/20K–60K；不是 N1.7 官方复现超参 | [Paper: PDF p. 27, Table 6] |
| Table 7 | N1 预训练 592.9M 帧、8375.7 小时：robot3288.8、human2517.0、simulation1742.6、neural827.3 | [Paper: PDF p. 28, Table 7] |

[Analysis] Table 7 的 simulation1742.6 小时是预训练子集；正文“780K 轨迹/6500 小时”同时计入 pre/post 数据生成，不能二选一或者把总数又加到 Table 7。[Paper: PDF p. 7 and 28, Simulation Trajectories and Table 7]

### 10.4 N1.7 当前官方 benchmark 证据

[External: C5] LIBERO 四个 suite 的公开命令分别训练，各自 `max_steps=20000`、global batch640、8 GPU、state dropout0.2。它们不是基座原样 zero-shot 成绩。官方表存在计数与百分比不一致：

| Suite | 官方写出的计数 | 官方写出的百分比 | 按计数计算 |
| --- | --- | --- | --- |
| Spatial | 195/200 | 97.65% | 97.5% |
| Goal | 195/200 | 97.5% | 97.5% |
| Object | 197/200 | 98.45% | 98.5% |
| 10 Long | 189/200 | 94.35% | 94.5% |

不把这些微小差异补成新实验结果，也不自创平均排名。评测示例 `n_action_steps=8`、环境实例数5、单任务 `n_episodes=10`；示例命令本身不等于汇总表所有 rollout 的完整日志，精确评测次数/种子要在复现时核查。[External: C5, LIBERO Evaluate checkpoint]

[External: C5] SimplerEnv 官方平均值 N1.6→N1.7：Bridge **56.6%→62.3%**，Fractal **52.0%→72.5%**。但任务并非全面进步：WidowX eggplant-in-basket **89%→53%**、eggplant-in-sink **33%→2%**；同时 stack-cube **5%→48%**、close-drawer **73%→97%**。这说明平均提升可能伴随明显的技能分布迁移。公开复现命令 batch1024/8GPU/20K steps，Bridge state dropout0.8、Fractal0.5；post-training 数据和动作 wrapper 都属于评测条件。

### 10.5 EgoScale 给出的机制证据，不能改名为 N1.7 直接成绩

[External: E1, EgoScale PDF p. 6–11, Figures 3–8] 五个高灵巧任务：卷衣、分卡、夹具转移水果、旋瓶盖、注射器转液；前四类多数100条示范，卷衣20条。图4中无预训练平均 binary success 0.02，human pretrain + midtrain0.56，差值 **54 个百分点**；task completion score0.24→0.83 是另一指标。该结果由 Galaxea R1 Pro + 22DoF Sharpa 双手产生，不是 RM65-B 的值。

1K→20K 小时人类数据，post-train 后平均 task completion0.30→0.71，离线验证对2K人类视频 episode、每条随机20时刻、每时刻采16个生成动作并取均值。它支持所测试范围的相关性与可预测趋势，不能证明对本项目任何离线 loss 都能替代真机成功率。[External: E1, EgoScale PDF p. 7–9, Figure 5]

“One-shot” Fold Shirt0.88、Unscrewing0.55 的设定仍每任务/每瓶配100条对齐人类示范，且已有 pretrain 与 midtrain；不等于从零只有一条数据。三指 G1 迁移还加入 G1 play data，并用独立 Homie 策略处理下半身，不能解读为同一 VLA 自动解决全部平衡与运动。[External: E1, EgoScale PDF p. 9–10, Figures 6–7]

## 11 如何正确理解结论

**可以确认**：[Paper] N1 在所选仿真/GR-1真机任务上优于从头训练的常见模仿学习基线，异构预训练提高数据效率。[External] N1.7 已发布权重和自定义本体流程，使用 Cosmos/Qwen3 与扩大的接口；EgoScale 为相对腕运动、手关节动作监督及人机对齐提供原文机制证据。

**仍有边界**：N1 报告短程桌面为主，N1.7 README/模型卡和 N1.6 技术页不能拼成一篇没有发表过的、包含全部消融的“N1.7 论文”。上游训练、基准 post-train、开环拟合、真机闭环是不同层级。原 N1 的分阶段评分、EgoScale 的 completion、N1.7 仿真二元成功率不能直接当同一指标。[Paper: PDF p. 14–17, Sections 4.3–4.6] [External: E1, C5]

### 11.1 资源与部署边界

[External: C4] 当前 N1.7 硬件文档要求推理16GB+，fine-tuning 最低40GB+，默认 projector + diffusion head、冻结 LLM，峰值约35GB/卡；启用视觉或 LLM 微调推荐80GB+。**不能沿用 N1.5 “24GB 可微调”的旧材料替代本版本要求。**

[Analysis] 4×3090 每卡24GB，普通数据并行复制模型/优化器，各卡显存不会变成一块96GB显存。现有四卡可以作为低batch、限定模块、受支持分片/检查点方案的实测平台，但“正式默认微调够用”尚无证据。降低 batch 能减少激活，并不必然解决模型与优化器状态占用；只训练 projector 会改变适配容量，须与默认 head 微调区别报告。此次入口核查没有发现可直接沿用 openpi 的 LoRA 配方，不借第三方 LoRA 补丁声称官方默认已支持。

[External: C4] H100、单相机、4次去噪：eager11.7Hz、TensorRT35.9Hz；L40对应7.8Hz、26.0Hz。这是**动作块重规划频率**，不是伺服频率。三相机、RM65-B数据、3090和网络桥的p50/p95时延都待测，不能拿单视角结果做本项目保证。

### 11.2 官方材料冲突清单

| 冲突 | 本次处理方式 |
| --- | --- |
| HF模型卡旧段落仍称 SigLip2 + T5，另一段已称 Cosmos-Reason2 | 用固定 model_name 与 Qwen3Backbone 确认当前架构；不以遗留段落绘制 N1.7 架构 |
| HF链接文本Cosmos-Reason2-2B，链接地址却到8B | 采用 checkpoint `model_name=nvidia/Cosmos-Reason2-2B`；链接错误保留为文档问题 |
| README称GA，模型卡/4月文章称EA | 保存各材料日期；确切GA日未确认，不将两个时期当同一发布状态 |
| README概要有Apache2.0商业许可表述，License节与模型卡权重为NVIDIA Open Model License | 代码Apache2.0、权重NVIDIA Open Model License分别记录，不能把权重简化为Apache2.0 |
| README代码默认DiT16层/层选择12，发布权重32层/层选择16 | 两列配置同时记录，实际运行导出最终配置 |
| 硬件文档Thor eager8.9Hz/TRT12.4Hz，模型卡6.9Hz/10.7Hz；Orin加速模式和频率也不同 | 不拼接为同一受控测试，采用注明来源/版本的结果并要求本机复测 |
| 模型卡Fractal段落复用了Bridge数据链接与说明 | Fractal训练依据固定官方examples/SimplerEnv的fractal数据命令，数据来源错误不当作真实训练证明 |
| LIBERO计数与百分比、N1 DexMG主表与附录冲突 | 记录双方而不掩盖，精确比较需要原始rollout和checkpoint记录 |

来源：C1、C2、C4、C5、M1、V2；均为2026-10-03核查。

## 12 作者明确承认的局限

此处仅列 **N1 原论文作者明示**，不冒充 N1.7 当前仍未改善的结论。

| 局限 | 表现 | 作者提出的方向 | 来源 |
| --- | --- | --- | --- |
| 时间跨度和场景有限 | 主要短程桌面操作 | 扩展长程loco-manipulation，联动硬件、架构与训练语料 | [Paper: PDF p. 17, Section 4.6 Limitations] |
| VLM能力仍可加强 | 空间推理、语言理解、适配能力有提升空间 | 更强视觉语言主干 | [Paper: PDF p. 17, Section 4.6] |
| 合成数据质量与多样性受限 | 多样反事实与遵守物理规律难同时做到 | 改善合成方法，丰富数据金字塔 | [Paper: PDF p. 17, Section 4.6] |
| 鲁棒和泛化仍需提升 | 现有模型不是一般情况下的万能控制器 | 新架构及预训练策略 | [Paper: PDF p. 17, Section 4.6] |

相关作者观察（不是另造formal limitations）：N1单手微调后丢失交接技能；液体/长视频生成尚留待完整量化。[Paper: PDF p. 8 and 16, Post-training with Neural Trajectories and qualitative results] N1.6作者还指出小数据relative action漂移、微调过拟合、语言跟随和OOD泛化仍困难。[External: V1, Discussion] EgoScale结论讨论继续扩大数据/模型、弱标视频与缩小本体差距，未给出RM65-B具体边界。[External: E1, EgoScale PDF p. 12, Conclusion]

## 13 批判性分析

| [Analysis] 观察 | 替代解释或具体风险 | 如何检验 | 依据 |
| --- | --- | --- | --- |
| 模型胜过DP不等于单个模块贡献被证明 | 预训练规模、架构、参数量和算力同时变化 | 同一checkpoint和数据做模块消融；报告总/可训练参数与GPU hours | [Paper: PDF p. 14–15, Baselines and Tables 2–3] |
| cross-embodiment兼容不等于立即转移 | adapter只是接口，目标本体仍需监督和控制标定 | joint/EEF两表示用同一轨迹、预算、种子比较，而不是重收不同数据 | [External: C3; E1, EgoScale PDF p. 3–5] |
| 20K人类数据利于高DoF手不保证6通道手同等受益 | 可达手形、动力学和接触自由度受硬件限制 | 先测6通道投影误差和可执行抓姿覆盖，再测闭环收益 | [External: E1, EgoScale PDF p. 9–11, Figures 7–8] |
| 离线MSE可降低但闭环仍失败 | 状态捷径、曝光/动作错位、执行延迟或训练分布漂移 | held-out物体/措辞/位置，分模态误差，闭环完成率/约束违背/干预率 | [External: C3, open-loop interpretation; V1, Discussion] |
| N1.7平均改进伴随任务退化 | 数据混合、动作表示或post-train预算改变，而非单纯更强主干 | 在固定任务集逐任务分析，公开同seed与初始化，查看失败轨迹 | [External: C5, SimplerEnv task table] |
| 动作块增到40可能增加长开环风险 | 预测长度与执行长度不分，延迟/接触变化无法及时修正 | 固定H40，对execution horizon与RTC分别测成功、手滑/碰撞、p95与动作跳变 | [External: C2, RTC; C4, inference frequency] |
| 官方配置和成绩材料不完全同步 | 用户可能训练出不同架构却引用同名模型成绩 | 存code commit、HF revision、最终config、数据hash、eval脚本与原始计数 | [External: C1–C5, M1] |

[Analysis] 本次没有开展足以支持新方法查新声明的先验文献检索；下文方向都属于可检验候选。对用户课题最需要先厘清的是动作表示、灵巧手可执行约束和语言监督可辨识性，而不是给已有模型叠加多个无法验证贡献的 loss。

## 14 学到的知识

### Agent-derived knowledge candidates

1. **跨本体共享要分层**：共享腕/末端运动和动作生成器，手部及state保留适配接口；这比把不同维度补零后称“统一动作”更具体。[Paper: PDF p. 3–5, Section 2.1] [External: E1, EgoScale PDF p. 3–5]
2. **无动作视频的价值依赖监督构造**：N1的latent/IDM与EgoScale的几何腕姿态、关节重定向是不同路线，后者的数据规模优势不能自动归因“只增加视觉样本”。[Paper: PDF p. 5 and 24, Latent Actions and IDM] [External: E1]
3. **少量精确对齐数据可成为关键桥梁**：EgoScale把scale与alignment分开；目标传感视角与控制接口有明确对齐成本。[External: E1, EgoScale PDF p. 4–5 and 9]
4. **强基座不等于微调后保持全部能力**：任务专属数据可能压缩原先组合行为，需保留预训练能力评测集。[Paper: PDF p. 16, qualitative findings]
5. **报告需分开成功与过程分数**：N1部分分、EgoScale completion和binary success、开环MSE测不同内容，不能共享一个“成功率”标签。[Paper: PDF p. 14–15] [External: E1, EgoScale PDF p. 7 and Appendix B]
6. **部署配置也是科学证据**：动作预测/执行horizon、相机数量、checkpoint配置与微调覆盖参数能够改变结论；它们应与表格成绩一起存档。[External: C2–C5, M1]

## 15 与现有项目知识的联系

关联 [[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]]、[[VLA基座选择-InternVLA-A1.5、π0.5与GR00T N1.7]]、[[RM65-B灵巧手仿真数据采集-RoboTwin2自动生成路线]]。

### 15.1 RM65-B 的 joint-space 首轮适配

[User] 当前仿真使用双RM65-B，每臂6轴、每手6个主动控制通道，合计24维；双手不是两个binary gripper。**头部D435目前已安装，左右腕相机属于规划，并未安装**，所以三路RGB数据计划和实机当前采集条件分别记录。

[Analysis] 第一轮用关节接口更便于与已有RoboTwin24维记录一致：

```text
state/action order = [left_arm(6), left_hand(6), right_arm(6), right_hand(6)]
video plan = [head, left_wrist, right_wrist]
actual current camera = [head]
embodiment_tag = NEW_EMBODIMENT
```

[External: C3] `ActionType.NON_EEF` 配置确实允许关节/手控制量；SO100官方例子是 arm RELATIVE + gripper ABSOLUTE。它证明不必只能使用EEF，但其“gripper absolute较好”不能直接作为多指手的已测结论。RM65-B可设两臂相对、两手绝对为候选，仍需做相同数据的绝对/相对对照。

需要四个明确模态key和一一对应的ActionConfig；action_configs的顺序必须与modality_keys相同。数据保存绝对state与绝对目标action，由processor做relative转换，避免自己提前做差后又被转换一次。若state/action key名不同，要明确state_key。6通道手的量程、方向、弧度/SDK值、被动关节映射与时间戳要先在转换器验收。[External: C3, ActionConfig and relative-action guidance]

`meta/modality.json` 描述24维的语义切片，`modality-config-path`注册到 `NEW_EMBODIMENT`。不能借用GR1或G1 tag假装已预训练同一机器人。24维会padding/mask到模型上限；重新计算目标数据统计，不复用DROID/GR1统计。调整action delta_indices会改变relative_stats形状，要重新生成统计。SO100教程16-step只是实例配置，不能把它写成N1.7基座horizon必为16。[External: C1, C3]

### 15.2 更接近预训练的 relative EEF 路线

[Analysis] 第二条路线将双臂目标通过FK转为相对腕/EEF位姿，两手仍保持各自6控制通道。官方模态规范的EEF内部用位置3+rotation6D；两腕加两手因此可能是 $2\times(9+6)=30$ 维接口，**物理主动控制仍24维**。若原始旋转为rotvec，需用指定format转换，不能随意填入9维。更高维几何表示与更多机器人自由度不是同一回事。[External: C3, ActionType.EEF / XYZ_ROT6D / XYZ_ROTVEC]

这一路线可能更好继承EgoScale腕运动先验，但增加了坐标系、手腕基准点、TCP、FK/IK可达性和控制分支一致性问题。不能只把原joint数组改名为EEF。joint路线能训练这一事实与“偏离native相对EEF预训练监督”的潜在代价并不矛盾；是否值得转换要用同一组原始演示验证。[External: C1; E1, EgoScale PDF p. 3 and 5]

### 15.3 基座选择中 GR00T 的具体价值

[Analysis] GR00T适合承担“跨本体与人类动作先验”对照，尤其未来加入人类演示或双手细粒度抓取。它不因面向humanoid就自动优于π0.5，也不因参数较小就自动更省本项目训练显存。最先进行的是模型/数据/控制共同接口验收，再比较同预算joint与EEF，而不是用来自不同benchmark的均值直接决定主基座。

## 16 可检验研究候选

### Agent-derived research candidates

以下均为 `[Hypothesis]`，创新状态 **unverified**；没有先验查新完成，不能宣称原创性。每个候选先用小规模可失败的实验决定是否继续。

### A. 相对 EEF 是否比相对关节更有效继承人类先验

- **来源**：N1.7宣称relativeEEF共享人机监督，但自定义本体官方教程仍支持NON_EEF joint。[External: C1, C3; E1, EgoScale PDF p. 3–5]
- **核心假设**：在相同RM65-B演示和微调预算下，relativeEEF腕动作在未见物体位置/相机视角上比relativejoint更有效，且控制误差不抹平收益。
- **增量**：只改变arm表示，hand6通道、视觉、数据、种子与可训练参数固定；不增添新loss。
- **初始方法**：将同一24维原轨迹转换为joint和EEF两版，先验证往返FK/IK与当前状态参照，记录失真/不可达比例。
- **如何验证**：2–4个双臂任务，同训练演示、同3个种子、同梯度更新预算；报告训练/held-out误差、闭环成功、约束违背、IK失败和GPU hours；相同结果或收益仅来自控制器优势会否证“更好继承先验”。
- **可能失败**：FK/IK与实机参考系误差主导；6通道手的可执行抓姿成为瓶颈；joint原数据比EEF转换更准确。
- **创新状态**：unverified，需与UMI/人机迁移及已有相对动作研究查重。

### B. 小规模人机对齐能否补足 6 通道手的本体差距

- **来源**：EgoScale50小时human+4小时robot中训练，以及G1低DoF迁移；手关节重定向优于纯wrist监督。[External: E1, EgoScale PDF p. 4 and 9–11]
- **核心假设**：预算固定时，少量与目标视角/腕参考系匹配的人类演示，比等时长无对齐人类视频更能提升RM65-B灵巧手任务。
- **增量**：增加对齐数据桥与受约束的6通道手重定向，不复刻20K预训练，不把机器人示范数偷偷增加。
- **初始方法**：先在仿真/离线测手姿到6通道控制的可执行投影误差；等时长采集aligned与unaligned人类操作，重定向失败标签显式过滤。
- **如何验证**：保持目标robot演示数量一致，比较robot-only、加unaligned、加aligned三组；测未见物体闭环成功、手滑/接触失效、干预率与数据制作成本；若对齐组无优势或仅因标签更干净，缩小主张。
- **可能失败**：相机尚未安装导致对齐条件不成立；低DoF手不能实现人类运动；姿态估计噪声和retargeting系统误差压过先验收益。
- **创新状态**：unverified，先检查EgoScale及已有低DoF迁移工作。

### C. 微调时保留双手组合能力的最小数据策略

- **来源**：N1仅右手微调丢失左→右交接；N1.6建议pretrain数据co-training减过拟合。[Paper: PDF p. 16, Section 4.5] [External: V1, Discussion]
- **核心假设**：小比例双手交接/失败纠正回放可比增加正则loss更稳定保留基座组合能力。
- **增量**：只改数据混合，不先叠加多个残差/辅助目标；总训练样本和更新次数保持一致。
- **初始方法**：构建不会与训练任务泄漏的左右手分工与交接probe集，比较任务数据单独训练、回放混合、等量额外单手数据。
- **如何验证**：同时测目标任务收益、原技能保留和未见措辞跟随；每组同checkpoint、3 seeds、相同GPU预算；目标任务掉分抵消保留收益或只记住probe物体则否证该假设。
- **可能失败**：回放数据与新task控制模式冲突；probe太简单无法代表组合；小数据混合只是延缓过拟合而非形成可泛化能力。
- **创新状态**：unverified，需与持续学习/rehearsal/多任务微调已有方法核查。

### D. 24GB 卡上的可用性与异步执行联合验证

- **来源**：当前默认训练约35GB/卡与4×3090的24GB/卡限制；N1.7支持action chunk/RTC。[External: C2, C4]
- **核心假设**：限定模块、低microbatch和受支持内存优化能形成可复现训练方案；短执行块/RTC的收益需要超过额外延迟与计算成本。
- **增量**：工程配置与evaluation设计；不是凭硬件调参就宣称算法创新。
- **初始方法**：固定HF revision和代码commit，导出最终config、训练参数量；先100–200步测峰值显存、loss、step time与视频解码吞吐，再测推理p50/p95。
- **如何验证**：四卡/单卡可行配置报告有效batch、梯度累积、更新数和GPU hours；比较相同训练checkpoint的执行horizon与RTC；不能加载、长期不稳定或成功下降则判为不适合作主基座。
- **可能失败**：模型/优化器状态超24GB而batch缩减无效；三视角token激活过大；异步桥时间戳错位；较小执行块推理频率跟不上。
- **创新状态**：unverified，当前定位为选型验收，不作为论文创新点。

> [!todo] 后续需要真实证据
> RM65-B自定义模态往返验证；腕视角安装与标定；joint/EEF相同数据对照；4×3090显存和step time；N1.7最终加载配置；公开数字矛盾的原始rollout；本机任一基座的闭环成功率。本次完成文献/官方实现精读，未启动训练或控制机器人。
