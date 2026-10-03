---
title: "InternVLA-A1.5精读-理解与潜在预见统一基座"
date: 2026-10-03
updated: 2026-10-03
tags:
  - VLA
  - 论文精读
  - 基座选型
  - 世界模型
  - InternVLA
aliases:
  - InternVLA-A1.5 Paper Card
  - A1.5基座精读
status: 全文与附录精读-训练与实机复现待验
paper_type: methods
locator_mode: page-grounded
arxiv: 2607.04988v1
paper_date: 2026-07-06
code_commit: e6fc904f9edbfb14532e97095fc2372202517f76
sources:
  - https://arxiv.org/abs/2607.04988v1
  - https://internrobotics.github.io/internvla-a15.github.io/
  - https://github.com/InternRobotics/InternVLA-A-series/tree/e6fc904f9edbfb14532e97095fc2372202517f76
  - https://huggingface.co/InternRobotics/InternVLA-A1.5-base
---

# InternVLA-A1.5：理解、潜在预见与动作的统一基座精读

> Source coverage: Full paper，24 页，正文、参考文献与 Appendix A 均已读取。
> Extraction confidence: High for text；公式、注意力图、主结果图表另经页面图复核。
> Locator mode: page-grounded；以下 PDF p. N 均是文件内从 1 开始的页号。
> Primary analytical lens: methods。
> Secondary analytical lens: None；数据配方作为方法条件审计，本文并非独立数据集发布论文。
> Context verification: Targeted external check；官方项目页、模型卡与固定代码快照已核对，领域历史未做独立穷尽检索。
> Card completeness: Complete relative to supplied source；仍有原文内部报告矛盾，明确保留。

> [!abstract] 选型上最应记住的结论
> A1.5 是以 Qwen3.5-2B 为语义主干、460M unified expert 为连续控制分支的 VLA。50 个 foresight tokens 经冻结 WAN2.2-5B 的可微条件通道接受未来视频监督；部署删去 WAN，而非删去整个潜在预见表征。它适合研究“语义保持＋短时预测表征怎样改善动作泛化”。官方已发布 base、RoboTwin、Libero、DOMINO 检查点与训练代码；本项目 4×3090 的完整教师微调可行性尚无实测，不能从参数量或总显存直接宣布可训练。

> [!warning] 来源冲突须保留
> arXiv v1 Figure 8 已报告真机成功率，官方项目页截至本次读取却写真机数值尚未 finalized；MOF 附录称 20 次试验，而图中 76.4%、29.3% 无法由单组 20 次二元成功计数直接得到。Table 8 的“去视频损失”又被正文描述为更改 inference-time configuration，实验干预时机不清楚。因此这些数值可作为作者报告，不能当作已经复核的原始实验计数。

原文：[[InternVLA-A1.5.pdf]]。关联：[[VLA基座选择-InternVLA-A1.5、π0.5与GR00T N1.7]] · [[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]] · [[RM65-B双臂机器人]] · [[服务器-172.17.27.166]]。

## 01 基本信息

| 字段 | 内容与证据边界 |
| --- | --- |
| 完整标题 | InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization |
| 团队与作者 | 上海人工智能实验室 Physical Intelligence Team；Haoxiang Ma、Junhao Cai、Xiaoxu Xu、Hao Li、Yuyin Yang、Yang Tian、Jiafei Cao 等共 29 位作者；前七位标为核心贡献者，Weinan Zhang 为通讯作者。[Paper: PDF p. 22, Contributors] |
| 发布状态 | arXiv 2607.04988v1，2026-07-06 提交；本次查询未见 v2 或正式会议接收信息，不自行推断发表状态。[External] [arXiv 版本记录](https://arxiv.org/abs/2607.04988) |
| DOI | 10.48550/arXiv.2607.04988，arXiv 预印本标识 |
| 论文类型与问题 | 方法／系统论文；VLA、语义保持、组合泛化、潜在未来预测、跨本体模仿学习 |
| 阅读材料 | 官方 24 页 PDF，正文 PDF p. 1–17、参考文献 p. 18–21、贡献者及实验细节 p. 22–24；官方代码只是辅助实现来源，不与论文配方混写 |
| 官方代码 | [InternRobotics/InternVLA-A-series](https://github.com/InternRobotics/InternVLA-A-series)，固定 `master` 快照 `e6fc904f9edbfb14532e97095fc2372202517f76`；A1 旧代码已移至 `InternVLA-A1` 分支。[External] [固定 README](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/README.md) |
| 权重与许可 | 官方模型卡列 base、RoboTwin、Libero、DOMINO；base 是下游微调起点，benchmark 专用权重用于相应发布指标复现。代码与模型卡标 CC BY-NC-SA 4.0；研究用途与后续商用安排需按原许可处理。[External] [base 模型卡](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) |
| 本项目角色 | 候选语义＋未来表征基座；需重新适配双 RM65-B、四代手 24 维、当前头部单 RGB；规划三视角配置另行评估，不视作腕相机已装 |

术语账本：统一使用 **InternVLA-A1.5**（用户简称 A1.5）、**unified expert**（同时容纳预见与动作 token 的小专家）、**foresight tokens**（潜在预见查询槽）、**WAN2.2-5B**（冻结视频教师）、**FAST**（离散动作 tokenizer）、**flow matching**（连续速度场学习）、**compositional generalization**（已见因素的未见绑定组合）。不将 latent foresight 翻译成已实现的长期规划，也不将 A1、A1.5 与 InternVLA-M1 混作一个版本。

## 02 一句话总结

[Paper] InternVLA-A1.5 通过原生 VLM 聊天格式和持续文字监督保留语义，通过训练时冻结视频生成器监督少量预见 token，再让连续动作专家读取这些 token，在其所报告的六个仿真基准与部分真机组合泛化任务中取得优势，同时在部署时移除视频生成器。[Paper: PDF p. 1, Abstract; PDF p. 17, Conclusion]

## 03 研究问题

[Paper] 核心问题不是“机器人能否生成未来视频”，而是语言语义与物理动态怎样同时服务控制。VLM 带来目标识别、语义指令和空间定位，视频生成模型带来时空变化先验；直接把文字、像素预测和动作目标叠加，可能让语义能力在策略训练中漂移，并增加训练和部署负担。[Paper: PDF p. 2, Introduction]

精确问题可表述为：**能否在保留原生 VLM 语义训练的条件下，利用一个冻结的视频模型，把短时未来信息压缩到可被动作专家利用的少量 token，并在推理时完全省去视频生成？** 作者用组合绑定、动态场景、长流程及消融来回答，但不同实验分别支持不同部分；不能以一个总榜分数同时证明语义保持、因果动力学和计算效率。[Analysis; Paper: PDF p. 11, Experiments; PDF p. 15, Analysis]

本项目对应问题是：若同一场景出现“哪只手、哪个颜色、哪个容器、哪个先后顺序”的变化，模型是否按语言改动作，而非复现训练中的固定运动。这与单纯抓取成功率不同。要研究这一点，训练集必须让各因素可分辨，并设计未见绑定，而不是把某条指令永久绑定到一个固定场景。[Analysis; Paper: PDF p. 22, Sort Tubes]

## 04 研究背景与发展路径

[Paper-framed; external verification not performed] 下表是原文组织的路线，不是本次独立验证后的完整领域历史或首创性判断。[Paper: PDF p. 2, Introduction]

| 路线 | 获得的能力 | 代价或缺口 | A1.5 的回应 |
| --- | --- | --- | --- |
| VLM → VLA | 把图像与语言的语义先验迁移到动作 | 未必具备未来状态和动态交互表征 | 保留原生 VLM，同时加短时预见 |
| Video / World Action Model | 对未来视觉变化与动作联合建模 | 若部署必须生成视频，会有额外延迟 | WAN 只参与训练／可视化 |
| A1 等统一理解生成动作模型 | 一个框架同时承担不同目标 | 语义目标可能被弱化，生成模块可能从头学 | 持续 QA／subtask／FAST 监督，复用冻结预训练 WAN |
| A1.5 的 latent querying | 不学习一个新像素生成器，而学习可供现成生成器读取的未来条件 | 监督受 WAN 先验及一动作块时域限制 | 用紧凑条件码作为专家的共享上下文 |

[Analysis] A1.5 的“统一”主要是 token 布局、上下文与数据接口统一；Stage 2 仍存在 CE、视频 FM 和动作 FM 三项损失，并有 1:1:10 权重。不能把作者“更一致训练框架”的说法转述成“异质损失已经不存在”或“已经证明没有梯度冲突”。[Paper: PDF p. 7, Equation 8]

## 05 论文识别的核心痛点

| 痛点 | 表现 | 作者解释 | 证据与证据强度 |
| --- | --- | --- | --- |
| 语义漂移 | 理解模块停止大规模 QA／语言训练后指令跟随可能弱化 | 生成和动作目标覆盖原先语义目标 | 引言给出动机；真实未见绑定提供间接行为证据，未报告独立通用 VQA 保持曲线。[Paper: PDF p. 2, Introduction; PDF p. 12, Figure 9] |
| 目标干扰 | 离散语言、连续动作、未来视频监督不同 | 不同目标形式与尺度使联合优化困难 | Stage 2 保留三目标；注意力隔离 FAST 真值是明确防泄漏措施，但没有完整梯度冲突测量。[Paper: PDF p. 7, Equation 8; PDF p. 8, Figure 5] |
| 从头学习视觉未来 | 重建未来需要自己学习时空先验 | 现有视频生成预训练知识未被利用 | WAN 冻结＋条件梯度是设计证据；消融支持预见分支与表现相关，具体去损失协议仍待澄清。[Paper: PDF p. 6, Figure 4; PDF p. 15, Table 8] |
| 自回归动作慢 | FAST 动作逐 token 生成不适合高频闭环 | 离散动作在 Stage 1 有语义训练价值，连续控制另需接口 | Stage 2 动作块并行去噪，缓存 VLM 上下文；5090 推理约 0.1 s，仅是作者机器条件。[Paper: PDF p. 5, Section 3.2; PDF p. 11, Real-world Experiments] |
| 泛化与长时序依赖 | 记住训练绑定或丢失流程进度 | 需语义拆解与未来状态感知共同参与 | 三种未见绑定＋MOF；并未验证任意长时规划或触觉接触任务。[Paper: PDF p. 22, Appendix A.1; PDF p. 23, Figure 13] |

## 06 核心想法

**表面方法。[Paper]** 在原生 Qwen3.5-2B 上接一个同类型但更窄的专家，专家前半序列是 foresight tokens，后半是动作 embeddings。预见输出替代 WAN 的 T5 文字条件，视频 FM loss 给出监督；动作 FM 分支读取同一预见表示。[Paper: PDF p. 3, Figure 2; PDF p. 6, Figure 4]

**真正的机制。[Analysis]** 学习的是“当前图像＋指令＋状态中，哪些条件码能使固定 WAN 对正确未来去噪”，不是学习复述视频模型的文本，也不是先生成未来视频再让机器人照着走。动作 loss 还会让这些共享预见表征对控制有用。因此更准确的说法是**由视频教师约束、由动作学习利用的条件表征**。只有视频 loss 不能保证这一定是完整、可解释或动作充分的世界状态。[Paper: PDF p. 6, Equation 4; PDF p. 7, Equation 6]

**可迁移的一般认识。[Analysis]** 一个固定大模型的内部可微条件接口可充当训练信号，学生不一定要复制该模型全部输出能力；部署时保留学生表示即可。但省去了部署生成器，不意味着训练无需保存教师条件梯度经过的计算图，也不意味着教师监督廉价到任何 24GB 显卡都能容纳。[Paper: PDF p. 6, Figure 4; PDF p. 17, Limitations]

## 07 方法总览

### 7.1 一次训练样本的数据流

```text
当前 K 路 RGB + 总任务指令 + control mode + 归一化后离散状态
  → Qwen3.5 原生视觉编码与聊天模板
  → VLM context H_t
  ├─ language head：QA / subtask / FAST 动作 token → label-only CE
  └─ shared full attention → unified expert
       ├─ 50 个 foresight tokens → Z_t^f → P_WAN → C_t^f
       │    └─ 冻结 WAN：当前帧 + 4 个未来帧的 VAE latent → video FM loss
       └─ 带噪动作块 → 连续速度场 → action FM loss
```

[Paper] 每 3 层 Gated DeltaNet 与 1 层 full attention 交替，共图示 6 组；VLM 与专家只在 full-attention 层交换信息，各自 linear-attention 层独立处理。这里的 Mixture-of-Transformers 不是依任务选择一个稀疏 MoE 专家的门控结构，不能误写成动态专家路由。[Paper: PDF p. 3, Figure 2]

### 7.2 输入输出与表示

[Paper] 输入为 $K$ 路观察、语言、控制模式 `<joint>` / `<end_effector>` / `<vqa>`；机器人状态 $D\le32$，每维在归一化区间 $[-1,1]$ 用 256 桶编码。FAST 把长度 50 的动作块压缩为短离散序列，追加 2048 个动作词表项，与文字共用 embedding 与 LM head。QA 样本没有状态段和动作标签。[Paper: PDF p. 4, Input structure; PDF p. 5, Figure 3]

[External] 公开配置默认状态／动作上限 32、图像 224×224、chunk 50、动作采样 10 步；`num_video_frames=4`，`image_delta_indices` 从同一动作块均匀取当前帧与未来帧。视频抽取实现默认从 `image0` 取未来视频，其他相机保留当前帧供 VLM 用，不能据“多视角输入”推断教师同时预测三路未来视频。[固定配置](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py) · [ExtractVideoFramesTransformFn](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py)

### 7.3 部署数据流与删除边界

[Paper] 部署不解码 FAST，不调用 WAN。保留当前 VLM context、可选模型生成的 subtask、预见 token 与动作专家。动作从高斯噪声开始经 Euler 积分生成，VLM KV cache 可供多次去噪复用。FAST span 在训练存在、部署不存在，因此专家 position ids 按紧接可选 subtask 排列，避免相位差。[Paper: PDF p. 8, Attention Masking Pattern]

[Analysis] 这里没有“执行动作→生成未来→搜索候选”的测试时规划环；执行若干动作后重新观察才构成控制闭环。模型一次生成 50 步，不代表必须执行全部 50 步；RoboTwin 作者执行 18 步。预测 horizon 与执行 horizon 必须分开记录。[Paper: PDF p. 24, RoboTwin 2.0]

### 7.4 梯度与冻结

| 路径 | 哪些变量接受监督 | 何处阻断／冻结 | 对理解的影响 |
| --- | --- | --- | --- |
| VLM CE | 文字／subtask／FAST 的 label token | prompt 不计 CE；并非 prompt 对表示毫无影响 | “继续语义训练”取决于实际文本数据与标签，不只取决于一个布尔开关 |
| 视频 FM | $C_t^f$、产生它的投影与专家、可训练查询槽；共享上下文可接受上游信号 | WAN 参数冻结；教师条件输入的 Jacobian 仍需保留 | frozen ≠ 整个 WAN forward 都能 `no_grad()` |
| 动作 FM | 动作专家、动作投影与其读取的共享表征 | 专家对 FAST 真值 span 的 attention 被屏蔽 | 避免答案泄漏，不等于全部训练分支彼此无梯度影响 |

前两列的论文支持见 [Paper: PDF p. 6, Equation 4; PDF p. 7, Equation 6; PDF p. 8, Figure 5]。原文没有完整列出所有参数冻结表，所以视觉 encoder 是否冻结、知识隔离是否开启，要另查代码和具体实验配置。

[External] 固定实现 `knowledge_insulation=true` 会 detach 专家所看的 prefix K/V，阻止 suffix 梯度回到 VLM，但仍允许 attention 看 prefix 内容；`false` 才保留这条梯度。WAN 的 VAE encode 在 `no_grad()` 内，DiT 条件 forward 在视频训练中保持可微。泛用与 RoboTwin 微调脚本 `freeze_vision_encoder=false`、`train_expert_only=false`、`knowledge_insulation=false`；原始可学习槽默认在微调冻结，但专家产生的上下文预见输出仍可变化。[modeling 固定源码](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py) · [微调脚本](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/launch/internvla_a15_finetune.sh)

## 08 核心模块拆解

| 模块 | 功能与必要性 | 输入 → 输出 | 证据 | 去掉后的已测／预期效果 |
| --- | --- | --- | --- | --- |
| 原生 Qwen3.5 VLM | 保留预训练图像、空间、语言处理结构 | RGB、语言、mode、离散状态 → context 与文字 logits | Figure 2/3，PDF p. 3/5 | 未报告替换 VLM 或不保留 chat-template 的独立消融；语义漂移不能只凭设计宣布已消除 |
| subtask 文字支路 | 给动作提供当前流程步骤 | context → 子任务文字 | Equation 1，PDF p. 5；MOF PDF p. 12 | 有作者解释，没有“仅移除 subtask”的 MOF 数字；其因果贡献尚未单独隔离 |
| FAST 离散动作监督 | 给 VLM 一个动作相关 next-token 信号 | expert action → FAST labels → CE | PDF p. 4/5 | 没有单独去 FAST 消融；不能由低训练 loss 确认独占贡献 |
| unified expert | 在小维度内同时处理预见与动作，full attention 共享语义 | prefix context＋查询＋带噪动作 → contextual foresight＋velocity | Figure 2，PDF p. 3 | 未给同预算普通动作头与 MoT 架构对照 |
| foresight tokens | 做未来信息的紧凑查询接口；组内双向 | 50 个 learnable slots＋context → $Z_t^f$ | Equation 3；Table 8 | 去 token：LIBERO 98.6、Plus 77.9、RoboTwin 90.2、DOMINO 23.8；消融时机需核实 |
| WAN2.2-5B 与条件投影 | 让紧凑表示能解释未来视频，复用固定动态先验 | $Z_t^f$→WAN condition；当前／未来视频 latent → FM residual | Figure 4；Equation 4 | 去视频 loss：97.9、78.0、91.1、25.3；不能直接推断它证明物理因果性 |
| 动作 flow matching | 一次并行建模整个连续动作块 | 带噪 action＋time＋context＋foresight → velocity | Equation 5–7，PDF p. 7 | 未给等延迟 FAST-only 对照；低延迟依据是5090实测作者报告 |
| group-wise attention＋FAST mask | 跨组因果、组内双向，阻止动作读取真值 token | token group layout →合法 attention mask | Figure 5，PDF p. 8 | 泄漏避免是结构逻辑；未给故意开放真值注意力的性能数字 |

消融数字均为作者报告，统一出处 [Paper: PDF p. 15, Table 8]。不把“预期效果”补成已经做过的实验。

## 09 核心公式与符号

### 9.1 层级语义与 Stage 1：Equation 1–2

$$
\pi_\theta(a_{t:t+H},\hat\ell\mid o_t,\ell)
=\pi_\theta(a_{t:t+H}\mid o_t,\hat\ell)\,
\pi_\theta(\hat\ell\mid o_t,\ell).
$$

[Paper] $\ell$ 为总指令，$\hat\ell$ 为当前 subtask，$a_{t:t+H}$ 为动作块，$\theta$ 为策略参数。因为 subtask 标签先于动作 token，自回归顺序自然把动作条件化到子任务。论文记号 $o_t$ 在此包含多视角、指令与离散状态，和前文只用 $o_t$ 指图像的习惯略有变化；读公式时按本节定义理解。[Paper: PDF p. 5, Equation 1]

$$
\mathcal L_{\mathrm{stage1}}
=-\mathbb E_{(o_t,\ell,y)\sim\mathcal D}
\left[\sum_{i=1}^{M+N}\log p_\theta(y_i\mid o_t,\ell,y_{<i})\right].
$$

[Paper] $y$ 是 subtask 文字和 FAST 动作 labels 拼接；本式 $M,N$ 分别代表文字与动作 token 数；QA 样本只保留 answer。prompt 不计损失。其统一性是共享词表、head、CE，而不是语义和动作标签在所有样本等量存在。[Paper: PDF p. 5, Equation 2]

### 9.2 潜在预见查询与教师监督：Equation 3–4

$$
Z_t^f=\Phi_\theta([H_t;Q^f])_{\mathcal F},\qquad
C_t^f=P_{\mathrm{WAN}}(Z_t^f).
$$

[Paper] $H_t$ 为当前语义上下文；$Q^f\in\mathbb R^{M\times d}$ 为预见槽，此处 $M=50$ 与上节文字长度不是同一物理量；$\mathcal F$ 选择预见位置，$\Phi_\theta$ 为专家，$P_{\mathrm{WAN}}$ 匹配 WAN 条件维度。$Q^f$ 是模型参数槽，$Z_t^f$ 是随当前场景变化的输出，二者必须区分。[Paper: PDF p. 6, Equation 3]

$$
x_s=(1-s)x_0+s x_1,\quad v_s=x_1-x_0,\qquad
\mathcal L_{\mathrm{video}}
=\mathbb E\left[\left\|u(x_s,C_t^f,s)-v_s\right\|^2\right].
$$

[Paper] $x_1$ 是当前帧＋4 个未来帧经 WAN-VAE 编码的干净 latent，$x_0$ 是高斯噪声；$u$ 为冻结 WAN DiT。优化问题是固定 $u$ 时调整 $C_t^f$ 使速度预测逼近正确未来，梯度沿 $\partial u/\partial C_t^f$ 回传。论文方法段不用像素重建损失直接训练学生，但仍用真实未来视频建立 latent target。[Paper: PDF p. 6, Equation 4]

[External] 固定实现采用反向时间约定：视频 target 为 noise−clean、当前 latent 帧保持干净并令其速度为零。动作分支也用 noise−action 且从时间 1 积分到 0；只要插值与积分方向一致，并非与论文正向写法实质冲突。直接拼接别家 FM scheduler 时要同时核查三处符号。[modeling 源码](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py)

### 9.3 连续动作与采样：Equation 5–7

$$
a^\tau=(1-\tau)\epsilon+\tau a,\quad
\epsilon\sim\mathcal N(0,I),\quad \tau\sim\mathrm{Beta}(1.5,1.0),
$$

$$
\mathcal L_{\mathrm{action}}
=\mathbb E\left[\left\|v_\theta^{\mathrm{act}}(a^\tau,H_t,Q^f)-(a-\epsilon)\right\|^2\right].
$$

[Paper] $a$ 是整块真实连续动作，$a^\tau$ 是部分去噪块；$v_\theta^{\mathrm{act}}$ 预测把噪声送到示范轨迹的速度。预见 token 作为上下文影响动作专家，但不需要先渲染成视频再转换成控制。[Paper: PDF p. 7, Equation 5; PDF p. 7, Equation 6]

$$
a^{\tau+\Delta\tau}=a^\tau+
\Delta\tau\,v_\theta^{\mathrm{act}}(a^\tau,H_t,Q^f),\quad
\Delta\tau=1/K.
$$

[Paper] 从高斯块 $a^0$ 向 $a^1$ 做 $K$ 次 Euler 更新；这是动作生成的数值积分，不能等同于机器人关节对速度场做真实时间积分。输出动作之后仍需反归一化和执行器的目标位置解释。[Paper: PDF p. 7, Equation 7]

### 9.4 总损失：Equation 8

$$
\mathcal L_{\mathrm{stage2}}
=\mathcal L_{\mathrm{stage1}}
+\alpha\mathcal L_{\mathrm{video}}
+\beta\mathcal L_{\mathrm{action}},\qquad
\alpha=1,\ \beta=10.
$$

[Paper] 作者让语义、预见与连续动作保持共同训练。权重不同说明各项原始数值的大小不能横比；模型间 SFT loss 曲线更低，也可能受损失定义、归一化、初始模型和 action representation 影响。[Paper: PDF p. 7, Equation 8; PDF p. 15, Figure 10]

## 10 实验设计与证据链

### 10.1 训练数据、采样与预算

| Robot source | Episodes | Frames | 采样权重 |
| --- | ---: | ---: | ---: |
| InternData-A1，仿真 | 587,946 | 395.9M | 0.20 |
| AgiBotWorld，真实 | 112,988 | 206.3M | 0.25 |
| UMI，真实 | 377,018 | 201.3M | 0.10 |
| DROID，真实 | 95,658 | 27.6M | 0.15 |
| Galaxea，真实 | 19,085 | 25.0M | 0.20 |
| RoboMind 1.0，真实 | 8,638 | 5.4M | 0.10 |

[Paper] 六源约 1.2M episodes，分源 frames 加和为 861.5M，正文取约 861M；仿真 raw frame 占46%，真实54%。权重不是原始帧比例；跨组权重先由 Re-Mix 得到再人工调整，组内按帧数的 $\gamma$ 次幂取样，$\gamma=1$ 恢复组内帧比例。[Paper: PDF p. 9, Robot Manipulation Data; PDF p. 10, Figure 6; PDF p. 10, Sampling Strategy]

[Paper] 多模态四类 General QA 637K、Box QA 879K、Point QA 832K、Trajectory QA 684K，合计3032K，约3M；空间目标用绝对图像坐标表达。作者明确 robot:multimodal = **0.15:0.85**，因此大部分 sampled batch 在维持语义与grounding，不能以原始数据大就认为每个batch主要是机器人轨迹。[Paper: PDF p. 9, Multimodal Co-training Data; PDF p. 10, Figure 6; PDF p. 10, Sampling Strategy]

| 配方 | 步数 / batch | 优化设置 | 解释 |
| --- | --- | --- | --- |
| Stage 1 pretrain | 300K / 1024 | AdamW，常数 LR5e-5，warmup2K | VLM label-only CE，QA＋subtask＋FAST |
| Stage 2 pretrain | 600K / 1024 | 同上 | 加专家／50预见槽／WAN监督／动作FM |
| 通用 posttrain | 60K / 128 | cosine LR5e-5→5e-6，warmup2K | 原文允许下游保留视频分支 |
| RoboTwin 实际主结果 | 100K，24GPU×每卡16=384 | peak LR1e-4，decay horizon140K | 与通用 Table1 配方不同，不能统一抹平 |

[Paper] 所有阶段 BF16，weight decay0.01，gradient clipping1.0，预测chunk50。预训练GPU型号、总GPU时和wall-time未报告；RoboTwin给24卡数量未给型号。实机推理是**单RTX5090**，静态图／SDPA／flash-linear-attention约0.1秒一次策略推理。[Paper: PDF p. 8, Training Protocol; PDF p. 9, Table 1; PDF p. 24, RoboTwin 2.0; PDF p. 11, Real-world Experiments]

### 10.2 基准协议与主指标

| 基准 | A1.5训练／测试条件 | A1.5结果 | 比较边界 |
| --- | --- | --- | --- |
| LIBERO | 40任务，每任务50示范，共2000；一个模型混四suite；每suite500评测rollouts | Spatial98.6、Object99.8、Goal98.6、Long98.4、avg98.9 | π0.5 avg96.9、N1.7 avg97.0均为本文汇编baseline；非同骨干同预训练的机制对照 |
| LIBERO-Plus | LIBERO checkpoint零样本，无Plus训练 | total84.8 vs π0.5 84.4 | 只有0.4点总体差；Robot扰动55.1 vs73.6明显劣势 |
| RoboTwin2 | ALOHA-AgileX；50任务×(50clean＋500rand)=27500示范；绝对joint；执行chunk18；双split各5000rollouts | clean93.3、rand93.0、avg93.2；π0.5 82.7/76.8/79.8 | 已训练rand，不是clean-only跨域；没有N1.7列 |
| DOMINO | RoboTwin checkpoint静态→动态zero-shot；35个Level1 clean任务×100rollouts | zero-shot SR27.7/MS39.8；再微调29.3/42.5；π0.5 zero-shot7.5/20.4 | 最高SR仍不足30%，不能写动态问题已解决 |
| EBench | 26任务，batch128、100Ksteps；mobile manipulation | Test SR35.2/Score49.5；π0.5 29.5/45.6 | 作者称步数少于部分baseline；训练预算不完全相同 |
| SimplerEnv | WidowX/Bridge四任务，visual matching；batch128 | avg80.8；π0.5 57.1；GR00T-N1.5 61.9 | 此处GR00T为N1.5不是N1.7；baseline明确来自既有研究 |

出处：[Paper: PDF p. 13, Table 2; PDF p. 13, Table 3; PDF p. 13, Table 4; PDF p. 14, Table 5; PDF p. 14, Table 6; PDF p. 14, Table 7; PDF p. 24, Appendix A.2]。

三候选唯一同表的 LIBERO 行，便于[[VLA基座选择-InternVLA-A1.5、π0.5与GR00T N1.7]]使用：

| 本文 Table 5 方法 | Spatial | Object | Goal | Long | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.7 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| InternVLA-A1.5 | 98.6 | 99.8 | 98.6 | 98.4 | 98.9 |

[Analysis] 可据此说“A1.5论文汇编中LIBERO均值更高”，不能说“三模型已经用完全相同数据、算力、代码和多种子训练重跑，A1.5显著胜出”。论文多用published baselines，未给全部baseline共同微调样本量和统一预训练数据控制。接近饱和的98–99%还需要原始成功计数、种子和CI才能评价差异稳健性。[Paper: PDF p. 13, Table 2 caption; PDF p. 14, Simulation Benchmarks]

### 10.3 真机绑定、长流程与分项结果

| 任务 / Figure 8 | π0.5 | Motus | A1.5 | 直接支持的结论 |
| --- | ---: | ---: | ---: | --- |
| Sort Tubes | 77.8 | 64.8 | 75.9 | A1.5总体略低于π0.5；不是所有真机任务均赢 |
| Insert Tubes | 51.7 | 44.2 | 72.5 | 特定孔位插入任务提升20.8点 |
| Move Tubes | 72.7 | 56.2 | 80.5 | 指定搬运＋插入提升7.8点 |
| MOF | 29.3 | 0.0 | 76.4 | 作者报告13步流程完成率更高；计数／聚合口径待核 |

[Paper] 作者称三模型用同一示范与同一协议，随机物体位置重复试验；未给本次全部训练示范条数、模型内随机种子和置信区间。[Paper: PDF p. 11, Real-world Experiments; PDF p. 12, Figure 8]

Figure 9 展示选定绑定而不是全部绑定的原始计数：Sort（blue→left seen）π0.5/Motus/A1.5=86.0/64.6/83.0，blue→right OOD=63.6/54.5/65.0；Insert（orange→hole1/2/3 seen）60.0/36.7/80.0，orange→hole4 OOD=56.7/46.7/60.0；Move（orange→hole1/3 seen）71.9/50.0/87.5，orange→hole2/4 OOD=66.7/60.0/83.3。[Paper: PDF p. 12, Figure 9]

[Paper] Sort绑定的是arm与color，目标box由所用臂的同侧决定；held-out训练仅blue-left、orange-right，测试反转。Insert训练blue孔1/3/4、orange孔1/2/3，holdout为blue2和orange4；各绑定15trial。Move训练orange→1/3、blue→2/4，测试其余绑定，各绑定16trial。MOF为溶液配制前段，包含漏斗插入、倒液、移除、搬运、塞盖、启动搅拌等13步，不含未见绑定。[Paper: PDF p. 22, Sort Tubes; PDF p. 22, Insert Tubes; PDF p. 23, Move Tubes; PDF p. 23, Figure 13; PDF p. 24, MOF]

[Analysis] 这种划分比未见措辞更直接检验因素重组，但相同factor的不同目标孔物理难度不同；作者也提醒这一点。单组15或16trial不足以重构图9中全部小数；图8也不能从其宣称trial数直接重建。需作者提供seed、每条件n和汇总规则。[Paper: PDF p. 13, Seen/OOD discussion]

### 10.4 消融与训练效率

| 预见消融 / Table 8 | LIBERO | LIBERO-Plus | RoboTwin | DOMINO |
| --- | ---: | ---: | ---: | ---: |
| 完整A1.5 | 98.9 | 84.8 | 93.2 | 27.7 |
| w/o video loss | 97.9 | 78.0 | 91.1 | 25.3 |
| w/o foresight tokens | 98.6 | 77.9 | 90.2 | 23.8 |

[Paper] 去video loss分别低1.0、6.8、2.1、2.4点；去token分别低0.3、6.9、3.0、3.9点。更大差异出现在分布扰动或动态任务，支持预见接口与泛化相关。但原文称所有消融使用同一个两阶段预训练模型，并“varying inference-time configuration”；正常部署本来就不计算video loss，因此第一行干预到底是下游微调关闭loss、训练阶段重跑，还是某个推理flag，尚不清楚。[Paper: PDF p. 15, Table 8; PDF p. 15, Ablation Studies]

[Paper] Figure10比较π0.5、A1、A1.5在同一RoboTwin SFT setup训练60Ksteps（约1.2epochs）的loss曲线，A1.5下降更快；没有GPU-hour、显存峰值或达同等SR所需wall-time。Figure11给当前＋4未来帧的预测／GT示例，涉及机械臂运动、液位变化、zero-shot场景，但无大规模视频误差和动作反事实指标。[Paper: PDF p. 15, Figure 10; PDF p. 16, Figure 11]

### 10.5 主图表完整清单与论证作用

| 图表 | PDF页 | 作用／读图边界 |
| --- | --- | --- |
| Figure 1 | 1 | 全文总览，训练／推理区别、任务和结果；不是独立实验 |
| Figure 2 | 3 | MoT主架构，VLM与460M专家的hybrid attention连接 |
| Figure 3 | 5 | robot/VQA chat-template；label-only监督 |
| Figure 4 | 6 | 冻结WAN中的条件通道及梯度回传 |
| Figure 5 | 8 | 训练与推理attention mask；FAST span防泄漏 |
| Figure 6 | 10 | 六源robot数据和四类multimodal数据；raw frame占比与采样权重不同 |
| Figure 7 | 11 | 四真机任务场景，不直接给数量或性能 |
| Figure 8 | 12 | 四任务总体SR，A1.5并非每任务最高 |
| Figure 9 | 12 | 选定seen/OOD绑定SR，物理难度混杂需注意 |
| Figure 10 | 15 | SFT loss曲线，无原始GPU时或SR-vs-time |
| Figure 11 | 16 | 未来视频可视化，定性而非因果充分证据 |
| Figure 12，附录 | 22 | Sort绑定设计及shortcut阻断 |
| Figure 13，附录 | 23 | MOF13步与完成态；不是规划树 |
| Table 1 | 9 | 通用训练超参，须与benchmark附录分开 |
| Table 2 | 13 | SimplerEnv/WidowX，GR00T只列N1.5 |
| Table 3 | 13 | RoboTwin clean/random，训练已包括random |
| Table 4 | 13 | DOMINO zero-shot与fine-tuned，SR和MS不同 |
| Table 5 | 14 | LIBERO四suite，唯一三候选同表 |
| Table 6 | 14 | LIBERO-Plus七扰动，Robot项是A1.5失败边界 |
| Table 7 | 14 | EBench三个split，SR/Score不能互换 |
| Table 8 | 15 | 预见消融，干预阶段描述有歧义 |

清单覆盖全部11张主图、8张主表及2张附录图。提取工具的caption正则只识别冒号形式，原PDF使用句点，故原始bundle检测到0图0表；本表由完整页面文本与关键页面图手工补全，未修改技能脚本或原始bundle。公式Equation1–8亦已逐项解释。

## 11 正确理解结论

[Paper] 论文直接支持的范围是：发布A1.5在所列benchmark协议中的作者报告表现，已见robot示范基础上的未见factor绑定，RoboTwin→DOMINO及LIBERO→Plus的特定zero-shot迁移，和一个chemistry长流程。zero-shot是**相应下游分布未训练**，不是基座从未见过机器人动作、通用语义或任何相关视频。[Paper: PDF p. 11, Experiments; PDF p. 24, Appendix A.2]

[Analysis] 预见teacher目标在训练阶段使用未来GT frame，是合法监督；推理不读真实未来，也不允许连续专家读FAST真值。subtask训练有teacher forcing与部署生成之间的差别，长流程结果不能默认使用了人工oracle subtask；原文未给全部推理调度、subtask更新频率及错误传播细节，复现时须明确。[Paper: PDF p. 5, Equation 1; PDF p. 8, Figure 5]

[Analysis] “保留语义”目前以训练机制与组合绑定行为为支撑，并非在每个VQA基准上证明语义能力无损；“继承动态先验”并非证明token就是因果状态；“实时”是5090约0.1秒策略调用，不是3090端到端机器人伺服达到10Hz。相机解码、网络、动作队列、SDK执行、重规划与安全停止都不在该模型耗时数字内。[Paper: PDF p. 11, Real-world Experiments]

[Analysis] 作为基座应使用base checkpoint并对RM65数据微调。用RoboTwin专用checkpoint可评估仿真桥接，但其ALOHA动作约定、归一化、相机与机械结构不是RM65-B灵巧手原生24维；同品牌、相同总维度或发布代码支持padding都不能替代数据／执行验证。

有界结论：A1.5提供了一个部署不生成视频的语义＋短时未来表征策略，其作者实验支持特定泛化收益；是否更适合本项目，需统一示范、指令split、相机配置、控制权限与实际四卡预算验证。

## 12 作者明确承认的局限

| 作者局限 | 具体边界 | 作者未来方向 | 来源 |
| --- | --- | --- | --- |
| 预见仅覆盖一个动作块 | 获得local dynamics priors；尚未做长期imagination或显式world-model planning | 将在后续工作处理，未给具体完成方案 | [Paper: PDF p. 17, Limitations] |
| 视频生成器固定且通用 | 可继承先验受视频预训练覆盖具身场景的程度约束 | 将在后续工作处理，未声称已解决dexterity或接触不足 | [Paper: PDF p. 17, Limitations] |

相关作者说明但未列为正式局限：不同seen/OOD目标物理难度可能不同，seen→OOD差异不纯粹表示语言grounding。[Paper: PDF p. 13, Seen/OOD discussion]。下一节的算力、统计、协议担忧均为Agent分析，不能并入作者局限表。

## 13 批判性分析

| [Analysis] 观察 | 潜在问题／替代解释 | 为何影响选型 | 可检验方式 | 依据 |
| --- | --- | --- | --- | --- |
| Plus total仅比π0.5高0.4点，Robot低18.5点 | 总平均遮蔽本体／运动扰动薄弱 | RM65与ALOHA区别远大于图像背景改变 | 固定视觉扰动，单独改变关节/手/执行时延，并逐项评测 | [Paper: PDF p. 14, Table 6] |
| 去loss的消融被写成推理配置 | loss只存在训练；差异来源无法确定 | 不能按该表直接断言本机关闭教师仅掉固定几个点 | 要checkpoint lineage、训练flags、下游步数，重新做同init的教师开关对照 | [Paper: PDF p. 15, Table 8] |
| 真实20trial与百分比不兼容 | 可能多seed平均、条件平均或文稿未同步 | 影响真实成功率的确定性与置信度 | 要每条件success/n和聚合公式；原数值保留为作者报告 | [Paper: PDF p. 12, Figure 8; PDF p. 24, MOF] |
| 图10比较总loss曲线 | 三模型loss定义、权重和归一化未完全说明；低loss未必省GPU时 | 四卡调试最关心达目标SR的时间 | 统一held-out action metric，记录GPU时和SR-vs-samples | [Paper: PDF p. 15, Figure 10] |
| 预测视频看起来合理 | 可能依赖视觉运动相关性；未证反事实动作条件充分 | 接触／双灵巧手更依赖动作和约束 | 同obs/lang替换候选action，测预测是否区分结果；评估collision、contact而非只像素 | [Paper: PDF p. 16, Figure 11] |
| 使用大规模robot＋VQA预训练 | 增益同时含新主干、数据、模板、教师；未独立隔离 | 不能将对A1/π0.5全部优势归于新token | 同主干同数据固定，只切教师或预见表征 | [Paper: PDF p. 9, Data Recipe; PDF p. 15, Table 8] |
| 微调默认冻结预见原始槽与投影 | 新RM65场景的适配依赖专家context变化；教师并非“完全不能传梯度” | 手／相机分布转移可能需要不同冻结策略 | 固定WAN，比较Qf冻结/解冻，报per-block grad norm | [External] 固定modeling与finetune脚本 |

### 13.1 论文、仓库、模型卡三种来源不能混写

[External] 官方固定仓库微调脚本是BF16、batch8/卡、默认2进程、60Ksteps、LR5e-5→5e-6、`gradient_checkpointing=false`、保留WAN、`freeze_learnable_tokens=true`。论文RoboTwin主结果却是24GPU×16、100Ksteps、LR1e-4。脚本是可执行示例，不能自动当作复现93.2的完全配方。[固定RoboTwin脚本](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/launch/internvla_a15_finetune_robotwin.sh) · [Paper: PDF p. 24, RoboTwin 2.0]

[External] 公开pretrain tutorial默认只加载InternData-A1仿真数据，external VQA关闭；`enable_vqa_loss=true`在此仍承担robot文字／FAST标签，不自动装入3M VQA。启用示例`vqa_dataset.weight=0.15`也不能未查sampler语义就等同论文robot:multimodal=0.15:0.85。README还有示例VQA数据与教程待发布的TODO。重跑该示例不是自动重建完整六源＋3M配方。[预训练tutorial](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/tutorials/pretrain_internvla_a1_5_with_interndata_a1.md) · [固定README](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/README.md)

[External] 实机推荐 `inference_backend="optimized", action_loss_only=True`，作用是省WAN加载与优化推理。固定源码中该flag还冻结learnable槽及其in-proj，但不删除contextual foresight，不自动关闭VLM CE。因此“action_loss_only”这个名称不能按字面推断训练时只剩动作loss；要查实际forward与标签开关。用该flag训练属于无下游视频监督的变体，不等同完整教师微调。[固定README](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/README.md) · [固定modeling](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py)

### 13.2 RM65-B、视角与4×3090适配审计

[User] 当前实机为双RM65-B六轴＋双因时四代手每手6通道，共24个主动控制通道；头D435已装，腕部两相机未装，三RGB是规划。GPU池资源以[[服务器-172.17.27.166]]为历史实机核验入口，**4×3090是本轮实验预算**，不能将其与旧8卡池配置、161的A6000或本机Windows GPU混写。本次没有连接服务器重新核实空闲、互联或驱动。[本库依据] [[RM65-B双臂机器人]]。

| 适配对象 | 最小正确处理 | 尚无证明的部分 |
| --- | --- | --- |
| 24维本体 | 保留`[左臂6,左手6,右臂6,右手6]`原始语义；新robot type配置FEATURE/IMAGE/MASK与schema重排；可padding至32但须往返检查 | 官方ALOHA 14维适配不能直接覆盖灵巧手；不存在本次已验证的RM65官方配置 |
| abs/delta | 首轮用绝对关节/手目标清楚核查；若臂delta手abs则专门设置delta mask | `MASK_MAPPING`是是否减当前state的选择，**不是有效动作维度mask**；copy ALOHA会错分手通道 |
| 单头RGB | 先以一个真正可见视角微调和评测，缺失view按模型机制处理；不要复制头图伪造腕图 | 头视角是否充分观察抓取与插入、224分辨率是否损失孔/手指细节未测 |
| 三RGB规划 | 真正装好并标定腕相机后，记录顺序、时间戳、预处理；单/三视角是独立变量 | 仿真虚拟腕视角可研究上限，不能算真实已有相机 |
| 视频教师 | 当前帧和未来帧来自同episode/同source view，与chunk覆盖时域匹配；不能跨episode末尾误取 | 通用WAN对因时四代手接触、遮挡和触觉状态是否有可靠先验未测 |
| GPU/软件 | 独立Linux环境验证Python3.11、cu128/PyTorch2.10、Transformers5.2与custom patch、FLA/causal-conv1d kernels | 本文不提供4×3090 VRAM峰值、吞吐、训练天数或确定可训练结论 |

源码依据：[DeltaActionTransformFn与ReorderStateActionTransform](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/transforms/core.py) · [映射表](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/src/lerobot/transforms/constants.py) · [安装条件](https://github.com/InternRobotics/InternVLA-A-series/blob/e6fc904f9edbfb14532e97095fc2372202517f76/tutorials/installation.md)。

[Analysis] 4张24GB GPU通常不会自动构成一张96GB GPU；多卡复制还是分片决定每卡驻留权重与优化器成本。A1.5学生比更大基座参数量少，是可优先测的理由，但完整教师路线仍需WAN5B、学生权重／梯度／optimizer、条件梯度激活和视频解码；关闭教师可降低负担，却改变训练条件。原文没有提供足以外推“4卡训练几天”的GPU时；24卡实验也不能按线性比例简单乘6得到四卡wall-time。

## 14 学到的知识

### Agent-derived knowledge candidates

1. **训练时预测与测试时规划分开。** 未来监督可以只塑造策略表征，部署不需要生成未来图像；是否有规划取决于测试时是否搜索、评分、重选动作。[Paper: PDF p. 17, Limitations]
2. **可学习槽与上下文输出分开。** 固定$Q^f$不意味着$Z_t^f$固定；新观察经过可训练专家仍能得到变化的未来条件。[Paper: PDF p. 6, Equation 3]
3. **冻结参数与冻结计算图分开。** 视频教师无optimizer更新，但学生要从条件输入取得梯度；这对显存预算和低成本教师替代至关重要。[Paper: PDF p. 6, Equation 4]
4. **离散／连续动作可承担不同角色。** FAST用于将动作嵌入语言训练，FM用于高效控制；动作token在训练当标签，连续专家应防止读取其真值。[Paper: PDF p. 8, Figure 5]
5. **native prompt也是迁移接口。** 状态、mode、subtask纳入原生chat template可以减少输入处理结构改变，但稳定性是作者经验与机制解释，不是所有模板已量化比较。[Paper: PDF p. 17, Conclusion]
6. **组合泛化需要设计shortcut。** unseen arm-color或color-hole绑定把两个可能的memorization策略同时打断；不过目标孔难度仍需平衡。[Paper: PDF p. 22, Sort Tubes]
7. **榜单均值与工程适配分开。** A1.5在LIBERO-Plus总均值高，Robot扰动却明显低，这恰是跨本体候选不能只看最高avg的实例。[Paper: PDF p. 14, Table 6]

## 15 与已有知识的连接

| 连接 | 相似／冲突／可组合部分 | 来源边界与用途 |
| --- | --- | --- |
| 与π0.5 | 都使用subtask与离散action语言监督，并再提供连续控制；A1.5明确借用层级factorization与co-training思路，额外引入WAN-conditioned预见槽 | [Paper: PDF p. 4, Training Recipe; PDF p. 5, Equation 1]。π0.5完整机制由其独立精读卡核实，不能据本篇related work代写 |
| 与GR00T N1.7 | 均为待适配本体的VLA候选；本篇只给N1.7 LIBERO published row，未证明RM65迁移差距 | [Paper: PDF p. 14, Table 5]。N1.7发布规格与训练策略应看其官方原文，不能用N1或N1.5替代 |
| 与LingBot-VLA2选型笔记 | 本项目已有未来表征监督研究关注；共同问题是预测表示是否具有action relevance，以及资源预算下监督开关如何影响泛化 | [Analysis] 关联[[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]]；现有笔记关于三相机和8卡的旧假设以最新资源卡修正，不外推本次复现结果 |
| 与MICA-VLA语言约束 | A1.5的arm/color/target绑定适合作“语义变了动作是否跟着变”的底座检查；但本篇未验证禁止接触、先后约束和违反率 | [Hypothesis] 可另建约束split；语言性能不能只用完成率，需要正确目标率、禁止接触率、过程约束率 |
| 与动作残差研究 | 若预见槽对动作影响不充分，增加视频loss可能只让teacher易解释而动作支路忽略它 | [Hypothesis] 应先观察action Jacobian/attention利用与遮蔽效应，再加模块；原文没有token responsibility归因实验 |

这些连接均是Agent-derived candidates；没有声称以上组合在文献中未出现，也没有将候选假设写成用户已确认的研究贡献。

## 16 研究想法

### Agent-derived research candidates

### 16.1 候选：本体变化时预见表征是否真在改善动作

- **起点。** Plus中Robot55.1低于π0.5的73.6，作者承认冻结通用WAN受具身覆盖限制。[Paper: PDF p. 14, Table 6; PDF p. 17, Limitations]
- **[Hypothesis] 核心假设。** 对RM65-B24维接触任务，WAN视频loss改善视觉可预测性，却未必降低手与臂的动作误差；其收益随任务是否受可见动态主导而变化。
- **相对论文的delta。** 评价从既有ALOHA转换到新24维本体，加入分块action/接触指标，固定学生主干与数据。
- **初始方法。** 从同一base checkpoint取A：无下游WAN监督、保留预见槽；B：完整WAN监督、预见槽冻结；C：完整WAN监督、预见槽解冻。先不加入新的loss或残差模块。
- **如何验证。** 2个可见pick-place、2个遮挡/插入任务，每任务约50条示范作为起步预算；训练split、seed、有效样本数、动作表示和相机保持一致。先离线每臂/每手MAE、限位率和held-out指令目标率；随后每条件20个试验作为先导，再按方差扩展。比较completion、contact误差、p95延迟以及GPU小时；若视频指标改善而动作与成功率无改善，假设被支持；若接触收益同样稳定则反驳“仅视觉相关性”的解释。
- **可能失败。** 手部标定误差掩盖模型区别；四卡内完整WAN训练无法满足显存/吞吐预算；数据量不足以学新本体；teacher目标本身被错误时间对齐。
- **创新状态。** unverified，需prior-art search，不称新方法。

### 16.2 候选：接触关键区域对冻结视频教师监督是否更重要

- **起点。** Figure11只展示运动／液位，作者未评估局部接触真实性；通用WAN先验可能不足。[Paper: PDF p. 16, Figure 11; PDF p. 17, Limitations]
- **[Hypothesis] 核心假设。** 在相同训练预算下，把未来监督限定到手-物体接触区域或控制相关局部视野，能减少背景主导的监督，改善灵巧手控制；仅降低视频误差不是成功判据。
- **delta／初始方法。** 保持教师冻结、学生结构、50槽不变，比较原图source_view与由既有标定/检测得到的局部crop；不要同时换教师或增加多个loss。若用检测器，单独报告漏检与额外耗时。
- **如何验证。** 同一数据与训练sample预算，global vs crop；先用无新硬件的头RGB做2任务，记录手/臂分块误差、抓取滑移与约束违背；摄像头ROI遮挡或离开视野时单独评测。无稳健动作收益即否证，不能用更好重建图替代控制结论。
- **可能失败。** crop丢掉长程目标和空间语义；源相机无法看到接触；改变图像尺度破坏WAN预训练分布；检测错误与控制收益混杂。
- **创新状态。** unverified，需检索局部预测监督与object-centric VLA先验。

### 16.3 候选：组合绑定与真实约束的一致性

- **起点。** 作者证明arm-color与color-hole部分绑定，但未覆盖禁止对象或顺序约束。[Paper: PDF p. 22, Appendix A.1]
- **[Hypothesis] 核心假设。** 持续VQA/subtask训练可能改善对象目标绑定，却不足以保证双臂执行“不碰红杯”“先左再右”；在同场景变语言时，其约束违背率仍可能高。
- **delta／初始方法。** 扩展评价，不先加算法：同一scene分别只改arm、target、order或禁止对象，每项因素独立平衡；保留简单语言绑定作为正对照。
- **如何验证。** A1.5、π0.5、N1.7用共同原始示范和同一24维执行桥，报告任务完成率、目标正确率、约束违背率、干预率和失败阶段。训练/测试按factor组合划分，防止指令与scene固定绑定；每模型等量数据与同样动作重规划权限。若违背率和普通绑定一样稳健，否证“高层语义与约束脱节”的假设。
- **可能失败。** 训练指令本身没有可见差异；真实碰撞定义不可靠；动作时延与轨迹漂移被误归因语义；新增约束场景比基线更难，导致物理难度混杂。
- **创新状态。** unverified；这是候选实验问题，尚未完成先验检索与新颖性判断。

### 16.4 本项目最小开始顺序与停止条件

[Analysis] 先把资源和本体接口跑通，再决定选型。下面是待执行计划，本次没有下载大权重、安装机器人训练环境、启动训练或驱动真机。

1. **离线接口。** 取单episode，核对24维顺序、单位、手6通道、abs/delta往返、32维padding和反重排；用真实头RGB与语言，不伪造腕相机。检查当前＋4未来帧同episode、不越界、对应预测50步时域；可选subtask标签来源单列。
2. **action-only推理。** base checkpoint用官方optimized路径测试1与规划3视角的显存和延迟；报cuda allocated/reserved峰值、p50/p95策略调用、预处理、端到端队列时延。3视角仅可在真实已采或明确仿真素材上评估。
3. **四卡训练冒烟。** 先microbatch1，固定数据与原始loss，核查distributed策略到底复制还是分片；只跑约100–300步测各卡峰值、steps/s、有效samples/s、loss数值及各模块grad。分别测action-only变体和完整教师变体，后者若OOM停止并记录，不把前者通过当完整方法可训。
4. **小规模可比较SFT。** 在相同2–4任务和相同示范上做完整教师/无教师对照；预计天数只由实测吞吐外推，另加eval与checkpoint开销。不能按24卡论文训练或上一模型耗时直接套用。
5. **语言与控制验收。** 先离线held-out绑定，再经过仿真与已验证停止路径进入真机；简单抓取通过不算组合泛化或完整长流程成功。若手单位／相机遮挡／队列时序失真，先修数据和桥接再增加模型结构。

证据材料位置：`E:\Workspace\VLA-benchmark\research\2026-10-03-vla-foundation-reading\internvla-a15`，包括原脚本生成的`source_bundle.json`、24页rendered-pages、官方固定源码与最终`audit-report.json`。本次归档的是文献及静态实现精读，未改变既有机器人代码或训练状态。