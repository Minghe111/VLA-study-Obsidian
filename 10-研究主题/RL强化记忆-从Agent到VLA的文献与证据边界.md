---
title: RL强化记忆：从Agent到VLA的文献与证据边界
date: 2026-10-16
updated: 2026-10-16
tags: [VLA, 强化学习, 记忆, 文献调研, 硕士开题]
status: 原论文方法与对照核查-未复现
research_cutoff: 2026-10-16
sources:
  - https://arxiv.org/html/2506.15841v2
  - https://arxiv.org/html/2508.19828v5
  - https://arxiv.org/html/2509.25911v1
  - https://arxiv.org/html/2601.03192v2
  - https://arxiv.org/html/2507.02259v2
  - https://arxiv.org/html/2609.38641v1
  - https://arxiv.org/html/2610.22368v1
  - https://arxiv.org/html/2502.10550v3
  - https://arxiv.org/html/2508.19236v2
  - https://arxiv.org/html/2604.18933v2
  - https://arxiv.org/html/2606.21188v1
  - https://www.pi.website/download/Mem.pdf
  - https://www.pi.website/download/pi07.pdf
---

# RL强化记忆：从Agent到VLA的文献与证据边界

> [!abstract] 结论
> RL学习记忆写入、检索和使用已有明确先例，包括具身VLA。MEM1、Memory-R1有直接SFT对照；AD-Memo对驾驶VLA的记忆文本分配长期RL信用；VaM在π0.5真机后训练中利用价值历史。证据支持“部分设置下监督记忆的长程泛化仍有缺口，RL可继续改善”，不支持“监督记忆普遍很弱”“只有RL能学记忆”或“尚无人做物理记忆RL”。

## 1. 此前三式与MEM的关系

上一轮 $z_t=E(H_t^{short}),m_t=F(m_{t-1},z_t,a_{t-1}),a_t\sim\pi(z_t,m_t,g)$ 是**通用循环策略／历史压缩示意**，不是MEM原论文公式，也不是新算法。它允许长期表示直接进入动作策略；MEM原结构则由高层更新文本摘要并产生下一子任务，低层依据短期历史、子任务和总目标执行，长期摘要主要经高层影响低层。

| 记忆接口 | RL优化便利性 | 代价 |
| --- | --- | --- |
| 连续循环隐状态 | 可通过时间反传联合更新记忆与动作读出 | 长程信用、与flow链联合训练的计算开销、难审计 |
| MEM式显式语言摘要 | 易审查、纠正、复用冻结低层，便于隔离高层管理 | 离散文本需策略梯度／监督蒸馏等，长序列探索更难 |
| 小型结构化记忆 | 操作空间小，最适合先验证memory-only RL | 人工结构偏置强，规则／SFT可能已经足够 |

没有同任务、同预算比较，就不能称一种更优。任意有限历史／摘要都不自动恢复Markov性；需要其保留决策充分信息或近似belief。可以直接在POMDP训练有记忆策略。

## 2. 哪种证据才能支持RL优于SFT

核查四项：相同记忆结构／容量／上下文；实际更新记忆而非只更新动作／回答器；固定其他组件或有交叉消融；匹配标注、rollout、教师与训练预算。相同训练QA不等于等总算力，文本动作、离线驾驶和连续机械臂控制也不能互换。

## 3. LLM／Agent原论文

### MEM1：直接SFT对照，特定长程泛化差距

[MEM1 v2](https://arxiv.org/html/2506.15841v2)以PPO联合学习递归internal-state摘要与检索／推理／回答，每轮清除旧上下文。用2D attention mask重建采样时可见上下文，外部检索内容不计为策略生成token。

§4／Table3：相同Qwen2.5-7B和prompt／rollout框架，GPT-4o轨迹SFT的Wiki-RAG单目标EM **0.302**，RL **0.405**；F1 **0.358→0.471**。Appendix D.1中目标数量增加时SFT明显衰减。这支持该配方下SFT长程泛化不足，但RL同时改摘要、检索和推理，未验证等总计算预算。附录Table4多目标EM汇总分可超过1，不能当普通SR。

### Memory-R1：学操作与使用，但PPO不是每次都胜SFT

[Memory-R1 v5](https://arxiv.org/html/2508.19828v5)用PPO／GRPO训练Memory Manager的ADD／UPDATE／DELETE／NOOP，以及Answer Agent对记忆的选择与回答。更新一方时冻结另一方，主表是完整系统。

§4.1说明Memory-SFT用相同架构和训练QA，模仿GPT-5轨迹。Table1 LoCoMo Overall F1：

| 基座 | SFT | PPO | GRPO |
| --- | ---: | ---: | ---: |
| LLaMA-3.1-8B | 42.81 | 41.05 | 45.02 |
| Qwen-2.5-7B | 39.51 | 41.72 | 43.14 |

LLaMA下GRPO高2.21个百分点，PPO反低1.76个百分点。原文笼统称PPO优于所有non-RL基线，与该SFT行不完全一致，应以表为准。收益混合写入与回答更新；未找到完整SFT步数、教师成本与等FLOPs说明。

### Mem-α：单独训练memory writer，无SFT对照

[Mem-α v1](https://arxiv.org/html/2509.25911v1)用GRPO训练core／semantic／episodic记忆构建，BM25检索和Qwen3-32B回答器固定。奖励含QA正确、格式、压缩、语义有效性。同结构base均值0.389→RL 0.642，但平均记忆约3.9K→7.9K tokens，且没有同结构SFT，不能当等预算优于SFT的证明。

### MemRL与MemAgent：更新位置不同

- [MemRL v2](https://arxiv.org/html/2601.03192v2)：LLM冻结，记忆条目保存Intent–Experience–Utility，检索按语义与效用选择；实际效用更新 $Q\leftarrow Q+\alpha(r-Q)$。这是外部效用／检索学习，不是PPO／LoRA或摘要模型训练。ALFWorld是文字动作接口。
- [MemAgent v2](https://arxiv.org/html/2507.02259v2)：多会话DAPO按最终答案奖励训练前序记忆更新。无RL对照说明仅提供workflow可能不够，但不是高质量SFT对照。

## 4. VLA与物理控制的直接近邻

### AD-Memo：记忆文本有单独的长期信用

[AD-Memo v1](https://arxiv.org/html/2609.38641v1)用于自动驾驶VLA，先SFT后Da Capo（Deviation-Adjusted Causal Advantage Policy Optimization）。驾驶token主要用当前步轨迹奖励，记忆token使用后续驾驶表现与终局VQA，随后做group-relative更新。

Table1 All-way-stop：

| 指标 | SFT-only | 后续RL |
| --- | ---: | ---: |
| Avg. ADE，低为好 | 2.121 | 1.866 |
| Go SR | 40.66% | 45.01% |
| MCQ准确率 | 45.99% | 51.66% |

边界：驾驶画面由数据replay，预测动作不改变后续画面；只有记忆链是自生成闭环。Appendix F说明独立action expert未参与训练／评估，不能称flow专家记忆RL。主表联合改驾驶与记忆，非memory-only因果收益；一般驾驶Table3的MCQ 65.63→65.51%、minADE 1.005→1.091还变差。

额外核对：All-way-stop的standalone memory-writing SFT系数 $c_1=0$ **不等于无记忆监督**。§3.2中驾驶输出 $y^{drive}=(CoT,mem,\tau^{future})$，memory token仍被驾驶NLL直接监督；Appendix C.1.1也有规则生成记忆标签。此处只核查论文公式，未审代码loss mask。

### Value as Memory（VaM）：π0.5＋MEM式历史＋真机RL

[VaM v1](https://arxiv.org/html/2610.22368v1)首版提交2026-10-15 17:59:31 UTC，即北京时间2026-10-16 01:59:31。本次已获取公开HTML/API，未核查正式录用。

§3.2／§3.4：基于π0.5和MEM式时空编码，把近期value预测缓存与本体历史经GRU编码，隐藏状态追加在视觉token后，**确实输入动作专家**。价值专家用分布式TD，动作专家用优势条件化flow matching，属于RECAP式路线，不是PPO或ACoB的critic梯度反传。

流程为示范初始化→离线RL warmup→Online DAgger→Learn from Experience真机在线学习。Table1最后三个在线checkpoint均值：

| 方法 | 盒子组装 | 叠衣服 | 磁吸巴士组装 |
| --- | ---: | ---: | ---: |
| VaM | 96.7% | 88.9% | 76.7% |
| 无value cache | 81.7% | 48.9% | 43.3% |

平均29.4个百分点是对**无cache消融**；§4.4的32.8个百分点是对匹配梯度步数的**Offline RL**，都不能改写为相对SFT增益。误差跨checkpoint，不是多随机种子。缓存规则固定，学习的是价值内容和动作利用，并未学习自由的写入／删除。理论也未证明缓存恢复完全Markov性。

## 5. 监督方法的强反例与RL的限制

- **MEM** §III-B／§IV-B：用摘要监督以及失败上下文＋人工纠正SFT学到改变抓取／开门策略；弱的是无历史、天真拼接或记忆预训练不足，不是证明监督摘要无效。
- **π0.7** §VI-A／§IX-A：RL rollout作为额外行为示例／蒸馏数据；Fig.8在记忆任务上无需任务微调即可达到或超过π0.6-MEM专家。数据来自RL不表示记忆写入器被RL优化。π0.6*的RL成效也不是memory-only消融。
- [MemoryVLA v2](https://arxiv.org/html/2508.19236v2)：监督训练记忆融合与动作生成。
- [Gated Memory Policy v2](https://arxiv.org/html/2604.18933v2)：按有／无历史能否降低动作预测误差生成门控标签，非RL学习何时读，再以cross-attention学习读什么。
- [CAMP v1](https://arxiv.org/html/2606.21188v1)：自监督压缩动作史与监督动作目标塑造决策相关记忆；“对任务有用”不属于RL专有能力。
- [MIKASA v3](https://arxiv.org/html/2502.10550v3)：PPO-LSTM在部分任务优于MLP，但更长记忆或稀疏奖励下仍会失败；RL＋循环模块不是充分条件。

## 6. 可辩护的开题动机与验证

2026-10-16补充 [[记忆RL方案框架-冻结π0.5与显式进度记忆]]，给出执行／训练两张框架图、记忆操作接口和冻结边界；是拟议方案，未实现或实验验证。

可表述为：**闭环物理操作遇到未见失败和恢复历史时，监督训练的记忆更新可能失配；能否用后续任务回报，改善保留／更新／纠错，超过同架构、同交互和标注预算的强SFT？**

先固定低层，比较相同记忆容量、输入与操作集合下的规则、监督更新、纠正SFT及memory-RL；若同时更新动作专家，应做记忆／动作交叉组合，区分动作变强和记忆变好。检查真实进度准确率、漏步／重复操作、恢复成功、记忆污染与成本，而不是只比较摘要流畅性。

VaM须作为直接近邻；AD-Memo已覆盖给记忆与动作不同信用范围的宽泛构想，不能直接声称首次。先证明当前系统有该故障，再做等预算对照，最后定位方法差异；找到弱SFT基线不足以确认新算法或RL必要性。本条不自动替换已定开题题目／动作专家主线。

## 7. 核查范围与关联

截至2026-10-16；以上结果均为作者报告。本次核查原论文HTML、方法与对应表格，MEM／π0.7沿用同日官方全文核查；没有安装、训练、复现或真机操作。检索非穷尽综述，不确认其他近邻不存在。HTML来源校验清单见 [[20-论文与模型/证据/2026-10-16-RL强化记忆文献/来源与核查范围.json]]。

- [[MEM精读-短期视觉与长期语言记忆]]
- [[强化学习优势的数学本质-从序贯决策到具身选题]]
- [[VLA动作专家适应-条件利用、多解生成与闭环强化学习]]
- [[VLA-Precision精读-价值校准、动作专家更新与Q-VGM比较]]
