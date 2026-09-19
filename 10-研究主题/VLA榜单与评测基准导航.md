---
title: VLA榜单与评测基准导航
date: 2026-09-18
source_date: 2026-09-18
tags: [VLA, 评测基准, 榜单, 双臂操作, RL, 语言约束]
aliases: [VLA排行榜, 机器人模型榜单导航]
status: 用户资料归档-未独立复核网页
---

# VLA榜单与评测基准导航

关联：[[具身模型训练与仿真实验平台选型]] · [[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]]

> [!info] 来源与时效
> 本笔记整理自用户在 2026-09-18 提供的榜单汇总；原材料注明截至当日已核查。本次仅归档与结构化整理，未重新访问网页独立核验。下文任务数量、榜单设置、更新日期和开源要求均为该材料的时间快照，正式选型或论文引用前需回到链接核对。

> [!abstract] 怎么使用这些入口
> 各榜单对应不同机器人、任务和协议，没有适用于所有场景的统一 VLA 总榜。面向睿尔曼双臂、VLA＋RL 与语言约束研究，原材料建议的关注顺序是 **RoboTwin → RoboDojo／RoboChallenge → RoboCasa365**；社区汇总用于发现方法，Hugging Face 与官方仓库用于核查权重及复现条件。

## 1. 官方榜单：按研究目标选择

| 平台与地址 | 评测类型 | 主要看什么 | 对本研究的用途 | 比较时的关键边界 |
| --- | --- | --- | --- | --- |
| [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard) | 双臂仿真操作 | clean2clean、clean2random、Average 与逐任务成功率 | 优先用于双臂基线和随机化鲁棒性比较 | 区分 Co-train 与 Single-task SFT；总平均不能替代逐任务分析 |
| [RoboDojo](https://robodojo-benchmark.com/) | 仿真＋真机 | 泛化、记忆、精度、长时程、开放词汇指令跟随 | 观察模型能力分布，寻找语言与记忆相关评测 | 仿真与真机结果分别看；本体迁移需要另行验证 |
| [RoboChallenge](https://robochallenge.ai/) | 真实机器人 | Table30／Table30 V2 得分、成功率和执行记录 | 观察真实物理环境中的闭环表现 | 区分版本和比赛赛道；Score 不等于 Success Rate |
| [RoboCasa365 Leaderboard](https://robocasa.ai/leaderboard.html) | 厨房操作仿真 | Atomic-Seen、Composite-Seen、Composite-Unseen | 多任务与未见组合任务泛化 | “365”不是该榜单评测任务数；核对 horizon 与评测版本 |

### RoboTwin：双臂研究的首要入口

- 原材料记录的协议：50 个目标任务，每任务 50 条 clean 示范；clean 与 randomized 设置下分别进行每任务 100 次测试。
- **clean2clean**：较接近训练分布的执行表现。
- **clean2random**：随机化环境中的执行表现，适合重点观察鲁棒性。
- **Average**：以上两种设置的平均值。
- **Co-train**：一套策略联合学习多个任务；**Single-task SFT**：可能每个任务分别微调一个 checkpoint。必须查看 Track 列再比较。
- 原材料记录官方要求上榜模型提供公开代码、权重和技术报告；具体模型的链接完整性仍需逐项核查。

研究使用：优先查看 clean2random 和逐任务结果，同时记录训练轨道、数据预算与 checkpoint 数量，避免把不同训练成本的结果当成同条件比较。

### RoboDojo：观察能力分布

- 原材料记录 42 个仿真任务、18 个真实任务。
- 仿真部分按泛化、记忆、精度、长时程执行和开放词汇指令跟随划分能力。
- 真机部分提供标准化硬件、场景复位和远程评测接口。
- 对 MICA／语义控制研究，可关注语言跟随与记忆维度，而不只记录总成功率。

评测机器人的能力与睿尔曼本体上的可部署性需要分别验证。

### RoboChallenge：追溯真实执行记录

- 原材料记录首页已有 Table30 V2，同时存在 CVPR 2026 等独立比赛赛道。
- Table30、Table30 V2 与比赛赛道不应混合排名。
- **Score** 可能反映任务进度；**Success Rate** 反映任务成功比例，两者不可互换。
- 具体运行记录可能包含 checkpoint、推理代码、微调代码与 rollout 明细；选型时应点开记录追溯。

### RoboCasa365：观察组合泛化

- 原材料记录公开榜单评测 50 个目标任务。
- **Atomic-Seen**：已见原子任务。
- **Composite-Seen**：已见组合任务。
- **Composite-Unseen**：未见组合任务，是观察组合泛化的重要维度。
- 榜单提供提交者与开源标记；模型详情可用于查找代码、checkpoint 与训练配置。
- 原材料提到部分结果因评测 horizon 更新而重测，引用时需要核对版本及测试时长设置。

## 2. 社区汇总：发现方法与跨基准检索

| 平台 | 地址 | 特性 | 使用边界 |
| --- | --- | --- | --- |
| Evo-SOTA／Evo-Studio | [网站入口](https://sota.evomind-tech.com/) · [开源项目 MINT-SJTU/Evo-SOTA.io](https://github.com/MINT-SJTU/Evo-SOTA.io) | 汇总 LIBERO、LIBERO-Plus、CALVIN、Meta-World、RoboChallenge、RoboCasa365、RoboTwin，也包含灵巧操作；可按训练方法、评测设置和开源状态筛选 | 数据来自论文或第三方复现，不等于官方统一重测；原入口据材料会跳转 Evo-Studio |
| VLA Leaderboard | [网站入口](https://vlaleaderboard.com/) | 按 LIBERO、CALVIN、VLABench、Meta-World 等子榜浏览成绩，并查找论文、代码或数据 | 每个子榜独立更新；首页活跃不等于所有子榜最新 |

- **VLA＋RL 检索**：Evo-SOTA 区分监督微调与 RL 方法，适合发现候选方法，再回原论文核对训练预算和评测条件。
- **更新时间快照**：原材料记录 VLA Leaderboard 的 LIBERO、CALVIN 子榜更新至 2026 年 9 月，VLABench 子榜仍标注 2025 年 12 月；该信息不代表未来持续更新状态。
- **引用原则**：社区榜单作为研究导航，论文中的数字尽量追溯到原论文、官方结果或可复现的执行记录。

## 3. 常见基准的官方入口

| 基准 | 地址 | 重点关注 | 注意事项 |
| --- | --- | --- | --- |
| LIBERO | [官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) | Spatial、Object、Goal、LIBERO-10 等具体任务套件 | 仓库主要提供任务、数据和评测代码；不能只记“LIBERO 总分”，最新方法可结合社区榜与原论文查找 |
| LIBERO-Plus | [官方仓库与结果表](https://github.com/sylvestf/LIBERO-plus) | 相机、机器人、语言、光照、背景、噪声和布局变化 | 分开记录不同扰动维度；语言变化与语言鲁棒性研究直接相关 |
| CALVIN | [官方网站](https://calvin.cs.uni-freiburg.de/) | 连续完成任务数、平均链长度 | 区分训练／测试环境划分；原材料称官网展示偏早期，最新结果需结合原论文核查 |
| VLABench | [官方网站](https://vlabench.github.io/) | 实际交互执行的 VLA、VLM／LLM 工作流、非交互 VLM 评测 | 原材料记录网站仍有预览结果和“完整榜单待发布”说明，不宜视为完整实时榜 |

> [!warning] 语言理解与闭环执行是不同证据
> VLM 在图片上理解任务或生成计划的得分，不等于 VLA 在机器人闭环中完成任务的得分。比较 VLABench 等结果时，先确认是交互式策略评测还是非交互式语言模型评测。

## 4. Hugging Face：核查权重与复现条件

入口：[Hugging Face Robotics 模型页（Trending 排序）](https://huggingface.co/models?pipeline_tag=robotics&sort=trending)。

可查看模型卡、权重文件、更新情况、下载量与点赞；适合核查：

- 是否公开权重，以及 checkpoint 对应的机器人、任务和 benchmark。
- 许可证是否满足研究与使用需求。
- 模型卡是否说明动作空间、输入观测、归一化与推理条件。
- 官方代码和训练配置是否齐全，是否仍有人维护。

**Trending、下载量和点赞是热度信号，不是机器人控制能力排名。** 选型分两步：用榜单发现值得研究的模型，再用官方代码与模型卡判断是否值得投入复现。

## 5. 面向双臂、VLA＋RL 与语言约束的查阅路线

1. **双臂任务与随机化鲁棒性**：先查 RoboTwin，重点查看 clean2random、逐任务结果和 Track。
2. **语言、记忆与长时程能力**：查 RoboDojo，并用 LIBERO-Plus 的语言变化评测补充语言鲁棒性视角。
3. **真机闭环证据**：查 RoboChallenge 的具体运行记录，记录版本、赛道与机器人本体。
4. **未见组合泛化**：查 RoboCasa365 的 Composite-Unseen，核对 horizon。
5. **发现 RL 方法**：用 Evo-SOTA 的训练方法筛选，再回原论文确认 RL 数据、奖励和训练预算。
6. **复现可行性**：查官方仓库与 Hugging Face 的权重、许可证、训练配置和本体适配条件。

以上是按研究需求组织的检索建议，不代表已经在本机复现或在睿尔曼上验证过这些模型。

## 6. 后续记录单个模型时保留的字段

| 字段 | 建议记录内容 |
| --- | --- |
| 来源与日期 | 官方页面／论文／第三方复现，链接与实际核查日期 |
| 基准协议 | benchmark 版本、任务套件、赛道、clean／randomized、horizon |
| 训练条件 | SFT／RL、Co-train／Single-task、示范数量、额外数据与训练预算 |
| 指标 | Success Rate／Score／平均链长度、逐任务表现、测试次数与随机种子 |
| 模型与代码 | 模型版本、checkpoint、代码 commit、训练及推理配置 |
| 开源与本体 | 权重、许可证、机器人、末端执行器、动作与观测接口 |
| 证据等级 | 作者报告、第三方复现、本机仿真、真机实测；不要互相替代 |

> [!todo] 实际选型或引用前
> 重新核对候选模型对应的官方协议、权重与逐任务结果，并记录核查日期。所有榜单表现都需要通过本体适配与闭环实验，才能成为睿尔曼部署证据。
