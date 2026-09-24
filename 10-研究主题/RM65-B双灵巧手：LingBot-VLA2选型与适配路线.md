---
title: "RM65-B双灵巧手：LingBot-VLA2选型与适配路线"
date: 2026-09-18
updated: 2026-09-24
tags:
  - 具身智能
  - RM65-B
  - VLA
  - 基线选型
status: 调研完成-待实测
aliases:
  - 睿尔曼双臂VLA选型
---

# RM65-B 双灵巧手：LingBot-VLA 2.0 选型与适配路线

> [!abstract] 本次决策
> 主模型选 LingBot-VLA 2.0；第一批对照选 π0.5、SmolVLA、ACT。扩展对照选 LingBot-VLA 1.0 和 GR00T N1.7。优先完成固定底盘、三视角、双臂双手的共同数据与执行接口，再加入 RL 和语言约束。
> “支持自定义本体”与“已有 RM65-B 真机验证”分开记录。当前没有核实到可直接运行本机的 LingBot-VLA 2.0 官方适配包。

## 1. 硬件事实与假设

> [!info] 2026-09-24 仿真接入补充
> 下文保留 9 月 18 日的手册 BOM 与选型假设。本项目后续仿真已按用户提供的因时四代触觉手 SDK／URDF 资料接入双 RM65-B，原生动作是 24 维；这不等于手部编码、抓取参考系、相机或真机控制已经标定。当前采集限制与设计方案见 [[RM65-B灵巧手仿真数据采集-RoboTwin2自动生成路线]]，不要把早期 RH56DFX 假设直接当作当前四代手模型的确认结论。

| 项目 | 当前依据 | 对适配的影响 |
| --- | --- | --- |
| 双臂 | 用户确认 RM65-B，每臂 6 轴 | 双臂关节共 12 维，不能沿用 7+7 轴假设 |
| 双手 | 用户确认灵巧手；手册 BOM 为 RH56DFX，每手 6 个主动自由度、12 个运动关节 | 若实物与手册一致，双手控制共 12 维；被动关节不能当独立动作 |
| 相机 | 用户确认头部 D435 + 自加左右腕 D435 | 三视角 RGB 为共同输入；另存深度用于标定/扩展 |
| GPU | 用户确认 8 张 RTX 3090 | 每卡 24GB；显存不能自动合为一张 192GB 卡 |
| 板载机 | 手册为 Xavier NX 8GB，实际是否升级待查 | 建议工作站推理，板载机负责相机、状态与执行 |
| 底盘/升降/头部 | 手册列有相关机构 | 第一阶段固定；后续增加动作维度需重新定义数据与任务 |

手册来源：用户提供《双臂机器人平台使用手册 V1.2》，本地路径 `/home/admin123/文档/xwechat_files/wxid_94x8tkwbvx3722_e6dc/msg/file/2026-09/双臂机器人平台使用手册V1.2.pdf`。核查了机械臂、灵巧手、板载机、相机和 BOM；未将手册原件或其中的登录信息复制进笔记库。

## 2. 基线优先级

| 优先级 | 模型 | 研究角色 | RM65-B 证据与适配边界 |
| --- | --- | --- | --- |
| 主模型 | LingBot-VLA 2.0 | 强 VLA，后续加 RL/语言约束 | 官方统一动作表示覆盖手部，但本次代码快照没有 RM65 专用机器人配置/硬件驱动 |
| 必做 | π0.5 | 强通用 VLA 对照 | 使用官方 openpi；需要三视角、24维状态动作与归一化适配 |
| 必做 | SmolVLA | 轻量 VLA、快速验证语言和数据链路 | 有 RM65-B 社区权重线索，但不能将模型卡当作独立真机成功率证据 |
| 必做 | ACT | 模仿学习与控制链路对照 | 睿尔曼官方有 RM65 单臂 ACT 示例；双臂双手需扩展；标准 ACT 不承担语言跟随比较 |
| 扩展 | LingBot-VLA 1.0 | 前代性能参照 | 数据和模型均有变化，1.0/2.0 对比不是单一机制消融 |
| 扩展 | GR00T N1.7 | 跨本体 VLA 对照 | 按官方 NEW_EMBODIMENT 建立模态配置并微调，不能直接套其他机器人的 embodiment tag |

选型依据：[LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)、[openpi](https://github.com/Physical-Intelligence/openpi)、[SmolVLA 文档](https://huggingface.co/docs/lerobot/en/smolvla)、[睿尔曼 ACT 示例](https://develop.realman-robotics.com/AI/developerGuide/embodiedruiEyeminimummodule/)、[LingBot-VLA 1.0](https://github.com/Robbyant/lingbot-vla)、[GR00T 自定义本体](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md)。

这是一份面向本机的投入优先级，不是跨榜单总排名。第一阶段不加入完整视频生成式 WAM，以免在本体接口尚未稳定时同时引入额外模型和训练变量；后续若论文问题涉及预测规划，再单独建立 WAM 对照。

## 3. 为什么主选 LingBot-VLA 2.0，又不能直接部署

论文中的 Realman 是 **Rs-02：双臂 14 维、夹爪 2 维、身体 1 维**，与本机 6+6 轴、双灵巧手不同。覆盖同品牌不等于本体一致。模型有 Qwen3-VL 主干、flow-matching 动作专家及未来几何/语义表示监督；应归为带预测表征学习的 VLA，不能据此宣称已具备联合视频动作生成或 RL 训练能力。

论文 GM-100 双臂实验实际是其中 **9 个任务、两个平台**，单策略联合训练。平均成功率如下，均为作者报告，不是本机复现：

| 平台 | GR00T N1.7 | π0.5 | LingBot 1.0 | LingBot 2.0 |
| --- | --- | --- | --- | --- |
| AgileX Cobot Magic | 17.8% | 32.2% | 30.0% | 34.4% |
| Galaxea R1 Pro | 5.6% | 8.9% | 15.6% | 15.6% |

来源：[论文 Table 1、Table 5](https://arxiv.org/html/2607.06403v1)。2.0 不是每个任务都领先，R1 Pro 总成功率与 1.0 持平。RoboTwin 发布示例联合使用 clean 和 randomized 训练数据，不能与只用 clean 示范的 clean2random 结果直接横比。

## 4. 数据与动作接口建议

建议保存原始语义清晰的 24 维状态/动作：

```text
[左臂6, 左手6, 右臂6, 右手6]
images = [头部RGB, 左腕RGB, 右腕RGB]
language = 当前任务指令
```

这是手与手册一致时的设计方案，尚未通过硬件读数验证。不同模型用各自转换器，保留相同原始数据、任务与控制权限。

LingBot 的 canonical 表示为 55 维；应将左右臂拼到 arm.position、左右手拼到 hand.position，按官方转换器补齐并屏蔽无效维度。不要将 24 维直接覆盖全部输出头，不要把手写进两维夹爪字段，也不要假设每臂都要补一个第七关节。代码按特征块拼接与 padding，必须验证训练和推理顺序一致。

建议臂关节先采用相对当前观测的目标偏移，手采用标定后的绝对目标。动作块内每个未来臂目标均相对同一个当前状态，不能误当连续积分增量。关节弧度/角度、手指开合方向、伺服量程、左右顺序须逐项验证。手部绝对表示是本项目建议，不是模型已验证结论。

输入先统一为三路 RGB。虽然 D435 有深度，发布模型叫 native-depth，也不能推断原始深度已接入当前策略输入；应按实际预处理路径核查，深度扩展作为独立实验。

使用通用预训练 `robbyant/lingbot-vla-v2-6b` 微调，重新计算本机数据归一化统计；不直接复用 RoboTwin 权重的动作约定和统计量。

来源：[数据适配指南](https://github.com/Robbyant/lingbot-vla-v2/blob/main/lingbotvla/data/vla_data/README.md)、[示例映射](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/robot_configs/robotwin.yaml)、[转换实现](https://github.com/Robbyant/lingbot-vla-v2/blob/main/lingbotvla/data/vla_data/utils.py)。

## 5. 8×3090 的可行路线

ACT、SmolVLA 先做单卡小批量微调和数据闭环；余卡用于多种子或并行实验。π0.5 先测低 batch 的 LoRA：openpi 给出通用单卡 LoRA >22.5GB、全量 >70GB 的估计，三视角24维配置仍需实测；支持的 FSDP 可降低单卡占用，当前官方脚本不支持多节点训练，若8卡分散在多台机器需另行设计。

LingBot 2.0 应先做多卡显存冒烟，不能复制大卡配置。核查的真实机器人 YAML 默认 micro_batch_size=32、未启用 gradient checkpointing、enable_full_shard=false、Muon、动作专家 FP32。建议从 micro batch 1、梯度检查点、受支持的参数/优化器分片配置探索；教师模块开关和精度调整均需记录，关闭监督项的结果不得冒充完整方法。

官方 RoboTwin 发布验证采用 FP32，并明确 BF16 成功率可能明显不同。约6B参数若全部 FP32，权重本身已约24GB十进制字节，尚未计入激活；单卡24GB不能保证容纳正式评测配置。官方约130ms推理是在4090D、10步去噪上测得，不是3090速度，也不是机械臂伺服频率。

部署建议：工作站策略服务器 → 板载机执行桥。测端到端 p50/p95 延迟、动作队列滞后、相机时间偏差与实际执行频率，再确定动作块长度和重规划频率。不要为了安装新模型环境替换板载机器人原有 ROS/SDK 环境。

来源：[openpi 资源说明](https://github.com/Physical-Intelligence/openpi#requirements)、[LingBot 真实机器人配置](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/vla/real_robot/real_robot.yaml)、[训练说明](https://github.com/Robbyant/lingbot-vla-v2/blob/main/configs/vla/Training_Config.md)、[发布验证和推理说明](https://github.com/Robbyant/lingbot-vla-v2#deployment)。

## 6. 已找到的 RM65 社区资产及代码陷阱

[UCLA realman-env](https://github.com/UCLA-Robot-Intelligence-Lab/realman-env) 描述 RM65-B 双臂、6自由度手、D435，包含24维桥接接口，值得借鉴硬件层。

但检查其 `train_pi0_5.py`：实际导入 `PI0Config`、`PI0Policy`，没有显式的 π0.5 预训练 checkpoint 加载。部署循环也未见完整语言预处理链。因此脚本文件名和 README 支持列表不足以证明正确实现 π0.5/VLA 部署。建议参考其硬件桥，模型训练和加载以官方实现为准。

固定代码证据：[train_pi0_5.py](https://github.com/UCLA-Robot-Intelligence-Lab/realman-env/blob/55ae5f768ac95a25a959b611d1f20e5c2d6a03ad/src/rmc_aida_l_ros2-develop/train/vla/train_pi0_5.py)。

[SmolVLA RM65-B 社区权重](https://huggingface.co/JayCao99/smolvla-rm65b-sort-v0.0) 当前模型卡列训练检查点与 loss，但未给出足以复现的统一真机成功率对照；可以作为线索，不能直接认定适配本机双手和相机标定。

审查快照：LingBot `bc643d74a0127fab8788da993b261d4d64101138`；realman-env `55ae5f768ac95a25a959b611d1f20e5c2d6a03ad`。源码下载在临时目录，未执行这些第三方训练或控制脚本。

## 7. 实验顺序与验收

1. **接口验收**：核实手型号、关节单位、相机序号/标定/时间戳、SDK版本、手控制反馈与频率；对24维映射、归一化与反归一化做离线往返检查。真实执行前确认过期动作丢弃和停止路径。
2. **采集与闭环**：固定底盘等额外机构，用2个简单任务验证数据回放，先跑ACT/SmolVLA；三路图像、状态、动作、语言必须可追溯到同一时间轴。
3. **共同基线**：扩展到4–6个任务，同一示范集、同一任务划分比较LingBot2、π0.5、SmolVLA；ACT只比较基础操作。每任务50–100条示范作为起步预算，不是效果保证。先少量试验，再按方差扩大评测。
4. **语言约束**：同一场景仅改变目标物、左右手分工、先后顺序或禁止接触对象。划分未见措辞、物体和场景，避免一条指令永远对应一个固定场景。分别报告完成率、约束违背率、干预率和延迟，不能只报总成功率。
5. **RL研究**：从同一LingBot2 SFT检查点出发，比较 SFT、SFT+RL、SFT+语言约束、SFT+RL+语言约束。冻结/训练哪些模块、奖励、交互预算和随机种子保持可比；先离线或经过验证的仿真流程，再决定真机后训练。

> [!todo] 尚未验证
> 手实物是否RH56DFX；8卡是否同机及互联；现有遥操作/示范数据；训练显存与吞吐；3090推理时延；本机任何模型的成功率。本次完成的是文献与静态代码调研，没有训练、硬件连接或真机运行。

关联：[[笔记库使用指南]]。
