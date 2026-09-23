---
title: LingBot-VLA 2.0项目概述
tags: [模型, VLA, LingBot]
updated: 2026-09-23
status: 本地源码核对，未运行训练
---

# LingBot-VLA 2.0项目概述

LingBot-VLA 2.0 是 Robbyant（蚂蚁灵波）的视觉－语言－动作基础模型及训练、推理实现。它接收相机观测、语言任务和机器人状态，生成连续动作块。RoboTwin 则提供仿真、专家示范和成功率评测，两者承担不同工作。

本机评测命令：[[LingBot-VLA2本地RoboTwin评测-最简指令]]（先启动模型服务，再启动仿真客户端）。

## 本次核查对象

- 本地仓库：`/home/admin123/liminghe/vla-benchmark/lingbot-vla-v2`。
- commit：`ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8`。
- 默认解释范围：README 推荐的 RoboTwin 后训练入口 `configs/vla/robotwin/robotwin.yaml`，以及配套 `experiment/robotwin/start_robotwin_infer_and_eval.sh`。不把 Python 类的默认值或其他机器人配置混进该流程。
- 仓库存在未跟踪的环境修复材料、模型目录等，本次未修改，也不作为官方默认行为的证据。
- 本次 SSH 探查被服务器关闭连接，故以已找到的本地新拉取仓库为依据；未确认远端 LingBot checkout。

## 设计与预训练背景

公开模型以 `lingbot-vla-v2-6b` 发布，视觉语言骨干是 Qwen3-VL-4B-Instruct；动作侧使用包含稀疏 MoE 的动作专家。默认后训练配置在 36 层启用 MoE，每层 32 个路由专家、每 token 选择 4 个，并有共享专家。top-k 路由是计算分配机制，不能理解为“只微调固定的 4 个专家”。

统一状态/动作容量是 55 维：手臂关节 14、末端位姿 14、夹爪 2、灵巧手 12、腰 4、头 2、移动底盘 3、预留 4。具体机器人通过 YAML 映射有效分量，缺失分量填充并使用 mask；55 维不表示每台机器人都要执行 55 个控制量。

作者 README 报告约 6 万小时预训练数据，其中机器人轨迹约 5 万小时，覆盖 20 种机器人配置，另有约 1 万小时第一人称人类视频。这是作者报告的预训练规模，不是本地已下载数据，也不是默认 RoboTwin 微调数据量。

训练额外使用 LingBot-Depth/MoGe 和 DINO-Video 教师提供当前几何、未来几何与视频特征监督。模型学习未来特征有助于动作预测；默认部署仍输出动作，并不是部署时先生成未来 RGB 视频再调用独立规划器。

## 默认运行流程

1. RoboTwin 示范转换为 LeRobot 数据，配置机器人字段映射与归一化统计。
2. 从预训练 checkpoint 进行监督后训练：动作 flow-matching 损失加教师特征对齐与 MoE 路由辅助项。
3. 导出 `hf_ckpt`，在 LingBot 环境启动策略服务器。
4. RoboTwin 环境发送三路 RGB、当前状态和任务文本，等待动作块，执行后重新观测。

这条默认流程属于监督模仿学习/后训练，不含在线 RL 奖励优化。可选功能包括其他本体映射、冻结部分模块、其他优化器与推理精度，这里不展开。

## 与现有设备的关系

[[RM65-B双臂机器人]] 也是双六轴、三路相机，但其双灵巧手不同于默认 RoboTwin 的两维夹爪。不能直接把 14 维 RoboTwin 输出接到真机。已有 55 维统一表示提供适配空间，但仍需实际手部自由度映射、标定、控制接口和对应示范；当前没有验证 RM65-B 即插即用。

安装文档要求 Python 3.12、PyTorch 2.8.0，和 RoboTwin 环境分开管理。硬件信息见 [[服务器-172.17.27.166]]；默认 FP32 benchmark 的显存需求不能由“模型能加载”推断可跑。

关联：[[RoboTwin与LingBot-VLA默认流程对照]] · [[RoboTwin 2.0框架与默认任务]] · [[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]]

## 固定版本依据

- [README.md](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/README.md)
- [configs/vla/robotwin/robotwin.yaml](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/configs/vla/robotwin/robotwin.yaml)
- [configs/robot_configs/robotwin.yaml](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/configs/robot_configs/robotwin.yaml)
- [tasks/vla/train_lingbotvla.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/tasks/vla/train_lingbotvla.py)
架构详解：[[LingBot-VLA2与π0、SmolVLA架构比较]]。
