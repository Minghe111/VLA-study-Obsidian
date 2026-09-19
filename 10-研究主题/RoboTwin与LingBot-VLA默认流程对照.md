---
title: RoboTwin与LingBot-VLA默认流程对照
tags: [研究, VLA, 框架, 默认配置]
updated: 2026-09-19
status: 源码静态核查，未运行训练或评测
---

# RoboTwin与LingBot-VLA默认流程对照

> [!important] 默认设置的范围
> 本文以 LingBot `configs/vla/robotwin/robotwin.yaml` 后训练示例和配套 RoboTwin 评测 launcher 为主线。RoboTwin 本身不是一个固定策略，没有统一的“全量微调默认值”。类默认值、真机示例和 RoboTwin benchmark launcher 的默认值不同处，明确标注入口。

## 基本问题速查

| 问题 | 当前 RoboTwin | LingBot-VLA v2 的 RoboTwin 默认流程 |
| --- | --- | --- |
| 框架职责 | SAPIEN 仿真、脚本专家采集、环境随机化、任务成功判定；当前接入 XPolicyLab 策略层 | 模型、数据适配、监督后训练、动作推理服务 |
| 训练数据格式 | 当前原生采集输出 XPolicyLab v1.0 HDF5，另有可视化视频 | 示例转换流程产出 LeRobot v2.1；加载器也支持 v3.0 |
| 如何指定数据 | 采集任务配置与数据路径 | `data_name: multi`，文本清单列出机器人配置名＋LeRobot 路径 |
| 模型条件输入 | 环境提供图像、状态和任务语言 | 当前头部＋左右腕 RGB、当前关节/夹爪状态、任务文本 |
| 监督目标 | 示范的后续状态，可保存关节与末端位姿 | 默认 50 步连续动作块；同时有当前/未来教师特征监督 |
| 动作表示 | `take_action` 默认 `qpos`，也有 EE 接口 | **绝对关节角＋绝对夹爪位置**；默认不监督末端位姿 |
| 实际动作维数 | 默认 aloha-agilex 每臂 6 关节＋1 夹爪，共 14 | 原始/执行 14 维，网络统一填充到 55 维；有效分量才计动作损失 |
| 是否只训练专家 | 取决于所选策略 | **主策略全量微调**：视觉与语言骨干、动作专家及投影可训练；教师目标生成不反传 |
| 是否默认 RL | 默认专家示范生成与评测不等于 RL | 否；使用监督 flow matching 和特征蒸馏 |
| 动作生成 | 环境消费策略动作并执行控制 | 从噪声出发，默认 10 步 flow-matching 采样产生动作块 |
| 推理同步性 | 取决于策略适配器 | 此配套客户端同步请求－响应，执行动作块后再请求 |
| 是否异步 RTC | 不是统一框架保证 | 默认循环中没有边执行旧动作边提前计算新动作块的 RTC |
| 默认任务集 | 常用 50 个双臂操作任务，详见任务笔记 | launcher 默认全 50 任务，每任务评测 100 episode |
| 默认执行场景 | demo_clean 使用 aloha-agilex、三路 D435 RGB | launcher 默认 demo_clean；训练示例混合 clean 和 randomized |
| 默认评测精度 | 随所用策略 | **launcher 为 FP32**；直接启动 policy Python CLI 则默认 BF16，二者不可混写 |

## 1. 数据到底长什么样

LingBot 的直接训练入口是 LeRobot 数据目录，而不是任意 `.hdf5`。示例先整理 RoboTwin 轨迹，再转换；LeRobot 用表格数值数据、视频和 episode/task 元数据表达轨迹，通常对应 Parquet、MP4 与 meta 目录。v2.1 与 v3.0 的具体文件分片布局不同，当前 loader 负责读取，不需要先合并成一个大文件。

本仓库 `assets/training_data/robotwin.txt` 有 clean 和 randomized 的任务路径占位符。需要换成实际数据目录；文件名中的 `500`、`1000` 不能当作已核验的 episode 数量，也不能据清单算出实际训练样本量。个别条目名称含 piper，默认评测本体仍以任务配置为准。

机器人映射为：

```text
原始 state/action（14维）:
[左臂关节0..5, 左夹爪, 右臂关节0..5, 右夹爪]

observation.images.cam_high       → camera_top
observation.images.cam_left_wrist → camera_wrist_left
observation.images.cam_right_wrist→ camera_wrist_right
```

两个 action 字段均 `subtract_state: False`，因此学习绝对目标，不是当前状态的增量。`end.position: 14` 出现在训练配置中，但机器人 YAML 没有映射到这一组，不能据此说默认同时预测有效末端位姿。

归一化为 `bounds_99_woclip`：使用统计文件中的 1%/99% 分位数线性变换，不截断超界值。统计文件默认 `assets/norm_stats/robotwin.json`；部署时逆变换恢复机器人单位。

## 2. 一个训练样本的输入和输出

策略条件是当前 RGB、任务文本和当前状态；训练另从同一 episode 读取后续动作序列及未来图像，未来图像用于教师监督，不是要求部署时从未来获取相机画面。

```text
当前 RGB 三视角 + 当前状态 + 语言指令
                  ↓ VLM + 动作专家
训练：预测加噪动作的 flow 速度，与目标速度计算 L1 损失
部署：10 次积分更新 → 50×55 动作块 → 逆映射 → 50×14 执行动作
```

这里的 flow 速度是生成过程中的数学速度，不是机器人关节速度命令。模型内部输出 flow 速度，最终解码的控制目标仍是绝对关节位置。

动作窗口默认 50，数据 loader 按数据集 `fps` 生成时间偏移，未来图像取到窗口末端附近。因此“50 步”不能一律解释为固定秒数，也不能将推理频率、数据 fps 和物理仿真步频混为一谈。

动作主损失是 `L1_fm`，只对 mask 标记的有效维度聚合。默认还有当前/未来深度与视频教师特征对齐；几何/视频对齐权重配置为 0.004，MoE 序列辅助项 1e-3、z-loss 1e-4。它不是预测语言 token 的普通文本 SFT，也不是使用成功奖励的 RL。

## 3. 默认到底训练哪些参数

`freeze_vit=False`、`freeze_vision_encoder=False`、`train_expert_only=False`，且 `train_state_proj=True`。所以主策略属于全量微调，不是 LoRA，也不是只微调动作专家。Depth/MoGe 与 DINO 生成监督目标时使用 `no_grad`，应与可训练策略区分。

MoE 每 token 路由到 4/32 个专家，是稀疏激活；不同 token 可命中不同专家，不代表固定只训练 4 个专家，其余永远冻结。

| 后训练参数 | README 指向的 YAML 默认 |
| --- | --- |
| 优化器 | Muon（虽然通用说明提到 AdamW 默认，但此 YAML 显式覆盖） |
| 学习率 | 1e-4，cosine，最低 5e-5 |
| 训练步数 | max_steps=50000；不要把 num_train_epochs=29000 当作唯一停止条件 |
| micro / global batch | 32 / 1024；YAML 注释按 32 GPU 描述，不是按你的 8 GPU 定制 |
| 分布式 | FSDP2，enable_full_shard=false，module_fsdp_enable=true |
| 梯度检查点 | false |
| compile | true |
| checkpoint | 每 10000 step；推理读取导出的 hf_ckpt |

这些默认值不是“8×3090 保证跑得动”的证明，本次不擅自替你调参。

## 4. 同步与异步必须看控制循环

实际路径：`get_obs → WebSocket send → recv 等待 → 取返回动作块 → 循环 take_action → 再 get_obs`。默认 `chunk_ret=True`、`use_length=50`，成功或达到环境限制时可提前结束。

因此块间根据新观测闭环，块内连续执行预测动作；不应称为每一步都重新看图推理。WebSocket 服务端用 asyncio 或 launcher 同时跑多个任务，不会自动让单个机器人回路变成异步推理。

默认 launcher 是 1 GPU、每 GPU 1 服务、起始端口 9330、全 50 任务、FP32、compile=True。README 的 8 GPU 命令是显式示例覆盖。直接启动 `deploy/lingbot_vla_v2_policy.py` 的 CLI 默认端口 8006、BF16，属于另一入口。

本版本 launcher 先赋值 `enable_video=True`，随后又设为 False，所以实际默认关闭评测视频；以生效代码为准。

## 5. 语言怎样进入主干，是否先拆子任务

默认不会先生成“子任务1、子任务2”的文本计划。客户端每轮取 `TASK_ENV.get_instruction()`，以 `task` 字符串发送；`FeatureTransform` 将其包装成 `prompt=[item["task"]]`。

V2 配置默认 `use_qwen3_chat_template=True`。`prepare_language` 把指令放入单条 user 消息，应用 Qwen chat template，且 `add_generation_prompt=False`；随后 tokenizer 输出 `input_ids` 和 `attention_mask`。RoboTwin YAML 将 `tokenizer_max_length` 设为 72，右侧 padding，并启用 truncation。因此过长指令可能截断；72 是包含模板 token 的文本长度预算，不是 72 个汉字，也不包含另行构造的视觉 token。

`embed_prefix` 将三路当前图像的视觉 embedding（带视觉边界 token）与语言 token embedding 拼入多模态前缀，并加入感知对齐 query。机器人状态经 `state_proj` 进入动作侧 suffix，与加噪动作和生成时间嵌入一起参与动作生成。没有先由 VLM 输出一段文字计划、再把文字计划传给另一个执行器的默认步骤。

概念流程：当前三视角 RGB＋任务指令 → VLM 条件表示 → 动作专家结合当前状态 → 50步动作块 → 执行 → 新观测＋任务指令再次推理。

默认 RoboTwin client 每轮发送当前观测，不附带完整历史视频、历史聊天或显式子任务进度表。采样时的 prefix KV cache 主要复用本次观测条件下的计算，不能据此认为具有跨整个 episode 的长时程记忆。

因此更准确的说法是“直接在任务语言条件下生成局部动作块”，而不是“显式语言层级规划”，也不是“把整段长任务视频交给 VLM 一次推理到底”。长任务可能通过反复观测推进，但代码结构本身不能证明任意多阶段指令都能完成。

补充依据：`configuration_lingbot_vla.py` 的 V2 默认值、`transform.py:prepare_language`、`utils.py` 的 task→prompt 映射、`modeling_lingbot_vla_v2.py:embed_prefix`、父类 `modeling_lingbot_vla.py:embed_suffix`，以及上述评测 client。

## 6. 版本兼容与复现边界

- 当前 RoboTwin commit：`6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`，已采用 `scripts/`、`env_cfg/` 和 XPolicyLab。
- LingBot 附带指南固定 RoboTwin：`13c3c47ff4312dd62484bcd51be034af55c062d1`，使用旧 `script/`、`task_config/`、`policy/pi0` 路径。
- 当前采集 HDF5 的 schema 已改变；不能只修改文件路径就认定旧转换脚本兼容。
- LingBot launcher 会复制评测客户端并修补路径，实际运行会修改目标 checkout。本文仅静态读代码，没有执行该脚本。
- 作者发布分数采用 FP32，文档报告其栈中一个 FP32 server 加仿真约需 32 GB。3090 单卡 24 GiB 的默认全流程不能据此保证可跑；低精度仅作为可选路径提及，不与原分数混比。

关联：[[LingBot-VLA 2.0项目概述]] · [[RoboTwin 2.0框架与默认任务]] · [[服务器172.17.27.166-robotwin]]

## 代码依据

- [configs/vla/robotwin/robotwin.yaml](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/configs/vla/robotwin/robotwin.yaml)
- [configs/robot_configs/robotwin.yaml](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/configs/robot_configs/robotwin.yaml)
- [lingbotvla/data/vla_data/README.md](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/lingbotvla/data/vla_data/README.md)
- [lingbotvla/data/vla_data/base_dataset.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/lingbotvla/data/vla_data/base_dataset.py)
- [lingbotvla/data/vla_data/transform.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/lingbotvla/data/vla_data/transform.py)
- [tasks/vla/train_lingbotvla.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/tasks/vla/train_lingbotvla.py)
- [lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py)
- [lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py)
- [deploy/lingbot_vla_v2_policy.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/deploy/lingbot_vla_v2_policy.py)
- [deploy/websocket_client_policy.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/deploy/websocket_client_policy.py)
- [experiment/robotwin/eval_policy_client_lingbotvla.py](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/experiment/robotwin/eval_policy_client_lingbotvla.py)
- [experiment/robotwin/start_robotwin_infer_and_eval.sh](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/experiment/robotwin/start_robotwin_infer_and_eval.sh)
- [experiment/robotwin/README.md](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/experiment/robotwin/README.md)

补充：论文的数据制作阶段有离线子任务切分与语言标注，和这里讨论的在线推理不同。详见 [[LingBot-VLA2与π0、SmolVLA架构比较]]。
