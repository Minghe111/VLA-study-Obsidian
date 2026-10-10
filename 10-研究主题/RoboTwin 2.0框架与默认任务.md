---
title: RoboTwin 2.0框架与默认任务
tags: [框架, 仿真, RoboTwin]
updated: 2026-10-10
status: 当前源码核对
---

# RoboTwin 2.0框架与默认任务

RoboTwin 是双臂操作仿真与评测平台，任务脚本、机器人模型、相机、规划器、示范采集和成功判据在这里；VLA 的网络训练交给具体策略。当前仓库通过 XPolicyLab 管理策略接入，因此不能给整个 RoboTwin 指定一种数据归一化、微调方式或统一动作头。

当前核查 commit：`6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`；本地路径 `/home/admin123/liminghe/vla-benchmark/RoboTwin`。远端记录见 [[服务器172.17.27.166-robotwin]]。

> [!info] 2026-09-24 补充：默认流程与 RM65-B 灵巧手分开阅读
> 本页下文保留上述旧 commit 的默认本体与接口快照；本机 RM65-B 当前代码已到 `c826f28`。新本体支持 24 维控制，但专家抓取采集尚未开启，且下一帧测量状态不能自动视为实际下发指令。最新核查与采集设计见 [[RM65-B灵巧手仿真数据采集-RoboTwin2自动生成路线]]。

## 默认示范采集与观测

以 `env_cfg/task_config/demo_clean.yml` 为准：本体 aloha-agilex；每次配置 episode_num=50；头部与双腕 D435，采集 RGB、qpos、endpose；不采集 depth、pointcloud、segmentation。D435 是相机配置型号，不代表默认网络一定使用深度图。

clean 不启用背景、桌面杂物、光照和桌高随机化，评测指令设为 seen。randomized 同样是 50 条采集设置，启用随机背景、杂物、桌高与光照，指令为 unseen。采集 episode_num 与评测每任务 100 episode 是两个参数，不能混同。

专家示范由任务脚本和运动规划/控制生成，非 VLA 自己生成标签；场景、对象和语言构成任务变化。

## 真机采集的复用边界（2026-10-10）

官方 `collect_data.sh` / `scripts/collect_data.py` 是仿真示范采集入口：先在任务环境中探索成功种子和规划路径，再按种子重建场景、回放并导出。种子重置、物体位姿读取及任务成功检查依赖仿真。它不能通过切换 embodiment 或指定真机 IP 直接变成实体机器人采集器。[官方采集说明](https://robotwin-platform.github.io/doc/usage/collect-data.html) · [官方入口](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/scripts/collect_data.py)

官方真机实验与公开真机采集工具需区分：论文 `arXiv:2506.18088v2` §4.4 报告在 COBOT-Magic 双臂平台上使用 RDT 评测四项任务，对比10条真实示范、10条真实示范加1000条随机化仿真轨迹，以及仅1000条仿真轨迹。这是作者报告的真机迁移实验，不等于已经发布通用的真机自动示范采集器。2026-10-10 在线核对官方 main 的采集入口与文档，未发现可直接完成真实感知、抓取生成、实体控制、同步记录及场景重置的完整真机采集方案；此结论限定于已核对的公开主仓库与文档，不覆盖所有分支或其他平台工具，也不证明 RM65-B 真机可用。[论文 §4.4](https://arxiv.org/html/2506.18088v2#S4.SS4)

可复用的是任务/技能组织、规划方法、episode 管理、数据结构、图像编码及 LeRobot 转换方法；真机需要另行接入机械臂/手 SDK 或 ROS 驱动、实际相机/状态反馈、同步采样与命令记录。物体位姿须来自真实感知或标定；成功判据须来自实际传感器或人工标注；场景重置与物理随机化须通过真实摆放/设施完成。复用规划器的前提是实际坐标、几何、控制接口和执行约束已经匹配；仿真路径通过不等于真机执行通过。此为基于代码依赖的工程分析，未实现真机自动采集。

对当前 RealMan 数据接口，真机采集应分别保存测量状态、实际发送命令和各自时间戳；若沿用汽水瓶数据的24维契约，应固定 `[左臂6 rad, 左手6 SDK, 右臂6 rad, 右手6 SDK]` 的顺序，并说明手反馈的实际来源/单位。当前14维协同手任务接口与24维独立手接口需显式适配。通用 `create_xpolicylab_hdf5` 使用下一帧测量值构造 action 的行为，不能直接作为“真实发送命令”的记录器；真机数据导出需按选择的动作表示重新核对标签与时序。

推荐先用遥操作或已验证控制脚本完成一条真实 episode：真实相机与机器人状态/命令→按时间对齐记录→成功标注→统一导出→原始记录读回核对，再逐步迁移自动规划任务。记录层可以与仿真共用训练接口，实体执行层需要独立验证。

本地核查 HEAD `931ce21f4f0e63feab661f349634d5a25bd92681`；`integrations/realman_rm65b_gen4/hardware_contract.py` 明确为无硬件 I/O 的驱动字段转换，既有 RealMan 组件报告属于仿真/静态映射证据。本次没有连接、采集或操控实体机器人，也没有重新验证真机标定。关联 [[RM65-B灵巧手仿真数据采集-RoboTwin2自动生成路线]] · [[2026-10-09-RealMan从161同步与组件验收]]。

## 2026-10-10 域随机化范围核查

证据类型：官方在线配置/文档与本地源码静态检查；本地 HEAD 为 `931ce21f4f0e63feab661f349634d5a25bd92681`。本节补充不改变上文历史版本快照；未重新采集数据、训练或运行闭环评测。

域随机化主要在仿真示范采集时生成不同场景，策略训练再消费这些轨迹；不能把它等同于训练端自动启用图像增强。官方当前 `demo_randomized.yml` 与本地同名配置的相关字段一致：

| 项目 | 默认 randomized 设置与代码边界 |
| --- | --- |
| 背景纹理 | `random_background: true`，墙面与桌面分别采样纹理；代码按采集/评测模式使用 seen/unseen 纹理目录。 |
| 桌面杂物 | `cluttered_table: true`，随机选择干扰物与摆放位置，受禁止区域与放置尝试次数约束。 |
| 光照 | `random_light: true`，随机方向光与点光源的 RGB 颜色/强度；`crazy_random_light_rate: 0.02` 按 episode 抽取动态扰动模式，在渲染更新时重新扰动光色及环境光。 |
| 桌高 | `random_table_height: 0.03`；本地 `_base_task.py` 实际为 `uniform(-0.03, 0) + table_height_bias`，即相对配置基准向下 0～3 cm，不是 ±3 cm。 |
| 头部相机位置 | 支持 `random_head_camera_dis`，但默认值为 `0`，未开启；非零时采样三维方向和位移长度，不应扩大解读为默认扰动所有相机内外参。 |
| 保留干净状态 | `clean_background_rate: 0.02` 在代码中分别用于墙面、桌面纹理回退，以及跳过杂物生成；不能把它解释成严格 2% 的完整 clean episode。 |

任务物体的位姿与实例变化由任务脚本决定，并非上述配置独有：`pick_dual_bottles` 在 clean 下也会随机位置/姿态，但实例固定为瓶子 13、16；`pick_diverse_bottles` 则从 20 个实例中分别采样。不能说所有任务都会随机改变物体大小、形状或模型。

语言描述的多样性属于语言泛化维度；`language_num: 100` 与 `eval_instruction: unseen` 不属于物理参数随机化，也不表示 100 条示范。默认 clean 配置的评测指令为 seen。比较策略时应分别记录场景条件与指令划分。

默认配置未列出质量、摩擦、关节动力学、控制延迟或传感器噪声随机化，不应将这些算作默认已启用能力。官方文档中的 `random_embodiment` 标为实验性且未完整支持，当前默认 randomized 配置没有开启该项。

### 采集随机化与训练增强的边界（2026-10-10）

RoboTwin 的 `domain_randomization` 配置在环境初始化、场景构建及渲染阶段生效；已有 clean HDF5/LeRobot 数据不会因为修改该 YAML 而自动变成 randomized 数据。该配置也可作用于闭环评测环境，评测变化并不会增加训练示范。

训练端可以对已有图像做在线增强：亮度、对比度、饱和度、色调、锐度等通常保留原动作标签；其覆盖的是视觉外观扰动。它们不能完整替代仿真的光源变化、阴影和反射。桌高、物体位姿/实例、实际杂物布局或动力学变化需要生成对应的观测与有效动作轨迹；普通图像增强无法补出这些示范。相机三维位移涉及视差与遮挡，也不能简单等同于二维图像平移/旋转。

本地 `XPolicyLab/policy/SmolVLA/install.sh` 默认 `LEROBOT_REF=v0.4.4`；`train.sh` 未显式传入图像增强开关。官方 LeRobot v0.4.4 的 `ImageTransformsConfig.enable` 默认为 false，可在直接调用 `lerobot-train` 时传 `--dataset.image_transforms.enable=true`。该版本预设候选包含亮度、对比度、饱和度、色调、锐度和二维仿射，默认每次最多抽取三项；不能把此开关理解成开启 RoboTwin 的全部仿真随机化。此处核对安装脚本与官方固定版本源码，未确认当前 Conda 实际安装版本或运行训练。

选择增强时需保持监督语义：按颜色识别/排序任务应限制颜色扰动；裁剪、遮挡和翻转需检查目标可见性、左右臂与语言方向语义。为了单独分析增强效果，可比较 clean、clean 加图像增强、randomized、randomized 加图像增强四组，固定示范预算和评测指令划分。此为实验建议，尚无本机效果证据。

### 实际 RealMan 汽水瓶50条数据（2026-10-10 补充）

用户追问的已采集汽水瓶数据对应 `realman_rgb50_full50`，不是此前仅规划的 SmolVLA `pick_dual_bottles/demo_clean` 公共数据。此次读回本地 `data50_v1/native_index.json`、`source_training_files/scene_info.json` 与传输凭据：50条全部没有墙面/桌面随机纹理，全部没有杂物，场景文件SHA256与传输凭据一致。原始数据manifest SHA256为 `2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80`。

索引中的任务覆盖为雪碧20、可口可乐20、芬达10；左臂25、右臂25；直立20、横放15、斜放15，50条使用同一任务指令。这些是对象/任务姿态覆盖，不能据此称为开启完整的背景、光照、桌高、相机域随机化。本地当前 RealMan 配置也关闭光照、桌高和相机扰动，但未读回这50条的原始采集配置：原始166服务器连接返回 `No route to host`，因此后三项的历史实际状态仍待原配置核对，不用当前配置替代历史证据。

本次属于本地镜像元数据读回及哈希核对，未重新读取HDF图像、执行轨迹、验证物理成功或训练效果。记录见 [[2026-10-10-RealMan汽水瓶50条域随机化核查.json]]；对应训练历史快照见 [[2026-10-10-161-RealManπ0.5训练状态核查]]。

来源（在线核查日期 2026-10-10）：
- [官方配置说明](https://robotwin-platform.github.io/doc/usage/configurations.html)
- [官方 demo_randomized.yml](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/env_cfg/task_config/demo_randomized.yml)
- [官方 Base_Task](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py)
- [LeRobot v0.4.4 图像增强实现](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/transforms.py)
- [LeRobot v0.4.4 数据集配置](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/configs/default.py)
- 本地静态核查：`envs/_base_task.py`、`envs/camera/camera.py`、`envs/pick_dual_bottles.py`、`envs/pick_diverse_bottles.py`；对应上述本地 HEAD，未证明与官方 main 所有代码一致。

## 当前原生数据格式

`envs/utils/pkl2hdf5.py:create_xpolicylab_hdf5` 将逐帧缓存合并为 `data/episode_0000000.hdf5` 等文件，写入 XPolicyLab `data_format_version=v1.0`，包含：

```text
instructions                 JSON字符串形式的指令列表
additional_info/frequency    频率元数据
state/left_arm_joint_states  当前左臂关节
state/right_arm_joint_states 当前右臂关节
state/left_ee_joint_states    当前左夹爪
state/right_ee_joint_states   当前右夹爪
action/...                   对应的下一帧目标
state|action/*_ee_poses       若采集endpose则包含末端位姿
vision/<camera>/colors       编码RGB字节，并保存shape及可用相机参数
```

构造配对的代码明确使用 `state=values[:-1]`、`action=values[1:]`；相机观测也去掉末帧。也就是说，默认监督对来自相邻采样时刻，不是动作和状态不加区分地复制同一帧。另输出用于查看的 episode 视频。

`save_freq=15` 同时参与采集间隔及输出 frequency 字段；本次未测实际时序，不将其直接断言为真机控制 15 Hz。制作训练集时应核对导出时间语义。

LingBot 的公开示例消费 LeRobot 数据，不直接读这个新 HDF5 schema；旧版转换指南与当前采集格式必须分别看待。

## 默认动作执行

`take_action(action, action_type='qpos')` 默认接收左右臂关节目标与夹爪位置，将目标与当前状态组成路径，使用 TOPP 等步骤处理再推进仿真。也支持 `ee` 模式，但不是本文默认主线。

数据包含末端位姿，不表示策略必须预测末端位姿；采集格式与策略输出是两层定义。LingBot 的 RoboTwin 配置实际采用 qpos，详见 [[RoboTwin与LingBot-VLA默认流程对照]]。

## 50项任务

下列为 LingBot 配套 RoboTwin launcher 的完整任务 ID，涵盖取放、双臂交接、堆叠、排序、工具操作、开合与按钮动作。各任务具体成功条件以对应 `envs/<task>.py` 为准，不设一个通用距离阈值。

- `lift_pot`
- `hanging_mug`
- `stack_bowls_three`
- `scan_object`
- `handover_block`
- `click_bell`
- `put_object_cabinet`
- `open_microwave`
- `stack_blocks_three`
- `place_shoe`
- `adjust_bottle`
- `beat_block_hammer`
- `blocks_ranking_rgb`
- `blocks_ranking_size`
- `click_alarmclock`
- `dump_bin_bigbin`
- `grab_roller`
- `handover_mic`
- `move_can_pot`
- `move_pillbottle_pad`
- `move_playingcard_away`
- `place_cans_plasticbox`
- `place_container_plate`
- `place_dual_shoes`
- `place_empty_cup`
- `place_fan`
- `place_mouse_pad`
- `place_object_basket`
- `place_object_scale`
- `place_object_stand`
- `place_phone_stand`
- `move_stapler_pad`
- `open_laptop`
- `pick_diverse_bottles`
- `pick_dual_bottles`
- `place_a2b_left`
- `place_a2b_right`
- `place_bread_basket`
- `place_bread_skillet`
- `place_burger_fries`
- `place_can_basket`
- `press_stapler`
- `rotate_qrcode`
- `shake_bottle_horizontally`
- `shake_bottle`
- `stack_blocks_two`
- `stack_bowls_two`
- `stamp_seal`
- `turn_switch`
- `put_bottles_dustbin`

默认评测给定 clean 场景，按任务成功条件统计 success rate；50 步动作块不是 episode 长度，episode 上限另由任务控制。LingBot 自身没有独立的一套物理仿真任务，默认评测复用这 50 项。

## 当前接入与旧版接入

当前入口 `scripts/eval_policy.sh` 转到 XPolicyLab 评测、任务调度或策略服务。LingBot 自带的评测集成仍固定旧 RoboTwin revision；任务名字相同，不等于 CLI、数据 schema 和 Python 接口相同。本次未运行跨版本适配。

关联：[[LingBot-VLA 2.0项目概述]] · [[RoboTwin与LingBot-VLA默认流程对照]] · [[具身模型训练与仿真实验平台选型]]

## 固定版本依据

- [env_cfg/task_config/demo_clean.yml](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/env_cfg/task_config/demo_clean.yml)
- [env_cfg/task_config/demo_randomized.yml](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/env_cfg/task_config/demo_randomized.yml)
- [envs/_base_task.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/envs/_base_task.py)
- [envs/utils/pkl2hdf5.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/envs/utils/pkl2hdf5.py)
- [scripts/collect_data.py](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/scripts/collect_data.py)
- [scripts/eval_policy.sh](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/scripts/eval_policy.sh)
- [env_cfg/eval/all_tasks.yml](https://github.com/RoboTwin-Platform/RoboTwin/blob/6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755/env_cfg/eval/all_tasks.yml)
- [experiment/robotwin/start_robotwin_infer_and_eval.sh](https://github.com/Robbyant/lingbot-vla-v2/blob/ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8/experiment/robotwin/start_robotwin_infer_and_eval.sh)
