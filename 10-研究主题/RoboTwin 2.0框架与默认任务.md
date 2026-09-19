---
title: RoboTwin 2.0框架与默认任务
tags: [框架, 仿真, RoboTwin]
updated: 2026-09-19
status: 当前源码核对
---

# RoboTwin 2.0框架与默认任务

RoboTwin 是双臂操作仿真与评测平台，任务脚本、机器人模型、相机、规划器、示范采集和成功判据在这里；VLA 的网络训练交给具体策略。当前仓库通过 XPolicyLab 管理策略接入，因此不能给整个 RoboTwin 指定一种数据归一化、微调方式或统一动作头。

当前核查 commit：`6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`；本地路径 `/home/admin123/liminghe/vla-benchmark/RoboTwin`。远端记录见 [[服务器172.17.27.166-robotwin]]。

## 默认示范采集与观测

以 `env_cfg/task_config/demo_clean.yml` 为准：本体 aloha-agilex；每次配置 episode_num=50；头部与双腕 D435，采集 RGB、qpos、endpose；不采集 depth、pointcloud、segmentation。D435 是相机配置型号，不代表默认网络一定使用深度图。

clean 不启用背景、桌面杂物、光照和桌高随机化，评测指令设为 seen。randomized 同样是 50 条采集设置，启用随机背景、杂物、桌高与光照，指令为 unseen。采集 episode_num 与评测每任务 100 episode 是两个参数，不能混同。

专家示范由任务脚本和运动规划/控制生成，非 VLA 自己生成标签；场景、对象和语言构成任务变化。

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