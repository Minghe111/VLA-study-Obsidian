---
title: RoboTwin 2.0框架与默认任务
tags: [框架, 仿真, RoboTwin]
updated: 2026-10-22
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

## 当前原生数据格式

`envs/utils/pkl2hdf5.py:create_xpolicylab_hdf5` 将逐帧缓存合并为 `data/episode_0000000.hdf5` 等文件，写入 XPolicyLab `data_format_version=v1.0`，包含：

```text
instructions                 JSON字符串形式的指令列表
additional_info/frequency    频率元数据
state/left_arm_joint_states  当前左臂关节
state/right_arm_joint_states 当前右臂关节
state/left_ee_joint_states    当前左夹爪
state/right_ee_joint_states   当前右夹爪
action/...                   下一采样帧的绝对目标值（关节角/夹爪开合量）
state|action/*_ee_poses       若采集endpose则包含末端位姿
vision/<camera>/colors       编码RGB字节，并保存shape及可用相机参数
```

构造配对的代码明确使用 `state=values[:-1]`、`action=values[1:]`；相机观测也去掉末帧。也就是说，默认监督对来自相邻采样时刻，不是动作和状态不加区分地复制同一帧。另输出用于查看的 episode 视频。

在此原生导出与默认qpos动作表示下，`action[t]` 是记录序列中下一采样帧的绝对臂关节角/夹爪开合量，没有做 `values[1:] - values[:-1]` 的差分。例：某关节两次记录分别为0.50、0.55 rad，则该训练对为state=0.50、action=0.55，增量0.05不是这里保存的action。下一采样帧也不必等于下一物理仿真步；策略训练若另做归一化、差分或动作块构造，应依据具体策略转换器核对。2026-10-14补充为源码解释，未运行新采集。

> [!info] 2026-10-14 复核：字段名称不等于实时测量值
> 在线核对官方 main 的 `Base_Task.get_obs` → `Robot.get_left/right_arm_jointState`：常规本体的臂关节值读取 `get_drive_target()`，夹爪值读取保存的开合控制量，并非全部关节的实时测量反馈。每侧夹爪是连续0～1标量，0为闭合目标、1为张开目标；不是抓取成功标签，也不是手指所有物理关节角。6轴双臂例子为 `[左臂6 rad, 左夹爪1, 右臂6 rad, 右夹爪1]` 共14维，HDF5每侧臂字段通常为 `(N,6)`、夹爪字段为 `(N,1)`；其他本体按实际轴数确定。当前通用导出按相邻采样值构造state/action，不能据字段名自动断言为“测量状态/实际下发命令”。本机RealMan独立24维SDK接口应另行区分，不能用常规标量夹爪解释六通道灵巧手。本次仅官方在线源码与本地HEAD `931ce21f4f0e63feab661f349634d5a25bd92681` 静态核对，未重新采集或读回HDF5。来源：[官方Robot](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/robot/robot.py) · [官方采样入口](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py) · [官方导出](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/utils/pkl2hdf5.py)。

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

## 2026-10-07 官方夹爪数据与灵巧手关节的区别

用户询问官方数据是否没有灵巧手关节姿态而只有抓/不抓。本轮静态核对Git保存的官方基线56286567103cf58bfa6191ed889908c5c3a0baa3及当前官方文档：常规夹爪每侧记录一个连续0..1归一化开合量，没有五指各个独立关节状态/目标；不能把它叫boolean抓取成功或简单抓/不抓标签。0通常表示闭合目标、1表示张开目标，允许中间值；闭合命令不保证已形成真实抓持。

原生state/left_ee_joint_states、right_ee_joint_states每侧为(N,1)，action相同。每侧6轴的双臂例子是6+1+6+1=14维，其他臂轴数应按本体配置，不能把所有机器人都写死14维。可选left/right_ee_poses每侧7维是腕/末端空间位姿，不能当作手指姿态。基线Robot.get_left/right_arm_jointState读取臂关节drive target并追加保存的夹爪标量；这些字段不等于全部实际finger qpos。

官方基线Robot.set_gripper先clip到0..1，再按本体gripper_scale映射到基准夹爪关节，其余关节通过mimic系数/偏置联动；planner.plan_grippers使用np.linspace(now_val,target_val,200)，开合过程可以有连续中间值。导出仍为相邻观测配对state=values[:-1]、action=values[1:]。本机HEAD2c8ba401aa51fb13296f9b6e1d6bb19487523cf6已经包含RealMan接入，不使用HEAD新增is_dexterous分支作为官方原生证据；c826f28父提交正是上述562865基线。本轮只读源码，没有更改Git配置或官方代码，没有运行新的采集或推理。

与当前20条的区别：π0.5训练实际消费独立realman_dexterous/measured_state24与next_command24，顺序为左臂6＋左手6＋右臂6＋右手6；手部是SDK6控制/实测表示，逐物理步另保留full36和手部24关节实际qpos。正式HDF中官方14维兼容字段继续保留，不能将其中每手1维ee字段说成已自动变为6维或已经包含五指全部关节。当前24维训练监督属于RM65-B接入补充，不是直接从官方一个夹爪开合量获得的五指示教。来源：[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证]]及本机realman_dataset20_success_audit_20261007_v1全量HDF审计。

核查源码（本机路径E:/Workspace/VLA-benchmark/RoboTwin，行号为562865基线版本）：envs/robot/robot.py第494–506/526–536/628–659行；envs/_base_task.py第485–495行；envs/utils/pkl2hdf5.py第59–63/103–116行；envs/robot/planner.py第425–434行。未修改的pkl2hdf5/planner在当前工作树相同行号可读。

在线官方来源（核查日期2026-10-07）：[数据字段说明](https://robotwin-platform.github.io/doc/usage/configurations.html)明确夹爪float opening ratio为0..1；[控制API](https://robotwin-platform.github.io/doc/usage/API.html)定义close_gripper默认pos=0、open_gripper默认pos=1。关联：[[RM65-B灵巧手仿真数据采集-RoboTwin2自动生成路线]] · [[RM65-B接入RoboTwin仿真]]。

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

## 化学实验室桌面物体资产核查（2026-10-21）

> [!summary] 当前结论
> 当前安装包中未识别到明确标注的烧杯、量筒、试管、试管架、锥形瓶或移液器。已有普通瓶、药瓶、杯、托盘、塑料盒和电子秤，可用于干式容器取放流程的原型；这些通用资产不能直接认作化学器具或已验证的实验功能。

本次核查本机 `/home/admin123/liminghe/vla-benchmark/RoboTwin`，HEAD `931ce21f4f0e63feab661f349634d5a25bd92681`，实际资产根目录为 `assets/objects`；同时在线核对官方 RoboTwin-OD 图鉴与任务文档。没有连接161/166服务器、渲染场景、执行抓取或新采集。资产未被Git版本号覆盖，HEAD仅标识加载器和任务源码版本。

### 已有的相关资产

下表数量为本机实际找到非空 `visual/base<ID>.glb`、`collision/base<ID>.glb` 和对应 `model_data<ID>.json` 的文件组数，表示文件存在，不代表加载、碰撞几何或RM65-B灵巧手抓取验收通过。

| 物体 | 目录 | 本机文件组数 | 与实验室任务的关系 |
| --- | --- | --- | --- |
| 普通瓶 | `001_bottle`、`114_bottle` | 23、4 | 可用于瓶体操作原型；不是已确认的试剂瓶模型。 |
| 药瓶 | `080_pillbottle` | 5 | 可作为样品容器取放原型；不表示瓶盖能独立开合。 |
| 普通杯 | `021_cup` | 13 | 可用于容器取放；未标注为带刻度烧杯。 |
| 带液体外观的杯 | `022_cup-with-liquid` | 1 | 描述为蓝色饮用杯、黄色液体；当前刚体加载不提供液体流动证据。 |
| 托盘 | `008_tray` | 4 | 可用于器具集配与转运场景。 |
| 普通电子秤 | `072_electronicscale` | 5 | 可用于向秤台放物；没有据此确认真实称重读数或分析天平精度。 |
| 塑料盒 | `062_plasticbox` | 11 | 可用于容器收纳和归位。 |
| 双杆支架 | `040_rack` | 1 | 语言描述为带直杆、斜杆的支架，不是试管架。 |

官方入口：[RoboTwin-OD](https://robotwin-platform.github.io/doc/objects/index.html) · [药瓶](https://robotwin-platform.github.io/doc/objects/080_pillbottle.html) · [普通杯](https://robotwin-platform.github.io/doc/objects/021_cup.html) · [带液体杯](https://robotwin-platform.github.io/doc/objects/022_cup-with-liquid.html) · [电子秤](https://robotwin-platform.github.io/doc/objects/072_electronicscale.html) · [支架](https://robotwin-platform.github.io/doc/objects/040_rack.html)。

### 搜索范围与不能由名称推出的功能

检查了120个编号物体目录、附带Objaverse目录、1068个JSON文件，以及本地 `assets/objects.zip` 的10070个归档条目名称；对烧杯、量筒、试管、锥形瓶、移液器等中英文关键词检索，没有找到明确命中。并核对了 `description/objects_description`、任务脚本和资产加载器。没有逐个目视检查全部网格，故结论限定为“当前包未发现明确对应资产”，不声称整个Objaverse或所有第三方扩展均不存在化学器具。

官网写RoboTwin-OD为147类、731件，其中Objaverse部分27类、153件，且不列入编号图鉴；本地 `objaverse/list.json` 与注释文件则为26类、152件。保留这一差异，不将官网总数当成本机资产完整性证明。本地下载器 `assets/_download.py` 指向 [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main) 的 `objects.zip`；本次未更新资产包。

`envs/utils/create_actor.py:create_actor` 分别读取视觉/碰撞网格与元数据并建立PhysX刚体。即使网格显示液面，也不能据此推出液体倾倒、混合、体积读数或化学反应仿真。`022_cup-with-liquid` 的本地 `model_data0.json` 未包含常规资产的 `scale`，也不能仅凭三类文件存在就称其默认加载配置完善。

### 对当前课题的复用方式

- `move_pillbottle_pad` 已有药瓶抓取、移动到垫上的任务模板；可参考其物体创建、目标点和成功判据，但RM65-B六通道灵巧手抓取需要单独适配验证。
- `place_object_scale` 当前将鼠标、订书机或铃放到电子秤；成功判断检查位置、高度与夹爪状态，没有读取称重数值。它是放置任务，不是现成化学称量任务。
- 若下一步需要烧杯、量筒、试管及试管架，须补充专用网格、正确尺度、质量/摩擦与碰撞几何、抓取/放置功能点及任务成功条件。试管插架尤其需要保留孔洞和容器内腔的有效碰撞几何，不能只更换外观。此为适配建议，尚未实施。

对本课题，可先复用药瓶、托盘、盒子构造干式样品取放、交接、归位与扰动恢复，再补充专用实验器具。液体操作需要另行明确模拟精度与实现成本；当前资产核查没有证明相关功能已可用。关联 [[RM65-B接入RoboTwin仿真]] · [[RM65-B灵巧手仿真数据采集-RoboTwin2自动生成路线]]。

任务来源：[Move Pillbottle Pad](https://robotwin-platform.github.io/doc/tasks/move_pillbottle_pad.html) · [Place Object Scale](https://robotwin-platform.github.io/doc/tasks/place_object_scale.html)；本地对应 `envs/move_pillbottle_pad.py`、`envs/place_object_scale.py`、`envs/utils/create_actor.py`，核查日期2026-10-21，证据类型为文件存在性与静态源码核查。

### RealMan桌面资产多视角渲染（2026-10-21）

> [!info] 此节保留原包通用资产展示
> 本节保留2026-10-21只用官方素材时的初期展示；药品容器需要保留，不能将对普通瓶的否定扩展为移除药品。2026-10-22用户已允许导入外部器皿，当前功能资产见 [[2026-10-22-RoboTwin实验室器皿导入与功能点标注]]。

本机RTX 5090使用SAPIEN 3实际加载厂家RM65-B双臂四代灵巧手URDF和7件代表物体，按规则网格置于高度0.74米的桌面。导出正面、左前、右前、俯视、机器人侧高位和桌面近景六张1920×1440 PNG，并生成总览。双臂使用既有端点IK预览的静态关节姿态；每组资产各一件，没有摆出全部型号。原URDF、GLB和资产JSON未改，展示脚本和文件哈希保存在同目录清单。

同日用户纠正：可乐不应出现在化学实验室场景。已从初版9件中移除 `001_bottle/13` 可乐瓶及 `114_bottle/1` 另一只饮料瓶，重新渲染并替换全部六张图片、总览与清单；当前保留药瓶、普通杯、带液体外观杯、挂杯支架、托盘、电子秤和塑料盒。实验室布景不再使用饮料瓶代替实验器具；通用杯、支架仍不冒称烧杯或试管架。

![[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/multiview_overview.png|1000]]

高清图：[[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/01_front.png|正面]] · [[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/02_front_left.png|左前]] · [[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/03_front_right.png|右前]] · [[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/04_top.png|俯视]] · [[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/05_robot_overhead.png|机器人侧高位]] · [[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/06_table_detail.png|桌面近景]]。

六张PNG解码与尺寸检查通过，当前清单确认7件且不含两只饮料瓶；保留物体的位置沿用已检查的网格布局，其可见包围盒均位于桌面范围，二维占地互不重叠，网格底部统一高于桌面1毫米。已目视检查修订后六视角总览，确认饮料瓶移除。此为真实渲染运行证据，物体采用静态视觉网格，物理步数为0，不包含碰撞/抓取验收、轨迹执行、策略推理、数据采集或实体机器人操作。`022_cup-with-liquid` 因缺官方scale，显式取0.06展示缩放，杯高约11.4厘米；这不是官方尺度或液体仿真验证。所有画面均为外置展示相机，不冒称标定后的头部或腕部D435观测。

复现脚本：[[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/render_tabletop.py]]；资产、位姿和相机清单：[[30-实验记录/evidence/2026-10-21-realman-tabletop-assets/render_manifest.json]]。本次使用现有本机 `robotwin` 环境，与161/166服务器状态无关。

### LabUtopia专用器具导入与渲染（2026-10-21）

> [!info] 历史外部导入测试，未采用
> 用户于2026-10-21曾限定仅使用官方素材，本节保留当时未采用的静态导入测试。2026-10-22用户已允许重新使用外部现成模型，其中烧杯、锥形瓶在新功能资产包中复用，详见 [[2026-10-22-RoboTwin实验室器皿导入与功能点标注]]；本节旧渲染仍不是功能验证证据。

用户要求更接近化学实验室，普通杯、生活托盘不再作为主要布景。本轮改用 [LabUtopia作者资产库](https://huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset) 的8种实验室专用几何：烧杯、锥形瓶、容量瓶、带塞试管组、独立试管架、移液器与支架、实验室电子天平、加热板；烧杯显示两个尺寸，共9个物体或组合。源 `TestTube_01` 本身包含5支带塞试管和支架；源 `Pipette` 是多支移液器与支架，不按单根器具计数。本轮未找到明确量筒资产，不把容量瓶误称量筒。

![[30-实验记录/evidence/2026-10-21-realman-lab-specific/06_table_detail.png|1000]]

六视角总览：[[30-实验记录/evidence/2026-10-21-realman-lab-specific/multiview_overview.png]]；高清 [[30-实验记录/evidence/2026-10-21-realman-lab-specific/01_front.png|正面]]、[[30-实验记录/evidence/2026-10-21-realman-lab-specific/04_top.png|俯视]]。已完成实际SAPIEN渲染与目视检查，6张1920×1440 PNG解码通过，9个摆放位置均在桌面内、占地不重叠、几何底部高于台面1毫米。

这是外部专用资产的网格导入，不是RoboTwin原包已有这些器具，也不表示已经迁移LabUtopia整套任务或Isaac Sim环境。固定来源revision为 `e622f0ec8123c9098d913ea56a4116fd83a29662`，数据资产CC BY-NC 4.0；9个原始USD文件约7.45MB，包含试管架的相对依赖子文件。使用独立临时目录的 `usd-core 26.8` 将单位、节点变换和材质分组展开为OBJ，并在现有SAPIEN中渲染；未改动现有Conda包、原机器人模型或原RoboTwin资产。

Omni MDL采用近似SAPIEN材质，玻璃可透射，但不声称复现原纹理、刻度和光学属性。模型尺寸为显式选择的展示尺寸，不是实机标定。当前仅静态视觉网格，物理步数0；碰撞体、抓取点、质量/摩擦、液体、称重/加热反馈、任务脚本和成功判据均未移植，不能据图像称为可采集训练数据。

完整署名、许可、文件路径、适配范围与复现入口见 [[30-实验记录/evidence/2026-10-21-realman-lab-specific/来源与适配说明]]。本试验仅作为历史外部导入记录保留，当前选择以下官方版本。

### 仅官方素材的实验室近似实例筛选与渲染（2026-10-21）

> [!info] 历史筛选版本
> 本节7件物品包含4只药品容器。后续曾误把对普通瓶的否定扩展为排除药品，现已按用户纠正补回；当前8件桌面和修正后的默认姿态见“近处主要操作区与周边摆放”。

用户最终约束：**只使用RoboTwin官方素材库中相近的实例，不采用外部资产；保留原 `072_electronicscale/model 0` 电子秤。** 因此本次回到本机官方 `assets/objects`，按具体实例而非只按物体类别筛选。原饮料瓶、餐盘、蓝色饮用杯、黄色液体杯、挂杯支架和外部LabUtopia器具均未进入此场景。

| 所选官方实例 | 数量 | 外观与定位 |
| --- | --- | --- |
| `080_pillbottle/base2` | 2 | 白色旋盖瓶，作为样品容器外观近似；仍保留原标签纹理。 |
| `080_pillbottle/base4` | 2 | 棕色瓶身、白盖、宽白标签，外观接近试剂瓶；不宣称原类别就是试剂瓶。 |
| `021_cup/base5` | 1 | 无柄金属圆筒容器，避免原蓝色餐饮杯；不是专业烧杯。 |
| `062_plasticbox/base2` | 1 | 中性开口收纳容器，替换餐盘与蓝色生活收纳盒。 |
| `072_electronicscale/base0` | 1 | 沿用用户认可的原电子秤，文件SHA、型号与scale `[0.08,0.08,0.08]` 均与首版一致。 |

`021_cup/base6` 虽是银色金属杯，但官方描述明确有弯柄，故不选；`001_bottle/base17` 的内嵌纹理有醒目成品商标，`base20` 有花卉与提环，也未选。瓶子采用官方原模型多实例摆放，没有改颜色、去标签、重建器具或引入新的物体资产。

![[30-实验记录/evidence/2026-10-21-realman-official-lab-selection/06_table_detail.png|1000]]

完整六视角：[[30-实验记录/evidence/2026-10-21-realman-official-lab-selection/multiview_overview.png]]；高清 [[30-实验记录/evidence/2026-10-21-realman-official-lab-selection/01_front.png|正面]]、[[30-实验记录/evidence/2026-10-21-realman-official-lab-selection/04_top.png|俯视]]。

复现脚本 [[30-实验记录/evidence/2026-10-21-realman-official-lab-selection/render_official_selection.py]]；实际资产SHA、ID、尺度、位置和相机见 [[30-实验记录/evidence/2026-10-21-realman-official-lab-selection/render_manifest.json]]。物体网格全部位于本机官方 `RoboTwin/assets/objects`，六张1920×1440 PNG已解码、目视检查，7个物体均在台面内且占地分离，电子秤与原版本的网格SHA和缩放一致。当前仍为静态外观展示，未运行物理步、抓取或数据采集；不以相近外观证明真实化学器具功能。

### 烧杯、量筒与培养皿专项复查（2026-10-21）

用户进一步询问培养皿是否也可用。本轮重查官方图鉴、本机 `description/objects_description` 和 `assets/objects` 的JSON，检索 `beaker`、`graduated`、`measuring cylinder`、`petri`、`culture dish` 及中文名称，没有找到明确对应的专用资产。另检查 `003_plate`、`019_coaster` 和未列入编号图鉴的 `objaverse/plate` 全5个实例的原纹理与材质，未找到可确认的培养皿。

| 候选 | 实际核对结果 |
| --- | --- |
| `objaverse/plate/003`、`004` | 带花边图案的圆餐盘。 |
| `objaverse/plate/007` | 白色放射纹浅圆餐盘，是这些候选中最接近浅皿的形状；仍为陶瓷餐盘外观。 |
| `objaverse/plate/008` | 带花边图案的矩形餐盘。 |
| `objaverse/plate/009` | 浅绿色六边形盘，底部有商品标签。 |
| `003_plate/base0` | 普通餐盘，官方语言描述明确为 `food plate`。 |
| `019_coaster/base0` | 木纹杯垫，不是有容积的培养皿。 |

5个Objaverse盘子的MTL均为普通贴图材质，JPG不含alpha，未声明透明/玻璃透射；`003_plate` 与 `019_coaster` 的GLB材料未声明透明模式或透射扩展。此处结合原纹理和形状判断为餐盘/杯垫，不仅靠材质排除“培养皿”语义；也不以不透明本身认定所有培养皿都应透明。

因此结论限定为：**在当前下载的官方包及已复查实例里，未发现可确认的烧杯、量筒或培养皿专用模型；现有金属容器和浅盘只是近似候选。** 不把餐盘重命名为培养皿，不改动当前场景。若以后允许本地程序建模，可由SAPIEN原生几何构建这些简单器皿，但这属于自建模型，不是官方现成素材，且碰撞、抓取和物理功能须另外验证。

来源：[官方RoboTwin-OD](https://robotwin-platform.github.io/doc/objects/index.html)、[003_Plate](https://robotwin-platform.github.io/doc/objects/003_plate.html)、[021_Cup](https://robotwin-platform.github.io/doc/objects/021_cup.html)；本机实际文件 `assets/objects/objaverse/plate/*/textured.obj`、`*.obj.mtl`、`*.jpg`、`003_plate/visual/base0.glb`、`019_coaster/visual/base0.glb` 及对应语言描述。源码HEAD仍为 `931ce21f4f0e63feab661f349634d5a25bd92681`；静态资产与原纹理核查，没有新增仿真运行或采集。

### 电子秤称量纸专项复查（2026-10-21）

用户询问可放在电子秤秤盘上的纸，通常称为称量纸。本机官方包没有检索到独立 `weighing paper`、`weighing boat`、`filter paper` 或称量纸、称量舟、滤纸资产。`023_tissue-box` 为纸巾盒，7个GLB实例均是单一整体网格；`028_roll-paper` 为卷纸，不是可独立铺放的纸片；纸牌类别为整盒纸牌，也不适合直接替代。

特别检查了官方包附带的 `objaverse/tissue/004` 与 `005` 原纹理和URDF：前者明确为整包湿巾（包装含 `WET WIPES / 40片装`），后者为整包抽纸（含 `80抽` 等字样），两者都是单刚体、原配置质量0.1kg，不是独立称量纸。

若仅服务当前静态可视化，可使用RoboTwin已有的 `create_box` 原生几何接口创建极薄白色方片，按原电子秤秤盘的实际尺寸定位。这不需要外部模型，但属于程序新建纸片外观，不能说官方库已经提供称量纸，也不自动具备纸张形变、折叠或称重反馈。本轮只核查资产，没有创建纸片或修改渲染场景。

### 官方包透明物体与透明材质复查（2026-10-21）

> [!important] 专用实验器具缺失，不代表没有透明物体
> 本轮直接扫描原材质并实际渲染，确认官方包中存在半透明瓶体。此前筛选未单独列出这一类；“未找到烧杯、量筒、培养皿”不能扩展为“官方包没有透明物体”。

扫描了本机官方物体目录中572个 `visual/*.glb` 的材质，零解析错误；在 `114_bottle` 3个实例的共10个材质中发现 `alphaMode=BLEND` 与alpha小于1：

| 官方实例 | 瓶身部分材质alpha | 外观与适用边界 |
| --- | --- | --- |
| `114_bottle/base1.glb` | 约0.11 | 黄色瓶体；原贴图未见文字，不是专用实验器皿。 |
| `114_bottle/base2.glb` | 约0.41 | 无纹理图、无品牌文字，主要为纯色瓶身、瓶盖与色带；可作为中性透明容器候选，瓶型仍是普通瓶。 |
| `114_bottle/base4.glb` | 0.50 | 蓝盖透明瓶，有矿泉水文字包装；不用于当前实验室布景。 |

这些参数仅作用于部分材质，瓶盖、标签等其他部分可保持不透明；alpha不是经标定的光学透过率，不能当作折射率或物理玻璃模型。另扫描2114个MTL文件，部分烤箱、微波炉、厨房锅等部件以及辅助 `cube` 也含 `d<1` 的透明设置，故上述3瓶不代表全库只有这3件带透明部件的资产。

使用原GLB、原尺度和原材质，在SAPIEN RT中独立渲染3瓶与背后的棋盘格，确认瓶身部分区域可见后方棋盘格。未覆写材质，画面存在明显三角面/透明显示伪影，不声称高质量或物理准确玻璃渲染。图中从左到右为4、2、1号：

![[30-实验记录/evidence/2026-10-21-robotwin-transparency/transparent_candidates.png|1000]]

`088_wineglass` 的5个实例虽在语言描述中含transparent/clear，但当前GLB材质均缺省OPAQUE，贴图为RGB且没有alpha，也未声明玻璃透射扩展；因此语言里的“玻璃/透明外观”不能自动算作已验证的透光材质。这一类未在本轮执行透光渲染。

证据：[[30-实验记录/evidence/2026-10-21-robotwin-transparency/material_audit.json|原材质扫描与范围]]、[[30-实验记录/evidence/2026-10-21-robotwin-transparency/render_transparent_candidates.py|复现脚本]]、[[30-实验记录/evidence/2026-10-21-robotwin-transparency/render_receipt.json|实际渲染记录]]。这次只做透明性对比，没有把饮料瓶加入当前RealMan桌面，没有修改原模型或验证抓取/采集。

### 官方小盘子实际尺寸复查（2026-10-21）

用户询问小一点的盘子。本轮直接读取原视觉网格，并应用各自URDF的mesh scale（Objaverse为0.001）或GLB模型JSON的scale，计算外包尺寸，没有用注释中的 `object_size` 代替实际几何。

| 官方实例 | 当前网格外包尺寸 | 外观 |
| --- | --- | --- |
| `objaverse/plate/007` | 156.7 × 156.4 × 14.3 mm | 白色浅圆盘，边缘放射纹，是这5款Objaverse盘里最小、最浅的一款。 |
| `003_plate/base0` | 约190.3 × 189.7 × 25.0 mm（平面两轴与高度） | 常规圆盘。 |
| `objaverse/plate/003` | 192.2 × 191.7 × 38.6 mm | 带花边的圆盘。 |

推荐小盘候选为 `objaverse/plate/007`：原始尺寸约直径15.7厘米、整体高度1.43厘米；高度是外包高度，不等于盘内深度。若后续希望8～10厘米量级，可显式等比缩小这个官方原模型，模型来源不变，但那是人为缩放后的尺寸，并不是官方默认尺寸。本轮没有修改场景或缩放任何资产。

同日按用户随后要求执行实际渲染：新建独立预览，仅将原尺寸 `objaverse/plate/007` 与原电子秤 `072_electronicscale/base0` 放在RealMan桌面上作尺寸对照。盘子按原URDF的0.001缩放、identity姿态加载，未缩小或换材质；OBJ、MTL、JPG从官方原目录直接读取。4张1920×1440 PNG均解码通过、来源网格SHA及底面对齐核对通过，已目视检查近景与俯视。源盘面有扫描纹理和几何起伏，图像按官方默认姿态展示，不将其当作理想培养皿几何。静态渲染、物理步数0，没有抓取或采集。

![[30-实验记录/evidence/2026-10-21-official-small-plate/02_plate_and_scale.png|1000]]

查看 [[30-实验记录/evidence/2026-10-21-official-small-plate/03_top.png|俯视]]、[[30-实验记录/evidence/2026-10-21-official-small-plate/04_plate_close.png|盘子近景]]、[[30-实验记录/evidence/2026-10-21-official-small-plate/multiview_overview.png|四视角总览]]。复现脚本 [[30-实验记录/evidence/2026-10-21-official-small-plate/render_small_plate.py]]；资产、尺度和相机清单 [[30-实验记录/evidence/2026-10-21-official-small-plate/render_manifest.json]]。

### 完整桌面与左腕默认姿态修正（2026-10-21）

> [!info] 漏放药品的中间场景，已更正
> 用户接受官方小盘007，要求将物品一起摆上桌并修正左手外翻。本版当时只摆放4件官方物体：原电子秤、小盘007、无柄金属容器、开口收纳盒。将药品容器一并排除是对用户意图的误读；用户随后指出“药品呢？怎么没有了”，补回后的当前场景见下一节。

本机现有 `robotwin` 环境通过SAPIEN 3实际渲染厂家RM65-B双臂四代灵巧手URDF。4件物品按两排错位摆放在0.74米高的桌面上：原电子秤中心 `(-0.27, 0.27)` 米，小盘中心 `(0.02, 0.27)` 米，金属容器中心 `(-0.40, -0.01)` 米，收纳盒中心 `(0.38, 0.14)` 米。小盘保留官方0.001缩放，外包尺寸约156.7×156.4×14.3毫米；电子秤的源网格SHA及scale `[0.08, 0.08, 0.08]` 与原预览一致。所有物体可见包围盒均在台面内，水平足迹互不重叠，几何底面高于台面1毫米；这不是动态稳定性或可达性验证。

![[30-实验记录/evidence/2026-10-21-realman-tabletop-final/01_front.png|1000]]

**左手外翻修正**：原预览中 `l_joint6=-1.7524284124 rad`，本轮仅将该关节增加π，保存为 `+1.3891642412 rad`，位于原URDF的 `[-6.28, 6.28] rad` 限位内。关节按名称映射，不依赖36维活动关节向量的交错顺序。左右手掌面均对应各自手底座局部+Y；经实际SAPIEN正运动学核对，左掌世界法向由约 `[-0.7071, +0.7071, 0]` 翻为 `[+0.7071, -0.7071, 0]`，右掌保持 `[-0.7071, -0.7071, 0]`，双掌朝内。左手底座位置和手指长轴保持不变，其他关节未调整；目视检查正面、俯视、桌面近景及修正前后对比通过。

![[30-实验记录/evidence/2026-10-21-realman-tabletop-final/left_wrist_comparison.png|1000]]

默认姿态和物体位置已持久保存到 [[30-实验记录/evidence/2026-10-21-realman-tabletop-final/scene_preset.json|场景预设]]，[[30-实验记录/evidence/2026-10-21-realman-tabletop-final/render_scene.py|渲染脚本]]默认读取该文件，按关节名载入姿态并检查限位。它是**当前展示场景的默认值**；没有修改厂家URDF、RoboTwin所有任务的全局home配置，也没有下发真机动作。源URDF的SHA保持 `4bf28c444af1a28401a975e5de6e138c1ec043536aa786ea541c30c359bd65bd`。

高清 [[30-实验记录/evidence/2026-10-21-realman-tabletop-final/04_top.png|俯视]]、[[30-实验记录/evidence/2026-10-21-realman-tabletop-final/06_table_detail.png|桌面近景]]、[[30-实验记录/evidence/2026-10-21-realman-tabletop-final/07_hands_detail.png|双手近景]]、[[30-实验记录/evidence/2026-10-21-realman-tabletop-final/multiview_overview.png|六视角总览]]；完整位姿、掌面法向、限位及相机记录见 [[30-实验记录/evidence/2026-10-21-realman-tabletop-final/render_manifest.json|渲染清单]]。7张新姿态1920×1440 PNG及1张旧姿态对照已实际生成，均为外置展示相机，不是标定后的D435观测。

证据范围：2026-10-21本机实际静态渲染、几何位置与关节限位检查；RoboTwin源码HEAD `931ce21f4f0e63feab661f349634d5a25bd92681`，资产另以清单SHA标识。物理步数0，未验证全机器人碰撞、动态抓取、轨迹规划或数据采集，未运行策略推理或实体机器人控制。

### 药品补回与当前完整桌面（2026-10-21）

> [!info] 药品补回版本：8件物品，后续进一步调整摆放
> 按用户纠正，补回此前选定的4只药品容器：官方 `080_pillbottle/base2` 白色药瓶2只、`base4` 棕色药瓶2只。对普通饮料瓶的排除不应扩大为移除药品；当前保留药品容器，不加入可乐、矿泉水等饮料瓶。

原电子秤、小盘007、金属容器、收纳盒的位置与官方尺度保持不变；白色药瓶中心放在 `(-0.56, 0.27)`、`(-0.46, 0.27)` 米，棕色药瓶放在 `(0.21, 0.35)`、`(0.32, 0.35)` 米。4只药瓶沿用先前筛选的原GLB、原纹理和官方scale `[0.08, 0.08, 0.08]`，高度约15.2厘米。这里展示的是药品容器网格，不表示已有可分离药片、可开合瓶盖或真实化学功能。

![[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/01_front.png|1000]]

本机SAPIEN实际重新生成7个新姿态视角及旧腕姿态对照，正面和俯视目视检查通过。8件物体的二维包围盒均位于桌面内、互不重叠；已修正的 `l_joint6=+1.3891642412 rad` 与其余全部机器人关节值保持上一版不变。当前默认摆放及姿态已另存为 [[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/scene_preset.json|含药品的场景预设]]，渲染脚本默认读取该文件。

查看 [[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/04_top.png|俯视]]、[[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/06_table_detail.png|桌面近景]]、[[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/multiview_overview.png|六视角总览]]；[[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/render_scene.py|复现脚本]]与[[30-实验记录/evidence/2026-10-21-realman-tabletop-medicines/render_manifest.json|源文件SHA及位姿清单]]保留在同目录。来源仍为上述RoboTwin本机官方资产及未改动的厂家URDF；当前仅静态渲染，物理步数0，没有新增抓取、碰撞、策略或真机运行验证。

### 近处主要操作区与周边摆放（2026-10-22）

> [!info] 官方原包的近处摆放版本
> 按用户当时要求，把电子秤、小盘007和1只棕色药瓶移到双手前方近处；其余5件物品均匀分散在两侧及外圈。后续已授权导入专用器皿，最新功能场景见文末“外部器皿与功能药瓶”。

主要操作区三件物品的中心分别为：电子秤 `(-0.20, -0.04)` 米、小盘 `(0.04, -0.04)` 米、棕色药瓶 `(0.25, -0.04)` 米。相较10月21日药品补回版，它们沿桌面前后方向分别向机器人靠近31、31、39厘米。外围为金属容器 `(-0.62, 0.03)` 米、收纳盒 `(0.60, 0.15)` 米、两只白色药瓶 `(-0.43, 0.36)` 和 `(0.43, 0.36)` 米，以及另一只棕色药瓶 `(0, 0.42)` 米。全部沿用原网格、材质和尺度，机器人关节姿态与上一版相同。

![[30-实验记录/evidence/2026-10-22-realman-operation-zone/01_front.png|1000]]

**运动学核验与边界**：独立CPU脚本直接读取实际厂家URDF，对左臂接近电子秤 `(-0.20, -0.04, 0.82)` 米、右臂接近小盘 `(0.04, -0.04, 0.79)` 米、右臂接近药瓶 `(0.25, -0.04, 0.84)` 米分别求解完整位姿IK。三点均保持各自默认手腕朝向，满足原关节限位，数值位置残差均小于 `1e-8` 米。名义TCP取手底座局部+Z偏移0.18米，是未标定的几何参考点；本轮仅证明这3个接近端点在此运动学模型下有解，不代表无碰撞轨迹、真实接触抓取、三件物体同时操作或实机可达性已验收。记录见 [[30-实验记录/evidence/2026-10-22-realman-operation-zone/reach_check.json|最终三点IK结果]]，独立复核入口为 [[30-实验记录/evidence/2026-10-22-realman-operation-zone/check_reach.py|CPU运动学脚本]]。

本机 `robotwin` 环境实际SAPIEN渲染8张1920×1440 PNG，已检查正面、俯视和操作区近景；8件物体的可见包围盒均位于桌面内，水平足迹互不重叠。[[30-实验记录/evidence/2026-10-22-realman-operation-zone/04_top.png|俯视]] · [[30-实验记录/evidence/2026-10-22-realman-operation-zone/08_operation_close.png|主要操作区近景]] · [[30-实验记录/evidence/2026-10-22-realman-operation-zone/multiview_overview.png|六视角总览]]。当前摆放已保存为 [[30-实验记录/evidence/2026-10-22-realman-operation-zone/scene_preset.json|场景预设]]，[[30-实验记录/evidence/2026-10-22-realman-operation-zone/render_scene.py|渲染脚本]]默认读取它；完整源文件SHA、物体位姿及相机见 [[30-实验记录/evidence/2026-10-22-realman-operation-zone/render_manifest.json|渲染清单]]。源码HEAD仍为上述 `931ce21f4f0e63feab661f349634d5a25bd92681`，源URDF未改，物理步数0，未执行策略、抓取、数据采集或真机控制。

**同日药瓶内容物核查**：用户询问瓶内是否有药片。本机官方 `080_pillbottle/visual/base4.glb` 棕瓶及 `base2.glb` 白瓶均仅有1个节点、1个mesh、1个primitive和1个不透明材质；合并UV／法线接缝顶点后为单一封闭连通网格，未发现独立内含药片模型。原始接缝拆出的网格片段不能当成药片。当前场景也只创建整瓶静态视觉actor，没有单独药片刚体、可独立开合的瓶盖或内容物反馈。封闭网格不能证明具备可装物的有效中空内腔；倒药、数药、逐片称量任务需另加药片刚体并适配瓶口、内腔、瓶盖及碰撞体。本次为原GLB结构、几何与现有场景源码静态核查，没有更改场景或执行物理测试。

**同日官方拧盖标注核查**：需区分外观语言和可执行机构。`description/objects_description/080_pillbottle/base4.json:18` 确实含 `bottle with white screw-on cap`（带白色螺旋盖的瓶子）的外观描述，不能笼统说官方没有提到螺旋盖；但 `points_info.json` 只标瓶身／顶部抓持点和底面放置点，没有可拧开的盖子关节或开盖功能点。`envs/move_pillbottle_pad.py` 通过 `create_actor` 加载整瓶单刚体，只抓起、搬运和放到垫子，成功条件检查整瓶位置与夹爪状态；`envs/utils/create_actor.py:541–559` 对应单actor构建，而非瓶身与瓶盖的关节机构。资产目录没有该药瓶的URDF或独立瓶盖关节文件，当前本机任务中未发现针对该类药瓶的拧盖实现。[官方物体页](https://robotwin-platform.github.io/doc/objects/080_pillbottle.html)展示5种外观，[官方任务页](https://robotwin-platform.github.io/doc/tasks/move_pillbottle_pad.html)也将任务定义为把药瓶放到垫子。结论：语义描述为带旋盖的药瓶，现成仿真资产仍不能直接执行拧盖任务。核查日期2026-10-22，源码HEAD同上；网页与本机文件静态核查，未改场景或运行开盖实验。

**同日官方全包旋盖药瓶复查**：进一步检查当前本机官方 `assets/objects`（含Objaverse），没有找到可直接拧开瓶盖的药瓶。`080_pillbottle/base1–base5` 全5款均为单节点、单mesh的整瓶GLB，没有独立瓶盖或药瓶关节配置。共解析196个URDF、零解析错误，其中44个有非fixed关节，分布在笔、电脑、微波炉、水壶、柜、盒、锅、开关、烤箱，未发现瓶罐螺旋盖机构；Objaverse的152个URDF均为单link、无joint，其7个bottle及3个thermos也没有活动瓶盖。另扩查129个瓶罐及相关容器视觉GLB，未找到现成的瓶身—瓶盖分离旋盖配置。锅盖、水壶盖或盒盖的铰链／直线活动机构不能当作药瓶螺纹旋盖。清单见 [[30-实验记录/evidence/2026-10-22-realman-operation-zone/official_cap_asset_audit.json|官方包旋盖资产审计]]；同时在线核对官方物体图鉴和公开任务，未找到现成药瓶拧盖条目。结论范围为本次检查的本机官方包与公开入口，不对未检查的远端资产版本作绝对否定。若后续坚持不引入外部模型，可改造官方药瓶，但需要自行拆分瓶盖、建立运动／接触约束和成功判据，属于自建可操作资产；本轮未实施。

### 外部器皿与功能药瓶（2026-10-22）

用户已授权导入已有外部烧杯、器皿和透明量器，要求按官方流程标注，并优先支持瓶内药品、旋盖与倾倒。这一要求取代前期只用官方原包的限制。当前完成烧杯、锥形瓶、真正量筒、培养皿的现成模型导入，以及明确标为本地派生的分体药瓶／瓶盖；原电子秤与修正左手姿态保留。

六件资产的功能点通过原始RoboTwin加载器读取和世界坐标核对，五个开口容器通过实际落球内腔检查。药瓶与12颗独立药片完成3种子夹具驱动的闭盖倒置、导向旋盖释放及重力倒药入皿验证，卡盖负控也通过；药片初始化后没有pose写入。螺旋机构是按实测转角耦合抬升的D6约束近似，不是真实螺纹接触，也没有验证RM65-B自主抓持／拧盖或VLA/RL训练。

完整来源、许可、标注、视频、复现与后续机器人接入边界见 [[2026-10-22-RoboTwin实验室器皿导入与功能点标注]]；[[30-实验记录/evidence/2026-10-22-labware-integration/README.md|资产包使用入口]]和[[30-实验记录/evidence/2026-10-22-labware-integration/acceptance_report.json|最终验收记录]]。此次新增资产位于独立任务目录，通过绝对模型目录接入官方加载器，未覆盖官方assets或修改原加载器。

**同日后续官方Agent采集接入：**六件物体已补独立points_info、外底面功能点、模型映射与seen/unseen描述；原Agent enrichment和65个Actor点矩阵实测通过。用户授权使用当前Codex会话生成代码，原生成器解析与官方物理测试函数已实际执行，右臂空开口药瓶在声明的近位1cm×1cm XY区域10/10成功。沿用原 `collect_data.run` 的规划保存、独立reset重放、语言和合并主循环，通过任务局部SDK24 hooks完成一条389窗口／23.34秒的真实仿真示范；三路1167帧、实测state24、实际驱动target24、60ms及12＋12指令均独立读回通过。最终v2只修正头相机俯视，逐帧目标完整入画，原v1裁切证据保留。入口为 [[30-实验记录/evidence/2026-10-22-labware-agent-pipeline/README|实际采集命令与验收]]；旋盖新组件已移除夹具和位置电机，但机器人开盖／倒药与其他器皿任务仍未验收，不将空瓶单任务外推为全场景能力。
