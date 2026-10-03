---
title: RM65-B接入RoboTwin仿真
date: 2026-09-23
updated: 2026-10-03
tags: [环境, RoboTwin, 机器人, 研究收获]
status: 官方adjust_bottle持续完善中；无合格episode，先完成无推瓶路径与稳定物理抓握
sources:
  - https://robotwin-platform.github.io/doc/usage/new-embodiment.html
  - https://develop.realman-robotics.com/robot/download/model/
---

# RM65-B接入RoboTwin仿真

## 目标与接入约束

用户最初要求仅新增硬件支持、尽量保持官方结构与控制方式，保留官方已有代码；2026-10-02 后续明确授权尝试一个简单任务，仅生成首条数据。环境固定为 [[服务器172.17.27.166-robotwin]] 的 Conda `robotwin`。

本地工程：`E:\Workspace\VLA-benchmark\RoboTwin`。服务端工程：`/bigdata2/liminghe/VLA-benchmark/RoboTwin`。基线 commit `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`，XPolicyLab pin `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。

## 已确认的接口与实现

- 官方 Robot、任务和评测 bridge 均按每手一个标量开合量；即双RM65原生 qpos 动作为 `[左臂6,左手1,右臂6,右手1]` 共14维。单改 `ee_dim` 不能获得24维独立灵巧手控制。
- 新增目录 `integrations/realman_rm65b`，生成 `assets/embodiments/realman-rm65b-inspire-reference`，包含整机URDF、SRDF、config、左右CuRobo模板、碰撞球、网格及来源hash。厂家压缩包不修改。
- 生成副本保留双臂厂家关节原点/限位/惯性与底盘/头部结构，固定非任务自由度。原始底盘根坐标不在地面，固定链网格最低点为-0.2453455m，配置抬高底盘并等量降低平台，保持肩部工作高度。
- 三代手URDF每侧12活动关节、6条耦合关系；部分四指MCP限位超过DIP，生成副本收紧到交集1.6rad。厂家文件原样保留。
- SAPIEN 3.0b1 对URDF mimic会额外建立刚度1e5的 tendon。原型把手耦合统一放到RoboTwin `config.yml` 原生 `base+mimic`，生成URDF移除重复mimic标签，原关系保留在manifest；这仍是标量协同控制。
- 新增环境profile `realman_rm65b_inspire_reference`、任务配置 `realman_rm65b_reference`，三个元数据注册点分别在embodiment表、父仓库robot-info与XPolicyLab子模块robot-info。
- 左右法兰轴向不同，左手沿法兰-Z、右手沿+Z装配，分别设置旋转；44.5mm转接距离仍是仿真参考，不是厂家转接法兰标定。
- SAPIEN默认固定关节frame与TCP子link相差Rx(π)，通过官方既有`global_trans_matrix`配置补偿；对左右多组q检验native末端变换往返及CuRobo世界FK一致，不改控制代码。
- 手协同采用经过仿真自碰撞检查的受限范围：四指MCP/DIP闭合0.9rad、拇指屈曲0.05rad、对掌0.8rad；其余指关节沿原6条mimic关系。全行程同时闭合曾导致14.5mm拇指/食指穿透，因此未沿用。受限范围不等同于完整手能力。
- 笛卡尔动作走原生CuRobo轨迹驱动，qpos动作走原生MPLib TOPP。原生HDF5 state/action是drive target和commanded synergy，实际跟踪另读SAPIEN qpos。
- 正式任务RT图像核查发现厂家camera_link位于机壳表面，光轴前20.16mm仍有相机外壳；raster近裁剪曾掩盖该问题。新增camera_link局部`[0,-0.03,0]`m参考光心偏移，朝向桌面中心；30mm/40mm对照均看清锤子与红块，20mm仍有边缘遮挡。此为仿真光学预设，不是实机标定。

## 证据边界

> [!warning] 不能称为四代触觉手完整接入
> 本地可动手资产是三代短腕手；实机型号尚待确认。手法兰、TCP、腕相机外参是显式仿真预设，不是实机标定。独立六通道、触觉、电流、实体通信未接入。

官方任务运行时会将所有link质量设为1kg，保留URDF惯性不等于真实动力学。当前独立CuRobo规划碰撞模型只覆盖每侧臂与张开手，不随手姿变化更新，且不构成完整躯干/双臂/动态场景碰撞验证。

2026-09-23 实际验证结果：

| 层级 | 结果 |
| --- | --- |
| 厂家转换/静态资源 | 5项转换测试通过；生成61个link、60个joint、36个活动关节，动作14维 |
| 原生组件 | 双臂home实际跟踪、双手受限开合、三路图像通过 |
| 运动学与规划 | 左右各3组FK、原生末端坐标变换往返、CuRobo小幅位姿规划及SAPIEN实际执行通过 |
| 官方任务接口 | beat_block_hammer固定seed0的3个小幅qpos指令、三路320×240 RGB、原生HDF5及XPolicyLab解码通过；4帧构成3组相邻state/action |
| 旧平台回归 | Aloha既有profile的小幅控制与采集接口通过 |
| 专家任务 | seed0单次专家尝试正常返回但plan_success=false、check_success=false；没有成功抓锤示范或策略成功率证据 |

官方Python控制/规划/任务/相机/HDF5/评测代码无修改，仅追加3个既有注册表条目并新增独立配置/工具。最终报告绑定URDF/config/SRDF/CuRobo/碰撞球/manifest与验收脚本hash。历史诊断出现过重复mimic tendon数值不稳定、闭手穿插和腕部粗球误判，均不能作为通过记录；最终以sim_smoke与task_smoke JSON为准。

证据目录：服务端 `/bigdata2/liminghe/VLA-benchmark/deploy/realman/`，Windows `E:\Workspace\VLA-benchmark\deployment\realman\`。计划、操作与验收文档在工程 `integrations/realman_rm65b/PLAN.md`、`README.md`、`VALIDATION.md`。新增厂家资产遵循官方assets忽略规则，未提交/推送，也未确认再分发许可。

## 关联笔记

[[RM65-B双臂机器人]] · [[硬件资源索引]] · [[RoboTwin 2.0框架与默认任务]] · [[服务器172.17.27.166-robotwin]] · [[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]]

## 2026-10-01 服务器当前版本核查

> [!info] 本次为 SSH 只读核查
> 读取当前源码、环境元数据与已有报告，没有重跑仿真或执行实体机器人指令。下列运行结果来自 2026-09-23 的报告，不能称为 10 月 1 日新一轮仿真验收。

服务器当前提交为 56286567103cf58bfa6191ed889908c5c3a0baa3，工作树干净。9 月 23 日后续迁移已将四代手版本纳入私有项目；前文“未提交/推送”和“仅有三代参考资产”属于较早阶段，不能描述当前服务器。Windows 本机 RoboTwin 当前提交为 ea8b211，与服务器版本不同，本次没有同步或覆盖任一仓库。

### 当前四代手版本

- 主模型：/bigdata2/liminghe/VLA-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/。
- 代码：integrations/realman_rm65b_gen4/；实际服务器证据：/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/。
- 整机 93 个 link、92 个 joint、93 个网格；静止操作版本固定底盘与头部，保留 36 个活动关节。厂家质量和惯性保留。
- 控制动作为 [左臂6 rad, 左手6 SDK编码, 右臂6 rad, 右手6 SDK编码]，共 24 个独立控制量。每手另 6 个关节按厂家 mimic 关系生成目标。
- 通过官方 envs.robot.robot.Robot 加载模型；simulate.py 明确关闭 TOPP，并使用专用 DexterousController 驱动。官方夹爪、规划器及标准 14 维任务动作没有用于本版本的控制验收。

### 已有运行证据与未完成项

| 项目 | 当前证据 |
| --- | --- |
| 结构与输入合同 | delivery_checks.json 记录本机与服务器各 3 组测试通过 |
| 双手自由空间六通道 | 服务器 smoke_report.json 的 6 组检查全部通过；最大关节误差 0.0001453333 rad |
| 双臂小幅关节动作 | 最大实际跟踪误差 0.00001538694 rad |
| -1 保持指令 | 最大位置变化 2.5123e-8 rad |
| SDK 张手预设 | 最大误差 0.0003281322 rad |
| SDK 闭手预设 | preset_review.json 记录拇指/食指接触，最大穿透 5.2312e-6 m，最大目标误差 0.07861931 rad；不属于无接触目标跟踪通过 |
| 网页检查 | 图片、视频、服务代码存在；本次 ps 与 ss 核查未发现该服务进程或 8767 端口监听 |
| 原生规划与任务接口 | 四代模型的 CuRobo/MPLib 规划、笛卡尔控制与 24 维任务接口仍待接入和验收 |
| 硬件一致性 | SDK 编码至弧度仍为未标定线性映射；力控未实现，触觉仅有几何与接触代理；TCP、腕相机及安装外参待实测 |

旧 realman-rm65b-inspire-reference 的 14 维接口仍保留，8 个主要模型输入文件哈希与其 sim_smoke、task_smoke 报告一致。旧 control_example 的左臂回零实际误差为 0.06231957 rad；这属于旧参考版，不能据此判定四代版失败，也不能用旧版规划通过证明四代版规划已完成。

当前对官方主任务代码的改动为 envs/_base_task.py 中一处去噪器配置改为读取 ROBOTWIN_RT_DENOISER，默认仍为 oidn；属于迁移合并的渲染兼容改动。当前版本的官方 Robot 与规划代码未发生对应改写。旧 final_audit.json 是早期版本的审计记录，不能证明当前提交的全部 Python 文件均未修改。

当前优先缺口是四代手版本的原生规划/笛卡尔控制与统一 24 维接口；专家任务、数据集设计、VLA/RL 训练和真机部署不作为本次组件级成果。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]

## 2026-10-01 轨迹能力与关节导入逐项核查

核查提交仍为服务器 56286567103cf58bfa6191ed889908c5c3a0baa3，工作树干净。本次通过 SSH 在 Conda robotwin 中运行现有 test_contract.py，3 组模型合同测试通过；没有重跑动力学仿真或执行实体机器人指令。

### 目标运动与轨迹跟踪的区别

- integrations/realman_rm65b_gen4/controller.py 的 set_arm / apply_action 接收绝对关节角；step 每 0.002 s 将臂关节驱动目标向目标角度推进，单轴上限为 0.35 rad/s，再使用 SAPIEN drive 和被动力补偿执行。
- 该逐轴斜坡是仿真控制参数，不是睿尔曼 MoveJ 的速度百分比，也没有等时多轴到达、加速度约束或时间戳轨迹的实现与验收。连续更新关节目标具备代码基础，但已有小幅终点误差不能证明连续轨迹跟踪误差合格。
- simulate.py 第 38–40 行明确只复用官方 Robot 模型加载，need_topp=False，未使用官方夹爪、规划器和 14 维任务动作。当前四代版本没有直接接入末端位姿轨迹的逆运动学、CuRobo 规划与 MPLib/TOPP 时间参数化流程。
- RoboTwin 官方接入文档要求 config.yml、CuRobo 关节/末端/碰撞配置及坐标变换核查；当前四代手组件运行不能视为这一完整流程已完成。来源：https://robotwin-platform.github.io/doc/usage/new-embodiment.html 。

### 本次静态比对结果

| 比对 | 结果与边界 |
| --- | --- |
| 双手原始文件 | vendor_sources/left_hand.urdf、right_hand.urdf 的 SHA256 均与 source_manifest.json 记录匹配 |
| 双手导入 | 针对 robot.urdf 与 robot_sapien.urdf 共 990 项字段比对，关节类型、原点、轴、限位、父子链、惯性无差异；标准 URDF 的 mimic 引用与系数一致。仿真 URDF 按设计去掉 mimic 标签，联动在控制器执行 |
| 标准与仿真模型文件 | 两个 URDF 的当前哈希均与生成 manifest 匹配。这只证明生成版本一致，不替代真实硬件与完整源文件验证 |
| 手腕安装来源 | 保存的 sources/joint.urdf.xacro 哈希与 manifest 一致；左右手安装变换与原文件 RM65 分支一致，仅替换手部子连杆名称 |
| 双臂与本地旧包对照 | 每个 URDF 比较 98 项关节与惯性字段，各有 11 项差异。旧包为 realmanRM65/extracted/rmc_dual_arm_robot；当前模型声明来自另一版 rm_dual_arm_description，不能把跨版本差异直接判为导入错误 |

双臂差异涉及第一关节父连杆与原点、双侧第六关节原点旋转、左第六关节局部轴，以及基座连接件/第六连杆惯性。示例：旧第一关节原点 z=0.187 m，当前 z=0.2405 m 且父连杆不同；旧左第六关节 axis=[0,0,-1]，当前为 [0,0,1] 且 rpy 也改变。需要沿完整固定链及坐标约定核查，不能单独用其中一个数字判定物理轴错误。

当前部署模型目录未保存完整双臂原始 xacro，manifest 中 /home/admin123/liminghe/vla-benchmark/realman/rm/catkin_ws/src/rm_dual_arm_description 路径在该服务器已不存在。本次已检查项目缓存和迁移备份目录中直接保存的同名 xacro，未找到；未解包迁移快照。新版双臂还未完成独立逐项源文件比对，不能宣称“与厂商官方准确无误”。双手静态参数一致也不等于 SDK 编码映射、实机 TCP、伺服动态或接触耦合已标定。

本次没有修改服务器源码或模型。后续验收应先找回与 manifest 哈希一致的新版双臂源文件，核对全链及正运动学，再补齐官方规划通路，并用实际 qpos/末端位姿记录验证带时间轨迹的跟踪、同步、速度与加速度约束。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]

## 2026-10-01 新资料确认与实机关节接口对齐

用户明确确认：服务器双臂模型准确，应以当前模型为基准对齐实机；本机资料已更新。随后用户确认暂时没有 TCP、腕部 D435 外参及手指 SDK 编码映射的实机标定文件。这是用户确认与资料核对，不是现场测量。

新资料为 E:\Workspace\VLA-benchmark\realmanRM65\rm.7z，SHA256 为 5437782a46274c67dde63e65e4ce2abeec5d9e8be0fae744c86b2267c1d43a82。相关 XML、控制消息、驱动和启动定义已单独提取到 realmanRM65/extracted/rm-current-20261001/；没有覆盖服务器原 URDF。

### 对上一轮来源缺口的修正

21 个模型 xacro 及 9 个控制源文件哈希均与服务器原 source_manifest 一致。身体部分覆盖 33 个 link 和 34 个 joint，标准模型比较 271 项字段、仿真模型比较 215 项字段，共 486 项，无非预期差异。仿真版本固定底盘/头部的既有差异按配置核查，手部替换子连杆名称按既有接入方式核查。

前一节“新版双臂源文件缺失，尚不能独立核对”的缺口已被此次新资料解决。先前与 rmc_dual_arm_robot 旧包的差异是资料版本差异，不再作为当前模型正确性的疑点。当前两个 URDF 和 source_manifest 的哈希与修改前完全相同。

### 本轮已落地的硬件适配

服务器仍以 56286567103cf58bfa6191ed889908c5c3a0baa3 为提交基线，本轮改动尚未提交或推送。所有改动限定在 integrations/realman_rm65b_gen4/：

- 保存 36 个相关来源文件至 sources/vendor_workspace/，以 sources/vendor_workspace_manifest.json 绑定压缩包与来源哈希。
- 新增 hardware_alignment.json，记录 l_joint1..6 / r_joint1..6、ROS leftarm_joint1..6 / rightarm_joint1..6、左右命名空间、限位、手部安装变换及单位约定。
- 新增 hardware_contract.py，并在现有专用 controller.py 仅追加 set_arm_degrees 与 set_arm_from_joint_state 入口。六轴名称乱序按名称恢复；角度输入转换到弧度；错误侧别或不完整/重复名称拒绝。没有增加额外符号反转或零位偏移。
- MoveJ / JointPos 的 ROS joint 数组单位为弧度，底层控制器角度为度。归档驱动反馈使用 DEGREE_RAD=0.01745 的近似系数；仅显式指定 vendor_driver_rad 时修正该来源的反馈，标准 ROS 弧度仍直接使用。
- 没有修改官方 Robot/规划/任务代码，也没有改变双臂几何、既有 drive 参数或手部耦合方式。原控制器备份位于 deploy/realman_gen4/hardware_alignment_20261001/controller.before.py。

### 验证与仍待完成范围

Conda robotwin 内，30 个来源哈希检查、486 项身体字段检查、6 项名称/单位转换检查、4 项实际控制器入口的离线检查通过；原有 3 组模型合同测试通过；git diff --check 通过。此次没有启动动力学运动测试，也没有发出任何实体机器人指令。

服务器报告：integrations/realman_rm65b_gen4/validation/hardware_alignment_20261001.json。本机审阅副本：E:\Workspace\VLA-benchmark\deployment\realman_gen4_alignment_20261001/，含配置、报告、控制器及工具。资料与接口对齐不等于实机标定、连续轨迹跟踪或完整官方任务接入。TCP、腕相机外参和手指编码映射继续保留未标定；24 维规划/任务接口仍待接入。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]

## 2026-10-02 官方配置、灵巧手操作接口与头部 D435

本轮用户授权范围为补齐官方机器人接入配置、适配灵巧手操作接口并打通头部视角，完成后等待；未授权本轮开展专家任务、数据集采集、VLA评测或实体控制。用户同时澄清：头部已安装一台 D435，左右腕部相机尚未安装，先保留预留项。

### 接入实现

服务器仍以 `56286567103cf58bfa6191ed889908c5c3a0baa3` 为提交基线，使用 Conda `robotwin`。原 `robot.urdf`、`robot_sapien.urdf`、`source_manifest.json` 三个哈希均保持不变；新增 `config.yml`、SRDF、左右 CuRobo 模板/配置、碰撞球和来源清单。官方 Robot、planner 和 Camera 源文件未改；`Base_Task` 只追加配置触发的硬件类选择、厂商惯量保留、未安装腕相机保护、完整手状态附加及动作维度校验。原先的渲染去噪器改动保留。

`RealManRobot` 继承官方 Robot，臂控制继续使用 CuRobo、MPLib TOPP 与原生关节驱动；手控制只更新手部独立与 mimic 关节，不覆盖机械臂目标。官方开合操作映射为厂家示例 SDK 六通道手势；独立 `set_hand_angle` 支持六通道及 -1 保持。SDK↔角度仍是未标定的模型限位映射，未实现真实力阈值和电子触觉。

现有任务兼容动作保持 `[左臂6,左手开合1,右臂6,右手开合1]` 共14维，配置名 `realman_rm65b_inspire_gen4_synergy`。完整24维实测/驱动目标通过 `get_state24`、`hand_state` 和观测附加字段 `dexterous_state` 分别提供；这不等于完整24维数据/策略桥接完成。错误地将24维动作传给旧任务接口会明确拒绝。

模型前方对应厂商 -Y，世界放置使用 Rz(π)；独立适配层将官方桌面转换到规划器根坐标。实际核查发现 SAPIEN 固定安装 joint 与其子手掌 link 相差 Rx(π)，最终配置通过 `global_trans_matrix=diag(1,-1,-1)` 补偿。任务接近方向沿手掌 +Z，0.18m 偏置是未实机标定的几何参考。

### 相机状态与证据边界

头部图像绑定厂家 `camera_link`，通过官方 Camera/Base_Task.get_obs 返回320×240 RGB、合成深度和内外参，机器人根坐标改变时相机跟随。腕相机配置保留，但 `collect_wrist_camera=false`，误开启明确报错。

原模型头关节固定零位，按外壳前向推断的零位视角看不到近桌面；默认使用明确标为 `sim_optical_pitch_24_deg` 的24°光学仿真参考视角，另保留 `vendor_zero_view`。光心位置由外壳网格估计，不是实测镜头中心；没有把头部关节实际设置为24°，没有修改原模型，也未完成实机光学标定。40°只用于诊断对比，不是默认。RoboTwin D435 预设的37°视场及渲染深度不等于真实 D435 内参或主动双目噪声模型。

### 本轮实际验证

- 9项灵巧手、动作合同及场景重置测试通过。
- 官方场景中双手开合操作、14维兼容观测、额外24维实测状态、厂商质量保留、双臂 TOPP 小幅驱动和同任务重复初始化通过；两臂终点最大关节误差约3.87e-5rad、3.06e-5rad。
- 左右零位/扰动位四组 CuRobo FK 与 SAPIEN link 比较，最大位置误差约5.89e-7m；末端姿态转换往返通过。
- 左右小幅末端位姿规划均成功，各65个路径点，六关节轨迹有限且未越限；实际驱动后末端终点误差约0.0341mm、0.0270mm。这是仿真局部终点验证，不是连续跟踪精度或实机精度。
- 头部 RGB/合成深度/内外参、桌面三标记投影、根坐标平移2cm跟随、腕相机禁用与配置冷启动通过；原 Aloha 图像与原生小幅控制回归通过。

最终配置 SHA256：`4776ce11bd15222688e96ff61eef625a2baeffb0084f56d57c7a25cdacc77768`。早期 global=I 的组件运行已单独标为修正前记录，最终验收绑定上述配置，不能混用。

证据：服务器 `deploy/realman_gen4/native_support_20261002/`；规划报告 `integrations/realman_rm65b_gen4/validation/native_planning_20261002.json`；本机审阅副本 `E:\Workspace\VLA-benchmark\deployment\realman_native_20261002\delivery`。本机旧 RoboTwin checkout未被覆盖。所有本轮新增支持与注册改动尚未提交或推送。

规划器仍按独立臂与固定张手近似碰撞模型规划，相邻手指球体假重叠有显式豁免；未完成动态全身双臂/任意手势避碰、完整接触抓取任务、连续轨迹跟踪、24维数据策略适配或真机迁移。本轮已停止于硬件组件支持阶段，等待用户后续安排。

来源：[官方新机器人接入规范](https://robotwin-platform.github.io/doc/usage/new-embodiment.html)、[官方相机接入规范](https://robotwin-platform.github.io/doc/usage/new-camera.html)、2026-10-02服务器实际运行报告和用户相机安装说明。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]

### 2026-10-02 头部截图物体身份澄清

用户质疑头部截图出现夹爪。本次使用同一模型与配置重建测试场景，只做对象分割和视图核查，没有运行专家任务或采集数据。渲染实体 ID 确认黑色轮廓属于 `020_hammer`（羊角锤，ID 104）；该头部画面仅包含墙、桌、锤子和红方块，双臂与灵巧手均未入镜。临时场景移除锤子后的对照图中，黑色轮廓随之消失。

补充外部整机视图及左右手特写，确认加载的仍为 `realman-rm65b-inspire-gen4/robot_sapien.urdf` 与双五指灵巧手；URDF SHA256 `4bf28c444af1a28401a975e5de6e138c1ec043536aa786ea541c30c359bd65bd`、配置 SHA256 `4776ce11bd15222688e96ff61eef625a2baeffb0084f56d57c7a25cdacc77768` 均未改变。核查未修改源码、机器人模型或相机配置。

这只能证明模型与图中物体身份。头部图仍使用 `sim_optical_pitch_24_deg`，光心由网格估计、光轴人为下俯24°，并非实机标定结果，也没有实际转动模型头关节；不应将其描述为已与用户实机对齐的头部视角。

证据：[对象身份报告](file:///E:/Workspace/VLA-benchmark/deployment/realman_native_20261002/view_identity/view_identity_report.json)、[整机核查视图](file:///E:/Workspace/VLA-benchmark/deployment/realman_native_20261002/view_identity/whole_robot.png)、[左手特写](file:///E:/Workspace/VLA-benchmark/deployment/realman_native_20261002/view_identity/left_hand.png)、[移除锤子后的头部图](file:///E:/Workspace/VLA-benchmark/deployment/realman_native_20261002/view_identity/head_without_hammer.png)。服务器原始证据目录为 `deploy/realman_gen4/view_identity_20261002/`。后续工作仍等待用户安排。
## 2026-10-02 双臂桌面可达区域与头部视锥核查

用户要求查看双臂桌面操作范围和头部视角范围。本次在服务器 Conda `robotwin` 中进行离线端点逆解及相机几何投影计算，未执行轨迹、专家任务、数据集采集或实机控制，未修改原模型、源码或配置。基线提交与上一节相同；URDF/配置哈希保持 `4bf28c...bd65bd` / `4776ce...77768`，完整值在报告中。

### 方法与结果

桌面1.20×0.70m，高0.74m，近边Y=-0.35m；机器人根坐标Y=-0.65m。操作点为手掌基座沿局部+Z偏移0.18m的未标定名义TCP。分别采样桌上方5cm和10cm两层，每层336个5cm网格中心、24个向下/斜向姿态（0°/30°/60°倾角×8个方位），每姿态32个种子，保留8个运动学逆解候选，再检查现有单臂/固定张手CuRobo端点碰撞约束。

| TCP距桌面高度 | 左臂运动学可达点 | 右臂运动学可达点 | 左/右通过规划端点约束点 |
| --- | --- | --- | --- |
| 5cm | 137/336 | 137/336 | 0 / 0 |
| 10cm | 143/336 | 145/336 | 143 / 145 |

10cm平面中，两臂任一可达205点（网格面积估计0.5125m²，61.0%）；两臂分别均可达83点（0.2075m²，24.7%）；其中进入当前相机视锥29点（0.0725m²，8.6%）。最远成功采样点距近边42.5cm。所有成功逆解限位检查通过，两臂各16个候选经完整SAPIEN模型正运动学抽查，满足位置/姿态阈值。这是有限姿态和种子的端点证据，不是完整连续操作范围、同时双臂路径或实机抓取保证；着色方格只代表中心采样点，未搜索成功的点不能判为绝对不可达。

### 新发现的近桌面约束问题

5cm平面中，左8025个、右8026个保留的运动学成功候选全部被桌面碰撞约束拒绝，自碰撞拒绝为0；同一批碰撞球与真实渲染桌面盒体也相交。说明不能仅靠降低规划桌面的高度就宣称贴桌抓取已就绪，仍需核查名义TCP、张手碰撞近似与具体手势。

另外发现规划器桌面盒体尺寸为2.0×0.7×0.04m、中心Z=0.74m，即上表面0.76m；渲染桌面宽1.2m、上表面0.74m，存在宽度差与20mm表面高度差。10cm层左臂有2776个候选被规划桌面拒绝，但所有运动学成功候选的碰撞球与实际渲染桌面均不相交；另有可行候选使网格级覆盖仍为143点。本次只记录差异，没有修改已有配置。

### 头部相机与操作区域的重叠

当前仿真光心约[0.0005,-0.61395,1.26915]m，水平视场48.09°、垂直37°、24°人为光轴下俯。视锥与0.74m高桌面相交面积0.3170m²，占37.7%；近侧约31.4cm深桌面不在视野内，近端可见宽66.3cm、远端97.8cm。光轴落在桌面高度平面的交点超出桌子远边22.5cm。固定模型零位的水平参考视角不覆盖当前桌面。

在桌上方10cm，视锥几何面积为0.3665m²（43.6%），从距近边20.4cm开始；因此操作范围叠加图的相机边界与桌面视野图不同。当前可达区域偏近侧、视野偏远侧，共同可达且可见的区域较小。结果均未计遮挡或实机光学标定，也未验证躯干/另一只手臂碰撞与连续路径。

证据：[范围说明与方法](file:///E:/Workspace/VLA-benchmark/deployment/realman_workspace_20261002/RESULTS.md)、[操作范围叠加图](file:///E:/Workspace/VLA-benchmark/deployment/realman_workspace_20261002/workspace_overlay.png)、[桌面相机覆盖](file:///E:/Workspace/VLA-benchmark/deployment/realman_workspace_20261002/head_coverage.png)。逐点CSV、全部保留逆解NPZ、参数哈希及约束分解在同一目录；服务器证据目录 `deploy/realman_gen4/workspace_20261002/`。初始低效带碰撞优化扫描主动停止，不作为最终统计；最终采用 `scan_candidates.py` 的多解筛选结果。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]
### 2026-10-02 仿真场景图像导出

按用户要求，从服务器SAPIEN实际场景导出整机、桌面近景、俯视和头部画面，模型与配置哈希保持不变。场景为 `beat_block_hammer` seed 0，仅初始化；保留锤子与红方块，未执行专家任务或轨迹。另将上一轮有效端点逆解静态赋给左右臂：名义TCP分别为[-0.175,-0.075,0.84]和[0.175,-0.075,0.84]m；这是静态候选姿态预览，不是双臂路径、避碰或抓取成功的证据。

头部高清图为同光心、同37°垂直视场的1280×960新渲染，另保留官方320×240观测输出；继续使用未实机标定的24°光轴预设。图中桌面姿态使双五指灵巧手在头部画面下缘入镜；没有修改原始相机配置。模型、头部画面和物体均由渲染器输出，没有AI图像生成或后期重绘。

本机证据：[整机视图](file:///E:/Workspace/VLA-benchmark/deployment/realman_render_20261002/tabletop_external.png)、[头部视图](file:///E:/Workspace/VLA-benchmark/deployment/realman_render_20261002/tabletop_head_hd.png)、[俯视图](file:///E:/Workspace/VLA-benchmark/deployment/realman_render_20261002/tabletop_top.png)；同目录保存 `render_scene.py` 与 `render_report.json`。服务器目录：`deploy/realman_gen4/render_scene_20261002/`。
### 2026-10-02 左手预览朝向修正

用户指出上一张头部画面左手像是反了。核查确认：旧预览左侧露出手背，右侧露出掌面。左右手原始URDF及安装文件与manifest一致，RM65左右安装变换仍为[0,0,0.028]/[0,0,0.032]m、rpy=[0,0,3.1416]；掌面与拇指结构表明厂商双手沿局部X镜像，两侧掌面均朝局部+Y。问题是离线预览对左右臂分别选择了低关节角代价的候选姿态，没有检查两掌的解剖镜像朝向，并非此次导入时将左右手模型调换。

新预览以 `S @ R_right @ S`（S=diag[-1,1,1]）约束左掌镜像，针对这一个选定姿态将左第6关节从约-100.407°改为79.593°，变化180°；掌面镜像方向误差从180°降至数值精度内，名义TCP位移约1.13e-7m。修正端点关节限位与原有单臂规划碰撞检查通过。此180°是该预览姿态的修正，不是模型、SDK、零位或所有左臂指令的固定补偿。

原渲染与脚本保留，新增镜像检查版本 `deployment/realman_hand_orientation_20261002/render_scene_mirrored.py`。URDF、官方源码、安装变换、相机和接入配置均未改变；未执行运动轨迹或实机指令。上一轮范围扫描没有重跑，原数字仍限于当时的独立单臂采样姿态集，不能升级为掌面镜像匹配的双臂任务范围。

证据：[修正后头部渲染](file:///E:/Workspace/VLA-benchmark/deployment/realman_hand_orientation_20261002/tabletop_head_hd.png)、[修正后近景](file:///E:/Workspace/VLA-benchmark/deployment/realman_hand_orientation_20261002/tabletop_close.png)、[朝向检查报告](file:///E:/Workspace/VLA-benchmark/deployment/realman_hand_orientation_20261002/render_report.json)。
### 2026-10-02 头部机械下俯限位视角预览

按用户要求将仿真头部转到最低位置。核查发现现有 `robot_sapien.urdf` 将 `head_joint1/2` 固定，而厂商 `body_head.urdf.xacro` 与标准 `robot.urdf` 保留头部两自由度。此次在独立派生预览模型中恢复两关节，origin、axis、parent/child、effort/velocity和位置限位均逐项核对厂商来源；原URDF、config与manifest哈希不变。

预览偏航置0，俯仰置厂商下限 **-0.419 rad（向下约24.0069°）**。对正负两端分别做正运动学，确认负值朝下。相机采用壳体局部+Y前向参考，去掉上一轮人为24°光轴下俯，避免和实际头部关节下俯叠加；光心随头部前移约5.79mm、下降33.78mm。独立URDF正运动学与SAPIEN相机矩阵最大元素误差约3.23e-7。除头部及相机链外其他link变换变化为0，保留已修正的双掌镜像姿态；14维任务观测和24维扩展状态形状保持。

在同一37°垂直视场、1.20×0.70m、Z=0.74m桌面下，视锥几何覆盖由37.74%增至39.31%（0.3302m²），近边盲区深度由31.35cm缩至28.23cm。改善不大，因为先前已人为设置24°光轴倾角，此次是以真实头部关节下俯替代该预设，方向近似相同。实际SAPIEN渲染中双手、锤子和红方块可见，仍有较多墙面区域。覆盖数字不计遮挡；光心、光轴、相机内参均仍未实机标定。

范围：服务器Conda `robotwin`，官方场景仅初始化；仅静态关节姿态与视角检查，未测试头部电机轨迹跟踪、双臂任务、数据生成或实机。默认模型仍保持原配置，独立预览增加头部自由度不等于已完成通用头部动作接口。

证据：[最低俯仰头部画面](file:///E:/Workspace/VLA-benchmark/deployment/realman_head_down_20261002/head_down_head_hd.png)、[头部下俯外观](file:///E:/Workspace/VLA-benchmark/deployment/realman_head_down_20261002/head_down_external_close.png)、[说明与复现](file:///E:/Workspace/VLA-benchmark/deployment/realman_head_down_20261002/RESULTS.md)、[验证报告](file:///E:/Workspace/VLA-benchmark/deployment/realman_head_down_20261002/render_report.json)。服务器目录 `deploy/realman_gen4/head_down_20261002/`，原预览未覆盖。关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-02 官方其他机器人头部机位对照渲染

在服务器172.17.27.166、Conda `robotwin`、RoboTwin commit `56286567103cf58bfa6191ed889908c5c3a0baa3` 下，实际渲染已安装的ALOHA AgileX、Piper、ARX-X5、Franka Panda和UR5-WSG。五份官方 `config.yml` 的 `head_camera` 均为固定世界机位[-0.032,-0.45,1.35]m，forward=[0,0.6,-0.8]、left=[-1,0,0]，对应**向下53.130°**；由官方Camera类创建，不跟随可动头部。因此“头部相机”名称不能证明实机是头部安装，也不能与睿尔曼头部机械限位直接等同。

五者均用D435的37°垂直视场，光轴与Z=0.74m桌面交于[-0.032,0.0075,0.74]m，接近桌面中心；1.20×0.70m桌面的视锥覆盖约0.42047m²（50.06%，不计遮挡）。实渲画面以桌面为主，较睿尔曼24.01°下俯画面保留更少墙面。初始位姿的机械臂主要出现在边缘，部分夹爪位于视野外。

此次使用同一 `beat_block_hammer` 静态初始化场景，独立固定物体随机流以消除不同相机数量对随机采样的影响；五份锤子/红方块变换矩阵一致。保留官方初始关节配置；ALOHA为完整双臂整机，其他模型按官方左右臂加载方式以展示间距0.60m配对，此间距不是声称的官方默认值。静态显示将超出URDF限位的夹爪张开目标截至限位，并记录原始/显示值。未执行物理轨迹、专家任务或采集，未验证操作成功。官方URDF/config、Camera代码与相机配置哈希前后一致。

每种平台均交付1280×960同机位高清头部图、320×240原生图、整机外观图和JSON验证报告。证据：[五种平台图像入口与复现说明](file:///E:/Workspace/VLA-benchmark/deployment/official_head_views_20261002/RESULTS.md)、[ALOHA头部视角](file:///E:/Workspace/VLA-benchmark/deployment/official_head_views_20261002/aloha-agilex/head_hd.png)。服务器目录 `deploy/official_head_views_20261002/`。关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-02 用户确定保持实机一致的视角策略

用户明确选择“保持实机一致”。后续保留厂商头部机械限位和安装关系，不通过增加虚拟光轴倾角、扩大关节限位或套用官方其他机器人固定机位来声称实机视野改善；尚未获得实测内外参时，证据只能表述为与厂商模型一致，不能表述为已完成实机相机对齐。

本次发现服务器默认config仍选用旧的 `sim_optical_pitch_24_deg`，已仅将 `head_camera_mount.optical_preset` 改为 `vendor_zero_view`，同步更新 `NATIVE_SUPPORT.md`。旧预设保留作历史诊断，不再默认启用；其余配置字段不变。重新加载配置并构造MountedHeadCamera，局部前向为+Y（float32最大误差1.19e-7），Z分量为0，无额外光轴下俯；光心位置未变。`robot.urdf`、`robot_sapien.urdf`及source_manifest哈希与上一轮实渲报告一致。

默认URDF仍固定头部零位，因此默认水平参考视角不能覆盖近侧桌面；此前 `head_down_20261002` 的厂商头部机械下俯-0.419 rad独立预览继续保留，不能混称默认已具备头部运动接口。头部最低位置预览也仍使用未标定壳体光学参考。未修改硬件、执行运动或新增数据采集。

配置前后备份、文档diff与验证证据：[变更差异](file:///E:/Workspace/VLA-benchmark/deployment/realman_physical_view_policy_20261002/changes.diff)、[验证报告](file:///E:/Workspace/VLA-benchmark/deployment/realman_physical_view_policy_20261002/verification.json)。服务器证据目录 `deploy/realman_gen4/physical_view_policy_20261002/`。关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-02 首条双臂到位任务数据生成

用户本轮授权从硬件接入推进到一个简单任务的自动化数据生成，且仅生成第一个。已在 [[服务器172.17.27.166-robotwin]] 的 Conda `robotwin`、GPU 2 实际执行；服务器 RoboTwin HEAD 为 `56286567103cf58bfa6191ed889908c5c3a0baa3`。全程仅仿真，无实机指令、训练或后续批量采集。

**唯一成功样本**为新增独立任务 `realman_reach_targets`，seed 0、`episode_0000000`：左手到蓝色目标、右手到橙色目标，目标为桌面上方10cm的可视球，无碰撞形状，不要求接触或抓取。使用此前核验的镜像手掌朝向与名义TCP（手掌局部+Z 0.18m）。沿用 Base_Task 的初始化、CuRobo规划、原生关节驱动物理执行、成功判定、轨迹保存、重置回放和官方HDF5/视频导出；没有直接将目标qpos赋值为成功。

- 首轮左右规划分别449/458点，规划关节限位合法；物理执行与回放均满足双手位置误差小于10mm、朝向误差小于5°的判据。
- 回放实际模型末端误差：左0.3006mm/0.0381°，右0.3572mm/0.0332°。仅为仿真内部名义TCP误差，不能写成实机精度。
- 保存87帧原始观测、86组状态—动作配对，仿真时长4.828s，只有头部320×240 RGB、合成深度和内外参。逐帧解码、状态动作对齐、有限值、测量扩展、源文件稳定和成功指标等16项读回检查全部通过。
- 唯一HDF5的SHA256为 `5c262670a74bdc58a9f53fc393ed0e87b0929e7277af6e8e482a1685c1320bde`；本机副本与服务器一致。

**数据与视觉约定**：官方原生任务仍为每侧arm6+scalar，共14维；当前导出器原生state机械臂字段来自命令通道，不能混称测量。额外 `realman_dexterous` 组保留实际模型24维测量、下一采样命令及仿真时间戳，与原始PKL逐项对应；未启用独立24维策略训练桥接，model-SDK映射仍未实机标定。`save_freq=15` 是250Hz物理步的保存间隔，动作边界产生2组零时长配对；应读取显式时间戳，不将原生frequency字段15当作均匀15Hz。官方视频87帧/30FPS约2.9s，另导出按物理时间重采样的145帧/30FPS、4.833s等速预览，均来自实际保存图像。

本次采用派生URDF将头部固定在厂商机械最低俯角-0.419rad，偏航0，光学预设仍 `vendor_zero_view`、额外光轴下俯0；保留臂/手36DOF，原始厂商模型和默认配置不变。不是通用头部电机控制。相机仍未实机标定，头部画面保留较多墙面，没有改成官方其他机器人的虚拟俯视机位。运行配置单独将规划桌面设为中心Z=0.715m、1.2×0.7×0.05m，与渲染桌面上表面0.74m一致；原默认规划盒体未改。

**前置失败与必要修复**：先试官方 `click_bell` seed0，首次下降规划失败。排查发现硬件专属 `native_robot.py` 的scalar增量控制原来相对滞后的测量手位递增，驱动目标无法持续前进；改为相对上一驱动目标，匹配官方Robot语义。SDK通道-1保持仍以测量值为准。新增滞后回归在修复前失败，修复后全部9项测试通过。修复并对齐桌面后，按铃三段规划成功，但未触发真实铃接触，任务失败，未生成成功数据；中间一次预热尝试主动停止。失败报告与修复前后文件均保留，不计入唯一成功episode。官方已有任务与核心源码未在本轮修改。

证据：[结果说明与复现命令](file:///E:/Workspace/VLA-benchmark/deployment/realman_first_episode_20261002/RESULTS.md)、[数据读回检查](file:///E:/Workspace/VLA-benchmark/deployment/realman_first_episode_20261002/realman_reach_targets_native_seed0/dataset_validation.json)、[等速预览](file:///E:/Workspace/VLA-benchmark/deployment/realman_first_episode_20261002/realman_reach_targets_native_seed0/video/episode_0000000_realtime.mp4)、[唯一HDF5](file:///E:/Workspace/VLA-benchmark/deployment/realman_first_episode_20261002/realman_reach_targets_native_seed0/data/episode_0000000.hdf5)。服务端目录：`/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/first_episode_20261002/`。本机同名目录保存原始PKL、轨迹、报告与新增任务副本；采集脚本禁止覆盖已有尝试。

结论限于这个自定义到位任务的自动规划、物理回放与采集导出闭环已跑通，不是抓取/按压完成、动态双臂/躯干避碰保证、多种子成功率或实机迁移成功。本次语言通过官方模板生成，未调用在线LLM自动发明任务。仅生成首条后停止。关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-02 用户纠正任务范围：改为官方 Adjust Bottle

用户明确指出：上一条 `realman_reach_targets` 只是移动，且手臂从画面外突然进入，不能满足其需要。新的验收对象应为 **RoboTwin 2.0 官方物体操作任务**；此前自定义到位样本仅保留为采集链路测试证据，不能作为已完成用户要求的官方任务数据。

本轮在线核对 [官方任务列表](https://robotwin-platform.github.io/doc/tasks/index.html) 与 [Adjust Bottle说明](https://robotwin-platform.github.io/doc/tasks/adjust_bottle.html)：列表第一项是 `adjust_bottle`，要求用正确侧手臂抓起瓶子并保持瓶口朝上。列表未标注按难度排序，不能将“第一项”直接称为“官方认定最简单”。本地官方源码为抓瓶、抬升、转正/移到目标；原有成功条件检查瓶子功能点0的左右位置和高度大于0.9m。准备保留该任务与原成功判据，额外核对实际手—瓶接触及保持后成功，防止仅动作到位或物体飞过目标被当成有效抓取。

为改善起始画面，准备以先前实际物理到位的镜像双臂末态作为**候选**homestate，通过官方关节驱动初始化，录制前检查稳定误差和名义TCP视锥位置，并保留0.8s起始停留。实际初始碰撞、完整手部可见性、抓握接触与回放均尚未验证。相机继续为厂商头部机械下限-0.419rad、额外光学倾角0；不扩大关节限位、不改变厂商安装关系。

**当前阻塞**：本轮两次连接 `172.17.27.166:22` 均在认证前超时；Windows只读检查显示 OpenVPN TAP、Wintun、DCO 网卡均Disconnected，目标路由走WLAN的默认路由。未改变网络配置，未获得本轮服务器状态。已向用户请求重新连接学校VPN。

本机准备包：`E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/`，含官方任务本地源码快照、`run_adjust_bottle.py`、候选 `ready_home.json`、校验与等速预览脚本、Conda `robotwin` 启动入口及 `preparation_status.json`。这是**未部署、未运行、待语法和运行验证**的准备包，没有生成新的官方任务episode，也没有宣称灵巧手抓瓶适配完成。恢复连接后先核对服务器源码/模型版本，再执行并依据实际失败诊断修正硬件抓握适配，直到首条官方任务通过后停止；不得再次以自定义移动任务替代。

关联：[[RoboTwin 2.0框架与默认任务]] · [[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-03 官方 Adjust Bottle 实测：未通过，保留失败视频

学校VPN恢复后已通过SSH连接服务器，在Conda `robotwin`、GPU 2执行官方 `adjust_bottle`。官方任务源码SHA256为 `4f76b35e718a4c5a364dae564d35d68268755818d9062e3ed4ff393d158d0548`，RoboTwin HEAD为 `56286567103cf58bfa6191ed889908c5c3a0baa3`。**合格官方episode为0，没有新增HDF5；此前自定义到位样本不计为本次用户要求的任务成功。**

沿用官方专家的抓瓶、沿接近方向抬升、转正/移到指定一侧与原成功判据。在独立运行脚本中测试五指手抓握几何、开合预设、驱动参数及仿真接触限力；部分原生CuRobo路径与关节限位检查通过，但物理闭手会使瓶子偏移，抬升后失去稳定夹持。未完成官方操作、重置回放和正式采集。触觉限力诊断读取的是PhysX物体接触，不能写成实机触觉已接通；SDK模型映射也仍未标定。所有未通过参数均保留在诊断包，未写入默认适配器。

头部继续厂商机械下限−0.419rad、`vendor_zero_view`、额外光轴下俯0。双臂提升后的候选初态通过原生驱动稳定，实录最大关节误差约0.00036rad，刷新相机后名义TCP位于视锥；实际只保证画面两侧能看到部分手指，瓶子仍部分位于画面底边，没有声称完整手部/目标入镜或实机相机标定完成。腕部相机未启用。原始模型、默认config、官方任务、Base_Task和既有native_robot的保护哈希均未变。规划仍不覆盖另一臂/躯干，不能据此宣称全身动态避碰。

已保存两段**失败诊断视频**：头部320×240、旁观640×480，均191帧/25FPS、7.64秒，来自实际每10个250Hz物理步的渲染，逐帧解码及本机/服务器哈希核对通过。源物理时间末帧7.60秒；编码最后一帧显示0.04秒。包含0.8秒初始停留。没有附着/焊接瓶子、修改物体质量/重力/碰撞来制造成功，也没有实机IO、训练或批量采集。

新增诊断证据确认：接触列表中的零冲量邻近点不能当作握持成功。后续独立验收额外要求正确手非零接触冲量、瓶口竖直误差<15°、保持1秒位移<20mm，再通过原生回放和数据读回；这些是后续验收要求，不是已通过结果。49份尝试/审计报告包括初始化、规划失败、物理失败和时限中断，不构成多种子成功率统计。初始homestate共享变量造成的报告元数据覆盖已保留原报告并按初始配置/测量修正；首次录制相机未刷新导致的检查异常与修正后的v2记录均保留。诊断进程已停止。

证据：[结果与复现说明](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/RESULTS.md)、[逐帧与哈希检查](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/diagnostic_validation.json)、[头部失败视频](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/official_adjust_bottle_diagnostic_v2_seed7/failed_diagnostic_head_realtime.mp4)、[旁观失败视频](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/official_adjust_bottle_diagnostic_v2_seed7/failed_diagnostic_external_realtime.mp4)。服务器目录 `deploy/realman_gen4/adjust_bottle_20261002/`。后续需先完成可承力的抓握几何与闭合时序验证，再继续同一官方任务的转正和首条有效数据生成。

关联：[[RoboTwin 2.0框架与默认任务]] · [[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-03 持续完善：无推瓶轨迹与物理联动（进行中）

用户明确要求继续到首条合格官方任务轨迹，接近和抓取不得推移/碰倒目标。已建立持续目标：保持官方 `adjust_bottle`、厂商整机/手部模型和头部限位，接近阶段无有效手—瓶冲量且瓶子最大位移≤1mm/转角≤1°；闭合保持位移≤2mm/转角≤2°；抬升转正后拇指与其他手指持续相对法向承力、离桌、手内漂移≤2mm/2°并保持1秒，随后原生重置回放、读回唯一数据及真实等速视频。阈值是本次仿真工程验收，不是实机标定精度，也不代表求得全局最优。

只读并行审计确认旧同参数 `contact_force_04_seed7` 首个采样到的真实冲量发生在step400（1.6s、采样间隔0.1s）：全开无名指在预抓取关节路径中推瓶，晚于此才闭手；不能将这个时间当作v2失败视频的精确首撞时刻。原规划世界只有桌面，路径Success不保证避瓶。另查明 `robot_sapien.urdf` 移除了mimic，原适配器只同步目标、所有指节独立PD，实际受阻联动未被约束。新增独立物理联动与完整网格审计helper，原官方任务/厂商模型/config/既有适配器保护文件保持不变。

完整辅助碰撞模型直接复制PhysX几何并处理瓶子0.132缩放，93链接FK校验通过，最大位置/旋转误差约6.19e-7m/1.33e-6rad；线性关节插值按完整碰撞顶点位移上界≤1mm细分，包含躯干、另一臂、所有指节与传感器，并保留实际引擎已有固定/邻接过滤。候选B10°的第一段预抓路径瓶子连续间隙34.48mm、桌面97.97mm、自碰间隙1.75mm、无碰撞。v5被实测未命令手部3.76e-5rad数值越限的统一阈值拦截，尚未执行接近/闭手；需分别审计严格命令限位和实测数值容差，不能记为抓握失败或成功。

候选B/A侧向接近的早期物理测试在最后近接时被guard中止，瓶移仅约0.15mm以内；实际掌Z偏离约5mm、朝向偏离约1.15°解释了静态间隙仍提前触拇指。正在将最后接近换为框架已有MPlib screw直线并降低关节速度，闭合按全部指节厂家1rad/s速度上限重定时。恢复物理mimic后已能执行，但稳定保持时还有拇指速度抖动；高刚度1e4/1e5试验NaN，较低刚度受控对照保留失效边界，不能只以finite记为通过。SAPIEN3实际库默认材料为0.3/0.3/0.1，场景的Python default_physical_material赋值未保证传递给URDF；手部显式材料与传动阻尼正在独立对照，尚未写入默认正式配置。

**当前仍为0条合格官方episode、未新增HDF5，目标继续进行中。** 没有实机IO、训练、物体附着、模型换型或关闭外部碰撞。脚本和证据：[完整网格helper](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/full_mesh_path_audit.py)、[93链接FK](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/full_mesh_fk_validation_20261003.json)、[候选v5路径审计](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/coupled_side_entry_v5_B10_20261003_seed7/mesh_path_1.json)。服务器工作目录仍 `deploy/realman_gen4/adjust_bottle_20261002/`、Conda `robotwin`、GPU2。关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。
#### 同日继续核查：接近已通过，闭合与承重仍待验证

v6 使用原生 MPlib screw 做最后近接、保留官方专家任务；完整网格证书的最后近接瓶间隙下界4.644mm、桌面18.638mm、自碰1.700mm。物理接近无有效手—瓶冲量，瓶子最大位移约0.133mm；随后中指比拇指先接触40ms，闭合把瓶子推至2.002mm触发保护。静态精确拇指阈值约917.9 model-SDK；失败记录中的935 SDK已含瓶子被推向拇指后的变化，不能当成正确的未推瓶预闭合目标。

v7 独立 PhysX 接触反馈在首触冻结实际关节目标，闭合/保持最大瓶移1.50mm、转角1.42°，但官方撤退时瓶子滑脱留桌，转正规划失败。该候选四指法向偏向下压；用真实接触力的摩擦锥上界计算，µ=.3不足以产生向上支撑。不是完整抓取，也未生成合格数据。正在核查 B085 下方托持候选（hand_y=.085，pitch10°，hand_z=.165）：静态开手桌面余量约9.83mm、首触形状12.69mm，净力可行性优于原B10；完整质心/力矩与动态成功尚待实测。

另直接读取服务器 articulation，全部36关节的实际摩擦默认均为.05，包含手指；这不是URDF明确提供的厂家摩擦标定。独立CPU对照仅将24手关节设为.001、保留12臂.05：原手型稳态真实位置差分速度峰值由.23752降至.00465rad/s，0.2s角幅由.00910降至.000189rad，mimic残差由.000994降至.00001895rad。手部.001及显式接触材料.5/.5/0均属未标定仿真参数，不能称实机精度或厂家触觉控制。部分返回qvel与每步真实位置变化不一致，必须分开记录。

新增连续SDK6多段quintic helper保留整数SDK接口，按全primary/mimic速度限位重定时、仅给primary解析速度，接触后速度归零；左右手14项纯CPU检查通过，这不等于官方任务成功。V8 独立候选补上首触后其他手指减速、实际连续成对承力与手内漂移的撤退前门禁，待物理验证。VPN中断后恢复，原SSH绑定worker停止，改用独立日志与超时的worker4继续；Conda仍robotwin、GPU2。当前仍0条合格官方episode/0新增HDF5，持续目标未完成，原模型和官方源码保护哈希未变。

证据：[连续驱动helper](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/continuous_servo.py)、[精确接触几何](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/grasp_geometry_v6_refinement_20261003.json)、[物理联动交接](file:///E:/Workspace/VLA-benchmark/deployment/hand_coupling_20261003/hand_coupling_handover_20261003.md)。

#### 同日继续验证：三指静态抓持通过，完整持瓶任务仍未完成

v8 B085 的实际末端较规划目标仍偏约2.2mm，首触小指比静态阈值提前，拇指未到位；闭合在瓶移约1.18mm、旋转2.001°时中止。读取 installed MPlib `plan_screw` 实现确认默认 qpos_step=.1 会累积末端积分误差；通过现有 API 将精细接近的 qpos_step 设为 .005，独立末端FK预检位置误差2.70µm/朝向误差0°。这是仿真规划内部一致性，不是实机精度。

v9 保持相同厂商模型，采用中指、食指和拇指主三指抓持，小指/无名指保留打开余量。完整接近路径的瓶间隙下界1.847mm、桌面9.343mm、自碰1.707mm；闭合全过程瓶子最大位移1.126mm、转角1.919°，满足本轮2mm/2°门禁。撤退前真实相对法向承力连续3012步（12.048s）；额外2s保持的手内平移漂移0.056mm、旋转0.048°。实际拇指约1.412N、中指.576N、食指.827N，相对法向dot约−.9999985，不能用零冲量邻近点替代这些指标。

**v9仍未完成任务**：随后官方方向撤退虽抬起瓶子、三指仍有法向冲量，但原默认转正规划失败。原先约20mm手内滑动的估计混用了刚性假设目标与实际末端跟踪误差，不能作为手内漂移证据；同日用实际双臂关节日志与厂商URDF逐帧重算，得到撤退阶段手内最大平移约0.960mm、转角0.331°（当前完整FMA独立交叉核查得到0.9596mm/0.33048°，历史抓持参考位置误差0.513µm/朝向误差0.000066°）。该结果不能写成稳定抬升、完整抓取、专家成功或已采集数据。进一步使用官方 `get_place_pose` 已支持的 `constrain='align'` / world actor_axis / align_axis 参数核查8组16端点；`carry_ee_camera_z__x_minus` 两端点IK成功、目标关节限位合法，保留官方FP0目标[-.25,-.12,.95]与瓶口竖直。它只证明端点可达，不证明路径避碰或物理成功。

新增纯CPU六维接触力 LP，每次按真实世界接触点、施加在瓶子上的法向与读回质量中心求解；µ=.35、16边内摩擦锥，中/食/拇指下限.1N、上限1.5/1.5/2.5N。10项CPU测试通过，v9实际初始接触的候选法向分配约[.5892,.8277,1.4439]N；较大旋转需控制速度或补足期望惯性wrench。LP可行不能保证单通道实现多点分力，也不能替代实际手内滑动/丢失接触检查。v10正在独立worker4验证：小角抬升使用精细screw、逐步滑动保护、慢转正和真实接触分配；完整持瓶网格审计通过前不推进相应路径。

当前0条合格官方episode/0新增HDF5，原生重置回放及数据读回仍待完成。物理联动K100/D.02、手关节摩擦.001和手部材料.5/.5/0均属未实机标定的仿真参数；瓶子保留原.3/.3/.1材料、质量/惯量/质心、重力和碰撞，材料组合摩擦约.4不能误写成.5。不改官方任务、原Base_Task、既有native_robot或厂商URDF/config保护哈希；无实机IO。

证据：[v9实际报告](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/coupled_side_entry_v9_B085_20261003_seed7/attempt_report.json)、[CPU承力helper](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/purecontact_lp.py)、[CPU检查](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/purecontact_lp_cpu_results_20261003.json)。服务器证据仍在 `deploy/realman_gen4/adjust_bottle_20261002/`。关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B双臂机器人]]。

#### 完整持瓶网格预检与历史漂移更正

v10 三指抓持保持通过，额外2s手内漂移约0.0466mm/0.0529°。原方向抬升的完整持瓶路径检查发现 `body_base_link` 与 `l_link4`、`l_link5` 相交，4140个细分样本中对应碰撞出现3224次，已在物理carry之前拦截；没有新的成功episode。现有CuRobo同目标路径也有相同本体碰撞。世界Z+.1m的CuRobo候选逐点无碰撞，但近桌处瓶/table的连续保守下界仍为负，必须通过独立递归细分正间隙证书后才能执行，不能直接放宽碰撞阈值。

原任务显式 `move_axis='arm'` 在该侧入五指手姿态下实际后退约98mm、升高约17mm；正在对照官方工具已有 `move_axis='world'` 选项与完整模型多IK/RRT。所有变化位于独立硬件运行profile和helper，官方任务及核心源文件仍保持保护哈希。新的aux PlanningWorld端点预检曾报瓶子与本体相交，但独立诊断证实其AttachedBody返回的是机器人base坐标，缺少整机root的Rzπ和平移：与实测瓶位置差0.508m、朝向差π，是无效辅助碰撞。先前怀疑静态组件名称为空不是已证实根因；已独立命名为realman_table/held_bottle继续核对。准备使独立规划世界统一到base坐标，保持物理场景不变并重新验证；不能用该虚假碰撞判模型不可达。

历史v9逐帧实际关节FK的修正证据：[NumPy独立交叉核查](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/v9_observed_numpy_fk_crosscheck_20261003.json)。较慢位置servo的LP目标仍是仿真接触候选，完整v11路径、物理转正、原生回放与HDF读回尚未完成。
世界竖直抬升的同一CuRobo候选已获得独立递归瓶/table证书：925次距离计算、971个接受子区间、最多6层细分，连续距离下界约0.543µm为正，没有实际碰撞或未解决区间；这只补足同一冻结抓持路径的桌面几何证书，不等于实际物理抬升通过。v9实际历史完整非手网格检查确认 `body_base_link/l_link4` 相交422个记录样本（首step7598），因此保留避本体路径要求。关联证据：[实际历史FK](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/v9_observed_actual_fk_audit_20261003.json)。

#### 2026-10-03 分段抓持规划与实际竖直抬升（尚未完成任务）

辅助完整模型规划世界统一到机器人base坐标后，实测附着手掌/瓶子世界位置误差约0.5µm，7个瓶子碰撞形状一致，起点无辅助碰撞。24个IK分支的有限搜索中，原arm轴后退没有合法无碰撞端点；世界Z+.1m找到2个。该结果不是全局不可达证明，原任务源码和厂商模型仍保持保护哈希。

v11实际完成世界竖直抬升：完整持瓶网格审计和逐物理时间样本接触承力LP预检均通过；约3023个抬升记录步中，手瓶相对位移最大0.4255mm、朝向最大0.22384°，未触发实际非豁免自碰撞停止。末态功能点0高度0.87258m，瓶子仍未转正，**不满足官方任务成功**。第二段转正规划把初始冻结抓持关系与当前约0.424mm/0.217°微小变化按20µm阈值比较而被拦截；这是分段参考使用过严，不能称为抓取滑脱或模型坐标修复失败。

v12正在验证：每段规划与惯性承力预检分别冻结该段开始时实际掌瓶关系，保存到planned_carry_path NPZ和report；全过程仍相对首次抓持检查2mm/2°滑移，全部实际36DOF和非豁免自碰撞接触逐物理步记录。原生重置回放、实际完整网格读回、正式HDF5及等速视频均待完成，当前仍0条合格官方episode/0新增官方任务HDF5。无实机IO，手参数和SDK映射仍未实机标定。

证据：[base坐标端点预检](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/full_native_retreat_goal_preflight_v3_20261003.json)、[竖直候选递归桌面证书](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/server_evidence/world_z_lift_table_certificate_20261003.json)。实际v11服务器目录为`deploy/realman_gen4/adjust_bottle_20261002/coupled_side_entry_v11_B085_20261003_seed7/`；完整v12仍在worker4、Conda robotwin、GPU2运行。关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B双臂机器人]]。