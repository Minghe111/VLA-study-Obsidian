---
title: RM65-B接入RoboTwin仿真
date: 2026-09-23
updated: 2026-10-11
tags: [环境, RoboTwin, 机器人, 研究收获]
status: 50条专家示教验收完成；全量训练最近核查48550步；4万步八回合样例已归档；完整SR与实机标定待完成
sources:
  - https://robotwin-platform.github.io/doc/usage/new-embodiment.html
  - https://develop.realman-robotics.com/robot/download/model/
---

# RM65-B接入RoboTwin仿真

> [!info] 2026-10-08整体进度（历史快照）
> 2026-10-08归档：全新50条近臂三RGB示教已发布与独立验收，全50已接入161官方RoboTwin π0.5全量微调；旧20模型闭环未成功，新50模型SR尚未评测，实机标定未完成。
> 早期模型／20条／RGB50采集中段落按原日期保留。最新阶段见[[#2026-10-08 数据集录制与实验阶段总览]]；不能把早期三代参考手或标量协同接口当当前四代六通道版本。


> [!info] 合并后的最新记录（2026-10-11整理，证据截至2026-10-10）
> 全新50条专家示教已完成；161官方π0.5全量训练最近实际核查为2026-10-10 20:47的48,550／100,000步。40000检查点已有累计8条闭环样例，追加6条中2条抓起并保持；完整任务SR、最终模型效果与实机标定尚未完成。此次仅合并已有笔记，没有复查服务器或重跑实验。

## 项目进度时间线（2026-10-11合并）

| 记录日期 | 已有证据与阶段 | 对应记录 |
| --- | --- | --- |
| 2026-09-23至10-02 | 厂家模型、双臂／四代手、原生接口及头部视角接入；实机标定待补 | 本页此前带日期章节、[[RM65-B双臂机器人]] |
| 2026-10-03 | 官方adjust_bottle首条验收，随后缩短时长、五指闭合及20条示教；三RGB采集接口接通 | 本页10月3日章节、[[服务器172.17.27.166-robotwin]] |
| 2026-10-04至10-07 | 原20条模型训练与闭环失败分析、执行时钟对齐；旧模型失败不能改写为专家数据失败 | [[2026-10-04-π0.5-RM65B-RGB20微调与推理验证]] |
| 2026-10-07 | 新采50条：左／右25，三种瓶型与三类摆放；独立验收、发布及161交付 | [[2026-10-07-RM65B-RGB50双臂近臂抓瓶数据集]] |
| 2026-10-08 | 官方RoboTwin π0.5完整权重恢复，全参数100000步作业启动；global64／每卡16，GPU4–7，每20000步保存 | [[2026-10-07-π0.5-RM65B-RGB50官方RoboTwin全量微调]] |
| 2026-10-09至10-10 | 161新版代码同步到Linux本机，组件验收及两私库提交；并非完整任务benchmark或实机验收 | [[2026-10-09-RealMan从161同步与组件验收]] |
| 2026-10-10 | 40000检查点首轮2条失败，追加6条中2条抓起并保持；累计8条仅为样例，不能推广为完整SR。头部24°下俯仍存在近桌面裁切 | [[2026-10-10-RealManπ0.5四万步闭环推理与视频]] |
| 2026-10-10 20:47 | 训练最近核查48,550／100,000步，2万／4万检查点已保存；属于当时快照 | [[2026-10-10-161-RealManπ0.5训练状态核查]] |

后续结果按记录日期链接，保留原采集版本、单位、时钟、来源SHA和验收边界。50条专家成功与模型闭环效果分别记录；10月10日域随机化核查见[[RoboTwin 2.0框架与默认任务#实际 RealMan 汽水瓶50条数据（2026-10-10 补充）]]，不能把瓶型／姿态覆盖称为完整背景、光照及动力学随机化。

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
### 2026-10-03 官方 Adjust Bottle 首条严格验收通过

本轮持续目标已完成：在Conda `robotwin`、GPU2，以RM65-B双臂+四代Inspire手完成RoboTwin官方 `adjust_bottle` seed7（瓶模型13、左臂操作、右臂待命）的首次规划执行、1秒保持和全新场景原生保存轨迹回放。**正式合格官方数据从0增至1，仅保存episode_0000000；此前自定义reach与失败尝试不计入此数。** 采集worker4已退出，没有启动训练、后续批量任务或实机IO。

实际运动：接近阶段没有手瓶接触，瓶功能点位置变化最大0.133mm；闭手最大1.126mm/1.919°，不写成零移动。回放手瓶相对位移最大0.467mm、旋转最大0.231°；末态直立误差约0.20°，250个保持样本持续真实三指对向法向力、离桌，满足未改的官方成功判据和额外稳定门禁。三段完整持瓶规划与每个离散时间样本的承力预检通过，独立以原质量/惯量/质心重算三段全部保存的力与力矩，差异均0，原物理场景不变。规划辅助附件未进入物理场景，无焊接/外力托举或运动中的qpos瞬移。

回放22774个物理步全部36DOF/原生93-link网格与真实接触读回通过：0非豁免实际自碰撞、0实际几何碰撞事件、0离桌后再触桌；初始瓶子桌面支承另行保留。指令路径连续几何证书与实际离散物理步检查分别记录，不写成全局最优、多种子成功率或完整实际连续扫掠证明。手实际有限差分速度最大0.34889rad/s；engine qvel最大6.463仍单列，不能用它替代真实角速度。物理联动与手摩擦/材料/接触力参数仍是未实机标定仿真profile，SDK/TCP/真实相机标定尚缺。

正式HDF5包含1540组状态—动作配对、1541原观测、91.096004秒、仅320×240头部RGB/合成深度/内外参；官方14D兼容格式保留，另存实际24D臂+主手SDK状态/命令及实际36D关节/物理时钟。14D原native arm state仍属命令通道，测量应读明确标记的附加实际字段。独立全量校验全部PKL、RGB JPEG逐项重编码、深度、CV/GL矩阵、HDF/NPZ/JSON物理字段、时钟/边界及24项来源hash通过。11个重复动作边界保留显式时间，缓存覆盖step0到22774、最大间隔15，无丢尾。HDF SHA256 `355797ed4a8dcd399d3ac409d76247dab120957d980235c99719ec01872db052`，本机副本同hash。

视频：官方原head观察序列1541帧/30FPS为51.37秒，不能当等速。新head按真实缓存时钟导出2733帧/30FPS、91.10秒，source age最大0.06秒；旁观每10个实际物理步2278帧/25FPS、91.12秒，尾差小于一帧。全帧解码及时间映射通过，另存H264只作编码兼容，无重渲染/删帧/改运动时间。头部保留厂商安装关系、机械俯角−0.419rad、yaw0、extra optical pitch0；起点/终点瓶体部分出框，抓握手掌有遮挡，不能写成全过程无遮挡或完整物体入镜。旁观九帧与视频能看清完整抓取/抬升/转正保持。

原adjust_bottle/Base_Task/native_robot及厂商两URDF/config六项保护SHA不变。原arm轴后退碰躯干的问题，以独立硬件hook委托官方已有world抬升选项解决；转正使用官方get_place_pose已有align分支，官方功能点目标与成功判据未改。validated_profile_v12记录单一瓶型/种子仿真参数，未替换所有任务默认配置。辅助AttachedBody的base/world错误已更正；QA曾将inactive接触壳层列表非空误计为自接触，已修正并单独说明，真实active计数0来自已通过的全36审计，不使用错误字段。

本机交付：[说明与边界](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0/RESULTS.md)、[唯一HDF5](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0/data/episode_0000000.hdf5)、[旁观等速视频](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0/video/external_realtime_h264.mp4)、[头部等速视频](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0/video/head_realtime_h264.mp4)、[全量数据验收](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0/evidence/dataset_validation.json)。原服务器episode目录：`deploy/realman_gen4/adjust_bottle_20261002/coupled_side_entry_v12_B085_20261003_seed7/accepted_replay_20261003T080454184429Z/`，原大JSON/缓存/物理NPZ/轨迹及视频均保留。下一阶段等待用户指令，本条成功仅指仿真任务与数据导出闭环，不是实机迁移或训练评测成功。

关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B双臂机器人]] · [[RoboTwin 2.0框架与默认任务]]。

## 2026-10-03 官方 Adjust Bottle 缩短时长与五指闭合验收

用户要求按RoboTwin2.0节奏缩短单episode、消除抓到后长停顿，并让普通抓取五指共同启动。核查日期2026-10-03；服务器172.17.27.166、Conda robotwin、GPU2，RoboTwin commit `56286567103cf58bfa6191ed889908c5c3a0baa3`，四代RM65-B+Inspire Gen4厂家模型。六项官方/厂家保护SHA仍与上条验收相同。

旧v12为91.096004秒，主要来自诊断配置的多段慢闭合、2秒额外预抬升保持和12/40/12秒搬运最小时长。新v29移除这些诊断等待，使用现有CuRobo与MPlib/TOPP时间参数化；五指同一quintic曲线，按primary/mimic厂家速度限位重定时，保留原50% gripper padding。实际阶段：接近2.040+1.244、闭手1.720、抬升1.164、转正3.548、调整.568、最终验收1.000秒，合计**11.284000536秒/2821物理步**。闭手step1251直接接抬升；三指稳定确认至抬升.564秒属于既有闭合/padding，不存在额外保持步。

五指均真实运动：小指/无名指约28.84°/27.19°；中/食约30.56°/30.77°、拇指约3.13°。当前瓶型承力仍是拇指/食指/中指，小指和无名指未承力，不声称五指全接触。闭合功能点最大移位.670mm、旋转.477°，未放宽原2mm/2°保护；全程最大实际掌内漂移.346mm/.129°，官方成功且末端瓶口倾角.274°。

新初态手部primary PD从K5/D.1调整为K20/D.2，effort10和followers0/0/0、tendon100/.02不改；24手关节实际参数前后核验一致。相同原生节奏的K5候选未完成转正，K20完成；拇指驱动误差约.077→.0196rad，冻结材料点几何proxy末分离约2µm，相比历史失败约.58mm明显改善。这是实际36FK/COM对照和几何proxy，不是持久接触ID证明、实机标定或跨任务成功率。降低原生加速度(.8/.5)和抓位+15mm对照未改善，未采用。LP26仅针对HiGHS未知数值状态增加等价求解回退；原完整约束/残差/力上限检查保留，未知状态仍拒绝。

全新场景严格回放**已发布唯一新版episode_0000000**：204配对/205观测、2821物理步。实际36DOF/93链接逐步离散网格审计与禁止自碰撞冲量0、完整离桌接触证据、HDF/NPZ/JSON/PKL/RGB/深度/相机内外参/时间/source hashes独立读回均通过。真实桌面碰撞开启，额外深入1.496µm<原20µm预算，step1282达到1mm释放阈值，释放后活跃桌面冲量0。手关节真实位置差分峰值.99677rad/s通过，engine qvel峰值11.5639rad/s另存，不能把FD验收说成引擎瞬时速度也满足厂家限值。分配器法向上限是目标约束，不是实际瞬时力硬限制。

旁观H264283帧/25fps=11.32秒；头部H264339帧/30fps=11.30秒来自原记录RGB按物理时钟零阶保持（最大观测龄.06秒），未改变播放速度。原生观测序列视频205帧/30fps仅6.833秒，不能拿它当实际任务时长。头部仍厂家机械俯角−.419rad/yaw0/vendor_zero_view/额外光学pitch0，腕相机关闭；头部部分瓶体出画，没有擅自改变实机视角。

HDF SHA256 `27bf40ad25e2dcf9d173ef63348de178604195571db6349da6817c1fc21de569`；本机HDF/两视频与服务器SHA一致。服务器全量原trace/559MB网格审计/缓存位于 `deploy/realman_gen4/adjust_bottle_20261002/coupled_side_entry_v29_B085_20261003_seed7/accepted_replay_20261003T121935159324Z/`，独立验收在同目录。旧91秒版本保留在validated_episode0；新版为同首条的修正版，没有episode1、训练或实机IO。worker5已停止。当前单种子/瓶型profile未覆盖默认硬件配置；TCP/SDK/真实相机和力参数仍未标定。

本机：[结果说明](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/RESULTS.md)、[v29配置](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/validated_profile_v29.json)、[旁观等速视频](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/video/external_realtime_h264.mp4)、[头部等速视频](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/video/head_realtime_h264.mp4)、[HDF](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/data/episode_0000000.hdf5)、[独立数据验收](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/evidence/dataset_validation_independent_v29_20261003.json)、[实际离桌读回](file:///E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_20261002/validated_episode0_fast_five_finger/evidence/actual_support_readback_independent_v29_20261003.json)。

关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B双臂机器人]] · [[RoboTwin 2.0框架与默认任务]]。

## 2026-10-03 官方 Adjust Bottle 首批20条完整数据集

用户授权按已通过的v29时长与五指标准，自动生成20条完整episode。核查服务器172.17.27.166，沿用Conda `robotwin`、RoboTwin commit `56286567103cf58bfa6191ed889908c5c3a0baa3`及RM65-B双臂/Inspire Gen4厂家整机模型；本批为官方左臂`qpose_tag=0/model_id=13`分支，官方采样、任务与成功判据未改，双臂和双手状态均保存。

正式索引0–19、20个唯一官方seed全部完成。总计**4112组state/action、4132个原始观测（含20个终点）、56858物理步、227.432011秒**；单条动作时长最小/平均/最大**11.208/11.372/11.956秒**，全部闭手1.720秒。五指共同quintic曲线、无额外预抬升等待，均有SAPIEN实际运动；小指约28.84°、无名指27.19°、中/食约30°、拇指3.861–4.079°。承力仍是拇指/食指/中指，不称五指均接触瓶子。

所有官方任务成功、完整actual36网格/禁用自碰撞接触、支撑释放、抓取稳定性、HDF/缓存/图像/深度/仿真矩阵、视频和时钟独立验收通过。闭手相对preclose的功能点最大位移1.011mm、瓶原点1.259mm、旋转.782°；搬运时瓶原点相对掌心最大漂移1.160mm/.562°，原2mm/2°守卫未放宽。离桌额外深入最大3.830µm，释放后无活跃桌面重接触。这里的掌内漂移并非持久指尖材料点滑移测量；actual36覆盖全部离散物理步，不等于连续扫掠证明。有限差分手速与engine qvel分别保存，不将前者通过写成后者瞬时速度通过。

20条的初始位姿、首目标、首段/全部左臂数值轨迹，以及官方codec解码的首帧/完整RGB像素流均互异；比较排除了pickle/视频容器元数据，确认没有复制同一episode。14D原生接口保持；帧级实际24D与next_command24、逐物理步命令/反馈及full36/hand24弧度保留在HDF补充组。合法动作边界存在重复时间戳，使用显式sample_time_s/next_sample_time_s与物理时钟，不以原生30FPS观测序列替代真实时长。头部和旁观realtime H264仅转码，实际时钟未改。

批次编排为v32、物理标准仍v29，冻结178个关键输入后执行：首组5条合格v32 replay经完整门禁继承，GPU2–7补齐15条，每条接近均重新调用原生规划器。v31仅修复FCL初始支持碰撞集合为空时的误拒，仍要求原实际支持证书和离桌守卫；v32规范frozenset代码常量签名并校验该显式规划绑定，没有跳过源码比对或物理守卫。实际模块hashseed0/5/9及反例测试通过。六项官方/厂家保护文件及全部178项冻结源最终SHA未变，所有采集worker退出。

原v32尝试27条：5接受、17不支持分支、1candidate失败、4初始化异常；补齐批次129条：15接受、74不支持分支、18candidate失败、22初始化异常。失败含超过2mm漂移的候选及官方场景不稳定；这些未计入20条，不把筛选后的20成功示范报告为100%生成成功率或已训练策略评测。

服务器正式数据：`/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/batch_runs_v32_parallel/realman_adjust_bottle_batch20_20261003/dataset`。
本机：`E:/Workspace/VLA-benchmark/deployment/realman_adjust_bottle_batch20_20261003/v32_parallel/dataset`，20个HDF合计2705427134字节。README、episode_summary.csv、completed_summary.json、diversity_summary.json及final_runtime_recheck.json在本机上一级目录。manifest SHA `866b6da0fd6559fac0a32ae8dfeafa9b7061eb0287896cd7eb4319ef807c1874`；complete SHA `d987c91dfa59ec7ecff918603ee7d40c69052b8c7e6ca1ad6d562b5ed09a4d30`。

本机最终local_integrity_report通过20条全部训练HDF、语言、轨迹、80段视频、140份轻量报告及4份聚合元数据SHA核验；20份巨型actual36_geometry_audit仍在服务器，各source replay内原始trace、NPZ与.cache终点观测保留。不是把所有服务器诊断复制到本机。历史单条v29及v30/v31/v32目录均保留，不纳入新批次正式计数。

头部继续机械俯角−.419rad、yaw0、vendor_zero_view、extra optical pitch0；腕相机未安装，训练仅cam_head320×240，旁观640×480不作为训练镜头，原有部分遮挡/出框不修改。SDK/TCP/握力/真实相机外参仍待实机标定，无实体机器人IO或训练。后续等待用户指令。

关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B双臂机器人]] · [[RoboTwin 2.0框架与默认任务]]。

## 2026-10-03 三D435纯RGB视角与录制准备

用户本轮授权调整左右腕相机视角，指定头部及左右腕均D435，三路只录RGB、没有深度，按RoboTwin格式准备采集。当前用户说明确定型号与输出需求；不据此宣称实机腕相机安装位姿、序列号或标定已核验。

三路采用librealsense D435 USB3默认color profile：640×480、RGB8、30fps；USB2默认回退15fps不纳入本次仿真profile。真实曝光/白平衡/增益保持设备固件默认的要求写入配置，仿真没有模拟这些UVC参数；每台实际RGB K及畸变应读取对应device profile。模拟K仍来自RoboTwin Large_D435名义37°垂直FOV，不称真实D435出厂标定。来源：[默认stream源码](https://github.com/realsenseai/librealsense/blob/master/src/ds/d400/d400-factory.cpp#L636-L670)、[RGB8默认转换](https://github.com/realsenseai/librealsense/blob/master/src/ds/d400/d400-color.cpp#L169-L174)。

在法兰背侧、抓握侧及后移抓握侧做实际FK渲染对照；选定左l_link6局部光心[0,-.08,-.045]m、右r_link6[0,-.08,-.041]m，两侧forward[0,-.18,.983666610188635]/left[1,0,0]。右+4mm对应厂家左右掌连接偏移。视角固定跟随腕FK，不逐帧跟踪目标、不镜像RGB。这是未测量安装预设，未新增支架/相机实体碰撞体；不是新硬件碰撞证明。头部最低机械俯角−.419rad/yaw0/vendor_zero_view/extra optical pitch0保持，光心[0,.016,.0125]m不变。新头世界矩阵与原缓存误差0、全部相机跟随最大矩阵误差4.18e−7、瓶姿态重建3.58e−7；左光心最低Z=.82155m（桌面.74m），只是光心几何检查。

完成**1条完整格式预录制**：上一批20条中的episode0/seed7，重建每个原缓存的实际36关节与瓶体姿态后重新渲染三路RGB，循环不执行物理步。205组native状态动作/206个原始观测、源时长11.316s；这不是新增随机种子、再次物理抓取成功或策略评测。用现有原生pkl2hdf5 exporter保存vision/cam_head、cam_left_wrist、cam_right_wrist，各含colors/shape/intrinsic_matrix/extrinsics_matrix（GL cam2world）。RGB之外仍保留原14D兼容、真实仿真24D及逐步full36、语言、轨迹、实际时钟；无深度/分割/点云dataset。615个训练RGB帧全部官方CPUcodec解码、三相机矩阵与全部非视觉字段dtype/shape/值字节/属性独立读回通过，原状态动作时钟（含合法边界重复）未标成30Hz。3段实际时钟视频341帧/30fps=11.367s，零阶保持原观测，最大观察龄60ms；原生206帧/30fps观察序列不作实际时长视频。

新增单任务实例attach_rgb_recording适配器部署为integrations/realman_rm65b_gen4/three_camera_rgb.py，profile在camera_profiles/three_d435_rgb_default.yml；通过显式绑定后使用官方get_obs/_take_picture/exporter，不替换默认头部配置。实际Conda组件烟测调用两次官方_take_picture、转换2个静态cache为1个组件transition通过，状态恢复后禁止scene.step、0新增动力学步；不计作任务episode或训练样本。旧v32验收/发布器强制头部深度，正式三RGB批次应使用新的RGB合同，不能借旧passed标记。

服务器产物：/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/three_camera_rgb_20261003/rgb3_format_episode0。验证为rgb_only_validation.json、recording_report.json；组件IO证据api_smoke/native_recording_api_smoke.json。本机：E:/Workspace/VLA-benchmark/deployment/realman_three_camera_rgb_20261003，README/选定three_d435_rgb_v2.yml/数据/三视频/7组拼图/证据24文件SHA和字节长镜像通过；206份新原缓存与历史对照留服务器。HDF83069399字节，SHA6e9efdb0656fb2c8fe3137598ea0c219c88265e1eb16c650baa5e2cab662a32a；local_integrity_report通过，delivery manifest SHA683f07937692edb755d93df6b908a473a642d9bc58d3b2f7be7ae3b48faaeee9。

RoboTwin HEAD仍56286567103cf58bfa6191ed889908c5c3a0baa3，原178冻结源及官方/厂家代码保持；只新增3个相机集成文件，原20条未覆盖。尚缺实机RGB K/畸变/支架外参、TCP/SDK/握力标定；无实体IO或训练。头部原有终点部分出框、非动作右腕视角目标离开画面保留。关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。

## 2026-10-03 首批20条位置与目标外观多样性复核

用户要求验证20条汽水瓶是否位置随机、是否有多种目标样式且不全为可口可乐。本轮在服务器逐条读取manifest、accepted_candidate_report.official_sample.sampled、replay_report.final_bottle，并与既有diversity_summary初始矩阵核对；仅新增审计报告，不改变数据、模型、任务或采集配置。

结论：**位置/朝向随机化通过；目标瓶型多样性不通过**。20条model_id均为13、qpose_tag均为0/左臂。初始x范围[-.119896412,-.081806079]m、y[-.126071528,-.081576005]m，实际跨度3.809×4.450cm，覆盖官方左侧4×5cm采样区域；20个位置和yaw（−112.753°至−79.795°）均互异。范围指抓取前稳定后的瓶actor原点，不是全桌操作覆盖或功能点坐标。

当前官方adjust_bottle在[13,16]两瓶型、左右qpose_tag两分支中采样，左x[-.12,-.08]、右[.08,.12]、共同y[-.13,-.08]m，rotate_lim=[0,0,.4]rad。既有batch_adjust_bottle_v32.py:51明确过滤qpose_tag非0或model_id非13，因此正式20条全部同一model13，model16和右臂分支没有覆盖。过去“20数值轨迹/图像互异”只证明非重复轨迹和像素流，不证明物体类别/纹理/品牌随机化；应称**单瓶型、左臂、位置姿态变化的数据子集**，不能称达到完整官方瓶型覆盖。

官方pick_diverse_bottles另在0..19共20个瓶型中为左右各采样一瓶，是另一任务，不把它的范围误写成adjust_bottle默认配置。新增第二瓶型需要独立验证该几何/抓持/抬升/碰撞与物理门禁后再计入正式数据，本轮没有生成新episode或修改已验收20条。

审计：服务器deploy/realman_gen4/bottle_diversity_audit_20261003/audit_report.json和episode_positions_models.csv；本机E:/Workspace/VLA-benchmark/deployment/realman_bottle_diversity_audit_20261003同名文件，SHA与服务器一致（JSON68b1033afd391a10f7cf66db7e73d95d3d6931b9b1c3e396df4ce5d675e769f2；CSVb95cf2d8678a49b22f126e63baa2c7581244e357321154e98d845804e2b0ea26）。原manifest仍866b6da0fd6559fac0a32ae8dfeafa9b7061eb0287896cd7eb4319ef807c1874，official task SHA4f76b35e718a4c5a364dae564d35d68268755818d9062e3ed4ff393d158d0548、HEAD56286567103cf58bfa6191ed889908c5c3a0baa3。实际资产外观已核查：从服务器GLB内嵌baseColor纹理确认model13为红标可口可乐、model16为绿瓶雪碧（可见品牌文字），不是由ID或语言名称推定。当前20条为可口可乐20条、雪碧0条。官方adjust_bottle.py:13-29与rand_create_actor.py:24-38给出上述随机模型、位置及角度采样；pick_diverse_bottles.py:13-37属于另一项双瓶任务。当时仅由model_data.extents乘scale记录了68.7/92.3mm界定框尺度；2026-10-04实际PhysX凸体读回纠正：该尺度不能当作真实瓶径，见后续核查。第二瓶型须独立验证实际几何和抓取，不能仅替换纹理。GLB SHA256分别为bcf7da24d20a2d2b4b79ca6b7bf9de175ea649d53461dc820d857437df116cb7和8eec8d9b896accd7287a9756361a8ce3d7f58d55549fc4f579053840131b31b9。独立纹理预览位于本机审计目录appearance/base13_basecolor_preview.png及base16_basecolor_preview.png，SHA分别6a8d2d0fa9f06f683a1051e525ead7c13fd435a18eb42b0ec4b8984b8b7911ca和6302de1fd726d5a48f0efed4a679e155a4ce237ece8dcb12ad3bd96ba9178abd；它们是模型纹理图集证据，不是任务场景渲染。原审计JSON/CSV及20条数据未改。

关联：[[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。


## 2026-10-04 最终20条双瓶型三RGB构建进展与几何修正

本轮目标为完整20条同一官方adjust_bottle任务、至少两类目标、初始位置不同、头部及左右腕D435纯RGB以及全部原生状态/动作/语言/轨迹/实际时钟。新批次计划13/16各10条，沿用已验收的10条model13物理源并为model16重新严格物理验证；范围为左分支，双臂双手状态和三视角均保存。旧20条及官方/厂家文件保留，未改任务采样或成功条件。

截至当前阶段检查点：**10条model13三RGB重渲染暂存验收通过；model16尚无严格合格条目；最终20条仍未完成，也未发布正式manifest/complete。** 暂存合计2062组native状态动作、2072个raw观测（含10个终点）；6186个训练RGB帧全解码及所有非vision字段精确比对通过，重建后新增物理步0。这是继承已验收物理轨迹的传感器版本，不能写成10次新物理抓取成功、策略训练或实机效果。三路640×480RGB8，原非均匀采样时钟、合法边界重复和终帧保留，原save_freq=15为物理步间隔，保留原动作时钟，不冒称30Hz动作。新管线只有20条且两型各10、逐条source/asset/官方sample核对及独立完整合同审计通过才写完成标记。

原178项冻结输入已重新SHA核查全部相同。新增目录为服务器deploy/realman_gen4/final_batch20_20261004（requirements.json、model13_sources.json、brand_evidence.json、source_guard_before.json、original178_current_recheck.json、progress_checkpoint_10rgb13.json）及rgb20_final_20261004（渲染/发布器、录制暂存与model13_render_summary.json）。本机E:/Workspace/VLA-benchmark/deployment/realman_final_batch20_20261004已镜像阶段证据；rgb20_final_20261004/light_mirror保存10条轻量录制/验收/预览镜像，不把它说成已交付完整20条训练数据。

**修正此前宽度推断：metadata extents乘scale不是实际瓶径。** 从SAPIEN当前实际7个PhysxConvexMesh的vertices×scale与localpose读回actor坐标：model13 extents=[.0684887543,.2478884459,.0674401820]m，model16=[.0706707090,.2553243935,.0704546869]m；实际碰撞体横向尺度约68.49/70.67mm，而非metadata的68.7/92.3mm。读回位于model16_support_20261004/physx_actorframe_model13_readback与physx_actorframe_model16_readback，含summary及actual_physx_vertices_actorframe.npz（16顶点证据SHA ab95538bc578ac39bdeb4d4ef7143f7540351f38d0ca8ac69b9b48a82f312987）。这是实际仿真碰撞体数据，不称实物瓶子量测。实际仿真mass两型均约.01kg，惯量/质心保留官方加载值，未以尺寸推断改动。

model16/seed2000原几何的实际时序：1–679物理步瓶不动，step680拇指sensor_4出现零冲量几何manifold后开始滚动，step746功能点位移.639109mm/旋转1.024256°触发原approach门禁。active_contacts为空不等于没有几何接触。固定同初态4s对照位姿变化0，但仅wake_up后无机器人活跃冲量也发生明显滚动；原初态sleeping不能证明唤醒后稳定。瓶与手shape contact_offset各.01m，近触能先唤醒瓶。没有额外等待、set_pose/set_velocity、改质量/材质/offset来伪造成功。

hand_y在pregrasp表达式完全抵消，不能靠调整它退让；hand_z才控制径向间距。新增z190使两段接近真实通过，但闭手时被动ring力.1442N超原.1N守卫而拒绝，仍未accepted。继续新独立五指target/径向几何实验，保持同步quintic、厂家速度约束、原no-push/LP/力/自碰撞/支撑释放/2mm2°抓持门禁；失败与诊断不计正式条目。相关时序证据在rgb20_contract_review_20261004/model16_seed2000_approach_timeline_audit.json，静止/唤醒对照在model16_support_20261004/passive_seed2000_v4与passive_awake_seed2000_v5。

独立最终审计脚本已在当前单条三RGB及10条官方sample身份上验证：全36关节/物理时钟与原源逐帧绑定、独立URDF FK到GL/CV相机外参、全部非视觉字段、N+1终点、RGB解码、语言及轨迹检查。它尚未在最终20条上运行，组件/10条检查通过不替代20条验收。新增source_manifest采用每条不可变{version:1,sources:[entry]}，以实际HDF/model/seed/arm/replaypath核对，不能绑定动态追加总清单。无实体机器人IO或训练，实际腕相机外参与TCP/SDK/握力标定仍待实机。

关联：[[服务器172.17.27.166-robotwin]] · [[RM65-B双臂机器人]] · [[RoboTwin 2.0框架与默认任务]]。
2026-10-04 后续瓶肩试验：四个seed2000独立候选均已真实结束且严格接受0条。neck50/z184/thumb500第二段screw planner失败；neck60/z178/thumb500及neck70/z175/thumb500闭手转动分别约2.027°/2.047°超原门禁；neck60/z178/thumb650接近/闭手转动约0.116°，但拇指没有真实接触、paired=false，故不能撤离或计入成功。没有用小转动数值替代三指真实支撑。该批证据服务器final_batch20_20261004/model16_neck_checkpoint.json及本机同名文件SHA e145542cfca5b7c85de436ac5cef8fec55e44e8c39f8fc3c899cf4b641d48829已一致。另已启动三个不同官方左侧model16 seed组，结果待真实物理终态；这些启动不构成成功episode。

控制源码核查：feedback 0.16rad/s仅约束已latched主关节目标，未接触手指仍按原共同quintic推进；不能将拇指无力错误归因反馈速度上限。纯CPU手部STL/URDF FK间隙分析正在为下一步硬件姿态/SDK选择提供建议，静态分析不算物理成功。新来源接口预检发现原178冻结集合仅含model13资产，必须为新model16运行显式冻结visual/collision/metadata/description并绑定当前实际SHA；已经运行的旧worker与原冻结文件保持原样，其接受结果若出现须补完整独立接口/资产证据后才能进入RGB管线。不可变来源条目须明确source_episode_index=0、official_sample_json及真实配置，最终审计逐字段核对，不能补一个动态总清单代替。

## 2026-10-04 首条雪碧严格物理源与三RGB全条审计

首条model16/雪碧官方左分支seed1062已完成原任务全程物理执行及全新场景严格回放，2822物理步、11.288000536秒，204组native状态动作和205个原观测。沿用v29原几何、手/瓶原contact_offset各10mm、原材料/质量/惯量及全部抓取/碰撞/支撑/力门禁；本条没有采用后续数值接触参数试验。预撤离206步配对承力通过，掌内漂移约0.194mm/0.0893°，拇指/食指/中指约2.338/1.186/1.173N，真实对向法向点积约−0.994。实际全任务成功后再经独立reset回放、物理QA、36关节网格、支撑及数据验收，不把静态几何或仅抓持通过当作整条成功。

原v6采集器在已成功回放后的source_trajectory路径打包环节出现StopIteration；原报错记录和原完整物理源保留。新独立accepted_source_interfaces_v1/source_model16_0001062接口按真实replay_report.source_trajectory路径恢复不可变来源清单及真实配置，未重写旧日志或伪造物理通过。原物理HDF SHA122209996b5eb1a926f9d47794e3c170c13bc9c2114beedf3b62b18664c89928；不可变manifest SHAf994b24e305510753c17e415885206e26897041ee6b5375edfb8ae8da56af499。新v8仅修复该打包路径，不放宽物理门禁。

三RGB传感器版本位于服务器rgb20_final_20261004/rendered/model16_0001062，204组训练对、205完整raw（终点保留）、612个训练RGB帧。三路实际时间视频均实际解码339帧/30fps=11.3秒；按原物理clock零阶保持选帧，动作仍保留原时钟。独立CPU完整合同审计passed=true、episode_count=1、formal_dataset_complete=false：32非视觉dataset/5group共6,155,108值字节与原源的dtype、attrs、bytes精确一致；全36关节及物理时钟误差0；独立GL FK最大7.93e−7、CV外参1.04e−6；头部相对原源误差0；当前363冻结输入开始与结束SHA一致。审计报告rgb20_contract_review_20261004/model16_0001062_single_current_20261004/single_episode_independent_audit.json，SHAe159bad1035058855bf91c649a3ab16ae8799f3f91c9b685d5cf2c49b9b73a6d；RGB HDF SHA65623fd238355462fd3eac0e379e955d3c0b04b8c2da40e8921c985532d0dc50。本机轻量镜像E:/Workspace/VLA-benchmark/deployment/realman_rgb20_final_20261004/model16_0001062_light_mirror及审计目录；尚非完整20条训练数据交付。

当前合计11条录制暂存（可口可乐10、雪碧1）、2266 native配对/2277原观测/6798训练RGB；最终20条尚缺9条雪碧且未发布。40个raw yaw邻近seed的真实官方setup和1秒wake静止预筛全部结束，除对照1062外没有新增耐唤醒来源，不能把RNG筛选候选计为成功。已核查官方RNG抽样前后全部MT19937状态与两实际setup一致，未额外抽样、指定目标pose或改瓶模型。后续将在新独立硬件数值接触profile中核查手部contact_offset1mm（仅手，瓶/桌/臂保留10mm），并在全部原门禁下比较；此时该试验尚无新合格episode，不写成参数已经验证有效。

真实渲染确认绿瓶雪碧及五指灵巧手。头部保留实机机械最低俯角和厂家安装关系，起始瓶体靠画面下沿、终点存在裁切；左腕近距离瓶体占比大，右腕终点接近下沿。RGB/时钟/FK审计通过不意味着每路全程完整容纳目标；未为隐藏视角局限擅改头限位、腕安装或K。三路纯RGB，无实体IO或训练，实机K/畸变/腕外参、TCP/SDK/握力仍待标定。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。

2026-10-04 手部数值近触profile首轮真实对照：新增独立helper把左右60个Inspire碰撞shape contact_offset设为1mm，其余47个shape含瓶/桌/臂仍10mm，全部rest0；实际shape的材料/groups、24关节drive、原碰撞/视觉网格与四瓶资产未改。该参数是未实机标定的仿真数值配置，不能称原10mm参数不变或实体抓力标定。静态读回contract SHA74b3ac503426639862014bb0ee4093a7d8f417a18faaf53e6d668249852a5eb3通过；它仅证明参数读回，不是任务成功。数值壳层生成接触的含义见官方PhysX [PxShape文档](https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/_build/physx/latest/class_px_shape.html)。

四个完整真实候选均严格接受0条、reset0：seed2000 z177/thumb500 close阶段1.442mm/2.016°拒绝；z177/thumb650已产生三指真实配对51步，但close控制248/250的static_mg承力LP两次不可行（最后实际thumb/index/middle约.230/.137/.144N），没有撤离或抬升；原z165几何仅改skin仍approach .639mm/1.024°拒绝；seed1024 z177/thumb650 close1.046mm/2.051°拒绝。异常文字写approach但前两/末条的实际phase是close，以trace为准。独立审计754项当前输入首尾SHA一致、每run364冻结及原178/四资产正确绑定；此前不存在strict四phase和正式source_config，不能声称这些已通过。总审计SHA c01c5cfe0e61d241e61e4aeffdbe768451fea65aa4144be8b35e9428fa665370，服务器rgb20_contract_review_20261004/hand_skin1mm_v9_four_terminal_20261004/independent_four_terminal_contract_audit.json；本机进度证据realman_final_batch20_20261004/progress_checkpoint_11rgb_and_skin_trials_v2.json SHA182a3cb5741529a6131b37791e473f74b561ce6446d26b68588ba96214dafb6f。

实际三指接触和PhysX COM复算表明，不能把LP失败仅归因原最小1.15N要求；固定normal仅沿瓶轴移动也没有找到可行解。这些纯CPU诊断未修改运行门槛。新独立B075/B070及B075 pitch5°/15°对照正在验证降低主指接触高度与接触同步性；静态fresh mesh显示几何可达，不能预先称可承力或成功。正式录制暂存仍为10可口可乐+1雪碧，最终20条尚未发布。

## 2026-10-04 雪碧抓持的整段轨迹与严格拒绝边界

录制暂存仍为 **11/20（可口可乐10、雪碧1）**，没有新增正式accepted源或完成标记。当前初始位置图逐条绑定11个原始raw0，1µm量化下位置互异；官方sample JSON在这批数据记录的是setup稳定后的快照，不能冒称保存了全部settle前出生位姿。真实三RGB对照来自可口可乐seed7与雪碧seed1062各自终点hold帧；左腕可辨两型，头部和右腕的原有裁切保留，没有替换瓶纹理或拼成完整视野。图及诊断在本机deployment/realman_rgb20_final_20261004/evidence/previews/staging11_recorded_sources_20261004_v3，属于11条暂存证据，非最终20条。

新增B075、z177、pitch10°的硬件抓点对照通过闭合及首段垂直抬升，但后续带旋转的carry在掌内actor原点漂移2.00138mm时被原2mm门禁拒绝。独立36关节实际FK复算表明漂移主要径向（约1.976mm），轴向约0.165mm；功能点/COM漂移约0.744/0.882mm。不能用COM数值替换原actor原点指标称成功，也不能把失效简单归为沿瓶轴滑落。独立证据body_y075_carry_actual_FK_slip_components_v1.json SHA46e12c45be56b92be78e4ad8fe42eadb7ef71dba9d736d78c66fe131319b0da4。

真实掌坐标绕掌Z+5°折回原side-entry EE为局部Rx+5°（掌位姿关系E=P·H，H平移0.06m），不能直接将EE绕Z旋转。新增独立roll5/thumb610配置在seed2000完成candidate全任务，fresh strict回放亦完成2865步/11.46秒，掌内漂移最大1.257mm/0.454°；但最终全36关节网格检查在step1049检测到bottle与l_inspire_left_middle_2的non-touch重叠，故严格接受0、HDF exporter未调用。中指真实首次承力在1045（约0.483N），1046–1049短暂失触，1050恢复；这是锁存后接触丢失，不能误记为首次接触的一步延迟。时序证据palm_roll5_middle_first_force_timing_diagnosis_v1.json SHA663647fd61a0ceac8c7ef3678dc677070e897f3acf8e97f44afb3fa134738b02。后续仅微调middle SDK610→608/612并对照不同官方采样位置，全部五指共同quintic、速度和碰撞/承力/成功门禁保留；运行中不计episode。

增加明确schema的几何/控制身份验证：几何schema1只允许显式x=0/.030/.040m，schema2额外绑定掌roll5°和SDK；控制schema1显式绑定primary PD20/.2或40/.4，不能由几何schema暗推控制值。PD40/.4试验只改变手部primary刚性，effort10、follower0、tendon创建值100/.02及原物理门禁保留；实际24关节读回通过是组件证据，整条因支撑离开时非承力index_force_sensor_2与瓶重叠拒绝。独立身份/原生崩溃审计SHA2d2182e3317ec6630b5976d836dc9a91fc46944e874aa4a6f999d6309d62fbc7。没有把无真实法向力的相邻传感器加入豁免名单。

x30/x40首次worker在网格审计期间发生native Segmentation fault，原日志与stale running报告保留；真实子进程exitcode没有被采集，不能当成已观察到139。新线程环境OMP/OPENBLAS/MKL/NUMEXPR各1的独立重试均正常结束但物理严格接受0，分别出现index2或index_force_sensor_2非承力碰撞；这两次未复现native crash不证明根因修复。新physical_terminal_retry_summary.json SHAbe3e7e510c801dd0d07ce626478c9c93cd5bfc0ba08af7852b694ab4b889df01，365冻结输入当前一致。新profile/环境的actual readback和SHA将随真正accepted源进入三RGB管线，不混入首条1062的原10mm数值配置。官方/厂家网格与旧数据保持，无实机IO或训练。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。
## 2026-10-04 第二条雪碧严格成功与12条录制暂存

middle SDK608与612的两项同seed2000对照均完成candidate全任务及fresh strict回放，原actual36全状态网格、exact真实法向承力touch、原支撑释放、速度/力/LP/no-push/2mm2°门禁通过，均导出207组native配对。它们官方setup初始pose完全相同，故只选择 **middle608** 的source_id=model16_0002000计入；612保留为物理对照而非第二个不同位置episode。608为2866步/11.464000545秒、carry actor原点漂移最大0.473228mm/0.515616°（与actual36全审计的统计范围区分）；612为2864步/11.456000544秒、carry0.697170mm/0.480635°。该比较支持608的平移裕度更大，不证明所有位置均可成功。

第二条源仍为未实机标定的手contact_offset1mm、掌roll5°、B075/z177/pitch10°、SDK[900,900,608,610,610,0]与PD20/.2，原官方/厂家mesh、瓶mass/inertia/material不改。实际366冻结源和配置/entry/官方sample身份、两份四phase读回当前SHA核对通过。物理HDF SHA187c034bd9927074c5ac035c2acbf4b33033d540271c1070615b2e6742101bc0；immutablemanifest位于服务器model16_support_20261004/hand_skin1mm_body_y075_z177_palmroll5_thumb610_middle608_seed2000_native_v11/seed_0002000/candidate/accepted_replay。首触锁存的反证：SDK723.572694→722.787484对应URDF主关节q_target .397530→.398659rad，为更闭方向，而非张开回退；因此没有按错误假设改latch/feedback/filter。

当前已选12个物理来源（10可口可乐、2雪碧），root独立当前HDF SHA、官方sampler身份和stable snapshot位置核对得到12个互异位置（1µm量化）。新来源检查点selected_sources_12_physical_v1.json SHA f26225ba811af2520a159b6a96f4ccc8e7d1d9dde974d73e5f7d8961292d81d0；独立身份证据selected_sources_12_physical_identity_v1.json SHA5e67e9128a2dcea6c81e9acfa41b442d05cd9d4476c44555f25770d43c08d0c1，本机deployment/realman_final_batch20_20261004两文件与服务器SHA一致。它只证明所选物理源身份/位置，不替代最终20条三RGB审计。

新source三RGB重渲染暂存已完成：207训练对、208raw含终点、621训练RGB，HDF SHA7ed87c559c885762e83704958baa69ac771bd86687bd4bf19e1c9975f43a28dc，receipt SHAbdd591e9bdba3c4a9295c1d2b853429d00e2426b13786af6be1ccec45d12d637，恢复状态后无新增scene.step，366冻结源及nested contact/geometry读回前后守护通过。当前合计 **12条三RGB暂存，2473 native配对/2485原观测/7419训练RGB**；新条目的全条独立CPU合同审计尚待完成，最终20条仍缺8条雪碧，未生成正式20条完成标记。

已按同608不可变profile、v14/thread1在5个GPU采集11个不同官方seed候选，以取得余下8条合格源；正常setup已有sample证据的8/9/13/1024/1033/1041/1051/1087和仅诊断预筛的1069/1018/1020分别标明依据。所有候选仍真实执行未改sampler/pose/model的正常setup，既有稳定预筛不是本次完整成功保证；只有新strict accepted才计入。原roll610 seed13在thread1仍native中断，exitcode未采集、旧stale报告保留，不能称线程设置解决崩溃。没有实机IO或训练，最终20条/10+10和实际RGB外参与TCP/SDK/握力标定仍未完成。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。
## 2026-10-04 最终20条双瓶型三RGB录制汇总

正式选择已固定为 **10条可口可乐/model13与10条雪碧/model16**。20条来源均完成真实全任务与fresh strict物理验收，三相机重渲染暂存全部完成；此处记录录制汇总，尚不表示最终20条发布审计、完成标记或本机交付通过。官方 `adjust_bottle` 的左臂 qpose_tag=0 分支、官方采样与成功条件保留，双臂和双手数据保存；当前commit仍为 `56286567103cf58bfa6191ed889908c5c3a0baa3`，Conda `robotwin`。

root逐条只读当前recording_report.json并核对exact source_entry/finished/new_physics_episode后，汇总得到 **4135组原生状态动作配对、4155个原始观测（含20个终点）、12405张训练RGB**。实际物理动作总时长229.008010877秒，单条范围11.220000533–11.956000568秒。原始非均匀采样/action时钟保留；30fps等速视频按原始物理时钟零阶保持，不能称动作以30Hz采样。RGB重渲染恢复既有状态，未增加scene.step或新的物理成功条数。

雪碧固定种子为1062、2000、8、9、1024、1069、1041、1088、1059、1029；middle612 seed2000与chosen608同初始位置，排除重复；额外1047在official setup因UnStableError拒绝且自然结束，没有accepted/HDF。所有采集worker和GPU4渲染已结束。固定source清单为服务器 `model16_support_20261004/source_list/model16_selected10_sources_20261004_v1.json`，SHA `4c8f3a2c6a1f39f489f4696cad5a01b54a45b159047172b8f5b61e61e3015846`；10条Coke固定清单SHA `35c9912421123fb376a0519461218b5979eb4d47d66256cfade31187e1f4fb2c`。

新的安全JSON/current身份报告SHA `030458cf60fe9d22d76ba51c5d15d6da887fefa1ead5d770dcd1808d44e1c262`，在上述source_list目录 `model16_selected10_identity_current_20261004_v1.json`。原178输入、两瓶型各visual/collision/metadata/description共8项资产、所选363/366源冻结及源HDF/轨迹/官方sample/actual36/3QA当前SHA核对通过。该报告的初态来自entry、official_sample、strict before、physics_trace首步四方安全JSON核对；raw0只绑定文件SHA，不能把这一报告描述为已直接反序列化检查raw0。10条Sprite真实settled初始XY最近间距3.062871mm；全20发布数据的XY唯一性由后续新whole20审计直接验证，Z不用于位置多样性。1条baseline1062与9条middle608；只有8条v14来源明确记录thread1环境，1062/2000 legacy没有该字段，不补写线程历史。

前述2000新录制独立审计已实际通过：报告 `rgb20_contract_review_20261004/model16_0002000_single_current_20261004/single_episode_independent_audit_v3.json` SHA `df9cdb439032fb3f43bd1a88cc0ddf669bc7cc029975ce0f1983d54cc6a8bf25`；207pairs/208raw、32非视觉dataset字节/类型/属性、full36时钟/差分、相机独立URDF-FK与三视频全344帧PTS/ZOH均通过。后续source8完整独立审计也通过，206pairs/207raw，报告SHA `df7b4c0b593deed2152f68db35ddf87f8bbc5160cf886025cf8e5908175ac6e7`。它们仍不替代最终全部20条审计。

三路为头部/左腕/右腕D435仿真RGB，640×480，沿用RoboTwin的D435相机profile；不保存深度/IR/分割/点云载荷，保存相机内外参和安装配置。头部保持实机最低机械俯角；腕部固定安装并随厂家关节FK运动，不逐帧追踪目标或镜像图像。RoboTwin仿真内参不是每台D435实测标定，TCP/SDK/接触数值及腕外参仍未实机标定，头部和右腕终点原有出画裁切保留。没有实机IO、训练或跨任务成功率结论。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。
## 2026-10-04 全20独立验收、正式发布与本机交付完成

核查日期2026-10-04；本节记录最终完成状态，前述11/12/20条暂存段落保留为历史阶段。最终为 **10条可口可乐/model13 + 10条雪碧/model16**，均执行官方 `adjust_bottle` 左臂 `qpose_tag=0` 分支的完整仿真物理轨迹并通过原严格验收；双臂/双手状态均记录，不表述为20条双臂协同抓取。官方任务、厂家模型及原有数据保留，commit `56286567103cf58bfa6191ed889908c5c3a0baa3`，环境仍为服务器 Conda `robotwin`；没有实体机器人IO或训练。

**全20新独立审计已实际通过。** 从源和发布cache的原始第0帧直接核对模型、物理步0/时间0、actor世界位姿及当前SHA，在1µm量化下得到20个互异的桌面XY，Z不参与唯一性判定。最小实际XY间距0.796796702mm；位置不同但分布集中，不代表均匀覆盖全部工作台。最终固定safe来源接口SHA `e24125f0ab1308f7e60fc7a0155c68898241ed50d8e8913879c9ba3696d236a4`，同初态middle612和额外1047排除。全部原始cache使用绑定当前SHA的受限数值读取器验证，不使用未知pickle全局的无约束加载。

全量验收统计：4135组native状态动作、4155个含终点原观测、12405张训练RGB；物理总时长229.008010877秒，单条11.220000533–11.956000568秒。32个非视觉dataset的字节/dtype/attrs保持原源一致，全36关节/物理时钟与差分速度误差0，头部相对原源误差0；独立URDF-FK误差≤1.171e−6，CV相机外参误差≤1.317e−6。九条新profile的几何/控制、107个实际碰撞shape及四phase读回通过，源与输出当前SHA首尾一致。60个实际时间视频全部20637帧通过PTS、CFR30和原观测零阶保持像素绑定；保留原生非均匀action clock、14D兼容接口、24D实测/命令及full36轨迹，不能称动作30Hz。三路仅640×480 uint8 RGB，未录depth/IR/分割/点云。

正式服务器目录：`/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb20_final_20261004/dataset`。全20独立审计 `independent_all20_audit.json` SHA `ad27bf51e634ee49794010ef773282b56433e9a0c328e10e715c303914231c3b`；candidate SHA `8c71977e063679c80e8647a04caf7e56831e36fa50541f4f7b766b914531b359`；正式 `manifest.json` SHA `97b33109a740ae2d9b99155dbf228e5d2a19761b323e60fc622c8998aec0e577`；`complete.json` SHA `b403485fd02d4c95a2d083be21ef3252b5a9812b8b07904a5c1efc6f9e241df8`。新增 `complete_rgb20_safe_v6.py` 仅修正v5将reader SHA字符串加引号后比较的错误，原v5/候选/审计不变；其余发布门槛保持。独立最小差异审查SHA `989f5f5069826b09d42f32d444b980c820ace2523ab58c3cdb9f2eff3f2ba985`，首次序列化adapter身份读回SHA `ac24bcd330fc8f79fadd838da5e53165fc31ed5ff5585a16f1cfc81faa3e6a78`，不在完成后改写候选或报告。

**本机完整训练载荷镜像已通过逐文件校验**，目录 `E:/Workspace/VLA-benchmark/deployment/realman_rgb20_final_20261004/dataset`：816文件/2,577,332,727 bytes，包含20 HDF、20语言、20原轨迹、120视频（每条三native与三实际时间）、560预览、71证明及5元数据。inventory SHA `51f837a6d9dd82ea0d3158834ca9793eda290ae50dbd21814455b2a4e81b2550`，ZIP SHA `bc6ae1b86b0b4d4affdfec9c1620d002a98110ba6744a910267c02235453fdfa`；所有成员大小/SHA及服务器当前complete核对后才写本机正式marker。镜像报告为该目录 `_mirror_evidence/runs/20261004T0157296841185Z_d6b430eab3c4442ab6225ab1457faee0/local_integrity_report.json`，SHA `172acf6602ae805b11301a3cfa73c434f32afacae2bb65adb72f1701ec241496`，passed/local_all_selected_files_verified=true。**4155个raw cache/11,512,129,206 bytes完整保留服务器，未下载本机**；完整服务器数据4971文件/14,089,461,933 bytes，明确 `local_raw_cache_mirrored=false`。交付说明 `E:/Workspace/VLA-benchmark/deployment/realman_rgb20_final_20261004/README_complete_safe_v6.md`，原详细说明 `README_final20_delivery_v5_safe.md` 同目录。

最终位置CSV、PNG/SVG/PDF、两瓶型三视角终点原图对照和自动/逐面板视觉QA位于本机 `E:/Workspace/VLA-benchmark/deployment/realman_model16_support_20261004/final20_completed_20261004_v2`。`initial_positions.csv` SHA `b8d5811b283b75b24595eca31164a10b6fec817a1e8b4a87ee42049b83362e6e`；figure report SHA `314078c206bed55b02ba07008b9d433890183f12db4b7052230079baad900f83`，manual visual QA SHA `d704e7c435e77fe8abf4a75bab10b4150fdc59221d50839d44cbdb7bfaf6252d`。20点无jitter；示例分别取最终manifest每瓶型首条的终点，6路原RGB像素、安装与曝光未修改，图是代表性展示而非全部20条图像证明。PNG与24项输出及6张原图的服务器/本机SHA一致，PDF碰撞及对齐QA通过；仅在已有Conda robotwin中补装PyMuPDF1.28.2用于PDF QA，未新建环境或改变其他包/模型/物理代码。

适用边界：这是当前任务、左臂分支及两瓶型的仿真数据交付，不是实机执行、模型训练或跨任务泛化评测。头部保持实机最低机械俯角和厂家安装关系，双腕固定随FK移动；终点头视角瓶体部分出左界，右腕部分遮挡/出下界，不能称三路全程完整容纳目标。RoboTwin D435仿真profile不是各实体相机标定；TCP、SDK映射、握力/接触数值及腕外参仍待实机标定。后续应在确认实际腕安装后再改硬件安装配置并重新审计，不为隐藏视角问题更改头限位或目标位置。

关联：[[RM65-B双臂机器人]] · [[服务器172.17.27.166-robotwin]] · [[RoboTwin 2.0框架与默认任务]]。

## 2026-10-04 π0.5基座1000步训练与离线推理完成

正式使用当前10Coke+10Sprite三RGB20条数据做官方released flow π0.5微调；OpenPI独立clean commit215abfb217dbac7d5f1273282331b9b1866c0479，完整pi05_base已恢复。16条训练/4条whole-episode验证，train-only norms、SDK/1000及arms delta→absolute、H16/32pad保留24维物理顺序。Conda均名robotwin：原Python3.10仿真环境保留，Python3.11训练使用隔离full prefix；详细环境见[[服务器172.17.27.166-robotwin]]。

实际完成optimizer1000/final checkpoint999，用时1300.4769秒、正常退出0、采样参数非零更新、输入/官方代码SHA首尾不变。末日志step990的10步平均loss0.006376，不能等同finalstep单点或任务成功。final999全33文件/9,549,425,342 bytes发布SHA清单和推理加载一致，publication SHA1f29f604dc5243596d2909a798f6e20108d8ef39083580fd6ac2a4cae63258f7。

798个留出观测均真实输入三RGB/state/prompt，738完整H16和60尾部first-only全覆盖；raw first左臂MAE0.011325rad、左SDK3.9167，比hold当前状态基线改善；完整H16左臂MAE0.033876rad。所有query存在至少微小SDK越界，241/798会被原1SDK预算拒绝，arms越界0。离线报告SHA4857ca5af7dbfd8125f61ee8c77a76579310cd74fc1b9555c1e68172cccc4e81，已逐SHA镜像本机。验证index10/Sprite1062为旧10mm baseline，需按profile单列，不称纯位置泛化。

当前四条模型闭环未启动，不能称完整推理/抓取成功或0/4物理失败。两TRAIN新执行器回归仍被原动态LP拒绝；原strict默认必须通过。显式sim-only diagnostic入口保留全部实际停止门禁、strict success永远false，但自动审批两次拒绝启动，已等待用户明确批准诊断与SDK范围投影对照或选择先修严格回归。没有绕行、实机连接、修改原20条、厂家模型或官方任务。完整本次版本、误差、model/receipt身份及原失败边界见[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证]]。
## 2026-10-04 全20条四卡20000步π0.5新协议

用户明确取消上一轮16/4留出划分：现有20条全部训练，global batch8、FSDP4、GPU2/3/4/7、actual20000更新，从官方pi05_base重新初始化。新独立目录pi05_rgb20_all20_20k_b8_4gpu_20261004，旧实验与source20/官方代码不改。完整CPU转换/预训练恢复与四卡5步实际短跑通过；正式supervisor918271/worker918272已运行，最近核查日志optimizer451、50步均值loss0.01813069，不能当抓取成功率或完整训练完成。新场景直接推理将代替旧episode留出评估，并按真实抓住瓶子、瓶底离桌>=1cm且持稳1秒判定；旧LP只诊断，实际碰撞/防推倒/关节/mimic检查保留。新场景/自动后续链仍待接线和初始化核查，不称已模型闭环。来源、当前head-down固定URDF、归一化、四卡证据和实际状态见[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证]]；原1000步及此前拒绝记录保留为历史，用户新指令替代旧自设strict准入范围。
### 2026-10-04 新场景预检与训练后直接推理链

实际LEFT新seed30015、距训练XY至少12.66mm、三RGB/FK和实际36关节/碰撞/桌面门禁通过；官方setup4376步后只额外执行1个驱动目标不变的被动物理步，模型/策略命令0，不是抓取成功。V4新增修正正常30秒模型超时误计录像异常，5项CPU分类通过，物理/动作/相机参数均不变。最终V6协议与247项source SHA冻结。

服务器172.17.27.166的全20/batch8/四卡2/3/4/7正式20000步训练继续运行，固定UTC13:50读回optimizer3251，尚未完成。后续链PID935071已真实启动且phase=01_wait_training，等待actual20000/final19999发布后自动运行10新场景5Coke/5Sprite，用真实抓持瓶底离桌>=1cm持续1秒判定，保存三RGB视频；目前没有新模型抓取成功率。原source20、官方代码、1000步与所有旧版本保留。具体receipt/manifest/本机镜像与可复查入口见[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证#2026-10-04 新场景基础设施通过，训练后直接推理链已启动]] · [[服务器172.17.27.166-robotwin]]。
### 2026-10-05 π0.5四卡20000步已完成，新场景抓持0/10

服务器实际终态确认全20train、batch8/FSDP4、GPU2/3/4/7的20000更新完成，checkpoint19999发布校验通过；北京时间2026-10-05 05:14结束、约8小时52分钟。自动链07:33结束，10个新环境新位置模型场景真实查询162次，可口可乐0/5、雪碧0/5，基础设施失败0。6条触发瓶子推移/倾斜检查、2条无名指/小指接触力检查、2条手部FD/mimic/关节限位组合检查，均实际物理后提前停止，未达到瓶底离桌1cm持续抓持1秒的标准。全部30段三RGB视频导出成功，但模型任务成功仍0/10。

正常训练完成、低loss和workflow_completed=true不能写成成功抓取。本次只核查状态与归档，未改变原模型、门槛、源20数据或动作执行器。完整receipt/SHA及停止分类见[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证#2026-10-05 四卡20000步训练完成，新场景抓取验证0/10]]，本任务GPU与模型服务已正常释放，部署状态见[[服务器172.17.27.166-robotwin]]。
### 2026-10-05 五指SR评测口径修正

当前模型及源20条均是五指主要屈曲实际运动，但示教采用三指主要承载策略；旧V6还无条件禁止无名指/小指目标瓶力超过0.1N，不是官方RoboTwin统一SR标准。新实际首例seed40033因小指0.361405N触发该限制，停止后保留全部结果，不能报成完整新0/10。

按用户普通五指抓持要求新增V7，只取消这条过时承载限制，其余实际关节/碰撞/防推移/携持检查和1cm持续1秒真实空中抓持标准保留。26项CPU检查通过不能代替模型效果。最终20000步检查点未重训未修改，新10场景5Coke/5Sprite、seed50000、GPU6模型/GPU5仿真已实际启动，尚无新SR；旧V6的0/10独立保留，不能与V7合并。版本、SHA、启动与停止实证见[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证#2026-10-05 五指SR口径修正与新场景重跑]]。

## 2026-10-06 V9真实接触与加速采样新SR评测（进行中）

V7保留6例0成功/112查询的未完成记录，不与V9合并。V9只修正原30手部链接的真实接触存在性，并提前拒绝不可变的错误瓶型/左右分支；物理与成功门槛保留。33项CPU和4376步三RGB/CPU及CUDA RNG相同只证明组件、setup一致，不是SR。原20000/19999模型已启动新10例评测，尚无最终SR。

来源与版本：[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证#2026-10-06 V9真实接触与加速采样新SR评测（进行中）]] · [[服务器172.17.27.166-robotwin]]。

### 2026-10-06 V9十个新场景SR终态：0/10

原20000步π0.5在V9执行配置下完成10个新LEFT场景：可口可乐0/5、雪碧0/5，SR0%，基础设施失败0、159次模型查询。1条手指速度、4条关节下限、1条瓶子推移、4条瓶子姿态检查触发早停，均无实际对向抓持或1秒悬空保持。30段三RGB视频完整只是录制通过；模型与执行配置尚未通过该批抓瓶验证，停止分项不是已证明根因。旧协议不合并。

来源与实际SHA：[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证#2026-10-06 V9十个新场景SR终态：0/10]] · [[服务器172.17.27.166-robotwin]]。


### 2026-10-07 双臂近臂RGB50采集进行中

新任务覆盖左右近臂、可乐/雪碧/芬达与直立/横躺/斜向横躺。原20实际瓶身中心更靠近右肩却采用左臂，因此保留旧版本，另采全新50合规条目。首条左手直立可乐已实际抓起、空中持稳及三RGB/24维时间对齐独立核验通过；正式50仍未齐备。具体来源与未决边界见[[2026-10-07-RM65B-RGB50双臂近臂抓瓶数据集]]，不能写成新模型推理成功。

## 2026-10-08 数据集录制与实验阶段总览

> [!info] 本次记录的是整套实验阶段
> 仿真抓瓶链路与三RGB录制已可用；全新50条近臂专家示教已录制、独立验收和发布；全50已接入161官方RoboTwin π0.5全量微调。新50模型的闭环抓瓶效果尚未评测，实机迁移仍待标定。

记录日期2026-10-08。以下整理既有实际结果：录制与发布终态来自10月7日固定凭据，训练阶段以10月8日12:18回读为准。本次没有重新采集、训练或推理，也没有连接实体机器人；历史阶段记录保留。

| 阶段 | 状态 | 已完成内容与证据边界 |
| --- | --- | --- |
| RM65-B仿真接入 | 当前抓瓶链路已验证 | 双六轴臂、双四代灵巧手六通道、规划／关节接口及三RGB采集已支撑实际示教。沿用厂家模型和官方框架结构；不表示任意手势动态全身避碰或实机标定已完成 |
| 头部与腕部视角 | 仿真录制已完成 | 头部按厂家机械限位／安装关系保留，头＋左右腕三路640×480纯RGB。实机腕相机安装、支架、内外参仍未核验，仿名D435参数不等于实机默认profile标定 |
| 原20条数据 | 已录制、发布并保留 | 10可乐＋10雪碧，主要为左臂示教；按后来新增的瓶身碰撞中心近臂规则0/20合规，故保留历史版，重新采全新50；不能据此否定原物理成功 |
| 旧20条π0.5实验 | 已完成训练与失败验证 | 全20、四卡20000步旧模型完成，属于含LoRA／视觉及投影更新的非全量基线。V9新10场景SR0/10；不能由低loss认定会抓瓶 |
| 执行时间与动作语义对齐 | 已完成限定验证 | 专家原时钟与固定60ms回放代表例均成功；当前／下一动作、24维目标与真实执行时钟已核对。旧模型在新时钟的单条400动作持续推理仍失败；这不是新50模型的SR |
| 新50条专家采集 | 已完成并独立发布 | adjust_bottle相关抓瓶示教，左25／右25，按实际瓶身中心到肩部距离选近臂；可乐20／雪碧20／芬达10；直立20／横躺15／斜向横躺15 |
| 新50条质量与同步验收 | 已通过 | 选入发布集的50条源轨迹完整审计通过，数值边界例外0；每条末尾合格持稳至少1秒，瓶底离桌最小约93.925mm。复制后结构、SHA、RGB解码与时间／状态动作链核对通过；不代表全部候选采集成功率100% |
| 161训练准备 | 已完成 | 50HDF＋50instruction＋manifest／scene_info共102训练文件逐SHA同步；新50独立index／norm及官方输入变换、完整权重恢复、batch64四卡五步实际验收通过。完整物理证明继续保留166 |
| 新50条官方π0.5全量实验 | 正式运行中 | 全50训练，无验证划分；官方RoboTwin cotrain29999参数，51叶／3353433872全参数可训练，冻结与LoRA均0。global64／每卡16／GPU4–7，100000步、每20000步保存；本轮不是旧模型resume，尚未完成 |
| 新50模型推理与SR | 尚未开展 | 尚无新50全量模型的独立新环境成功率或推理视频；专家50条成功不能替代模型能力评估 |
| 实机部署 | 尚未完成 | TCP、手SDK编码／角度与力度、相机内外参、固件／触觉接口及控制时延需要现场核验，未连接或驱动实体机器人 |

### 数据与能力边界

- 正式50为独立新采集，未重标旧20，也未用模型推理代替专家示教。全50验收是发布集质量，不是所有生成尝试的产出率。
- 三路RGB的观测／动作共11098对，物理周期4ms、每15物理步采样一次约60ms；HDF共33294张RGB均已解码。保留原生14维兼容入口及24维双臂／灵巧手位置接口，仿真测量另存；驱动目标不能当实际关节测量。
- 每手保存6位置通道，但新50两手拇指旋转命令列11／23恒0，不表示六自由度都获得变化监督。力／触觉没有作为当前π0.5模型输入；物理接触／摩擦审计与实机传感器验证分开记录。
- 位置是两侧局部成功邻域，包含三瓶型／三姿态，不能称全桌面均匀覆盖。头部／活动腕视角仍存在目标边缘裁切和手臂遮挡，不能称全过程完整可见。

### 下一阶段

- [ ] 取得新50全量训练的完整检查点，核查训练终态；当前不把运行中写成100000步已完成。
- [ ] 使用新环境／新seed／未用于示教的初始位置做闭环推理，覆盖左右近臂与三瓶型／三姿态；按官方π0.5配置持续推理，不因瓶子推移提前结束或加入抓取修正规则。
- [ ] 依据真实物理瓶底离桌至少1cm且持续1秒的用户抓持标准记录结果，另保留官方任务判定、失败程度及三RGB视频；模型SR与专家验收分别统计。
- [ ] 如仍失败，区分动作预测、抓位／闭手、接触物理与执行时间问题；同数据LoRA／全量等可控消融尚未完成，不能把多项同时改动后的差异归因于单一因素。
- [ ] 真机迁移前补齐相机、TCP、手映射、力度和固件／延迟标定，单独完成实机验收。

### 证据入口

[[2026-10-07-RM65B-RGB50双臂近臂抓瓶数据集#最终发布与复制后验收（2026-10-07）]]保存完整录制／发布／统计证明；[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证]]保存旧模型训练、V9失败SR和60ms单条验证；[[2026-10-07-π0.5-RM65B-RGB50官方RoboTwin全量微调]]保存新模型准备、官方权重及实际全量运行证据；[[RM65-B双臂机器人]]保存实机资料与待标定边界。

本轮重新核对本机固定录制凭据SHA：manifest `2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80`；源50审计 `5ff5401ca4f503bdf9da881ef6f9e5ae86d65fca4b37440bccb71242ff91cf06`；复制后独立总结 `2a7086b57fbc17837ece17cb906b430a9188832e14d799279d109402983e93fe`。核对固定文件不等于重跑仿真；源位置在本机deployment/realman_rgb50_20261007_v1，实际正式数据在166的`/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_final_20261007/dataset`。
