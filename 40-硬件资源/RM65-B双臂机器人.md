---
title: RM65-B双臂机器人
tags: [资源, 机器人]
status: 三D435纯RGB仿真录制准备完成；实际腕部支架、相机内外参与实机标定待核验
updated: 2026-10-07
---

# RM65-B双臂机器人

以下来自本次对话中用户对实际硬件的说明，未进行实机核验。

| 项目 | 配置 |
| --- | --- |
| 机械臂 | 睿尔曼 RM65-B，双臂，每臂 6 轴 |
| 末端 | 双因时四代灵巧手；每手6独立控制通道，实机具体固件待核验 |
| 相机 | 头部D435×1已安装；左右腕D435×2（用户2026-10-03指定），三路只录RGB；腕支架位姿、安装状态与序列号未核验 |
| 研究方向 | LingBot-VLA 2.0 调研、VLA / WAM / RL 与语言约束 |

机器人控制地址、SDK/固件、相机序列号、标定文件、手部控制接口待补充。

关联：[[硬件资源索引]] · [[RM65-B双灵巧手：LingBot-VLA2选型与适配路线]]

## 2026-09-23 本地资料核查

Windows 材料目录：`E:\Workspace\VLA-benchmark\realmanRM65`。这次读取厂家资料与运行仿真，没有连接实体机器人。

- `rmc_dual_arm_robot.zip` 提供完整移动底盘、升降、头部、双 RM65 六轴臂模型；结合用户确认的 RM65-B 型号用于本次参考适配，尚未核对实机序列号/选配差异。左右腕关节原点/方向不完全镜像，适配保留两份厂家原始 FK。
- 整机包手选项名为 `aoyi`，网格目录为 `yinshi`，每手只有一个固定网格；不能用此目录名认定具体因时手型号。
- `四代触觉手.rar` 中 R1 手册描述 FTP 系列：12关节、6驱动自由度、12个触觉传感器。SDK `inspire_hand_SDK_2025_07_03.zip` 内可动模型却标注“三代手短手腕”，不能视为四代FTP的已核实模型。
- 六通道顺序为小指、无名指、中指、食指、拇指弯曲、拇指旋转；指令0–1000，-1保持。尚未建立硬件寄存器到仿真弧度的实测标定。
- 厂家 README 写明标品无腕相机。用户增装的左右 D435 仍需实际支架位姿、光学外参及序列号。

接入进展与验收边界见 [[RM65-B接入RoboTwin仿真]]。旧笔记中的 RH56DFX 是旧手册条件结论；本次新资料未消除实机手型号待确认项。

## 2026-10-01 双臂资料更新与用户确认

用户确认服务器现有双臂模型准确，以该模型作为对齐实机的基准；本机整机资料已更新为 E:\Workspace\VLA-benchmark\realmanRM65\rm.7z。此包内 rm/catkin_ws/src/rm_dual_arm_description 的 21 个模型源文件及 9 个驱动/控制源文件与服务器原来源哈希一致。身体关节原点、轴、限位、惯性与当前模型逐项比对一致，旧 rmc_dual_arm_robot 不再作为当前版本的基准。

实机 ROS 启动定义使用左臂 leftarm_joint1..6、右臂 rightarm_joint1..6，对应仿真 l_joint1..6、r_joint1..6；适配按六轴顺序和名称映射，不增加额外符号或零位偏移。ROS 运动指令单位为弧度，底层控制器使用度；物理安装变换保留当前厂商定义。

用户确认暂时没有 TCP、左右腕部 D435 外参或手指 SDK 编码到角度的标定文件。现场标定、相机序列号、手部实机型号/固件及实体通信状态仍未核验；本轮没有连接或驱动实体机器人。

资料对齐证据与适配实现见 [[RM65-B接入RoboTwin仿真]]，服务器状态见 [[服务器172.17.27.166-robotwin]]。

## 2026-10-02 相机安装状态澄清与仿真支持

用户明确说明：头部 D435 已安装一台；左右腕部相机目前尚未安装，先保留预留配置。此前“左右末端各已加装一台 D435、共三台”的记录据此修正为规划配置，不再视作当前安装事实。本轮没有连接实机或核验相机序列号。

当前四代灵巧手仿真已通过独立SDK六通道控制、官方开合手势接口与24维状态读取的组件检查；完整24维数据/策略接口留待后续。厂商双臂与头部原模型保持不变。

头部仿真相机已绑定 `camera_link` 并通过官方 RGB、合成深度和内外参接口验收；默认24°下俯角、镜头中心估计及0.18m手掌任务偏置均为明确的未标定仿真参考，不能称为实机外参/TCP。固定零位头部模型未改变，保留原始零位视角。腕相机默认关闭，误开启会明确报错。

来源：2026-10-02用户说明及 Conda robotwin 内运行的组件报告；完整实现、限制及证据路径见 [[RM65-B接入RoboTwin仿真]] · [[服务器172.17.27.166-robotwin]]。

## 2026-10-03 三D435 RGB要求与仿真安装预设

用户本轮明确实机腕相机也采用D435，头部/左腕/右腕三路只录RGB，不录深度。此说明更新型号与采集需求，未提供支架测量、序列号或相机标定；不能把它写成已现场核验安装。2026-10-02“腕相机未安装”的历史说明保留，不视为本轮重新核查后的状态。

仿真三路统一D435 USB3默认color profile640×480 RGB8/30fps；真实曝光/白平衡/增益应保持设备默认，各设备profile K及畸变后续实际读取。本轮仅用RoboTwin Large_D435 FOV37°名义K，不代表精确D435实机视场。左右光心作为未测量固定预设绑定l_link6[0,-.08,-.045]m、r_link6[0,-.08,-.041]m，头部最低机械俯角−.419rad与原光学安装估计不变。镜头由法兰抓握侧后移以减少掌遮挡及瓶体裁切；支架/相机实体碰撞未建模，不能替代实机安全或外参标定。

Conda robotwin中完成原成功轨迹的1条三RGB格式预录制及原生保存接口组件检查，未驱动实体机器人或新增物理成功episode。205组native配对/206原观测，保留14/24/full36状态动作和时钟，无深度/分割/点云dataset。完整来源和新参数见[[RM65-B接入RoboTwin仿真#2026-10-03 三D435纯RGB视角与录制准备]]；环境与数据路径见[[服务器172.17.27.166-robotwin]]。


## 2026-10-07 官方灵巧手SDK力反馈能力核查

本次查睿尔曼官方在线API2/JSON文档及本机厂家资料，未连接实机、读取设备传感器、更新固件或改控制。确认官方有灵巧手反馈数据接口；不能写成当前左右实机的固件/触觉能力已经核验。

- 普通UDP：`rm_udp_hand_state_t.hand_force[6]`是六自由度力反馈，官网标单位mN；同包有位置、角度、状态和错误。Python可用`rm_set_realtime_push`开启自定义`hand_state=1`，三线程模式注册`rm_realtime_arm_state_call_back`接收。不能把它当成腕部六维力，也未在当前手控制页核到独立`rm_get_hand_force()` getter。
- `rm_set_hand_force(force)`设置力阈值，官方范围1–1000，和实际力读取不同；不能把阈值编码500直接当500N或任意套用实机毫牛单位。
- 末端生态协议提供`rm_plus_state_info_t.force`（文档单位0.001N）及触觉`normal_force`、`tangential_force`、方向和原始触觉数据。先根据`rm_plus_base_info_t`的厂家/硬软件版本/`force`、`touch`、`touch_num`等实际能力选接口；当前官方样例的65535占位数不应解释成有效触觉力。三维触觉与普通6自由度力反馈的物理含义和维数不同，不直接拼入24维位置状态。
- `hand_follow_angle`/`hand_follow_pos`跟随控制最高50Hz，需要厂家定制末端工具固件；这是跟随功能条件，不泛化成所有反馈读取都必须先调用follow。该跟随接口明确顺序拇弯、食、中、无名、小、拇转；当前普通UDP正文未明确可直接沿用此顺序，实收反馈顺序仍待核验。一次搜索摘要曾写only-follow有效，当前正文未复现，故不采用为现行硬限制。

本机四代厂家SDK `sdk/python/四代手控制程序/demo_485.py` 有`forceAct=1582`和`read6`；同目录`demo_modbus.py`有Modbus读取示例。R1手册文本 `E:/Workspace/VLA-benchmark/realmanRM65/extracted/tactile_hand/manual_R1.txt` 第1193–1211行（PDF正文23/26）标`FORCE_ACT`为只读、有符号short，范围−4000–4000，单位g，顺序小、无名、中、食、拇弯、拇转。第1107–1134行`FORCE_SET`范围0–3000，用指尖握力g解释，且不同接触力臂产生不同有效力。此厂商协议与睿尔曼UDP的mN文档不能直接混用；完整接口应保留raw值、设备版本、通道名/顺序、单位、零点及校准依据。SDK演示中0–1000的旧阈值注释也不能覆盖R1手册范围，不擅自选择一种作为实机已验证范围。

`touch_data_ts.py` 的 `read_finger_data` 读取触觉法向/切向float数据；`touch_data.py`读取指尖/指腹阵列。因此资料层面同时有六路实际力与更细的触觉读取路径；具体触觉TS固件、FTP/R1型号、传感器数量与有效单位仍需实机信息核验。未执行这些demo（含设备连接/运动写入），也没有改动现有仿真或VLA策略。

核查日期2026-10-07，官网当前API2/JSON页面；页面版本号不等于实机安装版本。官方来源：

- [UDP灵巧手状态](https://develop.realman-robotics.com/robot/apipython/struct/udpHandState/)
- [UDP配置与回调](https://develop.realman-robotics.com/robot/apipython/classes/udpConfig/) · [自定义上报项](https://develop.realman-robotics.com/robot/apipython/struct/udpCustomConfig/)
- [手部控制/力阈值](https://develop.realman-robotics.com/robot/apipython/classes/handControl/)
- [末端生态实时信息](https://develop.realman-robotics.com/robot4th/apic/struct/plusState/) · [设备能力](https://develop.realman-robotics.com/robot4th/apic/struct/plusBase/) · [末端协议/跟随控制](https://develop.realman-robotics.com/robot4th/json/endTool/)

本机静态来源SHA：`demo_485.py` 985e993a9c56683e9cd43dc902a7c4b9e6450f8501c39718f3ee6570187e7579；`touch_data_ts.py` d7771247b1d07f3e858f7997a284c5bd0b21eeea120373f3691c6af84bc143b4。原20条仿真力/摩擦、是否作为模型输入见[[2026-10-04-π0.5-RM65B-RGB20微调与推理验证]]；接入仍见[[RM65-B接入RoboTwin仿真]]。
