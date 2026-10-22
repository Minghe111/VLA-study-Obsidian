# SDK24 示范数据契约

这些是机器人在 SAPIEN 中实际执行的示范，非真机数据、非物体逐帧设姿态回放，也非VLA模型rollout。

每行对应一个60ms控制窗口。窗口开始时采集三路RGB和实测state，下发一个绝对24维target，经过15个4ms物理步得到next_state。手臂与手部在窗口内使用已验证执行器的插值与驱动，不改变物体位姿。图像对应state时刻。

## HDF5字段

| 字段 | 含义 |
| --- | --- |
| `state/vector` | N×24，窗口开始时实际关节状态 |
| `action/vector` | N×24，该窗口实际下发的绝对目标；不是下一帧实测状态 |
| `next_state/vector` | N×24，窗口结束时实际关节状态 |
| `state/left_arm_joint_states` 等四个分组字段 | 与vector完全同值的6+6+6+6分块，供XPolicyLab式读取 |
| `action/left_arm_joint_states` 等四个分组字段 | 实际target的同样分块 |
| `vision/cam_head/colors` | N个原RoboTwin codec编码的640×480 RGB JPEG |
| `vision/cam_left_wrist/colors` | 同步左腕相机 |
| `vision/cam_right_wrist/colors` | 同步右腕相机 |
| `timestamp` | 窗口开始的模拟秒数，间隔0.06秒 |
| `additional_info/ticks` | 每行[start,end]，end−start=15，相邻窗口首尾连接 |
| `additional_info/control_dt_seconds` | 0.06 |
| `additional_info/frequency` | 浮点1/0.06，不舍入为15或16Hz |
| `language_instructions` | 原语言流水线生成的seen指令JSON列表 |
| `diagnostics/phase` | 实际动作阶段标签 |
| `diagnostics/physical_metrics` | 每控制边界实际接触、离桌、倾角、速度与稳定保持指标 |
| `diagnostics/task_receipt` | 成功、来源和错误状态 |

24维顺序固定为 `[左臂6, 左手6, 右臂6, 右手6]`。手臂为弧度；手部为当前URDF实测关节经原native映射得到的**模型SDK 0..1000坐标**。它不是手指关节弧度，也未完成真机编码器标定。三相机使用模拟安装预设，不能当作真机标定外参。

本次seen／unseen为本地编写的指令和物体描述措辞划分，不等于官方50任务的评测划分，也不构成未见物体或更大场景的泛化证明。物理成功标签来自仿真接触与几何判据，文本来自任务模板；二者分别保留来源。

## 验收

逐行验证state/next_state连续、动作维度及有限性、15物理步/60ms时钟；三路图像逐帧用原 `decode_image_bit` 解码，检查形状、非空和变化；原模板语言必须12条且完成物体/臂侧替换。规划和独立reset后的命令重放分别要求真实物理任务成功。带异常或未完成任务不导出为成功示范。

原 `scripts/collect_data.py::run` 不修改。任务局部hooks保存实际SDK24命令并重放，替换通用14维/下一状态式出口；这是明确的本体适配。`launch.json`和各receipt保存精确源SHA、候选出处与物理结果。

视频以50fps输出，每个控制窗口重复3次同一同步帧，精确对应60ms；它是诊断预览，学习和重放使用HDF5时间戳。后续转LeRobot或训练π0.5需沿用原24维顺序与各自数据adapter，不应直接套ALOHA14维配置。
