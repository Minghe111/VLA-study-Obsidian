---
type: independent-physical-evidence-audit
updated: 2026-10-22
scope: body-feedback-v4-first-window-read-only
tags: [RoboTwin, RealMan, labware, evidence]
---

# 瓶身反馈 v4：零误差下绝对 IK 映射破坏原 drive 预载

24 个冻结源 SHA 均相符。转动前的 q36、qvel、SDK24、drive、root、双掌、瓶身、瓶盖与 12 药片 pose 与 v3 完全一致；此前 738 个完整目标逐项一致。该运行仅新执行 11070→11085 的一个完整 60 ms 窗口，总计 739 目标/11085 步，状态/world 链连续，partial 为空。未完成任一 0.005 rad 增量，不能与 v3 已完成的四次增量直接比较效果。

首反馈记录的瓶身位置与旋转误差恰为零，几何 helper 的目标为当前实测右掌。但旧接口直接把 absolute IK 的实测关节解替代原 drive，导致最大 r_joint6 命令变化 −0.0080906965 rad，峰值命令速度 0.134845 rad/s（仍在 0.45 界内）。这不是物理误差要求的修正，而是把原 drive 与承载下 actual q 的差吸收到新命令中。

使用本次 URDF 的独立纯 FK，actual q 对实际右掌的残差为约 0.085 µm、3.22e−7 rad，验证坐标链。旧 drive 到 absolute-IK 目标的虚拟 FK 变化约 0.86878 mm、0.00869479 rad；改为 current_drive+(qIK−actual_q) 后，因 IK 数值残差仅改变约 0.331 µm、8.14e−7 rad。虚拟 drive FK 是运动学诊断，不是承载下的实际物理轨迹。严格零误差同命令还需要显式 zero-error 分支，以消除 ~1.33 µrad 的 IK 数值残差。

物理首窗中瓶身实际移动 1.13156 mm、转动 0.00863669 rad；右掌移动 0.39837 mm、转动 0.00742126 rad；左掌仅移动 20.95 µm、转动 0.00046578 rad。逐 4 ms 检查该窗双指对向条件均未保持，末态 index=0、thumb≈1.12169 N；新增右手—桌面/药片/自身/非手部瓶身，以及原左手禁碰类别的法向负载均为零。该时序支持先修映射再评估稳定效果，不能单凭这一轮完全分离所有接触因果。

下一次 proposal 因原始 thumb_2=−128.92517 µrad 超出 100 µrad 观察预算而拒绝，follower thumb_3=−693.84096 µrad。未修改实测值、命令界或物理参数；盖未释放、完整任务仍 false，不记作数据集成功。

独立 CPU 反事实共五项通过：零运动学增量精确保留 drive；有限增量叠加后仍限速；实际归档 IK 残差保持在微弧度尺度；严格关节上下界向外增量均拒绝。见 `independent_preload_mapping_cpu_audit.json`。这只证明映射性质，不证明修复后的物理效果。

SAPIEN/PhysX 限位源码另见上级 `joint_limit_solver_semantics_audit.json`：有限关节范围映射为 eLIMITED 与 low/high，PhysX 文档说明迭代数影响 contact/drive/limit 解算精度；此任务的 1/10/100 µrad 是观察校验，并非仿真器保证，也不是碰撞形状的 contact_offset。本次未新增读取实际 articulation 迭代数，不以默认值解释该具体越界。所有 reported impulse 仍仅为法向报告，不能反推出完整摩擦力矩。
