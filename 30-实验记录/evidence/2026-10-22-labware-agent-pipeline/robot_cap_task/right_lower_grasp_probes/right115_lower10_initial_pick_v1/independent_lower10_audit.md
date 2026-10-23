---
type: independent-physical-evidence-audit
updated: 2026-10-22
scope: lower10-private-pick-and-fixed-arm-observation
tags: [RoboTwin, RealMan, labware, evidence]
---

# Lower10：闭盖持瓶阶段已到达，固定臂观察中仍有显著相对运动

独审只读实际 terminal 证据，无新增 GPU 或物理步。23 份冻结源 SHA 匹配，入口 `1008e547…`；初始 q36/qvel、SDK24 drive、root、双掌、瓶/盖/药片等全字段与原真实初始参考一致。未要求复制旧 269/732 动作前缀。私人 teacher 设置中 hand_x 由 −0.015 改为 −0.005，其他默认姿态、SDK800、资产和被动组件保持；保留了 7 次真实 native 规划报告及实际动作记录。

共 280 个完整目标/4200 个 4 ms 步，每目标 15 步，状态、tick 和 world pre/post 链连续，无 partial。第 270 个目标、4050 步达到闭盖持身子阶段；随后真实执行十个 60 ms 固定臂目标窗口。左右臂与左手请求保持，右手继续原 native 接触反馈。arm 目标在实际 URDF 关节界内；raw SDK 读回携带最多 −4.37e−5 的浮点 canonicalization 微偏移，未以 clamp 改写成虚假实测。请求、drive 和实测关节状态分别记录。

固定臂观察的全部 150 个 4 ms 步仍满足原 closed 判据及右手对向法向载荷，瓶身—桌面载荷为零，既有禁碰类别法向载荷为零。真实 12×7 药片 pre/post pose 共 300 个状态独立重算完整胶囊几何，均在声明的 0.4 mm 接触容差内；最差径向原始余量 −63.73 µm、底面 −3.23 µm。这不是零容差水密 CAD 证明，也不等同整程采集成功。

首个右手触盖正法向载荷在 active tick 4182，即固定臂观察开始后 0.528 s，由 right index_2 产生约 0.020414 N。此前 131 个观察步（0.524 s）为零，未保持完整 0.6 s 的零触盖载荷；末两边界载荷约 0.15830/0.48978 N。固定臂命令不是物体静止：瓶身相对右掌变化约 8.05071 mm、7.97366°，右掌自身仅移动 0.14240 mm；瓶身实际线速度最小/平均/最大约 6.65/18.38/30.88 mm/s。相对位姿变化包含手指反馈、柔顺与可能的接触滑移，不能仅据掌相对位姿分离全部表面滑移机制，但它与迟发 index 触盖相伴，不能作为稳定不变的几何参考。

边界离桌量从 24.101 mm 降至 18.991 mm；4170 边界已低于原源码当前 held 判据的 20 mm。记录中的 stage true 是曾达到 0.5 s 的历史子阶段，不能替代当前条件或静止保持结论；这没有新增阈值。原始 qvel36 在 60 ms 边界记录，body 速度是 4 ms 数据；目标限速并非实际速度硬保证，手指瞬态 qvel 保留未裁剪。

源码另有重要边界：既有“LP advisory/direct targets”说明不是严格隔离证明。外层 `super(carry=False)` 仍可能进入 native 已配对 LP 分支，usable 时 allocation 会影响本次 SDK，再由外层恢复 force_targets；不能仅调整 direct_targets 就声称固定了本次执行的所有目标。该更正不改变本次真实物理记录、计划执行或载荷读回。

本次错误为空、退出码为 0，只说明该私人持瓶及观察程序走完。盖未 release，full_task_success=false；未完成开盖、倒药或官方采集闭环。完整源码哈希、规划报告、逐步接触、包含和关节速度证据见同目录 JSON 与脚本。
