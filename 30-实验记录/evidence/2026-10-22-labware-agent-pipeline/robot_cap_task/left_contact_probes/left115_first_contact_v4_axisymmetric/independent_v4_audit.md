---
title: v4实际末段接近与无载荷接触独立核查
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 独立审计
status: 160个闭合窗口耗尽，存在零冲量接触manifold，未首触
---

# v4实际末段接近与无载荷接触独立核查

> [!warning] 接触字段语义更正（2026-10-22）
> 已确认 PhysX/SAPIEN 的 point.impulse 由法向标量乘 normal 构造，不是完整接触冲量。本文原“切向 raw≈0”只能说明报告字段的投影，不能证明物理摩擦或摩擦扭矩为零。原始数值、法向首载荷与控制时序结论保留；解释以文末更正为准。

v4 真正完成了末段连接和 160 个闭合窗口，随后因预算耗尽退出。**原始 PhysX 日志有 LEFT-cap 接触 manifold，但所有保存点的法向冲量都为 0；不能将“未触发首触事件”表述为“完全没有接触点”。** 本记录无新物理运行，没有修改任何现场源码或旧结果。

## 控制与来源

22 项 source snapshot 的 SHA 全符，入口 `95fb52bf8a6d16009030a0d7c78b599cf39c16d5af3420f79dec3ce0ab67bfeb`，controller `202dee28d41db8655b6bbaa02b3560f7a3439de949fe99790d5355ba351f727d`。684 条完整目标=505 条动作/状态/next-state/tick 完全一致前缀＋19 条末段接近＋160 条闭合；160 份 proposal 均被实际执行。全部 transition 为 15 tick，状态和 tick 链连续；15635 条全程日志中有 10260 个动作期步骤，每目标恰 15 条；body/cap/12 药粒 pre/post 链无断裂，请求动作逐项对应。terminal state 匹配最后 next-state，partial 为 null。

## 原始接触判定

按 cap 实际 entity ID 104 与 `l_inspire_` link 匹配，动作期存在 32087 条 manifold 记录，包含 72025 个 contact point；其中闭合阶段有 29448 条 manifold。**正法向冲量点数为 0，全部点法向冲量精确为 0 Ns，聚合最大法向力为 0 N**。raw impulse 向量与 tangent impulse 向量的最大范数也为 0 Ns，故不是法向投影抵消而漏掉非零冲量。

首条记录出现在 active tick 7150、assembly step 12525、`left_approach_via2`，对手为 `l_inspire_left_thumb_4`；保存点 world 坐标约 `(0.091811,-0.081389,0.888150) m`，raw normal 约 `(-0.091004,0.844897,0.527131)`，raw impulse 为零。末条位于 active tick 10260、assembly step 15635，`l_inspire_index_force_sensor_3` 对 cap，保存点约 `(0.086040,-0.093632,0.880417) m`，normal 约 `(-0.132643,0.910222,-0.392303)`，impulse 仍为零。所有匹配记录连同 normals、positions、参与者 ID 另存，未丢弃 zero-load manifold。

该证据排除了“其实有非零弱载荷，只是低于 1 mN 事件阈值”的解释。零冲量 manifold 本身不能证明具备夹持载荷，也不能仅凭其存在证明精确网格已压入；PhysX 接触生成与几何间隙的关系须结合独立 actual-mesh gap 核查，不能据此推断 collision mask 或接触解算器有错误。

## 终态与因果边界

最后 proposal 的 index/opposition SDK 到达保留的 screened 端点 `(560.09765625,348.443359375)`，实际 drive 与之仅浮点读回差。两指始终未 latch，filtered force 为 0，ready=false。因此本轮是固定受筛范围内真实闭合预算耗尽，不是 controller 观测容差拒绝，也不是最终 IK 或路径检查拒绝。

preclose tick 7860 至 terminal tick 10260 的 9.6 s 中，cap 中心仅改变 0.03950 mm，轴方向约改变 0.05306°；该量不支持用“大幅漂移”解释本轮无载荷。cap 全闭合阶段仍 attached，却始终不符合原 closed 判据，终轴移为 1.315743 mm。没有抓盖、旋盖、倒药或完整任务成功；历史 bodyhold 子阶段标志不改变这一结果。

证据：[结构化独审](independent_v4_audit.json)、[逐点原始manifold子集](independent_v4_raw_left_cap_manifolds.json)、[计算源](independent_v4_audit.py)。


## 接触字段语义更正（2026-10-22）

已安装 PhysX 的 `PxContactPair::extractContacts` 在第 662–663 行将单标量转为 `normal * impulse`。SAPIEN callback 原样复制该向量，Python binding 原样暴露，因此取其切向投影接近零是接口构造的结果，不能用于排除未报告的真实摩擦冲量或扭矩。v6 的约 1.85e-11 Ns 与 v4 的零投影均保留为 **reported-normal-only 字段投影**，不当作完整切向测量。

原有法向载荷点数/大小、触点与 normal、source SHA、控制链与 partial 读回不变；没有追加物理运行或更改旧物理日志。真实扭转的归因应结合实际运动、动量/角动量及重力、线程被动力、关节/约束反力等已知力源，不能单凭该 report 推出完整 wrench。旧版独审 JSON/Markdown 已按原 SHA 保存至 `contact_report_semantics_sources/prior_independent_audits/`。

[完整源码链与实际反证](../../contact_report_semantics_audit.md)
