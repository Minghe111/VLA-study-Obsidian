---
title: v6实际首载荷、触点方向与partial独立核查
created: 2026-10-22
updated: 2026-10-22
tags: [RoboTwin, RealMan, 独立审计]
status: 一次thumb侧法向载荷已核验，未形成双指夹持或转盖证明
---

# v6实际首载荷、触点方向与partial独立核查

> [!warning] 接触字段语义更正（2026-10-22）
> 已确认 PhysX/SAPIEN 的 point.impulse 由法向标量乘 normal 构造，不是完整接触冲量。本文原“切向 raw≈0”只能说明报告字段的投影，不能证明物理摩擦或摩擦扭矩为零。原始数值、法向首载荷与控制时序结论保留；解释以文末更正为准。

v6 的实际左手—盖首载荷成立：active tick 10981、assembly step 16356，thumb4 对 cap 出现唯一正法向冲量点。该结论来自 world 日志与带 separation 的独立接触日志一致读回；本审计没有新建物理场景、调整阈值或修改运行源码。

## 执行与中断边界

23 项 source snapshot SHA 全符。684 条前缀的动作、state、next-state 和 tick 完全相同；新增完整 index 阶段 10 条、thumb 阶段 38 条，总 732 条完整 60 ms transition。随后第 49 个新 proposal 在第 1 个 4 ms 步骤首触，形成 partial 10980→10981，没有补写成完整 transition。

原始动作期为 10981 步：每个完整 action 恰有 15 条日志，最后 partial 仅 1 条；tick、请求动作、body/cap/12 药粒 pre/post 链全部连续。partial.before state 对应最后完整 next-state，partial.after 对应 terminal，实际 drive 读回也逐项一致。最终 drive 是部分执行后的读回，不冒充尚未执行完的 request endpoint。

## 唯一载荷与方向

参与者原始顺序为 `[l_inspire_left_thumb_4, derived115_closed_tablet_bottle_cap]`，entity ID 为 `[77,104]`。唯一正点的法向冲量为 `1.8851041926e-4 Ns`，实际步长为 `0.00400000019 s`，得到 `0.04712760258 N`，与事件记录精确一致。保存的 separation 为正 `3.75741456 µm`；首载荷不以人为改成负 separation 为前提。

触点 world 坐标为 `(0.116748512,-0.145077094,0.877645075) m`，raw normal 为 `(0.514505088,-0.853391945,0.083706625)`。按现有 observer 的参与者顺序归一到 cap 侧后，cap frame 内的触点为 `(0.012189551,-0.027426245,-0.009736736) m`，单位 normal 为 `(-0.382261641,0.924054104,0.000266841)`。

该 normal 与 outward radial 的点积约 `-0.999662`，与 cap 轴的点积约 `0.000267`，主要是侧壁径向挤压。raw tangent impulse 范数仅 `1.85e-11 Ns`，处于法向浮点投影残差量级，没有可据此主张的摩擦扭矩。按 observer 方向计算的法向力绕 cap 轴几何力矩约 `3.68e-5 Nm`，只是触点几何量，不是持盖扭矩或实际转盖能力；绝对 impulse-on-body 方向约定未在本次新增标定。

## 支持什么，不支持什么

动作期共有 56533 条 LEFT-cap manifold、131733 个零法向冲量点，只有上述 1 个正点。raw 左手—瓶身最大力为 0；末态左手—桌、左手—右手、左手—瓶身 force observer 均为 0，guard_failure=null。末态 index-cap 载荷仍为 0，left-cap opposed=false，不能把右手抓瓶身的历史 opposition 当作盖的双指 opposition。

cap 未 release，轴向间隙仍约 1.311094 mm，当前 closed=false；unscrew/pour/full task 均为 false。该结果是可靠首载荷证据，尚不包含双指持续加载、机器人诱发的实际扭转或数据集采集成功。下一有限实际探针可使用上述真实 normal/position 观察第二指载荷与切向冲量；此建议不增加新的人工成功标签或阻塞可逆仿真。

[结构化结果与唯一正触点原值](independent_v6_contact_audit.json) · [复核脚本](independent_v6_contact_audit.py)


## 接触字段语义更正（2026-10-22）

已安装 PhysX 的 `PxContactPair::extractContacts` 在第 662–663 行将单标量转为 `normal * impulse`。SAPIEN callback 原样复制该向量，Python binding 原样暴露，因此取其切向投影接近零是接口构造的结果，不能用于排除未报告的真实摩擦冲量或扭矩。v6 的约 1.85e-11 Ns 与 v4 的零投影均保留为 **reported-normal-only 字段投影**，不当作完整切向测量。

原有法向载荷点数/大小、触点与 normal、source SHA、控制链与 partial 读回不变；没有追加物理运行或更改旧物理日志。真实扭转的归因应结合实际运动、动量/角动量及重力、线程被动力、关节/约束反力等已知力源，不能单凭该 report 推出完整 wrench。旧版独审 JSON/Markdown 已按原 SHA 保存至 `contact_report_semantics_sources/prior_independent_audits/`。

[完整源码链与实际反证](../../contact_report_semantics_audit.md)
