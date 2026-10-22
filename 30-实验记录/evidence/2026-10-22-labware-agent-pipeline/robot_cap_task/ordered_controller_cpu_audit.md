---
title: 顺序单通道闭合API独立CPU核查
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 独立审计
status: 10项有限CPU接口检查通过，未运行物理
---

# 顺序单通道闭合API独立CPU核查

v5 真实重放 684 条前缀后，在新增动作之前被旧 controller 的“双通道均需非零区间”约束拒绝。该历史失败保留。root 增加可选 `allow_fixed_feedback_actuator=False`，默认严格行为不变；显式 True 时允许 index/thumb pair 中恰有一个零区间通道，两个都固定仍拒绝。此次独立核查只测试该最小 API 修复，没有修改 component 源码或仿真参数。

新 controller SHA 为 `cf7ee158a0d19ea6edaf38e03b09e4af13401f9fda9374dbfaca7042264b4abd`，原候选 SHA 为 `e6a6499750113d9747e9f21b3e6b188998a6197da56755f499a12fdaa7da6166`。使用 v5 实际 terminal q36/drive24 和原生 ContinuousHandMapping，完成 10 项检查，脚本实际 exit 0。

- index-only 阶段从真实 v5 drive 起步，constructor 与 propose 通过，固定 thumb drive 严格保持原值。
- thumb-only 阶段用 stage1 endpoint 作为模拟 drive transport fixture，constructor 与 propose 通过；此 fixture 不是已发生的物理 stage1 成功。
- 固定通道的实际 drive 比单点端点偏移 3e-7 rad，并观察到 0.2 N 接触力时，仍保持同一个实际 drive，没有被全局 clip 拉回标准端点，也不通过固定通道的 force 改动目标。
- 默认 strict 拒绝单固定；显式允许也拒绝 pair 都固定；非布尔 opt-in、超出 screened 起点及 SDK 区间的输入均拒绝。
- 在无力命令 transport fixture 中，index/thumb 分别用 8/39 个 proposal 到达端点，所有 primary/mimic 命令位置及速度检查通过；最大命令速度约 0.08 rad/s，仅有约 1e-15 的浮点表示误差。没有裁剪原始实测 q。

旧 propose 没有除以 `close_sign` 或闭合区间，零区间本身不造成除零。必须单独处理的是全局 clip 对固定通道微小 drive 表示偏差的影响；新实现已跳过该通道反馈，并在 clip 后恢复原 previous drive。映射层 SDK/mimic 严格合法性检查保持原样。

此次为真实原生 mapping 与归档实际输入上的接口测试，接触及命令执行由无写入 doubles 提供，没有新建物理场景、调用 scene.step 或写入任何实际 drive。它不证明首触、抓盖、旋盖、倒药或完整采集成功。后续同候选 v6 实际运行是独立证据。

[结构化结果](ordered_controller_cpu_audit.json) · [可复现测试源](ordered_controller_cpu_audit.py)
