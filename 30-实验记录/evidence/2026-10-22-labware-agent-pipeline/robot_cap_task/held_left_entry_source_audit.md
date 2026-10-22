---
title: 左手首接触入口动态内核独立审计
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 独立审计
status: CPU实际内核调用通过，尚不代表物理路径成功
---

# 左手首接触入口动态内核独立审计

本记录核查 `left_cap_contact_probe.py` 的实际动态编译/执行入口，并使用模拟距离与 FK 接口调用保留的原始 audit 方法体。没有创建仿真场景、写入驱动、改变物理碰撞 mask 或执行真实路径。

## 已发现并复核的两处实现问题

旧入口 SHA `40cf66a2556505bb685d96f15571f7f24341cabe30397a75c40d20912bf19b7e` 虽可通过顶层编译与导入，但调用 `moving_path_audit` 时，动态插入的 self-loop 条件缩进不匹配，`exec(compile(...))` 实际抛出 `IndentationError`。原失败源码及异常记录保存在同名子目录，未覆盖旧仿真失败记录。该问题由源码拥有者修复。

旧入口还只把末端 waypoint 传给 audit，实际控制五次曲线从 drive target 起步，存在被测 measured q 与实际 drive target 起点不一致的问题。新入口捕获一次实际 `actual_drive_left`，audit 接收 `[actual_drive_left, endpoint]`，五次曲线使用同一捕获值；原 audit 同时将 measured q 作为前置起点。归档真实静持快照中两者最大偏差约 `1.2672e-4 rad`，此次修复覆盖了实际执行线段，不涉及调整几何阈值或物理参数。

## 实际 CPU 复核

当前入口 SHA `a075af8dd6ab38f9ba6fbcb5120383c725ae34db38ab7635c888efbe2f28c100`，task SHA `38bbfea9850a3329b9f556ba6012ca28015157309a56dff240319ff2a477db85`，planner SHA `07c2fae9779d1339719508f3f36372a959fd68143d03f8b32e6182195f542f23`。

- 动态 `exec` 真实执行，无碰撞的非空路径通过。
- 空 `(0, 6)` 路径拒绝。
- 左手对盖、瓶身、右手、桌面、药粒的碰撞负控全部拒绝。
- 当前源从 live articulation 读取 q36；每个新 action 从实际 command24 起步，右臂保持实测结束 drive，右手保留原反馈，不使用 metadata 中右侧实测值充当 drive。
- controller 在左手实际预备姿态和 approach 后才构造；中途不足 15 tick 的命令另存 partial，不伪造完整 60 ms transition。

## 判定范围

`moving_path_audit` 是明确的 moving-left 采样诊断，跳过纯 frozen-only pair。负控中未许可右手 link 与瓶身的新碰撞被完整 held audit 拒绝，但被 moving-left scoped 入口排除。这是两种接口的范围差异，不能把 scoped 入口的通过描述为完整 held baseline-only audit 或连续无碰撞证书。

此次 CPU 复核只解决实际证明的入口实现错误，没有增加新物理 gate。真实首接触/抓盖/旋盖/倒药及完整官方 Agent 数据集采集仍须由各自实际运行证明。

证据：[结构化结果](held_left_entry_source_audit.json)、[持久测试源](held_left_entry_source_audit.py)、[完整 held planner 独审](held_left_planner_independent_audit.md)。
