---
title: v2左手首接触入口路径拒绝独立核查
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 独立审计
status: 未来路径采样被拒绝，未执行左手接近
---

# v2左手首接触入口路径拒绝独立核查

v2 在执行左手 approach 之前，被 CPU mesh 路径检查拒绝。它没有在仿真中执行报碰撞姿态，也没有产生左手碰盖事件。此次独审没有启动仿真或修改原始源码、结果和阈值。

## 实际运行停在哪里

归档 22 项 source snapshot 的 SHA 全部符合其 manifest，入口为 `a075af8dd6ab38f9ba6fbcb5120383c725ae34db38ab7635c888efbe2f28c100`。343 条完整目标由原 294 条动作/状态/next-state/tick 完全一致前缀，加 49 条 `left_prepare_open400` 组成；全部是 15 tick，状态链连续。没有任何 `left_approach` transition，没有 controller proposal，partial 为 null。

实际最后 active tick 为 5145，assembly step 为 10520。`actual_preapproach_snapshot.json` 与 `actual_terminal_snapshot.json` 的 q36、qvel36、state24、actual drive24、机器人根/左右掌/瓶身/盖姿态、12 颗药粒位姿和 tick 全部相同。最后一条 next-state 与 terminal state24 相同。最后实际左手—桌面力和左手—盖力均为 0 N。

因此 `rejected_path.json` 中 sample 1933（共计划 2104 个，已检查 1934 个）、path_index 1、fraction 0.919124643 的 `l_inspire_left_index_2 ↔ table` 是**未来 CPU 采样姿态**。零距离是 mesh 检查结果，不能解释成该姿态已在 PhysX 发生碰撞。关节限位检查通过。

## 坐标绑定造成什么改变

冻结入口第 155–178 行将所有 table/world transit 掌心转换为原 cap-relative，再乘当前实际 cap 矩阵；仅最终 endpoint 的 relative 另外被 screened endpoint 覆盖。这使整个远处过渡走廊跟随 cap 转动。

静持参考到 49 条左手预备动作后的 cap 原点位移为 0.5637 mm，旋转为 2.33354°；瓶身位移为 3.6066 mm，旋转为 2.01732°。独立由 `P_ref = C_ref C_actual^{-1} P_rebound` 重建原 via0 掌心后，cap-follow 使该掌心移动 6.9567 mm，其中 world Z 降低 5.9470 mm。via0 的 6 个关节也因此变化，最大约 0.02696 rad。IK 实际残差仅 1.64e-8 m、9.19e-8 rad，说明它准确求解了被改变的目标，并非 IK 未收敛。

当前 drive 与 measured 左臂起点最大差为 1.2672e-4 rad。冻结入口第 179–186 行已用同一个 drive capture 构建 `[drive, endpoint]` audit 和后续 quintic，原 audit 另加 measured 起点；此次拒绝不能归因于遗漏 drive 起点。

保留前段 world/table transit，仅末段跟随实际 cap，是针对上述坐标依赖的修正。该修正的实际输入 CPU 路径对照和后续真实 probe 是独立证据；本记录不宣称它已通过实际接近、碰盖、抓持、旋盖或完整采集。

证据：[独立结构化结果](independent_rejection_audit.json)、[计算脚本](independent_rejection_audit.py)、原 `result.json` / `rejected_path.json` / `actual_cap_rebindings.json` / `transitions.json`。
