---
title: v3实际左臂过渡与末段IK失败独立核查
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 独立审计
status: 五段真实过渡完成，最终IK失败，未首触
---

# v3实际左臂过渡与末段IK失败独立核查

本记录只读检查 v3 已结束的实际输出，没有新建仿真场景，没有修改正在运行的 v4 源码、参数或旧结果。

## 已实际执行

22 项 source snapshot 全部通过 SHA 复核，入口为 `1b034ea2fcc501ac76eb1b98aaeda93f4e4d196dd32affd515294d43f6c3a868`。505 条完整目标包括原 294 条动作/状态/next-state/tick 完全一致前缀、49 条手姿预备、162 条真实左臂过渡；via0–4 分别为 82、45、10、14、11 条。五份 moving-left scoped mesh 检查均无碰撞、关节限位通过，采样数分别为 2140、1352、108、155、97。它们保留原声明的 moving-left/采样检查范围，不是连续无碰撞证书。

全部 transition 都为 15 tick，状态与 tick 链完全连续。12950 条全程 world 日志中，7575 条属于动作期，active tick 严格递增，每 action 恰有 15 条；body、cap 与 12 颗药粒的 pre/post 链无断裂，requested action 逐项对应 transition。所有实际 drive24 有限，控制窗末 drive 与请求目标最大数值差约 5.08e-5（SDK/弧度混合向量），左手阶段右臂实际 drive 未改变。终态 tick 为 7575，匹配最后 next-state。

## 接触与失败边界

全动作期保存的 body/cap 原始接触日志中，左手—盖及左手—瓶身最大法向力均为 0 N；右手—盖最大约 0.918 N。60 ms 边界记录的左手—桌、左手—右手和左手—瓶身力均为 0 N，未触发逐 4 ms 运行的禁碰/释放 guard。注意原始 `contacts_post` 只保存 body/cap 参与的接触，故不能把它说成独立保存了每 4 ms 的左手—桌/右手全场景接触原始证据；逐步 observer 源码与实际无异常结束记录支持“guard 未触发”的结论。

最终第 6 个 waypoint（via5）的单 seed IK 报 `Current-cap rebound IK failed`，发生在生成/执行该段控制目标之前。没有该段规划通过记录，没有 controller proposal，也没有 partial command 或首触事件。因此这次停机不是首触 controller 的观测容差拒绝，也不能归为真实抓盖失败。

cap 始终未 release，但其轴向间隙在 active tick 5592、assembly step 10967 首次达到 1.000435 mm，超过冻结的 1 mm closed 条件；终态为 1.319379 mm，`cap_closed_observed=false`。历史 bodyhold 子阶段成功标记仍为 true，与当前闭盖条件和完整开盖倒药成功无关。完整任务成功仍为 false，不能把附着状态或 bodyhold 成功当闭盖/完整任务通过。

## 有限实际 prepared 读回

对 v3 真实 prepared 终态 q36/drive24，使用冻结的 `202dee…` controller 和原生 ContinuousHandMapping 做了一次只读 CPU 构造及 proposal。primary 实测最大越界为 5.14323e-7 rad，低于现有 1e-6 rad 观测容差；实际 drive 关节越界为 0。constructor 和 proposal 均通过，未裁剪实测 state、未写 drive。此快照没有提供提前放宽容差的依据；它不是对未来全部物理状态的保证。

证据：[结构化独审](independent_v3_audit.json)、[核查脚本](independent_v3_audit.py)、[prepared读回](independent_prepared_controller_readback.json)、[读回脚本](independent_prepared_controller_readback.py)。
