---
title: 加载与小角转动探针最小实现审查
created: 2026-10-22
updated: 2026-10-22
tags: [RoboTwin, RealMan, 独立审计]
status: 只读实现审查未发现确定新bug，物理结果另验
---

# 加载与小角转动探针最小实现审查

审查时 probe SHA 为 `073eeb1be21749e0ff605d8922351365b8ba195e92e490462bc35d42f3d11ef6`，observer SHA 为 `59e0591dce4e4526a5b623199d12d1ea22c296e4f6b9942ca4768bdeafad8e32`。未启动物理、修改源码或重测整个pipeline；pressure候选的最终绑定及SHA仍由拥有者确认。

原 v6 一步 partial 的 request 会被重新发给正常 driver，真正执行一个新的完整15步窗口；在10981点与旧after快照逐字段比较，driver完整返回后才记 complete，不通过补日志伪造。首触stop默认保留，LoadTurnTask显式关闭；逐步guard继续工作。右手shadow和partial target反馈调用时钟一致，后续新命令从实际drive读回构造。

两指均有法向载荷连续3窗与opposed连续3窗分别记录，只有two-loaded会触发有限转动尝试，不冒称已形成稳定grasp。小orbit从实际body/cap/palm算出palm目标，再做IK和robot joint-drive；没有物体pose写入或外加物体力矩。请求角度和实际cap转角分别存储。

新wrench observer使用reported_*名称，明确实际切向摩擦不可由该报告字段观测。异常保留pending的目标、起止tick、实际终态，未填成完整transition。没有发现会必然导致无谓终止或假成功的新实现错误，因此不新增gate。

本审查不证明双指持续加载、对向夹持或实际转动，也不等于完整任务、官方Agent采集或数据集成功。

[逐项来源行号及SHA](cap_load_turn_probe_independent_review.json)
