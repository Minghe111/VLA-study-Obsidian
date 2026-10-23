---
title: 装配体载荷分配器独立源码审查
updated: 2026-10-22
tags: [RoboTwin, RealMan, source-audit]
---

核查版本为 `assembly_contact_allocator.py` 的 `b11f9cda70980304cae839a32786be138318454114bc3e6e4ca6a77eba2f06d5`。本次仅核输入路由和已安装类的接口，没有创建场景、运行 GPU 或执行新的物理实验。

原生教师在构造时从模块全局取得 `BoundaryContactController`，恢复 MRO[1] 后会使用原类。其 `target` 同样从模块全局取得 `allocate_contact_forces`，因此实际进入新的装配体 shim；不再经过原适配类的 `direct_targets` 前后重置。shim 每次调用读取瓶身、瓶盖和全部药片实体的质量及世界质心，求总质量和质量加权质心，并以实际场景重力构造 `-M g`。原 LP 算法、上一轮分配、接触点与法线保留，力矩臂改用新的总质心计算。已安装接口确实包含 `Scene.get_physx_system` 与 `PhysxCpuSystem.get_config`。

恢复原控制器也恢复其原有判据：LP 可用、三个主指均有载、对向接触、滤波载荷达到目标的 80%，连续五个边界后才进入抬升；carry 时 LP 连续不可行达到 0.42 秒会终止。它与旧适配类强行 `carry=False` 的行为不同，不能只描述为替换日志输入。

这一分配仍是给定 μ、零净外力矩和准静态支撑假设下的模型；原生接触证据以瓶身接触为主，其他外部支持没有自动纳入。没有证明实际摩擦、动态稳定或真实抓持。释放后显式拒绝继续套用闭盖装配体模型。源中未发现 step、set_pose、关节驱动或物体外力写入调用；冻结原生文件未修改。

机器可读记录见 [[assembly_allocator_independent_source_audit.json]]。
