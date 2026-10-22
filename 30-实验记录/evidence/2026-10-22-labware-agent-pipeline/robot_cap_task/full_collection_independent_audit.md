---
title: 完整开盖倒药采集入口最小独立审查
created: 2026-10-22
updated: 2026-10-22
tags: [RoboTwin, RealMan, 独立审计]
status: 接线静态核查完成，三项最小修正已交拥有者，未物理采集
---

# 完整开盖倒药采集入口最小独立审查

审查时 collector SHA 为 `563d495c74ea425060bd523f63d455a7aed2d0df998f8a77a2d1156b3d90cb43`，config 为 `25f6583a669899e3f301431d0d2b82f09c688b1ce60e92a11948f9ffa5c4d759`，full task 为 `6205a476bf4db852c89e908504a806d6a95f8660cfd42668044e3fb0cea9bd20`。只读代码及执行 pure final_gate/preflight，没有导入物理任务、启动 GPU、仿真或生成 HDF5。

原始 collect.run 的 plan→保存成功轨迹→重置同seed→加载轨迹→capture→close→merge hooks 已接入。capture 由一个 central_replay 遍历全部 requested actions，并防止同次 setup 重放两次；不让每个技能 helper 重放整程。导出的 action 来自实际 clock 的 final drive24 与15tick记录，经回读比较，不取 next measured state。SDK24、三RGB与60ms出口沿现有 exporter，不改空瓶入口。

正常关闭路径先冻结 final verdict、metrics 和 info，再执行原 run 的 close→merge 次序。AST/factory 要求完整任务身份与生成 class，不回退 bodyhold；已有空瓶旧 receipt 缺完整 full source closure，不能通过新完整任务认证。实际 preflight 仍因为 FULL_WORKFLOW_READY=False 拒绝，完整任务 play_once 仍拒绝尚未验证的工作流。

三项最小修正已交文件拥有者：

1. final_gate 第63行额外要求 cap_end_state 字符串，但真实 full task 不产该字段。实际调用纯 gate，所有声明物理指标满足后仍只因该标签拒绝。应删除额外标签门槛，沿用 released、cap_physically_clear 与完整几何读回，不造人工标签。
2. main except 未将 status 从 running 改成 failed，故失败 finally receipt 状态不准确。
3. capture 运行异常或中断可直接退出 original.run，外层 finally 当前只写receipt。建议记录每setup是否已close，对未close实例做一次安全清理；清理失败另记，不覆盖首异常。

初始状态签名已覆盖 qpos/qvel、根、瓶/盖、12药粒、dish、assembly读回和seed。补记实际 initial drive24 有溯源价值，因为控制插值从 drive 起步；这不是新增成功阈值。新全文本/资产闭包的实际 Agent 测试 receipt 尚需接入，当前类与CPU检查不等于完整工作流或数据集通过。

[逐项结构化结果及来源SHA](full_collection_independent_audit.json)
