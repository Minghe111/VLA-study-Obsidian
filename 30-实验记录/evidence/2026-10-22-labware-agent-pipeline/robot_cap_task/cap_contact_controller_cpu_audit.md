---
title: 瓶盖接触控制器CPU独立核查
date: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 组件核查
---

# 瓶盖接触控制器CPU独立核查

本次修复cap_contact_controller.py并通过11组有意义CPU检查，使用保留工程的实际ContinuousHandMapping与12关节left-hand元数据，未创建SAPIEN scene、未触碰真实瓶盖、未验证物理持握或旋转。数值receipt见cap_contact_controller_cpu_audit.json，复现源见test_cap_contact_controller_cpu.py。

初版允许rate=.16，与要求的.08冲突；从实测q初始化会在PD误差较大时把首个drive命令跳到测量位置附近；tick经int截断且未绑定实际task计数；mimic端点与异常actualq缺乏完整核验。已分别修复。控制器现在从真实drive读回起步，首次接触仅按同一.06秒slew上限向actualq锁存；后续力反馈只在几何筛过的open/close区间内移动，不为达到.5N目标盲目越界。

force读数仍是index与thumb两组，执行器显式通过pair_actuator_indices=(3,4)或(3,5)映射；后者用于index屈曲加thumb opposition。闭合符号由每个执行器open→close求得，两个force目标和滤波各自独立。所有主关节与链式mimic都不超过.08rad/s及各自更低速度限制；负倍率/嵌套mimic与完整位置端点也经原mapping验证。

11组检查覆盖实体ID/手侧/实际hand link/切向冲量排除、接触顺序翻转、同tick深拷贝幂等、错clock与不合法tick、未执行上一proposal拒绝、首触actualq锁存与PD偏差、独立力反馈及(3,5)映射、反向闭合、SDK/嵌套mimic限速与筛选端点、异常actualq/drive/contact的无部分状态提交、三次独立对向边界ready诊断、真实mapping接口，以及静态无状态/驱动写入调用。mock状态和drive只由测试harness在模拟执行边界时更新，组件自身不写它们。

ready只说明三个连续控制边界出现所定义的对向力诊断，不表示盖已抓稳、能转动或已开盖。下一步仍需有限范围实际接触probe，核对完整碰撞、actualq/drive/force、瓶身受控及盖物理状态。相关范围见[[full_collection_contract]]。
