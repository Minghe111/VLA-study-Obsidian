---
title: 115毫米固定右臂保持的独立物理证据核查
date: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 物理证据
---

# 固定右臂保持：独立CPU核查

本run实际执行269条原轨迹加25条固定臂／原手部力反馈命令，共294 targets、4410 active ticks。独立复算前269条state/action/next_state/tick与v3源逐项相等，15份source snapshot SHA全部匹配；每动作15个4ms，state链及body/cap/12药片pre→post链精确连续。实际drive终点与action、内部线性时钟读回最大误差4.375e−5。terminal state24/q36/12药片pq与最后实测记录一致。原反馈shadow日志64个边界均SDK差0；本审计独立检查实际输出命令与状态，未重新运行teacher或仿真。

静持段375步，即1.500000071秒，右臂6维drive每4ms与段起点完全相同；左臂及左手12维drive也固定，右手6维继续原反馈，未冒充整手固定。真实table载荷每步为0，body对向接触每步成立。盖在全部静持步均满足原closed阈值，无release；相对轴移范围−0.207334至0.675828mm，实测相对转角−0.004674至0.023199rad。

12颗完整capsule逐4ms pre/post按冻结spec的0.4mm接触容差全部包含，保留零容差与原始margin在JSON；不声称零穿透。实际right index首次接盖为active4089，法向力0.137813N；静持全段cap-hand为0至0.578724N，最终index_2为0.499380N。末body支撑由thumb_4=2.419797N、index_2=1.339186N、middle_2=1.023250N组成。该接触没有触发解锁；它说明后续左手动作需要考虑真实右手与盖的当前接触状态。

按inv(body)·right_palm重算，整个1.5秒相对参考变化为2.328424mm／0.887041度。原result用最后10个60ms采样点得到1.482834mm／0.558477度，但点间时间跨度实际为0.54秒。严格最后0.60秒应取11点active4260–4410，变化为1.675438mm／0.643137度。末体速21.582mm/s，静持段速度范围4.852至46.165mm/s。

这些量描述实际几何参考怎样随时间变化，不是新加的否决阈值。actual_terminal_snapshot可作为该时刻的实测起点；不能假定它随后完全不动，也无需仅因轻接触或非零速度就宣布完整目标不可行。下一步应以实际当前几何筛选并有限执行左手接近，继续检查右手持瓶、盖状态与药片包含。

workflow所有294条记录的metrics.success及full_unscrew_pour_success均为false，check_success在源码中固定false；持瓶stage通过没有提前变成完整任务成功。本CPU审计新增物理trial为0，未修改probe源码、source snapshot或旧receipt。数值与哈希见independent_static_hold_audit.json，计算源见同名.py。
