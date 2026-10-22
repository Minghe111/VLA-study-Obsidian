---
title: 瓶盖控制器实际静持读回CPU烟测
date: 2026-10-22
tags:
  - RealMan
  - 组件核查
---

# 实际q36与drive输入烟测

使用held115_static_reference_v1的实际terminal q36／actual_drive_target24、保留工程ContinuousHandMapping，以及left115_static_rebind400的open／firsttouch SDK6。未创建新物理场景，未执行drive或状态写入，实际读数没有clamp。

当前左手primary最大轻微越URDF下界7.12317e−7rad；实际drive最大轻微越上界5.09262e−8rad，两者均通过控制器既有1e−6rad观察容差。actual mimic没有被强制等同于命令耦合；mapping严格范围核验作用于新命令。实测primary逆SDK略高于1000、drive逆SDK的末通道为−4.37473e−5，这些读回不应直接冒充合法SDK命令端点。

几何候选open=[0,0,0,1000,1000,400]、firsttouch=[0,0,0,560.09765625,1000,348.443359375]，pair=(3,5)，两个端点通过实际完整mapping。但此时实际左手仍factoryopen，前三指和opposition尚未预置。因此constructor按设计拒绝“Actual drive must start inside the screened posture interval”，没有返回proposal。

此拒绝不是actualq观察容差不足，也不是物理抓盖失败。应先在真实流程中完成候选open预置，再读取新的实际q/drive并构建controller。不能修改snapshot、假设drive已到位或clamp实测状态来强行获得单proposal通过。当前证据不支持放宽观察容差，也没有更改控制参数。明细与复现源见同名JSON和.py；物理触盖、抓稳或旋转仍未在本烟测验证。
