---
title: v4碰撞形状与分组加载链静态独审
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 独立审计
status: 未发现sensor-only或cap-left碰撞屏蔽源码，缺运行时属性归档
---

# v4碰撞形状与分组加载链静态独审

当前源码与已保存 v4 元数据没有提供“传感器 link 只做 query”或“cap 对左手被 mask 屏蔽”的证据。所有相关 parent/sensor 都有普通 URDF collision，已运行的 planner metadata 也把它们记录为实际 PhysX convex shape。此次只读源码与既有结果，没有启动仿真、构造物理场景或更改 mask。

## 机器人形状与碰撞组

归档 `robot_head_down.urdf` 中，thumb4、index2 及 thumb/index force-sensor 子 link 都各有一个普通 mesh collision。sensor 子 link 是固定关节接在对应指节上；名称含 sensor 不等于 query-only。

`labware_task.py:161–168` 用 LabwareRobot/RealManRobot 加载，随后把 articulation 上全部 shape 的 contact offset 设为 10 mm，rest offset 为 0。原生 `native_robot.py:75–78` 统一调用 `configure_collisions`；sensor 名称仅在第 118–123 行加入 gripper 接触名集合，没有独立 query/trigger 路径。`envs/robot/robot.py:121–127` 使用普通 SAPIEN URDFLoader.load，已安装 `sapien/wrapper/urdf_loader.py:407–445` 将相关 URDF mesh 加载为普通 convex collision record。

当前 `envs/robot/realman_rm65b/physics_setup.py:46–92` 仅为机器人内部固定组件/相邻组件分配 mask。由该源和 vendor URDF 推导：thumb4/thumbsensor3 为 `[1,1,8,2001]`，index2/indexsensor2/3 为 `[1,1,32,2001]`。cap 的普通 ActorBuilder 默认是 `[1,1,0,0]`。安装的 MPLib `srdf_exporter.py:8–39` 分组判据对这些 cap-left pair 返回允许碰撞；同手固定组件的自碰撞排除不作用于 cap。这里是**源码推导值，非归档运行时 readback**。

## cap 与 FCL 表示

冻结 `robot_cap_assembly.py:138–148` 为盖创建 32 个裙边 box 与一个顶盖 convex，使用普通动态 actor build。`lab_physics.py:61–69` 只将盖的 contact/rest offset 设为 spec 的 0.4 mm/0，没有另设 cap-left mask。

安装的 `mplib/sapien_utils/conversion.py:296–355` 遍历全部 `comp.collision_shapes`，复制 geometry、shape.local_pose 与 entity.pose；相关 native scale overlay 也不按传感器名称、collision groups、offset 或 trigger/query flag 筛选。FCL 包含这些形状，确实不自动代表它复刻了 PhysX 的全部解算条件。已运行的 planner metadata 明确记录 thumb4、index2、相关 sensor 都是 PhysxCollisionShapeConvexMesh。

两物体 contact offset 之和约 10.4 mm。metadata 报告 thumb4 与 cap 约 4.793 mm 正间隙，落在该生成距离内，与“已有 manifold 但冲量为零”一致。此为可支持的近接触解释，不解决另一路 index2 compound-FCL 报 collision 的矛盾；逐 piece 几何/transform 正由独立 metadata 核查处理，不能从零冲量直接归因于 mask、penetration 或 solver 错误。

## 尚未取得的运行时事实

v4 现有 snapshot/contract 没有保存逐 shape 的实际 collision_groups、contact/rest offsets、simulation/trigger/query flags、cooked hull/scale/local pose/vertices。外部原生 loader 与安装库也不是该 v4 的全部 22 项 snapshot 覆盖范围。本次已记录所读源的当前 SHA，但不能倒推其所有属性在归档运行时必然相同。Python PhysX stub 未暴露可直接追查的 query/trigger 属性，因此本次不伪称完成这些运行时核验。

结论是暂未找到 sensor-only/mask 屏蔽源码，而非证明所有运行时 flag 已核验通过。后续若已有物理运行需要补证，可以在同一次正常运行记录这些 shape 属性；本次不为此额外启动完整仿真。

[结构化来源、逐 link 碰撞组推导与 SHA](independent_collision_chain_audit.json)
