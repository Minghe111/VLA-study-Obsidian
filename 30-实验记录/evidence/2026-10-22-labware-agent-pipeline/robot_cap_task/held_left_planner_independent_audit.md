---
title: 持瓶左手规划器的碰撞基线独立核查
date: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 组件核查
---

# 持瓶左手规划器：只读CPU负控

锁定held_left_planner.py SHA `07c2fae9779d1339719508f3f36372a959fd68143d03f8b32e6182195f542f23`。本审计未改task源码、碰撞掩码、物体pose、刚性连接或旧receipt，未使用GPU。实际调用本文件installer与其编译出来的原bounded audit内核、RRT wrapper；碰撞距离场、FK/model与OMPL/PlanningWorld在负控中使用明确的CPU test doubles，不代表实际路径或物理可行性证明。

六组检查均通过：实际接触选择的entity ID及sourcepart边界；实际patched audit内核对新碰撞的拒绝；RRT的teacher_category对象名与ACM一致；私有namespace与无物理写调用；旧body+cap+pill compound被actualbody重新独立覆盖；原静持实际PhysX末step接触字段的离线选择。数值receipt与测试源见同名JSON及.py。

## 原碰撞覆盖完整保留

原bounded audit明确执行categories=[*self.static,"self"]，并逐static类别、逐robot link进入距离与碰撞查询；不是仅把threshold字典扩充。测试实际查询到瓶身、瓶盖、12颗独立pill代理及table共15类，以及self。未知right/body、right/cap、left/cap、left/body、left/right、table与pill碰撞全部被拒绝；原body-rightindex基线只在该link被冻结时获准，一旦该link运动，同pair也被拒绝。另设“起点终点清、中间碰撞”的距离场，原完整路径细分发现并拒绝，未用endpoint当path通过。

构造器独立重读actualbody component覆盖static["bottle"]，即使前面安装过整个闭盖assembly compound，也不会把body基线扩大到cap或pill。cap保留实际形状，pill仅CPU规划中用包含AABB代理，物理对象不改。构造负控通过，static仍为15类。

## 基线只能来自实际PhysX

新版基线函数必须提供真实robot_link_ids；同名但不同entity ID的伪hand对象不会获准。只选择body/cap对实际right thumb/index/middle的观察manifold；left、pill、ring、新对象与未承载cap接触均不产生许可。dt及contact向量shape/finite/unit-normal检查拒绝NaN样本，不允许NaN绕过cap力阈值。

从原静持assembly step9785逐字读取body_actor_ids、impulse、normal及position后，得到10个唯一pair：9个body link manifold（包括原来确实存在、法向冲量为0的sensor manifold）与1个cap-rightindex。实际承载的body thumb/index/middle约2.419797／1.339186／1.023250N，cap-rightindex约0.499380N。基线来自这份PhysX接触观察，没有根据FCL碰撞自动增补。零冲量body manifold是否允许是实现的显式约定，不能误写成所有获准link都正在承载。

## RRT与namespace

原exact_branch_planning_model复制audit.objects全部robot collider与既有self豁免，生成独立CPU ArticulatedModel，重查实际场景FK；strict RRT随后为每个audit.static key生成teacher_+key对象，新增基线ACM使用同一命名。RRT负控证明未知pair仍在start collision检查被拒绝，baseline link影响当前movegroup时显式报错。实际OMPL搜索未在本审计中运行。

inspect/exec编译后的audit globals虽保留dict副本中的旧class符号，但该函数code不解析FullMeshPathAuditor，因此不会转回旧class；RRT globals中的fma与当前private fma是同一个模块，installer重复调用幂等。

## 当前真实q与drive的接口边界

审计mock数据采用静持actual_terminal_snapshot中的真实q36数值；installer自身只接收调用者传来的q0，不负责获取当前机器人q36或SDK drive。新左手执行入口必须另行核对：从执行当时的实际q36/drive构建规划起点，完整path通过后再逐60ms执行，并检查实际物理变化。这里不把读取旧snapshot、CPU负控通过或候选endpoint检查记成实际路径通过。

保留NativeCPUArmPlanner.solve的源码已单独核对：它在调用时读取art.get_qpos()和command24(actualdrive)，构建当前auditor，记录actual_start_full_qpos/actual_start_drive_sdk24，冻结非目标臂及hands，并在finally用_assert_snapshot核查物理状态未变。直接线段或RRT整段均须通过完整mesh audit才返回；新入口仍须确认调用approach=True且audit_approach=True，不能仅凭默认solve在approach=False时也返回路径就宣称碰撞通过。
