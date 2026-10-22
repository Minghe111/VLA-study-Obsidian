---
title: 实验室药瓶资产物理接口与验收边界
updated: 2026-10-22
tags: [RoboTwin, 资产适配, 仿真, 物理验证]
---

# 实验室药瓶资产物理接口与验收边界

本目录的 `lab_physics.py` 是本次新增的派生功能机构，不是对原 RoboTwin 药瓶增加标签后就获得开盖功能，也不冒称导入的源模型原生支持旋盖。烧杯、培养皿等视觉资产来源、许可证、转换过程见同目录 `asset_interface.json`。药瓶本体、瓶盖的可视几何依据 `physics_spec.json` 参数派生，并与独立碰撞机构对齐。

## 物理与几何合同

- 瓶体坐标：米、Z 向上、原点为底面轴心。外径 50 mm，内径 42 mm，高 90 mm，底厚 3 mm；32 片重叠凸墙加独立底面保留内腔，不用整体凸包堵住瓶内。
- 瓶盖坐标：顶面圆盘中心为原点；圆盘 Z 范围为 −2～+2 mm，盖裙 Z 范围为 −12～0 mm；外径 60 mm，盖裙内径 51.6 mm，质量 12 g，明确由 `cap.mass_kg` 读取。闭合时盖原点位于瓶体 Z=93 mm。
- 螺旋导向：声明为 D6 限力、限扭弹簧驱动的近似；螺距 4.8 mm/圈，完全开盖 2.5 圈。轴向目标由**实际累计相对转角**计算，不由目标转角直接抬盖：`z_target = closed_z + pitch * measured_angle / (2π)`。
- 脱离判据：实际抬升至少 11.5 mm，实际累计转角至少 `5π−0.1` rad；提前调用 `release_cap()` 会报错。分离后取消瓶体导向，验证脚本的另一独立夹具接管并移离瓶盖。
- 药片：12 个独立动态 capsule 刚体，半径 3 mm、圆柱半长 2 mm，每粒 0.35 g。构建时设置一次初始位置，之后完全由重力、碰撞和刚体积分推进；关闭小刚体 sleep，避免移动容器失接触后药片睡眠悬停。
- 接收培养皿：与导入下皿 CAD 尺寸对齐，外径 87 mm、内径 85 mm、高 14 mm、底厚 1 mm。碰撞仍为派生凸分块。

动态模块的瓶体为 32 片凸墙；用于通用官方加载验收的 prepared 瓶体使用 48 片分块。二者内径、外径、高度与底厚相同，但离散方式不同，不能将本模块的动态验收称为 prepared 全部碰撞网格逐件相同的动力学验收。质量字段显式化后已重跑 3 个种子和卡盖负控，3 条动态轨迹与此前逐字节相同；旧合同报告保留为 `physics_prior_contract_*`。

## 使用接口

```python
from lab_physics import LabBottle, add_receiver, read_spec

spec = read_spec()
assets = ROOT / "prepared"
visuals = {
    "body": {"filename": str(assets / "reagent_bottle_body/visual.obj"), "scale": [1,1,1]},
    "cap": {"filename": str(assets / "reagent_bottle_cap/visual.obj"), "scale": [1,1,1]},
}
bottle = LabBottle(scene, spec, pose=bottle_pose, visual=True, visuals=visuals)
receiver = add_receiver(scene, spec, pose=dish_pose)

# 以下是声明的测试夹具动作，不代表机器人已抓住瓶体或瓶盖。
bottle.set_fixture_pose(fixture_target_pose)
bottle.command_unscrew(target_turns)   # 连续、小步递增至 2.5 圈
scene.step()
bottle.after_step()                    # 读取实际转角并更新虚拟螺旋导向

state = bottle.cap_state()
# 满足实际旋转、抬升判据后才可脱离。
bottle.release_cap()
bottle.command_cap_holder(cap_fixture_target_pose)
```

`visual=True` 的缺省图形仅用于几何调试；真实透明玻璃请按 `asset_interface.json` 中材质参数创建 SAPIEN RenderMaterial。简单 RGB/RGBA 列表传给 SAPIEN 的 primitive visual builder 会被当作 RGB 并强制 alpha=1，不可据此宣称透明效果。

## 复现与已完成验收

运行环境是本机 `robotwin`，物理无需渲染或 GPU：

```bash
cd /home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-integration
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python test_lab_physics.py --seed 0 --out physics_new_seed0.json
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python test_lab_physics.py --blocked-cap --out physics_new_blocked_cap.json
```

2026-10-22 实际 PhysX CPU 运行：3 个初始药片朝向种子，各运行 14,600 步、步长 1 ms。每条在闭盖倒置的全部 2,500 物理步逐步检查，12 颗均在瓶内；实际盖转角约 15.705 rad、实际抬升约 11.754 mm 后解除导向；倾倒后全部 12 颗落入培养皿，瓶内剩余 0、药片与地面的有效冲量接触为 0。独立卡盖负控中，即便请求 2.5 圈，盖实际转角为 0，轴向目标不升高，开盖解除被拒绝。

最终证据为 `physics_final_seed0.json`、`physics_final_seed1.json`、`physics_final_seed2.json`、`physics_blocked_cap.json` 及对应 50 Hz 轨迹。`physics_validation_summary.json` 验证当前源哈希、轨迹时钟、各阶段覆盖与药片数量。渲染应使用 `physics_final_seed0_trace.json`；按轨迹回放作图必须标记为**物理记录的可视化回放**，不能把回放过程中 set_pose 当作新物理验证。

早期大接收器测试和高位倾倒到小皿时漏 1 颗的失败结果保留在 `physics_pilot_large_receiver_*`、`physics_petri_seed0.json` 中；降低瓶口高度并调整固定接收皿落点后通过。默认 sleep 的开发失败及修正记录在 `physics_development_failures.json`。

## 不能据此声称完成的能力

这些结果证明资产组件的含片、封口、导向开盖与重力倒药链条可运行，**不等于 RM65-B 已自主拧盖或倒药**。当前瓶体由运动学测试夹具持有，盖由测试用约束驱动；没有机器人抓持、灵巧手接触稳定性、VLA/RL 策略训练或成功率评测。螺旋机构不包含真实螺纹接触、摩擦与扭矩标定、错扣、按压式儿童安全盖，也不模拟液体或真实药片材料参数。3 个种子只改变初始药片朝向，不能代表全面的随机化鲁棒性。

后续应先把测试夹具替换为机器人抓持与腕部旋转，保持药片动力学和开盖判据不变；再做双手协作开盖、倾角与落点控制及完整任务闭环。
