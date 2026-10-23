---
title: SAPIEN接触冲量字段语义独立核查与更正
created: 2026-10-22
updated: 2026-10-22
tags: [RoboTwin, RealMan, PhysX, 独立审计]
status: 已确认报告仅暴露法向冲量，已更正切向推论
---

# SAPIEN接触冲量字段语义独立核查与更正

**当前 SAPIEN 3.0.0b1 的 `point.impulse` 不是完整接触冲量。它在 PhysX 报告层被构造为法向标量乘 normal，因此切向投影近零不能证明实际摩擦或摩擦扭矩为零。** 原 v6 独审中约 1.85e-11 Ns 的投影仍是准确字段运算，但以它评价实际切向作用的解释应撤回。v4 的所有报告向量为零也只能按这一字段范围理解。

## 确切源码链

已安装包版本为 3.0.0b1。wheel 同带的 PhysX `PxSimulationEventCallback.h:626–666` 中，`PxContactPair::extractContacts` 从 `contactImpulses` 取单个 scalar，在第 663 行执行：

```cpp
dst.impulse = dst.normal * impulse;
```

该 installed header 的 Git blob 为 `5c4cb1813eb6b2f67497979bee6eeb829eba5baa`，与通过 GitHub 插件读取的 NVIDIA 官方 tag `105.1-physx-5.3.1`、commit `85befb639eb736bc8ec7d4a2e9726ec2ab1d0434` 文件完全相同。[PhysX 官方源码](https://github.com/NVIDIA-Omniverse/PhysX/blob/85befb639eb736bc8ec7d4a2e9726ec2ab1d0434/physx/include/PxSimulationEventCallback.h#L626-L666)

SAPIEN `DefaultEventCallback::onContact` 在第 50–55 行调用该 extractContacts，并把 `p.impulse` 原样放入 ContactPoint；installed callback 的 blob `8f6a8fd3bf660ccbda59d297d58be49015ac632c` 与官方 3.0.0b1 tag commit `d9031a351c39986e51bd0716bc3b51f9a05070d8` 相同。[SAPIEN 回调源码](https://github.com/haosulab/SAPIEN/blob/d9031a351c39986e51bd0716bc3b51f9a05070d8/include/sapien/physx/simulation_callback.hpp#L50-L55)

Python binding 第 327–330 行将 ContactPoint 的 position、normal、impulse、separation 作为 readonly 字段暴露，没有重建切向部分。installed Python stub 的 blob 也与该 tag 完全相同。[SAPIEN Python绑定](https://github.com/haosulab/SAPIEN/blob/d9031a351c39986e51bd0716bc3b51f9a05070d8/python/pybind/physx.cpp#L327-L330)

## 已有实际反证

独立读回 root 保存的 `contact_wrench_friction_limit_check.json`，不重跑物理：0.1 kg box 的初始水平速度为 0.1 m/s。第 5 步实际水平动量变化减重力项为 `(-0.0113842506,+0.0008959978) Ns`，而 binding 报告水平冲量为 0；法向动量残差只有约 `1.27e-9 Ns`。报告合计法向力约 5.712764 N。这个已实际运行的组件例子证明，摩擦会改变水平动量，即使 contact report 的切向投影仍为零。

此例是简单 box CPU 组件反证，不是机器人转盖或示范采集成功。源码链与该实际反证相互支持，不能将“字段不暴露”误写为“物理不存在”。

## 对既有独审的影响

- 保留 v6 的唯一正法向 point、0.0471276 N、触点位置/normal、source SHA、动作与 partial 控制链。
- v6 法向几何力矩仅是 **reported-normal-only** 部分，不是完整接触 wrench；未报告的摩擦力矩无法由该字段排除或测得。
- 保留 v4 的 manifold 与零报告法向冲量记录；不能据报告向量缺少切向分量排除实际摩擦作用。
- 真实转动归因应结合实际角动量变化与已知重力、被动线程力、关节/约束反力等；总动量/角动量变化也不能不加分解就全归给某一手指。

原 v4/v6 独审 JSON、Markdown 已按旧 SHA 保存至 `contact_report_semantics_sources/prior_independent_audits/`；当前两份报告在显著位置与文末追加更正，原始物理日志、观察器历史版本和实验结果没有改动。审计者未运行新的物理步骤，也未修改物理或 observer 源码。

[结构化来源、字节匹配和旧证据SHA](contact_report_semantics_audit.json) · [root实际组件反证](contact_wrench_friction_limit_check.json)
