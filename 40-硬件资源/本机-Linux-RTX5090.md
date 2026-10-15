---
title: 本机-Linux-RTX5090
date: 2026-10-13
updated: 2026-10-15
verified: 2026-10-13
tags: [资源, 硬件, RTX5090, 本机]
sources:
  - 本机nvidia-smi、lscpu、psutil实际只读查询
  - 30-实验记录/证据/2026-10-13-RLinf-RM65B-本机5090最小PPO/hardware_inventory.json
---

# 本机Linux RTX5090

- 类型／用途：本地Linux工作站，RM65-B三相机仿真、π0.5推理及单环境PPO小实验。
- 最近实际核验：2026-10-13；主机admin123-System-Product-Name，Linux7.0.0-38-generic／glibc2.39。不是历史Windows路径或161／166服务器。
- GPU：1×NVIDIA GeForce RTX5090；nvidia-smi总量32607MiB，UUID GPU-ca2dd5de-58fa-1e27-0c50-6de5293c94ac；驱动595.99.02。
- CPU：Intel Core i7-14700KF，28逻辑CPU；未作CPU吞吐／散热基准。
- 主存：33387954176字节，约31.10GiB可识别物理容量；swap约8GiB。实验中整机swap用量最高接近8GiB，不能套用为无swap的最低RAM要求。
- 工作目录：/home/admin123/liminghe/vla-benchmark；笔记库/home/admin123/liminghe/VLA-WAM-RL。任务根与源资产具体位置见环境卡。
- 实测能力：新统一Python环境三RGB＋SDK24仿真通过；单模型推理峰值约12.1GiB；PPO分桶同步、更新、保存／恢复峰值约29.3GiB，400动作窗最低剩余主存2.87GiB。
- 使用边界：B1／micro1小实验已通过；更多并行、全FP32专家更新、长时间稳定性未核验。建议后续扩展先增加主存至64GB，再独立测GPU容量；不由显卡理论参数推断与A6000／3090的实际训练速度排序。

关联：[[硬件资源索引]] · [[本机-RLinf-RM65B-5090独立环境]] · [[2026-10-13-RLinf-RM65B-本机5090最小PPO闭环验证]]。


## 2026-10-15 ACoB最小算例容量补充

本机JAX／OpenPI独立运行已通过RM65-B无纠正、batch1、H16/K1/ODE10的1次critic warmup＋1次专家LoRA joint update，训练阶段整卡显存峰值17.81GiB，独立进程重载并带三相机仿真再执行峰值21.04GiB；对应更新／重载阶段主存最低可用约6.49GiB。原完整状态JIT因冻结参数透传副本曾OOM，改为只返回动态参数的数值等价接口后通过。

此为2026-10-15实际算例容量，未重新核验全套硬件库存，不改上方10月13日清单的验证日期。不能推定完整Replay＋Correction batch2、并发ACoB-Stream、大回放或长期训练稳定性。见[[2026-10-15-VLA-Precision-RM65B仿真复现启动与验收]]及[[本机-VLA-Precision-RM65B最小ACoB闭环复现方案]]。


## 2026-10-15 完整ACoB-Stream实际容量补充

随后在另一独立根实际完成单环境、batch2、三RGB／SDK24／60ms的完整ACoB机制闭环：成功专家示范、489对在线纠正、三Buffer、并发Actor／Learner、3次LoRA及8次critic更新、保存和独立恢复。Stream整卡显存峰值21.25GiB，进程树RSS19.23GiB、最低可用主存6.43GiB；独立恢复并执行完整原生自主评测回合峰值21.14GiB、RSS20.41GiB、最低可用6.64GiB。无OOM或资源保护中止。

该结果补充了上节未验收的B2与本地共享内存Stream范围；没有重新核验硬件清单、更多环境、长训或跨主机网络。自主评测0/1，容量通过与任务学习分开。详细收据见[[2026-10-15-RM65B完整ACoB仿真闭环补齐]]。
