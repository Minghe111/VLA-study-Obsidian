---
title: LingBot-VLA2本地RoboTwin评测-最简指令
tags: [环境, LingBot, RoboTwin, 评测]
aliases: [LingBot双终端评测命令]
updated: 2026-09-23
status: 启动前检查通过，完整策略评测待验证
---

# LingBot-VLA2本地RoboTwin评测-最简指令

适用本机已安装的 `lingbotvla`、`robotwin` 环境和已下载的官方 RoboTwin 微调权重。打开两个终端，按顺序执行；模型服务已运行时直接执行第二步。

## 1. 终端一：启动模型服务

```bash
conda activate lingbotvla
cd /home/admin123/liminghe/vla-benchmark/lingbot-vla-v2

CUDA_VISIBLE_DEVICES=0 SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
QWEN3VL_PATH="$PWD/lingbot-vla/Qwen3-VL-4B-Instruct" \
python -u -m deploy.lingbot_vla_v2_policy \
  --model_path lingbot-vla/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
  --use_length 50 --use_bf16 False --use_fp32 True --use_compile True --port 9330
```

等服务开始监听 `9330` 后执行第二步，保持本终端运行。使用 FP32、50 步动作块和编译；行首变量只对本次命令生效，无需 `export`。

## 2. 终端二：启动可视化评测

```bash
conda activate robotwin
cd /home/admin123/liminghe/vla-benchmark/lingbot-vla-v2

CUDA_VISIBLE_DEVICES=0 python -s -u tools/start_lingbot_robotwin_client.py \
  --robotwin-root ../RoboTwin --lingbot-root . \
  --task lift_pot --render-freq 1
```

默认连接本机 `9330`，运行 `demo_clean`、100 个回合、seed=0、unseen 指令，打开实时窗口并保存录像。`-s` 等价于之前的 `PYTHONNOUSERSITE=1`，避免用户目录中的 Python 包干扰。

- 换任务：修改 `--task lift_pot`；只保存录像、不打开窗口：改为 `--render-freq 0`。
- 初始阶段先检查专家轨迹，首次模型推理可能编译；出现 `infer time` 表示已执行模型推理，回合结束后显示成功率。
- 停止评测：终端二按 `Ctrl+C`；关闭模型服务：终端一按 `Ctrl+C`。
- 如果终端尚未初始化 Conda，先执行：`source /home/admin123/liminghe/miniconda3/etc/profile.d/conda.sh`。

录像与结果目录（`_result.txt` 在全部回合结束后生成）：

```text
/home/admin123/liminghe/vla-benchmark/lingbot-vla-v2/output/robotwin_visual_eval/current_robotwin_lingbot_<时间>/lift_pot/
```

## 适用范围

截至 2026-09-23，模型健康检查、WebSocket 连接、仿真任务导入及配置路径检查通过；尚未验证完整策略评测回合。第二步使用[本地兼容启动文件](/home/admin123/liminghe/vla-benchmark/lingbot-vla-v2/tools/start_lingbot_robotwin_client.py)，调用官方评测循环，未修改模型和训练框架；当前 RoboTwin 版本与官方固定版本不同，不等同于严格复现官方分数。

参数依据：[官方策略服务入口](/home/admin123/liminghe/vla-benchmark/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py)、[官方评测循环](/home/admin123/liminghe/vla-benchmark/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py)。

关联：[[LingBot-VLA 2.0项目概述]] · [[RoboTwin与LingBot-VLA默认流程对照]] · [[环境与部署索引]]
