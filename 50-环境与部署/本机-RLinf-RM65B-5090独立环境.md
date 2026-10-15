---
title: 本机-RLinf-RM65B-5090独立环境
date: 2026-10-13
updated: 2026-10-13
verified: 2026-10-13
status: 依赖及单环境PPO计算闭环通过-非正式训练验收
tags: [环境, 部署, RLinf, RoboTwin, RTX5090, RealMan]
sources:
  - 独立venv实际安装与pip check
  - 30-实验记录/证据/2026-10-13-RLinf-RM65B-本机5090最小PPO/environment_packages.json
---

# 本机RLinf RM65-B独立环境

资源：[[本机-Linux-RTX5090]]。实验与效果边界：[[2026-10-13-RLinf-RM65B-本机5090最小PPO闭环验证]]。原双A6000规划保留于[[RLinf-RM65B-双A6000最小PPO闭环验证方案]]，没有在A6000执行。

## 持久路径与来源

任务根为 `/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rlinf_local_5090_minppo_20261013`。

| 内容 | 根下相对路径 |
| --- | --- |
| 统一Python | venv/bin/python（Python3.11.15） |
| RLinf／RoboTwin代码副本 | code/RLinf、code/RoboTwin |
| 转换权重／原统计 | models/pi05_80000_openpi/model.safetensors、realman_rgb50_full50/norm_stats.json |
| tokenizer与包缓存 | cache/openpi、cache；tokenizerSHA保留于任务reports |
| 仿真运行副本 | runtime（任务专属URDF和worker运行目录） |
| 入口／资源保护 | scripts/task_env.sh、run_isolated_rlinf.py、supervised_run.py |
| 完整版本与SHA | configs/environment-freeze.txt、reports/environment_packages.json、task_code_manifest.json |
| 经独立评测的PPO权重 | logs/minppo_resume1_to3_v3/robotwin_realman_local_5090_minppo/checkpoints/global_step_3 |
| 额外400窗容量检查点 | logs/reward_window400_resume3_v4/robotwin_realman_local_5090_minppo/checkpoints/global_step_4 |

原始代码仍在 `/home/admin123/liminghe/vla-benchmark/{RLinf,RoboTwin}`；HEAD为50f22ad9a7825eeaaf4180dca5300f2fa70758be／931ce21f4f0e63feab661f349634d5a25bd92681，加已有未提交RealMan接口。资产symlink指向原RoboTwin/assets，执行不编辑来源资产；原Conda和原入口保留。80000权重来源完整绝对路径在conversion_receipt／验收总表；没有操作服务器或换到99999。

## 实际依赖与必要兼容工作

| 层 | 本次实际版本 |
| --- | --- |
| GPU | Torch2.7.1+cu128、torchvision0.22.1+cu128、torchaudio2.7.1+cu128；capability12.0真实前后向通过 |
| OpenPI | rlinf-openpi0.1.1、rlinf-transformer-openpi4.53.2、tokenizers0.21.4；仅在新venv复制其Transformers替换 |
| 仿真ABI | NumPy1.26.4、SAPIEN3.0.0b1、MPLib0.2.1、TOPPRA0.6.10、SciPy1.15.3、Gymnasium0.29.1 |
| 辅助 | Open3D0.19.0、trimesh4.4.3、h5py3.16.0等；完整清单另存 |
| JAX辅助CPU | JAX／JAXlib0.5.3、Orbax0.11.13、Flax0.10.2；JAX_PLATFORMS=cpu，禁止预占GPU |
| 配置／调度 | Ray2.52.1、Hydra1.3.4、OmegaConf2.3.1；私有本机Ray |
| schema兼容 | Pydantic2.10.6、pydantic-settings2.8.1、numpydantic1.6.6；新版Pydantic实际导入失败后固定 |

原Torch／Conda／系统CUDA／驱动未改，没有为RM不使用的规划流程构建CuRobo／Warp／PyTorch3D。副本隔离可选CuRobo导入、补齐data/scripts源码辅助链；转换副本处理Orbax／NNX包装并采用分段低内存流式写出。

仓库test-install技能的只读resolve／dry-run没有本体OpenPI＋robotwin对应CI job；因此本轮使用用户明确授权的RM65-B自定义依赖／真实入口验证，**不声称requirements/install.sh或官方CI安装测试通过**。安装来源、约束与wheel SHA保留在任务configs／reports。

## 当前可复现配置

单卡actor／rollout／env共置，B1、micro1、no_shard／use_orig_params，禁用offload、模型compile、CUDA graph、AMP和grad scaler，保持原选择性BF16。原patch在step1后同步时OOM，当前使用原生bucket，512MiB目标／原dtype／立即载入。不要换回默认patch并仅以首次更新成功判定容量。

启动器建私有Ray／tmp短路径会话，object store1GiB、8CPU、随机dashboard端口；结束时shutdown本任务。NCCL本机loopback，禁IB／P2P；TORCHINDUCTOR_COMPILE_THREADS=1。需要新标签、新输出目录，历史路径禁止复用覆盖。supervised_run持续记录整卡显存／整机MemAvailable／swap，并在可用主存低于1GiB时只停止本任务进程组；本轮成功运行未触发该保护。

评测使用独立realman_sft_single_scene.py，默认初态2010和每query的7+index噪声；传原统计、checkpoint root、budget400及新输出。当前训练预设内置eval原继承200预算且本轮val关闭，因此本轮比较只采用这个已配对400预算的独立入口；不能直接把内置HF eval与该SFT结果合并。

## 验收状态

新环境原生仿真、完整π0.5加载与400动作推理、40项针对性测试、PPO四次更新／两次恢复／分桶同步与保存通过；pip check无冲突。结束时任务模型／Ray／仿真进程已退出。

保留24维、三RGB和15×4ms；native成功不等于转直保持。SFT／step3单场景均超时，448训练动作奖励全0；正式RL、留出SR、JAX／Torch完整采样等价和更多并行尚未通过。现环境是持久可复现的最小计算验证环境，不能自动等同于原SFT JAX部署环境或研究效果成立。

关联：[[环境与部署索引]] · [[RM65-B接入RoboTwin仿真]]。
