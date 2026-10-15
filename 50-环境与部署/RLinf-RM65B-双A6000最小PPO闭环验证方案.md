---
title: RLinf-RM65B-双A6000最小PPO闭环验证方案
date: 2026-10-13
updated: 2026-10-13
source_checked: 2026-10-13
status: 静态核查后的执行方案-未安装未转换未运行
tags: [VLA, RLinf, RoboTwin, RealMan, π0.5, PPO, A6000, 闭环验收]
sources:
  - 用户要求两张A6000最小PPO闭环方案并明确暂不执行
  - RLinf本地50f22ad9a7825eeaaf4180dca5300f2fa70758be加未提交RealMan适配
  - RoboTwin本地931ce21f4f0e63feab661f349634d5a25bd92681加未提交RealMan适配
  - https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html
  - https://pytorch.org/get-started/previous-versions/
  - https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479
  - https://github.com/google/orbax/tree/v0.11.13
---

# RLinf-RM65B：双A6000最小PPO闭环验证方案

> [!abstract] 推荐起点
> **GPU0负责Actor训练，GPU1负责Rollout采样＋单环境仿真。π0.5＋Flow-SDE＋PPO，冻结VLM，更新动作专家、动作侧投影和价值头，不加LoRA。** 先验依赖与转换，再测同协议SFT；用T16两轮验证计算链，随后用能覆盖真实奖励事件的T验证学习信号，保存、续跑并独立评测。

> [!important] 本文状态
> 本轮仅阅读代码、历史证据与官方资料，整理方案。没有登录服务器或预留GPU，没有安装依赖、修改工程代码/训练配置、恢复或转换权重、执行模型、启动评测或训练。文中配置、命令、补丁、报告名称均为后续执行计划。GPU容量、依赖组合及PPO效果尚未实测。

关联：[[π0.5-RealMan强化学习研究路线-算法与训练闭环]]、[[RLinf-VLA强化学习上手-RoboTwin与LIBERO]]、[[服务器-172.17.27.161]]、[[服务器172.17.27.161-robotwin]]、[[2026-10-12-RealManπ0.5八万步原场景本机推理与实时画面]]。

## 1. 固定实验合同与验收对象

| 项目 | 本轮保持的协议 |
| --- | --- |
| 物理状态/动作 | SDK24：左臂6 rad、左手6 SDK、右臂6 rad、右手6 SDK；使用实测状态，保留原单位与映射 |
| 相机 | head、left wrist、right wrist三RGB8，原生480×640；保留现有模型内部缩放，`center_crop=false` |
| 控制时钟 | 正常target执行15×4ms＝60ms；终态可提前停止，以实际physics ticks记账 |
| π0.5 | 物理24维、内部32维、预测H16、每次执行K1、10步采样；固定原Full50量化归一化统计与指令 |
| SFT/RL比较 | 都使用H16/K1、同初态、相同物理时限和成功/失败/超时判据；原JAX执行K16的结果单列 |
| 任务时限 | 候选400个target，名义上限24秒仿真时间；须经SFT验证足够，调整后重做同协议基线 |
| 初轮目标 | 工程闭环采用现有`native`成功/终止，独立记录对象姿态；不把native成功叫作转直并保持 |

60ms是仿真推进时间，不要求每60ms墙钟完成一次π0.5推理。H16/K1提高观测反馈频率，也增加查询成本；这项变化不能归为RL收益。SDK动作不套用ALOHA14维变换，不为了省显存减少相机或改变控制周期。

当前`native`只检查选侧功能点的x/z位置，不检查转直或连续持握。第一轮沿用其成功+1、明确失败−1、普通步/超时0的接口行为，并逐项核对终态。正式研究若改为`upright_hold`，先通过真实角度、接触、持握时长与视频验收，再重做SFT基线及奖励闭环；不能直接继承本方案native成功率。

**两级验收定义：**

1. **计算闭环**：当前策略交互→GAE→两次真实optimizer更新→新权重参与下一轮采样→保存→训练状态续跑→独立评测。全零奖励可以检验部分计算链，但不能证明任务回报学习。
2. **奖励学习信号闭环**：按SFT事件长度选择采样窗，在当前策略交互中出现真实正/负事件，优势、动作专家梯度与有效参数更新可追溯。少量更新不要求SR提高，也不构成研究效果结论。

## 2. 资源与独立目录

优先候选主机为[[服务器-172.17.27.161]]。多卡A6000是资源卡历史记录，本轮未核验在线占用。执行前确认主机、账户、绝对路径/realpath、两张空闲GPU的UUID与显存、驱动、CPU/RAM和磁盘，按任务实际分配为逻辑GPU0/1。A6000每卡48GB见[NVIDIA官方规格](https://www.nvidia.com/en-gb/products/workstations/rtx-a6000/)；两卡显存分别使用，当前OpenPI要求`no_shard`，不能视为给Actor提供96GB。

建议新任务根（尚未创建）：

```text
/bigdata/liminghe/rlinf_rm65b_minppo_20261013/
  code/RLinf/                    # 当前适配快照＋后续必要兼容补丁
  code/RoboTwin/                 # 独立代码副本，保留原入口
  envs/rlinf_rm65b/              # 统一PPO＋仿真Python
  envs/rm65b_ckpt_convert/        # CPU转换/参考环境
  models/pi05_rm65b_80000_openpi/
  configs/                      # 新预设，原预设不改
  runtime/                      # 各spawn仿真独立工作目录
  logs/ reports/
```

原服务器工程位于`/bigdata/liminghe/VLA-benchmark`；资产路径应在执行时核实，候选为其`RoboTwin/assets`。资产作为只读来源，校验vendor URDF/相机/控制合同相关文件SHA。原SFT环境、资产、checkpoint和运行入口保留。

代码来源是当前Linux本地`/home/admin123/liminghe/vla-benchmark/{RLinf,RoboTwin}`。两个HEAD分别为`50f22ad9a7825eeaaf4180dca5300f2fa70758be`、`931ce21f4f0e63feab661f349634d5a25bd92681`，**仅克隆HEAD不会包含未提交RM接口**。后续移交应记录commit、diff、未跟踪适配文件SHA及依赖wheel哈希，确认完整快照，而不是依赖旧`/tmp`副本。

工作盘建议预留至少100GB容纳转换权重、两个完整训练检查点、续跑及缓存/视频；CPU转换需要源树和目标张量，RAM峰值未测，建议先确认64–128GiB可用工作余量。这些是工程预留建议，不是实测最低要求。

## 3. 依赖准备：新环境分开，先锁定再安装

### 3.1 候选版本与安装顺序

| 层 | 候选与要求 | 后续验收 |
| --- | --- | --- |
| Python | 两个独立Python3.11环境；优先3.11.17以对应原SFT记录 | 主进程、Ray Worker及spawn子进程使用同一PPO环境绝对Python |
| PyTorch | `torch 2.7.1 / torchvision 0.22.1 / torchaudio 2.7.1`，cu126 wheel | A6000 CUDA/BF16、实际attention前后向；驱动与wheel兼容 |
| 仿真ABI | `numpy 1.26.4 / sapien 3.0.0b1 / mplib 0.2.1 / gymnasium 0.29.1 / opencv 4.10.0.84` | cp311 wheel、导入、三相机真实render、一次reset/step/close |
| 辅助仿真包 | `toppra / transforms3d / trimesh / open3d / imageio / pillow / h5py`等当前导入链 | 包括Base_Task、camera/save_file及Robot的真实导入链，不只顶层SAPIEN |
| RLinf | 当前源码＋pyproject基础依赖；Ray[default]要求≥2.47、Hydra约束沿用仓库 | 生成准确解析锁定版本；`pip check`、版本freeze及安装来源 |
| OpenPI运行时 | `rlinf-openpi==0.1.1`及其Transformers替换；tokenizers≥0.21,<0.22 | 保存wheel SHA、包文件SHA、实际tokenizer与RealMan DataConfig构造 |
| JAX辅助 | JAX/JAXlib 0.5.3、Orbax0.11.13、tyro1.0.13；PPO环境优先CPU JAX | 禁止辅助JAX预占训练GPU；需要的模块实际可导入 |
| 转换/参考 | CPU JAX/JAXlib0.5.3、Orbax0.11.13、NumPy1.26.4；转换器另需Torch/safetensors等 | CPU恢复及转换；JAX参考需要原OpenPI/Flax0.10.2等原版本 |

PyTorch组合是[官方列出的cu126组合](https://pytorch.org/get-started/previous-versions/)；其他跨RLinf/RoboTwin组合是候选约束，尚不是已经通过的lock。SciPy可从1.15.3候选开始解析。TorchCodec若实际需要，当前installer对Torch2.7派生区间为`>=0.3,<0.6`，仍须选兼容wheel并锁定，不套用最新版本。

后续顺序：确认主机→建立两个新环境→解析带约束的基础依赖→安装Torch→仿真ABI包→RLinf/OpenPI运行时→仅在新环境应用所需Transformers替换→真实导入与组件检查→冻结环境清单。转换器本身不要求安装原OpenPI；只有原JAX参考策略需要[原OpenPI commit215abfb](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)。参考运行与Torch顺序进行，避免两大模型同卡共驻；可使用原独立JAX环境或新参考环境，不能污染原训练环境。

不直接照搬整个RoboTwin旧requirements（会引入旧Torch/SciPy），也不把RLinf一键robotwin安装器当作本任务最小安装清单：它默认SAPIEN3.0.1，在NVIDIA路径还会构建PyTorch3D/CuRobo/Warp并修补新环境库；`--no-root`只跳过系统安装，不能跳过这些构建。其真实参数包括`--venv/--python/--torch/--no-root/--no-flash-attn/--install-rlinf`，没有已核对的skip-CuRobo开关。RM所需栈应按真实导入链裁剪；未通过实际attention导入/前后向前，不宣称FlashAttention一定可省略。

### 3.2 当前必须先补的兼容入口

| 前置工作 | 原因 | 通过条件 |
| --- | --- | --- |
| RoboTwin可选规划依赖隔离 | `native_robot→Robot→CuroboPlanner`仍有无条件导入；缺CuRobo时类未定义，即使RM不规划也可能导入失败 | 在独立代码副本修正可选导入；真正规划时报明确缺依赖；已安装CuRobo保留原行为。若不修此入口，就准备完整CuRobo依赖栈 |
| Orbax/NNX转换兼容 | 当前转换器只解外层params，未处理terminal value；metadata根类型也有兼容风险 | 下节恢复与转换测试通过，再转换全模型 |
| 新双卡训练/独立eval预设 | 原接口预设train2/eval2共卡，评测默认固定ID | 新预设完整替换placement；single env、遍历seed、统一时限；保存resolved config |
| 独立eval噪声控制 | env seed只控制场景；当前HF预测链未按评测回合注入独立flow初始noise | 在代码副本给eval按(reset ID,query index,repeat)注入/记录固定noise或generator，并记录噪声SHA；训练Flow-SDE随机性另记 |
| 诊断记录入口 | 日志结束不等于有效更新、同步或显存通过 | 增加/复用可核验的更新计数、梯度/参数指纹、同步版本、峰值和resume状态记录 |

这些尚未实现。历史79项组件通过及3项OpenPI跳过见[[RLinf-VLA强化学习上手-RoboTwin与LIBERO]]，属于2026-10-12证据；新环境应重新跑对应测试，补齐跳过的OpenPI测试及真实tokenizer/数据注册/原生spawn检查，不能只引用历史总数。

## 4. π0.5权重转换与一致性

### 4.1 固定输入

优先用已有80000 checkpoint，避免同时换起点。当前本地来源：

```text
/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/
pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/checkpoints/
pi05_realman_rgb50_all50_full_v1/
full50_100k_b64_fallback32_save20k_20261008_v4_bs64_formal100k/80000
```

以上换行仅为阅读，实际是一个连续绝对路径。来源有`params/`、`train_state/`、`assets/realman_rgb50_full50/norm_stats.json`。转换仅取params和assets，在新输出目录初始化PPO；不继承JAX optimizer/train_state。统计文件当前SHA256：

```text
34cbdd09c831e04549d6c4a4e2eb321fb975bdea13bb816ec8aa3a2332d1ccc5
```

### 4.2 转换器前置修正

当前`rlinf/utils/ckpt_convertor/openpi/jax_to_openpi.py::_load_jax_params`对Orbax metadata直接构造restore_args。本checkpoint记录有51个NNX叶，路径以`value`结尾，后续SigLIP转换却要求数组。

后续实现应：metadata取普通树（兼容`.tree`）→用同一树显式构造item与NumPy restore_args→CPU恢复→按需解外层params→**只剥叶路径最后的NNX value**→保持FP32数组。`attention/value/kernel/value`只能变为`attention/value/kernel`，不能删除attention语义节点。增加纯树、NNX包装、attention反例及TreeMetadata根的回归验收。

[Orbax0.11.13源码](https://github.com/google/orbax/tree/v0.11.13/checkpoint/orbax/checkpoint/_src)显示TreeMetadata参与PyTree结构，当前metadata/restore_args根类型存在静态兼容风险；本轮未restore，未声称已复现该异常。参考原OpenPI的恢复和terminal value处理，但不直接把JAX训练环境包替换进PPO环境。

### 4.3 后续转换命令模板

**下列只在前置补丁、依赖及小树恢复验收后执行。** `$CKPT`必须设为后续确认的源目录，`$OUT`设为新的models目录，`$CONVERT_PY`为新转换环境Python，PYTHONPATH指向代码副本；服务器路径不能照抄本机`/home/admin123`。

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
"$CONVERT_PY" -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode jax_to_openpi \
  --input-model "$CKPT" \
  --input-norm-stats "$CKPT/assets/realman_rgb50_full50/norm_stats.json" \
  --output-model "$OUT" \
  --output-norm-stats "$OUT/realman_rgb50_full50/norm_stats.json" \
  --action-dim 32 --action-horizon 16 --max-token-len 200 \
  --paligemma-variant gemma_2b --action-expert-variant gemma_300m
```

输出FP32单文件`model.safetensors`（按参数量估算约13.4GB）、config.json与原样norm stats。jax模式没有`--dtype bf16`参数；运行时模型factory不读取该config.json，需要预设显式定义π0.5/D32/H16/三图和物理D24/K1。

### 4.4 转换验收

- 恢复51个源叶的逻辑shape、dtype和有限值；转置/拼接/重命名后逐项回查数值，列清全部目标键，没有被静默忽略的权重。统计SHA与源文件相等。
- eval模型骨干零缺失、零unexpected；PPO新value head随机初始化是预期，先同步Actor/Rollout，再采样。列出允许缺失集合，不能笼统忽略missing keys。
- 固定三RGB/state24/指令/统计/token与同一显式`[B,16,32]`初始噪声，对比原JAX与转换模型的输入变换、flow速度场、10步ODE完整`16×24`输出。RGB/token逐项相同；浮点预处理误差≤1e−5可作为候选工程门槛。臂误差用rad、手误差用SDK分别报告max/P95。
- 跨JAX/Torch BF16输出不要求bitwise相等；在看结果前依据控制误差预算固定独立容差文件。当前未做本体容差定标，不能把临时数字声称为实测精度，也不能事后放宽门槛掩盖转换问题。
- 本节比的是ODE/确定固定噪声的转换一致性；PPO训练Flow-SDE noise0.3另做概率项验收。

## 5. SFT基线：先保证RL比较公平

复制当前`evaluations/robotwin/robotwin_realman_adjust_bottle_openpi_pi05_eval.yaml`形成新独立eval预设；使用`evaluations/eval_embodied_agent.py`入口，不通过训练入口创建Actor。

| 字段 | SFT和更新后RL共同取值 |
| --- | --- |
| placement | 完整替换为`env:1, rollout:1`，同一轮只启动一个评测场景 |
| model | π0.5、H16/K1、三图、10步ODE；`openpi.task=eval`、`add_value_head=false` |
| env | `total_num_envs=1 / auto_reset=false / ignore_terminations=false` |
| 回合 | `max_episode_steps=400`、`max_steps_per_rollout_epoch=400`、task budget400 |
| seed | 固定base seed，`rollout_epoch=8`、`use_fixed_reset_state_ids=false`；记录实际八个reset ID |
| 统计/终止 | 原Full50 stats；native判据、成功早停、失败及时间限制一致 |

现有eval池为2010–2017，训练池1010–1017；执行前固定种子池、检查初态覆盖与重复。**B1＋固定reset ID true＋8轮会反复测同一个初态**，所以要改为可复现遍历。不同ID不保证布局OOD或专家成功；应保存模型ID、左右臂、瓶姿、对象/机器人初态和场景SHA。

同场景不自动等于同模型noise。当前Pi0Eval支持显式noise/rng，但HF采样入口默认走torch.randn，env seed没有逐query控制该噪声。后续eval需按(reset ID,query index,repeat)锁定初始noise或独立generator，保存噪声SHA，才可称SFT/RL同噪声配对；该hook尚未实现。ODE表示确定去噪积分，不表示随机初始noise自动消失。

顺序：一条golden场景验证真实策略闭环→8个冻结标准初态快速SFT基线→8个独立轻扰动初态基线（额外冻结初态清单/配置）。轻扰动候选为固定初态XY各±1cm、yaw±5°，限制在可达、无初始碰撞范围，并保存实际坐标；这是待确认的难度，不是已通过的任务分布。轻扰动只改变reset对象位置/姿态，不修改机器人合同；当前adapter没有已验收的执行中动态扰动入口。若标准集并非精确原训练初态，应如实称“固定名义分布”，不自动标为训练原场景复现。

每组记录native成功分子/分母、明确失败/超时、成功前target数及实际物理时间、对象角度/持握视频、动作范围/有限值、推理耗时、显存与进程退出。八回合用于快速验收，不能代替正式SR统计或独立训练seed复现。

若在轻扰动分布中始终没有真实正回报，先诊断转换、闭环和难度；不要直接扩大PPO。若400步不足覆盖SFT事件，先修订共同物理预算并重做基线。按成功/失败出现的target长度确定第7节信号采样窗。

## 6. 双卡T16计算闭环预设

后续从现有`examples/embodiment/config/robotwin_realman_adjust_bottle_ppo_openpi_pi05.yaml`复制新预设。下面是**待创建的差异草案**，其余继承现有RM模型、任务、优化器和PPO设置；placement必须完整替换旧组合键，不能与`actor, env, rollout:0`合并。

```yaml
cluster:
  component_placement:
    actor: 0
    rollout: 1
    env: 1
runner:
  max_epochs: 2
  max_steps: 2
  val_check_interval: -1
  save_interval: 1
  weight_sync_interval: 1
  overlap_env_bootstrap: false
  use_training_pipeline: false
  resume_dir: null
  ckpt_path: null
env:
  enable_offload: false
  train:
    total_num_envs: 1
    rollout_epoch: 1
    group_size: 1
    max_episode_steps: 400
    max_steps_per_rollout_epoch: 16
    auto_reset: false
    ignore_terminations: false
    enable_offload: false
    task_config:
      max_episode_steps: 400
      renderer_device: cuda:0
actor:
  micro_batch_size: 1
  global_batch_size: 16
  enable_offload: false
  model:
    precision: null
    is_lora: false
    add_value_head: true
    action_dim: 24
    num_action_chunks: 1
    num_steps: 10
    openpi:
      task: rl
      model_action_dim: 32
      action_horizon: 16
      num_images_in_input: 3
      train_expert_only: true
      noise_method: flow_sde
      noise_level: 0.3
      torch_compile: false
  fsdp_config:
    strategy: fsdp
    sharding_strategy: no_shard
    use_orig_params: true
    gradient_checkpointing: false
    mixed_precision:
      param_dtype: null
      reduce_dtype: null
      buffer_dtype: null
    amp_autocast:
      enabled: false
    grad_scaler:
      enabled: false
rollout:
  pipeline_stage_num: 1
  enable_offload: false
  enable_cuda_graph: false
algorithm:
  update_epoch: 1
  recompute_logprobs: true
  filter_rewards: false
```

保留Actor LR5e−6、value LR1e−4、gamma0.99、GAE lambda0.95、clip0.2及原梯度裁剪，先不加KL模型、LoRA、残差或新奖励。Rollout模型设置继续引用Actor配置。新增字段若通过Hydra CLI传递，需要按其struct规则用`+`；实际以新预设的resolved配置与validator结果为准，本轮没有compose该草案。

`precision:null`沿用现有选择性精度：Gemma/SigLIP主体BF16，norm、视觉stem、动作/时间/value头FP32；不要全模型`.to(bf16)`或把FSDP三项精度直接改成BF16。当前`use_fp32_master_params=true`只允许LoRA，不能在`is_lora=false`全动作专家模式直接启用。必须检查专家有效梯度和实际权重变化；若低LR/BF16量化导致只有value/投影变化，计算链不能标为动作专家RL通过。后续候选是显式将可训练专家保留FP32，或单独适配并验证master optimizer；这是新增工作，不能为绕过校验改称LoRA方案。

**步数必须这样理解：**

- B1、K1、T16、rollout_epoch1、update_epoch1、Actor world1、global16/micro1：每轮16个样本，梯度累积16，预期每轮一次optimizer更新；两轮共两次。若global2则每轮八次，不能叫“两次更新”。
- `runner.max_steps=2`是两轮采样—更新循环，`max_epochs`同时设2。优化器初始化warmup不是数据更新；manager计数在nonfinite判断前增加，必须核对未跳步、Adam state step及有效参数delta。
- T16末端多一次value前向取得V16，不能因此执行第17个target。正确定义终态/截断bootstrap及valid mask。
- 每轮会reset；T16最多推进约0.96秒，不能把两轮累加成一个24秒任务。全零奖励的T16结果只作技术门槛。
- `renderer_device=cuda:0`指env worker可见设备的局部编号；必须用UUID/PCI/PID证实renderer落在逻辑GPU1，不能盲改cuda:1。

## 7. 实际奖励信号与PPO更新验收

计算链通过后，从固定SFT重新开始独立信号试验。取T覆盖第5节观察到的奖励事件，例如128或400；同时设global_batch_size=T、micro1、update_epoch1，继续每轮一次更新。T400的采样动作数是T16的25倍，实际耗时还受加载、同步和保存影响，不能直接宣称总时间25倍。

至少出现一个来自当前策略真实交互的正/负事件，记录对应视频、终态、reward与GAE。成功信号稀疏时优先使用已确认SFT能成功的轻扰动训练分布；不能伪造奖励、塞专家轨迹冒充on-policy样本、把reset/teleport记为策略成功，或临时放松success门槛。

| 必须记录 | 验收判断 |
| --- | --- |
| observation→action→physics trace | 三图/state24对应同一步；物理target24、时钟、早停和mask正确 |
| old/new logprob与去噪链 | 同一保存去噪步骤可重算；更新前相同策略的ratio接近1，重算偏差有界且无NaN/Inf |
| value、returns、GAE、loss/grad norm | 有限；终态不错误bootstrap；真实事件能追溯到优势 |
| trainable名单、梯度、参数delta | 冻结VLM梯度为空且参数指纹不变；动作专家与value有有效更新，区分数据梯度与单纯weight decay |
| optimizer实际更新 | 两次未跳过的正式step；warmup另列；保存前后Adam state/scheduler可核对 |
| Actor/Rollout权重版本 | 第2轮确实使用第1次更新；最终第2次更新的采样同步在续跑阶段确认 |

已有runner在每轮开始同步：`sync→collect→GAE→update→save`。关闭训练内eval后，最后一次update不会自动追加同步；不能只凭“global_step2保存”声称Rollout已经使用第2次更新。第8节恢复第3轮开头完成该确认；若要单独末尾同步，需要新增明确入口和验收。

## 8. 保存、重载、续跑与同协议评测

正常输出应包含：

```text
<log_path>/<experiment_name>/checkpoints/global_step_2/
  actor/dcp_checkpoint/                 # 模型＋optimizer＋scheduler＋Actor RNG
  actor/model_state_dict/full_weights.pt
```

1. 校验文件完整可读、参数/optimizer/scheduler/global step正确，记录SHA与日志。
2. 退出本任务Worker，核对无悬挂spawn场景/渲染进程；不停止主机上其他Ray或训练任务。
3. 新进程设置`runner.resume_dir=<global_step_2>`、**max_steps/max_epochs都改3**，T16/global16保持，则再采样16步并更新一次。核对恢复训练状态、Actor/Rollout同步版本和第三次实际step。
4. 当前checkpoint恢复Actor的CPU/NumPy/Python/CUDA RNG；不恢复env场景/seed池进度或Rollout RNG，因此验收是训练状态续跑，不是逐数值整条轨迹重启。
5. 独立eval对原SFT和`global_step_2`使用第5节同一场景/噪声/终止协议。新模型以`rollout.model.model_path=<global_step_2目录>`加载其full_weights，`runner.ckpt_path=null`；不要用含value head的PPO `.pt`走eval strict load。norm stats仍指定原文件。
6. 报告原SFT、两次更新后、可选续跑后三个版本的配对结果。允许SR不变/下降，仍可证明工程闭环；不能把8条样例的波动叫作有效RL增益。

## 9. 显存测试与失败处理

[RLinf官方π0文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html)提供分离placement及offload等容量调整方式。本任务先不开offload，train env1、训练内eval关闭、pipeline1、micro1、H16/K1，避免Actor/采样双模型和多场景共卡；两卡足够与否由实际峰值判定。

| 阶段 | GPU0：Actor | GPU1：Rollout＋env |
| --- | --- | --- |
| 初始化 | 模型、优化器状态warmup、FSDP缓冲 | 模型、SAPIEN三相机、Vulkan/Optix |
| 采样/末端value | 重算旧概率等实际Actor工作 | 当前策略推理/缓存、图像/动作/去噪链 |
| 反向＋step | 激活、梯度、Adam状态、更新临时张量 | 采样模型/渲染常驻占用 |
| 权重同步 | 发送/打包缓冲 | 接收/替换缓冲 |
| 保存/恢复 | 完整state gather/序列化、CPU RAM | 对应Worker占用及同步峰值 |

每阶段在对应Worker记录PyTorch allocated/reserved峰值，同时按200–500ms采样整卡NVML显存与全部PID；SAPIEN/Vulkan占用不在PyTorch allocator里。报告UUID、PCI、逻辑编号、Actor/Rollout/render PID，验证落卡而不是只看CUDA_VISIBLE_DEVICES设置。补记CPU RSS/转换峰值、单轮墙钟时间、target查询吞吐、实际仿真秒数及同步/保存耗时。

建议工程通过线：所有阶段无OOM、无非有限值、两次有效更新/同步/保存/恢复完成，**每卡峰值后至少留8GiB余量**（约40GiB以内，按设备实报换算）。8GiB是本方案的余量目标，不是已测出的最低需求；初始化加载成功和低空闲显存不代表PPO峰值通过。

若失败，按阶段定位：先确认未创建多余eval场景、GPU placement及采样缓存→缩短技术T并同步调整global保证计数/整除→保持micro1/环境1→再独立验证Actor/Rollout阶段CPU offload。RM环境offload保持false。奖励窗口不能为了省显存缩到永远看不到真实事件；降低并行或调整采样/批次拆分需重新验计数。三相机/SDK24/60ms与stats不作为降配手段。

## 10. 后续启动参数与入口模板

执行时确认路径后设置任务环境变量：`EMBODIED_PATH`、`REPO_PATH`、`PYTHONPATH`（RLinf＋RoboTwin代码副本）、`REALMAN_ROBOTWIN_ASSET_ROOT`、`REALMAN_PI05_MODEL_PATH`、`REALMAN_NORM_STATS_PATH`、`REALMAN_RUNTIME_ROOT`及两张实际分配GPU。辅助JAX设置`JAX_PLATFORMS=cpu`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`；缓存/tokenizer下载使用独立任务目录，保留模型来源/hash。不要改写HOME/CODEX_HOME，不复制旧环境二进制扩展。

拟定训练预设名`robotwin_realman_adjust_bottle_ppo_openpi_pi05_two_a6000_min`；eval预设名`robotwin_realman_adjust_bottle_openpi_pi05_two_a6000_eval`。**两者尚未创建**。下面命令只展示前述新环境/新预设通过验收后的入口；变量由后续核实的路径设置。

```bash
"$PPO_PY" "$RLINF_ROOT/examples/embodiment/train_embodied_agent.py" \
  --config-path "$RLINF_ROOT/examples/embodiment/config" \
  --config-name robotwin_realman_adjust_bottle_ppo_openpi_pi05_two_a6000_min \
  runner.logger.log_path="$RUN_ROOT/logs" \
  runner.logger.experiment_name=minppo_t16
```

任意Hydra覆写直接走Python训练入口；当前`run_embodiment.sh`第二参数是ROBOT_PLATFORM，只转发有限环境参数，不能把上述覆写任意接到末尾。

```bash
"$PPO_PY" "$RLINF_ROOT/evaluations/eval_embodied_agent.py" \
  --config-path "$RLINF_ROOT/evaluations/robotwin" \
  --config-name robotwin_realman_adjust_bottle_openpi_pi05_two_a6000_eval \
  runner.logger.log_path="$RUN_ROOT/logs/sft_eval" \
  rollout.model.model_path="$OUT" \
  runner.ckpt_path=null
```

RL评测仅替换model_path到保存目录并使用独立日志位置。Wrapper准确位置为`evaluations/run_eval.sh`，支持`robotwin <config_name> [Hydra overrides]`；不可与`examples/embodiment`混淆。

## 11. 执行顺序与交付证据

| 顺序 | 工作 | 进入下一阶段的门槛 | 计划报告 |
| --- | --- | --- | --- |
| 0 | 核实服务器/两卡/目录，完整代码快照 | 不碰原资产/入口，UUID和来源清楚 | preflight.json、source_manifest.json |
| 1 | 新环境依赖与必要兼容入口 | pip check、真实导入/组件/三图spawn通过 | environment_lock、import_and_contract_report |
| 2 | CPU恢复与80000转换 | tree/键/形状/数值/stats SHA合格 | conversion_report.json |
| 3 | JAX/Torch golden比较 | 预处理相等、模型/SDK输出满足预先固定容差 | parity_report.json |
| 4 | 同协议SFT标准/轻扰动基线 | 正确初态遍历；真实策略/终态/事件长度已测 | sft_baseline.json、episode视频 |
| 5 | T16两轮＋显存 | 两次真实更新、VLM冻结、专家有效变化、下一轮同步 | ppo_wiring_report、vram_profile.csv |
| 6 | 保存/重载/第3轮续跑/独立eval | 模型及训练状态恢复；最新同步可核验 | checkpoint_resume_report、paired_eval.json |
| 7 | 奖励信号试验 | 当前策略真实事件→非零优势→专家有效更新 | ppo_signal_report、事件视频 |

所有报告应带时间、主机、GPU UUID、代码快照、checkpoint/stats哈希、resolved配置、实际执行Python/依赖版本、退出状态与未通过项。未执行阶段保持“待验收”，不能用配置存在、日志打印或旧测试替代证据。

完成上述阶段后，才能称“RM65-B的π0.5在RLinf完成最小PPO闭环验证”；进一步扩大并行、正式PPO曲线、多训练seed/留出布局和真实任务成功评测，另立实验记录。

## 12. 本次静态核查来源与范围

| 结论 | 当前源码/历史入口 | 核查性质 |
| --- | --- | --- |
| 控制合同、adapter、已有组件证据 | [[RLinf-VLA强化学习上手-RoboTwin与LIBERO]]及2026-10-12接口证据 | 合同本轮静态复核；实际测试为历史证据 |
| 训练/采样/optimizer计数 | RLinf `rlinf/runners/embodied_runner.py`、`workers/actor/embodied_fsdp_actor_worker.py`、`workers/rollout/hf/huggingface_worker.py` | 2026-10-13静态核查 |
| 精度、master限制、状态恢复 | `rlinf/config.py`、`hybrid_engines/fsdp/{optim.py,fsdp_model_manager.py,strategy/checkpoint.py}`、`utils/utils.py` | 2026-10-13静态核查；未跑optimizer或DCP |
| 转换/权重路径 | `utils/ckpt_convertor/openpi/{convert.py,jax_to_openpi.py}`、`models/embodiment/openpi/checkpoint.py`及源checkpoint元数据 | 2026-10-13静态核查与文件读取；未restore |
| CuRobo导入阻断 | RoboTwin `integrations/realman_rm65b_gen4/native_robot.py`导入链、`envs/robot/{robot.py,planner.py}` | 2026-10-13静态核查；未运行导入 |
| 部署/评测 | `requirements/install.sh`、pyproject、OpenPI requirements、RM训练和`evaluations/robotwin`预设、`openpi/tasks/eval.py`及HF worker采样入口 | 2026-10-13静态核查；未compose/安装/运行 |

本方案是后续执行的完整门槛清单，依赖解析、转换容差、两卡实际峰值、真实奖励出现及PPO学习收益均待执行验收。

## 13. RTX A6000渲染能力与166的3090备选（2026-10-13）

用户提出“A6000不能渲染，166的3090是否足够”。需区分显卡硬件和服务器运行状态：**RTX A6000具备RT核心、图形渲染与Vulkan能力**，见[NVIDIA产品规格](https://www.nvidia.com/en-gb/products/workstations/rtx-a6000/)与[官方中文数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-a4-nvidia-1454980-r9-web-zhCN.pdf)。[SAPIEN官方安装说明](https://sapien.ucsd.edu/docs/latest/tutorial/basic/installation.html)也提供无显示器服务器的offscreen渲染验收。因此不能由“服务器没有显示器/当前渲染失败”推出“A6000硬件不支持渲染”。本轮没有161错误日志或实际render检查，不能确定其具体故障。

若161实际失败，后续需检查Vulkan ICD/驱动图形库、实际渲染设备、容器图形能力和SAPIEN/OptiX版本；若使用容器，[NVIDIA官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)指出默认compute/utility不包含graphics，CUDA可用也不代表图形渲染链可用。以上是排查对象，不是已确定原因。

[[服务器-172.17.27.166]]记录2026-09-19核验8×RTX3090、每卡24GiB、503GiB主存；[[服务器172.17.27.166-robotwin]]保留后续仿真组件/数据运行证据。本轮只读这些历史记录，未登录166核对空闲卡、驱动或当前环境，不能将其视为今日可用资源。当前RM65-B Full50/RLinf代码快照也需要在新任务目录完整移交，不自动等于166原工程已匹配。

| 3090分配 | 对本方案的判断 |
| --- | --- |
| 一卡Actor＋Rollout＋仿真 | 不作为首选：两个完整模型、优化器与渲染同占24GiB，余量紧 |
| 两卡：Actor0，Rollout＋env1 | 合理的最小容量验证候选；冻结VLM、全动作专家＋value、B1/micro1/T16/global16、H16/K1，关闭训练内eval。尚未证明24GiB能覆盖全部峰值 |
| 三卡：Actor0，Rollout1，env2 | 两卡中采样/渲染卡拥挤时的备选，独立分走SAPIEN渲染；当前CPU动作传输路径允许分离。Actor仍只有24GiB |

当前OpenPI限制`no_shard`；增加Actor副本或分开env不会合并Actor显存。Actor若仍OOM，需单独验证CPU offload等路径；如果为了有效专家更新将其改为FP32或适配master optimizer，重新测峰值，不能继续引用原BF16预算。原24维/三相机/60ms合同不改。

3090试验建议每卡保留3–4GiB峰值余量（约20GiB以内，按实际设备报告换算），这是工程目标，不是实测下限；不能直接套用双A6000的8GiB余量线。至少覆盖原生三相机render、第一轮反向/Adam step、权重同步、checkpoint保存/恢复；SAPIEN显存仍须看整卡NVML。技术T16通过后，再单独测包含真实奖励事件的较长T。

166历史驱动535.247.01、旧环境Torch2.4.1+cu121来自旧快照。CUDA12.x存在[最低驱动与小版本兼容机制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)，不能仅由nvidia-smi的CUDA字段断言cu126绝对不能运行；PTX/JIT及新特性仍可能要求较新驱动。因此新wheel/算子、Vulkan/OptiX和工具链须分别验收，不自动修改共享驱动或CUDA。

结论：166的多张3090可作为最小PPO验证平台，建议先两卡容量验证、需要时三卡分离仿真；两卡足够与否尚待实测。A6000原方案不因硬件渲染能力而失效，若161环境暂不可用，迁移166属于运行条件选择。本轮只修订方案判断，没有远端操作、依赖安装、配置修改或测试。

### 13.1 谁更快：理论算力、带宽与实际PPO吞吐

用户进一步关心速度。**不能直接断言RTX A6000的π0.5 PPO端到端速度比3090快。** 两者均为Ampere GA102，理论计算峰值接近，而3090显存带宽更高。标称数据来自[NVIDIA GA102架构白皮书](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)与[A6000数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-a4-nvidia-1454980-r9-web-zhCN.pdf)：

| 理论指标 | RTX A6000 | RTX3090公版 | 指标含义 |
| --- | --- | --- | --- |
| FP32峰值 | 38.7 TFLOPS | 35.6 TFLOPS | A6000标称高约9%，不等于BF16/PPO实测快9% |
| 显存带宽 | 768 GB/s | 936 GB/s | 3090标称高约22%，不等于模型推理快22% |
| 显存容量 | 48GB | 24GB | A6000可容纳更大并行/缓存，或避免显存不足造成的offload |

工程推测：同batch/并行/精度且两者都装得下时，不期待仅换A6000获得数量级提速，3090在访存受限部分可能更快；A6000若允许增加有效并行或避免CPU offload，整体吞吐可能更好。SAPIEN渲染同样不能只凭核心数判定赢家；环境推进、CPU、GPU散热/功耗、驱动与同步也影响最终时间。本轮没有两台服务器同配置性能测试，不能给训练/渲染快慢百分比或实测排名。

对当前最小闭环，先选实际环境可用、显存能覆盖峰值的两卡；比较时固定合同、T/global/micro、精度、初态和噪声，报告采样target/s、optimizer step耗时及完整一轮墙钟时间，且分别列GPU数与是否offload。正式扩大并行后再比较单位时间有效交互和更新吞吐，不能用单项TFLOPS替代。

### 13.2 本机5090与A6000/3090的标称参数

用户进一步问本机5090相比上述两卡的参数。本轮查询工具未能通过nvidia-smi读取GPU，故未重新确认本机具体型号、容量或驱动；该失败不能单独诊断实体主机驱动损坏。下表按**标准桌面RTX5090公版32GB**，不混用5090D、5090D V2或Laptop版参数。历史本机5090/cu128与渲染组件证据不等于本轮重新验收。

| 标称参数 | RTX5090 | RTX A6000 | RTX3090 |
| --- | --- | --- | --- |
| 架构 | Blackwell | Ampere | Ampere |
| CUDA核心 | 21760 | 10752 | 10496 |
| 显存 | 32GB GDDR7 | 48GB GDDR6 ECC | 24GB GDDR6X |
| 显存带宽 | 1792 GB/s | 768 GB/s | 936 GB/s |
| 理论FP32峰值 | 104.8 TFLOPS | 38.7 TFLOPS | 35.6 TFLOPS |
| RT核心 | 170／第4代 | 84／第2代 | 82／第2代 |
| Tensor核心 | 680／第5代 | 336／第3代 | 328／第3代 |
| 标称整卡功率 | 575W | 300W | 350W |

来源：[NVIDIA Blackwell白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)、[NVIDIA RTX5090规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)、[NVIDIA GA102白皮书](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)。不同厂牌超频/功率设定可不同，本表不是本机实际频率/功耗。

标称FP32比值为5090/A6000约2.71、5090/3090约2.94；带宽比值约2.33和1.91。GPU计算与渲染硬件上5090具备明显优势，A6000保留最大显存。上述是规格计算，不是π0.5/PPO或SAPIEN实测加速比；Tensor AI TOPS还涉及精度与稀疏定义，不能拿5090的FP4宣传值直接除以Ampere的BF16/FP16数值。

若Blackwell运行栈支持、单卡显存可容纳完整峰值，本机5090是值得优先尝试的调试/容量验证设备；一张卡承载Actor＋Rollout＋渲染仍须实测。不能由单卡算力推断一张5090必然快过两张分工Ampere卡。保持当前24维/三相机/60ms合同与原控制入口，本轮未安装、转换、启动模型/渲染或性能测试。

## 后续本机执行证据（2026-10-13）

用户随后明确改为本机5090，授权独立环境／转换与最小PPO。实际结果见[[2026-10-13-RLinf-RM65B-本机5090最小PPO闭环验证]]和[[本机-RLinf-RM65B-5090独立环境]]：四次真实更新、保存恢复与step3配对独立评测通过；成功bucket峰值约29.3GiB，原patch曾同步OOM。全部训练奖励0，学习信号门未通过；原JAX／Torch完整采样等价也未验。本页仍是双A6000规划，不改成A6000执行记录；额外旧概率重算在当前OpenPI配置被拒绝，实际采用原Flow-SDE概率路径。
