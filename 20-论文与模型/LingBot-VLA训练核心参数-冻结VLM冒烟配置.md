---
title: LingBot-VLA训练核心参数-冻结VLM冒烟配置
date: 2026-09-20
tags: [LingBot, 训练配置, 冒烟测试]
status: 配置与源码核对，不代表运行成功
---

# LingBot-VLA训练核心参数：冻结VLM冒烟配置

对象：本地 `configs/vla/robotwin/robotwin_local_frozen_smoke.yaml`。仓库位于 `/home/admin123/liminghe/vla-benchmark/lingbot-vla-v2`，HEAD `ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8`；本地测试配置及代码可能含工作区修改，应以实际文件为准。

这是一份读取已有6B权重、冻结VLM、关闭深度与视频教师、训练动作分支两步的测试配置。用于检查数据、前向、反向、优化器和保存流程，不用于判断收敛或任务成功率，也不等同于官方完整方法复现。

## 最关键的八组参数

| 参数 | 当前值 | 实际意义 |
| --- | --- | --- |
| `model_path` | 本地6B checkpoint目录 | 读取已有权重；并非从随机初始化重新预训练 |
| `train_expert_only` / `freeze_vision_encoder` / `train_state_proj` | true / true / true | 冻结整个VLM及其视觉部分；动作专家、相关可训练投影保留训练，状态投影也更新 |
| `align_params` / `use_future_image` | {} / false | 不构建这条配置下的Depth/Video对齐监督，不加载未来图像用于教师目标；这是简化版本 |
| `loss_type` | L1_fm | 对flow速度场进行L1监督，不是直接对最终关节角做普通L1回归；仍保留MoE辅助损失 |
| `action_dim` / `max_action_dim` / `max_state_dim` | 55 / 55 / 55 | 网络统一表示容量；本次机器人映射的有效执行量是12个关节角＋2个夹爪位置 |
| `micro_batch_size` / `global_batch_size` / `max_steps` | 1 / 1 / 2 | 每rank每微批1样本、全局每更新1样本、仅做2次训练更新；按单rank测试理解，不是跑完整数据集2遍 |
| `optimizer` / `lr` / `lr_min` | adamw / 1e-5 / 1e-5 | 使用AdamW；上下限相同且warmup为0，这两步中没有实质cosine降学习率 |
| `enable_gradient_checkpointing` / `use_compile` | true / false | 用反向时重算换取激活显存节省；不启用torch.compile，便于排查且避免两步测试被编译开销主导 |

“只训练动作专家”不表示只训练MoE中固定的几个专家，也不是LoRA；不应认为只有最后一个输出层更新。冻结权重仍需前向计算和显存存放，显存不会归零。

## 数据入口与动作约定

- `data_name: multi`：train_path指向数据清单，各行关联机器人配置和数据集。
- `robot_config_root`：决定如何从原始state/action与相机字段映射到统一表示。此次clean50映射使用双六轴关节与夹爪，动作 `subtract_state:false`，即绝对位置。
- `joints` 中保留 `end.position:14` 是布局容量，不能据此认定本次有效监督末端位姿；要看机器人YAML是否实际映射。
- `norm_type: bounds_99_woclip`：按1%/99%分位数做线性归一化，不截断；统计文件应对应当前数据。相关robot YAML也指向clean50统计文件。
- 三相机：头部、左腕、右腕RGB；`tokenizer_max_length:72` 是含文本模板的token预算，超长文本截断。
- `num_workers:0` 在主进程读数据，`pin_memory:false` 不固定CPU内存，适合缩小调试复杂度，不是吞吐最优配置。
- YAML未显式写chunk_size，训练参数类默认50；max_steps=2与动作块长度50是不同时间尺度。

## MoE参数：结构仍保留，计算稀疏

`use_moe:true`；层号0—35表示36层均安装MoE FFN；每层32个路由专家，每token激活4个，并有共享专家。路由/共享中间宽度分别512和704。`moe_implementation:fused` 选择融合实现，不改变“稀疏路由”的语义。

`use_shared_expert_gate:false` 不代表关闭共享专家，而是不使用额外共享专家门控。`use_moe_expert_lr:false` 关闭专门给路由专家的学习率缩放，不冻结它们。

`bias_update_speed:0` 不进行这一路由校正bias的动态更新；负载平衡主要由启用的sequence-wise辅助项承担。`sequence_wise_loss_coeff:1e-3` 和 `router_z_loss_coeff:1e-4` 分别约束专家分配与路由logit尺度。因此“action-only”在这里表示没有Depth/Video监督，不表示损失里绝对只有动作项。

## 精度：不能只看一个布尔值

当前源码的模型构建调用为：

```python
torch_dtype = 'float32' if enable_mixed_precision else 'bfloat16'
```

所以本配置 `enable_mixed_precision:false` 会选择BF16初始化/加载路径，而不是全模型FP32。`enable_fp32:false` 不请求FSDP整体FP32策略；不同模块还可能有独立精度处理。

`action_fp32:true` 在状态/动作输入、动作投影及输出等数值路径显式使用FP32。它不能简单等同于“全部动作专家参数和所有算子均FP32”。实际dtype和优化器状态要以运行时张量核查为准。

`adanorm_time:true` 使动作专家使用flow时间条件化的自适应归一化；这里的时间是生成过程的噪声/积分时间，不是机器人采样时间戳。

## FSDP与激活内存

`data_parallel_mode:fsdp2`、`module_fsdp_enable:true`、`vlm_fsdp:true` 选择分布式包装路径；不会改变VLM已经冻结的事实。

当前FSDP2代码将 `enable_full_shard:false` 传给 `reshard_after_forward`，重点是不在前向后立刻重新分片。不能直接将它解释成“完全不分片”；也不要把FSDP1分支的含义套到FSDP2。单rank没有多卡分片收益，配置也不会自动启动8张卡。

`enable_reentrant:false` 配合checkpoint使用非reentrant实现。`precompute_grid_thw:true` 减少重复视觉网格元信息计算；`attention_implementation:flex_cached` 选择注意力实现，不代表跨episode长期记忆。

## post_training不是是否微调的总开关

尽管名字叫post_training，设为false并不会取消model_path中的权重加载，也不会自动把冻结开关改回去。

当前源码中它参与checkpoint键映射，并影响优化器默认参数分组：存在名称含depth的对齐参数时，false可使该组学习率倍率为10，true为1。当前align_params为空，不能据这个字段推断存在启用的教师训练。训练语义应结合加载权重、可训练参数和损失共同判断。

## 保存与验收

- `enable_resume:false`：不恢复旧训练状态；与读取model_path中的预训练权重不同。
- `ckpt_manager:dcp`、`save_steps:2`：选择DCP保存路径并设置第2步保存；`save_epochs:0` 不使用周期epoch保存。实际落盘仍须检查运行日志与文件。
- `save_hf_weights:false`、`async_save_hf_weights:false`：本次不导出推理用HF权重，不能预期一定生成可部署的hf_ckpt。
- `output_dir`：本地测试输出目录，不能原样当成服务器路径。
- 最小验收应包括2步更新完成、loss与梯度有限、参数冻结符合预期、实际checkpoint可读取；本笔记仅解释配置，未启动训练或判定这些检查通过。

用户消息中的YAML经过Markdown排版后缩进合并，末尾还出现trueb；本地原文件缩进正常且为`action_fp32: true`，应以原文件为准。

## 源码位置

均相对本地仓库根目录：

- `configs/vla/robotwin/robotwin_local_frozen_smoke.yaml`：本次实际配置。
- `tasks/vla/train_lingbotvla.py`：参数默认值、模型dtype、数据、教师开关与优化器构建。
- `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`：冻结、MoE、L1_fm、action_fp32。
- `lingbotvla/distributed/torch_parallelize.py`：FSDP2与精度策略。
- `lingbotvla/models/loader.py`、`lingbotvla/optim/optimizer.py`：post_training的实际使用。

关联：[[LingBot-VLA 2.0项目概述]] · [[RoboTwin与LingBot-VLA默认流程对照]] · [[LingBot-VLA2与π0、SmolVLA架构比较]]
