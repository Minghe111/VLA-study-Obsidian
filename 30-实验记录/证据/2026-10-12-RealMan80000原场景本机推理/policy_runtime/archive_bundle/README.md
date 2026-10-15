# 80000 步本机推理运行包

本包保存本次实际使用的模型服务、准确 OpenPI 源码、tokenizer、依赖版本及验收证据。它不包含32.3 GB检查点、临时venv或pip下载缓存。

`native_bindings`另存本次全50训练工厂与24维数据适配器原始副本及SHA，用于核对本机工程中的对应文件；当前运行入口仍导入本机工程位置的原文件，不会自动把副本替换进工程。

本次运行于本机RTX5090、Python3.12，调用官方 `create_trained_policy`；模型为π0.5、32维输出、H16、bf16、10次去噪。官方全50数据工厂恢复左右臂绝对关节目标及因时手SDK绝对目标，物理输出为16×24，未作动作截断或专家替换。

## 本机前提

- 保留已下载的 `80000` 检查点及其 `params`、`assets` 子目录。恢复推理时不加载 `train_state`。
- 本机检查点和训练工厂位置已固定在运行脚本中：`/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1`。这是本次主机路径，并非其他主机的通用路径。
- 独立venv复用 `/home/admin123/liminghe/miniconda3/envs/robotwin-official-data` 的Python3.12、Torch2.8及现有依赖，其余运行依赖在临时venv覆盖。不会改变现有Conda。
- `requirements.txt` 包含JAX0.5.3、Flax0.10.2、Orbax0.11.13等准确版本；所有实际版本及来自venv或只读Conda的路径见 `environment_receipt.json`。

## 恢复到本次主机的临时路径

以下步骤仅在原临时运行目录不再存在、且没有正在运行的本次服务时执行。不要覆盖正在运行的环境或服务结果。需要网络下载依赖，模型权重使用已经保存在本机的80000检查点。

1. 将本包中的 `runtime80k.py`、`service80k.py`、`launch_local.sh`、`requirements.txt` 放回 `/tmp/realman-eval80k-20261012/policy_runtime`。
2. 将 `openpi215ab.tar` 解压到 `/tmp/realman-eval80k-20261012/official_openpi`，将 `source_identity.json` 放回 `/tmp/realman-eval80k-20261012/source_helpers/source_identity.json`。
3. 将 `paligemma_tokenizer.model` 放回 `/tmp/realman-eval80k-20261012/policy_runtime/cache/big_vision/paligemma_tokenizer.model`。
4. 创建只复用已有环境依赖的venv并在该venv安装运行包：

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin/python -m venv --system-site-packages /tmp/realman-eval80k-20261012/policy_runtime/venv
/tmp/realman-eval80k-20261012/policy_runtime/venv/bin/python -m pip install -r /tmp/realman-eval80k-20261012/policy_runtime/requirements.txt
```

5. 检查全50工厂与适配器仍存在于本机工程，然后以一个全新的服务结果目录启动：

```bash
bash /tmp/realman-eval80k-20261012/policy_runtime/launch_local.sh \
  --output /tmp/realman-eval80k-20261012/policy_runtime/service_rerun \
  --openpi-root /tmp/realman-eval80k-20261012/official_openpi \
  --source-identity /tmp/realman-eval80k-20261012/source_helpers/source_identity.json
```

服务仅绑定 `127.0.0.1:8769`；`GET /healthz`（或 `/health`）只读查询模型身份，`POST /infer`才执行推理，`POST /shutdown`正常停止模型服务并释放该进程资源。

`/infer`请求包含三路真实RGB（`cam_head`、`cam_left_wrist`、`cam_right_wrist`，640×480、base64 PNG）、24维实测状态 `state24` 及静态语言 `prompt`。必须使用新服务发布的 `model_identity.json` SHA验证客户端及响应，不能复用旧SHA。

## 本次已处理的环境差异

- 系统 `/usr/bin/ptxas` 12.0不支持5090的CC12.0，本任务独立venv使用CUDA12.9编译器；`launch_local.sh`只修改当前服务进程的PATH及XLA编译数据目录。没有改系统CUDA。
- 官方代码会导入Torch，但实际推理执行器为JAX。先导入Torch使其当前CUDA runtime可用，避免JAX0.5.3先载入旧runtime导致Torch导入符号错误。没有使用Torch执行模型。
- 本机数据镜像缺少派生 `data50_v1/assets`，因此通过官方 `AssetsConfig` 将同一全50工厂绑定到80000检查点自带的 `assets`。归一化SHA必须等于训练硬编码 `34cbdd09c831e04549d6c4a4e2eb321fb975bdea13bb816ec8aa3a2332d1ccc5`。
- XLA显存预分配60%，实测服务占用20681/32607 MiB；同卡还运行仿真与渲染。

`gpu_probe.json`是实际GPU矩阵乘通过证据，`load_acceptance.json`是实际参数恢复及健康证据。它们均不能单独证明抓取成功；抓取结果依赖对应episode的物理轨迹、视频和任务验收记录。

## 建议保留

永久保留本包、完整检查点、下载SHA回执、六场episode报告/视频、模型请求的原始RGB/state和原始16×24输出，以及最终输入动作来源审计。准确源代码与tokenizer已包含在本包内。

venv可暂留以便快速再次运行，但不能把venv或pip缓存当作可迁移的软件源代码；`/tmp`可能清理，长期使用应按独立的新部署任务迁移运行入口并重新核验。`pip_cache`可在所有下载成功及本次工作结束后删除，通常不需永久归档。
