# 161 专用原生路径简化候选

核查日期：2026-10-14。此目录是独立服务器候选，当前仅 CPU 组件通过；尚未运行 161 上的机器人采集。现有本机 34 号、旧 50 条、E 主覆盖层及 V5 runtime 均未改动。

34 号本机完整候选的真实仿真时长为 80.28 秒，真实时钟视频为 80.3 秒。初次接近包含 391 个 RRT 线段，逐段 TOPP 实际执行 53.132 秒；持物平移包含 173 个线段，执行 19.340 秒。逐段 TOPP 要求各线段两端速度接近零，小线段频繁起停是动作慢的重要来源。

本候选使用原生 MPlib 的 `planner.plan(..., simplify=True)` 减少这些中间节点。有效规划参数仅从当前 `time=30.0, range=0.01, simplify=False` 改为 `simplify=True`。保留较小 RRT 扩展，不改关节速度和加速度上限，不改逐段官方 TOPP，不改机器人、灵巧手、抓取参数、4ms 物理步、每 15 步观测及三路 RGB。

## 集成位置

只把此目录的 `conservative_ompl_overlay.py` 安装到**新的服务器 runtime** 的 `planning/conservative_ompl_overlay.py`。沿用原入口及安装顺序：原随机环境与运动质量覆盖层 → numeric window → 此候选 → 后续间隙、姿态、抓稳、流式存储与观察器覆盖层。函数名及 producer 字段兼容原入口；新增 `variant_id=realman_server_native_simplify_v1` 供结果识别。

不覆盖本机成功候选；新服务器 runtime 必须重新封存源 SHA 与生成函数 SHA。原 601 项来源和 116 项硬件约束照旧，本候选具有自己的 producer SHA。

原生简化不是连续碰撞证书。它返回的**新路径**仍需原最近数值终点、numeric window、raw/dense/cumulative motion quality、完整 native 网格审核、实际逐步几何审核、原任务成功及空中保持门禁。无法通过时应拒收该候选；不能因简化成功绕过这些验收。

## 已完成 CPU 验证

`test_native_simplification_cpu.py` 共 9 项通过。对三个完整 numeric-window 父函数 AST，候选恢复 `simplify=False` 后与现用保守覆盖层生成的 AST 完全一致；没有改动其它语句、审核调用、返回值或阈值。另验证父源不变、动态或重复调用拒收、无 numeric-window 证明拒收、过速／硬件越界／错误 2π 数值终点拒收。

实际本机 MPlib 0.2.1 的合成单自由度无障碍 CPU world 中，同一个 73 节点 raw path 经原生 `simplify_path` 变为 2 节点，精确保持两端点；冻结原逐段 TOPP 的执行时长由 14.116 秒降至 1.672 秒。`plan(simplify=True)` 也实际返回 2 节点。该实验没有真实机器人、随机杂物、SAPIEN scene、renderer 或 GPU，不能把其加速比例套用到 RM65-B。

验证命令（CPU，不需 GPU）：

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python test_native_simplification_cpu.py
```

机器可读证据：`variant_proposal.json`、`cpu_validation_receipt.json`；与冻结覆盖层的字节差异：`conservative_ompl_overlay.diff`。保护器、出口、视频 FPS、原始记录格式及接受状态均未通过此候选改变。

## 161 实测验收

先完整重采 34 并比较首段节点数、逐段起停数量、各动作实际物理秒数、总步数和观测数；同时记录各阶段墙钟与内存／显存／磁盘峰值。通过原物理、运动、RGB、视频及正常退出验收后才继续后续槽位。视频以真实 60ms 观测时钟制作；减少真实动作时长，不能通过改变视频播放 FPS 掩盖速度。实际轨迹与成功率可能因原生简化改变，效果需 161 重新采集验证。

源码语义依据：本机已安装官方 MPlib `pymp/planning/ompl.pyi` 第 77–78 行说明 `simplify` 调用 `_simplifyPath()`；原 `full_native_carry_planner_v31.py` 第 83–94 行逐线段调用官方 TOPP，并检查线段两端静止。全部依据在本次本机读取核查，未借用服务器未执行的结果。
