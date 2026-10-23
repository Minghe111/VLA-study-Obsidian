# 实时三相机预览

本地实际 SAPIEN 画面，固定入口 <http://127.0.0.1:8765>。2026-10-23 用户要求后，后续本任务的物理测试默认经 `run_visible_test.py` 启动，并在测试前打开此窗口。

显示头部、左腕、右腕真实 RGB、实验名、控制阶段、物理 tick、仿真时间、画面时刻及异常。默认目标3组画面／秒，受渲染和计算耗时影响；无新帧时明确标记等待，进程退出时保留最后一帧。每个视角可放大。CPU几何／规划检查没有物理动作，不能假装成机器人运动。

启动服务（仅监听本机回环地址）：

```bash
python3 -B /home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/live_preview/preview.py --port 8765
```

实际物理测试统一入口：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/robotwin_labware_mpl \
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python -B -u \
/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/run_visible_test.py \
  --run-name TEST_NAME --fps 3 /absolute/path/to/probe.py -- [原测试参数]
```

物理步保持4ms、SDK24窗口保持60ms；默认只对过快的动作窗口增加墙钟等待，方便观看，不插入物理步。渲染在仿真主线程读取，不写机器人或物体状态。每帧前后核对物理步、关节位置／速度、drive、本体state24及已有装配体位姿／速度；恢复监控前的 `last_observation`。该检查不能单独替代完整参考轨迹对照。

预览源代码另存到当前实验的 `preview_source_snapshot`，并保存 `preview_provenance.json`、`live_preview_receipt.json`。临时JPEG仅位于 `/tmp/robotwin-labware-live-01a0b3a1`，每轮保留最新8组，不替代原实验轨迹／最终图像。HTTP只提供预览页面、状态与相机JPEG，不提供笔记库文件浏览或机器人控制接口。服务需要持续运行；电脑重启或服务退出后，须重新启动服务。跨主机测试需另行接入，当前不声称支持远端画面。
