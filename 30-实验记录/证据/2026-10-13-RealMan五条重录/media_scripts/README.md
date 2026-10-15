# 新专家三相机视频合成与读回

`render_accepted_expert.py` 只处理已经通过完整验收的新随机化专家候选。它不会启动仿真、模型或真机，不修改候选目录、原视频、HDF、动作和验收报告。

用现有 robotwin Python 运行，提供 `--candidate` 和独立的新 `--output` 目录。FFmpeg 默认取 PATH 或该 Python 同目录的二进制，也可以用 `--ffmpeg` 指定已有二进制；无需安装环境。

输入候选须包含：

- `recording_report.json` 的 accepted=true，以及真实新物理仿真和 hardware_io=false。
- `independent_motion_quality.json` 的 overall_publish_gate_passed=true，且报告 SHA 与 recording_report 一致。
- `sample_spec.json` 的原槽位索引 17、18、20、29、34及 actual_domain_seed=seed。
- `data/episode_0000000.hdf5`，SHA 必须与上述两个验收报告绑定。
- `video/head_realtime.mp4`、`left_wrist_realtime.mp4`、`right_wrist_realtime.mp4`，每个为原导出的 640×480、30fps、原60ms图像时钟零阶保持视频。

工具逐个全量 FFmpeg 解码源视频，检查帧数、FPS、尺寸、容器时长与原导出报告；再从 HDF 只读验证 60ms/4ms 时钟、实际状态链和观察状态对应。三个源画面横排在 1920×600 视频中，明确标为新专家真实物理仿真及官方场景随机化，并标为非策略推理。

每帧显示槽位、品牌、执行臂、域随机种子、播放时间、该画面实际保持的原观察索引/时刻、实际4ms物理步及左右 J6 实测角度。原导出器的浮点时间边界比较保持不变，不通过取模改变角度。J6来自 dense measured_state24 左第5、右第17列，按该帧播放时刻之前最近的真实物理测量显示。画面不插值、不加速。

输出包括合成 MP4、`overlay.ass`、`actual_frame_clock_j6.csv`、全部解码日志及 `video_validation.json`。合成后再次全量解码，核帧数/FPS/时长；所有输入重新核 SHA，确认原始字节不变。失败保留 `video_validation_failed.json`，不会更改候选验收状态。

准备状态记录在 `tool_preparation_validation.json`。目前只进行了语法/CLI 和时钟标记边界检查，没有合格候选时不会生成或宣传已验证的新采集视频。媒体合成不能替代实际采集、规划质量、物理成功、随机化或碰撞守卫验收。
