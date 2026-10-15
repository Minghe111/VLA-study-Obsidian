# 五条随机化重录补丁验收与发布

本工具只在真正采集的源候选目录执行独立验收。它保留原50条、现有训练数据和所有失败候选；不合并45+5，不将五条补丁或混合数据称为“全50条随机化”。

目标槽固定为17、18、20、29、34，分别保留Fanta左臂、Fanta左臂、Fanta右臂、Coca-Cola左臂、Coca-Cola左臂及upright类别。实际新场景使用官方`demo_randomized`完整配置，默认种子为源种子加10000000乘重录尝试数，也支持已有明确种子覆写记录。官方2%的干净场景抽样仍然是合法的随机化采集结果。

`publish_randomized5_patch.py`默认只验收。复制该工具、原source和quality目录到工程持久采集目录后，以原robotwin环境运行该工具；只有显式传入`--publish`且五槽全部通过，才发布至该持久采集目录的`published/realman_randomized_rerecord5_patch_v1/dataset`。不要在笔记库中运行大数据发布。

## 验收字段

- 源50条manifest固定SHA为`2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80`，逐槽保留旧HDF SHA并要求新HDF不同。
- 原601文件源码逐文件SHA匹配，源文件字节不改。离线reader仅绑定准确生产RoboTwin工作目录与原CPU验证器的源码根路径；每个AST替换site、原SHA和编译AST SHA记录在独立验收回执中，不能替换已采集proof路径。
- 重跑原v3逐条物理、全36关节、24维SDK、15×4ms/60ms、RGB与原N+1缓存的完整读回；保留原数值边界验收，必要时由原验证器独立重建零步CPU事件，不创建新物理示范或模型推理。
- 在真实源候选目录重算`motion_gate`，验证已记录质量回执、HDF SHA、planning实际branch和upstream真实证明的SHA连结。绝不重写路径使复制位置冒充原采集地点。
- 重建完整实际physics trace上的随机杂物组件覆盖、零有效接触与≤1mm漂移；实收DR参数、实际种子和桌面高度必须与原配置一致。
- 前后真实硬件回执的静态关节、PD、质量/惯量、材质、碰撞offset和36关节/93link清单相同，实机IO为false，三路相机与精确时钟由原RGB/物理读回验证。
- 原生`video/episode_0000000*.mp4`通过实际cv2元数据检查及已安装ffmpeg全量解码、帧数终点核对，保留文件SHA。视频仅作预览；时钟事实以原raw/HDF/physicsNPZ为准。候选外合成预览独立链接，不混入原证据。
- 五槽全部通过后，按最低已验收尝试号各选一条；全部源inventory在验收前后、复制前后不变。新补丁独立复制全部候选证明，内部顶层HDF/视频只链接新补丁内的冻结副本；复制字节与已验收源SHA完全相同。

## 发布结果

顶层HDF保留原编号`episode_0000017`等；manifest包含5条`validated_source_candidate`、真实`physical_recording_location`、实际种子、源50与旧HDF SHA、新HDF SHA、复制HDF相同SHA、所有视频SHA、原完整source inventory和独立验收证明。原proof中的绝对路径与字节保持原样，永久源候选仍可追溯。

发布采用独立staging目录和最终原子rename。失败staging与候选保留，完整manifest只在所有复制文件校验通过后写入。`complete=true`仅表示此5条补丁完成，`full50_randomized=false`始终明确保留。

截至本工具准备时，只完成20项CPU反例、原50条manifest及601文件源码与v3 reader CPU导入检查，以及实际旧17视频的438帧全量解码API验证；尚未执行任何新候选完整验收或数据发布。CPU组件与测试通过不代表重录五条已经完成。
