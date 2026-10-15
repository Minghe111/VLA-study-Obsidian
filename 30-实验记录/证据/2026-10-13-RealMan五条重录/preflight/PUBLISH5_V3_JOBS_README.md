# 分散 V3/V4 采集目录的五条补丁验收

原 `publish_randomized5_patch.py` 保留原字节。派生入口 `publish_randomized5_patch_v3_jobs.py` 复用原完整物理、SDK24、RGB、全36关节、motion gate 和复制发布门槛，只增加两项来源限定兼容。

1. 候选从授权的 V3/V4 `jobs/slot{17,18,20,29,34}_runtime/new_physical_episodes` 原始位置发现。每个实际存在的隔离 runtime 分别验证601文件原源码、116条硬件来源保护，以及入口和新增 helper 与 E 内冻结源码的 SHA。原始候选、失败资料、proof 的路径和字节不改。其他目录或 symlink 不准入。
2. `pre_recording_clutter_settle_v1` 的末次实际矩阵与 heading 使用 float64 生产算术。只有原 helper SHA、完整准备 receipt、固定准备 proof SHA 在 expert 来源 registry 中、scene 全部实际 t0 字段与末次准备 readback 精确一致，才能采用 dtype 一处变化的私有 pose reader。继续以同一实际矩阵复算 heading，匹配容差仍为原 `1e-8` 度；姿态分类、最近手臂、碰撞与其他物理阈值全保留。其他生产者继续原 float32 分支。

当前三条 V3 样本的记录值与 float64 算术逐数相同，但与原 float32 算术分别差 `2.3583e-6`、`1.9900e-6`、`5.9294e-7` 度。这是来源明确的记录数值语义兼容，不能据此宣称任何完整物理 episode 已通过。

2026-10-14 实际只读来源核查：五个 V3 runtime 各601源码匹配，五个现有候选的准备来源/固定proof SHA绑定通过。12项 CPU 兼容与反例通过。尚未执行新候选完整物理发布验收，也未发布新数据；来源组件通过不等于完整示范通过。

## 调用

在实际工程中保存派生入口后，用原 robotwin Python 执行。默认仅验收 V3；若使用了 V4，显式传入两个 root。无需移动任何候选或修改 proof。

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python \
  /home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v3_acquisition/preflight/publish_randomized5_patch_v3_jobs.py \
  --collection-root /home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v3_acquisition \
  --collection-root /home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v4_so3
```

五个原编号槽分别通过全部原物理及新增质量门禁后，再使用相同调用追加 `--publish`。数据固定发布至 V3 `published/realman_randomized_rerecord5_v3_patch/dataset`。原50条不替换、不合并45+5、不标为全50条随机化。

V3 初始 `deployment_manifest.json` 中入口SHA为运行前早期版本 `ae2afa0f…`；实际五个 runtime 与 E 的冻结入口均为 `765fbf62…`。派生验收以实际生产来源和冻结 E SHA 绑定，不覆盖历史 manifest，也不将历史早期SHA作为当前入口SHA。
