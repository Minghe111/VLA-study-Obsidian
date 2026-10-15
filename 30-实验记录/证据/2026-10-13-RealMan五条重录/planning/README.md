# 新 5 条示范的最近 IK 分支与运动质量约束

核查日期：2026-10-13。范围：任务目录内的采集器覆盖层；没有修改原仓库、原 50 条数据、原训练或真机。

`collector_quality_overlay.py` 提供 `install_planning_quality(source_package=..., legacy_directory=...)`。在独立 worker 进程中安装一次，环境与场景覆盖层另行安装。原始采集器及实际 SDK24、三路 RGB、15×4ms 时钟、手部力控制、LP、TOPP、几何与持稳验收继续由原 worker 执行。

源快照实际文件为 `full_native_approach_equivalent_goals_v6.py` 和 `full_native_approach_equivalent_goals_dense_v7.py`，并不存在名为 `full_native_approach_finite_v6.py` 的文件。安装器核对两者固定 SHA，再生成任务内派生源。原文件保持不变。v9 的 dense_v7 子类继承该实际入口。

原入口对每个 IK 解预先声明有限关节范围内的等价数值端点，检查全部原生 FK 链节和完整碰撞世界，并按与当前实测关节状态的距离排序。不过，原 RRT 收到整族近、远数值目标，原 endpoint 验收允许匹配其中任意一个。新增覆盖层只将每族距离最近的合法数值目标交给 RRT；族间仍按原近解顺序尝试最多 6 族。原 IK 有限范围、冻结关节、全链节等价 FK 和碰撞声明不放宽。补充显式 palm 目标误差记录。

原始 RRT 终点、TOPP 返回的密集位置终点、最终几何审计顶点终点必须匹配同一个被选中的数值目标。合法远 2π 别名也会被拒绝。不会对返回或录制轨迹使用 modulo、unwrap、限位裁剪或逐点替换。

本轮新增示范质量阈值独立于硬件安全约束：

- 完整 pre-contact 接近阶段累计 J6 行程与跨度分别不超过 π rad，各关节跨度不超过 π rad。
- 关节向量折线路径长度不超过首尾直线距离的 3 倍加 0.15 rad。
- 相同 native palm 世界坐标原点的折线路径长度不超过首尾直线距离的 3 倍加 0.03 m。
- TOPP 密集命令相邻步差不超过原 official velocity limit × 实际 4ms × 1.02 + 1e-5 rad。稀疏 RRT 节点间距只记录，不冒充 4ms 执行间距。
- 两段接近路径合并评估，避免分别通过半圈门槛而累计绕一圈。原生 SAPIEN 实测关节状态与 palm 位置逐物理步记录，实际累计质量在手部闭合第一物理步之前验收。

规划路径的精确原始数值端点容差不套用到物理伺服跟踪误差。实际末点与已选命令目标的关节误差另行记录。持物后运动仍保留原完整安全验收；本轮接近质量阈值不代表全部任务阶段的运动策略。

原 `FullMeshPathAuditor` 仅将 bottle/table 放入环境碰撞世界，不能直接声称覆盖随机杂物。环境覆盖层由 root 的 `randomized_environment_guard.py` 提供。近解规划必须核对实际保留的 `_rerecord_clutter_entities` 与 `audit.metadata['randomized_environment_coverage']` 中计数、actor identity、几何分类和完整覆盖证据；未覆盖直接拒绝。只有原官方抽样到的明确 `native_clean_background_draw` 才允许随机化配置下的 0 杂物分支，丢失引用或静默关闭杂物不通过。

已完成的验证：

- `cpu_counterexample_receipt.json`：23 项纯 CPU 质量工具反例通过。
- `collector_overlay_counterexample_receipt.json`：11 项受控原 planner 调用流程反例通过，包括 raw/TOPP/audit_vertices 远别名、原端点正确但路径绕行、累计两阶段门槛、随机杂物漏覆盖等。测试使用受控 native/planner double，不是实际几何或物理通过证据。
- `native_fk_near_goal_receipt.json`：在实际冻结 RealMan URDF、原厂 mesh 和 SRDF 构造的 CPU MPLib/Pinocchio 模型上，左右臂各 93 个链节证明 synthetic `J6=-5.25` 与 `J6=1.033185...` 等价；当前 `J6=0.93` 时选择后者，前者不能通过 chosen endpoint 匹配。全模型矩阵误差分别约 6.66e-16、8.88e-16。没有创建 GPU/物理场景，也没有证明这些 synthetic pose 碰撞自由。
- 真正 robotwin Python 的 import/install smoke 通过，原源 SHA 未变，派生代码和新增模块 SHA 注册入专家来源集合。

这些验证证明覆盖层的选择与拒收逻辑，并不证明新随机化 5 条数据已经采集成功。新数据必须再通过原 worker、环境逐步验证、实际几何、持稳、原生 HDF 时钟与画面验收，才能成为可替换数据。

## 2026-10-14 持久恢复与重新验证

主机恢复后，原 `/tmp` 文件已丢失。本目录从已保存 agent 日志的 16 个实际 Add/Update 补丁顺序恢复 9 个最终文件，恢复证明见 `recovery_receipt_20261014.json`；未执行历史 shell 命令。核心三个模块保持原补丁内容，流程测试中唯一的路径修订是将写死的临时 source 路径改为本目录上级的持久 source 路径。

原 601 个 collector 源文件恢复并逐项 SHA 核对后，重新通过 23 项最近分支/运动质量反例、11 项 collector 调用流程反例、22 项严格随机桌面支撑反例。真正 MPLib/Pinocchio 左右各 93 链节 FK 验证通过，矩阵误差分别为 `6.66e-16`、`8.88e-16`；CPU installer 也完成原固定 SHA 核验与规划、闭合前实测、严格支撑检查接入。证据分别为三项 `*_counterexample_receipt.json`、`native_fk_near_goal_receipt.json` 和 `planning_support_install_cpu_receipt_20261014.json`。这些验证未创建仿真 Scene、未推进物理时钟、未启动 GPU，不能据此声明新随机化示范已采集通过。
