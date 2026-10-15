# 五条 upright 轨迹的独立运动质量门禁

本门禁仅适用于17、18、20、29、34的重录槽位，保留原品牌、左右臂及upright类型。新场景按用户要求使用官方demo_randomized域随机配置；不再要求恢复旧样本的完全相同初始场景。原45条保留，不把它们宣传为已经通过此新运动门禁。

root于2026-10-13确认阈值：接近窗口内，实际和命令每关节范围≤180°，累计往返≤225°；实际/命令4ms差≤1°，60ms下一目标差≤8°；腕相机中心累计路径≤0.55m，首末位移≥0.10m时路径比≤2。旧legacy EE位置长度仅附报，不作准入硬条件。它不是实机标定TCP。

接近窗口默认从第一条真实记录到应用命令中任一前五弯曲SDK通道首次低于950，排除第六个拇指旋转通道。这是“明显闭合之前”的诊断界限，不代表首次真实触碰。若新物理记录提供首次手–目标接触时间，另按该实际时刻计算预接触窗口，使用相同门槛。两者分开记载，不把代理界限写成接触事实。

短几何路径不自动代表可行或安全。门禁不会移动物体、修改状态、裁掉前缀、夹断角度或生成画面。采集不能达标时拒收，不通过放宽物理安全或重标成功来满足要求。

## 调用

`motion_gate.evaluate_candidate(numeric_path_or_hdf, metadata, config_path, branch_receipt_path, upstream_acceptance_path, actual_first_target_contact_time_s=None)`只读原始数值。输入可为HDF5或保持原字段名的NPZ；metadata须含index、brand、arm、pose_class。

输出分别保留`motion`、`planner_branch`、`planner_source_binding`、`original_upstream_acceptance`及`overall_publish_gate_passed`。只有确认范围、真实轨迹质量、真实分支receipt、原始规划记录与实际执行对应、全部上游验收均通过，才允许发布。缺证明就是未通过，不补造pass。

上游`upstream_acceptance.json`必须有下列字段，每项为`{passed:true,evidence_path:"...",evidence_sha256:"..."}`，实际文件SHA必须相符：

- `original_physical_success`
- `original_airhold_acceptance`
- `original_safety_acceptance`
- `official_demo_randomized_actual_parameters`
- `actual_clutter_collision_guard`
- `hardware_camera_physics_clock_unchanged`

真实随机参数与真实clutter碰撞守卫是必要证明。不能只给随机开关配置，也不能用旧成功报告代替新场景的实际成功。

## Planner receipt与实际执行对应

目前独立validator接受`nearest-periodic-branch-v1`。真实新planner需记录每次事件的arm/stage/sim_time_s/reference_kind、reference_q_rad、raw_ik_q_rad、selected_q_rad及实际枚举的`candidates:[{q_rad:[...]}]`，另提供每臂六个revolute关节类型和六对真实joint limit。

validator从记录的raw IK独立枚举所有限位内2π等价候选，检查chosen与raw周期等价、chosen合法、实际receipt确实含候选，并核对它是离记录的参考关节位置最近的周期分支。该代数检查不证明整段轨迹无碰撞，实际场景安全/物理成功是另一个必要门槛。

`planner_source_binding`进一步只读每个event指向的新`free_approach_*.json`并核SHA。逐字段核对唯一accepted原RRT分支的raw IK family、原actual before36关节、限位与类型、实际93-link FK声明与native collision候选、near chosen goal、原raw/TOPP/audited endpoint。再把planning时刻绑定HDF真正的measured_state24，并将原retimed position逐条与相同真实4ms时钟下的applied_command24比对。读回native drive target可能是float32，因此同时核原值与其明确float32存储表示，容差1e−7rad、另报两种最大误差；这不修改执行数组、角度或任何运动阈值。原规划成功记录必须全部覆盖、不能重复，新HDF与source应属于同一个candidate目录。缺源、错误对应或synthetic标记均拒收。这里核的是保存的实际证据及实际命令，不重新运行规划或编造FK/collision。

当前采集入口按保护的官方dense executor在planning后立即逐条执行同一position数组；若后续executor改变调度，必须记录真实执行起点并适配对应证明，不能删除执行对应检查来求pass。

### 辅助数值窗口的显式拒收记录

2026-10-14补充兼容独立规划模型窗口：窗口外的周期等价目标未执行碰撞检查，因此原规划回执只保存 `full_qpos_rad`、声明索引及明确窗口拒绝理由，没有 `collision_count`。这些记录只有在完整窗口证明有效、原36关节源次序与界值可核、current与实际测量吻合、窗口等于current±(π−1e−5)与原J6界的交集、目标实际超出窗口且与原93链接FK声明/冻结实际状态逐项对应时才可显式排除。不会伪造 collision_count=0，也不会泛忽略缺字段。真实检查得到0且位于窗口内的目标才可参与legal family。

`actual_full_vendor_limits_rad` 为辅助helper保存的原native auditor导出CPU模型 Pinocchio 36关节界值，命名沿helper兼容字段；不将它宣传为另一次真机厂家限位标定。六主动关节与原PhysX命令界读取对应；声明目标保持原精确主动界检查。冻结关节与实际初值精确相同，并保留原FMA对初始测量/冻结关节的1e−4 rad数值容差，接受−7e−7等微小测量差，不夹断数值。raw、TOPP命令及实际audit vertices也独立核窗口。

`test_window_source_binding.py` 包含25项纯CPU合成正/负例；有效显式排除与原测量容差通过内部一致性，伪理由、缺/错窗口、假碰撞计数、错误原FK/冻结数据及窗口外轨迹均拒绝。synthetic整体发布仍被阻止。执行证明见 `window_schema_validation_20261014.json`；旧无窗口合法schema及原来源SHA、末点、4ms执行对应、FMA证明继续保留。

## 已完成旧数据对照

`validate_old_reference.py`对原始数字实际执行门禁：旧五条全部被运动条件拒绝，短轨upright参考0、2、13、14、25、48的运动子门禁全部通过；另两条同组异常参考3、16被拒绝。

匹配限制保留：只有Coca-Cola左臂upright的2是目标组的精确正常对照。Fanta左臂除17/18无其他upright；Fanta右臂的3也绕行异常，不能当正常匹配对照。其他参考是同upright、同或镜像臂、跨品牌比较。旧样本没有真实新planner或新clutter证明，这些对照不会被宣称通过完整新数据发布门禁。

`test_gate_components.py`与`test_source_binding.py`为明确标记的纯CPU组件反例，不是新示范。后者所有临时fixture执行后删除，synthetic fixture即使内部字段一致也不能通过完整发布门禁。
