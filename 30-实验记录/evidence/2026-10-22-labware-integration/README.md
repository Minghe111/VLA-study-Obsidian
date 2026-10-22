# 实验室器皿与功能药瓶资产包

当前目录包含可直接加载的准备资产、RoboTwin兼容标注、独立物理组件和运行证据。它没有覆盖官方资产目录或修改原RoboTwin加载器，也没有连接真机。

## 先看结果

- `scene_visualization/01_front.png`：保留原电子秤和修正左手姿态的RealMan桌面。
- `scene_visualization/05_annotations.png`：几何推导的抓持中心、接近方向、支承面、口部和候选倒出点。
- `physics_visualization/fixture_driven_physical_trace.mp4`：14.6秒实际物理记录回放，永久标明测试夹具驱动。
- `physics_visualization/physics_six_stage_contact_sheet.png`：闭盖含片、倒置、旋转抬升、移盖、倒片、落皿。
- `acceptance_report.json`：最终检查入口；具体物理结果不能用这份文件替代，须查看原始试验报告。

## 资产与目录合同

`asset_interface.json` 是几何、材质、内径、底厚及来源的索引。统一米制、Z-up、scale=1；独立瓶盖原点例外明确写在interface中。

| 目录 | 来源 | 已完成范围 |
| --- | --- | --- |
| `prepared/beaker` | LabUtopia现成烧杯USD及已保存的展开OBJ | 透明材质近似、保留内腔的凸分段、RoboTwin标注 |
| `prepared/erlenmeyer` | LabUtopia现成锥形瓶 | 同上，颈口和锥身碰撞近似单独声明 |
| `prepared/graduated_cylinder` | revenexx/sample-data现成量筒STEP | 真实量筒几何，新增视觉刻度线，未做容量标定 |
| `prepared/petri_dish` | 同库现成培养皿STEP的下皿 | 不盖上源上盖，保留接收口与内底 |
| `prepared/reagent_bottle_body`、`reagent_bottle_cap` | 本地派生参数模型 | 分体可视/碰撞、独立动态盖与螺旋导向近似 |

药瓶没有冒充下载模型的原生旋盖能力。药片是12个独立动态capsule刚体，不是贴图。药瓶和盖的机构规格见 `physics_spec.json`，质量、摩擦及接触参数是工程假设，尚未标定实物。

每个准备目录保留官方刚体布局：`visual/base0.glb`、`collision/base0.glb`、`model_data0.json`。碰撞GLB保留独立凸节点，使用 `convex=True` 通过官方 `add_multiple_convex_collisions_from_file` 加载；单个整体凸包会封住内腔。标注生成依据官方流程，但这些新点由本地几何推导，并非官方或源作者提供的原标，也不是已经成功执行的灵巧手抓取姿态。

## 在当前环境使用

当前Python：`/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python`。以下命令在本目录运行，不需要重新安装或转换现成prepared文件。

标注schema和坐标自检：

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python annotation_contract.py --self-check
```

新做一次物理试验并输出独立报告，不覆盖冻结证据：

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python test_lab_physics.py --seed 0 --out physics_new_seed0.json
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python test_lab_physics.py --blocked-cap --out physics_new_blocked_cap.json
```

实际调用官方加载器：

```python
from pathlib import Path
import sapien
from annotation_contract import load_with_official_create_actor

# scene应为已初始化render/physics系统的SAPIEN Scene。
root = Path(__file__).resolve().parent
beaker = load_with_official_create_actor(
    scene, root / "prepared/beaker",
    sapien.Pose([0, 0, .741]), convex=True,
)
opening_world = beaker.get_point("functional", 0, "matrix")
support_world = beaker.get_point("target", 0, "matrix")
```

兼容函数实际导入本机原始 `envs.utils.create_actor`，保留官方Actor和矩阵变换，并在其默认质量设置后恢复标注中的显式质量。此处采用绝对模型目录，未修改官方全局assets。药瓶的运动机制另由 `lab_physics.LabBottle` 创建，不能只调用整瓶加载器就获得开盖功能。

重新渲染需要本机GPU访问权限：

```bash
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python render_labware_scene.py
/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python render_physics_record.py
```

完整接口、几何假设和标注含义见 `annotation_notes.md`、`physics_interface.md`。源模型重建见 `regeneration_notes.md`；它与直接使用现成prepared资产是两条不同流程。不要在冻结目录直接重建后沿用旧验收结论。

## 已实际验证及适用边界

- 六件资产通过原始官方加载器，全部关键点世界变换与质量读回通过。
- 五件开口容器分别通过1000步重力落球，球经开口停在真实代理内底。
- 三个初始药片朝向种子各14,600步：闭盖倒置阶段全2,500步含12颗；实测转约2.5圈并抬升11.75毫米后允许释放；12颗全部落入导入培养皿尺寸对应的分段碰撞接收器。
- 卡盖负控中，即使命令开盖，实际角度为0时不抬盖，释放也被拒绝。
- 药片在初始化后没有pose写入，之后受重力与碰撞推进；可视化则按保存记录回放，不作为新一次物理验收。

当前瓶体由运动学测试夹具持有，盖由限力限扭D6导向及独立移盖夹具控制。导向按实际转角耦合轴向位移，**不是真实螺纹表面接触**；没有错扣、螺纹摩擦、扭矩标定或儿童安全盖。尚未验证RealMan抓持、双手协作、碰撞自由轨迹或VLA/RL策略；官方抓持助手的固定0.12米夹爪偏移不能直接用于本机灵巧手。没有液体、粉末、化学反应和真实称重反馈。

## 来源与许可

使用GitHub连接器核对源仓库、固定版本和许可。LabUtopia数据资产CC BY-NC 4.0；revenexx/sample-data数据和CAD为CC BY 4.0，模型描述为虚构示例产品。保留源文件、许可与派生说明于 `sources/`、`source_manifest.json`、`LICENSES.md`。本轮没有Git提交或推送，也没有将第三方资产自动公开。
