# 外部实验室器具的来源与授权

核查日期：2026-10-22。以下是研究仿真资产导入，不是制造商数字孪生或容量计量认证。

## LabUtopia 烧杯与锥形瓶

- 项目：[Rui-li023/LabUtopia](https://github.com/Rui-li023/LabUtopia)。作者 Rui Li 等，2025。
- 数据：[Ruinwalker/LabUtopia-Dataset](https://huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset)。固定数据 revision：`e622f0ec8123c9098d913ea56a4116fd83a29662`。
- 官方 README 明确区分：代码 MIT，数据资产 **CC BY-NC 4.0**。当前导入的 USD、其派生 OBJ/GLB 继续按 CC BY-NC 4.0 记录，不能把代码的 MIT 许可证套给资产。
- 署名：Rui Li et al., *LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents*, 2025。
- [资产许可](https://creativecommons.org/licenses/by-nc/4.0/)。用途为非商业学术研究。2026-10-22随研究笔记归档时，保留本节署名、资产许可链接及下述派生改动说明。
- 改动：复用 2026-10-21 从源 USD 提取的可视几何，将原厘米单位、源变换转换到米制 Z-up；本轮将烧杯轴心平移到 XY 零点、所有静态器具底面置于 z=0；Omni MDL 玻璃材质由 SAPIEN 透明材质近似；另建立保持内腔的分段凸碰撞代理。原扫描网格和几何文件哈希保存在 `source_manifest.json` 与 `asset_interface.json`。
- 限制：烧杯可视扫描网格本身非完全水密；物理碰撞使用独立封闭凸件。没有继承 LabUtopia 的液体模型、控制器、任务逻辑、流体仿真或材质标定。

## 真正的量筒与培养皿 CAD

- 来源：[revenexx/sample-data](https://github.com/revenexx/sample-data)，固定 commit：`e7217fb87daf42adc2e61233a4a1bba61adef946`。
- 固定源：`packs/lab/assets/cad/demo-lab-4001-measuring-cylinder-100ml.step` 和 `demo-lab-3003-petri-dish-90.step`。STEP 是实际下载的现成几何，非图片逆建模。
- [LICENSE-DATA](https://github.com/revenexx/sample-data/blob/e7217fb87daf42adc2e61233a4a1bba61adef946/LICENSE-DATA) 指明 `base/`、`packs/` 下数据、媒体和 3D 模型为 **CC BY 4.0**。署名：**revenexx sample data, Copyright (c) 2026 revenexx**。
- [资产许可](https://creativecommons.org/licenses/by/4.0/)。`tools/` 脚本另受 MIT 约束，已保留源 LICENSE、LICENSE-DATA，以及仅作源依据的 CAD/可视化脚本。
- 改动：实际 STEP B-rep 经临时独立 Gmsh 4.14.1 转三角网格；毫米转米；统一 Z-up、底面 z=0；量筒添加侧面开放弧形刻度线，未添加堵住内腔的实心圆盘；分段凸碰撞代理为本地适配。
- 量筒是直管加底座的量筒，不是容量瓶。源 CAD 外径 31 mm、底座直径 58 mm、高 250 mm。源 CAD 没有印刷刻度；新增刻度仅为 15 mm 间隔视觉标记，**不代表毫升刻度标定**。源目录名的 100 mL 是作者的虚构产品名称。
- 培养皿 STEP 有两个独立实体：下皿约外径 87 mm、内径 85 mm、高 14 mm、底厚 1 mm；上盖外径 90 mm。导入实体 1 下皿供接收药片，源中的上盖保留而不盖住开口。
- 原库声明产品、制造商和公司均为虚构，不能当作真实厂家型号、准确容量或产品合规证据。

## 本地派生功能药瓶、瓶盖与碰撞近似

- 药瓶本体、独立瓶盖是依据 `physics_spec.json` 新建的参数化研究几何，**不是声称某个下载模型原生支持旋盖**。
- 本体外径 50 mm、内径 42 mm、高 90 mm、底厚 3 mm。瓶盖外径 60 mm、内径 51.6 mm，独立刚体；局部原点位于顶圆盘中心，与 `LabBottle` 接口一致。
- 旋盖由受力/力矩约束的螺旋导向近似完成；不把关节角变化、脚本移动或可视拆分说成真实螺纹接触验证。实际验证范围由物理测试输出单独记录。
- 药片为独立刚体胶囊近似，只表示可流动、可计数固体，不包含真实药物化学或材料性质。
- 所有新碰撞件为派生几何，不直接继承真实玻璃摩擦、质量、破裂、粘附、计量精度或流体性质。

## 工具范围

Gmsh 仅安装到 `/tmp/labware-cad-runtime` 用于离散化，没有修改现有 Conda 环境。原 STEP、USD 和出处声明随证据保留。现有 RoboTwin 原资产、RealMan URDF 和主任务默认配置均未被此转换程序覆盖。
