---
title: 外部实验器皿的RoboTwin功能点兼容契约
created: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - 资产导入
  - 功能点标注
---

# 外部实验器皿的RoboTwin功能点兼容契约

本记录对应 [[RoboTwin 2.0框架与默认任务]] 的实验室资产扩展。标注由实际模型几何与明确的工程规格推导，**不是外部资产作者原有的机器人标注，也不是RoboTwin官方提供的新标注**。矩阵通过官方代码读取与变换，抓持成功仍需本体与任务实测。

## 已核对的官方入口

官方 [Object Annotation指导](https://robotwin-platform.github.io/doc/usage/object-annotation.html) 和 [标注工具源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/scripts/create_object_data.py) 将点分为抓持、目标、功能和方向四类。桌面刚体与URDF关节物体使用不同JSON键名；后者每个点还需要所属link。交互工具可创建、旋转、分组并保存点，URDF模式额外记录link、初始关节量与质量。

本机实际依据 `/home/admin123/liminghe/vla-benchmark/RoboTwin` 的 `create_actor.py`、`actor_utils.py`、`create_object_data.py` 和 `_base_task.py:get_grasp_pose`。版本与文件SHA256保存在同目录 `annotation_specs.json`，后续若加载器变动需重新核验。

## 坐标、尺度与抓持语义

新增模型先整理成 **Z轴向上、米制、底面z=0**；统一使用 `scale=[1,1,1]`。不通过改变标注矩阵旋转来补偿错误的模型坐标。

刚体JSON的键名是 `contact_points_pose`、`target_pose`、`functional_matrix` 和 `orientation_point`，每个元素为4×4齐次矩阵。官方 `Actor.get_point` 只将局部矩阵的**平移**逐元素乘 `scale`，再左乘物体世界位姿；旋转不缩放。因而对于原始毫米网格和scale=0.001，JSON局部平移也需要采用原始毫米坐标，不能再次写成米，否则会发生二次缩小。生成器支持这个换算，首轮准备的资产直接使用米制以减少错误。

刚体 `create_actor` 没有应用 `transform_matrix` 字段，因此本次将几何变换烘焙进网格，字段保持单位矩阵。URDF关节标注则使用 `contact_points`、`target_points`、`functional_points`，每个点为 `{"base": "实际link名", "matrix": [...]}`，并由相应link的当前世界位姿变换。不能将两种schema混写。

抓持标记是**夹持中心及接近方向的候选坐标系**，不意味着标记坐标本身在手指接触的表面。官方药瓶也以瓶身中心附近作为抓持中心。本次同时保存实际三角网格上的左右表面候选点及triangle ID，区分夹持中心和表面锚点。四个侧向接近候选共享同一个夹持中心，组成一组，`contact_points_mask` 对应组数。

官方 `get_grasp_pose` 的固定坐标变换为 `C=[[0,0,1],[-1,0,0],[0,-1,0]]`，其手腕预抓取偏移是 `R_contact @ C @ [-0.12-pre_dis,0,0]`，等价于接触点坐标系的 **+Y方向向外退让**，真正接近为 **−Y方向**。生成器严格遵循这一约定；但官方还有0.12米工具偏移，不能直接拿来宣称RealMan灵巧手抓持姿态正确，仍需现有本体适配器转换与执行验证。

## 各物体的候选功能点

- 烧杯、锥形瓶、量筒和浅皿：底面中心供放置对齐；口部中心供接收或瓶口对齐；真实内底位置供接收区域判定；真实网格上的口沿候选点供倾倒目标使用。倒出点采用实际三角面最近点校正，是否为真实导流嘴另行核验。
- 药瓶瓶身：瓶身中部提供侧向夹持候选，瓶口中心与候选口沿提供开盖后对齐和倒药目标。是否能让药片通过由实际内腔/颈口碰撞决定，不能仅凭功能点推断。
- 独立瓶盖：盖体中部提供侧向夹持候选，盖底中心用于与瓶颈对齐，+Z标记旋转轴。该功能点本身不提供螺纹、解锁或可分离机制。

实际内半径、内底高度、抓持高度和倾倒口沿可由 `geometry` 覆盖；没有提供的内腔几何明确记录为未验证，不根据包围盒虚构中空能力。候选接触点并不包含灵巧手的指关节形状或力闭合结论。

## 碰撞与质量的真实加载核验

**必须保留内腔。** 官方 `create_actor(convex=True)` 使用 `add_multiple_convex_collisions_from_file`。已用17个独立GLB节点组成的中空杯夹具实际验证：16块杯壁与1块杯底导入后保留17个 `PhysxCollisionShapeConvexMesh`。`convex=False` 实际进入单个三角网格分支，并非多段凸形。本次动态器皿使用前者，不将整个容器合成一个凸包。

进一步使用半径4毫米的刚体小球，从杯口上方自由落下，运行500步、步长4毫秒后，小球中心停在内底以上约8毫米处，低于80毫米杯口。此结果证明**该加载路径下的测试夹具**口部未被单凸包封死；不能替代最终导入烧杯/药瓶的碰撞验证。

官方 `Actor.__init__` 默认把动态刚体质量重设为0.01千克，不会自动采用JSON中的自定义质量。本兼容入口在实际官方加载之后明确调用 `set_mass`，已验证输入0.1千克读回约0.1000000015千克。器皿质量暂属工程假设，未做真机称量；摩擦、接触偏置及物理材质还应由场景初始化统一指定。

## 可调用入口与检查范围

同目录 `annotation_contract.py` 提供：

```python
data = generate_model_data(
    mesh_path, "beaker", scale=(1, 1, 1),
    geometry={"inner_radius_m": 0.027, "floor_z_m": 0.0025},
    source={"repository": "...", "source_asset": "..."},
    mass_kg=0.10,
)
validate_annotations(data)
actor = load_with_official_create_actor(
    scene, asset_directory, pose, convex=True, mass_kg=0.10,
)
```

外部目录使用官方 `visual/base0.glb`、`collision/base0.glb` 与 `model_data0.json` 布局，OBJ也可使用官方支持的 `textured0.obj`。兼容入口先在RoboTwin所需的工作目录导入**原始官方模块**，再通过绝对物体目录调用原始 `create_actor`；不改官方全局 `assets`，也不复制重写加载器。

每次加载均逐一比较官方 `Actor.get_point` 的真实输出与参考世界变换。CPU自检另外验证非均匀尺度、旋转和平移、URDF link转换、错误旋转拒绝、分组与描述数量；这些检查与实际SAPIEN落球检查分别记录。纯CPU Scene缺少render系统不能满足官方visual加载要求，实际兼容核验使用完整SAPIEN Scene。

## 最终导入资产的实测结果

六个实际准备目录已经生成 `model_data0.json`，并在完整SAPIEN Scene内通过**原始官方** `create_actor` 逐一加载。烧杯、锥形瓶、量筒、培养皿、瓶身和瓶盖分别保留49、193、193、49、49、33个凸形碰撞件，与输入interface的分段数逐一相符。使用非零yaw旋转与非零平移核验全部四类点的世界坐标，质量读回也通过。细节与最终visual/collision GLB SHA256在 `annotation_specs.json` 的 `prepared_asset_validation` 中。

对五件开口容器分别再进行1000步、步长1毫秒的真实重力落球检查。半径3毫米、质量0.35克的小球，从口部上方4厘米开始，最终在烧杯、锥形瓶、量筒、培养皿、瓶身内分别停在距底面4.5、5.2、4.8、4.0、6.0毫米处，均与声明的内底高度加球半径相符。这证明**实际导入的分段碰撞代理**具有可通过的开口与承托内底，不只是visual看起来透明。真实玻璃厚度、摩擦和惯量尚未标定。

烧杯原扫描包含28个零面积三角面，原样视觉与SHA不变；最近表面查询时排除这些退化面，并保留映射回原始face ID的证据，避免数值NaN污染功能点。器皿轴心采用interface的 `x=y=0`，不将烧杯导流嘴导致的非对称包围盒中心误当瓶轴。瓶盖保持自身原点与 `z=-0.012..+0.002` 米范围，下口功能点法向为−Z，旋转轴法向为+Z。

当前通过了schema、坐标变换、实际官方加载、多凸段计数、质量读回和容器落球检查；没有把这些结果等同于RealMan抓持、机器人拧盖、倒药整任务或VLA训练成功。
