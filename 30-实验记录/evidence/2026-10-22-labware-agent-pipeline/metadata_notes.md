# 六件实验室资产的官方智能体元数据接入

核查日期：2026-10-22。基于本机 RoboTwin `931ce21f4f0e63feab661f349634d5a25bd92681`，记录实际源文件 SHA256。此处为本地扩展，不是 RoboTwin 官方原生素材或官方认证。

## 目录与保持不变的内容

`metadata/assets/objects/{asset_id}` 是独立对象目录。新写 `model_data0.json` 与 `points_info.json`，`visual/`、`collision/` 只链接到上一轮冻结包。没有修改官方 `assets/objects`，没有修改上一轮 prepared、manifest 或物理参数。生成前后核对原包 22 个文件的 SHA256。

`metadata/registry.json` 是接线入口，六件资产各自提供：`absolute_modelname`、`model_id=0`、`convex=True`、显式质量、语义点编号、对象描述路径。`episode_description_value` 是描述 JSON 的绝对路径去掉 `.json`；可以直接写入 episode 的 `info` 参数值，由原版语言替换函数读取。

## 编号与动作含义

五件容器原编号全部保留：functional 0=上口中心，1=内底中心，2=已保存的唇沿候选点；新增 3=外部底面中心。瓶盖保留 0=朝下的盖口，1=扭转轴标记；新增 2=外部下支承面中心。新增底点的 -Z 朝外，不能把口部或内底误当作桌面放置点。

四个 contact ID 0、1、2、3 分别是同一高度的侧向候选抓持中心，对应接近外侧 +X、+Y、-X、-Y。contact 坐标系 +Y 是预抓取方向，沿 -Y 接近。点表示候选抓持坐标系，并非已验证的指尖接触位置；RealMan 灵巧手的实际抓持需要另行验证。

`points_info.json` 为原 `enrich_actors` 实际读取的 `{id, description, usage}` 结构。`model_data0.json` 同时保留原 `functional_point_description` / `target_point_description`，添加复数描述别名，便于其他工具读取；真正传给智能体的是 points_info，而不是依赖这些描述别名。

## 官方代码的绑定方式

原 `code_gen/test_gen_code.py` 的 `OBJECTS_PATH` 固定指向官方资产根，不会自动扫描新目录。两种方式已实际核验为一致：

1. 用 `metadata/actor_list_template.json`，在当前进程把 `test_gen_code.OBJECTS_PATH` 绑定到 registry 的 `OBJECTS_PATH`，再调用原 `enrich_actors()`；用完恢复原值。
2. 用 `metadata/actor_list_absolute.json`，其中 modelname 是对象目录绝对路径。原函数的 `os.path.join` 支持绝对右操作数，因此不需要改变模块全局变量。

对象实际加载可直接使用原 `create_actor(scene, pose, modelname=entry['absolute_modelname'], model_id=0, convex=True)`。原函数的 `Path('assets/objects') / absolute_path` 也保留绝对路径；随后显式 `actor.set_mass(entry['mass_kg'])`，避免原 Actor 默认 0.01 kg 覆盖预期质量。物理药瓶需采用原派生机制的 body/cap 装配，不能把两个独立资产加载后就视为已获得旋盖机构。

`metadata/task_info_template.json` 按官方 task_info 的 task_name、task_description、current_code、actor_list 组织；它是资产清单模板，实际任务必须先实现 `load_actors()`、成功判据和需要的动作 API，再收敛到当前任务涉及的 actor 子集。不要把六件清单的读取成功当成示范采集成功。

## 语言描述来源

`metadata/description/objects_description/{asset_id}/base0.json` 使用官方 raw_description/seen/unseen 格式，内容由当前 Codex 根据资产名称与任务要求在本地编写，未额外调用远程 LLM API。seen/unseen 仅为措辞划分：每类仍只有一个几何模型，不代表未见物体或几何泛化评测。元数据内明确保留来源。透明外观、量筒刻度、容器容积没有计量或深度传感器精度认证。

## 本子项实际验证

运行 `robotwin` 环境中的 `prepare_agent_overlay.py`，真实导入原版 `enrich_actors`，六件资产的功能点和抓持点语义均读取成功。原版 `Actor` 在真实 SAPIEN Entity 上执行全部 65 个世界坐标点矩阵变换，矩阵绝对误差为 0。原 contact、target、orientation 与旧 functional ID 保持一致，底面点追加成功。

原版 `replace_placeholders` 和 `replace_placeholders_unseen` 均实际读取六件对象描述，替换 `{object}` / `{a}` 得到完整语句。详细结果保存在 `metadata_validation.json`。

此验收不包含机器人真实抓持、运动规划、旋盖倒药示范、智能体远程推理或数据导出。它只关闭元数据读取和点位编号不一致这两个前置问题；完整流程的其余验收由主任务记录。

## 收敛到实际空瓶抓起放回任务

`metadata/labware_pick_place_task_info.json` 是当前应使用的具体 Agent 输入，只包含 `self.bottle` 一个 actor，绑定 reagent_bottle_body 的绝对目录，并使用右臂。任务要求接触抓起空开口药瓶，离桌至少 2 cm，直立连续保持至少 0.5 s，放回原位、松手、手后退并等待稳定。明确要求仅调用已注册的本地 `grasp_lift_place` 接口，不伪称它是官方原生 API，不允许改 actor pose、加夹具、改物理或成功标签，也不把空瓶任务描述成旋盖倒药。

`metadata/description/task_instruction/labware_pick_place.json` 提供 12 条 seen 与 12 条 unseen 的同义任务指令，全部只使用 `{A}` 和 `{a}`。实际 episode 参数为 `{A}`=药瓶描述 `base0` 绝对路径、`{a}`=`right`。原版 `generate_episode_descriptions(max_descriptions=12, seed=42)` 已生成各 12 条唯一、无占位符的完整指令，见 `metadata/task_instruction_validation.json`。

将原语言模块 `parent_directory` 绑定到 `metadata/description/utils` 即可使用该独立任务指令根。**空 utils 目录必须真实存在**，因为原函数通过 `utils/../task_instruction` 路径打开文件；本地仅创建空目录和 `.gitkeep`，没有复制或替换原工具代码。
