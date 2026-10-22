# 115毫米完整任务的独立Agent语义入口

2026-10-22。这里只准备标注、语言与接口；没有实现或验收完整开盖倒药任务。

`registry.json` 将115毫米瓶身与瓶盖分别注册为 `reagent_bottle_body115`、`reagent_bottle_cap115`，沿用既有 `petri_dish` ID。model_data逐字节来自profiles/bottle115，visual为指向该profile的链接；points_info保留既有ID含义，局部坐标来自115矩阵，并保留 `grasp_execution_verified=false`。盖的装配初始中心0.118米不是局部功能点。

原 `test_gen_code.enrich_actors` 已实际处理 `self.bottle/self.cap/self.dish`；原语言函数已实际解析 `{A,B,C,a,b}`，生成12 seen和12 unseen措辞。记录见component_validation.json、enriched_actors.json与language_validation.json；输入源SHA未变。这只是组件核查，未运行几何loader、机器人整程、Agent专家或采集。

此目录故意没有body/cap碰撞目录。物理对象必须由 `Bottle115Assembly` 工厂加载真实115形状，不能让generic create_actor兜底旧90毫米凸碰撞。

`task_info_template.json` 是未来任务的数据模板，当前类体pass且标注PENDING，不应交给Agent假装已实现或已成功。建议task_name为 `labware_cap115_unscrew_pour`，最终名称须与新task/factory/collector一致。当前closed-bodyhold不是最终任务成功。

需要复核元数据时，在本目录外运行robotwin环境的Python执行prepare_metadata.py；脚本只重建该独立overlay，不改profiles、旧metadata或collector。接采集前须完成full_collection_contract.md列出的真实机器人整程、原Agent测试、原collect规划与重放、24维/60ms/三RGB读回。
