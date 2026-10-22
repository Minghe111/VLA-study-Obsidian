# v4无首触后的实际几何读回

实际left cap冲量对应力前后均0；末态真实q36下最近thumb gap 4.7927mm，index gap 0.0000mm。

各实际父指节及sensor均检查，不把sensor表面当必要条件。对比实际关节与已有drive、原first-surface和新yaw-minimal既有目标；没有新姿态族、没有增加close budget或物理执行。原始距离/FK/来源见JSON。

## 独立表面分离修正

原first-touch标签失效：由实际PhysX碰撞顶点重新构建ConvexHull面，原thumb/index对全部33盖piece均有分离平面，实际间隙下/上界分别为4.154–4.207mm、4.115–4.214mm。v4实际末态为thumb4.443–4.769mm、index4.058–4.135mm。上界来自双向顶点到三角面，非完整edge-edge最短距离；正分离下界已足以证明未触碰。

FCL原firsttouch thumb piece29/index piece10以及v4 index piece9的collision=True与独立分离平面、同pair正distance矛盾。无首触主要由旧错误FCL首表面区间提前终止解释；不能归为闭指期间大漂移或仅0.4mm掌残差。保留旧证据，不能再把该binary标签称作实际firstsurface。来源与转换库SHA见representation_source_receipt.json，逐piece证据见piece_consistency.json。
