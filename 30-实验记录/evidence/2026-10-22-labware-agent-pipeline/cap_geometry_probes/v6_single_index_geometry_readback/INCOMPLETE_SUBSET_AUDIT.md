# 不完整的局部网格诊断

该次一次性诊断只读取 `PhysxCollisionShapeConvexMesh`，漏掉瓶盖的 Box 碰撞件，所以其中最小约 0.452 mm 的值不是“全瓶盖到食指”的间隙，不能用于续闭规划。

完整 33 件瓶盖与食指父 link 的实际局部几何一致性复核见 `../v6_single_index_continuation_v3_bounded_upper/actual_geometry_identity_receipt.json`：局部顶点一致性误差小于 1 nm，全盖最小 SAT 投影约 -56.87 μm，最近为盖第 9 件。
