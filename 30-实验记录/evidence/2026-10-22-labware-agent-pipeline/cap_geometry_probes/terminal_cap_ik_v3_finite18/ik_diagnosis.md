# 实际terminal盖位的有限IK初值复核

快照physics_tick=7575；相同目标18个初值（实测current、原final、16个±0.04/0.12rad有限小扰动），合法收敛0，端点CPU几何筛查通过0。

仅左臂6关节可变，右臂及全部手指保持actual q36；原始solver error、FK位置/角度残差、冻结关节变化和jointlimits逐次保存，未clip。未扩展姿态族、未搜索path、未物理执行。

若存在preferred，left_arm6和returned_q36仅是IK候选，仍需actualscene规划与碰撞审计；这里不代表末段接近或真实首触成功。
