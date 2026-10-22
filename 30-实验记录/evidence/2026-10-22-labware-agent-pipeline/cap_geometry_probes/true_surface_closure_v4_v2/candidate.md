# v4实际末态的唯一真实表面续闭候选

CPU candidate_passed=True；SDK6从[0.0, 0.0, 0.0, 560.09765625, 1000.0, 348.443359375]到[0.0, 0.0, 0.0, 536.0999393463135, 1000.0, 190.54951238632202]，依次仅index(channel3)与thumbopp(channel5)。

未改变palm、物理参数、抓握姿态族或实际状态。使用真实PhysX碰撞顶点的完整SAT而非FCL hit；保留零表面bracket与至多一SDK unit/50μm负投影余量。全部forbidden body/right/table/self与其它手指cap在≤0.25mm顶点移动界采样，并记录保守区间分离下界。

该候选只供下一次精确回放后的低速限力续闭；每4ms真实法向力与actual geometry守卫优先，不表示机器人接触或开盖已通过。
