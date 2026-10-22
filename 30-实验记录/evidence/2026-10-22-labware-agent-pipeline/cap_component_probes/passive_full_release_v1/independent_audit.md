# 被动组件完整释放台架：独立读回审计

日期：2026-10-22。仅检查 `receipt.json`、两份完整trace和冻结源码，没有运行新仿真、GPU或修改旧receipt／源文件。**这是0.003Nm对偶外加扭矩的机械组件台架，不是机器人拧盖、移盖、倒药或示范数据。** 全部精确数字、SHA与检查项见 `independent_audit.json`。

## 释放与动态合同

两个dt均逐步连续、trace SHA匹配。4ms：第360步释放，力矩实际施加110步（名义0.44秒），θ=15.89204479rad、轴移11.90068793毫米。1ms：第1440步释放，施加440步（名义0.44秒），θ=15.67818115rad、轴移11.73892838毫米。每组前一步尚不满足双阈值，首释放步同时满足θ≥15.60796327rad和轴移≥11.5毫米，之后released不反转。

释放之后所有外加扭矩为0，没有holder，瓶盖按自由动力学下降至world z约13.42／13.45毫米。4ms后续实际188步即0.752秒，1ms750步即0.75秒；不把round造成的差异写成完全相同的持续时间。

实际合同读回body／cap为dynamic、12份药片质量均约0.00035kg、每步12个独立药片local position；源码为12次不同builder.build默认动态刚体构造，未给药片加固定约束。合同未逐片列kinematic flag，所以逐片非kinematic还依赖静态构造审计，而不是声称已有该项逐片日志。两径向导向所有motor参数实际读回均为0；源码没有后续启用motor，释放只disable导向。bench没有pose／velocity setter调用，组件只有初始化builder pose，未发现初始化后状态写入。

## 角速度残差的可解释部分

台架明确同时保存pre投影ω与post完整world ω，因此可在同一时序重算：

- 4ms力矩段：θ增量15.89204612rad。pre左矩形积分残差0.28575857rad，post右矩形残差仅0.00013698rad，梯形残差0.14294777rad。
- 1ms力矩段：θ增量15.67819503rad。pre残差0.07120879rad，post残差0.00004644rad，梯形残差0.03562761rad。

pre[k]与前一post[k−1]的最大差约3.8e−6rad/s。pre／post积分差基本就是dt乘末首速度之差，符合此运行的半隐式位置更新时序；改用post速度后大部分累计残差消失。梯形只减半，在dt缩小4倍时残差也约缩小4倍，与一阶端点采样误差一致；只有两个dt，不能据此声称一般收敛阶或完整能量守恒。补上完整swing-twist横向项对本台架积分几乎不改变，说明这里不是主要遗漏了随动项。

释放后组件故意冻结“螺纹转角”状态，不能继续拿这个冻结θ与实际自由旋转ω积分比较。台架的功由施力段τ×真实θ增量计算，独立重算与receipt一致；它不是无驱动自行开盖。

## 不能替代旧机器人失败解释

旧机器人强接触失败中θ净+15.615rad而pre ω积分−9.784rad，差约25.4rad；仅移一个端点采样约0.06rad不能解释该量级。因此，本台架解决了平滑扭矩情况下的pre/post采样残差解释，**没有证明旧手指碰盖过程正确，也没有证明机器人已能拧盖**。旧失败仍应由115profile的同tick完整world ω、pre/post pose、接触冲量／力矩日志独立复验。

## 115profile日志需求

每4ms保存：actual dt、global/active tick、phase与SDK24 action index；body/cap完整world pose（wxyz）、cmass local/world pose、pre/post完整world线速度和角速度；相对pose、wrapped/unwrapped θ、pre/post轴向投影和完整twist导数；逐link接触point/normal/normal与tangent impulse、作用臂及总力矩；螺旋／摩擦／座止／外加载荷和释放判据；尽量同步实际q36。保留pre左、post右与梯形三种诊断，而不是只保存一个混名scalar。闭盖失效和意外盖接触应独立触发guard，不弱化成功判据以通过。
