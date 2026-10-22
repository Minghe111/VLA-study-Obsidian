# 115毫米闭盖持瓶：独立离线观察器与动力学审计

2026-10-22。只读原11780个4ms物理步、427条24维命令和冻结source snapshot；没有运行新仿真，没有修改原result.json。原run仍为失败：teacher持握超时。所有13个source snapshot的SHA复算匹配。

## 真正误计的是药片内部接触

冻结labware_task.py第222行用 `'table' in other.lower()` 判断桌支撑。新实体prefix包含tablet，原table_contact_N因此把12颗药片与瓶身的内部碰撞载荷相加。本run最终0.044564826N全部来自tablet_00…11，实际唯一table实体的载荷为0；本run没有发现cap被计入这一支撑项。逐60ms点重现原table值的最大误差为1.19e−8N。这与index碰盖属于两个不同的接触事实。

用真实table身份、原有力/对向/离桌2cm/直立15度阈值重算，最长连续bodyhold为active3909–6405，共2497步、9.988秒；原substring判据最多0.004秒。加上原closed转角/轴移/径向/倾斜限制，最长连续窗口active3909–4717，共809步、3.236秒；另有4973–5115的0.572秒窗口。此为原轨迹的离线阶段证据，不改原在线失败结果，也不构成新控制闭环成功。

## 真实的盖轴移问题仍存在

首次closed=false为assembly10093／active4718，因轴移1.059443mm超过原1mm限制；此时转角0.021895rad、倾斜0.000748rad、径向误差约7.45nm，未release。最终轴移1.044602mm、转角0.048653rad，依然未release。不能借观察器修复抹去该问题。

药片中心仅428个60ms观察点可见，所有点12颗；没有每4ms药片完整pose，也没有本run完整capsule包含证明，因此不把dense closed+bodyhold直接升级为dense filled完整形状成功，更不宣称开盖倒药完成。

## 手指接触与相对滑移

首次真实cap-hand法向力在active4074／assembly9449，为right index_2约0.03549N；active4076首次超过0.05N，约0.13762N。此时瓶身仍由thumb_4约3.10293N、index_2约1.49664N、middle_2约1.28424N支撑（4074）。最终body的各link载荷与cap载荷见JSON，cap主要为index_2的0.67520N。index也是现有瓶身承载者，不能直接假定张开index不影响持瓶。

从60ms完整机器人状态计算inv(palm)·body，相对active3840参考：首次cap接触前4065已平移2.933mm／旋转1.442度，4080为3.280mm／1.478度；最终6405为11.900mm／4.341度。先滑移后碰盖与几何干涉解释相容，但没有通过干预实验单独证明因果。

## 角速度积分没有被端点或投影修正解释

用完整worldω转到body坐标，严格使用swing–twist导数：q=qb⁻¹qc，w=Rbᵀ(wc−wb)，theta_dot=wz+[wx(qw·qy+qz·qx)+wy(−qw·qx+qz·qy)]/(qw²+qz²)。active期pose的theta增量为0.04866643rad，pre左矩形积分0.52724169rad、post右矩形0.52718185rad、梯形0.52721177rad，残差均约−0.4785rad。pre/post端点差仅5.984e−5rad；swing修正最大每秒2.222e−4rad，积分绝对上界0.00570rad，均不足解释残差。

所有记录的pre[k]与post[k−1]的pose、线速度、角速度逐字值相同，步号与active tick无缺漏。该证据排除了可见的步间状态跳变，但不能检查PhysX接触约束内部的位置/速度更新次序。与无接触台架不同，这里不能凭unwrap正确就宣称速度一致，也不能凭残差认定theta计算错误或存在写pose。后续若要归因到接触解算修正，需独立受控诊断，不应改动被动约束来掩盖。

## 控制链

427条state/action/next_state均有限24维，每条15步，实测state链精确连续。每4ms记录的action_index与所属60ms行一致；requested24匹配该行endpoint。内部drive按原clock从上一drive线性插值，不能把所有4ms值与60ms终点直接判等。实际endpoint与保存action、后426段内部drive与按原clock公式预测的最大差见JSON；首段起始drive未单独保存，故只检验该段endpoint。

本审计新增物理trial数为0。空瓶成功、被动组件释放、当前闭盖持瓶阶段、未来完整双手开盖倒12片是不同范围，任何一项均不替代另一项。
