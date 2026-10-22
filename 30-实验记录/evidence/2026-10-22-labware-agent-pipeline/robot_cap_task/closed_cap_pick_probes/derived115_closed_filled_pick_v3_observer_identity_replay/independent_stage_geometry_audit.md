# 115毫米闭盖持瓶v3：独立完整胶囊与真实重放审计

2026-10-22。CPU只读审计本run已关闭的完整gzip、冻结snapshot及实际末态；审计者新增仿真trial为0。这里的通过仅为右手闭盖持瓶子阶段，不是开盖、倒药或官方完整数据采集成功。

## 真实命令和状态与v2前缀完全相同

重新比较269条24维state/action/next_state及tick，均与原失败v2前缀逐项相等，未新规划。总物理步9410，其中初始化5375步、active4035步；每动作15步，24维有限，state链精确连续。4ms context与60ms action行对齐，requested endpoint逐字相同，实际drive终点与命令、后268段内部drive与原clock插值预测最大误差均4.375e−5。首段起始drive未单独保存，只检验该段终点。

所有相邻pre/post的body、cap pose与速度及12颗药片pq逐字相同，物理步和active tick无缺漏。13个source snapshot SHA复算全部通过。terminal的24state、q36、qvel36、root/body/cap矩阵与12药片pq全部和最后实测记录逐项一致。原result、source snapshot及v2失败记录未更改。

## 支撑和闭盖阶段

contacts_post保留actor ID与name，唯一support为per_scene_id=6的table；名称与ID每步一致。持瓶窗口真实桌支撑为0，active期间cap-hand力始终为0。按原离桌至少2cm、对向接触、直立小于15度、闭盖转角/轴移/径向/倾斜及12中心规则重算，最长连续active3909–4035共127步，名义0.508秒，与在线stage一致；另一较短窗口为3826–3879的0.216秒。

## 完整胶囊包含须说明接触容差

使用原capsule_geometry.capsule_cylinder_margins，对每个实际4ms步的pre/post各12颗完整pq进行变换。胶囊半长2mm、半径3mm；瓶内半径21mm、底面z=3mm、瓶口z=115mm，均来自冻结spec。没有仅靠中心计数。

active全部4035步的pre/post，12颗完整胶囊在原spec contact_offset=0.4mm的显式接触容差下均包含，闭盖＋持瓶＋此完整包含的最长窗口同样为0.508秒。原始最差径向margin为−86.82微米、底面margin为−23.31微米；因此不能写成零穿透、零容差包含。零容差时最长连续整形包含持瓶窗口仅0.008秒，未达到0.5秒。初始化自然settle中最差floor margin为−0.6657mm，也单独保留，不能和active混写成全程无越界。

0.4mm是冻结仿真接触参数，未根据结果拟合，也未改变在线判据；它不等于硬件几何校准。在线stage仍采用中心计数，本审计只是补充完整形状的离线界限。未来full task须在采集前明确并冻结最终几何容差及稳定时长。

## 末态不能直接作为静止双手交接参考

terminal在lift阶段，瓶底离桌27.816mm、体速29.812mm/s。它是实际的运动中stage末态，不能用静态IK工作区检查代替后续稳持。仍须真实保持、取得静止后的左右palm/body/cap/药片和碰撞参考，再验证左手接盖。没有任何开盖、倒12片、接收皿稳定或完整采集成功在本run中发生。

数值与SHA明细见independent_stage_geometry_audit.json，核心离线计算源见同名.py。
