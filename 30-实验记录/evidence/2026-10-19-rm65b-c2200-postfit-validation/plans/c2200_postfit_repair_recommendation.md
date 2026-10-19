# C2200验证后诊断与下一轮门槛

**当前应先修数据设计与验收口径，不再给现有96条加步数，也不进入3000条训练。** C2200已恢复同分布协议输出，但未学稳记忆操作与角色冲突处理；旧动作闭环还缺正向基线。

## 实测证据

| 验证 | 严格正确 | 说明 |
|---|---:|---|
| 原val | 60/60 | 去重28/28；其中历史未推理的15个不同输入全部正确。仍共享训练的9种目标字符串。 |
| 两种同义goal | 54/56 | READ16/16、SUBTASK20/20；WRITE18/20，两个CORRECT输出破坏JSON。EOS56/56。 |
| 修正schema后的反事实v4 | 40/88 | READ8/8、WRITE16/40、SUBTASK16/40；JSON/schema48/48，EOS85/88。 |
| 只改revision的v5 | 8/24 | JSON/schema/EOS均24/24；8组三元的正确目标不变，模型操作却全部随revision改变。 |

四份报告均记录同一C2200权重SHA，模型冻结参数及144个LoRA张量前后不变、没有梯度或优化器。这里的“语义”主要是形式工具逻辑与左右角色词义代理，不是视觉定位或机器人成功率。

## 分清三类问题

**数据捷径已有局部因果证据。** 独立CPU复核确认v5每组三元只有一个revision数字token变化，图像、state、goal、value、confirmed均不变；8/8组均呈现`1→EMPTY、2→KEEP、4→CORRECT`。fit96中WRITE只配revision0，KEEP只配2，CORRECT只配4，EMPTY配1或3。这支持当前受控上下文中的revision捷径，不能扩展成“模型任何场景都只看revision”。

v4的32个active/revision1 WRITE全部输出EMPTY；SUBTASK在32个非空memory条件下全部跟随存储的左右值，目标冲突的16例因此错误。empty SUBTASK的8例全部发生JSON模式串扰，3例没有EOS。这些是需要保留的条件利用失败。

**评测缺陷与加载问题另记。** 旧v2把SUBTASK的原生`{selected, current_writes}`记忆对象误写成list，其40条不能作为模型失败证据，原报告与勘误应并存。v4已用真实EpisodeMemory及collator/controller对齐验证修复。当前四份语言报告和旧动作C2200的严格加载回执，没有支持“LoRA未加载”的证据；动作回执核实144个adapter逐张量一致、72个B非零。后续入口仍必须保留这份加载证明，不能只凭配置名判断。

**动作基线尚未通过。** 已完成的`H16/K1`无接管测试中，原SFT与C2200在同两种子上各0/2，均未建立有效抬升。原SFT本身失败，因此既不能宣称动作能力保留，也不能归因语言LoRA使动作退化。局部action/KV误差通过只证明固定观察处映射接近。K16及高层实际交互由主流程另行汇总；不能提前用未完成结果补足结论。

## 下一轮按门槛推进

1. **先重构数据，不训练。** 保留真实Memory工具、revision所有权与冲突检查。empty槽的revision0与WRITE关联是合法状态约束，不应伪造active WRITE或随意篡改revision。重点是在每个可达active revision内交叉覆盖正确/错误value、confirmed真假、goal左右及EMPTY/KEEP/CORRECT；用原生WRITE、幂等CORRECT/KEEP及真实读写序列生成可回放状态。不要直接删掉整个revision机制。若将来研究隐藏actor端revision，仅作独立视图消融，完整工具审计和并发校验继续保留。
2. **CPU门槛先过。** 按实际prefix与三相机像素去重、查冲突与split泄漏；逐字段校验READ/WRITE/SUBTASK可见输入等于真实tool/controller/collator。核对时间因果、三元干预仅改目标字段。统计`revision×标签`列联表，验证仅凭revision不能再确定active操作；同时覆盖goal、value与confirmed的反向干预。明确自然post-WRITE样本与stale/empty压力样本；先定义冲突时目标与记忆的权威关系及允许的复核行为，再定标签。
3. **具备前两门后才考虑新的小集拟合。** 预备约96–192个去重输入，数量以交叉格去重结果为准，沿用相同初始权重、参数范围与旧动作锚定。通过线预先写定：自由生成EOS/schema及形式逻辑小集全部正确，两次检查点稳定；revision三元8/8目标正确且不变，角色冲突不盲从旧memory。此项是后续门槛，本轮不执行训练。
4. **再看未见组合。** 将布局/episode/像素组与`goal×value×confirmed×revision×措辞`组合分组留出；各单因素在训练中出现，测试留新组合，避免只换文件名。按不同输入、各操作和最差组分报。现val/反事实已用于诊断，后续作为开发集；原test继续封存，并为新数据设计独立终测。小集通过不替代这一步。
5. **无接管动作基线独立验收。** 首先在原入口复现一个已有成功场景，再对相同实测观察、归一化、固定噪声张量比较原策略与Torch完整action chunk；固定H、执行步数K、时限、场景与24维/三相机/60ms合同后做配对回合。分开报告native判据与抓起/转直/保持判据。原SFT基线须先出现可复现成功；否则优先排查本体映射、权重转换、观察处理、执行协议和仿真场景，不将差异包装成语言学习收益。

以上门槛针对协议条件利用与执行一致性。当前数据没有新执行进度、接触或完成标签，不能据此宣布MEM视频理解、多步骤任务或RL收益成立。

## 证据位置与范围

核查日期：2026-10-19。只读CPU综合分析，未加载模型、未启动GPU、未改工程或原数据。

- `RUN/heldout_val/report.json`
- `RUN/paraphrase/report.json`
- `RUN/counterfactual_corrected/report.json`
- `RUN/revision_only/report.json`
- `/tmp/protocol_cf_revision_independent_audit.json`
- `RUN/old_task_baseline/report.json`与`RUN/old_task_c2200/report.json`

`RUN=/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2/reports/postfit_validation_20261019_v1`。
