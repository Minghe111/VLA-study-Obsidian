---
type: independent-source-review
updated: 2026-10-22
scope: right-index-bookkeeping-overlay-only
tags: [RoboTwin, RealMan, controller, evidence]
---

# 右 index 有限诊断的控制器状态复核

本记录为只读源码审查，不是物理试验、抓持或卸压成功证明。原 native `BoundaryContactController.target()` 会修改 filtered、真实接触触发的 latch、latched_q、free_search_q、计数和 previous_target；其中 previous_target 仅写入，不参与下一次目标计算。因而只覆盖 previous_target 不能解决被丢弃 index 提案的状态漂移。

对 `right_index_unload_probe.py` 版本 `2ad157ee…` 的实际源码复核表明：每窗先从真实已完成 drive 同步 index 的 latched_q/free_search_q/previous_target，再调用原有效 target 一次，仅替换 index 命令；未执行的原 index 提案随即从 bookkeeping 清除。其余 middle/thumb 等原反馈提案完整保留，latched 布尔、filtered 和 evidence 没有被人为赋成功。完整 15 步执行后才用实际 drive confirm，partial 不能作为完成 commit。

版本同时绑定候选有限预算、hold reference SHA、实际基准 right SDK，并单独验证新合成命令的 primary/mimic 速率。原始实测 SDK 微负值只作为读回保存，未被 clamp 成虚假实测，也未混作严格新命令验证。未发现妨碍该有限可逆物理诊断的确定接口错误，没有追加 CPU 测试或额外物理门槛。

随后 owner 仅调整诊断停止规则并启动冻结版本 `0ace18b3…`。该版本的真实接触、保持、whole capsule 包含和终态尚须依据其实际日志独立审计；源码与状态接线自洽不等于卸压、开盖、倒药或采集全流程完成。

追加源码更正：外层设置 direct targets 后调用 `super(carry=False)` 仍可能进入原生已接触配对的 LP 分支；usable allocation 会在本次 SDK 生成前覆盖 force_targets。外层返回前 reset 不撤销已生成命令。因此“LP 仅 advisory”的既有 source receipt 文字不能证明严格执行隔离。本审查关于 native 提案保留、真实 filtered 和 index commit 的结论不受影响。
