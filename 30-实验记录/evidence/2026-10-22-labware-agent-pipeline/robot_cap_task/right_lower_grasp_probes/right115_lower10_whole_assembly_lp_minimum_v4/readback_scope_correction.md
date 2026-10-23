# 独立读回范围修正

初次 `independent_assembly_lp_readback.json` 保留为审计开发证据，不作为当前控制链或禁碰结论。

- 原生 action clock 在每个60ms窗口内用15个4ms步线性插值驱动目标。不能将每个中间驱动直接与窗口终点action比较；修正版按原生插值重建，并单独核对最终驱动。
- 名称中的 `closed_tablet_bottle_body` 含有 `tablet`，不能将这个子串当作药片身份。修正版只按精确桌名和瓶身名称后缀读回已有body-table接触。
- 此dense接触trace仅保存与瓶身／瓶盖相关的接触，不足以独立证明其他机器人接触全部为零。实时任务守卫另行保留，但不冒充这份trace的可观测范围。

当前读回入口为 `independent_assembly_lp_readback_v2.json`。原始实验数据没有改写。
