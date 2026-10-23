# 无效候选，禁止执行

本目录保留 CPU 脚本负例。实际起点已经是约 57 μm 的 SAT 浅交叠，脚本却仍假设起点分离并二分“零面”。因此 `surface_bracket.disjoint` 实际也是负值，不能构成零面区间。没有把这个候选用于机器人执行。

修正版先承认起点浅交叠，再给有限额外预载；当前交付见 `../v6_single_index_continuation_v3_bounded_upper/candidate.json`。最终仍由实际 4-ms 接触载荷判断，不把 SAT 交叠当受力保证。
