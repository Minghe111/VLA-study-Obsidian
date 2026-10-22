# 115毫米双手开盖／倒12片：完整采集接口合同

2026-10-22，只读源码核查与未来接口设计；不是full task实现或验收。不修改已通过的空瓶入口、旧数据、active profiles或官方源码。建议独立task_name为 `labware_cap115_unscrew_pour`。

## 可以复用什么

原 `scripts.collect_data.run` 的外层顺序可完整保留：setup_demo→play_once→plan_success/check_success→关节限位→存trajectory/seed；然后同seed reset→load_tran_data/set_path_lst→真实play_once重放→语言→close_env→merge/export→cache清理→再次check_success。下层由独立full task/collector hooks承接，不把14维通用导出器当成SDK24。


- `setup_demo(now_ep_num,seed,**args)`：Official planning/capture orchestration; separate run dirs; deterministic reset；full task要求：Create115 assembly, twelve dynamic tablets, receiving dish, both robot hands and profile-specific annotations; identical initial-state fingerprint for plan/replay.
- `play_once() -> info`：Original run calls it in both planning and replay；full task要求：Single central replay gate executes every saved SDK24 row once; otherwise actual stage expert; return multi-actor info; never stop after body-hold.
- `plan_success / check_success()`：Original seed gate and post-merge assert；full task要求：Only final task return true. Preserve terminal metrics after close_env. Body-hold/open-cap are stage milestones, not final success.
- `planned_joints_legal()`：Left6/right6 slices plus all12 hand channels bounds；full task要求：Check complete dual-arm sequence; no joint-limit or SDK-range weakening.
- `save_traj_data / load_tran_data / set_path_lst`：Original plan→seed.txt→physical replay architecture；full task要求：Save actual full SDK24 rows/phases/command-intent fields; recompute cap release, containment and success during replay; never copy outcome flags.
- `on_transition / capture_transition`：Actual measured before/next states and drive endpoint readback；full task要求：Every stage routes through same60ms action clock; carry phase and full physical metrics including tablet identities/events.
- `merge_pkl_to_hdf5_video / remove_data_cache`：RGB codec,SDK24 export/readback,success-only cache cleanup；full task要求：Require final full-task receipt and full-language schema; retain failed/replay disagreement artifacts.
- `close_env`：Original run cleanup order；full task要求：Original capture closes before merge and check_success; immutable final metrics/receipt must survive removal of actors/assembly.

## 必须另立入口的硬编码


- `collect_official.py:29-43`：当前 Exact AST template and class hardcoded to labware_pick_place；需 New isolated full collector validates exact wrapper for declared new task and tested full candidate.
- `collect_official.py:54-59,66-102,137-140`：当前 CLI only empty reagent_bottle_body/right; task module, generated class and receipt task_name fixed；需 Newfull task_name/module/factory/both-hand role binding; never reinterpret empty10/10receipt as full-task acceptance.
- `collect_official.py:112-130`：当前 Replay provenance checks only empty task/expert and asset/arm；需 Bind profile115/spec/assembly/full expert/capsule geometry/seeded contents/dish/metadata/camera/clock/source hashes and initial-state fingerprint.
- `collect_official.py:229-235`：当前 Language info must be exactly {A,a}；需 New schema {A,B,C,a,b} for body/cap/dish/right/left; original language filter exact nonarm placeholder match.
- `collect_official.py:249-261`：当前 source_snapshot only five base code files plus optional framing；需 Extend full dependency closure; snapshot actual generated expert,receipts,full task/profile/spec/metadata/geometry/capsule predicate/native contracts.
- `agent_runtime.py:session_test,session_postprocess`：当前 Most names dynamic, but task_implementation_sha256 uses fixed ROOT/labware_task.py androbot_expert.py;default config empty-only helper；需 New isolated config/TaskInfo/full factory and explicit verified local APIs; configurable complete source manifest; no reference fallback.
- `task_config.yml / collect_official.py:137`：当前 YAML describes empty success; collector constructs args inline and does not read this YAML；需 New actual-consumed full config + resolved runtime args snapshot; do not claim changing YAML alone configures collection.
- `labware_cap_task.py:135-159;labware_cap115_task.py:136-158`：当前 closed_fill_hold>=.5 sets metrics.success and check_success terminates teacher while-loop；需 Keep in stage_success only; override final predicate/loop so no early stop when merely holding closed bottle.
- `labware_task.py:grasp_lift_place`：当前 Each helper can consume all replay_commands for the entire empty task；需 For multi-helper full play_once, one central replay_episode branch; never replay whole episode once per helper.

## 最小task接口与final-only成功

factory返回严格的full生成类与args。task至少暴露scene/robot/bottle/cap/dish/tablets/assembly/clock、真实physics_step_count、transition_log/physics_trace、metrics/info、phase/stage_success/final_success；保留setup_demo、play_once、apply_sdk24、capture、get_state24、observe_rgb、advance和check_success。capture输出24维state、physics_tick、instruction，以及需要记录时的三路480×640×3 uint8。

body_hold、left_cap_grasp、release、cap_clear都是阶段状态。`metrics.success`／`check_success`只能代表最终全流程完成；不能继承当前closed-hold在0.5秒时就返回成功的逻辑，否则Clock/teacher while-loop会早停。原run在capture后先close_env再merge并再次check_success，因此最终物理指标与receipt必须在close后仍可读，不能依赖已释放的live actor。

最终判据应冻结为：机器人实际支撑瓶身和操作瓶盖；按实测转角/轴移释放；盖真实清离瓶口并处于明确约定的安全持握或放置状态；原12颗独立capsule全部以完整几何处于接收皿内、瓶内0、其它/地面0并稳定一段实际时间，无guard/failure。可参考root新增capsule_geometry.py，不能只用12个中心点代替完整形状包含。具体阈值和盖末态须在实现时显式冻结，这里不虚填成功。

## 重放不能借用planning结果

full play_once需要一个中央replay分支，每条SDK24记录只执行一次；不能让多个stage helper各自把整段replay_commands重放一遍。阶段标签可以是命令意图，cap.released、药片数、真实抓持和final success必须从当次重放物理重新观察，不能从planning row复制。计划与capture应有相同seed和完整初始状态指纹（robot/body/cap/12药片/dish/被动机制状态）；版本/初态不同就拒绝偷用旧轨迹。

## 不允许退化的出口与来源

保留 `[左臂6,左手6,右臂6,右手6]` 的24维实测state、真实下发及drive读回action、next_state；每动作15×4ms=60ms，三路RGB、时间链、实际drive记录和逐阶段物理指标。动作不是next_state，图片不能占位或复用旧图。若将来需要1ms物理步，必须独立版本化时钟、步数、导出和验收，不能仍标15×4ms。

source_snapshot必须扩到full task/expert、115 spec/profile/matrices/visual与碰撞来源、冻结assembly、capsule_geometry、native clock/SDK/physics/camera合同、新TaskInfo/points_info/对象和任务语言、实际生成专家与parser/physical receipts。当前empty固定5文件snapshot不足。小文本源逐字节快照；大型已有几何保留不可变来源和SHA，避免未版本化路径。

## 语言和115 profile绑定

新info["info"]建议精确为{A,B,C,a,b}：115 body／cap／petri_dish描述路径、right／left。原语言函数本来支持多对象，但filter要求所有非arm占位符精确匹配；模板需A/B/C完整且同时使用a/b，不能额外塞phase等非语言字段。新12seen/12unseen仅为措辞split，不是新物理泛化成绩。

初始核查时，115 body/cap已有新model_data及visual/spec，但没有独立Agent aliases/points_info和语言注册。本轮已在 `full_agent_metadata/` 独立补齐：原enrich_actors三对象通过，原语言函数生成12 seen与12 unseen且两臂占位符完全解析，源文件SHA未变。该overlay不修改profiles目录，不使用旧90毫米modelname；body开口局部z=.115、cap装配closed origin=.118只在装配变换中使用，不能把.118写进盖子局部点。petri沿用已有实测资产ID。新body/cap仅注册Agent语义，物理加载须走Bottle115Assembly；未复制旧90碰撞、不宣称generic create_actor可用。geometry loader、机器人完整任务、专家生成和采集均未在此组件测试中执行。

本合同仅为接通准备。真正fullphys成功、full Agent测试、原collect规划＋capture、完整HDF5读回全部通过后，才可宣称full采集就绪。
