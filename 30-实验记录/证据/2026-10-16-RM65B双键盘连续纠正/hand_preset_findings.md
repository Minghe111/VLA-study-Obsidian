# RM65-B hand preset source audit (read-only, 2026-10-16)

## Contract and ordering
`[left_arm_rad6, left_hand_sdk6, right_arm_rad6, right_hand_sdk6]`. Per hand channel order is `[little, ring, middle, index, thumb_flexion, thumb_rotation]`; see `code/RoboTwin/integrations/realman_rm65b_gen4/build_model.py:74,121,130` and `tests_acob/fixtures/realman_source_manifest.json`. Left and right use same SDK meanings and limits. `envs/robot/realman_rm65b/controller.py:15-32`: `q=lower+(1-SDK/1000)*(upper-lower)`. First five channels 1000 = extended/open and reducing SDK bends fingers. The sixth channel is rotation, not additional closure; SDK0 maps q=1.1641rad, SDK1000 maps q=0.

## Existing source values
All paths below are relative to `/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1`.

- Original open source: `integrations/realman_rm65b_gen4/acob_expert.py:34-40` OPEN = `[1000,1000,1000,1000,1000,0]`. This equals native robot DEFAULT_SYNERGY open. Native generic close `[0,0,0,0,0,1000]` is NOT the bottle pose and must not be used as a bottle close target.
- Upright bottle search template: `acob_expert.py:38-40` close_sdk = `[900,900,608,610,610,0]`. This is the free closing search endpoint. Actual contact controller latches contact and adjusts target; it does not simply reach this vector and guarantee grasp.
- Lying bottle template: `acob_expert.py:35-37` = `[650,670,610,610,887,0]`; not appropriate as universal upright bottle pose.
- No previously validated independent middle `pregrasp` hand preset exists in these files. `READY_NEAR_SDK6` in historical helper is for another geometry/template, not current upright bottle success.

## Actual successful evidence (do not conflate drive/measured)
`data/expert_demo_v7/dataset.json`: seed2010, actual success=true, completed=true, right arm, 245 rows, upright_hold; terminal lift gap0.027656m, angle0.599deg, hold1s.
`data/expert_demo_v7/rows.jsonl` final index244 `actions[0][18:24]` actual right hand commanded drive target:
`[900,900,698.4251708984375,708.1975708007812,736.367431640625,0]`.
Same row measured-before right hand:
`[900.000061,900.003967,701.394897,710.645691,767.723877,0.930098]`.
These differ and measured values must not be substituted as commands. The success followed dynamic contact feedback, not this fixed hand vector alone. This evidence proves the right hand only. Left has symmetric model mapping, but same fixed preset has not been independently demonstrated successfully here.

## Recommended implementation design (new, not preexisting calibrated data)
Use source upright template for nominal grasp `[900,900,608,610,610,0]`, source open above, and new middle pregrasp = arithmetic half closure `[950,950,804,805,805,0]`; label middle and reuse to left as new design. If lower overclosure is desired, an alternative is source successful right drive rounded `[900,900,698,708,736,0]` with middle `[950,950,849,854,868,0]`. Do not claim either fixed choice is contact-calibrated. Parent should select one consistently and record source in config.

A preset key selects an intended SDK6; pressing/releasing movement keys should not unexpectedly change the selected hand pose. A separate hold-current cancels interpolation. Interpolate from actual current drive SDK target, not measured clipped state, with max abs increment50SDK per60ms. Keep thumb rotation separately fixed at0 for these bottle presets; transition only if current drive rotation differs, using same cap. Use +/- closure trim on the first five channels if desired, preserve sixth. Each side independent; only an active human-owned side may receive human preset commands, and all intermediates must be recorded as human replacement commands. Never re-label former episodes.

No simulator/GPU/robot execution or code edits were performed by this audit.
