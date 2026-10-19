# Conditional D: attention LoRA plus the final language norm

Status: reference design only, NOT applied. Parent reports C2000 already fit all 96; therefore D is NOT triggered. No GPU model load or training performed for this review. Only the existing tiny 4-layer/64-width Gemma implementation was executed on CPU without an optimizer to verify the dependency boundary.

Scope, if future evidence triggers D:

- New independent stage `language_readout`; keep existing `language_warmup` and all old entry points unchanged.
- Exactly the existing 144 language attention A/B tensors, 3,686,400 scalars, plus the existing `llm.final_norms.0.scale`, shape [2048], 2,048 scalars. Total 145 tensors, 3,688,448 scalars. This scale already exists in C2000 `full_weights.pt`; there is no new randomly initialized layer or vocabulary head.
- MEM bypassed; vision, history projection, all MLPs, all other norms, tied embedding and action expert frozen. Keep SDK24, three cameras, 60 ms, H16/K_exec1.
- High-level CE updates attention LoRA and final language norm. Native KV loss updates attention LoRA; final language norm has no KV gradient. No FM, FAST or value update in this diagnostic.

## Actual dependency boundary

`modules/gemma.py:637` collects all layer K/V caches before applying expert-specific final norms. `pi0.py:476` returns the final language hidden state and the earlier cache separately. The continuous expert consumes per-layer K/V, not language final hidden states, for ordinary fixed-prompt flow inference.

Thus changing only final_norms.0.scale changes language logits but leaves prefix K/V and fixed-prefix expert suffix outputs unchanged. CPU proof `/tmp/final_norm_cache_cpu_proof.json` verifies all three on the real tiny Gemma implementation, plus finite nonzero CE gradient to the scale. This is structural component evidence only; it does not prove jointly changing attention LoRA preserves actions, nor that a changed generated subtask remains correct. Any value/RLT consumer of final language hidden states would also see changed features, so the result must not be generalized to such branches.

## Integration points and traps

1. `recap_lora.py:138 configure_mem_parameters`: add the new stage to the allowed and language-only sets; select the exact final-norm key additionally. Do not unfreeze every norm or call it LoRA. Retain all current adapter module/key layouts.
2. `tasks/recap_hier.py:143 kv_anchor_loss`: currently rejects any trainable shared base tensor that is not an adapter. Without a targeted exception the new stage fails immediately. Exempt ONLY the exact final language norm key in the new stage, justified by the pre-final-norm KV boundary. Keep rejection for all other base tensors. Teacher disabling adapters still yields original K/V even when this irrelevant-to-KV scale changed; do not describe resulting final hidden outputs as original-policy outputs.
3. `__init__.py:358 _apply_openpi_param_dtypes`: critical precision trap. Current mixed cast first rounds all `llm` tensors to BF16, then converts norms to FP32. A new trained FP32 scale would lose small updates on reload unless snapshotted before casting and restored afterward. Add the new stage to adapter preservation and preserve the exact final-norm FP32 tensor too. Simply keeping its final dtype FP32 is insufficient. Test values that are not BF16-representable through save, full load and dtype conversion.
4. `scripts/fit_comparison.py:50 audit_model`: expected 3,688,448 and allow exactly this one non-LoRA key. Other selection checks remain strict. CLI/build_config must recognize new stage; KV lambda remains enabled for it. Native backend's text-only-stage FM prohibition must also include it.
5. `recap_training_manifest` already records stage, every trainable tensor name/shape and MEM config. The new stage naturally adds one entry. Preserve old manifests byte-for-byte in structure so old exact resumes keep working. Do not weaken `restore_trainer_state` equality to accommodate stage changes.
6. Compact adapter saves currently omit all base norm tensors. For a minimal D experiment, require `full_weights.pt` only and explicitly reject compact adapter save/load for this new stage; avoid silently losing the norm. A later compact format would need versioned norm inclusion and an exact base checkpoint reference. Do not globally reclassify this base norm as an adapter, which would break old compact-checkpoint expected keys.

## Controlled cross-stage initialization

- Use a named, explicit weights-only transition from source `language_warmup` C2000 to target `language_readout`. Stage changes must NEVER be labeled exact resume.
- Require source full checkpoint, source version/training manifest, audited same 96-row label SHA/provenance, same architecture/rank/alpha/tokenizer/camera/state contract. Source and target state_dict key/shape layouts must be identical; only requires_grad membership differs.
- Keep `load_full_weights(strict_recap=True)` with zero missing/unexpected tensors. Missing old LoRA, missing base final norm, partial adapter collections, unrelated extras or a compact-only source must fail. Never fallback to original SFT on failure.
- Explicitly validate the source→target stage pair in the weights-only path; reject any other transition. Current `validate_fork_checkpoint` checks label SHA and size only, so stage semantics need an added guard. Generic evaluation/inference should require the saved new stage config; do not silently load a trained norm under a mismatched old-stage config.
- Load all weights first, preserve exact FP32 LoRA/norm values during mixed cast, then verify D-at-zero-update equals source weights with a full state hash independent of requires_grad grouping. Frozen-only digest from the old stage cannot be compared directly with D: the norm moves from the old frozen set to D's trainable set.
- Reset AdamW, sampler, RNG and branch update count to zero, recording source checkpoint step separately. Run the old attention-only comparator from the same source checkpoint with the same reset convention and budget.
- Once a D checkpoint exists, D→D exact resume must restore its 145-parameter optimizer manifest, sampler/RNG, LR/KV/accumulation/clip contract exactly. D↔old-stage exact resume must fail.
- Training CE, full teacher forcing, cached generation and inference all use the same existing final norm and tied embedder.decode. Save/load reload equivalence must include greedy JSON/EOS outputs, not just parameter existence.

## Required CPU tests before any conditional GPU run

1. Production meta shape: exactly 145 trainable tensors/3,688,448 scalars; exact key set; vision/MEM/expert/embedding/other norms frozen; MEM runtime false.
2. Switch old→new→old stage without state key changes or resetting existing LoRA/base/norm values; unchanged old stage parameter selection and manifest.
3. CE backward reaches final norm and early/late language attention; frozen grads remain absent. Native KV backward reaches LoRA but not final norm; arbitrary other base unfreeze remains rejected.
4. Mutate only final language norm: language logits change, every prefix K/V and fixed-prefix action-expert suffix remain bitwise equal on CPU. Existing `/tmp/probe_final_norm_cache_cpu.py` demonstrates this boundary.
5. Full checkpoint with nontrivial LoRA AND non-BF16-representable final norm loads exactly after dtype conversion; missing LoRA/base norm, extras, wrong rank or stage fail. Compact D save/load explicitly fail.
6. Weights-only source old→new preserves all tensor values and resets optimizer/sampler/RNG; manifest-required exact resume across stages fails, while a D→D stochastic exact resume remains identical.
7. Mixed dtype training/inference use FP32 scale storage with existing BF16 compute outputs, consistent decoding. No alternate JSON repair, EOS injection or replacement language head.

Current decision: C2000 passed, so keep D as an unapplied contingency. Confirming the existing configuration takes precedence over parameter expansion.
