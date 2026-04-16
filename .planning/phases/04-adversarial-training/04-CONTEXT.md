# Phase 4: Adversarial Training - Context

**Gathered:** 2026-04-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a warm-started adversarial-training checkpoint for AWN on RML2016.10a.
The deliverable is `./checkpoint/2016.10a_AWN_at.pkl` plus a JSON config of the
hyperparameters used, trained with mixed FGSM/PGD/EAD-L1/EAD-EN attacks (CW
held out) and a clean+adversarial loss that preserves analog-modulation
accuracy. Evaluation of that checkpoint is Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Script Architecture
- **D-01:** New standalone script `adv_train.py` at the repo root, parallel to
  `synth_finetune.py`. Reuses `data_loader.Load_Dataset`, `data_loader.Dataset_Split`,
  `util.utils.create_model`, and `util.evaluation.Run_Eval`. Do NOT add a new
  mode to `main.py` (already 800+ lines) and do NOT extend `synth_finetune.py`
  (two distinct experiments with different data paths).

### Attack Generation During Training
- **D-02:** `ta_box="minmax"`, `eps=0.1` for all attacks. Matches the v1.0
  evaluation configuration so AT is trained against the same adversarial
  distribution the paper reports on.
- **D-03:** Attack iteration budget — FGSM=1 (free), PGD=7, EAD-L1=7, EAD-EN=7.
  Madry-style fast-AT budget; ~8× forward/backward per adversarial batch.
- **D-04:** Per-batch uniform random attack selection from
  {FGSM, PGD, EAD-L1, EAD-EN}. CW is NOT in the training mix — it is strictly
  held out for Phase 5 evaluation.
- **D-05:** Reuse the existing `Model01Wrapper` + `iq_to_ta_input_minmax` /
  `ta_output_to_iq_minmax` adapters from `util/adv_attack.py` — no new
  normalization code.

### Loss and Clean/Adv Mix
- **D-06:** Dual-batch weighted loss:
  `L = α·CE(model(x_adv), y) + (1-α)·CE(model(x_clean), y) + regu_sum`.
  Two forward passes per batch on the same labels; AWN's internal
  regularization (from `regu_details`, `regu_approx`) added once per forward
  (detail). Dual-batch was chosen over split-batch so the clean loss weight
  is a pure hyperparameter, not a function of batch composition.
- **D-07:** Alpha fixed at 0.5. No warmup, no sweep. The JSON config records
  this so ablation sweeps are a future-milestone concern.
- **D-08:** Analog modulations (WBFM, AM-DSB, AM-SSB) are kept clean in the
  adversarial stream — when a sampled label ∈ {analog}, substitute the clean
  input for the adversarial input before the adv forward pass. Implements the
  AT-03 "prevent catastrophic forgetting of analog modulations" requirement
  directly; also aligns with the paper threat model which does not claim
  attacks against analog mods.

### Training Data Scope
- **D-09:** Train only on SNR ≥ 0 dB from the RML2016.10a train split.
  Rationale: Phase 5 reports at SNR ∈ {0, 6, 12, 18} dB (ATEVAL-01) and
  attacks at very low SNR are poorly defined. The warm-started checkpoint
  already carries low-SNR behavior from v1.0 training.
- **D-10:** Validation split follows the same SNR filter (stratified
  train/val split, 85/15). Test-set evaluation at full SNR range happens in
  Phase 5.
- **D-11:** Do NOT mix in synthetic bursts from `synth_finetune.py`. Two
  confounded variables would muddy the AT ablation.

### Checkpoint Selection & Budget
- **D-12:** Best-epoch criterion is a weighted validation metric:
  `0.5·val_clean_acc + 0.5·val_robust_acc(FGSM)`. FGSM is a cheap robust proxy
  on the val set; CW stays held out for Phase 5. Val loss alone is NOT used.
- **D-13:** Max epochs = 30, early-stopping patience = 8. `ReduceLROnPlateau`
  (factor=0.5, patience=4) on the weighted val metric. Initial LR = 1e-4
  (same as `synth_finetune.py`), optimizer = Adam, batch size = 256.
- **D-14:** Warm-start weights loaded from `./checkpoint/2016.10a_AWN.pkl`
  before the first optimizer step (AT-04).

### Artifacts and Logging
- **D-15:** Checkpoint file: `./checkpoint/2016.10a_AWN_at.pkl` (AT-02).
- **D-16:** Config file: `./checkpoint/2016.10a_AWN_at.config.json` (AT-05) with
  keys: `epochs_total`, `epochs_trained`, `lr_initial`, `batch_size`,
  `ta_box`, `eps`, `attack_iters` (per-attack dict), `attack_list`,
  `alpha`, `snr_filter`, `analog_mods_kept_clean`, `best_epoch`,
  `best_val_weighted`, `seed`, `warm_start_ckpt`, `git_sha`.
- **D-17:** Per-epoch log written alongside checkpoint with columns:
  `epoch, lr, train_loss, train_loss_clean, train_loss_adv, val_clean_acc,
  val_robust_fgsm_acc, val_weighted, time_s`. Plain CSV is sufficient;
  no tensorboard dependency.

### Claude's Discretion
- Exact directory layout inside `adv_train.py` (helper function boundaries,
  CLI argparse grouping) — planner's choice as long as the decisions above
  are honored.
- Whether to cache FGSM-val adversarial examples once per epoch vs. regenerate
  each eval — planner may pick based on wall-time budget.
- Whether logging uses the existing `util.logger.create_logger` or a simple
  CSV writer.
- Reproducibility details (deterministic CuDNN, seed flow into attack RNG)
  beyond calling `util.utils.fix_seed`.

### Folded Todos

None — no cross-matched todos for this phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Specs
- `.planning/ROADMAP.md` §Phase 4 — goal and 4 success criteria (checkpoint
  loads, train log shows clean+robust accuracy + analog retention, JSON
  config saved, per-batch attack selection from {FGSM, PGD, EAD-L1, EAD-EN}
  with CW absent).
- `.planning/REQUIREMENTS.md` §Adversarial Training — AT-01..AT-05 the plan
  must satisfy 1:1.
- `.planning/PROJECT.md` — AT is a baseline, not the main contribution;
  Adaptive-K v2 remains primary.

### Reference Implementations
- `synth_finetune.py` — closest structural parallel. Pattern to follow:
  data prep → DataLoader → train loop with ReduceLROnPlateau + early stop →
  eval → checkpoint save.
- `util/adv_attack.py` — `Model01Wrapper` (lines 31-86),
  `iq_to_ta_input_minmax` (line 108), `ta_output_to_iq_minmax` (line 120).
  These are the adapters the training attack loop must use.
- `util/utils.py` — `create_model`, `fix_seed`.
- `util/training.py` — existing `Trainer` class; reuse `EarlyStopping`
  utility if structure aligns, otherwise replicate inline.
- `util/evaluation.py` — `Run_Eval` for post-training full-SNR evaluation
  hook (sanity check inside Phase 4, full matrix is Phase 5).
- `data_loader/data_loader.py` — `Load_Dataset`, `Dataset_Split`,
  `Create_Data_Loader`. Handles (modulation, SNR) stratification.

### Locked v1.0 Artifacts (read-only)
- `./checkpoint/2016.10a_AWN.pkl` — warm-start source (AT-04).
- `config/2016.10a.yml` — dataset hyperparameters (class mapping, signal
  length 128, num_classes 11).

### External
- torchattacks docs via `mcp__context7__*` if implementation questions arise
  for FGSM/PGD/EADL1/EADEN parameter signatures — no specific ADR exists.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `util.adv_attack.Model01Wrapper` — already wraps AWN's 3D IQ input to the
  torchattacks 4D [0,1] image-style interface; supports both `unit` and
  `minmax` modes via `set_minmax/clear_minmax`. Use directly in the attack
  loop.
- `util.adv_attack.iq_to_ta_input_minmax` / `ta_output_to_iq_minmax` —
  paired per-sample min-max normalization helpers. The attack loop computes
  `(a,b) = iq_to_ta_input_minmax(x)`, sets them on the wrapper, calls
  `torchattacks.<Attack>(wrapped)(x01, y)`, then inverts with the returned
  `(a,b)`.
- `util.utils.create_model` — dispatch by `cfg.model`. For AWN returns the
  network and loads `torch.load(cfg.ckpt_path, weights_only=True)` on
  request.
- `util.utils.fix_seed` — numpy/torch/random seeding with `PYTHONHASHSEED`.
- `data_loader.Dataset_Split` — stratifies by (modulation, SNR). A new
  `snr_min=0` kwarg or a filter pass can produce the SNR ≥ 0 subset.
- `torchattacks` library already pinned in requirements; verified via
  existing `sigguard_eval` mode.

### Established Patterns
- CLI scripts use argparse with `--mode {gen,finetune,eval}` (see
  `synth_finetune.py`). `adv_train.py` should follow the same convention:
  `--mode {train,eval}` with `train` as default.
- Config object pattern (`util.config.Config`) is used for the main modes;
  for a standalone script, either instantiate Config with dataset YAML or
  use a plain argparse Namespace (synth_finetune.py uses the latter).
  Planner may pick; plain argparse is simpler and avoids fighting the
  run-directory auto-increment.
- Seeding: every entry point calls `fix_seed(args.seed)` before any data
  or model code.
- Checkpoints saved as raw `state_dict` via `torch.save(model.state_dict(),
  path)`; loaded with `weights_only=True`.

### Integration Points
- Phase 5 (AT Evaluation) reads `./checkpoint/2016.10a_AWN_at.pkl` through
  the standard `--ckpt_path` flag in `main.py` eval modes. Therefore the
  checkpoint must be a pure AWN `state_dict` — not a dict-of-dicts with
  metadata wrapped in.
- The companion JSON config file must be separate so it does not interfere
  with `torch.load(..., weights_only=True)`.
- Per-SNR evaluation hook at the end of Phase 4 run should use
  `util.evaluation.Run_Eval` on the real RML test split (full SNR range)
  to log a sanity accuracy breakdown into the training log — gives early
  warning if analog-mod accuracy collapsed before Phase 5 begins.

</code_context>

<specifics>
## Specific Ideas

- Follow the `synth_finetune.py` structural template down to function
  boundaries where practical — reviewers can diff the two scripts and see
  that only the data-generation step (synthetic → adversarial) differs.
- The JSON config file is an AT-05 deliverable and doubles as
  Phase 5's evidence that the training setup was what the paper claims.
  It must be generated atomically after the final checkpoint save, not
  before training, so `epochs_trained` and `best_val_weighted` reflect
  reality.
- The per-epoch log will be cited in the paper (PAPRU-03 narrative
  references "training log shows analog classes retain non-trivial
  accuracy"). Keep columns stable so Phase 6 can plot from it without
  re-instrumentation.

</specifics>

<deferred>
## Deferred Ideas

- Alpha sweep (α ∈ {0.3, 0.5, 0.7}) ablation — 3× training cost; not required
  for v1.1 paper update. Future milestone.
- Warmup schedule for alpha (0 → 0.5 over first 5 epochs) — v1.0-style
  stability measures; not needed since warm-start from a mature checkpoint
  and α=0.5 are both conservative choices already.
- Mixing synthetic RML-like bursts into the AT loop — confounds two
  experiments; reopen if v1.1 AT robust accuracy under-performs.
- Adaptive attack evaluation (BPDA, transfer) — already deferred as
  EXTEVAL-01 in REQUIREMENTS.md. Mentioned here only so Phase 5 planner
  does not accidentally include it.
- Epsilon jitter per batch (`eps ~ U[0.05, 0.15]`) — broadens the robustness
  surface but complicates reporting in Table I. Future milestone if a
  reviewer asks for an eps ablation.

### Reviewed Todos (not folded)

None — no todos were surfaced by the cross-reference step.

</deferred>

---

*Phase: 04-adversarial-training*
*Context gathered: 2026-04-16*
