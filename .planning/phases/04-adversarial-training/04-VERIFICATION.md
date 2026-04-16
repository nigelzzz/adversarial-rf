---
phase: 04-adversarial-training
verified: 2026-04-16T02:50:34Z
status: human_needed
score: 8/9 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run full training (30 epochs) and confirm analog class (WBFM, AM-DSB, AM-SSB) accuracy stays above a non-trivial threshold (e.g., >= 30%) in the sanity eval output"
    expected: "Sanity eval prints WBFM, AM-DSB, AM-SSB individual class accuracy >= 30% after full convergence"
    why_human: "SC2 requires confirming analog classes retain non-trivial accuracy — this is a runtime convergence outcome, not a code property. The mechanism (analog substitution) is correctly implemented, but only a completed training run can confirm the model doesn't catastrophically forget analog classes."
---

# Phase 4: Adversarial Training Verification Report

**Phase Goal:** A trained AT checkpoint exists that robustifies AWN against gradient-based attacks without catastrophic forgetting of analog modulations
**Verified:** 2026-04-16T02:50:34Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | adv_train.py exists at repo root and is executable with --mode train | VERIFIED | File exists at repo root, 637 lines, syntax valid, argparse --mode train present |
| 2 | Data loader produces SNR >= 0 dB subset with 85/15 train/val split | VERIFIED | `build_loaders` calls `Load_Dataset(dataset, logger, snr_min=snr_min)` with `val_ratio=0.15`, uses `np.random.default_rng(seed).permutation` |
| 3 | Attack factory creates FGSM, PGD, EADL1, EADEN with correct eps and iteration budgets | VERIFIED | `make_attacks` instantiates all 4 attacks; `ead_iters_safe = max(ead_iters, 10)` guards against ZeroDivisionError; `binary_search_steps=ead_bss` configurable |
| 4 | Training loop performs dual-batch forward (clean + adversarial) with alpha-weighted loss | VERIFIED | `train_epoch` runs clean forward + adversarial forward; `loss = alpha * loss_adv + (1.0 - alpha) * loss_clean + sum(regu_adv)` at line 265 |
| 5 | Analog modulations (WBFM, AM-DSB, AM-SSB) receive clean input in the adversarial stream | VERIFIED | `ANALOG_INDICES = {3, 6, 10}` and `generate_adv_batch` applies `x_adv[analog_mask] = x[analog_mask]` after attack generation |
| 6 | Model warm-starts from pretrained AWN checkpoint before first optimizer step | VERIFIED | `model.load_state_dict(torch.load(args.warm_start, map_location=device, weights_only=True))` called before `adv_train()` |
| 7 | Training loop uses ReduceLROnPlateau(mode='max') on weighted val metric and early stopping with patience=8 | VERIFIED | `ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)`; `if no_improve >= args.patience: break`; `scheduler.step(val_weighted)` |
| 8 | Best-epoch checkpoint saved to ./checkpoint/2016.10a_AWN_at.pkl as a pure state_dict | VERIFIED | `torch.save(model.state_dict(), ckpt_path)` when `val_weighted > best_weighted`; ckpt_path = `os.path.join(args.ckpt_path, '2016.10a_AWN_at.pkl')` |
| 9 | Analog classes retain non-trivial accuracy after full training convergence | ? HUMAN NEEDED | Code mechanism exists (analog substitution); 2-epoch smoke test showed WBFM=23.8%, AM-DSB=55.7%, AM-SSB=100% (not converged). Full 30-epoch run required to verify. |

**Score:** 8/9 truths verified (1 needs human)

### Deferred Items

No items deferred to later phases.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `adv_train.py` | Core training script >= 250 lines, contains `def train_epoch` | VERIFIED | 637 lines, contains all 9 required functions |
| `adv_train.py` | Complete script >= 350 lines, contains `save_config` | VERIFIED | 637 lines, `def save_config(` present |
| `./checkpoint/2016.10a_AWN_at.pkl` | AT checkpoint (created at runtime) | NOT ON DISK (by design) | Gitignored; code creates it at line 425 when `val_weighted` improves; smoke test confirmed creation |
| `./checkpoint/2016.10a_AWN_at.config.json` | Training hyperparameter record (created at runtime) | NOT ON DISK (by design) | Gitignored; `save_config` writes all 16 D-16 keys via `json.dump` in `finally` block |
| `./checkpoint/2016.10a_AWN_at_log.csv` | Per-epoch training log (created at runtime) | NOT ON DISK (by design) | Gitignored; `adv_train()` writes 9 D-17 columns via `csv.DictWriter`; SUMMARY confirms 2 data rows produced |

Note: Runtime artifacts are gitignored (`checkpoint/.gitignore` contains `*`). They are produced when `python adv_train.py --mode train` runs. The SUMMARY for Plan 02 confirms all three were produced during the 2-epoch smoke test.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `adv_train.py` | `util/adv_attack.py` | `from util.adv_attack import Model01Wrapper, iq_to_ta_input_minmax, ta_output_to_iq_minmax` | WIRED | Import at line 44; all three used in `generate_adv_batch` |
| `adv_train.py` | `data_loader/data_loader.py` | `Load_Dataset` with `snr_min=snr_min` parameter | WIRED | Line 94: `Load_Dataset(dataset, logger, snr_min=snr_min)` |
| `adv_train.py` | `checkpoint/2016.10a_AWN.pkl` | `torch.load` warm-start | WIRED | Line 604-606: `torch.load(args.warm_start, map_location=device, weights_only=True)` |
| `adv_train.py:save_config` | `./checkpoint/2016.10a_AWN_at.config.json` | `json.dump` after training exits | WIRED | `json.dump(cfg, f, indent=2)` in `save_config`; called from `finally` block in `main()` |
| `adv_train.py:adv_train` | `./checkpoint/2016.10a_AWN_at.pkl` | `torch.save(model.state_dict(), ...)` on best weighted val metric | WIRED | Line 425: `torch.save(model.state_dict(), ckpt_path)` when `val_weighted > best_weighted` |
| `adv_train.py:run_sanity_eval` | `util/evaluation.py` | Custom per-SNR accuracy computation (not Run_Eval) | WIRED | `def run_sanity_eval` defined at line 491; uses `Dataset_Split` + custom inference loop |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `adv_train.py:adv_train` | `val_weighted` | `val_epoch()` returns `(val_clean_acc, val_robust_fgsm_acc)` | Yes — computed from model predictions on validation set | FLOWING |
| `adv_train.py:adv_train` | `best_weighted` | Updated when `val_weighted > best_weighted` | Yes — drives checkpoint save decision | FLOWING |
| `adv_train.py:save_config` | `cfg` dict | `args.*` fields + `subprocess.run` for git sha | Yes — all 16 D-16 keys populated from real runtime args | FLOWING |
| `adv_train.py:run_sanity_eval` | `preds` | Model inference over `Dataset_Split` test set | Yes — model forward pass over real test data | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Syntax validity | `python3 -c "import ast; ast.parse(open('adv_train.py').read())"` | No error | PASS |
| All 9 functions present | AST walk for function names | All 9 found: build_loaders, make_attacks, generate_adv_batch, train_epoch, val_epoch, adv_train, save_config, run_sanity_eval, main | PASS |
| D-16 all 16 config keys in save_config | String scan | All 16 present | PASS |
| D-17 all 9 CSV columns in log_fields | String scan | All 9 present | PASS |
| CW absent from ATTACK_NAMES | `grep ATTACK_NAMES adv_train.py` | `['fgsm', 'pgd', 'eadl1', 'eaden']` — CW absent | PASS |

Note: Step 7b full runtime smoke test is not run here (requires GPU + dataset). The 2-epoch smoke test was run during Plan 02 execution and documented in SUMMARY.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|------------|------------|-------------|--------|---------|
| AT-01 | Plan 01 + 02 | Training script finetunes AWN with mixed FGSM/PGD/EAD-L1/EAD-EN, per-batch random attack selection | SATISFIED | `random.choice(ATTACK_NAMES)` in `train_epoch`; all 4 attacks in `make_attacks`; `adv_train` orchestrates full loop |
| AT-02 | Plan 02 | Saves checkpoint `2016.10a_AWN_at.pkl` + per-epoch log with train/val loss and clean/robust accuracy | SATISFIED | `torch.save(model.state_dict(), ckpt_path)` on best epoch; CSV log with `val_clean_acc`, `val_robust_fgsm_acc` |
| AT-03 | Plan 01 | Mixed clean+adversarial loss with configurable alpha (default 0.5), prevents catastrophic forgetting | SATISFIED | `alpha * loss_adv + (1.0 - alpha) * loss_clean + sum(regu_adv)`; `ANALOG_INDICES = {3, 6, 10}` substitution |
| AT-04 | Plan 01 | Warm-start from pretrained `2016.10a_AWN.pkl`, not scratch | SATISFIED | `torch.load(args.warm_start, ..., weights_only=True)` before first optimizer step |
| AT-05 | Plan 02 | Training hyperparameters persisted to JSON config alongside checkpoint | SATISFIED | `save_config` writes 16 D-16 keys including epochs, LR, attack list, eps, ta_box, alpha, git_sha |

No orphaned requirements: All 5 AT-* requirements mapped to Phase 4 in REQUIREMENTS.md are claimed by Plans 01 and 02. ATEVAL-01 through ATEVAL-05 map to Phase 5 (deferred — not applicable here).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `adv_train.py` | 287 | `"deferred to Plan 02"` comment in docstring | Info | Stale doc comment — `val_epoch` docstring says "Full multi-attack evaluation is deferred to Plan 02" but Plan 02 is complete. Harmless doc inaccuracy. |

No blockers, no stubs, no empty implementations found.

### Human Verification Required

#### 1. Analog Class Retention After Full Training

**Test:** Run `python adv_train.py --mode train --epochs 30 --batch_size 256` to completion (expected ~30-90 minutes on GPU). Observe the sanity eval output at the end.

**Expected:** The "Analog class accuracy (catastrophic forgetting check)" section shows WBFM, AM-DSB, and AM-SSB each above a non-trivial threshold (e.g., >= 30% accuracy). The 2-epoch smoke test showed WBFM=23.8%, AM-DSB=55.7%, AM-SSB=100% at epoch 2 (not converged). The analog substitution mechanism (`x_adv[analog_mask] = x[analog_mask]`) should maintain these classes, but convergence must be confirmed.

**Why human:** ROADMAP SC2 requires "confirming analog classes retain non-trivial accuracy." This is a runtime convergence outcome — it cannot be verified from code inspection alone. Only a completed 30-epoch training run can show whether the model forgets analog classes despite the substitution mechanism.

### Gaps Summary

No code gaps found. All 8 verifiable must-haves pass. The single human verification item (analog class retention after full convergence) is a runtime quality check, not a code deficiency.

The runtime artifacts (`2016.10a_AWN_at.pkl`, `2016.10a_AWN_at.config.json`, `2016.10a_AWN_at_log.csv`) are correctly gitignored and confirmed produced during the 2-epoch smoke test. Their absence from the current working tree is expected.

One stale docstring comment ("deferred to Plan 02") in `val_epoch` docstring at line 287 is harmless.

---

_Verified: 2026-04-16T02:50:34Z_
_Verifier: Claude (gsd-verifier)_
