# Phase 4: Adversarial Training - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-16
**Phase:** 04-adversarial-training
**Areas discussed:** Script architecture, Attack params per step, Clean/adv mix + alpha, Data scope + ckpt selection

---

## Script Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone adv_train.py (Recommended) | New top-level script parallel to synth_finetune.py. Reuses data_loader/training helpers, keeps AT logic isolated. | ✓ |
| New --mode adv_train in main.py | Consistent with existing modes. Adds ~200 lines to main.py which is already 800+ lines. | |
| Extend synth_finetune.py | Add --mode at_finetune that replaces synthetic bursts with adv generation. Couples two experiments. | |

**User's choice:** Standalone adv_train.py
**Notes:** Parallels synth_finetune.py structure; easy to archive after paper.

---

## Attack Params Per Step

### ta_box mode

| Option | Description | Selected |
|--------|-------------|----------|
| minmax, eps=0.1 (Recommended) | Matches v1.0 eval setup. Adv distribution = reporting distribution. | ✓ |
| unit, eps=0.03 | Simpler mapping but mismatches paper eval. | |
| Mixed: minmax + eps jitter U[0.05, 0.15] | Broadens robustness surface; harder to ablate. | |

**User's choice:** minmax, eps=0.1

### Attack iterations

| Option | Description | Selected |
|--------|-------------|----------|
| PGD=7, EAD=7 (Recommended) | Madry-style fast AT budget. | ✓ |
| PGD=10, EAD=10 | ~1.4× cost, stronger robustness. | |
| PGD=20, EAD=20 | Near-eval strength, ~3× cost. | |

**User's choice:** PGD=7, EAD=7

### Per-batch attack selection

| Option | Description | Selected |
|--------|-------------|----------|
| Uniform random per batch (Recommended) | Simple, satisfies AT-01 wording. | ✓ |
| Weighted: oversample EAD | EAD is slow and dissimilar; weight it higher. | |
| Round-robin deterministic | Rotate every batch; loses stochasticity. | |

**User's choice:** Uniform random per batch

---

## Clean/Adv Mix + Alpha

### Mix shape

| Option | Description | Selected |
|--------|-------------|----------|
| Dual-batch weighted loss (Recommended) | L = α·CE(adv) + (1-α)·CE(clean). Two forward passes; clean weight is a pure hyperparameter. | ✓ |
| Split-batch (half clean / half adv) | Single forward pass; cheaper but weight tied to batch composition. | |
| Adv-only with periodic clean epochs | Every Nth epoch is pure clean retraining. Analog drift hard to control. | |

**User's choice:** Dual-batch weighted loss

### Alpha

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed α=0.5 (Recommended) | Matches REQUIREMENTS.md default; simplest to ablate. | ✓ |
| Warmup 0.0 → 0.5 over 5 epochs | Extra hyperparameter. | |
| Sweep α ∈ {0.3, 0.5, 0.7} | 3× training cost; deferred to future milestone. | |

**User's choice:** Fixed α=0.5

### Analog mods

| Option | Description | Selected |
|--------|-------------|----------|
| No — always clean for analog mods (Recommended) | Substitute clean input in adv stream when y ∈ analog. Directly implements AT-03. | ✓ |
| Yes — attack all 11 classes uniformly | Simplest; risks degrading analog accuracy. | |

**User's choice:** No — always clean for analog mods

---

## Data Scope + Checkpoint Selection

### SNR scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full RML train split (Recommended) | All SNRs; matches pretrained distribution. | |
| SNR ≥ -6 dB only | Skip very low SNR; shifts distribution vs warm start. | |
| SNR ≥ 0 dB only | Focus on paper eval SNRs; risks forgetting low-SNR behavior. | ✓ |

**User's choice:** SNR ≥ 0 dB only
**Notes:** Rationale accepted — eval is at SNR ∈ {0,6,12,18} dB, warm-start
carries low-SNR weights. Flagged: Phase 5 should sanity-check full-range
clean accuracy post-AT to confirm no regression on SNR < 0.

### Best checkpoint criterion

| Option | Description | Selected |
|--------|-------------|----------|
| Weighted val: 0.5·clean + 0.5·robust(FGSM) (Recommended) | Mirrors training loss; CW held out. | ✓ |
| Val clean accuracy | Standard but ignores robustness. | |
| Val robust accuracy only | Risks drifting clean accuracy below baseline. | |

**User's choice:** Weighted val (0.5·clean + 0.5·FGSM-robust)

### Budget

| Option | Description | Selected |
|--------|-------------|----------|
| ft_epochs=30, patience=8 (Recommended) | AT converges fast from warm start. | ✓ |
| ft_epochs=50, patience=12 | Risk of overfitting val FGSM proxy. | |
| ft_epochs=15, patience=5 | Aggressive cut; analog-mod loss may not stabilize. | |

**User's choice:** ft_epochs=30, patience=8

---

## Claude's Discretion

- Function decomposition inside `adv_train.py` (helper boundaries, argparse grouping).
- Whether to cache FGSM-val adversarial examples once per epoch or regenerate each eval.
- Logging backend choice: existing `util.logger.create_logger` vs. plain CSV writer.
- Reproducibility details beyond `util.utils.fix_seed`.

## Deferred Ideas

- Alpha sweep ablation (α ∈ {0.3, 0.5, 0.7}).
- Alpha warmup schedule (0 → 0.5 over 5 epochs).
- Mixing synthetic RML-like bursts into the AT loop.
- Adaptive attack evaluation (BPDA, transfer) — EXTEVAL-01 future milestone.
- Epsilon jitter per batch.
