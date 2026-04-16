# Phase 5: AT Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-16
**Phase:** 05-at-evaluation
**Areas discussed:** Script architecture, at_adaptive_k flavor, Attack matrix scope, CSV merge & calibration

---

## Script Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Re-run main.py --mode defense_compare | Run main.py twice (base + AT). Zero new code paths. Needs small shim for get_ckpt_name() filename mismatch. Recommended. | ✓ |
| Standalone at_eval.py | New repo-root script parallel to adv_train.py. Consistent with Phase 4 D-01. Wraps run_defense_compare() with AT-specific CSV merging. | |
| Extend defense_compare.py with --at_ckpt | Add optional AT checkpoint arg; swap classifier per-defense row internally. More invasive to v1.0 code. | |
| New main.py mode: at_eval | Add --mode at_eval. Phase 4 D-01 explicitly rejected adding to main.py (already 800+ lines). | |

**User's choice:** Re-run main.py --mode defense_compare
**Notes:** Zero new code paths beats parallelism with Phase 4. The `get_ckpt_name()` filename shim (accept a full file path on `--ckpt_path` when it points at a `.pkl`, or add `--ckpt_name` override) is a one-line implementation detail for the planner.

---

## at_adaptive_k Flavor

| Option | Description | Selected |
|--------|-------------|----------|
| adaptive_k_v2_snr only | Stack the proposed main contribution on AT. Single new row in Table I. Cleanest paper narrative. Recommended. | ✓ |
| adaptive_k_v2 (no SNR cap) | Isolates whether SNR-adaptivity matters when classifier is robust. Adds a variable outside v1.0's main comparison. | |
| adaptive_k (v1 baseline) | Weaker composition — inconsistent with v1.0 claim that v2-SNR is primary. | |
| Multiple flavors as separate rows | Reports multiple at_adaptive_k_* rows. More data but muddies Table I. | |

**User's choice:** adaptive_k_v2_snr only
**Notes:** Matches the paper's main-contribution story. Ablation variants are deferred to future milestones if reviewers ask.

---

## Attack Matrix Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full 5×10 matrix for both | All 5 attacks × 10 SNRs for both at and at_adaptive_k. Matches v1.0 CSV shape. Symmetric Table I merge. ~2-4h GPU. Recommended. | ✓ |
| Full for CW, sanity for others | CW × 10 SNRs + other 4 attacks at SNR {0, 12, 18}. Hybrid. Non-uniform CSV shape. | |
| Requirements-literal | CW at {0,6,12,18} + all 5 at SNR=18. Minimal compute. Holes in per-SNR plots. | |
| Full for at_adaptive_k, CW-only for at | Composition gets the full matrix; at-alone gets only CW. Saves compute on the less-paper-critical row. | |

**User's choice:** Full 5×10 matrix for both
**Notes:** Symmetric with v1.0 enables straightforward Phase 6 merge. Compute budget (2-4h) is acceptable on the single-GPU setup.

---

## CSV Merge & Output Layout

| Option | Description | Selected |
|--------|-------------|----------|
| New inference run directory | Let main.py auto-increment into a new `inference/2016.10a_<N>/` dir. v1.0 artifacts untouched. Clean provenance. Phase 6 merges. Recommended. | ✓ |
| Append rows to existing v1.0 CSV | Edit `inference/2016.10a_165/result/defense_compare/defense_compare.csv` in place. Risk: shipped v1.0 artifacts modified post-hoc. | |
| Separate defense_compare_at.csv alongside v1.0 | Parallel file in the v1.0 result directory, same schema, AT rows only. Halfway option. | |

**User's choice:** New inference run directory
**Notes:** Preserves v1.0 reproducibility. The directory number is assigned at runtime and should be recorded in Phase 5 SUMMARY for Phase 6's lookup.

---

## Calibration Policy for at_adaptive_k

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse v1.0 calibration_params.json | Apply same per-SNR Adaptive-K v2 SNR params as shipped in v1.0. Measures pure composition effect. Apples-to-apples. Recommended. | ✓ |
| Recalibrate for AT classifier | Run calibrate_defenses with AT ckpt to find optimal per-SNR params. Adds compute and muddies narrative. | |
| Reuse + sensitivity spot-check | Reuse v1.0 params for main row; separately sweep at SNR=18 to confirm not obviously suboptimal. Diagnostic only. | |

**User's choice:** Reuse v1.0 calibration_params.json
**Notes:** This is the cleanest composition story — "does our shipped pipeline stack with robust training, as-shipped?" Recalibration is deferred.

---

## Claude's Discretion

Areas where the planner has latitude:
- One-shot invocation vs split-by-attack (wall-time vs risk tradeoff)
- Exact form of the `--ckpt_path` shim (full-path vs `--ckpt_name` override)
- Whether to `--skip_budget` if compute tightens (but NEVER skip the 5×10 defense_compare matrix itself)

## Deferred Ideas

Captured in CONTEXT.md `<deferred>`:
- Recalibrating Adaptive-K for AT (future milestone)
- `at_adaptive_k_v1` ablation row (future milestone)
- Adaptive/BPDA attacks on AT (EXTEVAL-01, out of v1.1)
- SNR < 0 AT evaluation (AT wasn't trained there)
- Per-modulation breakdown of AT rows (Phase 6 supplement)
- Rebuilding freq_spectra_cw.pdf with AT samples (Phase 6 CRTD-02)
