# Phase 5: AT Evaluation - Context

**Gathered:** 2026-04-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce two new defense rows — `at` (AT classifier alone, no input defense) and
`at_adaptive_k` (AT classifier + Adaptive-K v2 SNR recovery) — in the defense
comparison matrix on the RML2016.10a test split, across all 5 paper attacks
(FGSM, PGD, EAD-L1, EAD-EN, CW) and 10 SNR points (0..18 step 2). Deliverables
are CSVs in the v1.0 schema plus per-SNR accuracy curves. Paper-side narrative
writing, Table I integration, and discussion paragraphs are Phase 6.

CW is held-out from AT training (Phase 4 D-04). Phase 5 is where that
generalization claim is measured.

</domain>

<decisions>
## Implementation Decisions

### Eval Entry Point
- **D-01:** Re-run the existing `main.py --mode defense_compare --ckpt_path
  ./checkpoint/2016.10a_AWN_at.pkl` with the AT checkpoint loaded as the
  classifier. No new script. No new main.py mode. The existing
  `util/defense_compare.py:run_defense_compare()` iterates 9 defenses × 5
  attacks × 10 SNRs against whatever `model` is passed in — supplying AT as
  that model produces all needed rows. Planner must resolve the small
  `get_ckpt_name()` mismatch (currently hardcodes `<dataset>_AWN.pkl`) either
  by (a) extending `--ckpt_path` to accept a full file path when it points at
  a `.pkl` file, or (b) adding a small `--ckpt_name` override flag. Prefer
  (a) to minimise API surface churn.

### Row Naming and Composition Semantics
- **D-02:** The `at` row in the final CSV corresponds to the `no_defense`
  row produced when `--ckpt_path` is the AT checkpoint (AT classifier, no
  input-side recovery).
- **D-03:** The `at_adaptive_k` row corresponds to the `adaptive_k_v2_snr`
  row produced when `--ckpt_path` is the AT checkpoint. Only the SNR-aware
  v2 flavor is used — this is the paper's proposed primary defense. Other
  Adaptive-K flavors (v1, v2-no-SNR) are NOT included, keeping Table I to
  one composition row, not an ablation grid.
- **D-04:** Row renaming from `no_defense` → `at` and `adaptive_k_v2_snr` →
  `at_adaptive_k` happens in Phase 6 (paper writing) at the point of
  LaTeX/table integration. Phase 5's CSVs keep the raw defense names from
  `DEFENSE_CONFIGS` so that downstream provenance matches the code.

### Attack and SNR Scope
- **D-05:** Full 5 × 10 matrix — all 5 attacks (`cw`, `eadl1`, `eaden`,
  `fgsm`, `pgd`) × all 10 SNR points (0, 2, 4, 6, 8, 10, 12, 14, 16, 18).
  Matches the v1.0 CSV shape exactly, enabling symmetric Table I merging in
  Phase 6 and clean per-SNR curve plots without holes. Estimated compute:
  2–4 hours on the single-GPU setup.
- **D-06:** ATEVAL-01 (CW at {0, 6, 12, 18}) and ATEVAL-02 (all 5 attacks at
  SNR=18) are subsets of D-05 — satisfied automatically by running the full
  matrix. No separate "literal requirements" pass needed.
- **D-07:** SNR < 0 is out of scope. AT was trained on SNR ≥ 0 only (Phase 4
  D-09); low-SNR behavior is inherited from the v1.0 warm-start and reporting
  on it would obscure the AT ablation. The `SNR_POINTS` constant in
  `util/defense_compare.py` already reflects this (0..18).

### Sample Budget and Epsilon
- **D-08:** `max_per_cell = 200` samples per (SNR, modulation) cell. Matches
  v1.0 defense_compare defaults (`util/defense_compare.py` D-04 note) so the
  statistical power of AT rows matches existing rows.
- **D-09:** Attack epsilon and normalization: `ta_box=minmax`, `eps=0.1`.
  Inherits from Phase 4 D-02 (the AT training adversarial distribution) and
  from v1.0 Phase 2 D-05 (the paper's evaluation configuration). Both AT
  training and evaluation therefore share the same threat model.

### Calibration Policy
- **D-10:** Reuse the v1.0 per-SNR calibration file at
  `inference/2016.10a_165/result/calibration_params.json` via the existing
  `--calibration_path` auto-detect path (main.py lines 572–579 pick the most
  recent glob hit). This measures the **pure composition effect** of AT +
  Adaptive-K v2 SNR using the v1.0-calibrated defense parameters — the
  cleanest apples-to-apples comparison against the shipped v1.0 Table I rows.
  Recalibrating Adaptive-K for AT is explicitly deferred (see Deferred Ideas).

### Output Layout
- **D-11:** Let `main.py` auto-increment into a new
  `inference/2016.10a_<N>/result/defense_compare/` directory for the AT run.
  The v1.0 artifacts at `inference/2016.10a_165/` are read-only — not
  modified in place — preserving reproducibility of the shipped v1.0 claim.
  The new directory will contain a full `defense_compare.csv` plus all six
  per-attack pivots (`defense_compare_cw.csv`, ..., `defense_compare_pgd.csv`)
  alongside confmats and budget curves if defaults run.
- **D-12:** Phase 5 produces the CSVs but does NOT rename rows, merge into
  v1.0, or edit Table I. Phase 6 is responsible for: extracting the
  `no_defense` and `adaptive_k_v2_snr` rows from the AT run, renaming to
  `at` and `at_adaptive_k`, and integrating into the v1.0 Table I LaTeX.

### Companion Artifacts (defaults honored, not re-specified)
- **D-13:** Confusion matrices and perturbation budget curves — the existing
  `--mode defense_compare` also produces these via
  `generate_confusion_matrices()` and `generate_budget_curves()` unless
  `--skip_confmat` / `--skip_budget` is passed. Default behavior (both ON)
  is kept for Phase 5 — extra artifacts are cheap during the same run and
  Phase 6 may want the AT confmat at (cw, SNR=18) for a qualitative figure.
  Planner may skip if compute budget tightens.

### Per-SNR Curve Plots (ATEVAL-05)
- **D-14:** The plot deliverable is data-level, not figure-level. The
  per-attack pivot CSVs (`defense_compare_<attack>.csv`) contain the per-SNR
  accuracy curve for each defense row. Phase 5 ensures these CSVs exist and
  are correctly shaped so `paper/scripts/generate_figures.py:plot_defense_overview`
  (or a Phase 6 variant of it) can consume both v1.0 and AT rows together.
  Generating the actual PDF figure lives in Phase 6 (paper update).

### Pre-Flight Sanity Check
- **D-15:** Before launching the full 2–4h run, the planner must include a
  quick sanity step: load the AT checkpoint, run `main.py --mode eval
  --ckpt_path ./checkpoint/2016.10a_AWN_at.pkl --mod_filter QPSK --snr_filter
  18` (or equivalent) and confirm non-trivial accuracy on a clean subset.
  Guards against any serialization mismatch (state_dict key drift between
  the Phase 4 saver and the Phase 5 loader).

### Claude's Discretion
- Whether to run the AT defense_compare in one invocation or split by attack
  (e.g., `--attack_list cw` first to confirm the held-out CW numbers look
  reasonable before running the other four attacks) — planner's choice based
  on wall-time vs risk tradeoff.
- How to organize the small `--ckpt_path` shim (D-01 option a vs b) — pick
  whichever is least invasive to existing eval modes that also use
  `--ckpt_path`.
- Whether to run `--skip_budget` for Phase 5 if compute is tight — the v1.0
  run already has budget curves; Phase 6 may or may not need AT budget curves.

### Folded Todos

None — no cross-matched todos for this phase.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Specs
- `.planning/ROADMAP.md` §Phase 5 — phase goal and 4 success criteria
  (defense_compare CSV has `at`/`at_adaptive_k` rows with per-attack per-SNR
  values for all 5 attacks; AT vs CW reportable at SNR {0, 6, 12, 18};
  per-SNR plots for both new defenses).
- `.planning/REQUIREMENTS.md` §Evaluation — ATEVAL-01..ATEVAL-05 the plan
  must satisfy 1:1.
- `.planning/PROJECT.md` — "AT is a baseline, not the main contribution;
  Adaptive-K remains primary." Drives Table I framing for Phase 6.

### Phase 4 Artifacts (consumed by Phase 5)
- `.planning/phases/04-adversarial-training/04-CONTEXT.md` — carried-forward
  decisions: `ta_box=minmax`, `eps=0.1`, attack list {FGSM, PGD, EAD-L1,
  EAD-EN}, CW held-out, analog classes kept clean, SNR ≥ 0 training scope.
- `./checkpoint/2016.10a_AWN_at.pkl` — AT checkpoint (state_dict only).
- `./checkpoint/2016.10a_AWN_at.config.json` — training config record (epoch
  28 best, weighted val 0.766, 30 epochs total).
- `./checkpoint/2016.10a_AWN_at_log.csv` — per-epoch training log (val_clean
  reached 0.9174, val_robust_fgsm reached 0.618).

### Core Implementation Files
- `util/defense_compare.py` — `run_defense_compare()` (line 365),
  `DEFENSE_CONFIGS` (line 91), `SNR_POINTS` (line 79), `ATTACKS` (line 77),
  `generate_confusion_matrices()` (line 599), `generate_budget_curves()`
  (line 841). All the heavy lifting already exists.
- `util/defense_registry.py` — `DEFENSE_REGISTRY` (lines 147–159) with
  `adaptive_k_v2_snr`, `adaptive_k`, `adaptive_k_v2`, `spectral_gated`, and
  the five classical filters.
- `main.py` §`elif args.mode == 'defense_compare'` (lines 558–622) — the
  dispatch that Phase 5 re-runs. Note the auto-detect logic for
  `calibration_path` at lines 572–579.
- `util/adv_attack.py` — `Model01Wrapper`, `iq_to_ta_input_minmax`,
  `ta_output_to_iq_minmax`. Unchanged from Phase 4; AT classifier wraps the
  same way.

### Locked v1.0 Artifacts (read-only — do NOT modify)
- `inference/2016.10a_165/result/defense_compare/defense_compare.csv` — v1.0
  paper evidence; Phase 5 reads schema + weighted_avg format only.
- `inference/2016.10a_165/result/defense_compare/defense_compare_<attack>.csv`
  — one per attack {cw, eadl1, eaden, fgsm, pgd}; v1.0 pivot schema Phase 6
  merges AT rows into.
- `inference/2016.10a_165/result/calibration_params.json` — per-SNR
  calibrated filter params for classical + Adaptive-K variants (D-10 reuse).
- `./checkpoint/2016.10a_AWN.pkl` — v1.0 base classifier; reference for
  delta reporting but not reloaded.

### Phase 6 Interface (for researcher/planner awareness, not Phase 5 deliverable)
- `paper/latex/sections/results.tex` — Table I location where Phase 6 will
  add the two new rows.
- `paper/scripts/generate_figures.py:plot_defense_overview()` — figure
  generator that will consume both v1.0 and AT per-attack pivots.

### External
- torchattacks library signatures accessed via existing
  `util/defense_compare.py:create_attack()`. No new attack objects introduced
  in Phase 5, so no library spec reads required.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `util.defense_compare.run_defense_compare(model, sig_test, lab_test, SNRs,
  test_idx, cfg, logger, detector=None, attacks=None, snr_points=None,
  max_per_cell=200, batch_size=64, calibration_path=None)` — already does
  everything Phase 5 needs when invoked with an AT `model`. No re-writes.
- `util.defense_compare.DEFENSE_CONFIGS` (dict of 9 defenses + `no_defense`)
  — unchanged; the two new paper rows fall out as `no_defense` and
  `adaptive_k_v2_snr` from the AT run.
- `util.defense_compare.generate_confusion_matrices`,
  `generate_budget_curves` — default-on companion artifacts; Phase 5 keeps
  them on unless the planner proves compute is tight.
- `main.py:get_ckpt_name()` — currently returns `<dataset>_AWN.pkl` hardcoded;
  needs the small shim from D-01 (either full-path `--ckpt_path` handling or
  a `--ckpt_name` override) so the AT checkpoint file can be resolved
  without copying it to the canonical filename.

### Established Patterns
- `--ckpt_path` across main.py eval modes loads whatever state_dict sits at
  `os.path.join(args.ckpt_path, get_ckpt_name())`. Phase 5's shim preserves
  that pattern where possible (treat a file path as a direct checkpoint).
- Auto-increment run directories — main.py modes create
  `inference/<dataset>_<N>/result/` on each invocation; Phase 5 benefits
  from this automatically (D-11). No manual path management needed.
- Calibration params are JSON with `{defense_name: {snr: {param: value}}}`
  nested structure. The existing `_set_calibrated_params` /
  `_restore_cfg_params` in `util/defense_compare.py` already handles this;
  Phase 5 just points `--calibration_path` at the v1.0 file (or lets
  auto-detect pick it).
- Checkpoint saved/loaded as raw `state_dict` with `weights_only=True`. AT
  checkpoint matches this convention per Phase 4 D-15 (verified in Phase 4
  VERIFICATION §Required Artifacts).

### Integration Points
- Phase 6 reads CSVs from Phase 5's new `inference/2016.10a_<N>/` directory.
  The directory number is assigned at runtime — planner should record it in
  the phase SUMMARY so Phase 6 has an unambiguous pointer.
- Phase 6 Table I integration uses the per-attack pivot CSV
  `weighted_avg` column for a single-column Table I, or the `{0, 6, 12, 18}`
  columns for a per-SNR Table I row. Format choice is a Phase 6 concern;
  Phase 5 ensures both are present.
- The `plot_defense_overview()` figure script uses `weighted_avg` as the
  primary bar height, so Phase 5's AT pivots must have the `weighted_avg`
  column computed (default behavior of `run_defense_compare`).

</code_context>

<specifics>
## Specific Ideas

- The narrative target for Phase 6 is roughly: "Adversarial training alone
  improves over undefended at high SNR but remains below Adaptive-K on CW;
  AT + Adaptive-K v2 SNR achieves the best of both." Phase 5's numeric
  deliverable must be able to support (or contradict) this story
  quantitatively — if `at_adaptive_k` does NOT beat `adaptive_k_v2_snr`
  alone, Phase 6 narrative will need revision, not Phase 5 data.
- Single-GPU run of 2–4h is acceptable; the planner should NOT parallelize
  across GPUs or split the run into dozens of per-(attack, SNR) invocations.
  Keep it to one `main.py --mode defense_compare` call end-to-end so the
  CSV and pivots are atomically produced from one logger/one directory.
- If budget curves or confmats push the run past ~6h, the planner may
  `--skip_budget` but must NOT skip the core 5 × 10 defense_compare matrix.
  The matrix is the paper deliverable; budget curves are ablation support.

</specifics>

<deferred>
## Deferred Ideas

- **Recalibrating Adaptive-K for the AT classifier** — would answer "best
  achievable AT + Adaptive-K" but adds a calibration sweep and muddies the
  "does our shipped pipeline stack with robust training" narrative. Revisit
  in a future milestone if a reviewer asks whether composition was tuned.
- **Ablation row `at_adaptive_k_v1`** (AT + v1 baseline Adaptive-K) — useful
  to show that SNR-aware cap still matters when classifier is robust, but
  adds a Table I row that dilutes the main comparison. Future milestone if
  the ablation section is expanded.
- **Adaptive attack evaluation on AT** — BPDA, transfer attacks. Explicitly
  deferred as EXTEVAL-01 in REQUIREMENTS.md. Must not sneak into Phase 5.
- **SNR < 0 evaluation for AT rows** — AT was trained on SNR ≥ 0 only
  (Phase 4 D-09); low-SNR AT behavior is a separate research question.
- **Per-modulation breakdown of AT rows** — interesting for a paper
  supplemental but not required for Table I. Phase 6 may generate this
  from confmats produced as a side effect of D-13; not a Phase 5 goal.
- **Rebuilding `freq_spectra_cw.pdf` with AT-model CW samples** — that
  figure is a Phase 6 CRTD-02 deliverable; Phase 5 produces only numeric
  results, not figure assets.

### Reviewed Todos (not folded)

None — no todos were surfaced by the cross-reference step.

</deferred>

---

*Phase: 05-at-evaluation*
*Context gathered: 2026-04-16*
