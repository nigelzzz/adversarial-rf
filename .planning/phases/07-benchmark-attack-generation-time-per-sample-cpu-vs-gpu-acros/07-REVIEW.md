---
phase: 07-benchmark-attack-generation-time-per-sample-cpu-vs-gpu-acros
reviewed: 2026-04-26T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - util/attack_bench.py
  - main.py
  - paper/scripts/plot_attack_bench_latency.py
findings:
  critical: 0
  warning: 3
  info: 7
  total: 10
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-04-26
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 7 introduces a clean, well-documented latency benchmark module
(`util/attack_bench.py`), the `attack_bench` dispatch in `main.py`, and a
standalone CSV-to-PDF plotter. Overall the code is correct, defensive
(state_dict snapshot, CUDA availability check, paper-default pinning), and
follows project conventions (snake_case helpers, NumPy-style docstrings,
logger usage, `weights_only=True` checkpoint load).

No security issues or critical bugs were found. The findings below are
warnings about a subtle state-leak risk between attacks within a device
pass, redundant duplicated logic, and a few code-quality items that would
improve readability and future maintenance.

## Warnings

### WR-01: Model state not reloaded between attacks within a device pass

**File:** `util/attack_bench.py:363-388`
**Issue:** The original `state_dict` is reloaded once per device (line 367),
but inside the device loop the five attacks run sequentially without a
reload between them. The module's own threat-model comment (line 341,
"mitigates T-07-01 — in-place mutation by attacks") motivates the
snapshot, yet the mitigation is only applied across device boundaries —
not across attacks. Some torchattacks attacks call `model.eval()` /
`model.train()` or temporarily mutate `requires_grad` on parameters; if
any of the five attacks leaves the model in an altered state, later
attacks within the same device pass will be timed against a perturbed
model. This contaminates intra-device measurements (especially CW/EAD
which manipulate gradient state heavily) before the cross-device reload
ever runs.
**Fix:** Reload the snapshot at the start of every `(device, attack)`
cell, not just per device:
```python
for attack_name in tqdm(ATTACKS_5, desc=f"attack_bench[{device}]", leave=False):
    # Reload snapshot before EVERY attack to neutralize T-07-01.
    model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
    model.eval()
    attack = create_attack(attack_name, wrapped_model, cfg)
    ...
```
If the cost of reload is non-trivial, do it after `eval()` warmup but
before the timed reps.

### WR-02: `_stamp_paper_defaults` contains dead/contradictory branches

**File:** `util/attack_bench.py:251-274`
**Issue:** Each of the five `cfg.<param>` blocks first computes a
"preserve existing or default" value via nested `getattr` + `or` /
ternary, then immediately overwrites the result with the paper default
when it differs. The first computation is therefore dead code: the final
state is always the D-05 paper default. Worse, the pattern is inconsistent
(`or` for floats, ternary for ints) and obscures intent. A reader trying
to confirm "does this respect a user-set `cfg.cw_c=2.0`?" has to trace
two branches to discover the answer is "no". The same defaults are also
already pinned in `main.py:444-449` before the function is called, making
this a third copy of the same constants.
**Fix:** Replace the body with unconditional assignments matching D-05
intent, or remove `_stamp_paper_defaults` entirely and rely on
`main.py`'s pre-call assignment:
```python
def _stamp_paper_defaults(cfg) -> None:
    """Force D-05 paper-default hyperparameters onto cfg in-place."""
    cfg.attack_eps = 0.03
    cfg.cw_c = 1.0
    cfg.cw_steps = 100
    cfg.cw_lr = 0.01
    cfg.ead_max_iterations = 100
    if not getattr(cfg, 'ta_box', None):
        cfg.ta_box = 'unit'
```

### WR-03: D-05 paper defaults duplicated in three places

**File:** `main.py:444-449`, `util/attack_bench.py:17-22` (docstring),
`util/attack_bench.py:251-274`
**Issue:** The constants `attack_eps=0.03`, `cw_c=1.0`, `cw_steps=100`,
`cw_lr=0.01`, `ead_max_iterations=100`, `ta_box='unit'` are written out
in the docstring, hard-coded in `main.py` immediately before
`run_attack_bench_5x2`, and re-asserted inside `_stamp_paper_defaults`.
Three copies of the same magic numbers means any future paper-default
update must touch three places, and they will eventually drift.
**Fix:** Hoist a single `PAPER_DEFAULTS = {...}` dict at module top of
`util/attack_bench.py`, have `_stamp_paper_defaults(cfg)` iterate it,
delete the redundant assignments in `main.py:444-449`, and reference
the constant in the docstring (or generate it at doc-render time).

## Info

### IN-01: Awkward `if/pass/else` row-count check

**File:** `paper/scripts/plot_attack_bench_latency.py:88-91`
**Issue:** `if len(df) == 10: pass else: raise` is harder to read than
the natural `if len(df) != 10: raise`.
**Fix:**
```python
if len(df) != 10:
    raise ValueError(f"Expected 10 rows, got {len(df)}")
```

### IN-02: `total_seconds` semantics under-described in CSV

**File:** `util/attack_bench.py:243-248`, `406-428`
**Issue:** `total_seconds` is the sum across all `n_reps`, not the time
of one rep. The docstring documents this, but the CSV header
("`total_seconds`") and the JSON sidecar do not, so a downstream reader
who looks only at the artifact may mistake it for a single-rep wall time.
**Fix:** Rename the CSV/JSON column to `total_seconds_all_reps` or add a
`# total_seconds is summed across n_reps` line to the env comment.

### IN-03: `test_idx` parameter accepted but unused

**File:** `util/attack_bench.py:283-284, 322-323`
**Issue:** `test_idx` is documented as "Reserved for future stratification
refinements; currently unused", and silenced with `_ = test_idx`. Dead
parameters bit-rot quickly. If the wiring isn't planned for this phase,
drop it from the signature; the caller already builds `snrs_test` from
`test_idx` in `main.py:454`.
**Fix:** Remove the parameter and the silencer. Re-add when the
stratification refinement actually lands.

### IN-04: Hard-coded dataset glob in plotter default

**File:** `paper/scripts/plot_attack_bench_latency.py:59`
**Issue:** `_resolve_default_csv` globs only `inference/2016.10a_*/...`.
That matches Phase 7's RML2016.10a-only constraint, but if a future
reviewer runs the bench against another dataset the plotter silently
emits "No attack_bench.csv found". A clearer error or a
`--dataset` argument would help.
**Fix:** Either widen the glob to `inference/*/result/attack_bench.csv`
or accept a `--dataset` flag that builds the pattern.

### IN-05: Magic batch-size width in plotter

**File:** `paper/scripts/plot_attack_bench_latency.py:101`
**Issue:** `w = 0.38` is a hand-tuned bar width with no comment. A reader
adjusting the figure for two devices vs three (e.g., adding "MPS") would
have to re-derive the constant.
**Fix:** Add a one-line comment, e.g.
`# w = 0.38 leaves ~0.24 inter-group gap with two device bars.`

### IN-06: Lexicographic sort of inference dirs assumes 1-digit indices

**File:** `paper/scripts/plot_attack_bench_latency.py:68-70`
**Issue:** `sorted(glob.glob(...))[-1]` is comment-justified as
"Lexicographic sort matches chronological order because the index
auto-increments." That holds while the index stays single-digit, but
once `inference/2016.10a_10/` exists alongside `..._9/`, the lexicographic
"newest" becomes `..._9` (since `'2'<'9'` lexicographically). The
existing `Config.init_dir()` auto-increment will eventually trigger this.
**Fix:** Sort by mtime instead:
```python
paths.sort(key=os.path.getmtime)
```
or extract the numeric suffix and sort by int.

### IN-07: `_time_one_cell` does not call `torch.cuda.empty_cache()` between reps

**File:** `util/attack_bench.py:228-245`
**Issue:** Long EAD/CW reps allocate large gradient tensors. Between
reps, allocator reuse usually works, but for n_samples=512, batch=128,
n_reps=5 across attack types there is a small risk of GPU memory
fragmentation that biases later reps. This is performance-adjacent
(out of v1 review scope) but worth a one-line note for paper
reproducibility.
**Fix:** Optional — between reps:
```python
if device == 'cuda':
    torch.cuda.empty_cache()
```
Document the choice (added or skipped) in the module docstring so the
benchmark methodology is unambiguous.

---

_Reviewed: 2026-04-26_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
