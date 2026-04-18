---
phase: 05-at-evaluation
plan: 02
subsystem: eval
tags: [sanity-check, checkpoint, adversarial-training, analog-retention]

# Dependency graph
requires:
  - "05-01: resolve_ckpt_path shim enabling --ckpt_path ./checkpoint/2016.10a_AWN_at.pkl"
provides:
  - "Verified AT checkpoint loads with strict=True into fresh AWN (26/26 keys, 124,043 params)"
  - "QPSK@SNR=18 clean accuracy = 100% (digital class fully retained)"
  - "WBFM@SNR=18 clean accuracy = 36.5% (accepted as AT trade-off; base model was 41.4%)"
  - "Human approval gate cleared — Plan 03 full defense_compare matrix is unblocked"
affects: [05-03]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - .planning/phases/05-at-evaluation/05-02-sanity.log
  modified: []

key-decisions:
  - "WBFM@18=36.5% accepted as known AT trade-off (base model was 41.4%, only 4.9pp drop); paper will report honestly"
  - "Plan 03 cleared to launch despite WBFM below original 50% threshold — user judged the threshold overly conservative for an already-weak class"

patterns-established: []

requirements-completed: [ATEVAL-04]

# Metrics
duration: 5min
completed: 2026-04-17
---

# Phase 5 Plan 2: AT Checkpoint Sanity Check Summary

**AT checkpoint integrity verified (26/26 keys, strict=True) with QPSK@18=100% and WBFM@18=36.5% (accepted AT trade-off); Plan 03 full matrix unblocked**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-17T04:22:00Z
- **Completed:** 2026-04-17T04:27:00Z
- **Tasks:** 3 (2 auto + 1 human-verify checkpoint)
- **Files modified:** 1

## Accomplishments
- Confirmed AT checkpoint state_dict has zero key drift: 26/26 keys match fresh AWN with strict=True, 124,043 parameters (identical architecture to v1.0 base model)
- QPSK@SNR=18 clean accuracy = 100% -- digital modulation classification is fully retained after adversarial training
- WBFM@SNR=18 clean accuracy = 36.5% -- below the plan's conservative 50% threshold but accepted by human reviewer as a known AT trade-off (base model was only 41.4% on WBFM, the weakest class)
- Human approval gate cleared with rationale: WBFM was already the weakest class; 4.9pp drop is honestly reportable as an AT cost; training-free adaptive-K defense is the paper's main contribution (not AT)

## Task Commits

Each task was committed atomically:

1. **Task 1: Quick checkpoint-integrity check (state_dict keys align with AWN)** - `c8a9892` (chore)
2. **Task 2: Run main.py --mode eval on QPSK@18 and WBFM@18 via Plan 01 shim** - `d133528` (chore)
3. **Task 3: Human verifies sanity accuracies** - checkpoint approved (no code change, no commit)

## Files Created/Modified
- `.planning/phases/05-at-evaluation/05-02-sanity.log` - Full sanity check output: checkpoint integrity (keys, param count), QPSK@18=100%, WBFM@18=36.5%, config dumps from both eval runs

## Sanity Check Results

| Check | Result | Threshold | Status |
|-------|--------|-----------|--------|
| state_dict key match | 26/26 keys, strict=True | All match | PASS |
| Parameter count | 124,043 | Same as v1.0 AWN | PASS |
| QPSK@SNR=18 accuracy | 100.0% | >= 70% | PASS |
| WBFM@SNR=18 accuracy | 36.5% | >= 50% (revised) | ACCEPTED |

**WBFM acceptance rationale:** Base model WBFM@18 was 41.4% (already below original 50% threshold). The 4.9pp AT degradation (41.4% -> 36.5%) is a small additional cost on an already-weak class. The paper will argue that training-free adaptive-K outperforms AT on unseen attacks (CW held out from AT training), making WBFM AT cost an honest trade-off disclosure rather than a blocking issue.

## Inference Directories Created
- `inference/2016.10a_162/` - QPSK@SNR=18 sanity run (throwaway; Plan 03 will create its own)
- `inference/2016.10a_163/` - WBFM@SNR=18 sanity run (throwaway; Plan 03 will create its own)

## Decisions Made
- Accepted WBFM@18=36.5% despite being below the plan's original 50% threshold, based on user review that base model was only 41.4% (WBFM is inherently difficult for AWN)
- Cleared Plan 03 to launch the full 2-4h defense_compare matrix without analog class revision

## Deviations from Plan

None - plan executed exactly as written. The WBFM threshold miss was handled through the human-verify checkpoint as designed (Task 3 exists precisely to catch this scenario).

## Issues Encountered
- WBFM@18 accuracy (36.5%) fell below the plan's 50% threshold, but this was evaluated at the human checkpoint and accepted as a known AT trade-off rather than a training regression

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 03 (full defense_compare matrix) is unblocked by human approval
- AT checkpoint confirmed loadable via Plan 01 shim at ./checkpoint/2016.10a_AWN_at.pkl
- Paper should note WBFM AT trade-off when reporting AT baseline results

## Self-Check: PASSED

- [x] `.planning/phases/05-at-evaluation/05-02-sanity.log` exists in commit d133528
- [x] CHECKPOINT_INTEGRITY_OK present in sanity log
- [x] SANITY_ACCURACIES present in sanity log
- [x] Commit c8a9892 (Task 1) exists in history
- [x] Commit d133528 (Task 2) exists in history
- [x] SUMMARY.md file created at expected path
- [x] Accuracy numbers match: QPSK@18=1.0 (100%), WBFM@18=0.365 (36.5%)

---
*Phase: 05-at-evaluation*
*Completed: 2026-04-17*
