---
phase: 05-at-evaluation
plan: 01
subsystem: cli
tags: [checkpoint, path-resolution, eval-infrastructure]

# Dependency graph
requires: []
provides:
  - "resolve_ckpt_path() helper in main.py that accepts directory OR full .pkl file paths"
  - "--ckpt_path ./checkpoint/2016.10a_AWN_at.pkl resolves verbatim to AT checkpoint"
affects: [05-02, 05-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "resolve_ckpt_path pattern: .pkl suffix + isfile check = verbatim; else join with get_ckpt_name()"

key-files:
  created: []
  modified:
    - main.py

key-decisions:
  - "Patched all 14 classifier-load call sites (plan estimated >= 9; extra 5 from modes added after plan was written: sigguard_eval, calibrate_adaptive_k, adaptive_eval, power_budget_eval)"
  - "resolve_ckpt_path defined at line 185 inside if __name__ == '__main__' block, sibling to get_ckpt_name at line 182"

patterns-established:
  - "resolve_ckpt_path: any --ckpt_path ending in .pkl that exists as a file passes through verbatim; directories fall through to os.path.join(ckpt_path, get_ckpt_name())"

requirements-completed: [ATEVAL-04]

# Metrics
duration: 3min
completed: 2026-04-17
---

# Phase 5 Plan 1: Checkpoint Path Resolution Shim Summary

**resolve_ckpt_path() helper enabling file-form --ckpt_path for AT checkpoint loading across all 14 eval modes in main.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-17T04:14:40Z
- **Completed:** 2026-04-17T04:17:56Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added resolve_ckpt_path() helper at line 185 of main.py that resolves --ckpt_path to either a verbatim .pkl file path or directory + get_ckpt_name() join
- Patched all 14 classifier-load torch.load call sites (eval, visualize, adv_eval, freq_compare, freq_topk_eval, freq_topk_adv_eval, adv_bench, multi_attack_eval, sigguard_eval, calibrate_adaptive_k, adaptive_eval, power_budget_eval, defense_compare, calibrate_defenses)
- Verified both directory-form (./checkpoint) and file-form (./checkpoint/2016.10a_AWN_at.pkl) resolve correctly
- Sanity-tested end-to-end: AT checkpoint loads and evaluates successfully on QPSK/SNR=18 subset

## Task Commits

Each task was committed atomically:

1. **Task 1: Add resolve_ckpt_path helper and apply at all classifier-load call sites** - `6cb5d38` (feat)

## Files Created/Modified
- `main.py` - Added resolve_ckpt_path() helper (lines 185-198), replaced 14 os.path.join(args.ckpt_path, get_ckpt_name()) calls with resolve_ckpt_path(args.ckpt_path)

## Decisions Made
- Patched 14 call sites instead of the 9+ estimated in the plan; the additional 5 modes (sigguard_eval, calibrate_adaptive_k, adaptive_eval, power_budget_eval, and one more) were added to the codebase after the plan was written
- Did NOT modify train mode (loads from cfg.model_dir), adv_train mode (loads from cfg.model_dir), train_detector, calibrate_detector, build_psd_mask, or build_cw_psd_mask as they do not use args.ckpt_path for classifier loading

## Deviations from Plan

None - plan executed exactly as written. The extra call sites beyond 9 were anticipated by the plan's "grep for the pattern and wrap" instruction.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- resolve_ckpt_path shim is in place; Plans 02 and 03 can now invoke defense_compare and other eval modes with --ckpt_path ./checkpoint/2016.10a_AWN_at.pkl
- The AT checkpoint at ./checkpoint/2016.10a_AWN_at.pkl loads successfully through the shim

---
*Phase: 05-at-evaluation*
*Completed: 2026-04-17*
