# Phase 2: Experimental Results - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 02-experimental-results
**Areas discussed:** Evaluation matrix scope, Epsilon budget curves, Paper table format, Confusion matrix scope

---

## Evaluation Matrix Scope

### Which attacks?

| Option | Description | Selected |
|--------|-------------|----------|
| 5 required only (Recommended) | CW, EAD L1, EAD EN, FGSM, PGD — matches roadmap. Covers L2/L1/EN/Linf norms. | ✓ |
| 8 attacks (+ DeepFool, BIM, APGD) | Broader picture, more compute time. | |
| All 17 available | Everything in torchattacks. Very thorough but expensive. | |

**User's choice:** 5 required only
**Notes:** None

### Sample limit per cell?

| Option | Description | Selected |
|--------|-------------|----------|
| 200 per cell (Recommended) | Good statistical power, CW/EAD tractable (~45 min/attack). Matches Phase 1 cap. | ✓ |
| Full test set | ~1000 per cell. Most rigorous but hours per attack. | |
| 50 per cell | Fast iteration (~10 min/attack). | |

**User's choice:** 200 per cell
**Notes:** None

### SNR points?

| Option | Description | Selected |
|--------|-------------|----------|
| All 10 (0 through 18 dB) | Full picture. 400 cells in main table. | ✓ |
| 4 representative (0, 6, 12, 18) | Compact tables. Curves still show all 10. | |
| 3 representative (0, 10, 18) | Minimal table size. | |

**User's choice:** All 10
**Notes:** User specified "only test snr >= 0" as a global constraint across all areas.

### Which defenses?

| Option | Description | Selected |
|--------|-------------|----------|
| All 8 (Recommended) | No defense + unified pipeline + spectral gated + 5 classical + RS = 9 rows. | ✓ |
| Core 6 | Drop SG and FIR if underperforming. | |

**User's choice:** All 8 (9 rows including no-defense)
**Notes:** None

---

## Epsilon Budget Curves

### Linf epsilon range (FGSM, PGD)?

| Option | Description | Selected |
|--------|-------------|----------|
| 0.01 to 0.3, 8 points (Recommended) | Covers subtle to overwhelming. Matches prior minmax experience. | ✓ |
| 0.03 to 0.5, 6 points | Wider range, fewer points. | |
| Fixed eps only | Skip budget curves for Linf. | |

**User's choice:** 8 points [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
**Notes:** None

### L2 epsilon range (CW)?

| Option | Description | Selected |
|--------|-------------|----------|
| Vary c: [0.01, 0.1, 1.0, 10.0] (Recommended) | Sweep confidence parameter. Shows defense vs attack strength. | ✓ |
| Vary steps: [10, 50, 100, 200] | Attack convergence sweep. | |
| Fixed c=1.0 only | Skip L2 budget curve. | |

**User's choice:** Vary c = [0.01, 0.1, 1.0, 10.0]
**Notes:** None

### L1/EN epsilon range (EAD)?

| Option | Description | Selected |
|--------|-------------|----------|
| Vary c: [0.01, 0.1, 1.0, 10.0] | Same approach as CW. Consistent presentation. | ✓ |
| Fixed c=1.0 only | Skip EAD budget curves. | |

**User's choice:** Vary c = [0.01, 0.1, 1.0, 10.0]
**Notes:** Consistent c-sweep across all optimization attacks.

---

## Paper Table Format

### Table layout?

| Option | Description | Selected |
|--------|-------------|----------|
| One table per attack (Recommended) | 5 tables. Rows=defenses (9), columns=SNR (10) + average. Compact, standard. | ✓ |
| Single mega-table | 45 rows. Everything in one place. Very large. | |
| Rows=attacks, columns=defenses | One table per SNR (10 tables). | |

**User's choice:** One table per attack
**Notes:** None

### Fixed epsilon for main tables?

| Option | Description | Selected |
|--------|-------------|----------|
| eps=0.03 minmax for Linf, c=1.0 for CW/EAD (Recommended) | Moderate attack strength. Standard for AMC papers. | ✓ |
| eps=0.1 minmax for Linf, c=1.0 for CW/EAD | Stronger Linf. More dramatic but may look unrealistic. | |
| Multiple eps per table | 2-3 eps per attack. More complete but larger. | |

**User's choice:** eps=0.03 minmax, c=1.0
**Notes:** None

### Statistical reporting?

| Option | Description | Selected |
|--------|-------------|----------|
| Single run, accuracy % (Recommended) | Standard in AMC adversarial literature. Deterministic results. | ✓ |
| 3 runs with mean +/- std | 3x compute. More rigorous. | |
| Single run + bold best | Like option 1 with bold highlighting. | |

**User's choice:** Single run, accuracy %
**Notes:** User accepted that deterministic model + deterministic attacks = reproducible.

### Average column?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, weighted average (Recommended) | Weighted by samples per SNR. Easy comparison. | ✓ |
| Yes, macro average | Simple mean across SNR. | |
| No average column | Reader interprets per-SNR. | |

**User's choice:** Weighted average
**Notes:** None

---

## Confusion Matrix Scope

### Which attacks?

| Option | Description | Selected |
|--------|-------------|----------|
| CW + EAD-EN (Recommended) | Strongest L2 and EN attacks. 4 matrices. | |
| CW only | Single strongest attack. 2 matrices. | |
| CW + EAD-L1 + EAD-EN | All three optimization attacks. 6 matrices per SNR. | ✓ |

**User's choice:** CW + EAD-L1 + EAD-EN
**Notes:** User chose the most thorough option.

### At which SNR?

| Option | Description | Selected |
|--------|-------------|----------|
| SNR=18 dB (Recommended) | High SNR, attack effect most visible. | |
| SNR=10 and SNR=18 | Mid and high. Doubles figure count. | |
| SNR=0, 10, 18 | Low/mid/high. 6-9 panels per attack. | ✓ |

**User's choice:** SNR=0, 10, 18
**Notes:** User chose thorough coverage. 18 total matrices (3 attacks x 3 SNRs x before/after).

### Confusion matrix size?

| Option | Description | Selected |
|--------|-------------|----------|
| Full 11x11 (Recommended) | All modulations. Complete picture. | ✓ |
| 8 digital only | Drop AM-DSB, AM-SSB, WBFM. | |
| 6 PSK/QAM only | BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4. | |

**User's choice:** Full 11x11
**Notes:** None

### Confusion matrix format?

| Option | Description | Selected |
|--------|-------------|----------|
| Heatmap with percentage values (Recommended) | Row-normalized. Color-coded. Standard in AMC papers. | ✓ |
| Heatmap without numbers | Pattern-focused. | |
| Raw count table | For appendix. | |

**User's choice:** Heatmap with percentages
**Notes:** None

---

## Claude's Discretion

- `--mode defense_compare` implementation details
- Budget curve script organization
- CSV column naming conventions
- Confusion matrix save format
- Intermediate progress logs

## Deferred Ideas

None — discussion stayed within phase scope
