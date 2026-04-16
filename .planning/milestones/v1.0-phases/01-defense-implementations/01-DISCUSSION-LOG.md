# Phase 1: Defense Implementations - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 01-defense-implementations
**Areas discussed:** Filter parameters, Pipeline interface, Smoothing design, Latency method

---

## Filter Parameters

| Option | Description | Selected |
|--------|-------------|----------|
| Literature values | Use published parameter values from RF defense papers, fixed across all SNR/modulations | |
| Auto-calibrate | Sweep parameters on validation set, pick best per-filter | ✓ |
| Both | Start with literature defaults, then sweep around them | |

**User's choice:** Auto-calibrate
**Notes:** None

### Calibration Metric

| Option | Description | Selected |
|--------|-------------|----------|
| Clean accuracy | Maximize accuracy on unperturbed signals | |
| Defended accuracy | Maximize accuracy on CW-attacked signals | |
| Composite score | Weighted average of clean + defended accuracy | ✓ |

**User's choice:** Composite score

### Calibration Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Global | One set of params per filter across all SNRs | |
| Per-SNR | Separate optimal params at each SNR point | ✓ |

**User's choice:** Per-SNR

---

## Pipeline Interface

### Registry Design

| Option | Description | Selected |
|--------|-------------|----------|
| Dict + functions | DEFENSE_REGISTRY = {'kalman': kalman_filter, ...} | ✓ |
| Class-based | Each defense is a class with .apply(x) and .name | |

**User's choice:** Dict + functions

### Normalization

**User's choice:** Use minmax normalization before attack, denormalize after attack
**Notes:** User specified custom approach rather than choosing from options

---

## Smoothing Design

### Number of Copies (k)

**User's choice:** k=20 (custom value, not from options)
**Notes:** User specified k=20 and σ=0.01 together

### Sigma

| Option | Description | Selected |
|--------|-------------|----------|
| σ=0.01 fixed | As specified in requirements | ✓ |
| Sweep sigma | Test multiple sigma values | |

**User's choice:** σ=0.01 fixed

---

## Latency Method

### Timing Approach

| Option | Description | Selected |
|--------|-------------|----------|
| CUDA events | torch.cuda.Event for GPU, time.perf_counter for CPU | ✓ |
| Wall clock only | Simple time.time() | |

**User's choice:** CUDA events

### Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Per-sample | Divide batch time by batch size | ✓ |
| Both | Per-sample and throughput | |

**User's choice:** Per-sample

---

## Claude's Discretion

- File organization for new code
- Batch size for latency benchmarking
- Number of calibration iterations
- scipy vs pure torch for filter implementations

## Deferred Ideas

None
