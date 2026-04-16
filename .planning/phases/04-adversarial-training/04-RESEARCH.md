# Phase 4: Adversarial Training - Research

**Researched:** 2026-04-16
**Domain:** Adversarial training loop for PyTorch-based RF modulation classifier (AWN) using torchattacks
**Confidence:** HIGH

## Summary

Phase 4 produces a single new file, `adv_train.py`, at the repo root. The script warm-starts from `./checkpoint/2016.10a_AWN.pkl`, runs a dual-batch training loop (clean + adversarial forward passes with alpha=0.5 weighted loss), and saves `./checkpoint/2016.10a_AWN_at.pkl` plus a JSON config. The entire implementation is grounded in existing project code — no new libraries, no new normalization conventions.

The closest structural template is `synth_finetune.py`. The only architectural difference from that script is the replacement of synthetic data generation with an in-loop attack generation step. All data loading, model creation, optimizer configuration, early stopping, evaluation, and checkpointing patterns are directly reusable.

The one non-obvious implementation risk is EAD attack iteration budget during training. `torchattacks.EADL1` and `torchattacks.EADEN` have two nested loop parameters: `binary_search_steps` (default 9, outer) and `max_iterations` (inner). Using `max_iterations=7` with the default `binary_search_steps=9` produces 63 inner iterations per batch — roughly 9x more expensive than PGD(7). Decision D-03 ("EAD-L1=7, EAD-EN=7") should be interpreted as `max_iterations=7, binary_search_steps=1` to match the "Madry-style fast-AT budget" framing. This is confirmed as a planner decision point.

**Primary recommendation:** Implement `adv_train.py` as a near-copy of `synth_finetune.py` with the training loop body replaced by the attack generation + dual-batch loss logic from D-06.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** New standalone script `adv_train.py` at repo root, parallel to `synth_finetune.py`. Reuses `data_loader.Load_Dataset`, `data_loader.Dataset_Split`, `util.utils.create_model`, `util.evaluation.Run_Eval`. No new mode in `main.py`. No extension of `synth_finetune.py`.
- **D-02:** `ta_box="minmax"`, `eps=0.1` for all attacks.
- **D-03:** Attack iteration budget — FGSM=1 (free), PGD=7, EAD-L1=7, EAD-EN=7. Madry-style fast-AT budget.
- **D-04:** Per-batch uniform random attack selection from {FGSM, PGD, EAD-L1, EAD-EN}. CW is NOT in the training mix.
- **D-05:** Reuse `Model01Wrapper` + `iq_to_ta_input_minmax` / `ta_output_to_iq_minmax` from `util/adv_attack.py`. No new normalization code.
- **D-06:** Dual-batch weighted loss: `L = α·CE(model(x_adv), y) + (1-α)·CE(model(x_clean), y) + regu_sum`. Two forward passes per batch; AWN regularization added once per detail forward.
- **D-07:** Alpha fixed at 0.5. No warmup, no sweep. JSON config records this.
- **D-08:** Analog modulations (WBFM, AM-DSB, AM-SSB) are kept clean in the adversarial stream — substitute clean input for adversarial when label ∈ {3, 6, 10} (WBFM, AM-DSB, AM-SSB class indices).
- **D-09:** Train only on SNR ≥ 0 dB from RML2016.10a train split.
- **D-10:** Validation split follows same SNR filter (85/15 train/val split). Test-set evaluation at full SNR range is Phase 5.
- **D-11:** Do NOT mix synthetic bursts from `synth_finetune.py`.
- **D-12:** Best-epoch criterion: `0.5·val_clean_acc + 0.5·val_robust_acc(FGSM)`. Val loss alone is NOT used.
- **D-13:** Max epochs=30, early-stopping patience=8. `ReduceLROnPlateau`(factor=0.5, patience=4) on weighted val metric. Initial LR=1e-4, Adam, batch_size=256.
- **D-14:** Warm-start from `./checkpoint/2016.10a_AWN.pkl` before first optimizer step.
- **D-15:** Checkpoint: `./checkpoint/2016.10a_AWN_at.pkl`.
- **D-16:** Config JSON: `./checkpoint/2016.10a_AWN_at.config.json` with keys: `epochs_total`, `epochs_trained`, `lr_initial`, `batch_size`, `ta_box`, `eps`, `attack_iters` (per-attack dict), `attack_list`, `alpha`, `snr_filter`, `analog_mods_kept_clean`, `best_epoch`, `best_val_weighted`, `seed`, `warm_start_ckpt`, `git_sha`.
- **D-17:** Per-epoch log CSV with columns: `epoch, lr, train_loss, train_loss_clean, train_loss_adv, val_clean_acc, val_robust_fgsm_acc, val_weighted, time_s`.

### Claude's Discretion

- Exact directory layout inside `adv_train.py` (helper function boundaries, CLI argparse grouping).
- Whether to cache FGSM-val adversarial examples once per epoch vs. regenerate each eval.
- Whether logging uses `util.logger.create_logger` or a simple CSV writer.
- Reproducibility details (deterministic CuDNN, seed flow into attack RNG) beyond calling `util.utils.fix_seed`.

### Deferred Ideas (OUT OF SCOPE)

- Alpha sweep (α ∈ {0.3, 0.5, 0.7}) ablation.
- Warmup schedule for alpha (0 → 0.5 over first 5 epochs).
- Mixing synthetic RML-like bursts into the AT loop.
- Adaptive attack evaluation (BPDA, transfer) — deferred as EXTEVAL-01.
- Epsilon jitter per batch.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AT-01 | Training script finetunes AWN using mixed FGSM/PGD/EAD-L1/EAD-EN with per-batch random attack selection | torchattacks 3.5.1 has `FGSM`, `PGD`, `EADL1`, `EADEN`; per-batch random selection via `random.choice` before each batch |
| AT-02 | Saves checkpoint `./checkpoint/2016.10a_AWN_at.pkl` and per-epoch log with train/val loss and clean/robust accuracy | `torch.save(model.state_dict(), ...)` pattern confirmed; CSV log follows D-17 columns |
| AT-03 | Mixed clean+adversarial loss (alpha=0.5) to prevent catastrophic forgetting of analog modulations | Analog indices {3, 6, 10} identified; substitute x_clean for x_adv before adversarial forward when label is in analog set |
| AT-04 | Warm-start from `./checkpoint/2016.10a_AWN.pkl` | Checkpoint confirmed to exist as pure `state_dict` (OrderedDict, 26 keys); loads with `weights_only=True` |
| AT-05 | Hyperparameters persisted to JSON config alongside checkpoint | All D-16 fields resolvable at runtime; `git_sha` via `subprocess.run(['git', 'rev-parse', '--short', 'HEAD'])` |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchattacks | 3.5.1 | FGSM, PGD, EADL1, EADEN attack generation | Already pinned in requirements; used by all existing attack modes [VERIFIED: runtime check] |
| torch | 1.7+ (1.8.1 tested) | Model training, loss, optimizer, DataLoader | Core framework [VERIFIED: codebase] |
| numpy | (existing) | Array ops for data split and label filtering | Used throughout project [VERIFIED: codebase] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| util.logger.create_logger | internal | Training log file + console output | AT script should write log to `./checkpoint/adv_train.log` or a side log file |
| util.early_stop.EarlyStopping | internal | Patience-based early stopping | Can reuse with a wrapper; monitors loss but D-12 requires max of weighted acc, so easier to track `no_improve` counter inline |
| pandas | (existing) | Per-epoch CSV log | Simple `csv.writer` also works per planner discretion |

**No new installations required.** All dependencies are already present. [VERIFIED: runtime import checks]

## Architecture Patterns

### Recommended Project Structure

`adv_train.py` at repo root, parallel to `synth_finetune.py`. Internal layout mirrors `synth_finetune.py`:

```
adv_train.py
├── Module-level constants    # ANALOG_INDICES, IDX_TO_MOD, ATTACK_LIST
├── build_loaders()           # Load_Dataset(snr_min=0) -> 85/15 split -> DataLoaders
├── make_attack()             # Factory: attack_name -> torchattacks object
├── generate_adv_batch()      # x_clean + y -> x_adv (with analog substitution)
├── train_epoch()             # one epoch: adv gen + dual-batch loss
├── val_epoch()               # clean accuracy + FGSM robust accuracy
├── adv_train()               # main loop: epochs, early stopping, checkpoint save
├── run_eval()                # post-training full-SNR Run_Eval hook
└── main() / argparse         # CLI entry point
```

### Pattern 1: Dual-Batch Loss with Analog Substitution (D-06 + D-08)

**What:** Two forward passes per batch — one on clean signals, one on adversarial. For samples where the label is an analog modulation, substitute x_clean into the adversarial batch before the adversarial forward pass.

**When to use:** Every training batch.

```python
# Source: D-06, D-08 (CONTEXT.md) + util/adv_attack.py patterns
ANALOG_INDICES = {3, 6, 10}  # WBFM=3, AM-DSB=6, AM-SSB=10

def generate_adv_batch(wrapped_model, x, y, attack, ta_box='minmax'):
    """Generate adversarial examples; substitute clean for analog labels."""
    from util.adv_attack import iq_to_ta_input_minmax, ta_output_to_iq_minmax

    x01_4d, a, b = iq_to_ta_input_minmax(x)
    wrapped_model.set_minmax(a, b)
    with torch.enable_grad():
        adv01_4d = attack(x01_4d, y)
    wrapped_model.clear_minmax()
    x_adv = ta_output_to_iq_minmax(adv01_4d, a, b)

    # Substitute clean signal for analog modulations
    analog_mask = torch.tensor([yi.item() in ANALOG_INDICES for yi in y],
                                dtype=torch.bool, device=x.device)
    if analog_mask.any():
        x_adv[analog_mask] = x[analog_mask]

    return x_adv.detach()

# In train loop:
x_adv = generate_adv_batch(wrapped_model, x, y, attack_obj)
logit_clean, regu_clean = model(x)
logit_adv,  regu_adv   = model(x_adv)
loss = alpha * criterion(logit_adv, y) + (1 - alpha) * criterion(logit_clean, y) + sum(regu_adv)
```

Note: regu_sum is added once (from the adv forward) per D-06 ("added once per forward (detail)"). [ASSUMED] — the exact interpretation ("once per forward" could mean "sum from both forwards") should be confirmed by reading D-06 literally: "AWN's internal regularization added once per forward (detail)" means one regu_sum addition from the adversarial pass only, not both.

### Pattern 2: Per-Batch Random Attack Selection (D-04)

**What:** Instantiate all four attack objects once before training. Each batch, draw one uniformly at random.

```python
# Source: D-04 (CONTEXT.md)
import random

attacks = {
    'fgsm': torchattacks.FGSM(wrapped_model, eps=eps),
    'pgd':  torchattacks.PGD(wrapped_model,  eps=eps, alpha=eps/4, steps=pgd_steps),
    'eadl1': torchattacks.EADL1(wrapped_model, kappa=0, lr=0.01,
                                  max_iterations=ead_iters, binary_search_steps=1),
    'eaden': torchattacks.EADEN(wrapped_model, kappa=0, lr=0.01,
                                  max_iterations=ead_iters, binary_search_steps=1),
}
# Per batch:
attack_name = random.choice(list(attacks.keys()))
attack_obj  = attacks[attack_name]
```

### Pattern 3: Weighted Validation Metric and Best-Epoch Checkpoint (D-12)

**What:** After each epoch, compute `val_weighted = 0.5 * val_clean_acc + 0.5 * val_robust_fgsm_acc`. Save checkpoint when this exceeds all previous values.

```python
# Source: D-12 (CONTEXT.md)
val_weighted = 0.5 * val_clean_acc + 0.5 * val_robust_fgsm_acc
if val_weighted > best_weighted:
    best_weighted = val_weighted
    best_epoch    = epoch
    torch.save(model.state_dict(), ckpt_path)
    no_improve = 0
else:
    no_improve += 1

scheduler.step(val_weighted)   # ReduceLROnPlateau in 'max' mode
if no_improve >= patience:
    break
```

### Pattern 4: Data Loading for SNR ≥ 0 with 85/15 Split (D-09, D-10)

**What:** `Load_Dataset` already accepts `snr_min=0` parameter (verified). Then split 85/15 train/val using numpy permutation (same approach as `synth_finetune.py:make_loaders`). Do NOT use `Dataset_Split` for the AT loop — it only supports `val_size + test_size` and would produce a 3-way split with the wrong ratios for D-10.

```python
# Source: data_loader.py (Load_Dataset signature verified), D-09/D-10
import logging
logger = logging.getLogger('at_data')
Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger, snr_min=0)
# Signals: [N, 2, 128] torch.Tensor (N ~ 110000 for SNR>=0 all mods)
# Labels: [N] torch.int64

rng = np.random.default_rng(seed)
idx = rng.permutation(len(Labels))
n_val = int(len(Labels) * 0.15)
val_idx, train_idx = idx[:n_val], idx[n_val:]
```

### Anti-Patterns to Avoid

- **Using `Dataset_Split` for AT data:** It returns (train, test, val, test_idx) with default 60/20/20 ratios. D-10 specifies 85/15 with no separate test split. Use `make_loaders`-style numpy permutation instead.
- **Keeping `binary_search_steps=9` for EAD during training:** Default is 9, causing 63 iterations per batch. Use `binary_search_steps=1` to match the intended Madry-style budget of 7 actual iterations.
- **Wrapping metadata into the checkpoint dict:** Phase 5 loads via `torch.load(..., weights_only=True)`. The checkpoint must be a pure `state_dict` OrderedDict. JSON config must be a separate file (D-16).
- **Regenerating the FGSM wrapper with updated minmax params inside val loop without calling `clear_minmax`:** The wrapper stores `(a, b)` as instance state. Forgetting `clear_minmax()` after each adversarial batch leaks normalization parameters across batches.
- **Calling `model.eval()` during attack generation:** `torchattacks` requires `model.train()` mode (or at minimum `torch.enable_grad()`) to compute gradients. The `model_wrapper.base` should be in eval mode only during clean inference passes, not during attack generation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| IQ ↔ [0,1] normalization for torchattacks | Custom mapping | `iq_to_ta_input_minmax` / `ta_output_to_iq_minmax` | Already tested, handles edge cases like zero-range signals via `.clamp_min(1e-6)` |
| torchattacks model adapter | 4D wrapper | `Model01Wrapper` | Already handles dim squeezing, minmax state management |
| Model factory | AWN instantiation | `util.utils.create_model(cfg, 'awn')` | Handles Config-based hyperparameter wiring |
| Seed control | Manual seeding | `util.utils.fix_seed(seed)` | Already sets numpy/torch/random/PYTHONHASHSEED |
| FGSM/PGD/EAD attack objects | Custom gradient-based attacks | `torchattacks.FGSM`, `torchattacks.PGD`, `torchattacks.EADL1`, `torchattacks.EADEN` | Verified 3.5.1 in environment; all four classes available |

**Key insight:** The entire attack generation and normalization stack already exists and is tested by the existing `sigguard_eval` and `multi_attack_eval` modes. The AT loop reuses these exactly — no new attack code needed.

## Common Pitfalls

### Pitfall 1: EAD `binary_search_steps` Explosion

**What goes wrong:** `torchattacks.EADL1(model, max_iterations=7)` with default `binary_search_steps=9` runs 9 × 7 = 63 inner gradient steps per batch, not 7. This makes EAD roughly 9× slower than intended and can cause per-epoch training to take hours instead of minutes.

**Why it happens:** EAD has an outer binary search loop over the regularization constant `c`, and `binary_search_steps` defaults to 9. The `max_iterations` parameter only controls the inner loop.

**How to avoid:** Set `binary_search_steps=1` in training. The paper already uses `max_iterations=50` for evaluation (from `plot_all_attacks_iq_constellation.py:166`) but training must be faster.

**Warning signs:** First epoch takes >10 minutes when PGD(7) epoch takes <1 minute.

### Pitfall 2: Analog Accuracy Collapse (Silent)

**What goes wrong:** Training log shows high overall accuracy but WBFM/AM-DSB/AM-SSB classes silently drop to 0% because the adversarial perturbation corrupts even the "clean-substituted" analog examples if D-08 is implemented incorrectly.

**Why it happens:** The analog substitution mask must be applied *before* the adversarial forward pass, not after. If x_adv is computed for all samples first and then analog samples are overwritten, the clean forward pass sees correct signals but adversarial forward still uses perturbed analog signals.

**How to avoid:** Apply `x_adv[analog_mask] = x[analog_mask]` on the returned tensor immediately after `ta_output_to_iq_minmax` and before the adversarial forward pass.

**Warning signs:** Val log shows analog class accuracy (WBFM=3, AM-DSB=6, AM-SSB=10) near 0% at any SNR. Add per-class accuracy to val log at end of each epoch.

### Pitfall 3: `model.eval()` vs `torch.enable_grad()` Interaction

**What goes wrong:** Setting `model.eval()` before attack generation (to disable BatchNorm/Dropout randomness) also sets `torch.no_grad()` in some PyTorch versions, preventing gradient computation needed by PGD/EAD.

**Why it happens:** `model.eval()` and `torch.no_grad()` are independent, but confusion arises when val-mode code is copy-pasted into the attack loop.

**How to avoid:** Always wrap attack generation in `with torch.enable_grad():`. During training, the model should be in `.train()` mode for the adversarial forward; clean forward for loss can also be `.train()` mode (consistent with normal training).

**Warning signs:** `RuntimeError: element 0 of tensors does not require grad` during attack generation.

### Pitfall 4: ReduceLROnPlateau in Wrong Mode

**What goes wrong:** `ReduceLROnPlateau` defaults to `mode='min'` (monitoring a loss), but D-12 uses it to monitor a weighted accuracy (higher is better). Using `mode='min'` causes LR to be reduced when accuracy *improves*.

**Why it happens:** Copy-paste from `synth_finetune.py` which passes `val_acc` to `scheduler.step(val_acc)` — that script also uses `mode='max'` correctly. AT script must do the same.

**How to avoid:** `ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)`. Pass `val_weighted` to `scheduler.step(val_weighted)`.

### Pitfall 5: JSON Config Written Before Training Completes

**What goes wrong:** Config JSON is written at the start of training (before `epochs_trained`, `best_epoch`, `best_val_weighted` are known), so the file reports wrong values if training is interrupted.

**Why it happens:** CONTEXT.md specifics note this explicitly: "generated atomically after the final checkpoint save, not before training."

**How to avoid:** Write the JSON file only after the training loop exits and the final checkpoint is confirmed saved. Include a `try/finally` block so the config is still written even on keyboard interrupt.

## Code Examples

Verified patterns from existing codebase:

### torchattacks EAD Attack Instantiation (Training Budget)
```python
# Source: torchattacks 3.5.1 (VERIFIED: runtime), D-03 (CONTEXT.md)
import torchattacks

EPS = 0.1           # D-02
PGD_STEPS = 7       # D-03
EAD_ITERS = 7       # D-03 (max_iterations only; binary_search_steps=1 for speed)

attacks = {
    'fgsm':  torchattacks.FGSM(wrapped, eps=EPS),
    'pgd':   torchattacks.PGD(wrapped, eps=EPS, alpha=EPS/4, steps=PGD_STEPS),
    'eadl1': torchattacks.EADL1(wrapped, kappa=0, lr=0.01,
                                  max_iterations=EAD_ITERS, binary_search_steps=1),
    'eaden': torchattacks.EADEN(wrapped, kappa=0, lr=0.01,
                                  max_iterations=EAD_ITERS, binary_search_steps=1),
}
```

### Data Loading with SNR Filter
```python
# Source: data_loader.py Load_Dataset signature (VERIFIED: runtime inspect)
import logging
logger = logging.getLogger('at_loader'); logger.setLevel(logging.WARNING)
Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger, snr_min=0)
# Signals shape: [~110000, 2, 128], Labels: [~110000]
```

### Warm-Start from Pretrained Checkpoint
```python
# Source: synth_finetune.py lines 466-467 (VERIFIED: codebase)
model.load_state_dict(
    torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=device, weights_only=True))
```

### minmax Attack Loop (Full Pattern)
```python
# Source: plot_all_attacks_iq_constellation.py:173-191 (VERIFIED: codebase)
from util.adv_attack import (Model01Wrapper,
                              iq_to_ta_input_minmax, ta_output_to_iq_minmax)
wrapped = Model01Wrapper(model)

x01_4d, a, b = iq_to_ta_input_minmax(x)    # x: [N,2,128] -> x01_4d: [N,2,128,1]
wrapped.set_minmax(a, b)
with torch.enable_grad():
    adv01_4d = attack(x01_4d, y)
wrapped.clear_minmax()
x_adv = ta_output_to_iq_minmax(adv01_4d, a, b)  # [N,2,128]
```

### JSON Config Save (Atomic, Post-Training)
```python
# Source: D-16 (CONTEXT.md) + subprocess (VERIFIED: runtime)
import json, subprocess

def save_config(path, args, best_epoch, best_val_weighted, epochs_trained, seed):
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = 'unknown'
    cfg = {
        'epochs_total': args.epochs,
        'epochs_trained': epochs_trained,
        'lr_initial': args.lr,
        'batch_size': args.batch_size,
        'ta_box': 'minmax',
        'eps': args.eps,
        'attack_iters': {'fgsm': 1, 'pgd': args.pgd_steps,
                          'eadl1': args.ead_iters, 'eaden': args.ead_iters},
        'attack_list': ['fgsm', 'pgd', 'eadl1', 'eaden'],
        'alpha': args.alpha,
        'snr_filter': 'snr_min=0',
        'analog_mods_kept_clean': ['WBFM', 'AM-DSB', 'AM-SSB'],
        'best_epoch': best_epoch,
        'best_val_weighted': best_val_weighted,
        'seed': seed,
        'warm_start_ckpt': './checkpoint/2016.10a_AWN.pkl',
        'git_sha': sha,
    }
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
```

### Per-Epoch CSV Log
```python
# Source: D-17 (CONTEXT.md), synth_finetune.py CSV pattern
import csv

log_path = './checkpoint/2016.10a_AWN_at_log.csv'
with open(log_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'epoch','lr','train_loss','train_loss_clean','train_loss_adv',
        'val_clean_acc','val_robust_fgsm_acc','val_weighted','time_s'])
    writer.writeheader()

# Per epoch, append:
row = dict(epoch=ep, lr=lr_now, train_loss=..., ...)
with open(log_path, 'a', newline='') as f:
    csv.DictWriter(f, fieldnames=row.keys()).writerow(row)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Train from scratch (standard AT) | Warm-start from pretrained v1.0 checkpoint | v1.1 D-14 | ~10x less compute; preserves analog-class features from v1.0 training |
| Single attack type per experiment | Per-batch random attack selection from {FGSM, PGD, EAD-L1, EAD-EN} | v1.1 D-04 | Broader robustness surface without 4x training cost |
| Val loss for early stopping | Weighted `0.5·clean_acc + 0.5·robust_fgsm_acc` | v1.1 D-12 | Checkpoint selected for both accuracy dimensions; loss alone could improve while robustness degrades |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | D-03's "EAD-L1=7, EAD-EN=7" means `max_iterations=7, binary_search_steps=1` | Standard Stack / Pitfall 1 | If it means `max_iterations=7, binary_search_steps=9` (default), training is 9x slower; acceptable only on fast GPU |
| A2 | "AWN's internal regularization added once per forward (detail)" (D-06) means regu_sum from the adversarial forward only, not both forwards | Code Examples / Pattern 1 | If both regu_sums should be added, loss computation changes; low risk since regularization terms are small |
| A3 | The 85/15 train/val split in D-10 uses simple random permutation (not per-(mod,snr) stratification) | Architecture Patterns / Pattern 4 | If stratification is required, must use `Dataset_Split(val_size=0.15, test_size=0.0)` instead; SNR-filtered dataset is already balanced so simple split is likely fine |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.
(Table has 3 items — planner should clarify A1 and A2 before finalizing the training loop body.)

## Open Questions (RESOLVED)

1. **EAD iteration budget interpretation (A1)** — RESOLVED: `binary_search_steps=1`, `--ead_bss` CLI flag defaulting to 1. Plan 01 Task 1 implements this.
   - What we know: D-03 says "EAD-L1=7, EAD-EN=7" under "Madry-style fast-AT budget"
   - Resolution: `max_iterations=7, binary_search_steps=1` to match the "fast" framing

2. **regu_sum from one or both forward passes (A2)** — RESOLVED: `sum(regu_adv)` only (adversarial forward), not both forwards. Plan 01 Task 1 implements this.
   - What we know: D-06 says "AWN's internal regularization added once per forward (detail)"
   - Resolution: Add from adversarial forward only (cleaner loss semantics, lower scale)

3. **Per-class accuracy logging in val loop** — RESOLVED: per-class analog accuracy printed in `run_sanity_eval` console output. Plan 02 Task 2 implements this.
   - What we know: D-17 columns do not include per-class accuracy
   - Resolution: Per-class analog accuracy breakdown in post-training sanity eval console print (CSV columns kept stable per D-17)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| torchattacks | AT-01 attack generation | Yes | 3.5.1 | None needed |
| `./checkpoint/2016.10a_AWN.pkl` | AT-04 warm-start | Yes | — (pure state_dict confirmed) | None (required) |
| `./data/RML2016.10a_dict.pkl` | AT-01 training data | Yes (assumed from existing eval runs) | — | None (required) |
| CUDA GPU | Training speed | [ASSUMED] present from existing usage | — | CPU fallback via `torch.cuda.is_available()` |

**Missing dependencies with no fallback:** None identified.

**Missing dependencies with fallback:** None — all confirmed or assumed available from existing successful training history.

## Security Domain

Security enforcement is not applicable to this phase. This phase implements a training script (producing a checkpoint file), not a service or user-facing interface. There are no network endpoints, authentication surfaces, input validation requirements from untrusted users, or cryptographic operations. The only security-adjacent concern is the `subprocess.run` call to read `git rev-parse --short HEAD` for the JSON config — this command is fixed (no user input) and poses no injection risk.

## Sources

### Primary (HIGH confidence)
- `util/adv_attack.py` (codebase) — `Model01Wrapper`, `iq_to_ta_input_minmax`, `ta_output_to_iq_minmax` implementation details [VERIFIED: Read]
- `synth_finetune.py` (codebase) — structural template: data prep, DataLoader, train loop, early stopping, eval, checkpoint save [VERIFIED: Read]
- `data_loader/data_loader.py` (codebase) — `Load_Dataset(snr_min=0)` parameter confirmed [VERIFIED: runtime inspect]
- `util/utils.py` (codebase) — `create_model`, `fix_seed` signatures [VERIFIED: Read]
- torchattacks 3.5.1 runtime — `FGSM`, `PGD`, `EADL1`, `EADEN` class availability and constructor signatures [VERIFIED: runtime python -c]
- `./checkpoint/2016.10a_AWN.pkl` — confirmed as pure `state_dict` (OrderedDict, 26 keys), loads with `weights_only=True` [VERIFIED: runtime torch.load]

### Secondary (MEDIUM confidence)
- `plot_all_attacks_iq_constellation.py:131-191` — reference implementation for minmax attack loop pattern (run_attack function) [VERIFIED: Read]

### Tertiary (LOW confidence)
- Wall-time estimates for AT training (1-5 min/epoch on GPU) — extrapolated from batch count × assumed fwd/bwd time; not empirically measured [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified at runtime
- Architecture: HIGH — patterns copied directly from verified codebase files
- Pitfalls: HIGH for P1/P2/P4/P5 (sourced from code inspection); MEDIUM for P3 (standard PyTorch behavior)

**Research date:** 2026-04-16
**Valid until:** 2026-05-16 (stable libraries; torchattacks API unlikely to change)
