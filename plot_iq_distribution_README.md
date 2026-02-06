# IQ Distribution Plotter for RML2016.10a

Generate 2x2 constellation scatter plots (BPSK / QPSK / QAM16 / QAM64) from
RadioML RML2016.10a data, with configurable SNR selection, weighted multi-SNR
mixing, and optional phase/CFO correction.

## Requirements

```bash
pip install numpy matplotlib
```

## Quick Start

### One-shot comparison (recommended)

Produce two figures side-by-side for easy visual comparison:

```bash
python plot_iq_distribution.py --data ./data/RML2016.10a_dict.pkl --make_comparison
```

Outputs:
- `iq_snr18.png` -- SNR=18 only (400 frames/mod), clean constellation
- `iq_mixed.png` -- SNR 0/6/12/18 weighted mix (300 frames/mod), fuzzier cloud

### Single SNR=18

```bash
python plot_iq_distribution.py --data ./data/RML2016.10a_dict.pkl \
    --snrs 18 --n_examples_total_per_mod 400 --out iq_snr18.png
```

### Mixed weighted SNRs

```bash
python plot_iq_distribution.py --data ./data/RML2016.10a_dict.pkl \
    --snrs 0 6 12 18 \
    --snr_weights 0=0.5 6=0.25 12=0.15 18=0.10 \
    --n_examples_total_per_mod 300 \
    --out iq_mixed.png
```

### With phase/CFO correction

```bash
python plot_iq_distribution.py --data ./data/RML2016.10a_dict.pkl \
    --snrs 18 --phase_correct --cfo_correct --out iq_corrected.png
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | (required) | Path to `RML2016.10a_dict.pkl` |
| `--mods` | BPSK QPSK QAM16 QAM64 | Modulation types to plot |
| `--snrs` | 18 | SNR values to sample from |
| `--snr_weights` | uniform | Weights: `0=0.5 6=0.25` or `0.5 0.25` |
| `--n_examples_total_per_mod` | 200 | Total frames per modulation |
| `--seed` | 0 | Random seed |
| `--out` | iq_distribution.png | Output file (single-figure mode) |
| `--alpha` | 0.20 | Scatter point transparency |
| `--point_size` | 3 | Scatter point size |
| `--max_points` | 0 (no cap) | Subsample points per subplot |
| `--equalize_axes` | true | Same x/y limits across subplots |
| `--phase_correct` | false | Per-frame constant phase derotation |
| `--cfo_correct` | false | Simple CFO compensation |
| `--strict` | false | Error on missing (mod, SNR) keys |
| `--make_comparison` | false | Produce both SNR=18 and mixed figures |

### Comparison-mode arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--out_snr18` | iq_snr18.png | Output path for SNR=18 figure |
| `--out_mixed` | iq_mixed.png | Output path for mixed-SNR figure |
| `--snr18_value` | 18 | SNR value for single-SNR figure |
| `--snr18_total` | 400 | Frames per mod for SNR=18 figure |
| `--mixed_snrs` | 0 6 12 18 | SNRs for mixed figure |
| `--mixed_weights` | 0=0.5 6=0.25 12=0.15 18=0.10 | Weights for mixed figure |
| `--mixed_total` | 300 | Frames per mod for mixed figure |

## Sampling Logic

For each modulation, frames are allocated across requested SNRs proportional
to the specified weights using the **largest-remainder method**, which ensures
deterministic integer allocation that sums exactly to the requested total.
If a specific (modulation, SNR) key is missing, it is skipped with a warning
(or raises an error with `--strict`). If fewer frames are available than
requested, sampling with replacement is used.

## Corrections

- **Phase correction** (`--phase_correct`): estimates a constant phase offset
  via the 4th-power method and derotates. Works well for PSK/QAM.
- **CFO correction** (`--cfo_correct`): estimates a linear phase slope from
  consecutive sample phase differences and removes it.

Both corrections fail gracefully (skip) if estimation is unstable.
