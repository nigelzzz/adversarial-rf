# MCLDNN / VTCNN2 Attack Evaluation (same protocol as AWN)

Runs CW, PGD, FGSM, EAD-L1, EAD-EN on MCLDNN and VTCNN2 with the same
`defense_compare` protocol used for AWN in the paper:

- Full RML2016.10a split, seed 2022 (same test set as AWN)
- SNR 0, 2, ..., 18 dB only (filtered inside `defense_compare`)
- 200 samples per (SNR, modulation) cell
- Defenses: no_defense, Adaptive-K v2, and classical baselines

## Commands

```bash
# MCLDNN
python main.py --mode defense_compare --dataset 2016.10a --model mcldnn \
  --ckpt_path ./checkpoint --attack_list cw,pgd,fgsm,eadl1,eaden \
  --skip_confmat --skip_budget --dir_name dc_mcldnn

# VTCNN2
python main.py --mode defense_compare --dataset 2016.10a --model vtcnn2 \
  --ckpt_path ./checkpoint --attack_list cw,pgd,fgsm,eadl1,eaden \
  --skip_confmat --skip_budget --dir_name dc_vtcnn2
```

Checkpoints loaded: `checkpoint/2016.10a_MCLDNN.pkl`, `checkpoint/2016.10a_VTCNN2.pkl`.

Quick trial run: add `--max_per_cell 50`.

## Output

`inference/dc_<model>/result/defense_compare/`

- `defense_compare.csv` — columns `attack, snr, defense, accuracy, n_samples`
- `defense_compare_<attack>.csv` — pivot table per attack (rows = defenses, cols = SNR)

## Notes

- Do NOT add `--snr_min 0`. It filters the data before the split, so the
  test set would differ from AWN's.
- Classical filter parameters (Kalman, Wiener, ...) are auto-loaded from the
  latest `inference/*/result/calibration_params.json`, which was calibrated
  on AWN. For a fair baseline per model, first run
  `python main.py --mode calibrate_defenses --dataset 2016.10a --model <m> --ckpt_path ./checkpoint`
  and pass the result with `--calibration_path`. Adaptive-K v2 needs no calibration.
- MCLDNN contains an LSTM. `defense_compare` disables cuDNN for LSTM models
  (cuDNN RNN backward fails in eval mode), so MCLDNN attacks are slower.
- EAD-L1 / EAD-EN are the slowest attacks.
