Here’s a concise README snippet you can drop in.

  Adversarial Quickstart

  - Prereqs:
      - pip install torchattacks
      - Have a pretrained checkpoint (e.g., 2016.10a_AWN.pkl) in ./checkpoint or repo root.
  - Clean baseline:
      - python main.py --mode eval --dataset 2016.10a --snr_filter 0 --mod_filter QPSK --eval_limit 64

  CW Attack

  - Basic (torchattacks backend):
      - python main.py --mode adv_eval --dataset 2016.10a --attack cw --attack_backend torchattacks --cw_steps 100 --cw_c 1.0
  - Make it weaker/easier to recover:
      - add --cw_scale 0.3 (post-scales the CW delta to 30%)
      - reduce steps/c (e.g., --cw_steps 5 --cw_c 0.01)
  - Fallback (no torchattacks):
      - add --attack_backend internal

  Low-Frequency Attack

  - Single tone jammer:
      - python main.py --mode adv_eval --dataset 2016.10a --attack spectral --spec_type cw_tone --tone_freq 0.02 --spec_eps 0.25
  - Band-limited noise:
      - python main.py --mode adv_eval --dataset 2016.10a --attack spectral --spec_type psd_band --spec_band_low 0.00 --spec_band_high 0.05 --spec_eps 0.25

  IDFFT Recovery (FFT-domain)

  - Notch a band (IDFT-based):
      - python main.py --mode adv_eval --dataset 2016.10a ... --defense fft_notch --def_band_low 0.00 --def_band_high 0.08 --cmp_defense True
  - Soft notch (tapered transitions):
      - ... --defense fft_soft_notch --def_band_low 0.00 --def_band_high 0.08 --def_notch_depth 0.6 --def_notch_trans 4 --cmp_defense True
  - Adaptive notch (auto-detect low-frequency peak):
      - ... --defense auto_soft_notch --def_auto_fmax 0.1 --cmp_defense True
  - Top‑K FFT keep (IFFT, AWN_All style):
      - ... --defense fft_topk --def_topk 50 --cmp_defense True
  - Tip: Use --cmp_defense True to print accuracy before and after recovery in one run.

  Visualization

  - Per-sample time/FFT/IQ (saves PNGs to cw_analysis/):
      - python visualize_cw_fft.py
  - Grid differences across mods (I/Q scatter grids + avg |FFT|):
      - python visualize_cw_iq_freq_grid.py --snr 18 --samples_per_mod 500 --cw_steps 100 --topk 50 --ta_box minmax
      - Outputs: cw_analysis/iq_grid.png, cw_analysis/freq_grid.png, cw_analysis/accuracy_diff.json
  - Spectrum/energy summary for a few samples:
      - python analyze_freq_spectrum.py

  Tips

  - Use --mod_filter and --snr_filter to focus on a slice (e.g., QPSK @ SNR=0/18).
  - Use --eval_limit to speed up quick checks.
  - For smoother CW that’s easier to notch, try --lowpass True --lowpass_kernel 33 and smaller --cw_scale.
  - Frequency top-k eval (keep top 10-50% FFT bins, SNR>=0, GPU):
      - python main.py --mode freq_topk_eval --dataset 2016.10a --ckpt_path ./checkpoint --snr_min 0 --freq_percents 0.1,0.2,0.3,0.4,0.5 --device cuda
      - Saves per-mod/overall accuracies to training/<run>/result/freq_topk/freq_topk_eval.json
To continue this session, run codex resume 019a3356-647e-7b21-a4e2-a02409892434.
