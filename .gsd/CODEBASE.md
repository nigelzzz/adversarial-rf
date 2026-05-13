# Codebase Map

Generated: 2026-05-13T11:16:46Z | Files: 500 | Described: 0/500
<!-- gsd:codebase-meta {"generatedAt":"2026-05-13T11:16:46Z","fingerprint":"ebe9f17426f02a6773b51051eeff34a6b73939c6","fileCount":500,"truncated":true} -->
Note: Truncated to first 500 files. Run with higher --max-files to include all.

### (root)/
- *(41 files: 17 .py, 9 .md, 4 (no ext), 4 .tflite, 3 .pkl, 2 .json, 1 .zip, 1 .pt)*

### MCLDNN/
- `MCLDNN/dataset2016.py`
- `MCLDNN/mcldnn_pytorch.py`
- `MCLDNN/MCLDNN.py`
- `MCLDNN/mltools.py`
- `MCLDNN/README.md`
- `MCLDNN/test.py`
- `MCLDNN/train.py`

### MCLDNN/gittt/
- `MCLDNN/gittt/config`
- `MCLDNN/gittt/description`
- `MCLDNN/gittt/HEAD`
- `MCLDNN/gittt/index`
- `MCLDNN/gittt/packed-refs`

### MCLDNN/gittt/hooks/
- `MCLDNN/gittt/hooks/applypatch-msg.sample`
- `MCLDNN/gittt/hooks/commit-msg.sample`
- `MCLDNN/gittt/hooks/fsmonitor-watchman.sample`
- `MCLDNN/gittt/hooks/post-update.sample`
- `MCLDNN/gittt/hooks/pre-applypatch.sample`
- `MCLDNN/gittt/hooks/pre-commit.sample`
- `MCLDNN/gittt/hooks/pre-merge-commit.sample`
- `MCLDNN/gittt/hooks/pre-push.sample`
- `MCLDNN/gittt/hooks/pre-rebase.sample`
- `MCLDNN/gittt/hooks/pre-receive.sample`
- `MCLDNN/gittt/hooks/prepare-commit-msg.sample`
- `MCLDNN/gittt/hooks/push-to-checkout.sample`
- `MCLDNN/gittt/hooks/update.sample`

### MCLDNN/gittt/info/
- `MCLDNN/gittt/info/exclude`

### MCLDNN/gittt/logs/
- `MCLDNN/gittt/logs/HEAD`

### MCLDNN/gittt/logs/refs/heads/
- `MCLDNN/gittt/logs/refs/heads/master`

### MCLDNN/gittt/logs/refs/remotes/origin/
- `MCLDNN/gittt/logs/refs/remotes/origin/HEAD`

### MCLDNN/gittt/objects/pack/
- `MCLDNN/gittt/objects/pack/pack-a4f27f1f97f2defc9680ebc6cac51ac5bf63f08b.idx`
- `MCLDNN/gittt/objects/pack/pack-a4f27f1f97f2defc9680ebc6cac51ac5bf63f08b.pack`

### MCLDNN/gittt/refs/heads/
- `MCLDNN/gittt/refs/heads/master`

### MCLDNN/gittt/refs/remotes/origin/
- `MCLDNN/gittt/refs/remotes/origin/HEAD`

### MCLDNN/predictresult/
- `MCLDNN/predictresult/Predictresult.txt`

### awn_fpga/
- `awn_fpga/2016.10a_AWN_at_log.csv`
- `awn_fpga/2016.10a_AWN_at.config.json`
- `awn_fpga/2016.10a_AWN_at.pkl`
- `awn_fpga/2016.10a_AWN_ft.pkl`
- `awn_fpga/2016.10a_AWN.pkl`
- `awn_fpga/2016.10a_MCLDNN.pkl`
- `awn_fpga/2016.10a_VTCNN2.pkl`
- `awn_fpga/analysis_notes.md`
- `awn_fpga/AWN_quan_int8_simple.tflite`
- `awn_fpga/awn_to_tpu_case.py`
- `awn_fpga/detector_ae.pth`
- `awn_fpga/latency_notes.md`
- `awn_fpga/lifting.py`
- `awn_fpga/model.py`
- `awn_fpga/README.md`
- `awn_fpga/systolic_optimization.md`

### awn_fpga/rtl/
- `awn_fpga/rtl/avgpool1d_s8.v`
- `awn_fpga/rtl/bram_feeder_a.v`
- `awn_fpga/rtl/bram_feeder_b.v`
- `awn_fpga/rtl/eltwise_addsub_s8.v`
- `awn_fpga/rtl/gemm_s8.v`
- `awn_fpga/rtl/global_buffer.v`
- `awn_fpga/rtl/leaky_relu_s8.v`
- `awn_fpga/rtl/lut_s8.v`
- `awn_fpga/rtl/mul_s8.v`
- `awn_fpga/rtl/pe_s8.v`
- `awn_fpga/rtl/relu_s8.v`
- `awn_fpga/rtl/requantize_s32_s8.v`
- `awn_fpga/rtl/systolic_mesh_s8.v`

### awn_fpga/sw/
- `awn_fpga/sw/iohex.py`
- `awn_fpga/sw/profile_awn.py`
- `awn_fpga/sw/quantize_awn.py`
- `awn_fpga/sw/refmodel.py`
- `awn_fpga/sw/run_op_test.py`
- `awn_fpga/sw/test_systolic.py`

### awn_fpga/tb/
- `awn_fpga/tb/tb_avgpool1d_s8.v`
- `awn_fpga/tb/tb_eltwise_addsub_s8.v`
- `awn_fpga/tb/tb_gemm_s8.v`
- `awn_fpga/tb/tb_leaky_relu_s8.v`
- `awn_fpga/tb/tb_lut_s8.v`
- `awn_fpga/tb/tb_mul_s8.v`
- `awn_fpga/tb/tb_pe_s8.v`
- `awn_fpga/tb/tb_relu_s8.v`
- `awn_fpga/tb/tb_requantize_s32_s8.v`
- `awn_fpga/tb/tb_systolic_mesh_s8.v`

### awn_fpga/vectors/
- *(126 files: 126 .hex)*

### checkpoint/
- `checkpoint/.gitignore`

### config/
- `config/2016.10a.yml`
- `config/2016.10b.yml`
- `config/2018.01a.yml`

### crc_experiment_results/
- `crc_experiment_results/crc_vs_amc.csv`
- `crc_experiment_results/crc_vs_amc.json`

### cw_analysis/
- `cw_analysis/accuracy_diff_qam64_top13_all_snr_ge0_last.json`
- `cw_analysis/accuracy_diff_qam64_top13_all_snr_ge0.json`
- `cw_analysis/accuracy_diff_qam64_top13.json`
- `cw_analysis/accuracy_diff.json`
- `cw_analysis/energy_keep_metrics.json`
- `cw_analysis/freq_grid.pdf`
- `cw_analysis/iq_grid.pdf`
- `cw_analysis/notch_subset_metrics.json`
- `cw_analysis/notch_variants_metrics.json`
- `cw_analysis/top20_subset_metrics.json`
- `cw_analysis/topk_variants_metrics.json`

### data/
- `data/.gitignore`

### data_loader/
- `data_loader/data_loader.py`

### doc-CN/
- `doc-CN/multi_attack_eval_guide.md`
- `doc-CN/README.md`

### inference/
- `inference/.~lock.all_33_tables_snr18_eps003.csv#`
- `inference/.~lock.all_33_tables_snr18_eps010.csv#`
- `inference/.gitignore`

### inference/2016.10a_0/log/
- `inference/2016.10a_0/log/log.txt`

### inference/2016.10a_1/log/
- `inference/2016.10a_1/log/log.txt`

### inference/2016.10a_10/log/
- `inference/2016.10a_10/log/log.txt`

### inference/2016.10a_106/log/
- `inference/2016.10a_106/log/log.txt`

### inference/2016.10a_11/log/
- `inference/2016.10a_11/log/log.txt`

### inference/2016.10a_12/log/
- `inference/2016.10a_12/log/log.txt`

### inference/2016.10a_120/log/
- `inference/2016.10a_120/log/log.txt`

### inference/2016.10a_121/log/
- `inference/2016.10a_121/log/log.txt`

### inference/2016.10a_121/result/
- `inference/2016.10a_121/result/multi_attack_snr_mod_eval.csv`
- `inference/2016.10a_121/result/sigguard_eval_table.txt`
- `inference/2016.10a_121/result/sigguard_eval.csv`

### inference/2016.10a_121/result/freq_topk/
- `inference/2016.10a_121/result/freq_topk/freq_topk_eval.json`

### inference/2016.10a_121/result/freq_topk_adv/
- *(129 files: 129 .json)*

### inference/2016.10a_122/log/
- `inference/2016.10a_122/log/log.txt`

### inference/2016.10a_123/log/
- `inference/2016.10a_123/log/log.txt`

### inference/2016.10a_124/log/
- `inference/2016.10a_124/log/log.txt`

### inference/2016.10a_125/log/
- `inference/2016.10a_125/log/log.txt`

### inference/2016.10a_126/log/
- `inference/2016.10a_126/log/log.txt`

### inference/2016.10a_127/log/
- `inference/2016.10a_127/log/log.txt`

### inference/2016.10a_127/result/
- `inference/2016.10a_127/result/sigguard_eval_table.txt`
- `inference/2016.10a_127/result/sigguard_eval.csv`

### inference/2016.10a_128/log/
- `inference/2016.10a_128/log/log.txt`

### inference/2016.10a_129/log/
- `inference/2016.10a_129/log/log.txt`

### inference/2016.10a_13/log/
- `inference/2016.10a_13/log/log.txt`

### inference/2016.10a_132/log/
- `inference/2016.10a_132/log/log.txt`

### inference/2016.10a_133/log/
- `inference/2016.10a_133/log/log.txt`

### inference/2016.10a_134/log/
- `inference/2016.10a_134/log/log.txt`

### inference/2016.10a_134/result/
- `inference/2016.10a_134/result/multi_attack_snr_mod_eval.csv`
- `inference/2016.10a_134/result/sigguard_eval_table.txt`
- `inference/2016.10a_134/result/sigguard_eval.csv`

### inference/2016.10a_135/log/
- `inference/2016.10a_135/log/log.txt`

### inference/2016.10a_136/log/
- `inference/2016.10a_136/log/log.txt`

### inference/2016.10a_136/result/
- `inference/2016.10a_136/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_137/log/
- `inference/2016.10a_137/log/log.txt`

### inference/2016.10a_138/log/
- `inference/2016.10a_138/log/log.txt`

### inference/2016.10a_139/log/
- `inference/2016.10a_139/log/log.txt`

### inference/2016.10a_139/result/
- `inference/2016.10a_139/result/.~lock.multi_attack_snr_mod_eval.csv#`
- `inference/2016.10a_139/result/multi_attack_snr_mod_eval.csv`
- `inference/2016.10a_139/result/sigguard_eval_table.txt`
- `inference/2016.10a_139/result/sigguard_eval.csv`

### inference/2016.10a_14/log/
- `inference/2016.10a_14/log/log.txt`

### inference/2016.10a_14/result/
- `inference/2016.10a_14/result/psd_mask_QAM16_18.npy`

### inference/2016.10a_140/log/
- `inference/2016.10a_140/log/log.txt`

### inference/2016.10a_141/log/
- `inference/2016.10a_141/log/log.txt`

### inference/2016.10a_142/log/
- `inference/2016.10a_142/log/log.txt`

### inference/2016.10a_142/result/
- `inference/2016.10a_142/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_143/log/
- `inference/2016.10a_143/log/log.txt`

### inference/2016.10a_143/result/
- `inference/2016.10a_143/result/sigguard_eval_table.txt`
- `inference/2016.10a_143/result/sigguard_eval.csv`

### inference/2016.10a_145/log/
- `inference/2016.10a_145/log/log.txt`

### inference/2016.10a_145/result/
- `inference/2016.10a_145/result/adaptive_k_calibration.json`

### inference/2016.10a_146/log/
- `inference/2016.10a_146/log/log.txt`

### inference/2016.10a_146/result/
- `inference/2016.10a_146/result/adaptive_k_calibration.json`

### inference/2016.10a_147/log/
- `inference/2016.10a_147/log/log.txt`

### inference/2016.10a_147/result/
- `inference/2016.10a_147/result/sigguard_eval_table.txt`
- `inference/2016.10a_147/result/sigguard_eval.csv`

### inference/2016.10a_148/log/
- `inference/2016.10a_148/log/log.txt`

### inference/2016.10a_148/result/
- `inference/2016.10a_148/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_149/log/
- `inference/2016.10a_149/log/log.txt`

### inference/2016.10a_149/result/
- `inference/2016.10a_149/result/adaptive_k_calibration.json`
- `inference/2016.10a_149/result/sigguard_eval_table.txt`
- `inference/2016.10a_149/result/sigguard_eval.csv`

### inference/2016.10a_15/log/
- `inference/2016.10a_15/log/log.txt`

### inference/2016.10a_150/log/
- `inference/2016.10a_150/log/log.txt`

### inference/2016.10a_150/result/
- `inference/2016.10a_150/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_151/log/
- `inference/2016.10a_151/log/log.txt`

### inference/2016.10a_151/result/
- `inference/2016.10a_151/result/sigguard_eval_table.txt`
- `inference/2016.10a_151/result/sigguard_eval.csv`

### inference/2016.10a_152/log/
- `inference/2016.10a_152/log/log.txt`

### inference/2016.10a_152/result/
- `inference/2016.10a_152/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_153/log/
- `inference/2016.10a_153/log/log.txt`

### inference/2016.10a_153/result/
- `inference/2016.10a_153/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_154/log/
- `inference/2016.10a_154/log/log.txt`

### inference/2016.10a_154/result/
- `inference/2016.10a_154/result/multi_attack_snr_mod_eval.csv`

### inference/2016.10a_155/log/
- `inference/2016.10a_155/log/log.txt`

### inference/2016.10a_157/log/
- `inference/2016.10a_157/log/log.txt`

### inference/2016.10a_158/log/
- `inference/2016.10a_158/log/log.txt`

### inference/2016.10a_16/log/
- `inference/2016.10a_16/log/log.txt`

### inference/2016.10a_16/result/
- `inference/2016.10a_16/result/cw_psd_mask_QAM16_18.npy`

### inference/2016.10a_165/log/
- `inference/2016.10a_165/log/log.txt`

### inference/2016.10a_165/result/
- `inference/2016.10a_165/result/attack_bench_env.json`
- `inference/2016.10a_165/result/attack_bench.csv`
- `inference/2016.10a_165/result/calibration_params.json`

### inference/2016.10a_165/result/defense_compare/budget_curves/
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_curves_agg.csv`
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_curves_detail.csv`
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_cw.csv`
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_eaden.csv`
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_eadl1.csv`
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_fgsm.csv`
- `inference/2016.10a_165/result/defense_compare/budget_curves/budget_pgd.csv`

### inference/2016.10a_165/result/defense_compare/confmat/
- `inference/2016.10a_165/result/defense_compare/confmat/confmat_summary.csv`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr0_after_pct.csv`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr0_after.npy`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr0_before_pct.csv`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr0_before.npy`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr10_after_pct.csv`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr10_after.npy`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr10_before_pct.csv`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr10_before.npy`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr18_after_pct.csv`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr18_after.npy`
- `inference/2016.10a_165/result/defense_compare/confmat/cw_snr18_before.npy`
