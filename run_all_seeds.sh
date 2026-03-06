#!/usr/bin/env bash
# run_all_seeds.sh — Run key experiments across 5 seeds for statistical rigor.
#
# Usage:
#   bash run_all_seeds.sh [DATASET] [CKPT_PATH]
#   bash run_all_seeds.sh 2016.10a ./checkpoint
#
# Produces per-seed results in inference/<dataset>_seed<N>/ directories.
# After all seeds complete, aggregates mean +/- std.

set -euo pipefail

DATASET="${1:-2016.10a}"
CKPT_PATH="${2:-./checkpoint}"
SEEDS=(2022 2023 2024 2025 2026)
MODELS="awn,vtcnn2,resnet1d,lstm"
ATTACKS="fgsm,pgd,cw"

echo "======================================================"
echo " Multi-Seed Experiment Runner"
echo " Dataset: ${DATASET}"
echo " Seeds:   ${SEEDS[*]}"
echo " Models:  ${MODELS}"
echo " Attacks: ${ATTACKS}"
echo "======================================================"

# ---- Phase 1: Train all models (each seed) ----
echo ""
echo "=== Phase 1: Training ==="
for seed in "${SEEDS[@]}"; do
    for model in awn vtcnn2 resnet1d lstm; do
        echo "[seed=${seed}] Training ${model}..."
        python3 main.py --mode train --dataset "${DATASET}" \
            --model "${model}" --seed "${seed}" \
            --dir_name "${DATASET}_${model}_seed${seed}" \
            2>&1 | tail -1
    done
done

# ---- Phase 2: Transfer evaluation (each seed) ----
echo ""
echo "=== Phase 2: Transfer Evaluation ==="
for seed in "${SEEDS[@]}"; do
    echo "[seed=${seed}] Running transfer eval..."
    python3 main.py --mode transfer_eval --dataset "${DATASET}" \
        --ckpt_path "${CKPT_PATH}" --seed "${seed}" \
        --model_list "${MODELS}" --attack_list "${ATTACKS}" \
        --eval_limit 2000 \
        --dir_name "${DATASET}_transfer_seed${seed}" \
        2>&1 | tail -3
done

# ---- Phase 3: Adaptive evaluation (each seed) ----
echo ""
echo "=== Phase 3: Adaptive Evaluation ==="
for seed in "${SEEDS[@]}"; do
    echo "[seed=${seed}] Running adaptive eval..."
    python3 main.py --mode adaptive_eval --dataset "${DATASET}" \
        --ckpt_path "${CKPT_PATH}" --seed "${seed}" \
        --attack_list "${ATTACKS}" --sigguard_topk "20,50" \
        --eval_limit 2000 \
        --dir_name "${DATASET}_adaptive_seed${seed}" \
        2>&1 | tail -5
done

# ---- Phase 4: SigGuard evaluation (each seed) ----
echo ""
echo "=== Phase 4: SigGuard Evaluation ==="
for seed in "${SEEDS[@]}"; do
    echo "[seed=${seed}] Running sigguard eval..."
    python3 main.py --mode sigguard_eval --dataset "${DATASET}" \
        --ckpt_path "${CKPT_PATH}" --seed "${seed}" \
        --attack_list "${ATTACKS}" --no_plot_iq \
        --dir_name "${DATASET}_sigguard_seed${seed}" \
        2>&1 | tail -5
done

# ---- Phase 5: Power budget evaluation (each seed) ----
echo ""
echo "=== Phase 5: Power Budget Evaluation ==="
for seed in "${SEEDS[@]}"; do
    echo "[seed=${seed}] Running power budget eval..."
    python3 main.py --mode power_budget_eval --dataset "${DATASET}" \
        --ckpt_path "${CKPT_PATH}" --seed "${seed}" \
        --attack_list "fgsm,pgd" --eval_limit 2000 \
        --dir_name "${DATASET}_power_seed${seed}" \
        2>&1 | tail -3
done

# ---- Aggregate results ----
echo ""
echo "======================================================"
echo " Aggregating results across seeds..."
echo "======================================================"

python3 -c "
import glob, pandas as pd, numpy as np, os

base = 'inference'

# Aggregate transfer eval
files = sorted(glob.glob(f'{base}/${DATASET}_transfer_seed*/result/transfer_eval.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs)
    agg = combined.groupby(['attack', 'source_model', 'target_model']).agg(
        attack_acc_mean=('attack_acc', 'mean'),
        attack_acc_std=('attack_acc', 'std'),
        n_seeds=('attack_acc', 'count'),
    ).reset_index()
    agg.to_csv(f'{base}/transfer_eval_aggregated.csv', index=False)
    print(f'Transfer: {len(files)} seeds aggregated -> transfer_eval_aggregated.csv')
    print(agg.to_string(index=False))

# Aggregate adaptive eval
files = sorted(glob.glob(f'{base}/${DATASET}_adaptive_seed*/result/adaptive_eval.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs)
    agg = combined.groupby(['attack', 'topk']).agg(
        std_no_def_mean=('std_no_def_acc', 'mean'),
        std_no_def_std=('std_no_def_acc', 'std'),
        std_with_def_mean=('std_with_def_acc', 'mean'),
        adaptive_mean=('adaptive_acc', 'mean'),
        adaptive_std=('adaptive_acc', 'std'),
    ).reset_index()
    agg.to_csv(f'{base}/adaptive_eval_aggregated.csv', index=False)
    print(f'\nAdaptive: {len(files)} seeds aggregated -> adaptive_eval_aggregated.csv')
    print(agg.to_string(index=False))

# Aggregate sigguard eval
files = sorted(glob.glob(f'{base}/${DATASET}_sigguard_seed*/result/sigguard_eval.csv'))
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs)
    num_cols = [c for c in combined.columns if c != 'sample_type']
    agg = combined.groupby('sample_type')[num_cols].agg(['mean', 'std']).reset_index()
    agg.to_csv(f'{base}/sigguard_eval_aggregated.csv', index=False)
    print(f'\nSigGuard: {len(files)} seeds aggregated -> sigguard_eval_aggregated.csv')

print('\nDone! Results saved in inference/ directory.')
"

echo ""
echo "All experiments complete."
