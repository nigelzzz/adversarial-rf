#!/bin/bash
# Full pipeline: train + evaluate with SNR >= 0 only, 6:2:2 split
# Usage: bash run_snr0_pipeline.sh
set -e

source ~/opensource/gpu_env/tile-lang/bin/activate

echo "=== Step 1: Train AWN on SNR >= 0 ==="
python main.py --mode train --dataset 2016.10a --snr_min 0

echo ""
echo "=== Step 2: Find latest checkpoint ==="
CKPT_DIR=$(ls -td training/2016.10a_*/models | head -1)
echo "Using checkpoint: $CKPT_DIR"

echo ""
echo "=== Step 3: Evaluate clean accuracy ==="
python main.py --mode eval --dataset 2016.10a --snr_min 0 --ckpt_path "$CKPT_DIR"

echo ""
echo "=== Step 4: Run defense comparison (all 5 attacks x 9 defenses) ==="
python main.py --mode defense_compare --dataset 2016.10a --snr_min 0 --ckpt_path "$CKPT_DIR"

echo ""
echo "=== Done ==="
