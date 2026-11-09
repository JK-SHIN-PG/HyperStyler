#!/bin/sh
# Build the stage-2 input from the published generation author pairs.
# Usage: bash preprocessing/prepare_generation_input.sh <blog|news|mud>
set -e

DATASET=${1:?usage: $0 <blog|news|mud>}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK_DIR="$ROOT/data/intermediate/$DATASET"
GPU=${GPU_ID:-0}

# 1) Source and target texts for each author pair
python "$ROOT/preprocessing/prepare_generation_pairs.py" \
    --pairs-path "$ROOT/data/splits/$DATASET/generation_author_pairs.jsonl" \
    --dataset-path "$ROOT/data/processed/$DATASET/dataset_with_idx" \
    --output-dir "$WORK_DIR/generation_pairs"

# 2) Paraphrase the source texts
#    -> data/intermediate/<dataset>/generation_pair_paraphrases/topp0.8_tmp1.5
CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/preprocessing/common/generate_paraphrases.py" \
    --in_dir "$WORK_DIR/generation_pairs" \
    --out_dir "$WORK_DIR/generation_pair_paraphrases" \
    --num_return_sequences 5 --temp 1.5 --top_p 0.8
