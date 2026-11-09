#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
RAW_JSONL=${1:-"$ROOT/data/raw/mud/data.jsonl"}
PROTECTED_AUTHORS=${2:-"$ROOT/data/evaluation/reddit/protected_authors.txt"}
WORK_DIR=${3:-"$ROOT/data/intermediate/mud"}
OUTPUT_DIR=${4:-"$ROOT/data/processed/mud"}
GPU=${GPU_ID:-0}
COMMON="$ROOT/preprocessing/common"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

# 1) Assign authors to train, validation, and held-out sets using the
#    published author lists (reproduces the paper's split)
python "$COMMON/apply_author_lists.py" \
    --path "$RAW_JSONL" \
    --author_list_dir "$ROOT/data/splits/mud" \
    --out_dir "$WORK_DIR"

# Alternative: random author split with the evaluation authors held out. Use this
# instead of the command above to create a new split; it does not reproduce the
# published lists.
# python "$COMMON/split_authors.py" \
#     --path "$RAW_JSONL" \
#     --protected_authors_path "$PROTECTED_AUTHORS" \
#     --out_dir "$WORK_DIR"

# 2) Keep at most 10 texts per author
python "$COMMON/subsample_data.py" \
    --path "$RAW_JSONL" \
    --author_splits_path "$WORK_DIR/author_splits.json" \
    --per_author 10

# 3) Split the JSONL files into shards
python "$COMMON/split_files.py" --dir "$WORK_DIR" --max_per_file 50000

# 4) Generate paraphrases
CUDA_VISIBLE_DEVICES=$GPU python "$COMMON/generate_paraphrases.py" \
    --in_dir "$WORK_DIR/split_files" \
    --out_dir "$WORK_DIR/paraphrased_files" \
    --temp 1.5 --top_p 0.8 --idx 0 --num_workers 1 \
    --batch_size 32 --max_input_length 60 --max_output_length 60

# 5) Remove failed paraphrases
python "$COMMON/filter_empty_paraphrases.py" \
    --in-dir "$WORK_DIR/paraphrased_files/topp0.8_tmp1.5" \
    --out-dir "$WORK_DIR/filtered_paraphrases"

# 6) Convert JSONL files to a DatasetDict
python "$COMMON/convert_to_dataset_format.py" \
    --in_dir "$WORK_DIR/filtered_paraphrases" \
    --out_dir "$WORK_DIR/dataset_format" \
    --name mud_dataset

# 7) Add style embeddings
CUDA_VISIBLE_DEVICES=$GPU python "$COMMON/preprocess_add_source_text_embeddings.py" \
    --dataset_path "$WORK_DIR/dataset_format/mud_dataset" \
    --style_model_name AnnaWegmann/Style-Embedding

STYLE_DATASET="$WORK_DIR/dataset_format/mud_dataset_with_style_embeds_AnnaWegmann_Style-Embedding"

# 8) Build author embedding LMDBs and add StyleNavigator indices
python "$ROOT/preprocessing/build_author_embedding_lmdb.py" \
    --dataset-path "$STYLE_DATASET" \
    --output-root "$OUTPUT_DIR" \
    --splits train val

python "$COMMON/add_style_indices.py" \
    --dataset-path "$STYLE_DATASET" \
    --output-path "$OUTPUT_DIR/dataset_with_idx" \
    --train-lmdb "$OUTPUT_DIR/author_embs_train.lmdb" \
    --val-lmdb "$OUTPUT_DIR/author_embs_val.lmdb"
