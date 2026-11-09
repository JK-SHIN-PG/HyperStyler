#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BLOG_CSV=${1:-"$ROOT/data/raw/blog/blogtext.csv"}
WORK_DIR=${2:-"$ROOT/data/intermediate/blog"}
OUTPUT_DIR=${3:-"$ROOT/data/processed/blog"}
GPU=${GPU_ID:-0}
COMMON="$ROOT/preprocessing/common"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

# 0) Prepare Blog Authorship Corpus
python "$ROOT/preprocessing/blog/0_preprocess_blog.py" \
    --input-path "$BLOG_CSV" \
    --output-dir "$WORK_DIR"

# 1) Assign authors to train, validation, and held-out sets using the
#    published author lists (reproduces the paper's split)
python "$COMMON/apply_author_lists.py" \
    --path "$WORK_DIR/preprocessed_blog_dataset.jsonl" \
    --author_list_dir "$ROOT/data/splits/blog" \
    --out_dir "$WORK_DIR"

# Alternative: random 90/5/5 author split (seed 42). Use this instead of the
# command above to create a new split; it does not reproduce the published lists.
# python "$COMMON/split_authors.py" \
#     --path "$WORK_DIR/preprocessed_blog_dataset.jsonl" \
#     --protected_authors_path "$WORK_DIR/protected_author.jsonl" \
#     --train_split 0.9 \
#     --val_split 0.05 \
#     --test_split 0.05 \
#     --out_dir "$WORK_DIR"

# 2) Keep at most 10 texts per author
python "$COMMON/subsample_data.py" \
    --path "$WORK_DIR/preprocessed_blog_dataset.jsonl" \
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
    --name blog_dataset

# 7) Add style embeddings
CUDA_VISIBLE_DEVICES=$GPU python "$COMMON/preprocess_add_source_text_embeddings.py" \
    --dataset_path "$WORK_DIR/dataset_format/blog_dataset" \
    --style_model_name AnnaWegmann/Style-Embedding

STYLE_DATASET="$WORK_DIR/dataset_format/blog_dataset_with_style_embeds_AnnaWegmann_Style-Embedding"

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
