#!/bin/bash
set -e

ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
RAW_DATASET=${1:-"$ROOT/data/raw/news/all_the_news_component_one"}
WORK_DIR=${2:-"$ROOT/data/intermediate/news"}
OUTPUT_DIR=${3:-"$ROOT/data/processed/news"}
GPU=${GPU_ID:-0}
COMMON="$ROOT/preprocessing/common"

FILTERED_DATASET="$WORK_DIR/one_author_article_dataset_split_filtered"
PREPARED_DIR="$WORK_DIR/prepared"
SPLIT_DIR="$PREPARED_DIR/split_files"
PARAPHRASE_DIR="$WORK_DIR/paraphrased_files/topp0.8_tmp1.5"
FILTERED_PARAPHRASE_DIR="$WORK_DIR/filtered_paraphrases"
LIMITED_DIR="$WORK_DIR/limited"
DATASET_DIR="$WORK_DIR/dataset_format"
DATASET="$DATASET_DIR/news_dataset"
STYLE_DATASET="$DATASET_DIR/news_dataset_with_style_embeds_AnnaWegmann_Style-Embedding"
INDEXED_DATASET="$WORK_DIR/dataset_with_idx_author"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

regenerate() {
    path=$1
    label=$2

    if [ ! -e "$path" ]; then
        return 0
    fi

    printf '%s already exists: %s\n' "$label" "$path"
    printf 'Delete and regenerate it? [y/N] '
    read -r answer
    case "$answer" in
        y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

# 0) Filter authors and assign them to train and validation sets using the
#    published author lists (reproduces the paper's split). Drop
#    --author-list-dir to create a new random split instead.
if regenerate "$FILTERED_DATASET" "Filtered News dataset"; then
    rm -rf "$WORK_DIR/one_author_article_dataset_split" "$FILTERED_DATASET"
    python "$ROOT/preprocessing/news/0_preprocess_news.py" \
        --input-path "$RAW_DATASET" \
        --output-dir "$WORK_DIR" \
        --author-list-dir "$ROOT/data/splits/news"
fi

# 1) Write train and validation JSONL files
if regenerate "$PREPARED_DIR" "Prepared News JSONL files"; then
    rm -rf "$PREPARED_DIR"
    python "$ROOT/preprocessing/news/1_prepare_news.py" \
        --input-path "$FILTERED_DATASET" \
        --output-dir "$PREPARED_DIR"
fi

# 2) Split the JSONL files into shards
if regenerate "$SPLIT_DIR" "News JSONL shards"; then
    rm -rf "$SPLIT_DIR"
    python "$COMMON/split_files.py" --dir "$PREPARED_DIR" --max_per_file 50000
fi

# 3) Generate paraphrases
if regenerate "$PARAPHRASE_DIR" "News paraphrases"; then
    rm -rf "$PARAPHRASE_DIR"
    CUDA_VISIBLE_DEVICES=$GPU python "$COMMON/generate_paraphrases.py" \
        --in_dir "$SPLIT_DIR" \
        --out_dir "$WORK_DIR/paraphrased_files" \
        --temp 1.5 --top_p 0.8 --idx 0 --num_workers 1 \
        --batch_size 32 --max_input_length 60 --max_output_length 60
fi

# 4) Remove failed paraphrases
if regenerate "$FILTERED_PARAPHRASE_DIR" "Filtered News paraphrases"; then
    rm -rf "$FILTERED_PARAPHRASE_DIR"
    python "$COMMON/filter_empty_paraphrases.py" \
        --in-dir "$PARAPHRASE_DIR" \
        --out-dir "$FILTERED_PARAPHRASE_DIR"
fi

# 5) Combine shards and keep 10 texts per author
if regenerate "$LIMITED_DIR" "News files with 10 texts per author"; then
    rm -rf "$WORK_DIR/paraphrased" "$LIMITED_DIR"
    mkdir -p "$WORK_DIR/paraphrased" "$LIMITED_DIR"

    for split in train val; do
        mapfile -t shards < <(find "$FILTERED_PARAPHRASE_DIR" -maxdepth 1 \
            -name "${split}_*_paraphrased.jsonl" -print | sort -V)
        if [ ${#shards[@]} -eq 0 ]; then
            echo "No paraphrase shards found for $split" >&2
            exit 1
        fi

        python "$ROOT/preprocessing/news/utilities/concat_paraphrased_shards.py" \
            --input-paths "${shards[@]}" \
            --output-path "$WORK_DIR/paraphrased/$split.jsonl"
        python "$ROOT/preprocessing/news/utilities/limit_texts_per_author.py" \
            --input-path "$WORK_DIR/paraphrased/$split.jsonl" \
            --output-path "$LIMITED_DIR/$split.jsonl" \
            --per-author 10 \
            --author-key author
    done
fi

# 6) Convert JSONL files to a DatasetDict
if regenerate "$DATASET_DIR" "News DatasetDict"; then
    rm -rf "$DATASET_DIR"
    python "$COMMON/convert_to_dataset_format.py" \
        --in_dir "$LIMITED_DIR" \
        --out_dir "$DATASET_DIR" \
        --name news_dataset
fi

# 7) Add style embeddings
if regenerate "$STYLE_DATASET" "Style-embedded News DatasetDict"; then
    rm -rf "$STYLE_DATASET"
    CUDA_VISIBLE_DEVICES=$GPU python "$COMMON/preprocess_add_source_text_embeddings.py" \
        --dataset_path "$DATASET" \
        --style_model_name AnnaWegmann/Style-Embedding
fi

# 8) Build author embedding LMDBs
if regenerate "$OUTPUT_DIR/author_embs_train.lmdb" "News author embedding LMDBs"; then
    rm -rf "$OUTPUT_DIR/author_embs_train.lmdb" "$OUTPUT_DIR/author_embs_val.lmdb"
    python "$ROOT/preprocessing/build_author_embedding_lmdb.py" \
        --dataset-path "$STYLE_DATASET" \
        --output-root "$OUTPUT_DIR" \
        --author-column author \
        --splits train val
fi

# 9) Add StyleNavigator indices
if regenerate "$INDEXED_DATASET" "Indexed News DatasetDict"; then
    rm -rf "$INDEXED_DATASET"
    python "$COMMON/add_style_indices.py" \
        --dataset-path "$STYLE_DATASET" \
        --output-path "$INDEXED_DATASET" \
        --train-lmdb "$OUTPUT_DIR/author_embs_train.lmdb" \
        --val-lmdb "$OUTPUT_DIR/author_embs_val.lmdb" \
        --author-column author
fi

# 10) Use the same author column name as Blog and MUD
if regenerate "$OUTPUT_DIR/dataset_with_idx" "Final News training dataset"; then
    rm -rf "$OUTPUT_DIR/dataset_with_idx"
    python "$ROOT/preprocessing/news/2_finalize_news_dataset.py" \
        --input-path "$INDEXED_DATASET" \
        --output-path "$OUTPUT_DIR/dataset_with_idx"
fi
