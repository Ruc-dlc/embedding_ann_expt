#!/usr/bin/env bash
################################################################################
# run_all_experiments.sh — DACL-DR 全流程自动化实验脚本
#
# 功能：
#   Phase 1: 训练 DACL-DR (NQ + TriviaQA)
#   Phase 2: 验证 Baseline Checkpoint (DPR / ANCE / Contriever，需提前手动下载)
#   Phase 3: 编码 21M passages + 构建 FAISS 索引 (4 个模型)
#   Phase 4: 评估所有模型 × 所有索引 (NQ + TriviaQA)
#   Phase 5: Embedding 分析 + t-SNE 可视化 + 结果绘图
#
# 使用方式：
#   cd /path/to/DACL-DR/code
#   chmod +x run_all_experiments.sh
#   nohup ./run_all_experiments.sh > logs/master.log 2>&1 &
#
# 查看进度：
#   tail -f logs/master.log          # 主进度
#   tail -f logs/train_nq.log        # NQ 训练详情
#   ls -lt logs/                     # 查看最近更新的日志
#
# 注意事项：
#   - 脚本会在失败时立即停止 (set -e)
#   - 每个步骤有独立日志文件，互不覆盖
#   - 所有 GPU 操作使用 CUDA_VISIBLE_DEVICES=0 (单卡 A6000)
#   - 编码时使用 fp16 以节省显存和加速
################################################################################

set -euo pipefail

# ==================== 配置 ====================
# 项目根目录（脚本所在位置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# GPU 设置
export CUDA_VISIBLE_DEVICES=0

# Conda 环境
CONDA_ENV="dacl-dr"

# 数据路径
DATA_DIR="./data_set"
CORPUS_PATH="./data_set/psgs_w100.tsv"

# 输出路径
CKPT_DIR="./checkpoints"
EMB_DIR="./embeddings"
RESULT_DIR="./results"
FIGURE_DIR="./results/figures"
LOG_DIR="./logs"

# Backbone 路径
BERT_BACKBONE="./bert-base-uncase-backbone"
# DPR / ANCE: context encoder 和 question encoder 分开存放为子目录
# 命名必须包含 "context"/"question"，以便 analyze_embeddings.py 路径推导
DPR_CTX_PATH="./dpr-backbone/context"
DPR_QUERY_PATH="./dpr-backbone/question"
ANCE_CTX_PATH="./ance-backbone/context"
ANCE_QUERY_PATH="./ance-backbone/question"
CONTRIEVER_PATH="./contriever-backbone"

# 数据集列表
DATASETS=("nq" "trivia")

# ==================== 工具函数 ====================
log_section() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"
    echo ""
}

log_step() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        exit 1
    fi
}

check_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: Required directory not found: $1"
        exit 1
    fi
}

elapsed_since() {
    local start_ts=$1
    local now_ts
    now_ts=$(date +%s)
    local diff=$((now_ts - start_ts))
    printf '%02dh:%02dm:%02ds' $((diff/3600)) $((diff%3600/60)) $((diff%60))
}

# ==================== 初始化 ====================
GLOBAL_START=$(date +%s)

log_section "DACL-DR Full Experiment Pipeline Starting"

# 激活 conda 环境
log_step "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
log_step "Python: $(which python), version: $(python --version 2>&1)"

# 创建目录
mkdir -p "$LOG_DIR"
mkdir -p "$CKPT_DIR"/{nq,trivia}
mkdir -p "$EMB_DIR"/{dacl-dr-nq,dacl-dr-trivia,dpr,ance,contriever}
mkdir -p "$RESULT_DIR"/{nq,trivia}
mkdir -p "$FIGURE_DIR"

# 验证数据文件
log_step "Verifying data files..."
check_file "$CORPUS_PATH"
check_file "$DATA_DIR/NQ/nq-train.json"
check_file "$DATA_DIR/NQ/nq-dev.json"
check_file "$DATA_DIR/NQ/nq-test.csv"
check_file "$DATA_DIR/TriviaQA/trivia-train.json"
check_file "$DATA_DIR/TriviaQA/trivia-dev.json"
check_file "$DATA_DIR/TriviaQA/trivia-test.csv"
check_dir "$BERT_BACKBONE"
log_step "All data files verified."

# GPU 信息
log_step "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

################################################################################
# Phase 1: 训练 DACL-DR 模型
################################################################################
log_section "Phase 1: Training DACL-DR Models"

for DS in "${DATASETS[@]}"; do
    PHASE1_START=$(date +%s)
    TRAIN_LOG="$LOG_DIR/train_${DS}.log"
    MINE_LOG="$LOG_DIR/mine_hardneg_${DS}.log"
    TRAIN_LOG_S3="$LOG_DIR/train_${DS}_stage3.log"
    OUTPUT="$CKPT_DIR/$DS"

    log_step "=== Training DACL-DR on $DS (Stage 1 + Stage 2) ==="
    log_step "Log: $TRAIN_LOG"

    # Stage 1 + Stage 2 训练
    python train.py \
        --dataset "$DS" \
        --data_dir "$DATA_DIR" \
        --model_name "$BERT_BACKBONE" \
        --output_dir "$OUTPUT" \
        --batch_size 128 \
        --learning_rate 2e-5 \
        --stage1_epochs 10 \
        --stage2_epochs 20 \
        --fp16 \
        --distance_weight 0.4 \
        --temperature 0.05 \
        --num_hard_negatives 7 \
        --seed 42 \
        > "$TRAIN_LOG" 2>&1

    log_step "Stage 1 + 2 training complete for $DS ($(elapsed_since $PHASE1_START))"

    # 验证 Stage 2 best model 存在
    check_dir "$OUTPUT/best_model_stage2"

    # Mining: 编码全量语料库 + 挖掘 Model Hard Negatives
    MINE_START=$(date +%s)
    if [ "$DS" = "nq" ]; then
        MINED_PATH="$DATA_DIR/NQ/nq-train-mined.json"
    else
        MINED_PATH="$DATA_DIR/TriviaQA/trivia-train-mined.json"
    fi

    log_step "=== Mining hard negatives for $DS ==="
    log_step "Log: $MINE_LOG"

    python mine_hard_negatives.py \
        --checkpoint_dir "$OUTPUT/best_model_stage2" \
        --dataset "$DS" \
        --data_dir "$DATA_DIR" \
        --corpus_path "$CORPUS_PATH" \
        --output_path "$MINED_PATH" \
        --top_k 200 \
        --keep_top_n 50 \
        --encode_batch_size 512 \
        --query_batch_size 256 \
        --fp16 \
        > "$MINE_LOG" 2>&1

    log_step "Mining complete for $DS ($(elapsed_since $MINE_START))"
    check_file "$MINED_PATH"

    # Stage 3: Model Hard Negatives 训练
    S3_START=$(date +%s)
    log_step "=== Training DACL-DR on $DS (Stage 3, resume from Stage 2 best) ==="
    log_step "Log: $TRAIN_LOG_S3"

    python train.py \
        --dataset "$DS" \
        --data_dir "$DATA_DIR" \
        --model_name "$BERT_BACKBONE" \
        --output_dir "$OUTPUT" \
        --batch_size 128 \
        --learning_rate 2e-5 \
        --stage1_epochs 10 \
        --stage2_epochs 20 \
        --stage3_max_epochs 10 \
        --early_stopping_patience 5 \
        --fp16 \
        --distance_weight 0.4 \
        --temperature 0.05 \
        --num_hard_negatives 7 \
        --seed 42 \
        --mined_data_path "$MINED_PATH" \
        --resume_checkpoint "$OUTPUT/best_model_stage2" \
        --resume_stage 3 \
        > "$TRAIN_LOG_S3" 2>&1

    log_step "Stage 3 training complete for $DS ($(elapsed_since $S3_START))"
    log_step "Phase 1 ($DS) total time: $(elapsed_since $PHASE1_START)"
done

log_step "Phase 1 complete: All DACL-DR models trained."

################################################################################
# Phase 1.5: 训练 w=0 Baseline（用于 t-SNE 对比）
################################################################################
log_section "Phase 1.5: Training w=0 Baseline (for t-SNE comparison)"

# 只在 NQ 上训练 w=0 模型（用于对比可视化）
W0_LOG="$LOG_DIR/train_nq_w0.log"
W0_OUTPUT="$CKPT_DIR/nq_w0"
mkdir -p "$W0_OUTPUT"

log_step "=== Training w=0 baseline on NQ (Stage 1 + Stage 2 only) ==="
log_step "Log: $W0_LOG"

python train.py \
    --dataset nq \
    --data_dir "$DATA_DIR" \
    --model_name "$BERT_BACKBONE" \
    --output_dir "$W0_OUTPUT" \
    --batch_size 128 \
    --learning_rate 2e-5 \
    --stage1_epochs 10 \
    --stage2_epochs 20 \
    --fp16 \
    --distance_weight 0.0 \
    --temperature 0.05 \
    --num_hard_negatives 7 \
    --seed 42 \
    > "$W0_LOG" 2>&1

log_step "w=0 baseline training complete."

################################################################################
# Phase 2: 验证 Baseline Checkpoints（需提前手动下载并 scp 到对应目录）
################################################################################
log_section "Phase 2: Verifying Baseline Checkpoints"

log_step "Checking baseline checkpoint directories..."

BASELINE_DIRS=(
    "$DPR_CTX_PATH"
    "$DPR_QUERY_PATH"
    "$ANCE_CTX_PATH"
    "$ANCE_QUERY_PATH"
    "$CONTRIEVER_PATH"
)

BASELINE_LABELS=(
    "DPR context encoder (facebook/dpr-ctx_encoder-single-nq-base)"
    "DPR question encoder (facebook/dpr-question_encoder-single-nq-base)"
    "ANCE context encoder (castorini/ance-dpr-context-multi)"
    "ANCE question encoder (castorini/ance-dpr-question-multi)"
    "Contriever (facebook/contriever)"
)

ALL_OK=true
for i in "${!BASELINE_DIRS[@]}"; do
    DIR="${BASELINE_DIRS[$i]}"
    LABEL="${BASELINE_LABELS[$i]}"
    if [ ! -d "$DIR" ] || [ -z "$(ls -A "$DIR" 2>/dev/null)" ]; then
        echo "ERROR: Baseline checkpoint missing or empty: $DIR"
        echo "       Expected: $LABEL"
        echo "       Please download on local machine and scp to this directory."
        ALL_OK=false
    else
        FILE_COUNT=$(find "$DIR" -type f | wc -l)
        log_step "  OK: $DIR ($FILE_COUNT files)"
    fi
done

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "========================================="
    echo "  ABORT: Missing baseline checkpoints."
    echo "  Please download them on your local machine (with internet access)"
    echo "  and scp to the GPU server before running this script."
    echo "========================================="
    exit 1
fi

log_step "Phase 2 complete: All baseline checkpoints verified."

################################################################################
# Phase 3: 编码 21M Passages + 构建 FAISS 索引
################################################################################
log_section "Phase 3: Encoding Passages & Building Indexes"

# --- 定义模型列表 ---
# 格式: "model_key model_type ctx_encoder_path emb_subdir"
# (ctx_encoder_path 用于 encode_passages.py)
MODELS=(
    "dacl-dr-nq    dacl-dr  $CKPT_DIR/nq/best_model_nq       dacl-dr-nq"
    "dacl-dr-trivia dacl-dr  $CKPT_DIR/trivia/best_model_trivia dacl-dr-trivia"
    "dpr           dpr      $DPR_CTX_PATH                     dpr"
    "ance          ance     $ANCE_CTX_PATH                    ance"
    "contriever    contriever $CONTRIEVER_PATH                 contriever"
)

for MODEL_ENTRY in "${MODELS[@]}"; do
    read -r MODEL_KEY MODEL_TYPE CTX_PATH EMB_SUBDIR <<< "$MODEL_ENTRY"

    ENCODE_LOG="$LOG_DIR/encode_${MODEL_KEY}.log"
    INDEX_LOG="$LOG_DIR/build_index_${MODEL_KEY}.log"
    EMB_OUTPUT="$EMB_DIR/$EMB_SUBDIR"

    # --- 编码 ---
    ENCODE_START=$(date +%s)
    log_step "=== Encoding passages: $MODEL_KEY ==="
    log_step "  model_type=$MODEL_TYPE, path=$CTX_PATH"
    log_step "  output=$EMB_OUTPUT"
    log_step "  Log: $ENCODE_LOG"

    python encode_passages.py \
        --model_type "$MODEL_TYPE" \
        --model_path "$CTX_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --output_dir "$EMB_OUTPUT" \
        --batch_size 512 \
        --max_passage_length 256 \
        --fp16 \
        --save_float16 \
        > "$ENCODE_LOG" 2>&1

    log_step "Encoding complete: $MODEL_KEY ($(elapsed_since $ENCODE_START))"
    check_file "$EMB_OUTPUT/passage_embeddings.npy"
    check_file "$EMB_OUTPUT/passage_ids.json"

    # --- 构建所有索引 ---
    INDEX_START=$(date +%s)
    log_step "=== Building all indexes: $MODEL_KEY ==="
    log_step "  Log: $INDEX_LOG"

    python build_index.py \
        --embeddings_dir "$EMB_OUTPUT" \
        --index_type all \
        > "$INDEX_LOG" 2>&1

    log_step "Index building complete: $MODEL_KEY ($(elapsed_since $INDEX_START))"
done

log_step "Phase 3 complete: All models encoded and indexed."

################################################################################
# Phase 4: 评估所有模型 × 所有数据集
################################################################################
log_section "Phase 4: Evaluation"

# 定义评估任务
# 格式: "eval_key model_type query_encoder_path emb_subdir dataset"
EVAL_TASKS=()
for DS in "${DATASETS[@]}"; do
    # DACL-DR: 对应数据集的模型评估对应数据集
    if [ "$DS" = "nq" ]; then
        DACL_MODEL_PATH="$CKPT_DIR/nq/best_model_nq"
        DACL_EMB="dacl-dr-nq"
    else
        DACL_MODEL_PATH="$CKPT_DIR/trivia/best_model_trivia"
        DACL_EMB="dacl-dr-trivia"
    fi
    EVAL_TASKS+=("dacl-dr_${DS}  dacl-dr     $DACL_MODEL_PATH     $DACL_EMB   $DS")
    EVAL_TASKS+=("dpr_${DS}      dpr         $DPR_QUERY_PATH      dpr         $DS")
    EVAL_TASKS+=("ance_${DS}     ance        $ANCE_QUERY_PATH     ance        $DS")
    EVAL_TASKS+=("contriever_${DS} contriever $CONTRIEVER_PATH     contriever  $DS")
done

for EVAL_ENTRY in "${EVAL_TASKS[@]}"; do
    read -r EVAL_KEY MODEL_TYPE QUERY_PATH EMB_SUBDIR DS <<< "$EVAL_ENTRY"

    EVAL_LOG="$LOG_DIR/eval_${EVAL_KEY}.log"
    EMB_PATH="$EMB_DIR/$EMB_SUBDIR"
    INDEX_PATH="$EMB_PATH/indexes"
    OUTPUT_JSON="$RESULT_DIR/$DS/${EVAL_KEY}.json"

    EVAL_START=$(date +%s)
    log_step "=== Evaluating: $EVAL_KEY ==="
    log_step "  model_type=$MODEL_TYPE, query_encoder=$QUERY_PATH"
    log_step "  embeddings=$EMB_PATH, indexes=$INDEX_PATH"
    log_step "  dataset=$DS, output=$OUTPUT_JSON"
    log_step "  Log: $EVAL_LOG"

    python evaluate.py \
        --embeddings_dir "$EMB_PATH" \
        --index_dir "$INDEX_PATH" \
        --dataset "$DS" \
        --data_dir "$DATA_DIR" \
        --corpus_path "$CORPUS_PATH" \
        --model_type "$MODEL_TYPE" \
        --model_path "$QUERY_PATH" \
        --output_path "$OUTPUT_JSON" \
        --max_query_length 256 \
        --query_batch_size 256 \
        --fp16 \
        --top_k_values "10,20,50,100" \
        --hnsw_ef_search "8,16,32,64,128,256,512" \
        --ivf_nprobe "1,4,8,16,32,64,128,256" \
        > "$EVAL_LOG" 2>&1

    log_step "Evaluation complete: $EVAL_KEY ($(elapsed_since $EVAL_START))"
    check_file "$OUTPUT_JSON"
done

log_step "Phase 4 Evaluation complete."

################################################################################
# Phase 4.5: Embedding 空间分析 (stats)
################################################################################
log_section "Phase 4.5: Embedding Space Analysis"

# 对每个模型 × 每个数据集做 embedding stats
STATS_TASKS=()
for DS in "${DATASETS[@]}"; do
    if [ "$DS" = "nq" ]; then
        DACL_MODEL_PATH="$CKPT_DIR/nq/best_model_nq"
    else
        DACL_MODEL_PATH="$CKPT_DIR/trivia/best_model_trivia"
    fi
    STATS_TASKS+=("dacl-dr_${DS}  dacl-dr     $DACL_MODEL_PATH     $DS")
    # DPR stats: model_path = question encoder (code derives context encoder)
    STATS_TASKS+=("dpr_${DS}      dpr         $DPR_QUERY_PATH      $DS")
    # ANCE stats: model_path = context encoder (code derives question encoder)
    STATS_TASKS+=("ance_${DS}     ance        $ANCE_CTX_PATH       $DS")
    STATS_TASKS+=("contriever_${DS} contriever $CONTRIEVER_PATH     $DS")
done

for STATS_ENTRY in "${STATS_TASKS[@]}"; do
    read -r STATS_KEY MODEL_TYPE MODEL_PATH DS <<< "$STATS_ENTRY"

    STATS_LOG="$LOG_DIR/stats_${STATS_KEY}.log"
    STATS_OUT="$RESULT_DIR/$DS/stats_${STATS_KEY}.json"

    STATS_START=$(date +%s)
    log_step "=== Embedding stats: $STATS_KEY ==="
    log_step "  Log: $STATS_LOG"

    python analyze_embeddings.py stats \
        --model_type "$MODEL_TYPE" \
        --model_path "$MODEL_PATH" \
        --dataset "$DS" \
        --data_dir "$DATA_DIR" \
        --output_path "$STATS_OUT" \
        --batch_size 128 \
        --fp16 \
        > "$STATS_LOG" 2>&1

    log_step "Stats complete: $STATS_KEY ($(elapsed_since $STATS_START))"
done

log_step "Phase 4.5 complete."

################################################################################
# Phase 5: t-SNE 可视化 + 结果绘图
################################################################################
log_section "Phase 5: Visualization"

# --- 5.1 t-SNE: w=0.4 (DACL-DR) vs w=0 (baseline) 在 NQ dev 上 ---
TSNE_LOG="$LOG_DIR/tsne_visualization.log"
log_step "=== t-SNE visualization ==="
log_step "  Log: $TSNE_LOG"

{
    # w=0.4 模型
    python analyze_embeddings.py tsne \
        --model_type dacl-dr \
        --model_path "$CKPT_DIR/nq/best_model_nq" \
        --dataset nq \
        --data_dir "$DATA_DIR" \
        --output_dir "$FIGURE_DIR" \
        --n_samples 500 \
        --fp16 \
        --label "DACL-DR (w=0.4)"

    # w=0 baseline 模型
    python analyze_embeddings.py tsne \
        --model_type dacl-dr \
        --model_path "$W0_OUTPUT/best_model_nq" \
        --dataset nq \
        --data_dir "$DATA_DIR" \
        --output_dir "$FIGURE_DIR" \
        --n_samples 500 \
        --fp16 \
        --label "Baseline (w=0)"

} > "$TSNE_LOG" 2>&1

log_step "t-SNE visualization complete."

# --- 5.2 结果对比图：用 plot_results.py 绘制各类曲线 ---
for DS in "${DATASETS[@]}"; do
    PLOT_LOG="$LOG_DIR/plot_${DS}.log"
    log_step "=== Plotting results for $DS ==="
    log_step "  Log: $PLOT_LOG"

    # 收集该数据集下所有评估结果 JSON
    RESULT_FILES=()
    for f in "$RESULT_DIR/$DS"/dacl-dr_*.json "$RESULT_DIR/$DS"/dpr_*.json \
             "$RESULT_DIR/$DS"/ance_*.json "$RESULT_DIR/$DS"/contriever_*.json; do
        if [ -f "$f" ]; then
            RESULT_FILES+=("$f")
        fi
    done

    if [ ${#RESULT_FILES[@]} -gt 0 ]; then
        python plot_results.py \
            --result_files "${RESULT_FILES[@]}" \
            --output_dir "$FIGURE_DIR" \
            --dataset_label "$DS" \
            > "$PLOT_LOG" 2>&1
    else
        log_step "WARNING: No result files found for $DS, skipping plots."
    fi
done

log_step "Phase 5 complete."

################################################################################
# 完成
################################################################################
log_section "ALL EXPERIMENTS COMPLETE"

TOTAL_ELAPSED=$(elapsed_since $GLOBAL_START)
log_step "Total elapsed time: $TOTAL_ELAPSED"
log_step ""
log_step "Output summary:"
log_step "  Checkpoints:  $CKPT_DIR/"
log_step "  Embeddings:   $EMB_DIR/"
log_step "  Results JSON: $RESULT_DIR/"
log_step "  Figures:      $FIGURE_DIR/"
log_step "  Logs:         $LOG_DIR/"
log_step ""
log_step "Result files:"
find "$RESULT_DIR" -name "*.json" -type f | sort
log_step ""
log_step "Figure files:"
find "$FIGURE_DIR" -type f | sort
log_step ""
log_step "Done. Exiting successfully."
