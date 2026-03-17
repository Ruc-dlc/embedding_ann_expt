#!/usr/bin/env bash
################################################################################
# run_all_experiments.sh — DACL-DR 全流程自动化实验脚本（增量模式 + 断点续跑）
#
# 执行策略：增量流水线（训练 → 编码 → 评估 → 持久化 → 清理 → 下一个模型）
#
# Phase 1: DPR 评估（编码 → 评估 NQ-test + TriviaQA-test）  ← 先跑 baseline 验证流水线
# Phase 2: ANCE 评估（编码 → 评估 NQ-test + TriviaQA-test）
# Phase 3: Contriever 评估（编码 → 评估 NQ-test + TriviaQA-test）
# Phase 4: DACL-DR NQ 全流程（训练 → 编码 → 评估 NQ-test）
# Phase 5: DACL-DR TriviaQA 全流程（训练 → 编码 → 评估 TriviaQA-test）
# Phase 6: w=0 Baseline 全流程（训练 → 编码 → 评估 NQ-test + TriviaQA-test）
# Phase 7: Embedding 空间分析 + t-SNE 可视化 + 结果绘图
#
# 断点续跑：每个步骤前检查产出文件，已完成的自动跳过
#
# 使用方式：
#   cd /home/users/dailongchao/workspace/thesis/DACL-DR/code
#   chmod +x run_all_experiments.sh
#   mkdir -p logs
#   conda activate dacl-dr
#   nohup bash run_all_experiments.sh > ./logs/master.log 2>&1 &
#
# 查看进度：
#   tail -f logs/master.log                 # 主进度
#   tail -f logs/train_nq.log               # NQ 训练详情
#   ls -lt logs/                            # 查看最近更新的日志
#
# 删除重训（如需从头训练某个模型）：
#   必须同时删除该模型的 checkpoints + embeddings + results，否则断点续跑
#   会跳过已存在的旧结果，导致数据不一致！
#   例如重训 NQ 模型：
#     rm -rf checkpoints/nq/ embeddings/dacl-dr-nq/ results/nq/dacl-dr_nq.json
#
# 注意事项：
#   - 脚本会在失败时立即停止 (set -e)
#   - 支持断点续跑：任何时候 kill 后重新启动，已完成步骤自动跳过
#   - 每个模型评估完后自动删除索引文件（保留向量用于后续分析）
#   - 单卡 A6000 (48GB)，使用 fp16 加速
################################################################################

set -euo pipefail

# ==================== 配置 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0

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
DPR_CTX_PATH="./dpr-backbone/context"
DPR_QUERY_PATH="./dpr-backbone/question"
ANCE_CTX_PATH="./ance-backbone/context"
ANCE_QUERY_PATH="./ance-backbone/question"
CONTRIEVER_PATH="./contriever-backbone"

# w=0 baseline 输出路径
W0_OUTPUT="$CKPT_DIR/nq_w0"

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

# 断点续跑检测函数
skip_if_file_exists() {
    local filepath="$1"
    local step_name="$2"
    if [ -f "$filepath" ]; then
        log_step "[SKIP] $step_name -- 产出文件已存在: $filepath"
        return 0
    fi
    return 1
}

skip_if_dir_exists() {
    local dirpath="$1"
    local step_name="$2"
    if [ -d "$dirpath" ] && [ -n "$(ls -A "$dirpath" 2>/dev/null)" ]; then
        log_step "[SKIP] $step_name -- 产出目录已存在: $dirpath"
        return 0
    fi
    return 1
}

# 编码 + 建索引 + 评估 + 清理 的通用函数
# 参数: model_key model_type ctx_encoder_path query_encoder_path emb_subdir eval_datasets...
run_encode_evaluate() {
    local MODEL_KEY="$1"
    local MODEL_TYPE="$2"
    local CTX_PATH="$3"
    local QUERY_PATH="$4"
    local EMB_SUBDIR="$5"
    shift 5
    local EVAL_DATASETS=("$@")  # 剩余参数为评估数据集列表

    local EMB_OUTPUT="$EMB_DIR/$EMB_SUBDIR"
    # 使用 EMB_SUBDIR 命名日志文件（保证唯一性，避免 Phase 1/2 的 dacl-dr 冲突）
    local ENCODE_LOG="$LOG_DIR/encode_${EMB_SUBDIR}.log"
    local INDEX_LOG="$LOG_DIR/build_index_${EMB_SUBDIR}.log"

    # --- 前置检查：如果所有评估结果都已存在，直接跳过整个流程 ---
    local ALL_EVALS_DONE=true
    for DS in "${EVAL_DATASETS[@]}"; do
        if [ ! -f "$RESULT_DIR/$DS/${MODEL_KEY}_${DS}.json" ]; then
            ALL_EVALS_DONE=false
            break
        fi
    done
    if [ "$ALL_EVALS_DONE" = true ]; then
        log_step "[SKIP] $MODEL_KEY 所有评估结果已存在，跳过编码/索引/评估"
        return 0
    fi

    # --- Step A: 编码 21M passages ---
    if skip_if_file_exists "$EMB_OUTPUT/passage_embeddings.npy" "编码 $MODEL_KEY"; then
        : # 已完成
    else
        local ENCODE_START
        ENCODE_START=$(date +%s)
        log_step "=== 编码 passages: $MODEL_KEY ==="
        log_step "  model_type=$MODEL_TYPE, ctx_encoder=$CTX_PATH"
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

        log_step "编码完成: $MODEL_KEY ($(elapsed_since $ENCODE_START))"
        check_file "$EMB_OUTPUT/passage_embeddings.npy"
        check_file "$EMB_OUTPUT/passage_ids.json"
    fi

    # --- Step B: 构建 4 个索引 ---
    local INDEX_PATH="$EMB_OUTPUT/indexes"
    if skip_if_dir_exists "$INDEX_PATH" "建索引 $MODEL_KEY"; then
        : # 已完成
    else
        local INDEX_START
        INDEX_START=$(date +%s)
        log_step "=== 构建所有索引: $MODEL_KEY ==="
        log_step "  Log: $INDEX_LOG"

        python build_index.py \
            --embeddings_dir "$EMB_OUTPUT" \
            --index_type all \
            > "$INDEX_LOG" 2>&1

        log_step "索引构建完成: $MODEL_KEY ($(elapsed_since $INDEX_START))"
    fi

    # --- Step C: 评估每个数据集 ---
    for DS in "${EVAL_DATASETS[@]}"; do
        local EVAL_KEY="${MODEL_KEY}_${DS}"
        local EVAL_LOG="$LOG_DIR/eval_${EVAL_KEY}.log"
        local OUTPUT_JSON="$RESULT_DIR/$DS/${EVAL_KEY}.json"

        if skip_if_file_exists "$OUTPUT_JSON" "评估 $EVAL_KEY"; then
            continue
        fi

        local EVAL_START
        EVAL_START=$(date +%s)
        log_step "=== 评估: $EVAL_KEY ==="
        log_step "  model_type=$MODEL_TYPE, query_encoder=$QUERY_PATH"
        log_step "  embeddings=$EMB_OUTPUT, indexes=$INDEX_PATH"
        log_step "  dataset=$DS, output=$OUTPUT_JSON"
        log_step "  Log: $EVAL_LOG"

        python evaluate.py \
            --embeddings_dir "$EMB_OUTPUT" \
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

        log_step "评估完成: $EVAL_KEY ($(elapsed_since $EVAL_START))"
        check_file "$OUTPUT_JSON"
    done

    # --- Step D: 清理索引文件（保留向量 .npy 用于后续分析） ---
    if [ -d "$INDEX_PATH" ]; then
        local INDEX_SIZE
        INDEX_SIZE=$(du -sh "$INDEX_PATH" 2>/dev/null | cut -f1)
        log_step "清理索引文件: $INDEX_PATH ($INDEX_SIZE)"
        rm -rf "$INDEX_PATH"
        log_step "索引已删除，向量文件保留: $EMB_OUTPUT/passage_embeddings.npy"
    fi
}

# DACL-DR 训练通用函数（Stage 1+2 → Mining → Stage 3）
# 参数: dataset distance_weight output_dir [skip_stage3]
train_dacl_dr() {
    local DS="$1"
    local DIST_WEIGHT="$2"
    local OUTPUT="$3"
    local SKIP_STAGE3="${4:-false}"  # 默认不跳过 Stage 3

    local LOG_SUFFIX
    if [ "$DIST_WEIGHT" = "0.0" ]; then
        LOG_SUFFIX="_w0"
    else
        LOG_SUFFIX=""
    fi
    local TRAIN_LOG="$LOG_DIR/train_${DS}${LOG_SUFFIX}.log"
    local MINE_LOG="$LOG_DIR/mine_hardneg_${DS}${LOG_SUFFIX}.log"
    local TRAIN_LOG_S3="$LOG_DIR/train_${DS}${LOG_SUFFIX}_stage3.log"

    # 确定最终模型目录名
    local FINAL_MODEL_DIR
    if [ "$SKIP_STAGE3" = "true" ]; then
        FINAL_MODEL_DIR="$OUTPUT/best_model_stage2"
    else
        FINAL_MODEL_DIR="$OUTPUT/best_model_$DS"
    fi

    # --- Step 1: Stage 1 + Stage 2 训练 ---
    if skip_if_dir_exists "$OUTPUT/best_model_stage2" "Stage 1+2 训练 ($DS, w=$DIST_WEIGHT)"; then
        : # 已完成
    else
        local TRAIN_START
        TRAIN_START=$(date +%s)
        log_step "=== 训练 DACL-DR: $DS (Stage 1+2, w=$DIST_WEIGHT) ==="
        log_step "  Log: $TRAIN_LOG"

        python train.py \
            --dataset "$DS" \
            --data_dir "$DATA_DIR" \
            --model_name "$BERT_BACKBONE" \
            --output_dir "$OUTPUT" \
            --batch_size 128 \
            --stage2_batch_size 32 \
            --stage2_gradient_accumulation_steps 4 \
            --learning_rate 2e-5 \
            --stage1_epochs 10 \
            --stage2_epochs 20 \
            --fp16 \
            --distance_weight "$DIST_WEIGHT" \
            --temperature 0.05 \
            --num_hard_negatives 7 \
            --seed 42 \
            > "$TRAIN_LOG" 2>&1

        log_step "Stage 1+2 训练完成: $DS ($(elapsed_since $TRAIN_START))"
        check_dir "$OUTPUT/best_model_stage2"
    fi

    # 如果跳过 Stage 3（w=0 baseline），直接返回
    if [ "$SKIP_STAGE3" = "true" ]; then
        log_step "跳过 Stage 3 (w=$DIST_WEIGHT baseline, 仅 Stage 1+2)"
        return 0
    fi

    # --- Step 2: Mining hard negatives ---
    local MINED_PATH
    if [ "$DS" = "nq" ]; then
        MINED_PATH="$DATA_DIR/NQ/nq-train-mined.json"
    else
        MINED_PATH="$DATA_DIR/TriviaQA/trivia-train-mined.json"
    fi

    if skip_if_file_exists "$MINED_PATH" "Mining hard negatives ($DS)"; then
        : # 已完成
    else
        local MINE_START
        MINE_START=$(date +%s)
        log_step "=== Mining hard negatives: $DS ==="
        log_step "  Log: $MINE_LOG"

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

        log_step "Mining 完成: $DS ($(elapsed_since $MINE_START))"
        check_file "$MINED_PATH"
    fi

    # --- Step 3: Stage 3 训练 ---
    # 检测 best_model_stage3（只有 Stage 3 实际运行过才会创建此目录）
    # 注意：不能检测 best_model_$DS，因为旧版 train.py 在跳过 Stage 3 时也会误创建它
    if skip_if_dir_exists "$OUTPUT/best_model_stage3" "Stage 3 训练 ($DS)"; then
        : # 已完成
    else
        local S3_START
        S3_START=$(date +%s)
        log_step "=== 训练 DACL-DR: $DS (Stage 3, resume from Stage 2 best) ==="
        log_step "  Log: $TRAIN_LOG_S3"

        python train.py \
            --dataset "$DS" \
            --data_dir "$DATA_DIR" \
            --model_name "$BERT_BACKBONE" \
            --output_dir "$OUTPUT" \
            --batch_size 128 \
            --stage2_batch_size 32 \
            --stage2_gradient_accumulation_steps 4 \
            --learning_rate 2e-5 \
            --stage1_epochs 10 \
            --stage2_epochs 20 \
            --stage3_max_epochs 10 \
            --early_stopping_patience 5 \
            --fp16 \
            --distance_weight "$DIST_WEIGHT" \
            --temperature 0.05 \
            --num_hard_negatives 7 \
            --seed 42 \
            --mined_data_path "$MINED_PATH" \
            --resume_checkpoint "$OUTPUT/best_model_stage2" \
            --resume_stage 3 \
            > "$TRAIN_LOG_S3" 2>&1

        log_step "Stage 3 训练完成: $DS ($(elapsed_since $S3_START))"
        check_dir "$FINAL_MODEL_DIR"
    fi
}

# ==================== 初始化 ====================
GLOBAL_START=$(date +%s)

# 提前创建日志目录（确保 nohup 重定向能正常工作）
mkdir -p "$LOG_DIR"

log_section "DACL-DR Full Experiment Pipeline Starting (Incremental Mode)"

log_step "Activating conda environment: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
log_step "Python: $(which python), version: $(python --version 2>&1)"

mkdir -p "$CKPT_DIR"/{nq,trivia,nq_w0}
mkdir -p "$EMB_DIR"/{dacl-dr-nq,dacl-dr-trivia,dpr,ance,contriever,w0-nq}
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

# 验证 Baseline Checkpoints
log_step "Verifying baseline checkpoints..."
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
    echo "  Please download them and scp to the GPU server."
    echo "========================================="
    exit 1
fi
log_step "All baseline checkpoints verified."

# GPU 信息
log_step "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

################################################################################
# Phase 1: DPR 评估（编码 → 评估 NQ-test + TriviaQA-test）
#   先跑 baseline，验证评估流水线（编码/索引/评估）无误
################################################################################
log_section "Phase 1: DPR Evaluation (encode + evaluate NQ & TriviaQA)"
PHASE1_START=$(date +%s)

run_encode_evaluate \
    "dpr" "dpr" \
    "$DPR_CTX_PATH" \
    "$DPR_QUERY_PATH" \
    "dpr" \
    "nq" "trivia"

log_step "Phase 1 完成: DPR ($(elapsed_since $PHASE1_START))"
log_step ">>> DPR 结果已持久化: $RESULT_DIR/nq/dpr_nq.json, $RESULT_DIR/trivia/dpr_trivia.json <<<"

################################################################################
# Phase 2: ANCE 评估（编码 → 评估 NQ-test + TriviaQA-test）
################################################################################
log_section "Phase 2: ANCE Evaluation (encode + evaluate NQ & TriviaQA)"
PHASE2_START=$(date +%s)

run_encode_evaluate \
    "ance" "ance" \
    "$ANCE_CTX_PATH" \
    "$ANCE_QUERY_PATH" \
    "ance" \
    "nq" "trivia"

log_step "Phase 2 完成: ANCE ($(elapsed_since $PHASE2_START))"
log_step ">>> ANCE 结果已持久化: $RESULT_DIR/nq/ance_nq.json, $RESULT_DIR/trivia/ance_trivia.json <<<"

################################################################################
# Phase 3: Contriever 评估（编码 → 评估 NQ-test + TriviaQA-test）
################################################################################
log_section "Phase 3: Contriever Evaluation (encode + evaluate NQ & TriviaQA)"
PHASE3_START=$(date +%s)

run_encode_evaluate \
    "contriever" "contriever" \
    "$CONTRIEVER_PATH" \
    "$CONTRIEVER_PATH" \
    "contriever" \
    "nq" "trivia"

log_step "Phase 3 完成: Contriever ($(elapsed_since $PHASE3_START))"
log_step ">>> Contriever 结果已持久化: $RESULT_DIR/nq/contriever_nq.json, $RESULT_DIR/trivia/contriever_trivia.json <<<"

log_section "=== Baseline 评估全部完成 ==="
log_step "DPR/ANCE/Contriever 结果均已生成，请检查 $RESULT_DIR/ 下的 JSON 文件。"
log_step "若结果正确，后续 Phase 将继续训练和评估 DACL-DR 模型。"

################################################################################
# Phase 4: DACL-DR NQ 全流程（训练 → 编码 → 评估 NQ-test）
################################################################################
log_section "Phase 4: DACL-DR NQ (train + encode + evaluate NQ-test)"
PHASE4_START=$(date +%s)

# 训练 Stage 1+2+3
train_dacl_dr "nq" "0.4" "$CKPT_DIR/nq"

# 编码 + 评估 NQ-test + 清理索引
run_encode_evaluate \
    "dacl-dr" "dacl-dr" \
    "$CKPT_DIR/nq/best_model_nq" \
    "$CKPT_DIR/nq/best_model_nq" \
    "dacl-dr-nq" \
    "nq"

log_step "Phase 4 完成: DACL-DR NQ ($(elapsed_since $PHASE4_START))"
log_step ">>> DACL-DR NQ 结果已持久化: $RESULT_DIR/nq/dacl-dr_nq.json <<<"

################################################################################
# Phase 5: DACL-DR TriviaQA 全流程（训练 → 编码 → 评估 TriviaQA-test）
################################################################################
log_section "Phase 5: DACL-DR TriviaQA (train + encode + evaluate TriviaQA-test)"
PHASE5_START=$(date +%s)

# 训练 Stage 1+2+3
train_dacl_dr "trivia" "0.4" "$CKPT_DIR/trivia"

# 编码 + 评估 TriviaQA-test + 清理索引
run_encode_evaluate \
    "dacl-dr" "dacl-dr" \
    "$CKPT_DIR/trivia/best_model_trivia" \
    "$CKPT_DIR/trivia/best_model_trivia" \
    "dacl-dr-trivia" \
    "trivia"

log_step "Phase 5 完成: DACL-DR TriviaQA ($(elapsed_since $PHASE5_START))"
log_step ">>> DACL-DR TriviaQA 结果已持久化: $RESULT_DIR/trivia/dacl-dr_trivia.json <<<"

################################################################################
# Phase 6: w=0 Baseline 全流程（训练 → 编码 → 评估 NQ-test + TriviaQA-test）
################################################################################
log_section "Phase 6: w=0 Baseline (train + encode + evaluate NQ-test + TriviaQA-test)"
PHASE6_START=$(date +%s)

mkdir -p "$W0_OUTPUT"

# 训练 Stage 1+2（无 Stage 3）
train_dacl_dr "nq" "0.0" "$W0_OUTPUT" "true"

# 编码 + 评估 NQ-test + TriviaQA-test + 清理索引
# w=0 的最终模型就是 best_model_stage2
run_encode_evaluate \
    "w0" "dacl-dr" \
    "$W0_OUTPUT/best_model_stage2" \
    "$W0_OUTPUT/best_model_stage2" \
    "w0-nq" \
    "nq" "trivia"

log_step "Phase 6 完成: w=0 Baseline ($(elapsed_since $PHASE6_START))"
log_step ">>> w=0 结果已持久化: $RESULT_DIR/nq/w0_nq.json, $RESULT_DIR/trivia/w0_trivia.json <<<"

################################################################################
# Phase 7: Embedding 空间分析 + t-SNE 可视化 + 结果绘图
################################################################################
log_section "Phase 7: Embedding Space Analysis & Visualization"

# --- 7.1 Embedding 空间统计 ---
log_step "=== Phase 7.1: Embedding Space Stats ==="

STATS_TASKS=(
    "dacl-dr_nq        dacl-dr     $CKPT_DIR/nq/best_model_nq                 nq"
    "dacl-dr_trivia     dacl-dr     $CKPT_DIR/trivia/best_model_trivia         trivia"
    "dpr_nq             dpr         $DPR_QUERY_PATH                            nq"
    "dpr_trivia         dpr         $DPR_QUERY_PATH                            trivia"
    "ance_nq            ance        $ANCE_CTX_PATH                             nq"
    "ance_trivia        ance        $ANCE_CTX_PATH                             trivia"
    "contriever_nq      contriever  $CONTRIEVER_PATH                           nq"
    "contriever_trivia  contriever  $CONTRIEVER_PATH                           trivia"
)

for STATS_ENTRY in "${STATS_TASKS[@]}"; do
    read -r STATS_KEY MODEL_TYPE MODEL_PATH DS <<< "$STATS_ENTRY"

    STATS_LOG="$LOG_DIR/stats_${STATS_KEY}.log"
    STATS_OUT="$RESULT_DIR/$DS/stats_${STATS_KEY}.json"

    if skip_if_file_exists "$STATS_OUT" "Embedding stats $STATS_KEY"; then
        continue
    fi

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

    log_step "Stats 完成: $STATS_KEY ($(elapsed_since $STATS_START))"
done

# --- 7.2 t-SNE 可视化: w=0.4 vs w=0 在 NQ dev 上 ---
log_step "=== Phase 7.2: t-SNE Visualization ==="
TSNE_LOG="$LOG_DIR/tsne_visualization.log"
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
        --model_path "$W0_OUTPUT/best_model_stage2" \
        --dataset nq \
        --data_dir "$DATA_DIR" \
        --output_dir "$FIGURE_DIR" \
        --n_samples 500 \
        --fp16 \
        --label "Baseline (w=0)"

} > "$TSNE_LOG" 2>&1

log_step "t-SNE 可视化完成."

# --- 7.3 结果对比图 ---
log_step "=== Phase 7.3: Plotting Results ==="
for DS in nq trivia; do
    PLOT_LOG="$LOG_DIR/plot_${DS}.log"
    log_step "=== Plotting results for $DS ==="
    log_step "  Log: $PLOT_LOG"

    RESULT_FILES=()
    for f in "$RESULT_DIR/$DS"/*.json; do
        # 排除 stats 文件
        case "$(basename "$f")" in
            stats_*) continue ;;
        esac
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

log_step "Phase 7 完成."

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
