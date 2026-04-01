#!/usr/bin/env bash
################################################################################
# 为了跑所有的实验，专门编写该脚本作为实验汇总，以nohup ..... & 形式执行即可
# run_all_experiments.sh — DACL-DR 全流程自动化实验脚本（增量模式 + 断点续跑）
#
# 执行策略：增量流水线（训练 → 编码 → 评估 → 持久化 → 清理 → 下一个模型）
#
# Phase 1: DPR 评估（编码 → 评估 NQ-test + TriviaQA-test）  先跑 baseline 验证流水线
# Phase 2: ANCE 评估（编码 → 评估 NQ-test + TriviaQA-test）
# Phase 3: Contriever 评估（编码 → 评估 NQ-test + TriviaQA-test）
# Phase 4: DACL-DR NQ 全流程（训练 Stage1+2 → 编码 → 评估 NQ-test → 零样本 TriviaQA-test）
# Phase 5: DACL-DR TriviaQA 全流程（训练 Stage1+2 → 编码 → 评估 TriviaQA-test）
# Phase 6: w=0 Baseline 全流程（训练 Stage1+2 → 编码 → 评估 NQ-test + TriviaQA-test）
# Phase 7: Embedding 空间分析 + 5模型 t-SNE + 综合对比图 + ANN 曲线图
# Phase 8: BEIR 零样本评测（5 数据集 × 5 模型）+ 汇总 + 绘图
# Phase 9: 距离权重敏感性分析绘图（w sweep）
#
# 断点续跑：每个步骤前检查产出文件，已完成的自动跳过
#
# 使用方式：
#   cd 到 code 目录下执行
#   chmod +x run_all_experiments.sh 加全自动脚本权限
#   mkdir -p logs
#   conda activate dacl-dr
#   nohup bash run_all_experiments.sh > ./logs/master.log 2>&1 &
#
# 查看进度：
#   tail -f logs/master.log                 # 主进度
#   tail -f logs/train_nq_w0.4.log          # NQ w=0.4 训练详情
#   ls -lt logs/                            # 查看最近更新的日志
#
# 删除重训（如需从头训练某个模型）：
#   必须同时删除该模型的 checkpoints + embeddings + results，否则断点续跑，三者分别是模型、语料库编码、评估结果
#   会跳过已存在的旧结果，导致数据不一致问题
#   例如重训 NQ 模型：
#     rm -rf checkpoints/nq/ embeddings/dacl-dr-nq/ results/nq/dacl-dr_nq.json
#
# 注意事项：
#   - 脚本会在失败时立即停止 (set -e)
#   - 支持断点续跑：任何时候 kill 后重新启动，已完成步骤自动跳过
#   - 每个模型评估完后自动删除索引文件（保留向量用于后续分析）
#   - 本次实验在单卡 A6000 (48GB) 环境下，使用 fp16 加速
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
EMB_FIGURE_DIR="./results/embedding/figures"  # t-SNE .npz 数据存放目录（供 plot_embedding_compare.py 读取）
BEIR_RESULT_DIR="./results/beir"              # BEIR 评测结果目录
WSWEEP_DIR="./experiments/results/w_sweep"    # w sweep 结果与绘图脚本目录

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

# DACL-DR 训练通用函数（Stage 1+2）
# 参数: dataset distance_weight output_dir
train_dacl_dr() {
    local DS="$1"
    local DIST_WEIGHT="$2"
    local OUTPUT="$3"

    # 日志文件名包含 dataset 和 distance_weight，确保互不覆盖
    local WEIGHT_TAG
    WEIGHT_TAG=$(echo "$DIST_WEIGHT" | tr '.' '_')
    local TRAIN_LOG="$LOG_DIR/train_${DS}_w${WEIGHT_TAG}.log"

    # --- Stage 1 + Stage 2 训练 ---
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

        log_step "Stage 1+2 训练完成: $DS w=$DIST_WEIGHT ($(elapsed_since $TRAIN_START))"
        check_dir "$OUTPUT/best_model_stage2"
        check_dir "$OUTPUT/best_model_$DS"
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
mkdir -p "$EMB_FIGURE_DIR"
mkdir -p "$BEIR_RESULT_DIR"

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
# Phase 4: DACL-DR NQ 全流程（训练 → 编码 → 评估 NQ-test → 零样本 TriviaQA-test）
################################################################################
log_section "Phase 4: DACL-DR NQ (train + encode + evaluate NQ-test + zero-shot TriviaQA-test)"
PHASE4_START=$(date +%s)

# 训练 Stage 1+2
train_dacl_dr "nq" "0.4" "$CKPT_DIR/nq"

# 编码 + 评估 NQ-test（同域评估）
# 产出: results/nq/dacl-dr_nq.json
run_encode_evaluate \
    "dacl-dr" "dacl-dr" \
    "$CKPT_DIR/nq/best_model_nq" \
    "$CKPT_DIR/nq/best_model_nq" \
    "dacl-dr-nq" \
    "nq"

# NQ 训练模型在 TriviaQA 上的零样本评估（复用 dacl-dr-nq embedding，只做检索评估）
# 产出: results/trivia/dacl_dr_w0.4_trivia_zero_shot.json（区别于 Phase 5 的 in-domain 结果）
# 注意：embedding 已由上方 run_encode_evaluate 编码完成，此处仅重建索引并评估
if skip_if_file_exists "$RESULT_DIR/trivia/dacl_dr_w0.4_trivia_zero_shot.json" \
    "DACL-DR w=0.4 NQ→TriviaQA 零样本评估"; then
    :
else
    ZEROSHOT_NQ_START=$(date +%s)
    ZEROSHOT_NQ_LOG="$LOG_DIR/eval_dacl_dr_w0.4_trivia_zero_shot.log"
    EMB_NQ="$EMB_DIR/dacl-dr-nq"
    INDEX_NQ="$EMB_NQ/indexes"

    log_step "=== NQ w=0.4 模型 → TriviaQA 零样本评估 ==="
    log_step "  Log: $ZEROSHOT_NQ_LOG"

    # 重建索引（如果已被清理）
    if [ ! -d "$INDEX_NQ" ]; then
        log_step "  重建 dacl-dr-nq 索引..."
        python build_index.py \
            --embeddings_dir "$EMB_NQ" \
            --index_type all \
            >> "$ZEROSHOT_NQ_LOG" 2>&1
    fi

    python evaluate.py \
        --embeddings_dir "$EMB_NQ" \
        --index_dir "$INDEX_NQ" \
        --dataset "trivia" \
        --data_dir "$DATA_DIR" \
        --corpus_path "$CORPUS_PATH" \
        --model_type "dacl-dr" \
        --model_path "$CKPT_DIR/nq/best_model_nq" \
        --output_path "$RESULT_DIR/trivia/dacl_dr_w0.4_trivia_zero_shot.json" \
        --max_query_length 256 \
        --query_batch_size 256 \
        --fp16 \
        --top_k_values "10,20,50,100" \
        --hnsw_ef_search "8,16,32,64,128,256,512" \
        --ivf_nprobe "1,4,8,16,32,64,128,256" \
        >> "$ZEROSHOT_NQ_LOG" 2>&1

    log_step "零样本评估完成: DACL-DR w=0.4 NQ→TriviaQA ($(elapsed_since $ZEROSHOT_NQ_START))"
    check_file "$RESULT_DIR/trivia/dacl_dr_w0.4_trivia_zero_shot.json"

    # 评估完后清理索引
    if [ -d "$INDEX_NQ" ]; then
        rm -rf "$INDEX_NQ"
        log_step "  索引已清理: $INDEX_NQ"
    fi
fi

log_step "Phase 4 完成: DACL-DR NQ ($(elapsed_since $PHASE4_START))"
log_step ">>> DACL-DR NQ 结果已持久化: $RESULT_DIR/nq/dacl-dr_nq.json <<<"
log_step ">>> NQ→TriviaQA 零样本结果已持久化: $RESULT_DIR/trivia/dacl_dr_w0.4_trivia_zero_shot.json <<<"

################################################################################
# Phase 5: DACL-DR TriviaQA 全流程（训练 → 编码 → 评估 TriviaQA-test）
################################################################################
log_section "Phase 5: DACL-DR TriviaQA (train + encode + evaluate TriviaQA-test)"
PHASE5_START=$(date +%s)

# 训练 Stage 1+2
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

# 训练 Stage 1+2（w=0），注意是在nq数据集上进行训练
train_dacl_dr "nq" "0.0" "$W0_OUTPUT"

# 编码 + 评估 NQ-test + TriviaQA-test + 清理索引
# w=0 的最终模型为 best_model_nq（由 train.py 统一保存）
run_encode_evaluate \
    "w0" "dacl-dr" \
    "$W0_OUTPUT/best_model_nq" \
    "$W0_OUTPUT/best_model_nq" \
    "w0-nq" \
    "nq" "trivia"

log_step "Phase 6 完成: w=0 Baseline ($(elapsed_since $PHASE6_START))"
log_step ">>> w=0 结果已持久化: $RESULT_DIR/nq/w0_nq.json, $RESULT_DIR/trivia/w0_trivia.json <<<"

################################################################################
# Phase 7: Embedding 空间分析 + t-SNE 可视化 + ANN 曲线图
################################################################################
log_section "Phase 7: Embedding Space Analysis & Visualization & ANN Curve Plots"

# --- 7.1 Embedding 空间统计（5 个模型 × NQ dev） ---
# 产出: results/nq/stats_*.json（共 5 个）
# 注：stats 仅用于论文 tab:embedding_analysis，在 NQ dev 上计算，不重复跑 trivia
log_step "=== Phase 7.1: Embedding Space Stats (NQ dev) ==="

STATS_TASKS=(
    "dacl_dr_w0.4  dacl-dr     $CKPT_DIR/nq/best_model_nq   nq"
    "dacl_dr_w0.0  dacl-dr     $W0_OUTPUT/best_model_nq      nq"
    "dpr            dpr         $DPR_QUERY_PATH               nq"
    "ance           ance        $ANCE_CTX_PATH                nq"
    "contriever     contriever  $CONTRIEVER_PATH              nq"
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

# --- 7.2 t-SNE 可视化：5 个模型在 NQ dev 上（生成 .npz 数据文件） ---
# 产出: results/embedding/figures/tsne_data_<label>.npz
# label 必须与 plot_embedding_compare.py 中硬编码的 MODELS 列表完全一致：
#   DPR, DACL-DR_w0, DACL-DR_w0.4, ANCE, Contriever
log_step "=== Phase 7.2: t-SNE Data Generation (5 models) ==="

TSNE_TASKS=(
    "DACL-DR_w0.4  dacl-dr     $CKPT_DIR/nq/best_model_nq"
    "DACL-DR_w0    dacl-dr     $W0_OUTPUT/best_model_nq"
    "DPR            dpr         $DPR_QUERY_PATH"
    "ANCE           ance        $ANCE_CTX_PATH"
    "Contriever     contriever  $CONTRIEVER_PATH"
)

for TSNE_ENTRY in "${TSNE_TASKS[@]}"; do
    read -r TSNE_LABEL MODEL_TYPE MODEL_PATH <<< "$TSNE_ENTRY"

    TSNE_NPZ="$EMB_FIGURE_DIR/tsne_data_${TSNE_LABEL}.npz"

    if skip_if_file_exists "$TSNE_NPZ" "t-SNE $TSNE_LABEL"; then
        continue
    fi

    TSNE_START=$(date +%s)
    TSNE_LOG="$LOG_DIR/tsne_${TSNE_LABEL}.log"
    log_step "=== t-SNE: $TSNE_LABEL ==="
    log_step "  Log: $TSNE_LOG"

    python analyze_embeddings.py tsne \
        --model_type "$MODEL_TYPE" \
        --model_path "$MODEL_PATH" \
        --dataset nq \
        --data_dir "$DATA_DIR" \
        --output_dir "$EMB_FIGURE_DIR" \
        --n_samples 500 \
        --fp16 \
        --label "$TSNE_LABEL" \
        > "$TSNE_LOG" 2>&1

    log_step "t-SNE 完成: $TSNE_LABEL ($(elapsed_since $TSNE_START))"
    check_file "$TSNE_NPZ"
done

# --- 7.3 生成 t-SNE 综合对比图和余弦相似度分布图 ---
# 读取 results/embedding/figures/tsne_data_*.npz
# 产出: Image/tsne_comparison.pdf, Image/cosine_distribution.pdf
IMAGE_DIR="./Image"
mkdir -p "$IMAGE_DIR"

if skip_if_file_exists "$IMAGE_DIR/tsne_comparison.pdf" "t-SNE 综合对比图"; then
    :
else
    COMPARE_LOG="$LOG_DIR/plot_embedding_compare.log"
    log_step "=== 生成 t-SNE 综合对比图和余弦分布图 ==="
    log_step "  Log: $COMPARE_LOG"

    python plot_embedding_compare.py \
        --npz_dir "$EMB_FIGURE_DIR" \
        --output_dir "$IMAGE_DIR" \
        > "$COMPARE_LOG" 2>&1

    log_step "综合对比图生成完成."
    check_file "$IMAGE_DIR/tsne_comparison.pdf"
    check_file "$IMAGE_DIR/cosine_distribution.pdf"
fi

# --- 7.4 建立绘图脚本所需的文件名别名（cp，不覆盖已有文件） ---
# plot_ann_curves.py 硬编码读取 results/nq/dacl_dr_w0.4_nq.json 和 dacl_dr_w0.0_nq.json
# plot_trivia_ann_curves.py 硬编码读取 results/trivia/*_zero_shot.json
log_step "=== Phase 7.4: 建立绘图脚本所需文件名别名 ==="

# NQ 结果文件别名（用于 plot_ann_curves.py）
declare -A NQ_ALIASES=(
    ["$RESULT_DIR/nq/dacl-dr_nq.json"]="$RESULT_DIR/nq/dacl_dr_w0.4_nq.json"
    ["$RESULT_DIR/nq/w0_nq.json"]="$RESULT_DIR/nq/dacl_dr_w0.0_nq.json"
)
for SRC in "${!NQ_ALIASES[@]}"; do
    DST="${NQ_ALIASES[$SRC]}"
    if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
        cp "$SRC" "$DST"
        log_step "  别名已建立: $(basename $SRC) → $(basename $DST)"
    elif [ -f "$DST" ]; then
        log_step "  [SKIP] 别名已存在: $(basename $DST)"
    else
        log_step "  WARNING: 源文件不存在，跳过: $SRC"
    fi
done

# TriviaQA 零样本结果文件别名（用于 plot_trivia_ann_curves.py）
# DPR/ANCE/Contriever 的 trivia 结果本身即为零样本，但脚本期望特定文件名
declare -A TRIVIA_ALIASES=(
    ["$RESULT_DIR/trivia/dpr_trivia.json"]="$RESULT_DIR/trivia/dpr_trivia_zero_shot.json"
    ["$RESULT_DIR/trivia/w0_trivia.json"]="$RESULT_DIR/trivia/dacl_dr_w0.0_trivia_zero_shot.json"
    ["$RESULT_DIR/trivia/dacl_dr_w0.4_trivia_zero_shot.json"]="$RESULT_DIR/trivia/dacl_dr_w0.4_trivia_zero_shot.json"
)
for SRC in "${!TRIVIA_ALIASES[@]}"; do
    DST="${TRIVIA_ALIASES[$SRC]}"
    if [ "$SRC" = "$DST" ]; then
        continue  # 源和目标相同，无需 cp
    fi
    if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
        cp "$SRC" "$DST"
        log_step "  别名已建立: $(basename $SRC) → $(basename $DST)"
    elif [ -f "$DST" ]; then
        log_step "  [SKIP] 别名已存在: $(basename $DST)"
    else
        log_step "  WARNING: 源文件不存在，跳过: $SRC"
    fi
done

# --- 7.5 NQ ANN 曲线图（论文核心图，PDF 格式） ---
# 产出: Image/hnsw_recall_efsearch.pdf, hnsw_recall_latency.pdf, hnsw_recall_ndc.pdf,
#       ivf_recall_nprobe.pdf, ivf_recall_latency.pdf, hnsw_recall_qps.pdf
if skip_if_file_exists "$IMAGE_DIR/hnsw_recall_efsearch.pdf" "NQ ANN 曲线图"; then
    :
else
    ANN_CURVES_LOG="$LOG_DIR/plot_ann_curves_nq.log"
    log_step "=== 生成 NQ ANN 曲线图（PDF） ==="
    log_step "  Log: $ANN_CURVES_LOG"

    python scripts/plot_ann_curves.py \
        > "$ANN_CURVES_LOG" 2>&1

    log_step "NQ ANN 曲线图生成完成."
    check_file "$IMAGE_DIR/hnsw_recall_efsearch.pdf"
    check_file "$IMAGE_DIR/hnsw_recall_ndc.pdf"
    check_file "$IMAGE_DIR/ivf_recall_nprobe.pdf"
fi

# --- 7.6 TriviaQA 零样本 ANN 曲线图（论文核心图，PDF 格式） ---
# 产出: Image/trivia_ann_curves_subplots.pdf
if skip_if_file_exists "$IMAGE_DIR/trivia_ann_curves_subplots.pdf" "TriviaQA 零样本 ANN 曲线图"; then
    :
else
    TRIVIA_CURVES_LOG="$LOG_DIR/plot_ann_curves_trivia.log"
    log_step "=== 生成 TriviaQA 零样本 ANN 曲线图（PDF） ==="
    log_step "  Log: $TRIVIA_CURVES_LOG"

    python scripts/plot_trivia_ann_curves.py \
        > "$TRIVIA_CURVES_LOG" 2>&1

    log_step "TriviaQA ANN 曲线图生成完成."
    check_file "$IMAGE_DIR/trivia_ann_curves_subplots.pdf"
fi

# --- 7.7 快速预览图（PNG，供日常查看） ---
log_step "=== Phase 7.7: 生成快速预览图（PNG） ==="
for DS in nq trivia; do
    PLOT_LOG="$LOG_DIR/plot_preview_${DS}.log"
    log_step "=== Plotting preview for $DS ==="
    log_step "  Log: $PLOT_LOG"

    RESULT_FILES=()
    for f in "$RESULT_DIR/$DS"/*.json; do
        # 排除 stats 文件和别名文件（避免重复数据混入）
        case "$(basename "$f")" in
            stats_*|dacl_dr_*|dpr_*_zero_shot*|*_zero_shot*) continue ;;
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
        log_step "WARNING: No result files found for $DS, skipping preview plots."
    fi
done

log_step "Phase 7 完成."

################################################################################
# Phase 8: BEIR 零样本评测（5 数据集 × 5 模型）
################################################################################
log_section "Phase 8: BEIR Zero-shot Evaluation"
PHASE8_START=$(date +%s)

# 5 个模型的 BEIR 评测任务
# 格式: "model_name model_type model_path [query_encoder_path ctx_encoder_path]"
BEIR_TASKS=(
    "dacl_dr_w0.4  dacl-dr     $CKPT_DIR/nq/best_model_nq"
    "dacl_dr_w0.0  dacl-dr     $W0_OUTPUT/best_model_nq"
    "dpr            dpr         $DPR_CTX_PATH"
    "ance           ance        $ANCE_CTX_PATH"
    "contriever     contriever  $CONTRIEVER_PATH"
)

for BEIR_ENTRY in "${BEIR_TASKS[@]}"; do
    read -r BEIR_MODEL_NAME MODEL_TYPE MODEL_PATH <<< "$BEIR_ENTRY"

    # 断点续跑：检查该模型所有 5 个数据集结果是否都存在
    ALL_BEIR_DONE=true
    for BEIR_DS in scifact nfcorpus fiqa trec-covid fever; do
        if [ ! -f "$BEIR_RESULT_DIR/${BEIR_MODEL_NAME}_${BEIR_DS}.json" ]; then
            ALL_BEIR_DONE=false
            break
        fi
    done
    if [ "$ALL_BEIR_DONE" = true ]; then
        log_step "[SKIP] BEIR $BEIR_MODEL_NAME -- 所有数据集结果已存在"
        continue
    fi

    BEIR_START=$(date +%s)
    BEIR_LOG="$LOG_DIR/beir_${BEIR_MODEL_NAME}.log"
    log_step "=== BEIR 评测: $BEIR_MODEL_NAME (model_type=$MODEL_TYPE) ==="
    log_step "  Log: $BEIR_LOG"

    # DPR/ANCE 需要分别指定 query_encoder 和 ctx_encoder
    if [ "$MODEL_TYPE" = "dpr" ]; then
        python beir_evaluate.py \
            --model_type "$MODEL_TYPE" \
            --query_encoder_path "$DPR_QUERY_PATH" \
            --ctx_encoder_path "$DPR_CTX_PATH" \
            --model_name "$BEIR_MODEL_NAME" \
            --data_dir "$DATA_DIR/beir" \
            --output_dir "$BEIR_RESULT_DIR" \
            --batch_size 128 \
            > "$BEIR_LOG" 2>&1
    elif [ "$MODEL_TYPE" = "ance" ]; then
        python beir_evaluate.py \
            --model_type "$MODEL_TYPE" \
            --query_encoder_path "$ANCE_QUERY_PATH" \
            --ctx_encoder_path "$ANCE_CTX_PATH" \
            --model_name "$BEIR_MODEL_NAME" \
            --data_dir "$DATA_DIR/beir" \
            --output_dir "$BEIR_RESULT_DIR" \
            --batch_size 128 \
            > "$BEIR_LOG" 2>&1
    else
        python beir_evaluate.py \
            --model_type "$MODEL_TYPE" \
            --model_path "$MODEL_PATH" \
            --model_name "$BEIR_MODEL_NAME" \
            --data_dir "$DATA_DIR/beir" \
            --output_dir "$BEIR_RESULT_DIR" \
            --batch_size 128 \
            > "$BEIR_LOG" 2>&1
    fi

    log_step "BEIR 评测完成: $BEIR_MODEL_NAME ($(elapsed_since $BEIR_START))"
done

# 汇总所有模型的 BEIR 结果
BEIR_SUMMARY="$BEIR_RESULT_DIR/summary.json"
if skip_if_file_exists "$BEIR_SUMMARY" "BEIR 结果汇总"; then
    :
else
    BEIR_SUMMARY_LOG="$LOG_DIR/beir_summarize.log"
    log_step "=== 汇总 BEIR 结果 ==="
    python beir_evaluate.py \
        --summarize \
        --output_dir "$BEIR_RESULT_DIR" \
        > "$BEIR_SUMMARY_LOG" 2>&1
    check_file "$BEIR_SUMMARY"
    log_step "BEIR 汇总完成: $BEIR_SUMMARY"
fi

# 生成 BEIR 对比柱状图
# 产出: Image/beir_average.pdf, Image/beir_ndcg10_comparison.pdf, Image/beir_recall100_comparison.pdf
if skip_if_file_exists "$IMAGE_DIR/beir_average.pdf" "BEIR 绘图"; then
    :
else
    BEIR_PLOT_LOG="$LOG_DIR/plot_beir.log"
    log_step "=== 生成 BEIR 柱状图 ==="
    log_step "  Log: $BEIR_PLOT_LOG"

    python scripts/plot_beir_results.py \
        > "$BEIR_PLOT_LOG" 2>&1

    log_step "BEIR 绘图完成."
    check_file "$IMAGE_DIR/beir_average.pdf"
fi

log_step "Phase 8 完成: BEIR ($(elapsed_since $PHASE8_START))"

################################################################################
# Phase 9: 距离权重敏感性分析绘图（w sweep）
# 注：w sweep 实验数据已预先存放于 experiments/results/w_sweep/ 目录
#     plot.py 使用硬编码路径，需在其所在目录下执行
################################################################################
log_section "Phase 9: Distance Weight Sensitivity Plot (w sweep)"
PHASE9_START=$(date +%s)

WSWEEP_PDF="$WSWEEP_DIR/w_sensitivity_analysis.pdf"
IMAGE_WSWEEP="$IMAGE_DIR/w_sensitivity_analysis.pdf"

if skip_if_file_exists "$IMAGE_WSWEEP" "w sweep 绘图"; then
    :
else
    WSWEEP_LOG="$LOG_DIR/plot_wsweep.log"
    log_step "=== 生成距离权重敏感性分析图 ==="
    log_step "  Log: $WSWEEP_LOG"

    # plot.py 使用相对路径读写文件，必须在脚本目录下执行
    (cd "$WSWEEP_DIR" && python plot.py) > "$WSWEEP_LOG" 2>&1

    check_file "$WSWEEP_PDF"
    # 复制到 Image/ 目录供论文引用（不覆盖已有文件以外的任何内容）
    cp "$WSWEEP_PDF" "$IMAGE_WSWEEP"
    log_step "w sweep 图已复制到: $IMAGE_WSWEEP"
fi

log_step "Phase 9 完成: w sweep ($(elapsed_since $PHASE9_START))"

################################################################################
# 完成
################################################################################
log_section "ALL EXPERIMENTS COMPLETE"

TOTAL_ELAPSED=$(elapsed_since $GLOBAL_START)
log_step "Total elapsed time: $TOTAL_ELAPSED"
log_step ""
log_step "Output summary:"
log_step "  Checkpoints:   $CKPT_DIR/"
log_step "  Embeddings:    $EMB_DIR/"
log_step "  Results JSON:  $RESULT_DIR/"
log_step "  BEIR Results:  $BEIR_RESULT_DIR/"
log_step "  Figures (PDF): $IMAGE_DIR/"
log_step "  Figures (PNG): $FIGURE_DIR/"
log_step "  t-SNE data:    $EMB_FIGURE_DIR/"
log_step "  w sweep:       $WSWEEP_DIR/"
log_step "  Logs:          $LOG_DIR/"
log_step ""
log_step "Result files:"
find "$RESULT_DIR" -name "*.json" -type f | sort
log_step ""
log_step "BEIR result files:"
find "$BEIR_RESULT_DIR" -name "*.json" -type f | sort
log_step ""
log_step "Figure files (PDF):"
find "$IMAGE_DIR" -name "*.pdf" -type f | sort
log_step ""
log_step "Done. Exiting successfully."
