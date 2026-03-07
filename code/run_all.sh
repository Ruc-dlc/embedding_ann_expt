#!/bin/bash
# DACL-DR 全流程自动化脚本 (断点续跑版)
#
# 用法:
#   cd code && nohup bash run_all.sh &
#
# 特性:
#   - 自动创建 logs/ 目录并将主日志写入 logs/main.log
#   - 每个子任务成功后写入完成标记 (logs/done/), 重跑时自动跳过已完成任务
#   - 任务失败则无标记, 修复 bug 后重新执行即可从断点继续
#   - SSH 断连不会中断执行

# 先创建目录 (必须在 exec 重定向之前)
mkdir -p logs logs/done
# 脚本自身重定向: 所有后续输出追加写入 logs/main.log
exec >> logs/main.log 2>&1

set -e

DONE="logs/done"

# 时间戳函数
ts() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(ts)] $1"
}

# ====== Phase 1: 构建 5M 语料子集 ======
log "===== Phase 1: 构建 5M 语料子集 ====="
if [ -f "$DONE/phase1_corpus" ]; then
    log "  已完成, 跳过."
else
    python build_corpus_subset.py > logs/phase1_corpus.log 2>&1
    touch "$DONE/phase1_corpus"
    log "  Phase 1 完成. 查看详情: logs/phase1_corpus.log"
fi

# ====== Phase 2: 训练 5 个模型 ======
log "===== Phase 2: 训练模型 ====="
for w in 0.0 0.4 0.6 0.8 1.0; do
    if [ -f "$DONE/phase2_train_w${w}" ]; then
        log "  Wmax=${w} 训练已完成, 跳过."
    else
        log "  开始训练 Wmax=${w} ..."
        python train.py \
            --wmax ${w} \
            --output_dir experiments/models/w${w} \
            > logs/phase2_train_w${w}.log 2>&1
        touch "$DONE/phase2_train_w${w}"
        log "  Wmax=${w} 训练完成. 查看详情: logs/phase2_train_w${w}.log"
    fi
done
log "Phase 2 全部完成."

# ====== Phase 3: 编码 passages + 构建索引 ======
log "===== Phase 3: 编码 passages + 构建索引 ====="
for w in 0.0 0.4 0.6 0.8 1.0; do
    for ckpt in best_nq best_trivia; do
        # 编码
        if [ -f "$DONE/phase3_encode_w${w}_${ckpt}" ]; then
            log "  编码 w=${w} ${ckpt} 已完成, 跳过."
        else
            log "  编码 w=${w} ${ckpt} ..."
            python encode_passages.py \
                --checkpoint experiments/models/w${w}/${ckpt} \
                --corpus experiments/corpus/corpus_5m.tsv \
                --output experiments/embeddings/w${w}_${ckpt} \
                > logs/phase3_encode_w${w}_${ckpt}.log 2>&1
            touch "$DONE/phase3_encode_w${w}_${ckpt}"
            log "  编码完成: w=${w} ${ckpt}"
        fi

        # 构建索引
        if [ -f "$DONE/phase3_index_w${w}_${ckpt}" ]; then
            log "  索引 w=${w} ${ckpt} 已完成, 跳过."
        else
            log "  构建索引 w=${w} ${ckpt} ..."
            python build_index.py \
                --embeddings experiments/embeddings/w${w}_${ckpt} \
                --output experiments/indices/w${w}_${ckpt} \
                > logs/phase3_index_w${w}_${ckpt}.log 2>&1
            touch "$DONE/phase3_index_w${w}_${ckpt}"
            log "  索引完成: w=${w} ${ckpt}"
        fi
    done
done
log "Phase 3 全部完成."

# ====== Phase 4: efSearch 参数扫描 ======
log "===== Phase 4: efSearch 参数扫描 ====="
for w in 0.0 0.4 0.6 0.8 1.0; do
    # NQ
    if [ -f "$DONE/phase4_sweep_nq_w${w}" ]; then
        log "  ef_sweep NQ w=${w} 已完成, 跳过."
    else
        log "  ef_sweep NQ w=${w} ..."
        python ef_sweep.py \
            --checkpoint experiments/models/w${w}/best_nq \
            --index experiments/indices/w${w}_best_nq \
            --embeddings experiments/embeddings/w${w}_best_nq \
            --test_file data_set/NQ/nq-test.csv \
            --corpus experiments/corpus/corpus_5m.tsv \
            --output experiments/results/nq/w${w}_ef_sweep.json \
            > logs/phase4_sweep_nq_w${w}.log 2>&1
        touch "$DONE/phase4_sweep_nq_w${w}"
        log "  ef_sweep NQ w=${w} 完成."
    fi

    # TriviaQA
    if [ -f "$DONE/phase4_sweep_trivia_w${w}" ]; then
        log "  ef_sweep TriviaQA w=${w} 已完成, 跳过."
    else
        log "  ef_sweep TriviaQA w=${w} ..."
        python ef_sweep.py \
            --checkpoint experiments/models/w${w}/best_trivia \
            --index experiments/indices/w${w}_best_trivia \
            --embeddings experiments/embeddings/w${w}_best_trivia \
            --test_file data_set/TriviaQA/trivia-test.csv \
            --corpus experiments/corpus/corpus_5m.tsv \
            --output experiments/results/trivia/w${w}_ef_sweep.json \
            > logs/phase4_sweep_trivia_w${w}.log 2>&1
        touch "$DONE/phase4_sweep_trivia_w${w}"
        log "  ef_sweep TriviaQA w=${w} 完成."
    fi
done
log "Phase 4 全部完成."

# ====== Phase 5: t-SNE 可视化 ======
log "===== Phase 5: t-SNE 可视化 ====="
if [ -f "$DONE/phase5_tsne_best_nq" ]; then
    log "  t-SNE best_nq 已完成, 跳过."
else
    python visualize_tsne.py \
        --compare_all \
        --ckpt_type best_nq \
        --output_dir experiments/plots \
        > logs/phase5_tsne_best_nq.log 2>&1
    touch "$DONE/phase5_tsne_best_nq"
    log "  t-SNE best_nq 完成."
fi

if [ -f "$DONE/phase5_tsne_best_trivia" ]; then
    log "  t-SNE best_trivia 已完成, 跳过."
else
    python visualize_tsne.py \
        --compare_all \
        --ckpt_type best_trivia \
        --output_dir experiments/plots \
        > logs/phase5_tsne_best_trivia.log 2>&1
    touch "$DONE/phase5_tsne_best_trivia"
    log "  t-SNE best_trivia 完成."
fi
log "Phase 5 完成."

# ====== Phase 6: 生成图表 ======
log "===== Phase 6: 生成图表 ====="
if [ -f "$DONE/phase6_plots" ]; then
    log "  图表已完成, 跳过."
else
    python plot_figures.py > logs/phase6_plots.log 2>&1
    touch "$DONE/phase6_plots"
    log "  Phase 6 完成."
fi

# ====== 完成 ======
log "======================================================"
log "全部实验完成!"
log "======================================================"
log "实验产出目录: experiments/"
log "  模型:    experiments/models/"
log "  嵌入:    experiments/embeddings/"
log "  索引:    experiments/indices/"
log "  结果:    experiments/results/"
log "  图表:    experiments/plots/"
log "  日志:    logs/"
log "  完成标记: logs/done/"
log "======================================================"
