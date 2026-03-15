"""
Hard Negative Mining 脚本

用 Stage 2 训练完成的模型对全量 psgs_w100.tsv（~21M passages）进行离线难负例挖掘。

流程：
  1. 流式读取 TSV + 编码全部 21M passages → memory-mapped float32 向量文件
  2. 释放 model + GPU 显存
  3. 构建 FAISS Flat 索引（从 mmap 分批添加）
  4. 重新加载 model，编码 query，检索 top-200
  5. 释放 FAISS 索引
  6. 根据 doc_idx 从 TSV 按需读取 title/text
  7. 去除已知正例，保留前 50 个难负例
  8. 保存为 nq-train-mined.json / trivia-train-mined.json

内存优化（vs 旧版）：
  - 旧版峰值 RAM：passage 向量 60 GB (numpy) + FAISS 复制 60 GB + 文本 12 GB ≈ 132+ GB → OOM Kill
  - 新版峰值 RAM：FAISS 索引 60 GB + 杂项 < 10 GB ≈ 65-70 GB
  - 核心改动：向量写 mmap 文件不占 RAM；编码时流式读 TSV 不持有全部文本；
    检索完释放 FAISS 后再按需读 TSV 构建结果

用法：
  python mine_hard_negatives.py \\
    --checkpoint_dir ./checkpoints/nq/best_model_stage2 \\
    --dataset nq \\
    --output_path ./data_set/NQ/nq-train-mined.json

参考：
  - experiments.md 4.2 节 "Stage 2 → Stage 3 过渡"
  - 3stage_hard_samples_mining.md
"""

import argparse
import gc
import json
import logging
import os
import time

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models import BiEncoder
from src.utils.data_utils import normalize_question, normalize_passage

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Hard Negative Mining")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Stage 2 best model 目录（含 bi_encoder_config.json + bi_encoder.pt）")
    parser.add_argument("--dataset", type=str, required=True, choices=["nq", "trivia"])
    parser.add_argument("--data_dir", type=str, default="./data_set")
    parser.add_argument("--corpus_path", type=str, default="./data_set/psgs_w100.tsv",
                        help="psgs_w100.tsv 语料库路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出 mined JSON 路径")
    parser.add_argument("--top_k", type=int, default=200,
                        help="每个 query 检索的候选文档数")
    parser.add_argument("--keep_top_n", type=int, default=50,
                        help="去除正例后保留的难负例数")
    parser.add_argument("--encode_batch_size", type=int, default=512,
                        help="Passage 编码 batch size")
    parser.add_argument("--query_batch_size", type=int, default=256,
                        help="Query 编码 batch size")
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--max_query_length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", default=True)
    return parser.parse_args()


def load_corpus_ids_only(corpus_path):
    """仅加载 passage_ids（省 ~10 GB RAM，不加载 title/text）。"""
    logger.info("Loading passage IDs from %s ...", corpus_path)
    passage_ids = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 1:
                continue
            passage_ids.append(parts[0])

            if (line_idx + 1) % 5000000 == 0:
                logger.info("  Loaded %d passage IDs...", line_idx)

    logger.info("Loaded %d passage IDs", len(passage_ids))
    return passage_ids


def lookup_passages_from_tsv(corpus_path, doc_indices_set):
    """从 TSV 文件按需读取指定 doc_idx 的 title 和 text。

    Args:
        corpus_path: psgs_w100.tsv 路径
        doc_indices_set: 需要读取的 doc_idx 集合（0-based，不含表头）

    Returns:
        dict: {doc_idx: {"title": ..., "text": ...}}
    """
    logger.info("Looking up %d passages from TSV...", len(doc_indices_set))
    result = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            doc_idx = line_idx - 1  # 0-based
            if doc_idx in doc_indices_set:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    result[doc_idx] = {
                        "title": parts[2],
                        "text": parts[1],
                    }
                if len(result) == len(doc_indices_set):
                    break

    logger.info("Looked up %d passages", len(result))
    return result


def count_corpus_lines(corpus_path):
    """统计语料库行数（不含表头）。"""
    logger.info("Counting passages in %s ...", corpus_path)
    n = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    n -= 1  # 减去表头
    logger.info("Total passages: %d", n)
    return n


@torch.no_grad()
def encode_passages_to_mmap(model, tokenizer, corpus_path, args, device, mmap_path):
    """流式读取 TSV + 编码 → 直接写入 mmap 文件。

    不在 RAM 中持有全部 corpus 文本，也不持有全部向量。
    每次只有一个 batch 的文本和向量在内存中。

    Returns: (n_passages, dim)
    """
    model.eval()

    n_passages = count_corpus_lines(corpus_path)

    logger.info("Encoding %d passages (batch_size=%d) → mmap file...",
                n_passages, args.encode_batch_size)
    t_start = time.time()

    dim = None
    mmap_embs = None
    batch_titles = []
    batch_texts = []
    global_idx = 0

    def flush_batch():
        nonlocal dim, mmap_embs, global_idx
        if not batch_titles:
            return

        encodings = tokenizer(
            batch_titles,
            text_pair=batch_texts,
            max_length=args.max_passage_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        if args.fp16:
            with torch.cuda.amp.autocast():
                emb = model.encode_document(input_ids, attention_mask)
        else:
            emb = model.encode_document(input_ids, attention_mask)

        emb_np = emb.cpu().numpy()

        if mmap_embs is None:
            dim = emb_np.shape[1]
            mmap_embs = np.memmap(mmap_path, dtype="float32", mode="w+",
                                  shape=(n_passages, dim))
            logger.info("Created mmap file: %s (shape=[%d, %d], ~%.1f GB on disk)",
                        mmap_path, n_passages, dim, n_passages * dim * 4 / 1e9)

        mmap_embs[global_idx:global_idx + len(emb_np)] = emb_np
        global_idx += len(emb_np)

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                batch_titles.append("")
                batch_texts.append("")
            else:
                batch_titles.append(parts[2])
                batch_texts.append(normalize_passage(parts[1]))

            if len(batch_titles) >= args.encode_batch_size:
                flush_batch()
                batch_titles.clear()
                batch_texts.clear()

                if (global_idx // args.encode_batch_size) % 2000 == 0:
                    logger.info("  Encoded %d / %d passages...", global_idx, n_passages)

    # 最后一个不满 batch
    flush_batch()

    if mmap_embs is not None:
        mmap_embs.flush()

    elapsed = time.time() - t_start
    logger.info("Passage encoding complete: %.1f seconds (%.0f passages/sec)",
                elapsed, n_passages / elapsed)
    logger.info("Encoded %d passages, dim=%d", global_idx, dim)

    # 关闭 mmap 引用
    del mmap_embs

    return n_passages, dim


@torch.no_grad()
def encode_queries(model, tokenizer, questions, args, device):
    """编码所有 query，返回 numpy float32 向量矩阵。"""
    model.eval()
    all_embs = []
    n = len(questions)

    logger.info("Encoding %d queries...", n)

    for start in range(0, n, args.query_batch_size):
        end = min(start + args.query_batch_size, n)
        batch_qs = questions[start:end]

        encodings = tokenizer(
            batch_qs,
            max_length=args.max_query_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        if args.fp16:
            with torch.cuda.amp.autocast():
                emb = model.encode_query(input_ids, attention_mask)
        else:
            emb = model.encode_query(input_ids, attention_mask)

        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 加载模型
    model = BiEncoder.from_pretrained(args.checkpoint_dir)
    model.to(device)
    logger.info("Model loaded from %s", args.checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)

    # =========================================================================
    # Step 1: 编码 passages → mmap 文件（支持断点恢复）
    #   如果 mmap 文件已存在且大小匹配，跳过编码（省 ~7 小时）
    #   RAM 占用：仅 model (~0.5 GB) + 1 个 batch 的文本和向量 (~几十 MB)
    #   mmap 文件 ~60 GB 在磁盘，OS 按需 page in/out
    # =========================================================================
    output_dir = os.path.dirname(args.output_path) or "."
    mmap_path = os.path.join(output_dir, f".passage_embs_{args.dataset}.mmap")

    # 先数行数，用于校验 mmap 文件
    n_passages = count_corpus_lines(args.corpus_path)
    dim = 768  # bert-base hidden_size，后续会被实际值覆盖

    # 检查是否有可复用的 mmap 文件
    mmap_reused = False
    if os.path.exists(mmap_path):
        file_size = os.path.getsize(mmap_path)
        # mmap 文件大小 = n_passages * dim * 4 (float32)
        # 尝试用 dim=768 校验（BERT base 的 hidden_size）
        expected_size = n_passages * 768 * 4
        if file_size == expected_size:
            dim = 768
            logger.info("Found existing mmap file: %s (%.1f GB, matches %d x %d)",
                        mmap_path, file_size / 1e9, n_passages, dim)
            logger.info("Skipping passage encoding (reusing cached embeddings)")
            mmap_reused = True
        else:
            logger.info("Existing mmap file size mismatch (got %d, expected %d). Re-encoding.",
                        file_size, expected_size)
            os.remove(mmap_path)

    if not mmap_reused:
        n_passages, dim = encode_passages_to_mmap(
            model, tokenizer, args.corpus_path, args, device, mmap_path
        )

    # 释放 model 和 GPU 显存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Model released, GPU memory freed")

    # =========================================================================
    # Step 2: 构建 FAISS 索引（从 mmap 分批添加）
    #   RAM 占用：FAISS 索引逐步增长到 ~60 GB + 每次 1M batch ~3 GB
    #   注意：FAISS 内部 std::vector 可能在增长时 realloc
    # =========================================================================
    passage_embs = np.memmap(mmap_path, dtype="float32", mode="r", shape=(n_passages, dim))
    logger.info("Passage embeddings mmap opened: shape=(%d, %d)", n_passages, dim)

    logger.info("Building FAISS Flat index (dim=%d, n=%d)...", dim, n_passages)
    index = faiss.IndexFlatIP(dim)

    ADD_BATCH = 1_000_000
    for add_start in range(0, n_passages, ADD_BATCH):
        add_end = min(add_start + ADD_BATCH, n_passages)
        batch = np.array(passage_embs[add_start:add_end])
        index.add(batch)
        del batch
        if (add_start // ADD_BATCH) % 5 == 0:
            logger.info("  Added %d / %d vectors to index", add_end, n_passages)

    logger.info("Index built: %d vectors", index.ntotal)

    del passage_embs
    gc.collect()

    # =========================================================================
    # Step 3: 加载训练数据 + 编码 queries
    # =========================================================================
    if args.dataset == "nq":
        train_path = os.path.join(args.data_dir, "NQ", "nq-train.json")
    else:
        train_path = os.path.join(args.data_dir, "TriviaQA", "trivia-train.json")

    logger.info("Loading training data from %s ...", train_path)
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info("Loaded %d training samples", len(train_data))

    questions = []
    positive_pid_sets = []
    valid_indices = []

    for idx, sample in enumerate(train_data):
        if not sample.get("positive_ctxs"):
            continue
        valid_indices.append(idx)
        questions.append(normalize_question(sample["question"]))
        pos_pids = set()
        for pctx in sample["positive_ctxs"]:
            pid = str(pctx.get("passage_id", ""))
            if pid:
                pos_pids.add(pid)
        positive_pid_sets.append(pos_pids)

    logger.info("Valid queries for mining: %d", len(questions))

    # 重新加载 model 编码 queries
    logger.info("Reloading model for query encoding...")
    model = BiEncoder.from_pretrained(args.checkpoint_dir)
    model.to(device)

    query_embs = encode_queries(model, tokenizer, questions, args, device)
    logger.info("Query embeddings shape: %s", query_embs.shape)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================================
    # Step 4: 检索 top-K
    # =========================================================================
    logger.info("Searching top-%d for each query...", args.top_k)
    scores, result_ids = index.search(query_embs, args.top_k)
    logger.info("Search complete")

    # 释放 FAISS 索引（~60 GB）
    del index
    del query_embs
    gc.collect()
    logger.info("FAISS index released, ~60 GB RAM freed")

    # =========================================================================
    # Step 5: 按需读取 passage 内容
    #   先收集所有需要的 doc_idx，然后一次遍历 TSV 读取
    #   RAM 占用：passage_ids ~2 GB + lookup 结果（远小于全量）
    # =========================================================================
    passage_ids = load_corpus_ids_only(args.corpus_path)

    needed_doc_indices = set()
    for i in range(len(valid_indices)):
        for rank in range(args.top_k):
            doc_idx = int(result_ids[i, rank])
            if doc_idx >= 0:
                needed_doc_indices.add(doc_idx)
    logger.info("Unique passages needed for results: %d", len(needed_doc_indices))

    passage_lookup = lookup_passages_from_tsv(args.corpus_path, needed_doc_indices)

    # =========================================================================
    # Step 6: 构建 mined hard negatives
    # =========================================================================
    logger.info("Building mined hard negatives (keep_top_n=%d)...", args.keep_top_n)
    mined_data = []

    for i, orig_idx in enumerate(tqdm(valid_indices, desc="Mining")):
        sample = train_data[orig_idx]
        pos_pids = positive_pid_sets[i]

        hard_negative_ctxs = []
        for rank in range(args.top_k):
            doc_idx = int(result_ids[i, rank])
            if doc_idx < 0:
                continue
            pid = passage_ids[doc_idx]
            if pid in pos_pids:
                continue

            info = passage_lookup.get(doc_idx, {})
            hard_negative_ctxs.append({
                "title": info.get("title", ""),
                "text": info.get("text", ""),
                "passage_id": pid,
                "score": float(scores[i, rank]),
            })
            if len(hard_negative_ctxs) >= args.keep_top_n:
                break

        mined_sample = {
            "dataset": sample.get("dataset", ""),
            "question": sample["question"],
            "answers": sample.get("answers", []),
            "positive_ctxs": sample["positive_ctxs"],
            "negative_ctxs": [],
            "hard_negative_ctxs": hard_negative_ctxs,
        }
        mined_data.append(mined_sample)

    logger.info("Mined %d samples", len(mined_data))

    # =========================================================================
    # Step 7: 保存结果 + 清理
    # =========================================================================
    save_dir = os.path.dirname(args.output_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    tmp_path = args.output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(mined_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, args.output_path)
    logger.info("Saved mined data to %s", args.output_path)

    # 清理 mmap 临时文件
    if os.path.exists(mmap_path):
        os.remove(mmap_path)
        logger.info("Cleaned up mmap file: %s", mmap_path)

    # 统计
    hn_counts = [len(s["hard_negative_ctxs"]) for s in mined_data]
    logger.info(
        "Hard negatives stats: min=%d, max=%d, mean=%.1f, median=%.1f",
        min(hn_counts), max(hn_counts), np.mean(hn_counts), np.median(hn_counts),
    )


if __name__ == "__main__":
    main()
