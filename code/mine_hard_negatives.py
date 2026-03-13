"""
Hard Negative Mining 脚本

用 Stage 2 训练完成的模型对全量 psgs_w100.tsv（~21M passages）进行离线难负例挖掘。

流程：
  1. 编码全部 21M passages → float32 向量
  2. 构建临时 FAISS Flat 索引（精确最近邻搜索）
  3. 对训练集中每个 query 检索 top-200
  4. 去除已知正例（positive_ctxs 中的 passage_id）
  5. 保留前 50 个作为候选难负例
  6. 保存为 nq-train-mined.json / trivia-train-mined.json

用法：
  python mine_hard_negatives.py \
    --checkpoint_dir ./checkpoints/nq/best_model_stage2 \
    --dataset nq \
    --output_path ./data_set/NQ/nq-train-mined.json

参考：
  - experiments.md 4.2 节 "Stage 2 → Stage 3 过渡"
  - 3stage_hard_samples_mining.md
"""

import argparse
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


def load_corpus(corpus_path):
    """加载 psgs_w100.tsv 语料库。

    格式：id \\t text \\t title（带表头行）
    返回：(passage_ids, titles, texts) 三个列表
    """
    logger.info("Loading corpus from %s ...", corpus_path)
    passage_ids = []
    titles = []
    texts = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                # 跳过表头
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, text, title = parts[0], parts[1], parts[2]
            passage_ids.append(pid)
            texts.append(normalize_passage(text))
            titles.append(title)

            if (line_idx + 1) % 5000000 == 0:
                logger.info("  Loaded %d passages...", line_idx)

    logger.info("Corpus loaded: %d passages", len(passage_ids))
    return passage_ids, titles, texts


@torch.no_grad()
def encode_passages(model, tokenizer, titles, texts, args, device):
    """编码全部 passages，返回 numpy float32 向量矩阵。

    采用分批编码 + 追加到列表的策略，避免一次性分配 OOM。
    """
    model.eval()
    all_embs = []
    n = len(texts)

    logger.info("Encoding %d passages (batch_size=%d)...", n, args.encode_batch_size)
    t_start = time.time()

    for start in tqdm(range(0, n, args.encode_batch_size), desc="Encoding passages"):
        end = min(start + args.encode_batch_size, n)
        batch_titles = titles[start:end]
        batch_texts = texts[start:end]

        # Tokenize: [CLS] title [SEP] text [SEP]
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

        all_embs.append(emb.cpu().numpy())

    elapsed = time.time() - t_start
    logger.info("Passage encoding complete: %.1f seconds (%.0f passages/sec)", elapsed, n / elapsed)

    return np.concatenate(all_embs, axis=0)


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

    # 加载语料库
    passage_ids, titles, texts = load_corpus(args.corpus_path)

    # 构建 passage_id → index 映射
    pid_to_idx = {pid: idx for idx, pid in enumerate(passage_ids)}

    # 编码全部 passages
    passage_embs = encode_passages(model, tokenizer, titles, texts, args, device)
    dim = passage_embs.shape[1]
    logger.info("Passage embeddings shape: %s", passage_embs.shape)

    # 构建 FAISS Flat 索引
    logger.info("Building FAISS Flat index (dim=%d)...", dim)
    # 使用 inner product（向量已 L2 归一化，IP ≡ cosine）
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embs)
    logger.info("Index built: %d vectors", index.ntotal)

    # 释放 passage_embs 节省内存
    del passage_embs

    # 加载训练数据
    if args.dataset == "nq":
        train_path = os.path.join(args.data_dir, "NQ", "nq-train.json")
    else:
        train_path = os.path.join(args.data_dir, "TriviaQA", "trivia-train.json")

    logger.info("Loading training data from %s ...", train_path)
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    logger.info("Loaded %d training samples", len(train_data))

    # 收集 queries 和 positive passage ids
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

    # 编码 queries
    query_embs = encode_queries(model, tokenizer, questions, args, device)
    logger.info("Query embeddings shape: %s", query_embs.shape)

    # 批量检索 top-K
    logger.info("Searching top-%d for each query...", args.top_k)
    scores, result_ids = index.search(query_embs, args.top_k)
    logger.info("Search complete")

    # 构建 mined hard negatives
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
            # 去除已知正例
            if pid in pos_pids:
                continue
            hard_negative_ctxs.append({
                "title": titles[doc_idx],
                "text": texts[doc_idx],
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

    # 原子写入
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = args.output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(mined_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, args.output_path)
    logger.info("Saved mined data to %s", args.output_path)

    # 统计
    hn_counts = [len(s["hard_negative_ctxs"]) for s in mined_data]
    logger.info(
        "Hard negatives stats: min=%d, max=%d, mean=%.1f, median=%.1f",
        min(hn_counts), max(hn_counts), np.mean(hn_counts), np.median(hn_counts),
    )


if __name__ == "__main__":
    main()
