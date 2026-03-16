"""
BEIR Zero-shot 评测脚本

对 5 个 BEIR 数据集做 zero-shot 密集检索评测，报告 NDCG@10 和 Recall@100。
使用 FAISS Flat 精确搜索（不涉及 ANN 索引）。

支持模型：
  - dacl-dr: 我们训练的 BiEncoder (query_encoder + doc_encoder)
  - dpr: facebook/dpr-*-single-nq-base (DPRQuestionEncoder + DPRContextEncoder)
  - ance: castorini/ance-dpr-*-multi (同 DPR 架构)
  - contriever: facebook/contriever-msmarco (共享 encoder, mean pooling)

数据集放置路径：
  ./data_set/beir/{dataset_name}/
  例如: ./data_set/beir/scifact/corpus.jsonl

用法：
  # 评测 DACL-DR (w=0.4)
  python beir_evaluate.py \
    --model_type dacl-dr \
    --model_path ./checkpoints/nq/best_model_nq \
    --datasets scifact nfcorpus fiqa trec-covid fever \
    --data_dir ./data_set/beir \
    --output_dir ./results/beir \
    --batch_size 128

  # 评测 DPR
  python beir_evaluate.py \
    --model_type dpr \
    --query_encoder_path ./dpr-backbone/question \
    --ctx_encoder_path ./dpr-backbone/context \
    --datasets scifact nfcorpus fiqa trec-covid fever \
    --data_dir ./data_set/beir \
    --output_dir ./results/beir \
    --batch_size 128

  # 评测 ANCE
  python beir_evaluate.py \
    --model_type ance \
    --query_encoder_path ./ance-backbone/question \
    --ctx_encoder_path ./ance-backbone/context \
    --datasets scifact nfcorpus fiqa trec-covid fever \
    --data_dir ./data_set/beir \
    --output_dir ./results/beir \
    --batch_size 128

  # 评测 Contriever
  python beir_evaluate.py \
    --model_type contriever \
    --model_path ./contriever-backbone \
    --datasets scifact nfcorpus fiqa trec-covid fever \
    --data_dir ./data_set/beir \
    --output_dir ./results/beir \
    --batch_size 128

  # 汇总所有结果
  python beir_evaluate.py --summarize --output_dir ./results/beir

参考：
  - experiments.md 第七节 (实验5: BEIR Zero-shot 泛化评测)
  - BEIR.md (操作手册)
"""

import argparse
import glob
import json
import logging
import os
import time

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder, DPRContextEncoder

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据加载（不依赖 beir 库，直接读取 BEIR 标准 JSONL 格式）
# ---------------------------------------------------------------------------

def load_beir_dataset(data_folder):
    """加载 BEIR 数据集。

    Args:
        data_folder: 数据集目录，包含 corpus.jsonl, queries.jsonl, qrels/test.tsv

    Returns:
        corpus: dict[str, dict] — {doc_id: {"title": ..., "text": ...}}
        queries: dict[str, str] — {query_id: query_text}
        qrels: dict[str, dict[str, int]] — {query_id: {doc_id: relevance_score}}
    """
    # 加载 corpus
    corpus = {}
    corpus_path = os.path.join(data_folder, "corpus.jsonl")
    logger.info("Loading corpus from %s ...", corpus_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            doc_id = str(obj["_id"])
            corpus[doc_id] = {
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
            }
    logger.info("  Corpus: %d documents", len(corpus))

    # 加载 queries
    queries = {}
    queries_path = os.path.join(data_folder, "queries.jsonl")
    logger.info("Loading queries from %s ...", queries_path)
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            query_id = str(obj["_id"])
            queries[query_id] = obj["text"]
    logger.info("  Queries: %d total", len(queries))

    # 加载 qrels (test split)
    qrels = {}
    qrels_path = os.path.join(data_folder, "qrels", "test.tsv")
    logger.info("Loading qrels from %s ...", qrels_path)
    with open(qrels_path, "r", encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                # 跳过表头（如果有）
                if line.strip().startswith("query-id") or line.strip().startswith("query_id"):
                    continue
                # 如果第一行不是表头，回退处理
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            query_id, doc_id, score = str(parts[0]), str(parts[1]), int(parts[2])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score

    # 只保留 qrels 中有标注的 queries
    test_query_ids = set(qrels.keys())
    queries = {qid: text for qid, text in queries.items() if qid in test_query_ids}
    logger.info("  Test queries (with qrels): %d", len(queries))

    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Encoder 加载
# ---------------------------------------------------------------------------

class BaseEncoder:
    """Encoder 基类，统一接口。"""

    def __init__(self, device):
        self.device = device

    def encode_queries(self, texts, batch_size, max_length=256):
        raise NotImplementedError

    def encode_corpus(self, corpus_list, batch_size, max_length=256):
        raise NotImplementedError


class DACLDREncoder(BaseEncoder):
    """DACL-DR BiEncoder."""

    def __init__(self, model_path, device):
        super().__init__(device)
        from src.models import BiEncoder
        self.model = BiEncoder.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_name)
        logger.info("Loaded DACL-DR BiEncoder from %s", model_path)

    @torch.no_grad()
    def encode_queries(self, texts, batch_size, max_length=256):
        all_embs = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, max_length=max_length, truncation=True,
                                 padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.cuda.amp.autocast():
                emb = self.model.encode_query(input_ids, attention_mask)
            # DACL-DR 内部已 L2 归一化，但为保险再做一次
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def encode_corpus(self, corpus_list, batch_size, max_length=256):
        all_embs = []
        titles = [item["title"] for item in corpus_list]
        texts = [item["text"] for item in corpus_list]
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
            end = min(start + batch_size, len(texts))
            batch_titles = titles[start:end]
            batch_texts = texts[start:end]
            # [CLS] title [SEP] text [SEP]
            enc = self.tokenizer(batch_titles, text_pair=batch_texts,
                                 max_length=max_length, truncation=True,
                                 padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.cuda.amp.autocast():
                emb = self.model.encode_document(input_ids, attention_mask)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


class DPREncoder(BaseEncoder):
    """DPR / ANCE Encoder (双 encoder, DPRQuestionEncoder + DPRContextEncoder)."""

    def __init__(self, query_encoder_path, ctx_encoder_path, device):
        super().__init__(device)
        self.q_model = DPRQuestionEncoder.from_pretrained(query_encoder_path)
        self.q_model.to(device)
        self.q_model.eval()
        self.q_tokenizer = AutoTokenizer.from_pretrained(query_encoder_path)

        self.ctx_model = DPRContextEncoder.from_pretrained(ctx_encoder_path)
        self.ctx_model.to(device)
        self.ctx_model.eval()
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_encoder_path)
        logger.info("Loaded DPR/ANCE encoder from q=%s, ctx=%s",
                     query_encoder_path, ctx_encoder_path)

    @torch.no_grad()
    def encode_queries(self, texts, batch_size, max_length=256):
        all_embs = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding queries"):
            batch = texts[start:start + batch_size]
            enc = self.q_tokenizer(batch, max_length=max_length, truncation=True,
                                   padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.cuda.amp.autocast():
                emb = self.q_model(input_ids=input_ids,
                                   attention_mask=attention_mask).pooler_output
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    @torch.no_grad()
    def encode_corpus(self, corpus_list, batch_size, max_length=256):
        all_embs = []
        titles = [item["title"] for item in corpus_list]
        texts = [item["text"] for item in corpus_list]
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
            end = min(start + batch_size, len(texts))
            batch_titles = titles[start:end]
            batch_texts = texts[start:end]
            enc = self.ctx_tokenizer(batch_titles, text_pair=batch_texts,
                                     max_length=max_length, truncation=True,
                                     padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.cuda.amp.autocast():
                emb = self.ctx_model(input_ids=input_ids,
                                     attention_mask=attention_mask).pooler_output
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


class ContrieverEncoder(BaseEncoder):
    """Contriever Encoder (共享 encoder, mean pooling)."""

    def __init__(self, model_path, device):
        super().__init__(device)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Loaded Contriever from %s", model_path)

    @staticmethod
    def _mean_pooling(token_embs, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embs / sum_mask

    @torch.no_grad()
    def _encode_texts(self, texts, batch_size, max_length):
        all_embs = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, max_length=max_length, truncation=True,
                                 padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                emb = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    def encode_queries(self, texts, batch_size, max_length=256):
        return self._encode_texts(texts, batch_size, max_length)

    def encode_corpus(self, corpus_list, batch_size, max_length=256):
        # Contriever: 将 title + sep + text 拼接为单一文本
        texts = []
        for item in corpus_list:
            title = item.get("title", "")
            text = item.get("text", "")
            if title:
                texts.append(title + " " + text)
            else:
                texts.append(text)
        return self._encode_texts(texts, batch_size, max_length)


def build_encoder(args, device):
    """根据命令行参数构建 encoder。"""
    if args.model_type == "dacl-dr":
        return DACLDREncoder(args.model_path, device)
    elif args.model_type in ("dpr", "ance"):
        return DPREncoder(args.query_encoder_path, args.ctx_encoder_path, device)
    elif args.model_type == "contriever":
        return ContrieverEncoder(args.model_path, device)
    else:
        raise ValueError("Unknown model_type: %s" % args.model_type)


# ---------------------------------------------------------------------------
# 评测指标
# ---------------------------------------------------------------------------

def compute_ndcg_at_k(qrels, results, k):
    """计算 NDCG@K。

    Args:
        qrels: dict[str, dict[str, int]] — {query_id: {doc_id: relevance}}
        results: dict[str, dict[str, float]] — {query_id: {doc_id: score}}
        k: int
    """
    ndcg_scores = []
    for qid in qrels:
        if qid not in results:
            ndcg_scores.append(0.0)
            continue

        # 获取 top-k 结果
        sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]

        # DCG
        dcg = 0.0
        for rank, (doc_id, _score) in enumerate(sorted_docs):
            rel = qrels[qid].get(doc_id, 0)
            dcg += (2 ** rel - 1) / np.log2(rank + 2)  # rank+2 因为 rank 从 0 开始

        # Ideal DCG
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels):
            idcg += (2 ** rel - 1) / np.log2(rank + 2)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_recall_at_k(qrels, results, k):
    """计算 Recall@K。

    Args:
        qrels: dict[str, dict[str, int]] — {query_id: {doc_id: relevance}}
        results: dict[str, dict[str, float]] — {query_id: {doc_id: score}}
        k: int
    """
    recall_scores = []
    for qid in qrels:
        if qid not in results:
            recall_scores.append(0.0)
            continue

        # 获取 top-k doc ids
        sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        retrieved_ids = set(doc_id for doc_id, _ in sorted_docs)

        # 相关文档（relevance > 0）
        relevant_ids = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)

        if len(relevant_ids) == 0:
            continue

        recall_scores.append(len(retrieved_ids & relevant_ids) / len(relevant_ids))

    return np.mean(recall_scores) if recall_scores else 0.0


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def retrieve_with_faiss(query_embs, corpus_embs, corpus_ids, top_k=100):
    """使用 FAISS Flat 精确搜索。

    Args:
        query_embs: np.ndarray [Q, D]
        corpus_embs: np.ndarray [C, D]
        corpus_ids: list[str], 与 corpus_embs 行对应的 doc_id
        top_k: 检索的 top-K 数量

    Returns:
        results: dict[str, dict[str, float]] — 可直接用于评测的格式
                 （实际上 query_id 对应的是整数索引的字符串，需要外部映射）
    """
    dim = corpus_embs.shape[1]
    logger.info("Building FAISS Flat index: %d vectors, dim=%d", len(corpus_ids), dim)

    # 构建 Inner Product 索引（向量已 L2 归一化）
    index = faiss.IndexFlatIP(dim)
    # 分批添加（避免一次性 add 太大的矩阵）
    add_batch_size = 500000
    for start in range(0, len(corpus_ids), add_batch_size):
        end = min(start + add_batch_size, len(corpus_ids))
        index.add(corpus_embs[start:end].astype(np.float32))
    logger.info("Index built: %d vectors", index.ntotal)

    # 搜索
    logger.info("Searching top-%d for %d queries ...", top_k, len(query_embs))
    scores, indices = index.search(query_embs.astype(np.float32), top_k)

    # 转换为 {query_idx: {doc_id: score}} 格式
    results = {}
    for qi in range(len(query_embs)):
        doc_scores = {}
        for rank in range(top_k):
            idx = indices[qi, rank]
            if idx < 0:
                continue
            doc_id = corpus_ids[idx]
            doc_scores[doc_id] = float(scores[qi, rank])
        results[qi] = doc_scores

    return results


# ---------------------------------------------------------------------------
# 单数据集评测
# ---------------------------------------------------------------------------

def evaluate_single_dataset(encoder, dataset_name, data_dir, output_dir,
                            batch_size, max_length, model_name):
    """对单个 BEIR 数据集评测。"""
    data_folder = os.path.join(data_dir, dataset_name)
    if not os.path.exists(data_folder):
        logger.warning("Dataset folder not found: %s, skipping.", data_folder)
        return None

    logger.info("=" * 60)
    logger.info("Evaluating on %s", dataset_name)
    logger.info("=" * 60)

    # 加载数据
    corpus, queries, qrels = load_beir_dataset(data_folder)

    # 准备 corpus 列表（保持顺序）
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[cid] for cid in corpus_ids]

    # 准备 query 列表（保持顺序）
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    # 编码
    t0 = time.time()
    query_embs = encoder.encode_queries(query_texts, batch_size, max_length)
    t_query = time.time() - t0

    t0 = time.time()
    corpus_embs = encoder.encode_corpus(corpus_list, batch_size, max_length)
    t_corpus = time.time() - t0

    logger.info("Query encoding: %d queries in %.1fs", len(query_texts), t_query)
    logger.info("Corpus encoding: %d docs in %.1fs", len(corpus_list), t_corpus)

    # 检索
    t0 = time.time()
    raw_results = retrieve_with_faiss(query_embs, corpus_embs, corpus_ids, top_k=100)
    t_search = time.time() - t0
    logger.info("Search: %.1fs", t_search)

    # 映射 query_idx -> query_id
    results = {}
    for qi, qid in enumerate(query_ids):
        if qi in raw_results:
            results[qid] = raw_results[qi]

    # 计算指标
    ndcg_10 = compute_ndcg_at_k(qrels, results, k=10)
    recall_100 = compute_recall_at_k(qrels, results, k=100)

    logger.info("Results on %s:", dataset_name)
    logger.info("  NDCG@10:    %.4f", ndcg_10)
    logger.info("  Recall@100: %.4f", recall_100)

    # 保存
    result_obj = {
        "model": model_name,
        "dataset": dataset_name,
        "NDCG@10": round(ndcg_10, 4),
        "Recall@100": round(recall_100, 4),
        "num_queries": len(query_texts),
        "num_corpus": len(corpus_list),
        "time_query_encode_s": round(t_query, 1),
        "time_corpus_encode_s": round(t_corpus, 1),
        "time_search_s": round(t_search, 1),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{dataset_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False)
    logger.info("Saved to %s", out_path)

    # 释放内存
    del query_embs, corpus_embs, raw_results, results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_obj


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def summarize_results(output_dir):
    """汇总所有结果为 summary.json。"""
    pattern = os.path.join(output_dir, "*_*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith("summary.json")]

    if not files:
        logger.warning("No result files found in %s", output_dir)
        return

    summary = {}
    for fpath in sorted(files):
        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)
        model = obj["model"]
        dataset = obj["dataset"]
        if model not in summary:
            summary[model] = {}
        summary[model][dataset] = {
            "NDCG@10": obj["NDCG@10"],
            "Recall@100": obj["Recall@100"],
        }

    # 计算平均值
    for model in summary:
        datasets = summary[model]
        if datasets:
            avg_ndcg = np.mean([v["NDCG@10"] for v in datasets.values()])
            avg_recall = np.mean([v["Recall@100"] for v in datasets.values()])
            summary[model]["Avg"] = {
                "NDCG@10": round(float(avg_ndcg), 4),
                "Recall@100": round(float(avg_recall), 4),
            }

    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to %s", out_path)

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("BEIR Zero-shot Evaluation Summary")
    print("=" * 80)
    all_datasets = ["scifact", "nfcorpus", "fiqa", "trec-covid", "fever", "Avg"]
    header = f"{'Model':<25}" + "".join(f"{'  ' + ds:>14}" for ds in all_datasets)
    print(header)
    print("-" * len(header))

    for model in sorted(summary.keys()):
        row_ndcg = f"{'  ' + model + ' (NDCG@10)':<25}"
        row_recall = f"{'  ' + model + ' (R@100)':<25}"
        for ds in all_datasets:
            if ds in summary[model]:
                row_ndcg += f"{summary[model][ds]['NDCG@10']:>14.4f}"
                row_recall += f"{summary[model][ds]['Recall@100']:>14.4f}"
            else:
                row_ndcg += f"{'--':>14}"
                row_recall += f"{'--':>14}"
        print(row_ndcg)
        print(row_recall)
        print()

    return summary


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="BEIR Zero-shot Evaluation")

    parser.add_argument("--model_type", type=str, default=None,
                        choices=["dacl-dr", "dpr", "ance", "contriever"],
                        help="模型类型")

    # DACL-DR / Contriever 使用 --model_path
    parser.add_argument("--model_path", type=str, default=None,
                        help="DACL-DR BiEncoder 目录 或 Contriever 模型目录")

    # DPR / ANCE 使用分开的 encoder 路径
    parser.add_argument("--query_encoder_path", type=str, default=None,
                        help="DPR/ANCE 的 question encoder 目录")
    parser.add_argument("--ctx_encoder_path", type=str, default=None,
                        help="DPR/ANCE 的 context encoder 目录")

    # 模型别名（用于结果文件名）
    parser.add_argument("--model_name", type=str, default=None,
                        help="结果文件中的模型名称（默认自动推断）")

    parser.add_argument("--datasets", nargs="+",
                        default=["scifact", "nfcorpus", "fiqa", "trec-covid", "fever"],
                        help="要评测的 BEIR 数据集名称列表")
    parser.add_argument("--data_dir", type=str, default="./data_set/beir",
                        help="BEIR 数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./results/beir",
                        help="结果输出目录")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)

    # 汇总模式
    parser.add_argument("--summarize", action="store_true",
                        help="仅汇总已有结果，不运行评测")

    return parser.parse_args()


def main():
    args = get_args()

    # 汇总模式
    if args.summarize:
        summarize_results(args.output_dir)
        return

    # 参数校验
    if args.model_type is None:
        raise ValueError("--model_type is required when not using --summarize")

    if args.model_type in ("dacl-dr", "contriever") and args.model_path is None:
        raise ValueError("--model_path is required for model_type=%s" % args.model_type)

    if args.model_type in ("dpr", "ance"):
        if args.query_encoder_path is None or args.ctx_encoder_path is None:
            raise ValueError("--query_encoder_path and --ctx_encoder_path are required "
                             "for model_type=%s" % args.model_type)

    # 推断模型名称
    if args.model_name is None:
        args.model_name = args.model_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 构建 encoder
    encoder = build_encoder(args, device)

    # 逐数据集评测
    all_results = {}
    for dataset_name in args.datasets:
        result = evaluate_single_dataset(
            encoder=encoder,
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_length=args.max_length,
            model_name=args.model_name,
        )
        if result is not None:
            all_results[dataset_name] = result

    # 打印汇总
    if all_results:
        print("\n" + "=" * 60)
        print(f"Results for {args.model_name}")
        print("=" * 60)
        for ds, res in all_results.items():
            print(f"  {ds:<15} NDCG@10={res['NDCG@10']:.4f}  Recall@100={res['Recall@100']:.4f}")

        avg_ndcg = np.mean([r["NDCG@10"] for r in all_results.values()])
        avg_recall = np.mean([r["Recall@100"] for r in all_results.values()])
        print(f"  {'Avg':<15} NDCG@10={avg_ndcg:.4f}  Recall@100={avg_recall:.4f}")


if __name__ == "__main__":
    main()
