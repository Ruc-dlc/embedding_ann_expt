"""
检索评估脚本

对每个模型×每种索引，评估：
  - Top-K Accuracy (K=10,20,50,100): has_answer token 级匹配
  - Recall@K (K=20,100): 对比 Flat 精确搜索
  - QPS / Latency (ms/query)
  - 距离计算次数 (HNSW: faiss.cvar.hnsw_stats.ndis)

支持 HNSW/IVF/IVF-PQ 的参数扫描。

用法：
  python evaluate.py \
    --embeddings_dir ./embeddings/dacl-dr \
    --index_dir ./embeddings/dacl-dr/indexes \
    --dataset nq \
    --model_type dacl-dr \
    --model_path ./checkpoints/nq/best_model_nq \
    --output_path ./results/dacl-dr_nq.json

参考：
  - experiments.md 第七节、第十二节 Phase 4
"""

import argparse
import ast
import json
import logging
import os
import time

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils.data_utils import normalize_question, has_answer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="包含 passage_embeddings.npy 和 passage_ids.json 的目录")
    parser.add_argument("--index_dir", type=str, required=True,
                        help="包含 FAISS 索引文件的目录")
    parser.add_argument("--dataset", type=str, required=True, choices=["nq", "trivia"])
    parser.add_argument("--data_dir", type=str, default="./data_set")
    parser.add_argument("--corpus_path", type=str, default="./data_set/psgs_w100.tsv")

    # Query encoder
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["dacl-dr", "dpr", "ance", "contriever"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="dacl-dr: BiEncoder 目录; dpr/ance: question_encoder 目录; contriever: 共享 encoder 目录")

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=256)
    parser.add_argument("--query_batch_size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", default=True)

    # 参数扫描
    parser.add_argument("--hnsw_ef_search", type=str, default="8,16,32,64,128,256,512")
    parser.add_argument("--ivf_nprobe", type=str, default="1,4,8,16,32,64,128,256")
    parser.add_argument("--top_k_values", type=str, default="10,20,50,100")

    return parser.parse_args()


def load_test_data(dataset, data_dir):
    """加载测试集 (question, answers)。

    测试集格式为 TSV：question \\t answers
    answers 列是 Python 列表的字符串表示。
    """
    if dataset == "nq":
        path = os.path.join(data_dir, "NQ", "nq-test.csv")
    else:
        path = os.path.join(data_dir, "TriviaQA", "trivia-test.csv")

    questions = []
    answers_list = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = normalize_question(parts[0])
            answers = ast.literal_eval(parts[1])
            questions.append(question)
            answers_list.append(answers)

    logger.info("Loaded %d test queries from %s", len(questions), path)
    return questions, answers_list


def load_corpus_text(corpus_path):
    """加载语料库原始文本，用于 has_answer 判断。

    返回 dict: passage_id (str) -> (text, title)
    """
    logger.info("Loading corpus text from %s ...", corpus_path)
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, text, title = parts[0], parts[1], parts[2]
            corpus[pid] = (text, title)

            if (line_idx + 1) % 5000000 == 0:
                logger.info("  Loaded %d passages...", line_idx)

    logger.info("Corpus text loaded: %d passages", len(corpus))
    return corpus


def load_query_encoder(model_type, model_path, device):
    """加载 query encoder。

    各模型的 query encoder 不同于 context encoder：
      - dacl-dr: BiEncoder.encode_query
      - dpr: DPRQuestionEncoder
      - ance: DPRQuestionEncoder (castorini/ance-dpr-question-multi)
      - contriever: 同 context encoder (共享)

    Returns:
        (query_encode_fn, tokenizer)
    """
    if model_type == "dacl-dr":
        from src.models import BiEncoder
        model = BiEncoder.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)

        def query_encode_fn(input_ids, attention_mask):
            return model.encode_query(input_ids, attention_mask)

        return query_encode_fn, tokenizer

    elif model_type == "dpr":
        from transformers import DPRQuestionEncoder
        q_model = DPRQuestionEncoder.from_pretrained(model_path)
        q_model.to(device)
        q_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def query_encode_fn(input_ids, attention_mask):
            return q_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        return query_encode_fn, tokenizer

    elif model_type == "ance":
        from transformers import DPRQuestionEncoder
        q_model = DPRQuestionEncoder.from_pretrained(model_path)
        q_model.to(device)
        q_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def query_encode_fn(input_ids, attention_mask):
            return q_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        return query_encode_fn, tokenizer

    elif model_type == "contriever":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def query_encode_fn(input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
            sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embs / sum_mask

        return query_encode_fn, tokenizer

    else:
        raise ValueError("Unknown model_type: %s" % model_type)


@torch.no_grad()
def encode_test_queries(model_type, model_path, questions, args, device):
    """编码测试 queries，返回 numpy float32 矩阵 [Q, D]（已 L2 归一化）。"""
    query_encode_fn, tokenizer = load_query_encoder(model_type, model_path, device)

    all_embs = []
    for start in range(0, len(questions), args.query_batch_size):
        end = min(start + args.query_batch_size, len(questions))
        batch = questions[start:end]

        encodings = tokenizer(
            batch,
            max_length=args.max_query_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        if args.fp16:
            with torch.cuda.amp.autocast():
                emb = query_encode_fn(input_ids, attention_mask)
        else:
            emb = query_encode_fn(input_ids, attention_mask)

        # 统一 L2 归一化
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        all_embs.append(emb.cpu().numpy())

    query_embs = np.concatenate(all_embs, axis=0)
    logger.info("Encoded %d queries -> shape %s", len(questions), query_embs.shape)
    return query_embs


def compute_top_k_accuracy(result_pids, answers_list, corpus, k_values):
    """计算 Top-K Accuracy (has_answer 匹配)。

    Args:
        result_pids: list of list of str, 每个 query 的检索结果 passage_ids
        answers_list: list of list of str
        corpus: dict pid -> (text, title)
        k_values: list of int

    Returns:
        dict: {k: accuracy}
    """
    max_k = max(k_values)
    num_queries = len(answers_list)
    hits = np.zeros((num_queries, max_k), dtype=bool)

    for qi in range(num_queries):
        answers = answers_list[qi]
        pids = result_pids[qi][:max_k]
        for rank, pid in enumerate(pids):
            if pid in corpus:
                text = corpus[pid][0]
                if has_answer(answers, text):
                    hits[qi, rank] = True

    results = {}
    for k in k_values:
        top_k_hit = hits[:, :k].any(axis=1).mean()
        results[k] = float(top_k_hit)

    return results


def compute_recall_at_k(result_ids_ann, result_ids_flat, k_values):
    """计算 Recall@K（以 Flat 精确搜索为 ground truth）。

    Args:
        result_ids_ann: [Q, max_k] ANN 搜索结果 (FAISS index IDs)
        result_ids_flat: [Q, max_k] Flat 精确搜索结果 (FAISS index IDs)
        k_values: list of int

    Returns:
        dict: {k: recall}
    """
    results = {}
    for k in k_values:
        recall_sum = 0.0
        for qi in range(result_ids_ann.shape[0]):
            ann_set = set(result_ids_ann[qi, :k].tolist())
            flat_set = set(result_ids_flat[qi, :k].tolist())
            if len(flat_set) > 0:
                recall_sum += len(ann_set & flat_set) / len(flat_set)
        results[k] = recall_sum / result_ids_ann.shape[0]
    return results


def search_with_timing(index, query_embs, k, single_thread=True):
    """搜索并测量 latency（逐条 query 计时）。

    Args:
        index: FAISS index
        query_embs: [Q, D] float32
        k: top-k
        single_thread: 使用单线程以准确测量 latency

    Returns:
        (scores [Q,k], indices [Q,k], avg_latency_ms, qps)
    """
    if single_thread:
        faiss.omp_set_num_threads(1)

    n_queries = query_embs.shape[0]
    latencies = []
    all_scores = []
    all_indices = []

    for i in range(n_queries):
        q = query_embs[i:i+1]
        t0 = time.time()
        D, I = index.search(q, k)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)  # ms
        all_scores.append(D)
        all_indices.append(I)

    avg_latency = np.mean(latencies)
    qps = 1000.0 / avg_latency if avg_latency > 0 else 0

    scores = np.concatenate(all_scores, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    if single_thread:
        faiss.omp_set_num_threads(0)  # 0 = auto

    return scores, indices, avg_latency, qps


def get_hnsw_distance_count(index, query_embs, k, ef_search):
    """获取 HNSW 的平均距离计算次数。

    仅对前 min(100, Q) 个 query 测量以节省时间。
    """
    faiss.omp_set_num_threads(1)
    index.hnsw.efSearch = ef_search

    n = min(100, query_embs.shape[0])
    total_ndis = 0
    for i in range(n):
        faiss.cvar.hnsw_stats.reset()
        index.search(query_embs[i:i+1], k)
        total_ndis += faiss.cvar.hnsw_stats.ndis

    faiss.omp_set_num_threads(0)
    return total_ndis / n


def main():
    args = get_args()
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    k_values = [int(x) for x in args.top_k_values.split(",")]
    max_k = max(k_values)
    hnsw_ef_values = [int(x) for x in args.hnsw_ef_search.split(",")]
    ivf_nprobe_values = [int(x) for x in args.ivf_nprobe.split(",")]

    # 加载 passage_ids (index position -> passage_id 映射)
    ids_path = os.path.join(args.embeddings_dir, "passage_ids.json")
    with open(ids_path, "r") as f:
        passage_ids = json.load(f)
    logger.info("Loaded %d passage IDs", len(passage_ids))

    # 加载测试数据
    questions, answers_list = load_test_data(args.dataset, args.data_dir)

    # 加载语料库文本 (has_answer 需要原始 text)
    corpus = load_corpus_text(args.corpus_path)

    # 编码 test queries
    logger.info("Encoding test queries with %s encoder...", args.model_type)
    query_embs = encode_test_queries(args.model_type, args.model_path, questions, args, device)
    query_embs = query_embs.astype(np.float32)

    results = {
        "model_type": args.model_type,
        "dataset": args.dataset,
        "num_queries": len(questions),
        "indexes": {},
    }

    # ====== Flat (精确搜索，作为 upper bound 和 Recall 基准) ======
    flat_indices = None  # 初始化，用于后续 ANN recall 计算

    flat_path = os.path.join(args.index_dir, "flat.index")
    if os.path.exists(flat_path):
        logger.info("=" * 50)
        logger.info("Evaluating Flat index...")
        flat_index = faiss.read_index(flat_path)

        flat_scores, flat_indices, flat_latency, flat_qps = search_with_timing(
            flat_index, query_embs, max_k, single_thread=True,
        )

        flat_pids = [[passage_ids[idx] for idx in row if idx >= 0] for row in flat_indices]

        top_k_acc = compute_top_k_accuracy(flat_pids, answers_list, corpus, k_values)
        logger.info("Flat Top-K Accuracy: %s", top_k_acc)

        results["indexes"]["flat"] = {
            "top_k_accuracy": {str(k): v for k, v in top_k_acc.items()},
            "latency_ms": flat_latency,
            "qps": flat_qps,
        }
        del flat_index
    else:
        logger.warning("Flat index not found at %s; Recall@K will not be computed.", flat_path)

    # ====== HNSW 参数扫描 ======
    hnsw_files = sorted([f for f in os.listdir(args.index_dir) if f.startswith("hnsw") and f.endswith(".index")])
    if hnsw_files:
        hnsw_path = os.path.join(args.index_dir, hnsw_files[0])
        logger.info("=" * 50)
        logger.info("Evaluating HNSW index: %s", hnsw_path)
        hnsw_index = faiss.read_index(hnsw_path)

        hnsw_results = {}
        for ef in hnsw_ef_values:
            hnsw_index.hnsw.efSearch = ef
            scores, indices, latency, qps = search_with_timing(hnsw_index, query_embs, max_k)

            pids = [[passage_ids[idx] for idx in row if idx >= 0] for row in indices]
            top_k_acc = compute_top_k_accuracy(pids, answers_list, corpus, k_values)

            recall = {}
            if flat_indices is not None:
                recall = compute_recall_at_k(indices, flat_indices, [20, 100])

            avg_ndis = get_hnsw_distance_count(hnsw_index, query_embs, max_k, ef)

            entry = {
                "top_k_accuracy": {str(k): v for k, v in top_k_acc.items()},
                "latency_ms": latency,
                "qps": qps,
                "avg_distance_computations": avg_ndis,
            }
            if recall:
                entry["recall"] = {str(k): v for k, v in recall.items()}

            hnsw_results["ef_%d" % ef] = entry
            logger.info("  efSearch=%d: Top-100=%.4f, Recall@100=%.4f, Latency=%.2fms, QPS=%.0f, NDis=%.0f",
                        ef, top_k_acc.get(100, 0), recall.get(100, 0), latency, qps, avg_ndis)

        results["indexes"]["hnsw"] = hnsw_results
        del hnsw_index

    # ====== IVF / IVF-PQ 参数扫描 ======
    for prefix, idx_name in [("ivf_nlist", "ivf"), ("ivf_pq_nlist", "ivf_pq")]:
        idx_files = sorted([f for f in os.listdir(args.index_dir) if f.startswith(prefix) and f.endswith(".index")])
        if not idx_files:
            continue

        idx_path = os.path.join(args.index_dir, idx_files[0])
        logger.info("=" * 50)
        logger.info("Evaluating %s index: %s", idx_name, idx_path)
        ivf_index = faiss.read_index(idx_path)

        ivf_results = {}
        for nprobe in ivf_nprobe_values:
            ivf_index.nprobe = nprobe
            scores, indices, latency, qps = search_with_timing(ivf_index, query_embs, max_k)

            pids = [[passage_ids[idx] for idx in row if idx >= 0] for row in indices]
            top_k_acc = compute_top_k_accuracy(pids, answers_list, corpus, k_values)

            recall = {}
            if flat_indices is not None:
                recall = compute_recall_at_k(indices, flat_indices, [20, 100])

            entry = {
                "top_k_accuracy": {str(k): v for k, v in top_k_acc.items()},
                "latency_ms": latency,
                "qps": qps,
            }
            if recall:
                entry["recall"] = {str(k): v for k, v in recall.items()}

            ivf_results["nprobe_%d" % nprobe] = entry
            logger.info("  nprobe=%d: Top-100=%.4f, Recall@100=%.4f, Latency=%.2fms, QPS=%.0f",
                        nprobe, top_k_acc.get(100, 0), recall.get(100, 0), latency, qps)

        results["indexes"][idx_name] = ivf_results
        del ivf_index

    # 保存结果
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", args.output_path)

    # 打印摘要
    logger.info("=" * 50)
    logger.info("Evaluation summary for %s on %s:", args.model_type, args.dataset)
    if "flat" in results["indexes"]:
        flat_res = results["indexes"]["flat"]
        logger.info("  Flat: Top-100=%.4f, Latency=%.2fms",
                    flat_res["top_k_accuracy"].get("100", 0), flat_res["latency_ms"])


if __name__ == "__main__":
    main()
