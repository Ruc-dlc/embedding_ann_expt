"""
Embedding 空间分析与可视化脚本

功能：
  1. Embedding 空间统计指标（experiments.md 第 7.3 节）
     - Positive Cosine Mean / Var
     - Negative Cosine Mean / Var
     - Alignment: E[||q - d+||^2]
     - Uniformity: log E[exp(-2||x - y||^2)]

  2. t-SNE 可视化（experiments.md 第 8.1 节）
     - 从 dev 集采样 query-positive 对
     - t-SNE 降维至 2D
     - 输出散点图（query vs doc 不同颜色/标记，正样本对连线）

  3. 正样本对余弦相似度分布直方图（experiments.md 第 8.4 节）

用法：
  # 计算统计指标
  python analyze_embeddings.py stats \
    --model_type dacl-dr --model_path ./checkpoints/nq/best_model_nq \
    --dataset nq --data_dir ./data_set \
    --output_path ./results/embedding_stats_dacl-dr_nq.json

  # t-SNE 可视化
  python analyze_embeddings.py tsne \
    --model_type dacl-dr --model_path ./checkpoints/nq/best_model_nq \
    --dataset nq --data_dir ./data_set \
    --output_dir ./results/figures \
    --n_samples 500

  # 同时计算 w=0 baseline 的对比
  python analyze_embeddings.py tsne \
    --model_type dacl-dr --model_path ./checkpoints/nq/baseline_w0 \
    --dataset nq --data_dir ./data_set \
    --output_dir ./results/figures --label "w=0"

参考：
  - experiments.md 第 7.3 节、第 8.1 节、第 8.4 节
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils.data_utils import normalize_question, normalize_passage

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Embedding space analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ====== stats 子命令 ======
    stats_parser = subparsers.add_parser("stats", help="Compute embedding space statistics")
    stats_parser.add_argument("--model_type", type=str, required=True,
                              choices=["dacl-dr", "dpr", "ance", "contriever"])
    stats_parser.add_argument("--model_path", type=str, required=True)
    stats_parser.add_argument("--dataset", type=str, required=True, choices=["nq", "trivia"])
    stats_parser.add_argument("--data_dir", type=str, default="./data_set")
    stats_parser.add_argument("--output_path", type=str, required=True)
    stats_parser.add_argument("--max_length", type=int, default=256)
    stats_parser.add_argument("--batch_size", type=int, default=128)
    stats_parser.add_argument("--n_neg_samples", type=int, default=10000,
                              help="负样本对随机采样数量")
    stats_parser.add_argument("--fp16", action="store_true", default=True)
    stats_parser.add_argument("--seed", type=int, default=42)

    # ====== tsne 子命令 ======
    tsne_parser = subparsers.add_parser("tsne", help="t-SNE visualization")
    tsne_parser.add_argument("--model_type", type=str, required=True,
                             choices=["dacl-dr", "dpr", "ance", "contriever"])
    tsne_parser.add_argument("--model_path", type=str, required=True)
    tsne_parser.add_argument("--dataset", type=str, required=True, choices=["nq", "trivia"])
    tsne_parser.add_argument("--data_dir", type=str, default="./data_set")
    tsne_parser.add_argument("--output_dir", type=str, required=True)
    tsne_parser.add_argument("--max_length", type=int, default=256)
    tsne_parser.add_argument("--batch_size", type=int, default=128)
    tsne_parser.add_argument("--n_samples", type=int, default=500)
    tsne_parser.add_argument("--fp16", action="store_true", default=True)
    tsne_parser.add_argument("--seed", type=int, default=42)
    tsne_parser.add_argument("--label", type=str, default=None,
                             help="图表标签（例如 'w=0' 或 'w=0.4'）")
    tsne_parser.add_argument("--perplexity", type=float, default=30.0)

    return parser.parse_args()


def load_dev_data(dataset, data_dir):
    """加载 dev 集（DPR JSON 格式），返回 (questions, positive_titles, positive_texts)。

    每条样本取 positive_ctxs[0] 作为正例文档。
    """
    if dataset == "nq":
        path = os.path.join(data_dir, "NQ", "nq-dev.json")
    else:
        path = os.path.join(data_dir, "TriviaQA", "trivia-dev.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    pos_titles = []
    pos_texts = []

    for item in data:
        if not item.get("positive_ctxs"):
            continue
        q = normalize_question(item["question"])
        pos = item["positive_ctxs"][0]
        title = pos.get("title", "")
        text = normalize_passage(pos.get("text", ""))

        questions.append(q)
        pos_titles.append(title)
        pos_texts.append(text)

    logger.info("Loaded %d dev samples from %s", len(questions), path)
    return questions, pos_titles, pos_texts


def load_encoders(model_type, model_path, device):
    """加载 query encoder 和 document encoder。

    Returns:
        (query_fn, doc_fn, tokenizer)
        query_fn: (input_ids, attention_mask) -> [B, D]
        doc_fn: (input_ids, attention_mask) -> [B, D]
    """
    if model_type == "dacl-dr":
        from src.models import BiEncoder
        model = BiEncoder.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)

        def query_fn(input_ids, attention_mask):
            return model.encode_query(input_ids, attention_mask)

        def doc_fn(input_ids, attention_mask):
            return model.encode_document(input_ids, attention_mask)

        return query_fn, doc_fn, tokenizer

    elif model_type in ("dpr", "ance"):
        from transformers import DPRQuestionEncoder, DPRContextEncoder

        if model_type == "dpr":
            q_path = model_path
            c_path = model_path.replace("question_encoder", "ctx_encoder").replace("question", "context")
        else:
            q_path = model_path.replace("context", "question")
            c_path = model_path

        q_model = DPRQuestionEncoder.from_pretrained(q_path)
        q_model.to(device)
        q_model.eval()

        c_model = DPRContextEncoder.from_pretrained(c_path)
        c_model.to(device)
        c_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(q_path)

        def query_fn(input_ids, attention_mask):
            return q_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        def doc_fn(input_ids, attention_mask):
            return c_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        return query_fn, doc_fn, tokenizer

    elif model_type == "contriever":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def mean_pool(input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embs = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
            sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embs / sum_mask

        return mean_pool, mean_pool, tokenizer

    else:
        raise ValueError("Unknown model_type: %s" % model_type)


@torch.no_grad()
def encode_pairs(query_fn, doc_fn, tokenizer, questions, pos_titles, pos_texts,
                 max_length, batch_size, fp16, device):
    """编码 query-document 对，返回 L2 归一化后的 embeddings。

    Returns:
        query_embs: [N, D] numpy float32
        doc_embs: [N, D] numpy float32
    """
    n = len(questions)
    q_embs_list = []
    d_embs_list = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_q = questions[start:end]
        batch_titles = pos_titles[start:end]
        batch_texts = pos_texts[start:end]

        # 编码 queries
        q_enc = tokenizer(batch_q, max_length=max_length, truncation=True,
                          padding=True, return_tensors="pt")
        q_ids = q_enc["input_ids"].to(device)
        q_mask = q_enc["attention_mask"].to(device)

        # 编码 documents: [CLS] title [SEP] text [SEP]
        d_enc = tokenizer(batch_titles, text_pair=batch_texts,
                          max_length=max_length, truncation=True,
                          padding=True, return_tensors="pt")
        d_ids = d_enc["input_ids"].to(device)
        d_mask = d_enc["attention_mask"].to(device)

        if fp16:
            with torch.cuda.amp.autocast():
                q_emb = query_fn(q_ids, q_mask)
                d_emb = doc_fn(d_ids, d_mask)
        else:
            q_emb = query_fn(q_ids, q_mask)
            d_emb = doc_fn(d_ids, d_mask)

        # L2 归一化
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
        d_emb = torch.nn.functional.normalize(d_emb, p=2, dim=-1)

        q_embs_list.append(q_emb.cpu().numpy())
        d_embs_list.append(d_emb.cpu().numpy())

        if (start // batch_size) % 10 == 0:
            logger.info("  Encoded %d / %d pairs", end, n)

    return np.concatenate(q_embs_list, axis=0), np.concatenate(d_embs_list, axis=0)


def compute_alignment(query_embs, doc_embs):
    """Alignment = E[||q - d+||^2]

    对 L2 归一化向量: ||q - d+||^2 = 2 - 2 * cos(q, d+) = 2(1 - q·d+)
    """
    diff = query_embs - doc_embs
    sq_dist = np.sum(diff ** 2, axis=1)
    return float(np.mean(sq_dist))


def compute_uniformity(embs, n_pairs=50000, seed=42):
    """Uniformity = log E[exp(-2||x - y||^2)]

    从 embs 中随机采样 n_pairs 对计算。
    """
    rng = np.random.RandomState(seed)
    n = embs.shape[0]
    if n < 2:
        return 0.0

    # 随机采样不重复的 pairs
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    # 确保 a != b
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % n

    diff = embs[idx_a] - embs[idx_b]
    sq_dist = np.sum(diff ** 2, axis=1)  # [n_pairs]

    # log E[exp(-2 * ||x-y||^2)]
    # 使用 log-mean-exp 技巧避免数值溢出
    vals = -2.0 * sq_dist
    max_val = np.max(vals)
    uniformity = max_val + np.log(np.mean(np.exp(vals - max_val)))

    return float(uniformity)


def compute_stats(args):
    """计算 embedding 空间统计指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载 dev 数据
    questions, pos_titles, pos_texts = load_dev_data(args.dataset, args.data_dir)

    # 加载 encoders
    query_fn, doc_fn, tokenizer = load_encoders(args.model_type, args.model_path, device)

    # 编码所有 query-positive 对
    logger.info("Encoding %d query-positive pairs...", len(questions))
    query_embs, doc_embs = encode_pairs(
        query_fn, doc_fn, tokenizer, questions, pos_titles, pos_texts,
        args.max_length, args.batch_size, args.fp16, device,
    )
    logger.info("Embeddings shape: query=%s, doc=%s", query_embs.shape, doc_embs.shape)

    # 正样本对余弦相似度 (L2 归一化后, cos = dot product)
    pos_cos = np.sum(query_embs * doc_embs, axis=1)
    pos_cos_mean = float(np.mean(pos_cos))
    pos_cos_var = float(np.var(pos_cos))

    # 负样本对余弦相似度（随机配对 query 和 不匹配的 doc）
    rng = np.random.RandomState(args.seed)
    n = len(questions)
    n_neg = min(args.n_neg_samples, n * (n - 1))

    neg_q_idx = rng.randint(0, n, size=n_neg)
    neg_d_idx = rng.randint(0, n, size=n_neg)
    # 确保不配对到自身的 positive
    mask = neg_q_idx == neg_d_idx
    neg_d_idx[mask] = (neg_d_idx[mask] + 1) % n

    neg_cos = np.sum(query_embs[neg_q_idx] * doc_embs[neg_d_idx], axis=1)
    neg_cos_mean = float(np.mean(neg_cos))
    neg_cos_var = float(np.var(neg_cos))

    # Alignment
    alignment = compute_alignment(query_embs, doc_embs)

    # Uniformity (混合 query + doc embeddings)
    all_embs = np.concatenate([query_embs, doc_embs], axis=0)
    uniformity = compute_uniformity(all_embs, seed=args.seed)

    results = {
        "model_type": args.model_type,
        "dataset": args.dataset,
        "num_samples": n,
        "positive_cosine_mean": pos_cos_mean,
        "positive_cosine_var": pos_cos_var,
        "negative_cosine_mean": neg_cos_mean,
        "negative_cosine_var": neg_cos_var,
        "alignment": alignment,
        "uniformity": uniformity,
        "positive_cosine_distribution": {
            "min": float(np.min(pos_cos)),
            "max": float(np.max(pos_cos)),
            "p25": float(np.percentile(pos_cos, 25)),
            "p50": float(np.percentile(pos_cos, 50)),
            "p75": float(np.percentile(pos_cos, 75)),
        },
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Embedding stats results:")
    logger.info("  Positive Cosine: mean=%.4f, var=%.6f", pos_cos_mean, pos_cos_var)
    logger.info("  Negative Cosine: mean=%.4f, var=%.6f", neg_cos_mean, neg_cos_var)
    logger.info("  Alignment: %.4f", alignment)
    logger.info("  Uniformity: %.4f", uniformity)
    logger.info("Results saved to %s", args.output_path)


def run_tsne(args):
    """t-SNE 可视化。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载 dev 数据
    questions, pos_titles, pos_texts = load_dev_data(args.dataset, args.data_dir)

    # 随机采样 n_samples 对
    n = len(questions)
    if args.n_samples < n:
        indices = np.random.choice(n, size=args.n_samples, replace=False)
        questions = [questions[i] for i in indices]
        pos_titles = [pos_titles[i] for i in indices]
        pos_texts = [pos_texts[i] for i in indices]
    n_samples = len(questions)
    logger.info("Using %d query-positive pairs for t-SNE", n_samples)

    # 加载 encoders
    query_fn, doc_fn, tokenizer = load_encoders(args.model_type, args.model_path, device)

    # 编码
    query_embs, doc_embs = encode_pairs(
        query_fn, doc_fn, tokenizer, questions, pos_titles, pos_texts,
        args.max_length, args.batch_size, args.fp16, device,
    )

    # 合并后 t-SNE: [query_0, ..., query_n, doc_0, ..., doc_n]
    combined = np.concatenate([query_embs, doc_embs], axis=0)  # [2N, D]
    logger.info("Running t-SNE on %d vectors (dim=%d)...", combined.shape[0], combined.shape[1])

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        n_iter=1000,
    )
    coords = tsne.fit_transform(combined)  # [2N, 2]

    q_coords = coords[:n_samples]
    d_coords = coords[n_samples:]

    # 计算正样本对的 t-SNE 空间距离
    pair_dists = np.sqrt(np.sum((q_coords - d_coords) ** 2, axis=1))

    # 构建 label
    label = args.label if args.label else "%s (%s)" % (args.model_type, args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 图 1: t-SNE 散点图 ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.scatter(q_coords[:, 0], q_coords[:, 1],
               c="#4A90D9", marker="o", s=15, alpha=0.6, label="Query", zorder=3)
    ax.scatter(d_coords[:, 0], d_coords[:, 1],
               c="#E74C3C", marker="^", s=15, alpha=0.6, label="Document", zorder=3)

    # 画正样本对连线（使用灰色细线，避免遮挡）
    for i in range(n_samples):
        ax.plot([q_coords[i, 0], d_coords[i, 0]],
                [q_coords[i, 1], d_coords[i, 1]],
                color="#CCCCCC", linewidth=0.3, alpha=0.5, zorder=1)

    ax.set_title("t-SNE: %s" % label, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    safe_label = label.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
    fig_path = os.path.join(args.output_dir, "tsne_%s.png" % safe_label)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("t-SNE scatter saved to %s", fig_path)

    # --- 图 2: 正样本对距离分布直方图 ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    ax2.hist(pair_dists, bins=50, color="#4A90D9", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("t-SNE Pair Distance", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Positive Pair Distance Distribution: %s" % label, fontsize=13)
    ax2.axvline(np.mean(pair_dists), color="red", linestyle="--", linewidth=1.5,
                label="Mean=%.1f" % np.mean(pair_dists))
    ax2.legend(fontsize=11)

    hist_path = os.path.join(args.output_dir, "pair_dist_%s.png" % safe_label)
    fig2.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("Pair distance histogram saved to %s", hist_path)

    # --- 图 3: 正样本对余弦相似度分布 ---
    pos_cos = np.sum(query_embs * doc_embs, axis=1)
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    ax3.hist(pos_cos, bins=50, color="#2ECC71", alpha=0.7, edgecolor="white")
    ax3.set_xlabel("Cosine Similarity (q, d+)", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Positive Pair Cosine Distribution: %s" % label, fontsize=13)
    ax3.axvline(np.mean(pos_cos), color="red", linestyle="--", linewidth=1.5,
                label="Mean=%.4f" % np.mean(pos_cos))
    ax3.legend(fontsize=11)

    cos_path = os.path.join(args.output_dir, "pos_cosine_%s.png" % safe_label)
    fig3.savefig(cos_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    logger.info("Positive cosine histogram saved to %s", cos_path)

    # 保存数值数据（供后续对比绘图使用）
    data_path = os.path.join(args.output_dir, "tsne_data_%s.npz" % safe_label)
    np.savez(data_path,
             q_coords=q_coords, d_coords=d_coords,
             pair_dists=pair_dists, pos_cos=pos_cos)
    logger.info("t-SNE numeric data saved to %s", data_path)


def main():
    args = get_args()

    if args.command == "stats":
        compute_stats(args)
    elif args.command == "tsne":
        run_tsne(args)
    else:
        raise ValueError("Unknown command: %s" % args.command)


if __name__ == "__main__":
    main()
