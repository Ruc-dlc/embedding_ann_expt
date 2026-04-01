"""
语料库编码脚本

对全量 psgs_w100.tsv (~21M passages) 编码，支持多种 encoder：
  - dacl-dr: 我们自己训练好的双塔checkpoint BiEncoder (doc_encoder)
  - dpr: facebook/dpr-ctx_encoder-single-nq-base 注意是官方在nq上fine-tuning过的模型checkpoint
  - ance: castorini/ance-dpr-context-multi
  - contriever: facebook/contriever 或 facebook/contriever-msmarco

所有模型编码后统一 L2 归一化，存储为 float16 以节省空间。

优化策略：
  使用 DataLoader + num_workers 实现 CPU tokenization 与 GPU inference 流水线并行，加快编码速度，单个模型大约需要4.5小时推理编码完成

用法：
  # DACL-DR
  python encode_passages.py --model_type dacl-dr \
    --model_path ./checkpoints/nq/best_model_nq --output_dir ./embeddings/dacl-dr

  # DPR
  python encode_passages.py --model_type dpr \
    --model_path ./dpr-backbone --output_dir ./embeddings/dpr

  # ANCE
  python encode_passages.py --model_type ance \
    --model_path ./ance-backbone --output_dir ./embeddings/ance

  # Contriever
  python encode_passages.py --model_type contriever \
    --model_path ./contriever-backbone --output_dir ./embeddings/contriever

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DPRContextEncoder

from src.utils.data_utils import normalize_passage

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Encode passages")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["dacl-dr", "dpr", "ance", "contriever"])
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（dacl-dr 为 BiEncoder 保存目录，其他为 HuggingFace 模型目录）")
    parser.add_argument("--corpus_path", type=str, default="./data_set/psgs_w100.tsv")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--save_float16", action="store_true", default=True,
                        help="以 float16 存储向量（节省约 50% 空间）")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 数量，用于并行 tokenize（默认 4）")
    return parser.parse_args()


def load_corpus(corpus_path):
    """加载语料库，返回 (passage_ids, titles, texts)。"""
    logger.info("Loading corpus from %s ...", corpus_path)
    passage_ids = []
    titles = []
    texts = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
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


def load_encoder(model_type, model_path, device):
    """根据模型类型加载 encoder 和 tokenizer。

    Returns:
        (encode_fn, tokenizer)
        encode_fn: callable(input_ids, attention_mask) -> embeddings [B, D]
    """
    if model_type == "dacl-dr":
        from src.models import BiEncoder
        model = BiEncoder.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)

        def encode_fn(input_ids, attention_mask):
            return model.encode_document(input_ids, attention_mask)

        return encode_fn, tokenizer

    elif model_type == "dpr":
        # DPR context encoder
        model = DPRContextEncoder.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def encode_fn(input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.pooler_output  # [B, 768], 未归一化

        return encode_fn, tokenizer

    elif model_type == "ance":
        # ANCE uses same architecture as DPR (DPRContextEncoder compatible)
        # castorini/ance-dpr-context-multi 可用 DPRContextEncoder 加载
        model = DPRContextEncoder.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def encode_fn(input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.pooler_output

        return encode_fn, tokenizer

    elif model_type == "contriever":
        # Contriever 使用 mean pooling
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def encode_fn(input_ids, attention_mask):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling（Contriever 标准做法）
            token_embs = outputs.last_hidden_state  # [B, L, D]
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
            sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embs / sum_mask  # [B, D]

        return encode_fn, tokenizer

    else:
        raise ValueError("Unknown model_type: %s" % model_type)


class PassageDataset(Dataset):
    def __init__(self, titles, texts, tokenizer, max_length):
        self.titles = titles
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.titles[idx],
            text_pair=self.texts[idx],
            max_length=self.max_length,
            truncation=True,
        )
        # 返回 Python list（高效 pickle，避免 tensor 序列化开销）
        return encoding["input_ids"], encoding["attention_mask"]


def collate_passages(batch):
    """将单条 tokenize 结果 pad 到 batch 内最长长度并转为 tensor。"""
    input_ids_list, attn_mask_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)
    bsz = len(batch)

    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    attention_mask = torch.zeros(bsz, max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_mask_list)):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :length] = torch.tensor(mask, dtype=torch.long)

    return input_ids, attention_mask


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 加载编码器
    encode_fn, tokenizer = load_encoder(args.model_type, args.model_path, device)
    logger.info("Encoder loaded: type=%s, path=%s", args.model_type, args.model_path)

    # 加载语料库
    passage_ids, titles, texts = load_corpus(args.corpus_path)

    # 编码
    n = len(texts)

    # 构建 DataLoader（多 worker 异步 tokenize，与 GPU inference 流水线并行）
    dataset = PassageDataset(titles, texts, tokenizer, args.max_passage_length)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_passages,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    all_embs = []
    t_start = time.time()

    logger.info("Encoding %d passages (batch_size=%d, num_workers=%d)...",
                n, args.batch_size, args.num_workers)

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Encoding"):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            if args.fp16:
                with torch.cuda.amp.autocast():
                    emb = encode_fn(input_ids, attention_mask)
            else:
                emb = encode_fn(input_ids, attention_mask)

            # L2 归一化：DACL-DR 模型内部已归一化，baseline 需外部统一归一化，避免后续评估时，向量携带模长信息，造成评估不准确
            if args.model_type != "dacl-dr":
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

            all_embs.append(emb.cpu().numpy())

    elapsed = time.time() - t_start
    all_embs = np.concatenate(all_embs, axis=0)  # [N, D]
    logger.info("Encoding complete: %s, %.1f seconds (%.0f passages/sec)",
                all_embs.shape, elapsed, n / elapsed)

    # 保存
    if args.save_float16:
        all_embs = all_embs.astype(np.float16)

    emb_path = os.path.join(args.output_dir, "passage_embeddings.npy")
    np.save(emb_path, all_embs)
    logger.info("Saved embeddings to %s (dtype=%s, size=%.1f GB)",
                emb_path, all_embs.dtype, all_embs.nbytes / 1e9)

    # 保存 passage_ids 映射
    ids_path = os.path.join(args.output_dir, "passage_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(passage_ids, f)
    logger.info("Saved passage IDs to %s", ids_path)


if __name__ == "__main__":
    main()

