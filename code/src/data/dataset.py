"""
DPR 格式训练数据集与 Collator

支持三阶段训练所需的数据加载：
  - Stage 1 (in_batch): 每个样本只取 query + positive，负例由 batch 内其他 query 的 positive 构成
  - Stage 2/3 (hard_neg): 每个样本取 query + positive + 7 个 hard negatives

数据格式遵循 DPR 标准 JSON（question, positive_ctxs, hard_negative_ctxs 等）。

参考：
  - DPR/dpr/data/biencoder_data.py
  - experiments.md 第四节、第九节
"""

import json
import logging
import random
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.utils.data_utils import normalize_question, normalize_passage

logger = logging.getLogger(__name__)


class DPRDataset(Dataset):
    """DPR 格式训练数据集。

    加载 DPR 标准 JSON 文件，返回原始文本数据。
    tokenization 和 tensor 构建由 BiEncoderCollator 完成。

    Args:
        data_path: JSON 文件路径
        num_hard_negatives: 每个样本采样的 hard negative 数量（Stage 2/3 使用）
        stage: 训练阶段 ("in_batch" 或 "hard_neg")
        shuffle_positives: 是否随机选择 positive（多个 positive 时）
    """

    def __init__(
        self,
        data_path: str,
        num_hard_negatives: int = 7,
        stage: str = "in_batch",
        shuffle_positives: bool = False,
    ):
        self.data_path = data_path
        self.num_hard_negatives = num_hard_negatives
        self.stage = stage
        self.shuffle_positives = shuffle_positives

        self.data = self._load_data(data_path)
        logger.info(
            "Loaded %d samples from %s (stage=%s)",
            len(self.data), data_path, stage,
        )

    def _load_data(self, path: str) -> List[Dict]:
        """加载 JSON 数据，过滤 positive_ctxs 为空的样本。"""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 过滤无 positive 的样本（与 DPR JsonQADataset 一致）
        original_count = len(raw_data)
        data = [r for r in raw_data if len(r.get("positive_ctxs", [])) > 0]
        filtered = original_count - len(data)
        if filtered > 0:
            logger.warning(
                "Filtered %d samples with empty positive_ctxs (%.1f%%)",
                filtered, 100.0 * filtered / original_count,
            )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        """返回一个样本的原始文本数据。

        Returns:
            dict with keys:
                - question (str): 预处理后的 query 文本
                - positive (dict): {"title": str, "text": str}
                - hard_negatives (list of dict): [{"title": str, "text": str}, ...]
                    Stage "in_batch" 时返回空列表
        """
        sample = self.data[index]

        # Query 预处理
        question = normalize_question(sample["question"])

        # 选择 positive
        positive_ctxs = sample["positive_ctxs"]
        if self.shuffle_positives and len(positive_ctxs) > 1:
            pos = random.choice(positive_ctxs)
        else:
            pos = positive_ctxs[0]

        positive = {
            "title": pos.get("title", ""),
            "text": normalize_passage(pos.get("text", "")),
        }

        # Hard negatives
        hard_negatives = []
        if self.stage == "hard_neg":
            hn_ctxs = sample.get("hard_negative_ctxs", [])
            # 随机采样 num_hard_negatives 个（若不足，后续在 collator 中补齐）
            if len(hn_ctxs) > self.num_hard_negatives:
                hn_ctxs = random.sample(hn_ctxs, self.num_hard_negatives)
            for ctx in hn_ctxs:
                hard_negatives.append({
                    "title": ctx.get("title", ""),
                    "text": normalize_passage(ctx.get("text", "")),
                })

        return {
            "question": question,
            "positive": positive,
            "hard_negatives": hard_negatives,
        }


class BiEncoderCollator:
    """Batch 构建器：tokenize + ghost negative 补齐 + tensor 打包。

    Stage "in_batch":
        返回 query tensors [B, seq_len] + positive tensors [B, seq_len]
        负例由训练循环中 batch 内其他 query 的 positive 构成（通过相似度矩阵 B×B）

    Stage "hard_neg":
        返回 query tensors [B, seq_len] + context tensors [B*8, seq_len]
        每个 query 对应 1 positive + 7 hard negatives（不足 7 个时从 batch 内其他 positive 补齐）
        positive 在每个 query 的 context 块中的位置固定为 index 0

    Args:
        tokenizer: HuggingFace tokenizer 实例
        max_query_length: query 最大 token 数
        max_passage_length: passage 最大 token 数
        stage: "in_batch" 或 "hard_neg"
        num_hard_negatives: hard negative 目标数量（默认 7）
    """

    def __init__(
        self,
        tokenizer,
        max_query_length: int = 256,
        max_passage_length: int = 256,
        stage: str = "in_batch",
        num_hard_negatives: int = 7,
    ):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length
        self.stage = stage
        self.num_hard_negatives = num_hard_negatives
        self._ghost_neg_count = 0  # 记录补齐次数

    @property
    def ghost_neg_count(self) -> int:
        return self._ghost_neg_count

    def reset_ghost_neg_count(self):
        self._ghost_neg_count = 0

    def _tokenize_passage(self, title: str, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize passage: [CLS] title [SEP] text [SEP]

        使用 tokenizer 的 text_pair 功能实现 title + text 拼接。
        """
        return self.tokenizer(
            title,
            text_pair=text,
            max_length=self.max_passage_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

    def _tokenize_query(self, question: str) -> Dict[str, torch.Tensor]:
        """Tokenize query: [CLS] question [SEP]"""
        return self.tokenizer(
            question,
            max_length=self.max_query_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        if self.stage == "in_batch":
            return self._collate_in_batch(samples)
        else:
            return self._collate_hard_neg(samples)

    def _collate_in_batch(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Stage 1: In-batch negatives collate.

        Returns:
            dict with:
                - query_input_ids: [B, max_q_len]
                - query_attention_mask: [B, max_q_len]
                - ctx_input_ids: [B, max_p_len]  (只有 positive)
                - ctx_attention_mask: [B, max_p_len]
        """
        query_encodings = []
        ctx_encodings = []

        for s in samples:
            query_encodings.append(self._tokenize_query(s["question"]))
            ctx_encodings.append(
                self._tokenize_passage(s["positive"]["title"], s["positive"]["text"])
            )

        query_batch = self.tokenizer.pad(
            query_encodings,
            padding=True,
            return_tensors="pt",
        )
        ctx_batch = self.tokenizer.pad(
            ctx_encodings,
            padding=True,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "ctx_input_ids": ctx_batch["input_ids"],
            "ctx_attention_mask": ctx_batch["attention_mask"],
        }

    def _collate_hard_neg(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Stage 2/3: Hard negatives collate (含 ghost negative 补齐).

        每个 query 对应 1 positive + num_hard_negatives 个 hard negatives。
        不足时从 batch 内其他 query 的 positive 采样补齐（ghost negatives）。
        补齐时排除当前 query 自身的 positive，避免假负例。

        Returns:
            dict with:
                - query_input_ids: [B, max_q_len]
                - query_attention_mask: [B, max_q_len]
                - ctx_input_ids: [B * (1 + num_hard_negatives), max_p_len]
                - ctx_attention_mask: [B * (1 + num_hard_negatives), max_p_len]
                - positive_indices: [B]  每个 query 的 positive 在 ctx 中的绝对索引
        """
        batch_size = len(samples)
        n_ctx_per_query = 1 + self.num_hard_negatives  # 1 positive + 7 negatives = 8

        # 收集所有 positive 文本（用于 ghost negative 补齐）
        all_positives = [
            (s["positive"]["title"], s["positive"]["text"]) for s in samples
        ]

        query_encodings = []
        ctx_encodings = []
        positive_indices = []

        for i, s in enumerate(samples):
            # Query
            query_encodings.append(self._tokenize_query(s["question"]))

            # Positive（放在每个 query 的 ctx 块的第一个位置）
            ctx_start = i * n_ctx_per_query
            positive_indices.append(ctx_start)
            ctx_encodings.append(
                self._tokenize_passage(s["positive"]["title"], s["positive"]["text"])
            )

            # Hard negatives
            hard_negs = s["hard_negatives"]

            # Ghost negative 补齐
            if len(hard_negs) < self.num_hard_negatives:
                deficit = self.num_hard_negatives - len(hard_negs)
                self._ghost_neg_count += 1

                # 候选：batch 内其他 query 的 positive
                ghost_candidates = [
                    all_positives[j] for j in range(batch_size) if j != i
                ]
                # 采样补齐
                if len(ghost_candidates) >= deficit:
                    ghosts = random.sample(ghost_candidates, deficit)
                else:
                    # 极端情况：batch 太小，循环采样
                    ghosts = []
                    if len(ghost_candidates) > 0:
                        while len(ghosts) < deficit:
                            ghosts.extend(ghost_candidates)
                        ghosts = ghosts[:deficit]
                    else:
                        # batch_size=1 且无其他 query 的 positive 可用，
                        # 复制当前 query 的 positive 作为 ghost negative
                        ghosts = [all_positives[i]] * deficit

                for title, text in ghosts:
                    hard_negs.append({"title": title, "text": text})

            # Tokenize hard negatives
            for hn in hard_negs[:self.num_hard_negatives]:
                ctx_encodings.append(
                    self._tokenize_passage(hn["title"], hn["text"])
                )

        query_batch = self.tokenizer.pad(
            query_encodings,
            padding=True,
            return_tensors="pt",
        )
        ctx_batch = self.tokenizer.pad(
            ctx_encodings,
            padding=True,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "ctx_input_ids": ctx_batch["input_ids"],
            "ctx_attention_mask": ctx_batch["attention_mask"],
            "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
        }
