"""
DPR 标准 Batch Collation

将 (query, positive, hard_negative) 样本列表组装为 DPR 标准 batch:
  - query_input_ids:       [B, max_q_len]
  - query_attention_mask:  [B, max_q_len]
  - passage_input_ids:     [2B, max_d_len]    前B个positive, 后B个hard_negative
  - passage_attention_mask:[2B, max_d_len]
  - batch_size:            B
"""

import torch
from typing import Dict, List, Any


class DPRCollator:
    """
    DPR 标准 collator

    遵循 temp.md #9-#12 的 batch 构造规范:
    - B 个 query
    - 前 B 个 passage 为 positive
    - 后 B 个 passage 为 hard_negative
    - 总 passage 数 = 2B
    """

    def __init__(self, tokenizer, max_query_len: int = 64, max_doc_len: int = 256):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [x["query"] for x in batch]
        positives = [x["positive"] for x in batch]
        hard_negatives = [x["hard_negative"] for x in batch]

        # Tokenize queries
        q_encoded = self.tokenizer(
            queries,
            max_length=self.max_query_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize passages: positives 在前, hard_negatives 在后
        passages = positives + hard_negatives
        p_encoded = self.tokenizer(
            passages,
            max_length=self.max_doc_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "query_input_ids": q_encoded["input_ids"],
            "query_attention_mask": q_encoded["attention_mask"],
            "passage_input_ids": p_encoded["input_ids"],
            "passage_attention_mask": p_encoded["attention_mask"],
            "batch_size": torch.tensor(len(queries)),
        }
