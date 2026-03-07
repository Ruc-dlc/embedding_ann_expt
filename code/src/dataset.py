"""
合并训练数据集

加载 NQ train + TriviaQA train，合并 shuffle 为单一训练集。
加载 NQ dev / TriviaQA dev 作为验证集。

数据格式: DPR JSON
每条样本返回 (query, positive_text, hard_negative_text)
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _passage_text(ctx: Dict) -> str:
    """拼接 title + text 为 passage 文本"""
    title = ctx.get("title", "").strip()
    text = ctx.get("text", "").strip()
    if title:
        return f"{title} {text}"
    return text


class DPRDataset(Dataset):
    """
    DPR 格式数据集

    从 JSON 文件加载，每条样本返回:
        query:         str
        positive:      str  (title + text)
        hard_negative: str  (title + text)

    过滤掉没有正例或没有负例的样本。
    """

    def __init__(self, json_path: str, max_samples: Optional[int] = None):
        self.samples: List[Dict] = []
        self._load(json_path, max_samples)

    def _load(self, json_path: str, max_samples: Optional[int]) -> None:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {json_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        skipped = 0
        for item in raw:
            pos_list = item.get("positive_ctxs", [])
            # 负例: 优先 hard_negative_ctxs，其次 negative_ctxs
            hard_list = item.get("hard_negative_ctxs", [])
            neg_list = item.get("negative_ctxs", [])

            if not pos_list or (not hard_list and not neg_list):
                skipped += 1
                continue

            pos_ctx = pos_list[0]
            neg_ctx = hard_list[0] if hard_list else neg_list[0]

            self.samples.append({
                "query": item["question"],
                "positive": _passage_text(pos_ctx),
                "hard_negative": _passage_text(neg_ctx),
            })

            if max_samples and len(self.samples) >= max_samples:
                break

        logger.info(f"加载 {path.name}: {len(self.samples)} 条有效样本, 跳过 {skipped} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MergedTrainDataset(Dataset):
    """
    合并 NQ + TriviaQA 训练集并 shuffle
    """

    def __init__(self, data_dir: str, seed: int = 42, max_samples: Optional[int] = None):
        nq_path = f"{data_dir}/NQ/nq-train.json"
        trivia_path = f"{data_dir}/TriviaQA/trivia-train.json"

        nq = DPRDataset(nq_path, max_samples=max_samples)
        trivia = DPRDataset(trivia_path, max_samples=max_samples)

        self.samples = nq.samples + trivia.samples
        random.Random(seed).shuffle(self.samples)

        logger.info(f"合并训练集: NQ={len(nq)}, TriviaQA={len(trivia)}, 总计={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
