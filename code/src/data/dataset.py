"""
数据集类定义（NQ、TriviaQA）

本模块实现DPR格式数据集的PyTorch Dataset封装：
- Natural Questions (NQ): Google开放域问答数据集
- TriviaQA: 大规模开放域问答数据集

数据格式为DPR标准JSON，包含 question, answers, positive_ctxs, hard_negative_ctxs 等字段。

论文章节：第5章 5.1节 - 实验设置
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    DPR格式数据集基类

    负责加载DPR标准JSON文件，兼容NQ和TriviaQA两种字段命名方式。
    自动过滤 positive_ctxs 为空的记录。

    参数:
        data_path: 数据文件路径（JSON格式）
        num_hard_negatives: 每条样本使用的难负例数量
        max_samples: 最大加载样本数（None表示全部加载，调试时可设小值）
    """

    def __init__(
        self,
        data_path: str,
        num_hard_negatives: int = 7,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.num_hard_negatives = num_hard_negatives
        self.max_samples = max_samples
        self.samples: List[Dict] = []

    def _load_dpr_json(self) -> None:
        """
        加载DPR格式的JSON数据文件

        自动兼容 passage_id / psg_id 两种字段名，
        过滤 positive_ctxs 为空的记录。
        """
        logger.info(f"正在加载数据文件: {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        logger.info(f"原始记录数: {len(raw_data)}")

        # 过滤并规范化
        valid_count = 0
        skipped_count = 0

        for item in raw_data:
            # 过滤 positive_ctxs 为空的记录
            positive_ctxs = item.get('positive_ctxs', [])
            if not positive_ctxs:
                skipped_count += 1
                continue

            # 获取难负例
            hard_negatives = item.get('hard_negative_ctxs', [])

            # 规范化段落ID字段名（兼容 passage_id 和 psg_id）
            normalized_positives = []
            for ctx in positive_ctxs:
                normalized_positives.append({
                    'text': ctx.get('text', ''),
                    'title': ctx.get('title', ''),
                    'passage_id': ctx.get('passage_id', ctx.get('psg_id', '')),
                })

            normalized_hard_negs = []
            for ctx in hard_negatives:
                normalized_hard_negs.append({
                    'text': ctx.get('text', ''),
                    'title': ctx.get('title', ''),
                    'passage_id': ctx.get('passage_id', ctx.get('psg_id', '')),
                })

            sample = {
                'question': item.get('question', ''),
                'answers': item.get('answers', []),
                'positive_ctxs': normalized_positives,
                'hard_negative_ctxs': normalized_hard_negs,
            }

            self.samples.append(sample)
            valid_count += 1

            # 调试模式：限制加载数量
            if self.max_samples is not None and valid_count >= self.max_samples:
                break

        logger.info(
            f"加载完成: 有效样本 {valid_count}, "
            f"跳过(无正例) {skipped_count}"
        )

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本

        返回:
            字典包含:
            - question: 查询文本
            - answers: 答案列表
            - positive_text: 正例段落文本
            - positive_title: 正例段落标题
            - hard_negatives: 难负例列表 [{text, title}]
        """
        sample = self.samples[idx]

        # 随机选择一个正例
        pos_ctx = random.choice(sample['positive_ctxs'])

        # 选取难负例（随机采样指定数量）
        hard_negs = sample['hard_negative_ctxs']
        if len(hard_negs) > self.num_hard_negatives:
            hard_negs = random.sample(hard_negs, self.num_hard_negatives)

        hard_neg_items = [
            {'text': neg['text'], 'title': neg['title']}
            for neg in hard_negs
        ]

        return {
            'question': sample['question'],
            'answers': sample['answers'],
            'positive_text': pos_ctx['text'],
            'positive_title': pos_ctx['title'],
            'hard_negatives': hard_neg_items,
        }


class NQDataset(BaseDataset):
    """
    Natural Questions 数据集

    Google开放域问答数据集，包含来自真实用户搜索的问题和Wikipedia答案段落。

    参数:
        data_path: NQ数据文件路径（如 data_set/NQ/nq-train.json）
        num_hard_negatives: 每条样本使用的难负例数量
        max_samples: 最大加载样本数
    """

    def __init__(
        self,
        data_path: str,
        num_hard_negatives: int = 7,
        max_samples: Optional[int] = None
    ):
        super().__init__(data_path, num_hard_negatives, max_samples)
        self._load_dpr_json()
        logger.info(f"NQ数据集初始化完成，共 {len(self.samples)} 条样本")


class TriviaQADataset(BaseDataset):
    """
    TriviaQA 数据集

    大规模开放域问答数据集，问题来自trivia爱好者编写。

    参数:
        data_path: TriviaQA数据文件路径（如 data_set/TriviaQA/trivia-train.json）
        num_hard_negatives: 每条样本使用的难负例数量
        max_samples: 最大加载样本数
    """

    def __init__(
        self,
        data_path: str,
        num_hard_negatives: int = 7,
        max_samples: Optional[int] = None
    ):
        super().__init__(data_path, num_hard_negatives, max_samples)
        self._load_dpr_json()
        logger.info(f"TriviaQA数据集初始化完成，共 {len(self.samples)} 条样本")


class ConcatRetrievalDataset(Dataset):
    """
    合并检索数据集

    将NQ和TriviaQA两个数据集合并为一个统一数据集，
    用于联合训练场景。

    参数:
        datasets: 数据集列表
    """

    def __init__(self, datasets: List[BaseDataset]):
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes: List[int] = []

        cumsum = 0
        for ds in datasets:
            cumsum += len(ds)
            self.cumulative_sizes.append(cumsum)

        logger.info(
            f"合并数据集初始化完成，共 {len(datasets)} 个子数据集，"
            f"总样本数 {self.cumulative_sizes[-1]}"
        )

    def __len__(self) -> int:
        """返回总样本数"""
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """根据全局索引定位到对应子数据集并获取样本"""
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                local_idx = idx - (self.cumulative_sizes[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        raise IndexError(f"索引 {idx} 超出范围 [0, {len(self)})")