"""
三阶段训练数据加载器

本模块实现支持三阶段训练策略的数据加载器：
- 阶段1（预热）: 仅使用In-Batch负例，纯InfoNCE训练
- 阶段2（距离引入）: 使用DPR预挖掘的难负例 + In-Batch负例
- 阶段3（联合优化）: 使用动态难负例 + In-Batch负例

论文章节：第4章 4.3节 - 三阶段训练策略
"""

import logging
import random
from typing import Dict, List, Optional, Iterator, Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: Any,
    max_query_length: int = 64,
    max_doc_length: int = 256,
    include_hard_negatives: bool = False
) -> Dict[str, torch.Tensor]:
    """
    批次整理函数

    将原始文本样本tokenize并整理为模型输入张量。

    参数:
        batch: 样本列表（每个元素来自 Dataset.__getitem__）
        tokenizer: HuggingFace tokenizer
        max_query_length: 查询最大token长度
        max_doc_length: 文档最大token长度
        include_hard_negatives: 是否包含难负例

    返回:
        张量字典，包含 query/pos_doc 的 input_ids 和 attention_mask，
        以及可选的 neg_doc 张量
    """
    questions = [item['question'] for item in batch]

    # 拼接标题和正文作为文档文本
    pos_docs = []
    for item in batch:
        title = item.get('positive_title', '')
        text = item.get('positive_text', '')
        doc_text = f"{title} {text}".strip() if title else text
        pos_docs.append(doc_text)

    # Tokenize查询
    query_encoded = tokenizer(
        questions,
        max_length=max_query_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Tokenize正例文档
    pos_doc_encoded = tokenizer(
        pos_docs,
        max_length=max_doc_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    result = {
        'query_input_ids': query_encoded['input_ids'],
        'query_attention_mask': query_encoded['attention_mask'],
        'pos_doc_input_ids': pos_doc_encoded['input_ids'],
        'pos_doc_attention_mask': pos_doc_encoded['attention_mask'],
    }

    # 处理难负例
    if include_hard_negatives:
        all_neg_docs = []
        neg_counts = []
        for item in batch:
            hard_negs = item.get('hard_negatives', [])
            for neg in hard_negs:
                neg_title = neg.get('title', '')
                neg_text = neg.get('text', '')
                neg_doc_text = f"{neg_title} {neg_text}".strip() if neg_title else neg_text
                all_neg_docs.append(neg_doc_text)
            neg_counts.append(len(hard_negs))

        if all_neg_docs:
            neg_doc_encoded = tokenizer(
                all_neg_docs,
                max_length=max_doc_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 重塑为 [batch_size, num_negatives, seq_length]
            max_negs = max(neg_counts) if neg_counts else 0
            batch_size = len(batch)
            seq_length = neg_doc_encoded['input_ids'].size(1)

            neg_input_ids = torch.zeros(batch_size, max_negs, seq_length, dtype=torch.long)
            neg_attention_mask = torch.zeros(batch_size, max_negs, seq_length, dtype=torch.long)

            offset = 0
            for i, count in enumerate(neg_counts):
                if count > 0:
                    neg_input_ids[i, :count] = neg_doc_encoded['input_ids'][offset:offset + count]
                    neg_attention_mask[i, :count] = neg_doc_encoded['attention_mask'][offset:offset + count]
                offset += count

            result['neg_doc_input_ids'] = neg_input_ids
            result['neg_doc_attention_mask'] = neg_attention_mask
            result['num_negatives'] = torch.tensor(neg_counts, dtype=torch.long)

    return result


class ThreeStageDataLoader:
    """
    三阶段训练数据加载器

    根据训练阶段动态调整数据加载策略：
    - 阶段1: 仅返回 query + pos_doc（In-Batch负例由损失函数处理）
    - 阶段2: 返回 query + pos_doc + DPR预存难负例
    - 阶段3: 返回 query + pos_doc + 动态难负例

    参数:
        dataset: 训练数据集
        tokenizer: HuggingFace tokenizer
        batch_size: 批次大小
        max_query_length: 查询最大token长度
        max_doc_length: 文档最大token长度
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        pin_memory: 是否使用pinned memory
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        batch_size: int = 128,
        max_query_length: int = 64,
        max_doc_length: int = 256,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.current_stage = 1
        self._dataloader: Optional[DataLoader] = None
        self._rebuild_dataloader()

    def set_stage(self, stage: int) -> None:
        """
        设置训练阶段

        参数:
            stage: 训练阶段 (1, 2, 或 3)
        """
        assert stage in [1, 2, 3], f"阶段必须为 1, 2 或 3，当前为 {stage}"
        if self.current_stage != stage:
            logger.info(f"切换训练阶段: {self.current_stage} -> {stage}")
            self.current_stage = stage
            self._rebuild_dataloader()

    def _rebuild_dataloader(self) -> None:
        """根据当前阶段重建数据加载器"""
        include_hard_negatives = self.current_stage >= 2

        def _collate(batch):
            return collate_fn(
                batch,
                tokenizer=self.tokenizer,
                max_query_length=self.max_query_length,
                max_doc_length=self.max_doc_length,
                include_hard_negatives=include_hard_negatives
            )

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=_collate,
            drop_last=True
        )

        logger.info(
            f"数据加载器已重建: 阶段{self.current_stage}, "
            f"batch_size={self.batch_size}, "
            f"包含难负例={'是' if include_hard_negatives else '否'}"
        )

    def refresh_hard_negatives(self, encoder: Any) -> None:
        """
        刷新动态难负例（阶段3使用）

        使用当前编码器重新编码并检索最新的难负例。
        该方法由 HardNegativeMiner 配合使用。

        参数:
            encoder: 当前的BiEncoder模型
        """
        if self.current_stage < 3:
            logger.warning("仅阶段3支持动态难负例刷新，当前为阶段%d", self.current_stage)
            return

        logger.info("开始刷新动态难负例...")
        # 动态难负例挖掘在 HardNegativeMiner 中实现，
        # 此处预留接口，训练时由 Trainer 协调调用
        pass

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代数据批次"""
        if self._dataloader is None:
            self._rebuild_dataloader()
        return iter(self._dataloader)

    def __len__(self) -> int:
        """返回批次数量"""
        return len(self._dataloader) if self._dataloader is not None else 0

    def get_stage(self) -> int:
        """返回当前训练阶段"""
        return self.current_stage


class InBatchNegativeSampler(Sampler):
    """
    批次内负例采样器

    确保同一批次内的样本可以互作为负例，最大化负例利用率。
    打乱数据后按固定batch_size分组，保证每个batch内样本多样性。

    参数:
        data_source: 数据源
        batch_size: 批次大小
        drop_last: 是否丢弃最后不完整的批次
        seed: 随机种子
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch，用于控制随机打乱"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """生成打乱后的采样索引"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        n = len(self.data_source)
        indices = torch.randperm(n, generator=g).tolist()

        # 按batch_size分组
        if self.drop_last:
            indices = indices[:n - n % self.batch_size]

        return iter(indices)

    def __len__(self) -> int:
        """返回样本数量"""
        n = len(self.data_source)
        if self.drop_last:
            return n - n % self.batch_size
        return n