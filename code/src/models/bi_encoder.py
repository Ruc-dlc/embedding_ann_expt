"""
双塔编码器

本模块实现双塔（Dual Encoder）架构，用于Query和Document的独立编码。
双塔结构允许预计算Document向量，实现高效的大规模检索。

支持共享或独立的底层Transformer，可配置池化策略。

论文章节：第4章 4.1节 - 双塔编码器架构
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoConfig

from .query_encoder import QueryEncoder
from .doc_encoder import DocEncoder
from .pooling import get_pooling_layer


class BiEncoder(nn.Module):
    """
    双塔编码器

    包含Query编码器和Document编码器，支持：
    - 独立编码Query和Document
    - 共享或独立的底层Transformer
    - 可配置的池化策略（cls / mean / max / attention）

    参数:
        model_name: 预训练模型名称（如 "bert-base-uncased"）
        embedding_dim: 输出向量维度（None表示使用backbone原始维度）
        pooling_type: 池化类型 ("cls", "mean", "max", "attention")
        shared_encoder: Query和Document是否共享编码器backbone
        normalize: 是否对输出向量进行L2归一化
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        embedding_dim: Optional[int] = None,
        pooling_type: str = "cls",
        shared_encoder: bool = True,
        normalize: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling_type = pooling_type
        self.shared_encoder = shared_encoder
        self.normalize = normalize

        # 加载配置获取hidden_size
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.hidden_size

        # 初始化编码器
        self.query_encoder: Optional[QueryEncoder] = None
        self.doc_encoder: Optional[DocEncoder] = None
        self._init_encoders()

    def _init_encoders(self) -> None:
        """
        初始化Query和Document编码器

        共享模式下两个编码器使用同一个backbone；
        独立模式下各自加载独立的预训练模型。
        """
        # 加载backbone
        query_backbone = AutoModel.from_pretrained(self.model_name)

        if self.shared_encoder:
            doc_backbone = query_backbone
        else:
            doc_backbone = AutoModel.from_pretrained(self.model_name)

        # 创建池化层
        projection_dim = self.embedding_dim if self.embedding_dim != self.hidden_size else None

        query_pooling = get_pooling_layer(self.pooling_type, self.hidden_size)

        if self.shared_encoder:
            doc_pooling = query_pooling
        else:
            doc_pooling = get_pooling_layer(self.pooling_type, self.hidden_size)

        # 构建编码器
        self.query_encoder = QueryEncoder(
            backbone=query_backbone,
            pooling=query_pooling,
            projection_dim=projection_dim,
            normalize=self.normalize
        )

        self.doc_encoder = DocEncoder(
            backbone=doc_backbone,
            pooling=doc_pooling,
            projection_dim=projection_dim,
            normalize=self.normalize
        )

    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        编码Query

        参数:
            input_ids: Query的token ID [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]

        返回:
            Query向量 [batch_size, embedding_dim]
        """
        return self.query_encoder(input_ids, attention_mask, **kwargs)

    def encode_document(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        编码Document

        参数:
            input_ids: Document的token ID [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]

        返回:
            Document向量 [batch_size, embedding_dim]
        """
        return self.doc_encoder(input_ids, attention_mask, **kwargs)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，同时编码Query和Document

        参数:
            query_input_ids: Query的token ID [batch_size, query_length]
            query_attention_mask: Query注意力掩码 [batch_size, query_length]
            doc_input_ids: Document的token ID [batch_size, doc_length]
            doc_attention_mask: Document注意力掩码 [batch_size, doc_length]

        返回:
            (query_embeddings, doc_embeddings) 元组
        """
        query_emb = self.encode_query(query_input_ids, query_attention_mask)
        doc_emb = self.encode_document(doc_input_ids, doc_attention_mask)
        return query_emb, doc_emb

    def compute_similarity(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Query和Document向量之间的相似度分数（点积）

        参数:
            query_emb: Query向量 [batch_size, embedding_dim]
            doc_emb: Document向量 [batch_size, embedding_dim] 或 [batch_size, num_docs, embedding_dim]

        返回:
            相似度分数
        """
        if doc_emb.dim() == 2:
            return torch.sum(query_emb * doc_emb, dim=-1)
        else:
            return torch.bmm(doc_emb, query_emb.unsqueeze(-1)).squeeze(-1)

    def save_pretrained(self, save_path: str) -> None:
        """
        保存模型权重和配置

        参数:
            save_path: 保存目录路径
        """
        os.makedirs(save_path, exist_ok=True)

        # 保存模型配置
        config = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'pooling_type': self.pooling_type,
            'shared_encoder': self.shared_encoder,
            'normalize': self.normalize,
        }
        config_path = os.path.join(save_path, 'bi_encoder_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 保存模型权重
        model_path = os.path.join(save_path, 'bi_encoder.pt')
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, load_path: str, map_location: str = 'cpu') -> "BiEncoder":
        """
        从目录加载预训练模型

        参数:
            load_path: 模型目录路径
            map_location: 设备映射（默认'cpu'）

        返回:
            BiEncoder实例
        """
        # 加载配置
        config_path = os.path.join(load_path, 'bi_encoder_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 创建模型实例
        model = cls(
            model_name=config['model_name'],
            embedding_dim=config.get('embedding_dim'),
            pooling_type=config.get('pooling_type', 'cls'),
            shared_encoder=config.get('shared_encoder', True),
            normalize=config.get('normalize', True),
        )

        # 加载权重
        model_path = os.path.join(load_path, 'bi_encoder.pt')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=map_location)
            model.load_state_dict(state_dict)

        return model