"""
距离约束损失 L_dis

本模块实现距离感知对比学习的核心创新——距离约束损失。
通过最大化查询与正样本文档的余弦相似度，优化向量空间分布。

公式: L_dis = mean(1 - cos(q, d+))

论文章节：第4章 4.1节 - 距离约束损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DistanceLoss(nn.Module):
    """
    距离约束损失

    L_dis = mean(1 - cos(q, d+))

    通过最小化查询向量与正样本文档向量之间的余弦距离，
    显式约束正样本对在向量空间中的绝对位置关系，
    使向量分布更紧凑、更适合HNSW等ANN索引的图结构搜索。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_doc_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算距离约束损失

        参数:
            query_emb: 查询向量 [batch_size, embedding_dim]
            pos_doc_emb: 正样本文档向量 [batch_size, embedding_dim]

        返回:
            loss: 距离约束损失值（标量）
            loss_dict: 包含详细指标的字典
        """
        # 计算余弦相似度 [batch_size]
        cos_sim = F.cosine_similarity(query_emb, pos_doc_emb, dim=-1)

        # L_dis = mean(1 - cos(q, d+))
        loss = (1.0 - cos_sim).mean()

        loss_dict = {
            'distance': loss.item(),
            'cos_sim_mean': cos_sim.mean().item(),
        }

        return loss, loss_dict

    def get_distance_stats(
        self,
        query_emb: torch.Tensor,
        pos_doc_emb: torch.Tensor
    ) -> dict:
        """
        获取距离统计信息，用于监控训练过程中的向量空间分布变化

        参数:
            query_emb: 查询向量 [batch_size, embedding_dim]
            pos_doc_emb: 正样本文档向量 [batch_size, embedding_dim]

        返回:
            统计信息字典（余弦相似度的均值、标准差、最小值、最大值）
        """
        with torch.no_grad():
            cos_sim = F.cosine_similarity(query_emb, pos_doc_emb, dim=-1)

            stats = {
                'cos_sim_mean': cos_sim.mean().item(),
                'cos_sim_std': cos_sim.std().item(),
                'cos_sim_min': cos_sim.min().item(),
                'cos_sim_max': cos_sim.max().item(),
                'distance_mean': (1.0 - cos_sim).mean().item(),
            }

            return stats