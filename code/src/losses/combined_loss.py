"""
联合损失函数 L_total = L_InfoNCE + w · L_dis

本模块实现距离感知对比学习的核心损失函数，
将InfoNCE损失与距离约束损失进行加权组合。

通过调节权重w，在语义匹配和距离约束之间取得平衡。

论文章节：第4章 4.1节 - 联合损失函数
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .infonce_loss import InfoNCELoss
from .distance_loss import DistanceLoss


class CombinedLoss(nn.Module):
    """
    联合损失函数

    L_total = L_InfoNCE + w * L_dis
    其中 L_dis = mean(1 - cos(q, d+))

    参数:
        temperature: InfoNCE温度参数τ，默认0.05
        distance_weight: 距离损失权重w，默认0.6
        use_in_batch_negatives: 是否使用批内负例
    """

    def __init__(
        self,
        temperature: float = 0.05,
        distance_weight: float = 0.6,
        use_in_batch_negatives: bool = True
    ):
        super().__init__()

        self.temperature = temperature
        self.distance_weight = distance_weight

        # 初始化子损失函数
        self.infonce_loss = InfoNCELoss(
            temperature=temperature,
            use_in_batch_negatives=use_in_batch_negatives
        )

        self.distance_loss = DistanceLoss()

    def forward(
        self,
        query_emb: torch.Tensor,
        pos_doc_emb: torch.Tensor,
        neg_doc_embs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算联合损失

        参数:
            query_emb: 查询向量 [batch_size, embedding_dim]
            pos_doc_emb: 正样本文档向量 [batch_size, embedding_dim]
            neg_doc_embs: 负样本文档向量 [batch_size, num_negatives, embedding_dim]（可选）

        返回:
            loss: 总损失（标量）
            loss_dict: 各项损失的详细字典
        """
        # 计算InfoNCE损失
        infonce, infonce_dict = self.infonce_loss(query_emb, pos_doc_emb, neg_doc_embs)

        # 计算距离约束损失（仅正样本对）
        distance, distance_dict = self.distance_loss(query_emb, pos_doc_emb)

        # 加权组合
        total_loss = infonce + self.distance_weight * distance

        # 合并损失信息
        loss_dict = {
            'total': total_loss.item(),
            'infonce': infonce.item(),
            'distance': distance.item(),
            'distance_weighted': (self.distance_weight * distance).item(),
            'distance_weight': self.distance_weight,
            **{f'infonce_{k}': v for k, v in infonce_dict.items() if k != 'infonce'},
            **{f'dist_{k}': v for k, v in distance_dict.items() if k != 'distance'}
        }

        return total_loss, loss_dict

    def set_distance_weight(self, weight: float) -> None:
        """
        设置距离损失权重

        参数:
            weight: 新的权重值
        """
        self.distance_weight = weight

    def get_distance_weight(self) -> float:
        """
        获取当前距离损失权重

        返回:
            当前权重值
        """
        return self.distance_weight