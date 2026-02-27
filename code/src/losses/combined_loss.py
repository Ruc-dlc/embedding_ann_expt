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
        设置距离损失权重，用于三阶段训练中动态调整

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


class ScheduledCombinedLoss(CombinedLoss):
    """
    带调度的联合损失函数

    支持距离权重的动态调度，用于三阶段训练策略：
    - 预热阶段：权重为0（纯InfoNCE）
    - 增长阶段：权重线性增长
    - 稳定阶段：使用最终权重

    参数:
        temperature: InfoNCE温度参数
        initial_distance_weight: 初始距离损失权重
        final_distance_weight: 最终距离损失权重
        warmup_steps: 预热步数（权重为0）
        rampup_steps: 权重增长步数
        use_in_batch_negatives: 是否使用批内负例
    """

    def __init__(
        self,
        temperature: float = 0.05,
        initial_distance_weight: float = 0.0,
        final_distance_weight: float = 0.6,
        warmup_steps: int = 1000,
        rampup_steps: int = 5000,
        use_in_batch_negatives: bool = True
    ):
        super().__init__(
            temperature=temperature,
            distance_weight=initial_distance_weight,
            use_in_batch_negatives=use_in_batch_negatives
        )

        self.initial_distance_weight = initial_distance_weight
        self.final_distance_weight = final_distance_weight
        self.warmup_steps = warmup_steps
        self.rampup_steps = rampup_steps
        self.current_step = 0

    def step(self) -> None:
        """更新步数并调整权重"""
        self.current_step += 1
        self._update_weight()

    def _update_weight(self) -> None:
        """根据当前步数更新距离损失权重"""
        if self.current_step < self.warmup_steps:
            # 预热阶段：权重为0
            self.distance_weight = 0.0
        elif self.current_step < self.warmup_steps + self.rampup_steps:
            # 增长阶段：线性增长
            progress = (self.current_step - self.warmup_steps) / self.rampup_steps
            self.distance_weight = self.initial_distance_weight + \
                (self.final_distance_weight - self.initial_distance_weight) * progress
        else:
            # 稳定阶段：使用最终权重
            self.distance_weight = self.final_distance_weight

    def get_schedule_info(self) -> Dict[str, float]:
        """
        获取调度状态信息

        返回:
            调度状态字典
        """
        return {
            'current_step': self.current_step,
            'current_weight': self.distance_weight,
            'warmup_steps': self.warmup_steps,
            'rampup_steps': self.rampup_steps,
            'final_weight': self.final_distance_weight
        }