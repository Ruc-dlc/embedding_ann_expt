"""
DACL 损失函数与 w(t) 调度

L = InfoNCE + w(t) * AlignmentLoss

InfoNCE:        cross_entropy over sim_matrix [B, 2B], raw dot product (无温度系数)
AlignmentLoss:  mean(1 - cosine(q_i, d_pos_i))
w(t):           Wmax * sin^2(pi * t / T), 钟形曲线
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_w(step: int, total_steps: int, wmax: float) -> float:
    """
    钟形曲线调度: w(t) = Wmax * sin^2(pi * t / T)

    w(0)   = 0      训练开始: 纯 InfoNCE
    w(T/2) = Wmax   训练中期: 最大对齐压力
    w(T)   = 0      训练结束: 纯 InfoNCE, 恢复 uniformity
    """
    if total_steps == 0 or wmax == 0.0:
        return 0.0
    return wmax * math.sin(math.pi * step / total_steps) ** 2


class DACLLoss(nn.Module):
    """
    L = InfoNCE + w_t * AlignmentLoss

    sim_matrix = Q @ D^T               [B, 2B], raw dot product
    labels = arange(B)                  正例在对角线位置
    InfoNCE = cross_entropy(sim, labels)

    AlignmentLoss = mean(1 - cos(q_i, d_pos_i))
    """

    def forward(
        self,
        query_emb: torch.Tensor,    # [B, d]
        passage_emb: torch.Tensor,  # [2B, d]
        batch_size: int,
        w_t: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # InfoNCE
        sim_matrix = torch.mm(query_emb, passage_emb.t())  # [B, 2B]
        labels = torch.arange(batch_size, device=query_emb.device)
        infonce_loss = F.cross_entropy(sim_matrix, labels)

        # Alignment Loss
        pos_emb = passage_emb[:batch_size]  # [B, d]
        cos_sim = F.cosine_similarity(query_emb, pos_emb, dim=-1)  # [B]
        align_loss = (1.0 - cos_sim).mean()

        # Total
        total_loss = infonce_loss + w_t * align_loss

        loss_dict = {
            "total": total_loss.item(),
            "infonce": infonce_loss.item(),
            "align": align_loss.item(),
            "w_t": w_t,
            "pos_cos_mean": cos_sim.mean().item(),
        }

        return total_loss, loss_dict
