"""
DACL-DR 损失函数

L_total = L_InfoNCE + w * L_dis

- L_InfoNCE: 带温度系数的对比损失（全局 softmax）
- L_dis: 距离感知正则项 (1 - cos(q, d+))

两种计算模式：
  - in_batch: 相似度矩阵 B×B（Stage 1）
  - hard_neg:  相似度矩阵 B×8B（Stage 2/3）

参考：
  - DPR/dpr/models/biencoder.py  BiEncoderNllLoss
  - experiments.md 第三节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DACLLoss(nn.Module):
    """DACL-DR 联合损失函数。

    L = InfoNCE(τ) + w * L_dis

    InfoNCE 使用 F.cross_entropy 计算（内置 log-sum-exp 数值稳定性处理）。
    L_dis = mean(1 - q · d+)，对 batch 内所有 (query, positive) 对取平均。

    Args:
        temperature: 温度系数 τ（默认 0.05）
        distance_weight: 距离正则权重 w（默认 0.4）
    """

    def __init__(self, temperature: float = 0.05, distance_weight: float = 0.4):
        super().__init__()
        self.temperature = temperature
        self.distance_weight = distance_weight

    def forward(
        self,
        query_emb: torch.Tensor,
        ctx_emb: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> dict:
        """计算联合损失。

        Args:
            query_emb: query 向量 [B, D]（已 L2 归一化）
            ctx_emb: context 向量 [C, D]（已 L2 归一化）
                Stage "in_batch": C = B（只有 positive）
                Stage "hard_neg": C = B * 8（1 pos + 7 neg per query）
            positive_indices: 每个 query 的 positive 在 ctx_emb 中的索引 [B]
                Stage "in_batch": [0, 1, 2, ..., B-1]（对角线）
                Stage "hard_neg": [0, 8, 16, ..., 8*(B-1)]

        Returns:
            dict with keys:
                - loss: 总损失标量
                - infonce_loss: InfoNCE 损失标量
                - distance_loss: L_dis 损失标量
                - num_correct: 预测正确的 query 数（argmax == positive）
        """
        # 相似度矩阵 [B, C]
        scores = torch.matmul(query_emb, ctx_emb.t())  # [B, C]

        # InfoNCE: scores / τ → cross_entropy
        # F.cross_entropy 内部做 log_softmax + nll_loss，含 log-sum-exp 稳定性处理
        scaled_scores = scores / self.temperature
        infonce_loss = F.cross_entropy(scaled_scores, positive_indices)

        # L_dis: 1 - cos(q, d+) = 1 - q · d+（向量已 L2 归一化）
        # 从 scores 矩阵中按 positive_indices 提取正样本对的点积
        batch_size = query_emb.size(0)
        pos_scores = scores[torch.arange(batch_size, device=scores.device), positive_indices]  # [B]
        distance_loss = (1.0 - pos_scores).mean()

        # 总损失
        loss = infonce_loss + self.distance_weight * distance_loss

        # 统计预测正确数
        with torch.no_grad():
            predictions = scaled_scores.argmax(dim=1)
            num_correct = (predictions == positive_indices).sum().item()

        return {
            "loss": loss,
            "infonce_loss": infonce_loss,
            "distance_loss": distance_loss,
            "num_correct": num_correct,
        }
