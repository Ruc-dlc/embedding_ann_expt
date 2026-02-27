"""
标准InfoNCE损失函数

本模块实现InfoNCE（Noise Contrastive Estimation）损失函数，
是对比学习的核心损失函数，用于最大化正样本对的相似度并最小化负样本对的相似度。

论文章节：第3章 3.2节 - 对比学习基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失函数
    
    其中：
    - q: Query向量
    - d+: 正样本文档向量
    - d: 所有文档向量（正样本+负样本）
    - τ: 温度参数
    
    Args:
        temperature (float): 温度参数τ，控制分布的平滑程度，默认0.05
        use_in_batch_negatives (bool): 是否使用批内负例
        similarity_type (str): 相似度计算类型 ("cosine" 或 "dot")
    """
    
    def __init__(
        self,
        temperature: float = 0.05,
        use_in_batch_negatives: bool = True,
        similarity_type: str = "cosine"
    ):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
        self.similarity_type = similarity_type
        
    def forward(
        self,
        query_emb: torch.Tensor,
        pos_doc_emb: torch.Tensor,
        neg_doc_embs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算InfoNCE损失
        
        Args:
            query_emb: Query向量 [batch_size, embedding_dim]
            pos_doc_emb: 正样本文档向量 [batch_size, embedding_dim]
            neg_doc_embs: 负样本文档向量 [batch_size, num_negatives, embedding_dim]
                         如果为None且use_in_batch_negatives=True，则使用批内负例
            
        Returns:
            loss: InfoNCE损失值
            loss_dict: 包含详细信息的字典
        """
        batch_size = query_emb.size(0)
        
        # 计算正样本相似度
        pos_sim = self._compute_similarity(query_emb, pos_doc_emb)
        
        # 收集所有负样本相似度
        if self.use_in_batch_negatives:
            # 批内负例：使用其他样本的正样本作为负例
            all_doc_sim = self._compute_similarity_matrix(query_emb, pos_doc_emb)
        else:
            all_doc_sim = pos_sim.unsqueeze(1)
            
        # 添加显式负例
        if neg_doc_embs is not None:
            neg_sim = self._compute_similarity_batch(query_emb, neg_doc_embs)
            
            if self.use_in_batch_negatives:
                all_doc_sim = torch.cat([all_doc_sim, neg_sim], dim=1)
            else:
                all_doc_sim = torch.cat([all_doc_sim, neg_sim], dim=1)
        
        # 应用温度缩放
        pos_sim = pos_sim / self.temperature
        all_doc_sim = all_doc_sim / self.temperature
        
        # 计算InfoNCE损失
        # 正样本在第一位（对于批内负例）或单独处理
        if self.use_in_batch_negatives:
            # 对角线是正样本
            labels = torch.arange(batch_size, device=query_emb.device)
            loss = F.cross_entropy(all_doc_sim, labels)
        else:
            # 第一列是正样本
            labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
            loss = F.cross_entropy(all_doc_sim, labels)
        
        loss_dict = {
            'infonce': loss.item(),
            'pos_sim_mean': (pos_sim * self.temperature).mean().item(),
            'temperature': self.temperature
        }
        
        return loss, loss_dict
    
    def _compute_similarity(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        计算向量相似度
        """
        if self.similarity_type == "cosine":
            return F.cosine_similarity(query_emb, doc_emb, dim=-1)
        else:
            return torch.sum(query_emb * doc_emb, dim=-1)
    
    def _compute_similarity_matrix(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        计算相似度矩阵
        """
        if self.similarity_type == "cosine":
            query_norm = F.normalize(query_emb, p=2, dim=-1)
            doc_norm = F.normalize(doc_emb, p=2, dim=-1)
            return torch.mm(query_norm, doc_norm.t())
        else:
            return torch.mm(query_emb, doc_emb.t())
    
    def _compute_similarity_batch(
        self,
        query_emb: torch.Tensor,
        doc_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        批量计算相似度
        
        Args:
            
        Returns:
        """
        if self.similarity_type == "cosine":
            query_norm = F.normalize(query_emb, p=2, dim=-1)
            doc_norm = F.normalize(doc_embs, p=2, dim=-1)
            return torch.bmm(doc_norm, query_norm.unsqueeze(-1)).squeeze(-1)
        else:
            return torch.bmm(doc_embs, query_emb.unsqueeze(-1)).squeeze(-1)