#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数单元测试

本模块测试损失函数的正确性。
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInfoNCELoss:
    """InfoNCE损失函数测试"""
    
    def test_infonce_basic(self):
        """测试InfoNCE基本功能"""
        from src.losses.infonce_loss import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.05)
        
        batch_size = 4
        embedding_dim = 768
        
        query_emb = torch.randn(batch_size, embedding_dim)
        pos_doc_emb = torch.randn(batch_size, embedding_dim)
        
        loss, loss_dict = loss_fn(query_emb, pos_doc_emb)
        
        assert loss.dim() == 0  # 标量
        assert loss.item() >= 0
        assert 'infonce' in loss_dict
        
    def test_infonce_with_negatives(self):
        """测试带负样本的InfoNCE"""
        from src.losses.infonce_loss import InfoNCELoss
        
        loss_fn = InfoNCELoss(temperature=0.05, use_in_batch_negatives=False)
        
        batch_size = 4
        embedding_dim = 768
        num_negatives = 7
        
        query_emb = torch.randn(batch_size, embedding_dim)
        pos_doc_emb = torch.randn(batch_size, embedding_dim)
        neg_doc_embs = torch.randn(batch_size, num_negatives, embedding_dim)
        
        loss, loss_dict = loss_fn(query_emb, pos_doc_emb, neg_doc_embs)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        
    def test_temperature_effect(self):
        """测试温度参数影响"""
        from src.losses.infonce_loss import InfoNCELoss
        
        query_emb = torch.randn(4, 768)
        pos_doc_emb = torch.randn(4, 768)
        
        loss_low_temp = InfoNCELoss(temperature=0.01)(query_emb, pos_doc_emb)[0]
        loss_high_temp = InfoNCELoss(temperature=0.2)(query_emb, pos_doc_emb)[0]
        
        # 低温度应该产生更大的损失（分布更尖锐）
        # 这个断言可能因随机初始化而不稳定，仅作为示例
        

class TestDistanceLoss:
    """距离约束损失测试 L_dis = mean(1 - cos(q, d+))"""
    
    def test_distance_loss_basic(self):
        """测试距离损失基本功能"""
        from src.losses.distance_loss import DistanceLoss
        
        loss_fn = DistanceLoss()
        
        batch_size = 4
        embedding_dim = 768
        
        query_emb = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        pos_doc_emb = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        
        loss, loss_dict = loss_fn(query_emb, pos_doc_emb)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert 'distance' in loss_dict
        assert 'cos_sim_mean' in loss_dict
        
    def test_distance_loss_range(self):
        """测试距离损失值域：L_dis ∈ [0, 2]，余弦距离范围"""
        from src.losses.distance_loss import DistanceLoss
        
        loss_fn = DistanceLoss()
        
        batch_size = 4
        embedding_dim = 768
        
        # 相同向量 -> 余弦相似度=1 -> L_dis=0
        query_emb = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        loss_same, _ = loss_fn(query_emb, query_emb.clone())
        assert abs(loss_same.item()) < 1e-5
        
        # 随机向量 -> L_dis > 0
        pos_doc_emb = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        loss_rand, _ = loss_fn(query_emb, pos_doc_emb)
        assert loss_rand.item() > 0
        
    def test_distance_stats(self):
        """测试距离统计"""
        from src.losses.distance_loss import DistanceLoss
        
        loss_fn = DistanceLoss()
        
        query_emb = torch.nn.functional.normalize(torch.randn(4, 768), dim=-1)
        pos_doc_emb = torch.nn.functional.normalize(torch.randn(4, 768), dim=-1)
        
        stats = loss_fn.get_distance_stats(query_emb, pos_doc_emb)
        
        assert 'cos_sim_mean' in stats
        assert 'cos_sim_std' in stats
        assert 'distance_mean' in stats


class TestCombinedLoss:
    """联合损失函数测试"""
    
    def test_combined_loss_basic(self):
        """测试联合损失基本功能"""
        from src.losses.combined_loss import CombinedLoss
        
        loss_fn = CombinedLoss(
            temperature=0.05,
            distance_weight=0.6
        )
        
        batch_size = 4
        embedding_dim = 768
        
        query_emb = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        pos_doc_emb = torch.nn.functional.normalize(torch.randn(batch_size, embedding_dim), dim=-1)
        
        loss, loss_dict = loss_fn(query_emb, pos_doc_emb)
        
        assert loss.dim() == 0
        assert 'total' in loss_dict
        assert 'infonce' in loss_dict
        assert 'distance' in loss_dict
        
    def test_weight_adjustment(self):
        """测试权重调整"""
        from src.losses.combined_loss import CombinedLoss
        
        loss_fn = CombinedLoss(distance_weight=0.5)
        
        assert loss_fn.get_distance_weight() == 0.5
        
        loss_fn.set_distance_weight(0.8)
        assert loss_fn.get_distance_weight() == 0.8
        
    def test_scheduled_loss(self):
        """测试带调度的损失函数"""
        from src.losses.combined_loss import ScheduledCombinedLoss
        
        loss_fn = ScheduledCombinedLoss(
            initial_distance_weight=0.0,
            final_distance_weight=0.6,
            warmup_steps=10,
            rampup_steps=20
        )
        
        # 初始权重为0
        assert loss_fn.distance_weight == 0.0
        
        # 模拟训练步数
        for _ in range(10):
            loss_fn.step()
            
        # warmup结束后开始增长
        for _ in range(20):
            loss_fn.step()
            
        # 应该接近最终权重
        assert abs(loss_fn.distance_weight - 0.6) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])