#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
索引模块单元测试

本模块测试向量索引的构建和搜索功能。
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFlatIndex:
    """Flat精确索引测试"""
    
    def test_flat_index_build(self):
        """测试Flat索引构建"""
        from src.indexing.flat_index import FlatIndex
        
        dimension = 128
        num_vectors = 1000
        
        index = FlatIndex(dimension=dimension)
        vectors = np.random.randn(num_vectors, dimension).astype('float32')
        
        index.build(vectors)
        
        assert index.get_num_vectors() == num_vectors
        
    def test_flat_index_search(self):
        """测试Flat索引搜索"""
        from src.indexing.flat_index import FlatIndex
        
        dimension = 128
        num_vectors = 1000
        k = 10
        
        index = FlatIndex(dimension=dimension)
        vectors = np.random.randn(num_vectors, dimension).astype('float32')
        index.build(vectors)
        
        queries = np.random.randn(5, dimension).astype('float32')
        distances, indices = index.search(queries, k)
        
        assert distances.shape == (5, k)
        assert indices.shape == (5, k)
        assert np.all(indices >= 0)
        assert np.all(indices < num_vectors)


class TestHNSWIndex:
    """HNSW索引测试"""
    
    def test_hnsw_index_build(self):
        """测试HNSW索引构建"""
        from src.indexing.hnsw_index import HNSWIndex
        
        dimension = 128
        num_vectors = 1000
        
        index = HNSWIndex(dimension=dimension, M=16, ef_construction=100)
        vectors = np.random.randn(num_vectors, dimension).astype('float32')
        
        index.build(vectors)
        
        assert index.get_num_vectors() == num_vectors
        
    def test_hnsw_index_search(self):
        """测试HNSW索引搜索"""
        from src.indexing.hnsw_index import HNSWIndex
        
        dimension = 128
        num_vectors = 1000
        k = 10
        
        index = HNSWIndex(dimension=dimension, M=16, ef_construction=100)
        vectors = np.random.randn(num_vectors, dimension).astype('float32')
        index.build(vectors)
        
        queries = np.random.randn(5, dimension).astype('float32')
        distances, indices = index.search(queries, k)
        
        assert distances.shape == (5, k)
        assert indices.shape == (5, k)
        
    def test_ef_search_adjustment(self):
        """测试ef_search参数调整"""
        from src.indexing.hnsw_index import HNSWIndex
        
        index = HNSWIndex(dimension=128, ef_search=64)
        
        assert index.ef_search == 64
        
        index.set_ef_search(128)
        assert index.ef_search == 128


class TestIndexFactory:
    """索引工厂测试"""
    
    def test_create_flat_index(self):
        """测试创建Flat索引"""
        from src.indexing.index_factory import create_index
        
        index = create_index("flat", dimension=128)
        
        assert index is not None
        assert index.dimension == 128
        
    def test_create_hnsw_index(self):
        """测试创建HNSW索引"""
        from src.indexing.index_factory import create_index
        
        index = create_index("hnsw", dimension=128, M=32, ef_construction=200)
        
        assert index is not None
        assert index.M == 32
        
    def test_invalid_index_type(self):
        """测试无效索引类型"""
        from src.indexing.index_factory import create_index
        
        with pytest.raises(ValueError):
            create_index("invalid_type", dimension=128)


class TestIndexSaveLoad:
    """索引保存加载测试"""
    
    def test_save_load_hnsw(self, tmp_path):
        """测试HNSW索引保存和加载"""
        from src.indexing.hnsw_index import HNSWIndex
        
        dimension = 128
        num_vectors = 1000
        
        # 创建并构建索引
        index = HNSWIndex(dimension=dimension)
        vectors = np.random.randn(num_vectors, dimension).astype('float32')
        index.build(vectors)
        
        # 保存
        save_path = str(tmp_path / "test_index.faiss")
        index.save(save_path)
        
        # 加载
        loaded_index = HNSWIndex(dimension=dimension)
        loaded_index.load(save_path)
        
        assert loaded_index.get_num_vectors() == num_vectors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])