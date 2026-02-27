#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NQ数据预处理脚本

本脚本负责Natural Questions数据集的预处理，包括：
- 原始数据解析
- 格式转换
- 难负例生成
- 数据集划分

论文章节：第5章 5.1节 - 数据预处理

"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Preprocess Natural Questions dataset")
    
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to raw NQ data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save processed data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--generate_negatives",
        action="store_true",
        help="Whether to generate BM25 hard negatives"
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=30,
        help="Number of hard negatives per sample"
    )
    
    return parser.parse_args()

def load_raw_nq(input_path: str) -> List[Dict[str, Any]]:
    """
    加载原始NQ数据
    """
    # TODO: 实现NQ数据加载
    logger.info(f"Loading NQ data from {input_path}")
    pass

def extract_passages(document: Dict[str, Any]) -> List[str]:
    """
    从文档中提取段落
    """
    # TODO: 实现段落提取
    pass

def create_training_samples(
    data: List[Dict[str, Any]],
    generate_negatives: bool = False,
    num_negatives: int = 30
) -> List[Dict[str, Any]]:
    """
    创建训练样本
    """
    samples = []
    
    for item in tqdm(data, desc="Creating samples"):
        sample = {
            'query': item.get('question', ''),
            'positive_passages': [],
            'negative_passages': []
        }
        
        # TODO: 实现样本创建逻辑
        
        samples.append(sample)
        
    return samples

def generate_bm25_negatives(
    queries: List[str],
    corpus: List[str],
    num_negatives: int = 30
) -> Dict[int, List[int]]:
    """
    生成BM25难负例
    """
    # TODO: 实现BM25难负例生成
    pass

def save_processed_data(
    samples: List[Dict[str, Any]],
    output_path: str,
    split: str = "train"
) -> None:
    """
    保存处理后的数据
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{split}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Saved {len(samples)} samples to {output_file}")

def main():
    """主函数"""
    args = parse_args()
    
    logger.info("Starting NQ preprocessing...")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")
    
    # 加载数据
    raw_data = load_raw_nq(args.input_path)
    
    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        
    logger.info(f"Loaded {len(raw_data) if raw_data else 0} samples")
    
    # 创建训练样本
    samples = create_training_samples(
        raw_data,
        generate_negatives=args.generate_negatives,
        num_negatives=args.num_negatives
    )
    
    # 保存数据
    save_processed_data(samples, args.output_path)
    
    logger.info("NQ preprocessing completed!")

if __name__ == "__main__":
    main()