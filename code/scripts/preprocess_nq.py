#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NQ / TriviaQA 训练数据验证与统计分析脚本

本脚本对 DPR 格式的训练数据进行完整性验证和统计分析，包括：
- 字段完整性检查（positive_ctxs, hard_negative_ctxs, negative_ctxs）
- passage_id / psg_id 字段兼容性检查
- 正例/BM25负例/随机负例数量分布统计
- 文档池规模估算（去重后的唯一 passage 数，用于 Stage 3 挖掘规模评估）
- 数据质量抽样展示

使用方法:
    # 单个文件
    python scripts/preprocess_nq.py --input_path data_set/NQ/nq-train.json

    # 整个数据目录（自动扫描所有 JSON）
    python scripts/preprocess_nq.py --input_path data_set/ --scan_dir

    # 带文档池规模估算
    python scripts/preprocess_nq.py --input_path data_set/ --scan_dir --estimate_pool

论文章节：第5章 5.1节 - 数据预处理
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DPR训练数据验证与统计分析")

    parser.add_argument(
        "--input_path", type=str, required=True,
        help="DPR格式JSON文件路径，或包含JSON文件的目录"
    )
    parser.add_argument(
        "--scan_dir", action="store_true",
        help="将 input_path 视为目录，递归扫描所有 *train*.json 和 *dev*.json"
    )
    parser.add_argument(
        "--estimate_pool", action="store_true",
        help="估算训练数据文档池规模（Stage 3 挖掘用），按 passage_id 去重"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="最大处理样本数（调试用）"
    )
    parser.add_argument(
        "--show_examples", type=int, default=2,
        help="展示的样本数量（默认2）"
    )

    return parser.parse_args()


def load_dpr_json(filepath: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载DPR格式JSON文件"""
    logger.info(f"加载: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    logger.info(f"  加载 {len(data)} 条记录")
    return data


def get_passage_id(ctx: Dict[str, Any]) -> Optional[str]:
    """从 passage context 中提取 passage_id（兼容 passage_id / psg_id）"""
    pid = ctx.get("passage_id") or ctx.get("psg_id")
    return str(pid) if pid is not None else None


def validate_and_stats(data: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    """
    验证DPR数据完整性并收集统计信息

    返回统计字典
    """
    stats = {
        "filename": filename,
        "total_records": len(data),
        "has_question": 0,
        "has_answers": 0,
        "has_positive_ctxs": 0,
        "has_hard_negative_ctxs": 0,
        "has_negative_ctxs": 0,
        "positive_counts": [],
        "hard_neg_counts": [],
        "neg_counts": [],
        "id_field": Counter(),  # passage_id vs psg_id
        "empty_question": 0,
        "empty_positive": 0,
        "issues": [],
    }

    for i, record in enumerate(data):
        # 字段存在性检查
        if "question" in record and record["question"]:
            stats["has_question"] += 1
        else:
            stats["empty_question"] += 1
            if len(stats["issues"]) < 5:
                stats["issues"].append(f"记录 #{i}: question 缺失或为空")

        if "answers" in record:
            stats["has_answers"] += 1

        # positive_ctxs
        pos = record.get("positive_ctxs", [])
        if pos:
            stats["has_positive_ctxs"] += 1
            stats["positive_counts"].append(len(pos))
            # 检查 passage_id 字段名
            for ctx in pos:
                if "passage_id" in ctx:
                    stats["id_field"]["passage_id"] += 1
                elif "psg_id" in ctx:
                    stats["id_field"]["psg_id"] += 1
                else:
                    stats["id_field"]["missing"] += 1
        else:
            stats["empty_positive"] += 1
            if len(stats["issues"]) < 5:
                stats["issues"].append(f"记录 #{i}: positive_ctxs 缺失或为空")

        # hard_negative_ctxs
        hn = record.get("hard_negative_ctxs", [])
        if hn:
            stats["has_hard_negative_ctxs"] += 1
        stats["hard_neg_counts"].append(len(hn))

        # negative_ctxs
        neg = record.get("negative_ctxs", [])
        if neg:
            stats["has_negative_ctxs"] += 1
        stats["neg_counts"].append(len(neg))

    return stats


def estimate_document_pool(data_files: List[str], max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    估算训练数据文档池规模

    收集所有 positive_ctxs + hard_negative_ctxs + negative_ctxs 中的文档，
    按 passage_id 去重，返回统计信息。
    """
    all_passage_ids = set()
    source_counts = {"positive": 0, "hard_negative": 0, "negative": 0}
    total_records = 0

    for filepath in data_files:
        data = load_dpr_json(filepath, max_samples)
        total_records += len(data)

        for record in data:
            for ctx in record.get("positive_ctxs", []):
                pid = get_passage_id(ctx)
                if pid:
                    all_passage_ids.add(pid)
                    source_counts["positive"] += 1

            for ctx in record.get("hard_negative_ctxs", []):
                pid = get_passage_id(ctx)
                if pid:
                    all_passage_ids.add(pid)
                    source_counts["hard_negative"] += 1

            for ctx in record.get("negative_ctxs", []):
                pid = get_passage_id(ctx)
                if pid:
                    all_passage_ids.add(pid)
                    source_counts["negative"] += 1

    pool_size = len(all_passage_ids)
    # 估算编码时间和存储（768d float32）
    est_storage_gb = pool_size * 768 * 4 / (1024 ** 3)
    est_encode_min = pool_size / (256 * 60)  # batch=256, ~60 batches/min on A6000

    return {
        "total_records": total_records,
        "total_files": len(data_files),
        "unique_passages": pool_size,
        "source_counts": source_counts,
        "total_passages_before_dedup": sum(source_counts.values()),
        "dedup_ratio": 1 - pool_size / max(sum(source_counts.values()), 1),
        "estimated_storage_gb": round(est_storage_gb, 2),
        "estimated_encode_minutes": round(est_encode_min, 1),
    }


def print_stats(stats: Dict[str, Any]) -> None:
    """格式化打印统计信息"""
    print(f"\n{'=' * 60}")
    print(f"文件: {stats['filename']}")
    print(f"{'=' * 60}")
    print(f"总记录数: {stats['total_records']}")
    print()

    # 字段覆盖率
    total = stats["total_records"]
    print("字段覆盖率:")
    print(f"  question:           {stats['has_question']}/{total} ({100*stats['has_question']/max(total,1):.1f}%)")
    print(f"  answers:            {stats['has_answers']}/{total} ({100*stats['has_answers']/max(total,1):.1f}%)")
    print(f"  positive_ctxs:      {stats['has_positive_ctxs']}/{total} ({100*stats['has_positive_ctxs']/max(total,1):.1f}%)")
    print(f"  hard_negative_ctxs: {stats['has_hard_negative_ctxs']}/{total} ({100*stats['has_hard_negative_ctxs']/max(total,1):.1f}%)")
    print(f"  negative_ctxs:      {stats['has_negative_ctxs']}/{total} ({100*stats['has_negative_ctxs']/max(total,1):.1f}%)")
    print()

    # 数量分布
    if stats["positive_counts"]:
        counts = stats["positive_counts"]
        print(f"positive_ctxs 数量分布: min={min(counts)}, max={max(counts)}, "
              f"mean={sum(counts)/len(counts):.1f}, median={sorted(counts)[len(counts)//2]}")

    if stats["hard_neg_counts"]:
        counts = [c for c in stats["hard_neg_counts"] if c > 0]
        if counts:
            print(f"hard_negative_ctxs 数量分布: min={min(counts)}, max={max(counts)}, "
                  f"mean={sum(counts)/len(counts):.1f}, median={sorted(counts)[len(counts)//2]}")

    if stats["neg_counts"]:
        counts = [c for c in stats["neg_counts"] if c > 0]
        if counts:
            print(f"negative_ctxs 数量分布: min={min(counts)}, max={max(counts)}, "
                  f"mean={sum(counts)/len(counts):.1f}, median={sorted(counts)[len(counts)//2]}")

    # passage_id 字段名
    print()
    print(f"passage_id 字段名统计: {dict(stats['id_field'])}")
    if stats["id_field"].get("missing", 0) > 0:
        print(f"  WARNING: {stats['id_field']['missing']} 个 passage 缺少 passage_id/psg_id 字段!")

    # 问题
    if stats["issues"]:
        print()
        print(f"发现的问题（前{len(stats['issues'])}条）:")
        for issue in stats["issues"]:
            print(f"  - {issue}")

    if stats["empty_question"] == 0 and stats["empty_positive"] == 0:
        print()
        print("数据完整性检查: PASSED")
    else:
        print()
        print(f"数据完整性检查: WARNING (空question={stats['empty_question']}, 空positive={stats['empty_positive']})")


def show_examples(data: List[Dict[str, Any]], n: int = 2) -> None:
    """展示数据样本"""
    if n <= 0:
        return

    print(f"\n{'=' * 60}")
    print(f"数据样本（前 {min(n, len(data))} 条）")
    print(f"{'=' * 60}")

    for i, record in enumerate(data[:n]):
        print(f"\n--- 样本 #{i+1} ---")
        print(f"  question: {record.get('question', 'N/A')[:100]}")
        print(f"  answers: {record.get('answers', 'N/A')}")

        pos = record.get("positive_ctxs", [])
        if pos:
            ctx = pos[0]
            pid = get_passage_id(ctx) or "N/A"
            title = ctx.get("title", "")[:50]
            text = ctx.get("text", "")[:80]
            print(f"  positive_ctxs[0]: id={pid}, title=\"{title}\", text=\"{text}...\"")
        print(f"  positive_ctxs 数量: {len(pos)}")

        hn = record.get("hard_negative_ctxs", [])
        print(f"  hard_negative_ctxs 数量: {len(hn)}")

        neg = record.get("negative_ctxs", [])
        print(f"  negative_ctxs 数量: {len(neg)}")


def find_data_files(input_path: str, scan_dir: bool) -> List[str]:
    """查找数据文件"""
    path = Path(input_path)

    if not scan_dir:
        if path.is_file():
            return [str(path)]
        else:
            logger.error(f"文件不存在: {input_path}")
            return []

    # 递归扫描目录
    patterns = ["*train*.json", "*dev*.json"]
    files = []
    for pattern in patterns:
        files.extend(sorted(path.glob(f"**/{pattern}")))

    file_paths = [str(f) for f in files]
    logger.info(f"在 {input_path} 下发现 {len(file_paths)} 个数据文件:")
    for fp in file_paths:
        logger.info(f"  {fp}")

    return file_paths


def main():
    """主函数"""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DPR 训练数据验证与统计分析")
    logger.info("=" * 60)

    # 查找数据文件
    data_files = find_data_files(args.input_path, args.scan_dir)
    if not data_files:
        logger.error("未找到数据文件！")
        return

    # 对每个文件进行验证和统计
    for filepath in data_files:
        data = load_dpr_json(filepath, args.max_samples)
        if not data:
            logger.warning(f"文件为空或加载失败: {filepath}")
            continue

        stats = validate_and_stats(data, filepath)
        print_stats(stats)
        show_examples(data, args.show_examples)

    # 文档池规模估算
    if args.estimate_pool:
        # 仅对训练文件估算
        train_files = [f for f in data_files if "train" in f.lower()]
        if train_files:
            print(f"\n{'=' * 60}")
            print("训练数据文档池规模估算 (Stage 3 挖掘)")
            print(f"{'=' * 60}")

            pool_stats = estimate_document_pool(train_files, args.max_samples)
            print(f"  训练文件数: {pool_stats['total_files']}")
            print(f"  训练记录总数: {pool_stats['total_records']}")
            print(f"  文档来源统计 (去重前):")
            print(f"    positive_ctxs:      {pool_stats['source_counts']['positive']:>10,}")
            print(f"    hard_negative_ctxs: {pool_stats['source_counts']['hard_negative']:>10,}")
            print(f"    negative_ctxs:      {pool_stats['source_counts']['negative']:>10,}")
            print(f"    总计:               {pool_stats['total_passages_before_dedup']:>10,}")
            print(f"  去重后唯一文档数: {pool_stats['unique_passages']:,}")
            print(f"  去重比例: {pool_stats['dedup_ratio']:.1%}")
            print(f"  预估编码存储 (768d float32): {pool_stats['estimated_storage_gb']} GB")
            print(f"  预估编码时间 (A6000, batch=256): ~{pool_stats['estimated_encode_minutes']} 分钟")
        else:
            logger.warning("未找到训练文件（文件名需包含 'train'）")

    print(f"\n{'=' * 60}")
    print("分析完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
