"""
DPR格式数据验证与统计脚本

功能：
- 检查NQ和TriviaQA数据集的DPR格式是否正确
- 统计各字段分布（positive_ctxs数量、hard_negative_ctxs数量等）
- 验证字段兼容性（passage_id vs psg_id）
- 过滤无效记录并输出统计报告

使用方法:
    python scripts/preprocess_dpr.py --data_dir data_set/
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dpr_file(file_path: str) -> dict:
    """
    验证单个DPR格式JSON文件

    参数:
        file_path: JSON文件路径

    返回:
        验证统计结果字典
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"文件不存在: {path}")
        return {}

    logger.info(f"正在验证: {path}")
    logger.info(f"文件大小: {path.stat().st_size / (1024*1024):.1f} MB")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    logger.info(f"总记录数: {total}")

    stats = {
        'file': str(path),
        'total_records': total,
        'valid_records': 0,
        'empty_positive': 0,
        'empty_hard_negative': 0,
        'empty_question': 0,
        'id_field': set(),
        'pos_count_dist': Counter(),
        'hard_neg_count_dist': Counter(),
    }

    for i, item in enumerate(data):
        # 检查question字段
        question = item.get('question', '')
        if not question or not question.strip():
            stats['empty_question'] += 1

        # 检查positive_ctxs
        pos_ctxs = item.get('positive_ctxs', [])
        pos_count = len(pos_ctxs)
        stats['pos_count_dist'][pos_count] += 1

        if pos_count == 0:
            stats['empty_positive'] += 1
        else:
            stats['valid_records'] += 1

        # 检查hard_negative_ctxs
        hard_neg_ctxs = item.get('hard_negative_ctxs', [])
        hard_neg_count = len(hard_neg_ctxs)
        stats['hard_neg_count_dist'][hard_neg_count] += 1

        if hard_neg_count == 0:
            stats['empty_hard_negative'] += 1

        # 检查段落ID字段名
        for ctx in pos_ctxs + hard_neg_ctxs:
            if 'passage_id' in ctx:
                stats['id_field'].add('passage_id')
            if 'psg_id' in ctx:
                stats['id_field'].add('psg_id')

        # 抽样检查前3条的详细结构
        if i < 3:
            logger.info(f"  样本{i}: question='{question[:50]}...', "
                       f"pos={pos_count}, hard_neg={hard_neg_count}, "
                       f"answers={len(item.get('answers', []))}")

    return stats


def print_stats(stats: dict) -> None:
    """打印统计报告"""
    if not stats:
        return

    logger.info("=" * 60)
    logger.info(f"文件: {stats['file']}")
    logger.info(f"总记录数: {stats['total_records']}")
    logger.info(f"有效记录数（有正例）: {stats['valid_records']}")
    logger.info(f"无正例记录数: {stats['empty_positive']}")
    logger.info(f"无难负例记录数: {stats['empty_hard_negative']}")
    logger.info(f"空问题记录数: {stats['empty_question']}")
    logger.info(f"段落ID字段: {stats['id_field']}")

    # 正例数量分布
    pos_dist = stats['pos_count_dist']
    if pos_dist:
        counts = sorted(pos_dist.keys())
        logger.info(f"正例数量范围: {min(counts)} ~ {max(counts)}")
        if len(counts) <= 10:
            for k in counts:
                logger.info(f"  positive_ctxs={k}: {pos_dist[k]} 条")

    # 难负例数量分布
    neg_dist = stats['hard_neg_count_dist']
    if neg_dist:
        counts = sorted(neg_dist.keys())
        logger.info(f"难负例数量范围: {min(counts)} ~ {max(counts)}")

    logger.info("=" * 60)


def validate_corpus(corpus_path: str) -> None:
    """
    验证语料库文件（psgs_w100.tsv）

    参数:
        corpus_path: TSV文件路径
    """
    path = Path(corpus_path)
    if not path.exists():
        logger.warning(f"语料库文件不存在: {path}")
        return

    logger.info(f"正在验证语料库: {path}")
    logger.info(f"文件大小: {path.stat().st_size / (1024*1024*1024):.2f} GB")

    # 只读取前几行验证格式
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            parts = line.strip().split('\t')
            if i == 0:
                logger.info(f"  表头: {parts}")
            else:
                logger.info(f"  第{i}行: id={parts[0]}, "
                           f"text_len={len(parts[1]) if len(parts) > 1 else 0}, "
                           f"title='{parts[2][:30] if len(parts) > 2 else ''}'")

    # 统计总行数（流式读取避免内存问题）
    logger.info("正在统计语料库总行数（可能需要几分钟）...")
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    logger.info(f"语料库总行数: {count:,}（含表头）")


def main():
    parser = argparse.ArgumentParser(description='DPR格式数据验证工具')
    parser.add_argument(
        '--data_dir', type=str, default='data_set/',
        help='数据集根目录'
    )
    parser.add_argument(
        '--check_corpus', action='store_true',
        help='是否验证语料库文件'
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 验证NQ数据集
    nq_files = [
        data_dir / 'NQ' / 'nq-train.json',
        data_dir / 'NQ' / 'nq-dev.json',
    ]

    for f in nq_files:
        stats = validate_dpr_file(str(f))
        print_stats(stats)

    # 验证TriviaQA数据集
    trivia_files = [
        data_dir / 'TriviaQA' / 'trivia-train.json',
        data_dir / 'TriviaQA' / 'trivia-dev.json',
    ]

    for f in trivia_files:
        stats = validate_dpr_file(str(f))
        print_stats(stats)

    # 可选：验证语料库
    if args.check_corpus:
        corpus_path = data_dir / 'psgs_w100.tsv'
        validate_corpus(str(corpus_path))

    logger.info("数据验证完成！")


if __name__ == '__main__':
    main()