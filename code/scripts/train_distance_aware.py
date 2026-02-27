#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
距离感知模型训练脚本

训练距离感知对比学习模型，使用三阶段训练策略：
- 阶段1（预热）: 纯InfoNCE，distance_weight=0
- 阶段2（距离引入）: 引入距离约束，使用DPR预存难负例
- 阶段3（联合优化）: 完整联合损失 + 动态难负例

支持通过 --distance_weight 指定不同w值进行消融实验。

使用方法:
    python scripts/train_distance_aware.py --data_dir data_set/ --output_dir checkpoints/distance_aware/
    python scripts/train_distance_aware.py --distance_weight 0.4 --output_dir checkpoints/ablation_w0.4/

论文章节：第4章 - 距离感知对比学习方法
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from src.models.bi_encoder import BiEncoder
from src.losses.combined_loss import ScheduledCombinedLoss
from src.data.dataset import NQDataset, TriviaQADataset, ConcatRetrievalDataset
from src.data.dataloader import ThreeStageDataLoader
from src.training.trainer import Trainer
from src.training.training_args import TrainingArguments
from src.training.callbacks import CheckpointCallback, LoggingCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练距离感知对比学习模型")

    parser.add_argument("--data_dir", type=str, default="data_set/",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints/distance_aware",
                        help="模型输出目录")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="InfoNCE温度参数τ")
    parser.add_argument("--distance_weight", type=float, default=0.6,
                        help="距离损失权重w（消融实验可设 0/0.2/0.4/0.6/0.8/1.0）")
    parser.add_argument("--stage1_epochs", type=int, default=4,
                        help="阶段1（预热）轮数")
    parser.add_argument("--stage2_epochs", type=int, default=8,
                        help="阶段2（距离引入）轮数")
    parser.add_argument("--stage3_epochs", type=int, default=12,
                        help="阶段3（联合优化）轮数")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="损失权重预热步数")
    parser.add_argument("--rampup_steps", type=int, default=5000,
                        help="损失权重增长步数")
    parser.add_argument("--max_query_length", type=int, default=64,
                        help="查询最大token长度")
    parser.add_argument("--max_doc_length", type=int, default=256,
                        help="文档最大token长度")
    parser.add_argument("--num_hard_negatives", type=int, default=7,
                        help="每条样本的难负例数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="使用FP16混合精度")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大加载样本数（调试用）")

    return parser.parse_args()


def build_dataset(data_dir: str, num_hard_negatives: int, max_samples=None):
    """构建NQ + TriviaQA合并数据集"""
    data_dir = Path(data_dir)
    datasets = []

    nq_train_path = data_dir / "NQ" / "nq-train.json"
    if nq_train_path.exists():
        nq_dataset = NQDataset(
            str(nq_train_path),
            num_hard_negatives=num_hard_negatives,
            max_samples=max_samples
        )
        datasets.append(nq_dataset)
        logger.info(f"NQ训练集: {len(nq_dataset)} 条")

    trivia_train_path = data_dir / "TriviaQA" / "trivia-train.json"
    if trivia_train_path.exists():
        trivia_dataset = TriviaQADataset(
            str(trivia_train_path),
            num_hard_negatives=num_hard_negatives,
            max_samples=max_samples
        )
        datasets.append(trivia_dataset)
        logger.info(f"TriviaQA训练集: {len(trivia_dataset)} 条")

    if not datasets:
        raise FileNotFoundError(f"在 {data_dir} 下未找到训练数据文件")

    if len(datasets) == 1:
        return datasets[0]
    return ConcatRetrievalDataset(datasets)


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs

    logger.info("=" * 60)
    logger.info("训练距离感知对比学习模型（三阶段策略）")
    logger.info(f"  距离权重 w = {args.distance_weight}")
    logger.info(f"  阶段1: {args.stage1_epochs} epochs（纯InfoNCE预热）")
    logger.info(f"  阶段2: {args.stage2_epochs} epochs（引入距离约束）")
    logger.info(f"  阶段3: {args.stage3_epochs} epochs（联合优化）")
    logger.info(f"  总计: {total_epochs} epochs")
    logger.info("=" * 60)

    # 构建tokenizer
    logger.info(f"加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 构建数据集
    logger.info("加载数据集...")
    dataset = build_dataset(args.data_dir, args.num_hard_negatives, args.max_samples)
    logger.info(f"总训练样本数: {len(dataset)}")

    # 构建三阶段数据加载器
    train_dataloader = ThreeStageDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 构建模型
    logger.info(f"构建BiEncoder模型: {args.model_name}")
    model = BiEncoder(
        model_name=args.model_name,
        pooling_type='cls',
        shared_encoder=True,
        normalize=True
    )

    # 构建带调度的联合损失函数
    loss_fn = ScheduledCombinedLoss(
        temperature=args.temperature,
        initial_distance_weight=0.0,
        final_distance_weight=args.distance_weight,
        warmup_steps=args.warmup_steps,
        rampup_steps=args.rampup_steps,
        use_in_batch_negatives=True
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_epochs=total_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        distance_weight=args.distance_weight,
        fp16=args.fp16,
        enable_three_stage=True,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        seed=args.seed,
    )

    # 回调
    callbacks = [
        LoggingCallback(logging_steps=100),
        CheckpointCallback(save_dir=args.output_dir, save_steps=2000),
    ]

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        loss_fn=loss_fn,
        callbacks=callbacks
    )

    # 开始训练
    logger.info("开始三阶段训练...")
    results = trainer.train()

    # 保存最终模型
    save_dir = Path(args.output_dir) / "final_model"
    model.save_pretrained(str(save_dir))
    logger.info(f"距离感知模型训练完成！模型已保存至 {save_dir}")
    logger.info(f"训练结果: {results}")


if __name__ == "__main__":
    main()