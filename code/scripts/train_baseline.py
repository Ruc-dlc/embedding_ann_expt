#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline模型训练脚本

训练标准对比学习Baseline模型（纯InfoNCE，不含距离约束，distance_weight=0）。

使用方法:
    python scripts/train_baseline.py --data_dir data_set/ --output_dir checkpoints/baseline/

论文章节：第5章 5.1节 - Baseline模型
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer

from src.models.bi_encoder import BiEncoder
from src.losses.combined_loss import CombinedLoss
from src.data.dataset import NQDataset, TriviaQADataset, ConcatRetrievalDataset
from src.data.dataloader import ThreeStageDataLoader
from src.training.trainer import Trainer
from src.training.training_args import TrainingArguments
from src.training.callbacks import CheckpointCallback, LoggingCallback, EarlyStoppingCallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练Baseline对比学习模型")

    parser.add_argument("--data_dir", type=str, default="data_set/",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline",
                        help="模型输出目录")
    parser.add_argument("--model_name", type=str, default="./local_model_backbone",
                        help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="训练轮数（对标DPR 40epochs on NQ 58K，NQ+TriviaQA 118K等效20）")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="InfoNCE温度参数")
    parser.add_argument("--max_query_length", type=int, default=64,
                        help="查询最大token长度")
    parser.add_argument("--max_doc_length", type=int, default=256,
                        help="文档最大token长度")
    parser.add_argument("--num_hard_negatives", type=int, default=7,
                        help="每条样本的难负例数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="使用FP16混合精度（不推荐，低温度系数下易NaN）")
    parser.add_argument("--bf16", action="store_true", help="使用BF16混合精度（推荐，指数范围与FP32一致）")
    parser.add_argument("--distance_weight", type=float, default=0.0,
                        help="距离损失权重w（默认0，消融实验Model B可设0.6）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数（有hard neg时建议设为4，物理batch=32时有效batch=128）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大加载样本数（调试用）")

    return parser.parse_args()


def build_dataset(data_dir: str, num_hard_negatives: int, max_samples=None):
    """构建NQ + TriviaQA合并数据集"""
    data_dir = Path(data_dir)

    datasets = []

    # NQ训练集
    nq_train_path = data_dir / "NQ" / "nq-train.json"
    if nq_train_path.exists():
        nq_dataset = NQDataset(
            str(nq_train_path),
            num_hard_negatives=num_hard_negatives,
            max_samples=max_samples
        )
        datasets.append(nq_dataset)
        logger.info(f"NQ训练集: {len(nq_dataset)} 条")

    # TriviaQA训练集
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

    # 合并数据集
    if len(datasets) == 1:
        return datasets[0]
    return ConcatRetrievalDataset(datasets)


def build_dev_dataset(data_dir: str, num_hard_negatives: int):
    """构建验证集（用于训练过程中的模型选择，防止过拟合）"""
    data_dir = Path(data_dir)
    datasets = []

    nq_dev_path = data_dir / "NQ" / "nq-dev.json"
    if nq_dev_path.exists():
        nq_dev = NQDataset(str(nq_dev_path), num_hard_negatives=num_hard_negatives)
        datasets.append(nq_dev)
        logger.info(f"NQ验证集: {len(nq_dev)} 条")

    trivia_dev_path = data_dir / "TriviaQA" / "trivia-dev.json"
    if trivia_dev_path.exists():
        trivia_dev = TriviaQADataset(str(trivia_dev_path), num_hard_negatives=num_hard_negatives)
        datasets.append(trivia_dev)
        logger.info(f"TriviaQA验证集: {len(trivia_dev)} 条")

    if not datasets:
        logger.warning("未找到验证集文件，将无法基于Recall@5保存最佳模型")
        return None

    if len(datasets) == 1:
        return datasets[0]
    return ConcatRetrievalDataset(datasets)


def main():
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 60)
    logger.info("训练Baseline对比学习模型（纯InfoNCE）")
    logger.info("=" * 60)

    # 构建tokenizer
    logger.info(f"加载tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 构建训练数据集
    logger.info("加载训练数据集...")
    dataset = build_dataset(args.data_dir, args.num_hard_negatives, args.max_samples)
    logger.info(f"总训练样本数: {len(dataset)}")

    # 构建验证数据集（用于保存最佳模型，防止过拟合）
    logger.info("加载验证数据集...")
    dev_dataset = build_dev_dataset(args.data_dir, args.num_hard_negatives)

    # 构建数据加载器（Baseline不使用三阶段，固定阶段1）
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

    # 构建验证数据加载器
    eval_dataloader = None
    if dev_dataset is not None:
        eval_dataloader = ThreeStageDataLoader(
            dataset=dev_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size * 2,  # 验证时可用更大batch
            max_query_length=args.max_query_length,
            max_doc_length=args.max_doc_length,
            shuffle=False,
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

    # 构建损失函数（默认distance_weight=0纯InfoNCE，消融实验Model B可设w=0.6）
    loss_fn = CombinedLoss(
        temperature=args.temperature,
        distance_weight=args.distance_weight,
        use_in_batch_negatives=True
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        distance_weight=args.distance_weight,
        fp16=args.fp16,
        bf16=args.bf16,
        enable_three_stage=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )

    # 回调（含早停机制，patience=3表示连续3个epoch无改善则停止）
    callbacks = [
        LoggingCallback(logging_steps=100),
        CheckpointCallback(save_dir=args.output_dir, save_steps=2000),
        EarlyStoppingCallback(patience=3, monitor="eval_recall@5", mode="max"),
    ]

    # 训练器（传入eval_dataloader，实现基于dev集Recall@5的最佳模型保存）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        loss_fn=loss_fn,
        callbacks=callbacks
    )

    # 开始训练
    logger.info("开始训练...")
    results = trainer.train()

    # 训练完成后，best_model已由Trainer自动保存到 output_dir/best_model/
    # 创建final_model作为best_model的符号链接，保持兼容性
    best_model_dir = Path(args.output_dir) / "best_model"
    final_model_dir = Path(args.output_dir) / "final_model"
    if best_model_dir.exists():
        # 复制best_model为final_model（用于后续索引构建和评估）
        import shutil
        if final_model_dir.exists():
            shutil.rmtree(str(final_model_dir))
        shutil.copytree(str(best_model_dir), str(final_model_dir))
        logger.info(f"最佳模型已复制到 {final_model_dir}")
    else:
        # 兜底：如果best_model不存在，保存当前模型
        model.save_pretrained(str(final_model_dir))

    logger.info(f"Baseline训练完成！最佳模型: epoch={results.get('best_epoch', 'N/A')}")
    logger.info(f"训练结果: {results}")


if __name__ == "__main__":
    main()