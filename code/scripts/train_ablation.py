#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验自动化脚本

自动遍历不同距离损失权重 w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} 进行训练，
每个w值保存独立模型到 checkpoints/ablation_w{value}/，
训练完成后收集所有w值的验证指标并输出汇总报告。

使用方法:
    python scripts/train_ablation.py --data_dir data_set/
    python scripts/train_ablation.py --weights 0.0 0.2 0.6 --data_dir data_set/

论文章节：第5章 5.2节 - 混合权重实验
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认消融实验权重值
DEFAULT_WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="消融实验：遍历不同距离损失权重w")

    parser.add_argument("--weights", type=float, nargs='+', default=DEFAULT_WEIGHTS,
                        help="要测试的w值列表（默认: 0.0 0.2 0.4 0.6 0.8 1.0）")
    parser.add_argument("--data_dir", type=str, default="data_set/",
                        help="数据集根目录")
    parser.add_argument("--output_base", type=str, default="checkpoints",
                        help="输出根目录")
    parser.add_argument("--model_name", type=str, default="./local_model_backbone",
                        help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="物理训练批次大小（配合gradient_accumulation_steps=4达到有效batch=128）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="InfoNCE温度参数")
    parser.add_argument("--stage1_epochs", type=int, default=4, help="阶段1轮数")
    parser.add_argument("--stage2_epochs", type=int, default=8, help="阶段2轮数")
    parser.add_argument("--stage3_epochs", type=int, default=12, help="阶段3轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="使用FP16（不推荐）")
    parser.add_argument("--bf16", action="store_true", help="使用BF16（推荐）")
    parser.add_argument("--max_samples", type=int, default=None, help="调试用最大样本数")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要执行的命令，不实际运行")

    return parser.parse_args()


def build_command(args, weight: float) -> list:
    """为指定权重构建训练命令"""
    output_dir = str(Path(args.output_base) / f"ablation_w{weight}")

    cmd = [
        sys.executable, "scripts/train_distance_aware.py",
        "--data_dir", args.data_dir,
        "--output_dir", output_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--temperature", str(args.temperature),
        "--distance_weight", str(weight),
        "--stage1_epochs", str(args.stage1_epochs),
        "--stage2_epochs", str(args.stage2_epochs),
        "--stage3_epochs", str(args.stage3_epochs),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--seed", str(args.seed),
    ]

    if args.bf16:
        cmd.append("--bf16")
    elif args.fp16:
        cmd.append("--fp16")

    if args.max_samples is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])

    return cmd


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("消融实验：距离损失权重 w")
    logger.info(f"测试权重: {args.weights}")
    logger.info("=" * 60)

    results = {}

    for i, weight in enumerate(args.weights):
        logger.info(f"\n{'='*40}")
        logger.info(f"[{i+1}/{len(args.weights)}] 训练 w = {weight}")
        logger.info(f"{'='*40}")

        cmd = build_command(args, weight)
        output_dir = Path(args.output_base) / f"ablation_w{weight}"

        if args.dry_run:
            logger.info(f"[DRY RUN] 命令: {' '.join(cmd)}")
            continue

        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                capture_output=False,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"w={weight} 训练成功！模型保存至 {output_dir}")
                results[str(weight)] = {
                    "status": "success",
                    "output_dir": str(output_dir),
                }
            else:
                logger.error(f"w={weight} 训练失败，返回码: {result.returncode}")
                results[str(weight)] = {
                    "status": "failed",
                    "return_code": result.returncode,
                }
        except Exception as e:
            logger.error(f"w={weight} 训练异常: {e}")
            results[str(weight)] = {
                "status": "error",
                "message": str(e),
            }

    # 保存汇总报告
    if not args.dry_run:
        report_path = Path(args.output_base) / "ablation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'='*60}")
        logger.info("消融实验完成！")
        logger.info(f"汇总报告: {report_path}")
        logger.info(f"{'='*60}")

        for w, r in results.items():
            status = r.get('status', 'unknown')
            logger.info(f"  w={w}: {status}")


if __name__ == "__main__":
    main()