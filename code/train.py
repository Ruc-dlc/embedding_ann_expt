#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DACL-DR 训练脚本

训练 DPR 双编码器 + 钟形对齐损失调度:
    L = InfoNCE + w(t) * (1 - cos(q, d_pos))
    w(t) = Wmax * sin^2(pi * t / T)

用法:
    python train.py --wmax 0.6 --output_dir experiments/models/w0.6
    python train.py --wmax 0.0 --output_dir experiments/models/w0.0   # baseline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models import BiEncoder
from src.dataset import MergedTrainDataset, DPRDataset
from src.dataloader import DPRCollator
from src.loss import DACLLoss, compute_w

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Dev 评估: In-batch Recall@K
# ============================================================
@torch.no_grad()
def eval_inbatch_recall(model, dataloader, k: int, device) -> float:
    """
    In-batch Recall@K

    在 [B, 2B] 的相似度矩阵中, 检查正例 (位置 i) 是否在 query i 的 Top-K 中。
    """
    model.eval()
    hits = 0
    total = 0

    for batch in dataloader:
        batch = {
            key: val.to(device) if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()
        }
        bs = batch["batch_size"].item()

        q_emb = model.encode_query(batch["query_input_ids"], batch["query_attention_mask"])
        p_emb = model.encode_document(batch["passage_input_ids"], batch["passage_attention_mask"])

        sim = torch.mm(q_emb, p_emb.t())  # [B, 2B]
        actual_k = min(k, sim.size(1))
        _, topk_idx = sim.topk(actual_k, dim=1)

        for i in range(bs):
            if i in topk_idx[i]:
                hits += 1
            total += 1

    model.train()
    return hits / max(total, 1)


# ============================================================
# 训练主函数
# ============================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # --- 随机种子 ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    collator = DPRCollator(tokenizer, max_query_len=args.max_query_len, max_doc_len=args.max_doc_len)

    # --- 数据集 ---
    logger.info("加载训练数据...")
    train_dataset = MergedTrainDataset(args.data_dir, seed=args.seed, max_samples=args.max_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    logger.info("加载验证数据...")
    nq_dev = DPRDataset(f"{args.data_dir}/NQ/nq-dev.json")
    trivia_dev = DPRDataset(f"{args.data_dir}/TriviaQA/trivia-dev.json")
    nq_dev_loader = DataLoader(nq_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2)
    trivia_dev_loader = DataLoader(trivia_dev, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2)

    # --- 模型 ---
    logger.info(f"构建 BiEncoder: backbone={args.backbone}, shared_encoder=True")
    model = BiEncoder(
        model_name=args.backbone,
        shared_encoder=True,
        pooling_type="cls",
        normalize=True,
    )
    model.to(device)

    # --- 优化器 & 调度器 ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"优化器: AdamW, lr={args.lr}, total_steps={total_steps}, warmup={warmup_steps}")

    # --- 混合精度 ---
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16
    logger.info(f"混合精度: {'FP16' if use_amp else '关闭'}")

    # --- 损失 ---
    loss_fn = DACLLoss()

    # --- Checkpoint 追踪 ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_nq_recall = 0.0
    best_trivia_recall = 0.0
    global_step = 0
    training_log = []  # 每 epoch 的指标记录

    # --- 训练信息 ---
    logger.info("=" * 60)
    logger.info(f"Wmax = {args.wmax}")
    logger.info(f"调度: w(t) = {args.wmax} * sin^2(pi * t / T)  (钟形曲线)")
    logger.info(f"Epochs = {args.epochs}, Batch size = {args.batch_size}")
    logger.info(f"训练样本: {len(train_dataset)}, Steps/epoch: {len(train_loader)}")
    logger.info(f"输出: {output_dir}")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_align = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            bs = batch["batch_size"].item()

            # w(t) 调度
            w_t = compute_w(global_step, total_steps, args.wmax)

            # 前向
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                q_emb = model.encode_query(batch["query_input_ids"], batch["query_attention_mask"])
                p_emb = model.encode_document(batch["passage_input_ids"], batch["passage_attention_mask"])
                loss, info = loss_fn(q_emb, p_emb, bs, w_t)

            # 反向
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += info["total"]
            epoch_infonce += info["infonce"]
            epoch_align += info["align"]

            # 日志
            if global_step % args.log_steps == 0:
                logger.info(
                    f"  Step {global_step}: "
                    f"loss={info['total']:.4f}, "
                    f"infonce={info['infonce']:.4f}, "
                    f"align={info['align']:.4f}, "
                    f"w_t={info['w_t']:.4f}, "
                    f"pos_cos={info['pos_cos_mean']:.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # Epoch 统计
        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_infonce = epoch_infonce / n
        avg_align = epoch_align / n
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"loss={avg_loss:.4f}, infonce={avg_infonce:.4f}, align={avg_align:.4f}, "
            f"time={epoch_time:.0f}s"
        )

        # Epoch 指标记录
        epoch_record = {
            "epoch": epoch,
            "avg_loss": round(avg_loss, 6),
            "avg_infonce": round(avg_infonce, 6),
            "avg_align": round(avg_align, 6),
            "epoch_time_s": round(epoch_time, 1),
            "global_step": global_step,
            "w_t_at_epoch_end": round(compute_w(global_step - 1, total_steps, args.wmax), 6),
        }

        # ============================================================
        # Checkpoint 策略 (temp.md #17)
        # ============================================================
        if epoch <= 10:
            # Epoch 1-10: 按训练 loss 保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_pretrained(str(output_dir / "best_loss"))
                logger.info(f"  [Checkpoint] best_loss={avg_loss:.4f}")
            epoch_record["checkpoint_strategy"] = "best_loss"

        else:
            # Epoch 11-24: 按 dev recall@20 保存
            nq_recall = eval_inbatch_recall(model, nq_dev_loader, k=20, device=device)
            trivia_recall = eval_inbatch_recall(model, trivia_dev_loader, k=20, device=device)
            logger.info(f"  [Dev] NQ Recall@20={nq_recall:.4f}, TriviaQA Recall@20={trivia_recall:.4f}")

            if nq_recall > best_nq_recall:
                best_nq_recall = nq_recall
                model.save_pretrained(str(output_dir / "best_nq"))
                logger.info(f"  [Checkpoint] best_nq recall@20={nq_recall:.4f}")

            if trivia_recall > best_trivia_recall:
                best_trivia_recall = trivia_recall
                model.save_pretrained(str(output_dir / "best_trivia"))
                logger.info(f"  [Checkpoint] best_trivia recall@20={trivia_recall:.4f}")

            epoch_record["checkpoint_strategy"] = "dev_recall"
            epoch_record["nq_dev_recall@20"] = round(nq_recall, 4)
            epoch_record["trivia_dev_recall@20"] = round(trivia_recall, 4)

        training_log.append(epoch_record)

        # 每 epoch 保存训练日志 (防止中断丢失)
        log_path = output_dir / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)

    # 最终保存
    model.save_pretrained(str(output_dir / "final"))
    logger.info("=" * 60)
    logger.info(f"训练完成! best_loss={best_loss:.4f}, best_nq={best_nq_recall:.4f}, best_trivia={best_trivia_recall:.4f}")
    logger.info(f"输出: {output_dir}")
    logger.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="DACL-DR 训练")
    p.add_argument("--wmax", type=float, required=True, help="对齐权重最大值 Wmax")
    p.add_argument("--output_dir", type=str, required=True, help="模型输出目录")
    p.add_argument("--data_dir", type=str, default="data_set", help="数据目录")
    p.add_argument("--backbone", type=str, default="local_model_backbone", help="预训练模型路径")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_query_len", type=int, default=64)
    p.add_argument("--max_doc_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_steps", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=None, help="最大样本数(调试用)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
