"""
DACL-DR 三阶段训练 Pipeline

Stage 1 (epoch 0-9):   In-Batch Negatives, B×B
Stage 2 (epoch 10-29):  BM25 Hard Negatives, B×8B
Stage 3 (epoch 30-39):  Model Hard Negatives, B×8B, early stopping patience=5

验证策略：
  Stage 1-2: dev NLL loss 选最优 checkpoint
  Stage 3:   dev Average Rank 选最优 checkpoint + 早停

用法：
  python train.py --dataset nq --output_dir ./checkpoints/nq
  python train.py --dataset trivia --output_dir ./checkpoints/trivia

参考：
  - DPR/train_dense_encoder.py
  - experiments.md 第四节
"""

import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.models import BiEncoder
from src.losses import DACLLoss
from src.data.dataset import DPRDataset, BiEncoderCollator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description="DACL-DR Training")

    # 数据
    parser.add_argument("--dataset", type=str, required=True, choices=["nq", "trivia"],
                        help="数据集名称")
    parser.add_argument("--data_dir", type=str, default="./data_set",
                        help="数据集根目录")
    parser.add_argument("--mined_data_path", type=str, default=None,
                        help="Stage 3 mined hard negatives JSON 路径（为 None 则跳过 Stage 3）")

    # 模型
    parser.add_argument("--model_name", type=str, default="./bert-base-uncase-backbone",
                        help="预训练模型路径")
    parser.add_argument("--pooling_type", type=str, default="cls")
    parser.add_argument("--shared_encoder", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=True)

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--distance_weight", type=float, default=0.4)
    parser.add_argument("--num_hard_negatives", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # 序列长度
    parser.add_argument("--max_query_length", type=int, default=256)
    parser.add_argument("--max_passage_length", type=int, default=256)

    # 阶段 epoch 配置
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=20)
    parser.add_argument("--stage3_max_epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    # Average Rank 验证配置
    parser.add_argument("--val_av_rank_max_qs", type=int, default=2000,
                        help="Average Rank 验证时最多使用的 query 数")

    # 输出
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_steps", type=int, default=50)

    # 断点续训
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="从指定 checkpoint 恢复训练（模型权重路径，如 best_model_stage1 目录）")
    parser.add_argument("--resume_stage", type=int, default=1, choices=[1, 2, 3],
                        help="从第几个 stage 开始（1=Stage1, 2=Stage2, 3=Stage3），需配合 --resume_checkpoint")

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data_paths(args):
    """根据 dataset 名称返回训练/验证/测试文件路径。"""
    if args.dataset == "nq":
        train_path = os.path.join(args.data_dir, "NQ", "nq-train.json")
        dev_path = os.path.join(args.data_dir, "NQ", "nq-dev.json")
    else:
        train_path = os.path.join(args.data_dir, "TriviaQA", "trivia-train.json")
        dev_path = os.path.join(args.data_dir, "TriviaQA", "trivia-dev.json")
    return train_path, dev_path


# ============================================================
# 验证函数
# ============================================================

@torch.no_grad()
def validate_nll(model, loss_fn, dev_loader, device):
    """Stage 1-2 验证：计算 dev 集上的平均 NLL loss 和正确率。"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in dev_loader:
        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        ctx_input_ids = batch["ctx_input_ids"].to(device)
        ctx_attention_mask = batch["ctx_attention_mask"].to(device)

        query_emb = model.encode_query(query_input_ids, query_attention_mask)
        ctx_emb = model.encode_document(ctx_input_ids, ctx_attention_mask)

        # 构造 positive_indices
        if "positive_indices" in batch:
            positive_indices = batch["positive_indices"].to(device)
        else:
            # in_batch 模式：对角线
            bsz = query_emb.size(0)
            positive_indices = torch.arange(bsz, device=device)

        loss_dict = loss_fn(query_emb, ctx_emb, positive_indices)

        total_loss += loss_dict["loss"].item()
        total_correct += loss_dict["num_correct"]
        total_samples += query_emb.size(0)
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate_average_rank(model, dev_dataset, tokenizer, args, device):
    """Stage 3 验证：计算 dev 集上每个 query 的 gold passage 的 average rank。

    逻辑（参考 DPR validate_average_rank）：
    1. 对 dev 集中每个样本，编码 query 和其所有 context（1 pos + hard_neg）
    2. 汇总所有 query vector 和 ctx vector
    3. 计算全局 Q×C 相似度矩阵
    4. 对每个 query 找其 positive 在排序列表中的排名
    5. 取平均
    """
    model.eval()

    collator = BiEncoderCollator(
        tokenizer=tokenizer,
        max_query_length=args.max_query_length,
        max_passage_length=args.max_passage_length,
        stage="hard_neg",
        num_hard_negatives=args.num_hard_negatives,
    )

    # 限制验证 query 数量
    max_qs = min(args.val_av_rank_max_qs, len(dev_dataset))
    indices = list(range(max_qs))

    all_q_embs = []
    all_ctx_embs = []
    positive_idx_per_question = []
    total_ctxs = 0

    # 逐个处理（避免构造巨大的全局矩阵时 OOM）
    eval_batch_size = 32
    for start in range(0, max_qs, eval_batch_size):
        end = min(start + eval_batch_size, max_qs)
        samples = [dev_dataset[i] for i in indices[start:end]]
        batch = collator(samples)

        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        ctx_input_ids = batch["ctx_input_ids"].to(device)
        ctx_attention_mask = batch["ctx_attention_mask"].to(device)
        pos_indices = batch["positive_indices"]  # relative to this batch's ctx

        q_emb = model.encode_query(query_input_ids, query_attention_mask)
        ctx_emb = model.encode_document(ctx_input_ids, ctx_attention_mask)

        all_q_embs.append(q_emb.cpu())
        all_ctx_embs.append(ctx_emb.cpu())

        # 调整 positive indices 为全局偏移
        for idx in pos_indices.tolist():
            positive_idx_per_question.append(total_ctxs + idx)
        total_ctxs += ctx_emb.size(0)

    all_q_embs = torch.cat(all_q_embs, dim=0)      # [Q, D]
    all_ctx_embs = torch.cat(all_ctx_embs, dim=0)   # [C, D]

    logger.info(
        "Average Rank validation: q_vectors=%d, ctx_vectors=%d",
        all_q_embs.size(0), all_ctx_embs.size(0),
    )

    # 计算全局相似度矩阵 [Q, C]
    scores = torch.matmul(all_q_embs, all_ctx_embs.t())

    # 对每个 query，找 positive 的排名
    _, sorted_indices = scores.sort(dim=1, descending=True)

    total_rank = 0
    for i, gold_idx in enumerate(positive_idx_per_question):
        rank = (sorted_indices[i] == gold_idx).nonzero(as_tuple=False)
        if rank.numel() > 0:
            total_rank += rank.item()
        else:
            # 异常情况，positive 未出现（不应发生）
            total_rank += all_ctx_embs.size(0)

    q_num = all_q_embs.size(0)
    avg_rank = total_rank / max(q_num, 1)
    logger.info("Average Rank: %.2f (total questions=%d)", avg_rank, q_num)
    return avg_rank


# ============================================================
# Checkpoint 保存/加载
# ============================================================

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                    best_metric, args, filename):
    """保存完整训练状态。"""
    path = os.path.join(args.output_dir, filename)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "args": vars(args),
    }
    # 原子写入：先写临时文件再重命名
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    logger.info("Saved checkpoint: %s (epoch=%d, step=%d)", filename, epoch, global_step)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """加载 checkpoint 并恢复状态。"""
    logger.info("Loading checkpoint from %s", path)
    state = torch.load(path, map_location="cpu")

    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])

    return state["epoch"], state["global_step"], state.get("best_metric")


def save_best_model(model, args, stage_name):
    """保存最优模型（仅模型权重和配置）。"""
    best_dir = os.path.join(args.output_dir, "best_model_%s" % stage_name)
    model.save_pretrained(best_dir)
    logger.info("Saved best model to %s", best_dir)


# ============================================================
# 单阶段训练循环
# ============================================================

def train_stage(
    model, optimizer, scheduler, scaler, loss_fn,
    train_loader, dev_loader,
    stage_name, start_epoch, num_epochs,
    args, device,
    # Stage 3 专用
    use_average_rank=False,
    dev_dataset=None,
    tokenizer=None,
):
    """执行单个训练阶段。

    Args:
        stage_name: "stage1", "stage2", "stage3"
        start_epoch: 全局起始 epoch
        num_epochs: 本阶段 epoch 数
        use_average_rank: True 则用 Average Rank 验证 + 早停
    """
    best_metric = None
    patience_counter = 0
    global_step = 0

    for epoch_idx in range(num_epochs):
        epoch = start_epoch + epoch_idx
        model.train()
        epoch_loss = 0.0
        epoch_infonce = 0.0
        epoch_dist = 0.0
        epoch_correct = 0
        epoch_samples = 0
        num_batches = 0
        nan_batches = 0

        t_start = time.time()

        for step, batch in enumerate(train_loader):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            ctx_input_ids = batch["ctx_input_ids"].to(device)
            ctx_attention_mask = batch["ctx_attention_mask"].to(device)

            # 构造 positive_indices
            if "positive_indices" in batch:
                positive_indices = batch["positive_indices"].to(device)
            else:
                bsz = query_input_ids.size(0)
                positive_indices = torch.arange(bsz, device=device)

            # Forward
            if args.fp16:
                with autocast():
                    query_emb = model.encode_query(query_input_ids, query_attention_mask)
                    ctx_emb = model.encode_document(ctx_input_ids, ctx_attention_mask)
                    loss_dict = loss_fn(query_emb, ctx_emb, positive_indices)
                    loss = loss_dict["loss"]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
            else:
                query_emb = model.encode_query(query_input_ids, query_attention_mask)
                ctx_emb = model.encode_document(ctx_input_ids, ctx_attention_mask)
                loss_dict = loss_fn(query_emb, ctx_emb, positive_indices)
                loss = loss_dict["loss"]
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            # NaN/Inf 检测
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                logger.warning(
                    "[%s] Epoch %d Step %d: NaN/Inf loss detected, skipping batch (total skipped: %d)",
                    stage_name, epoch, step, nan_batches,
                )
                optimizer.zero_grad()
                continue

            # Backward
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 统计
            epoch_loss += loss_dict["loss"].item()
            epoch_infonce += loss_dict["infonce_loss"].item()
            epoch_dist += loss_dict["distance_loss"].item()
            epoch_correct += loss_dict["num_correct"]
            epoch_samples += query_emb.size(0)
            num_batches += 1

            # 日志
            if (step + 1) % args.log_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / num_batches
                logger.info(
                    "[%s] Epoch %d Step %d/%d | loss=%.4f infonce=%.4f dist=%.4f | lr=%.2e",
                    stage_name, epoch, step + 1, len(train_loader),
                    avg_loss, epoch_infonce / num_batches, epoch_dist / num_batches, lr,
                )

        # Epoch 结束统计
        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / max(num_batches, 1)
        accuracy = epoch_correct / max(epoch_samples, 1)
        logger.info(
            "[%s] Epoch %d finished | avg_loss=%.4f accuracy=%.4f | time=%.1fs | nan_batches=%d",
            stage_name, epoch, avg_loss, accuracy, epoch_time, nan_batches,
        )

        # 验证
        if use_average_rank:
            metric = validate_average_rank(model, dev_dataset, tokenizer, args, device)
            metric_name = "avg_rank"
            is_better = (best_metric is None) or (metric < best_metric)
        else:
            metric, val_acc = validate_nll(model, loss_fn, dev_loader, device)
            metric_name = "nll_loss"
            is_better = (best_metric is None) or (metric < best_metric)
            logger.info(
                "[%s] Epoch %d validation: %s=%.4f accuracy=%.4f",
                stage_name, epoch, metric_name, metric, val_acc,
            )

        # 保存 epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, global_step,
            best_metric, args, "checkpoint_epoch_%d.pt" % epoch,
        )

        # Best model 更新
        if is_better:
            best_metric = metric
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                best_metric, args, "best_%s.pt" % stage_name,
            )
            save_best_model(model, args, stage_name)
            logger.info(
                "[%s] New best %s=%.4f at epoch %d",
                stage_name, metric_name, best_metric, epoch,
            )
        else:
            patience_counter += 1
            if use_average_rank:
                logger.info(
                    "[%s] No improvement. patience=%d/%d",
                    stage_name, patience_counter, args.early_stopping_patience,
                )
            else:
                logger.info(
                    "[%s] No improvement. patience=%d (early stopping disabled)",
                    stage_name, patience_counter,
                )

        # 早停（仅 Stage 3）
        if use_average_rank and patience_counter >= args.early_stopping_patience:
            logger.info(
                "[%s] Early stopping triggered at epoch %d (patience=%d)",
                stage_name, epoch, args.early_stopping_patience,
            )
            break

    return best_metric


# ============================================================
# 主流程
# ============================================================

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Args: %s", json.dumps(vars(args), indent=2, ensure_ascii=False))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 数据路径
    train_path, dev_path = get_data_paths(args)
    logger.info("Train: %s", train_path)
    logger.info("Dev: %s", dev_path)

    # 模型
    model = BiEncoder(
        model_name=args.model_name,
        pooling_type=args.pooling_type,
        shared_encoder=args.shared_encoder,
        normalize=args.normalize,
    )
    model.to(device)
    logger.info("Model loaded: %s (shared=%s, normalize=%s, pooling=%s)",
                args.model_name, args.shared_encoder, args.normalize, args.pooling_type)

    # 损失函数
    loss_fn = DACLLoss(
        temperature=args.temperature,
        distance_weight=args.distance_weight,
    )

    # GradScaler
    scaler = GradScaler() if args.fp16 else None

    # 断点续训：加载已有模型权重并跳过已完成的 stage
    resume_stage = 1  # 默认从 Stage 1 开始
    if args.resume_checkpoint:
        logger.info("Resuming from checkpoint: %s (stage=%d)", args.resume_checkpoint, args.resume_stage)
        if os.path.isdir(args.resume_checkpoint):
            # 目录形式：best_model_stageN 目录
            model = BiEncoder.from_pretrained(args.resume_checkpoint)
            model.to(device)
            logger.info("Loaded model weights from directory: %s", args.resume_checkpoint)
        elif os.path.isfile(args.resume_checkpoint):
            # 文件形式：checkpoint_epoch_N.pt 或 best_stageN.pt
            state = torch.load(args.resume_checkpoint, map_location="cpu")
            model.load_state_dict(state["model_state_dict"])
            model.to(device)
            logger.info("Loaded model weights from checkpoint file: %s (epoch=%d)",
                        args.resume_checkpoint, state.get("epoch", -1))
        else:
            raise FileNotFoundError("Resume checkpoint not found: %s" % args.resume_checkpoint)
        resume_stage = args.resume_stage

    # ====================
    # Stage 1: In-Batch Negatives
    # ====================
    if resume_stage <= 1:
        logger.info("=" * 60)
        logger.info("Stage 1: In-Batch Negatives (epochs 0-%d)", args.stage1_epochs - 1)
        logger.info("=" * 60)

        train_ds_s1 = DPRDataset(train_path, stage="in_batch")
        dev_ds_s1 = DPRDataset(dev_path, stage="in_batch")

        collator_s1 = BiEncoderCollator(
            tokenizer=tokenizer,
            max_query_length=args.max_query_length,
            max_passage_length=args.max_passage_length,
            stage="in_batch",
        )

        train_loader_s1 = DataLoader(
            train_ds_s1, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator_s1, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        dev_loader_s1 = DataLoader(
            dev_ds_s1, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator_s1, num_workers=args.num_workers,
            pin_memory=True,
        )

        # Optimizer + Scheduler (Stage 1)
        total_steps_s1 = len(train_loader_s1) * args.stage1_epochs // args.gradient_accumulation_steps
        warmup_steps_s1 = int(total_steps_s1 * args.warmup_ratio)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps_s1, num_training_steps=total_steps_s1,
        )

        logger.info("Stage 1: total_steps=%d, warmup_steps=%d", total_steps_s1, warmup_steps_s1)

        best_s1 = train_stage(
            model, optimizer, scheduler, scaler, loss_fn,
            train_loader_s1, dev_loader_s1,
            "stage1", start_epoch=0, num_epochs=args.stage1_epochs,
            args=args, device=device,
        )
        logger.info("Stage 1 complete. Best NLL loss: %.4f", best_s1 if best_s1 else float("inf"))

        # 加载 Stage 1 最优模型（避免从 last epoch 继续）
        best_s1_dir = os.path.join(args.output_dir, "best_model_stage1")
        if os.path.exists(best_s1_dir):
            logger.info("Loading best Stage 1 model from %s", best_s1_dir)
            model = BiEncoder.from_pretrained(best_s1_dir)
            model.to(device)
        else:
            logger.warning("best_model_stage1 not found, continuing with last epoch model")
    else:
        logger.info("Skipping Stage 1 (resume_stage=%d)", resume_stage)

    # ====================
    # Stage 2: BM25 Hard Negatives
    # ====================
    if resume_stage <= 2:
        logger.info("=" * 60)
        logger.info("Stage 2: BM25 Hard Negatives (epochs %d-%d)",
                    args.stage1_epochs, args.stage1_epochs + args.stage2_epochs - 1)
        logger.info("=" * 60)

        train_ds_s2 = DPRDataset(train_path, stage="hard_neg", num_hard_negatives=args.num_hard_negatives)
        dev_ds_s2 = DPRDataset(dev_path, stage="hard_neg", num_hard_negatives=args.num_hard_negatives)

        collator_s2 = BiEncoderCollator(
            tokenizer=tokenizer,
            max_query_length=args.max_query_length,
            max_passage_length=args.max_passage_length,
            stage="hard_neg",
            num_hard_negatives=args.num_hard_negatives,
        )

        train_loader_s2 = DataLoader(
            train_ds_s2, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator_s2, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )
        dev_loader_s2 = DataLoader(
            dev_ds_s2, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator_s2, num_workers=args.num_workers,
            pin_memory=True,
        )

        # 重新初始化 Optimizer + Scheduler (Stage 2)
        total_steps_s2 = len(train_loader_s2) * args.stage2_epochs // args.gradient_accumulation_steps
        warmup_steps_s2 = int(total_steps_s2 * args.warmup_ratio)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps_s2, num_training_steps=total_steps_s2,
        )

        logger.info("Stage 2: total_steps=%d, warmup_steps=%d", total_steps_s2, warmup_steps_s2)

        best_s2 = train_stage(
            model, optimizer, scheduler, scaler, loss_fn,
            train_loader_s2, dev_loader_s2,
            "stage2", start_epoch=args.stage1_epochs, num_epochs=args.stage2_epochs,
            args=args, device=device,
        )
        logger.info("Stage 2 complete. Best NLL loss: %.4f", best_s2 if best_s2 else float("inf"))

        # 加载 Stage 2 最优模型（避免从 last epoch 继续）
        best_s2_dir = os.path.join(args.output_dir, "best_model_stage2")
        if os.path.exists(best_s2_dir):
            logger.info("Loading best Stage 2 model from %s", best_s2_dir)
            model = BiEncoder.from_pretrained(best_s2_dir)
            model.to(device)
        else:
            logger.warning("best_model_stage2 not found, continuing with last epoch model")
    else:
        logger.info("Skipping Stage 2 (resume_stage=%d)", resume_stage)

    # ====================
    # Stage 3: Model Hard Negatives (可选)
    # ====================
    if args.mined_data_path and os.path.exists(args.mined_data_path):
        logger.info("=" * 60)
        logger.info("Stage 3: Model Hard Negatives (epochs %d-%d, early stopping patience=%d)",
                    args.stage1_epochs + args.stage2_epochs,
                    args.stage1_epochs + args.stage2_epochs + args.stage3_max_epochs - 1,
                    args.early_stopping_patience)
        logger.info("=" * 60)

        train_ds_s3 = DPRDataset(
            args.mined_data_path, stage="hard_neg", num_hard_negatives=args.num_hard_negatives,
        )
        # Stage 3 dev 集用于 Average Rank 验证
        dev_ds_s3 = DPRDataset(dev_path, stage="hard_neg", num_hard_negatives=args.num_hard_negatives)

        collator_s3 = BiEncoderCollator(
            tokenizer=tokenizer,
            max_query_length=args.max_query_length,
            max_passage_length=args.max_passage_length,
            stage="hard_neg",
            num_hard_negatives=args.num_hard_negatives,
        )

        train_loader_s3 = DataLoader(
            train_ds_s3, batch_size=args.batch_size, shuffle=True,
            collate_fn=collator_s3, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )

        # Stage 3 重新初始化 Optimizer + Scheduler
        total_steps_s3 = len(train_loader_s3) * args.stage3_max_epochs // args.gradient_accumulation_steps
        warmup_steps_s3 = int(total_steps_s3 * args.warmup_ratio)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps_s3, num_training_steps=total_steps_s3,
        )

        logger.info("Stage 3: total_steps=%d, warmup_steps=%d", total_steps_s3, warmup_steps_s3)

        best_s3 = train_stage(
            model, optimizer, scheduler, scaler, loss_fn,
            train_loader_s3, None,  # dev_loader 不用于 NLL，用 validate_average_rank
            "stage3",
            start_epoch=args.stage1_epochs + args.stage2_epochs,
            num_epochs=args.stage3_max_epochs,
            args=args, device=device,
            use_average_rank=True,
            dev_dataset=dev_ds_s3,
            tokenizer=tokenizer,
        )
        logger.info("Stage 3 complete. Best Average Rank: %.2f", best_s3 if best_s3 else float("inf"))

        # 加载 Stage 3 最优模型作为最终模型
        best_s3_dir = os.path.join(args.output_dir, "best_model_stage3")
        if os.path.exists(best_s3_dir):
            logger.info("Loading best Stage 3 model from %s", best_s3_dir)
            model = BiEncoder.from_pretrained(best_s3_dir)
            model.to(device)
        else:
            logger.warning("best_model_stage3 not found, using last epoch model as final")
    else:
        logger.info("No mined data path provided or file not found. Skipping Stage 3.")
        logger.info("To run Stage 3, first run mine_hard_negatives.py, then re-run with --mined_data_path")

    # 最终保存
    final_dir = os.path.join(args.output_dir, "best_model_%s" % args.dataset)
    model.save_pretrained(final_dir)
    logger.info("=" * 60)
    logger.info("Training complete. Final model saved to %s", final_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
