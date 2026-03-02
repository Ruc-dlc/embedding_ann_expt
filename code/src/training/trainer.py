"""
训练器主类

本模块实现完整的训练流程，包括：
- 三阶段训练策略（In-Batch负例→BM25难负例→模型难负例）
- AdamW优化器（BERT层与其他层可设不同学习率）
- 梯度累积与梯度裁剪
- FP16混合精度训练
- NaN检测与梯度跳过（防止FP16溢出导致训练崩溃）
- 检查点保存与恢复

论文章节：第4章 4.3节 - 训练流程
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .training_args import TrainingArguments
from .callbacks import TrainingCallback

logger = logging.getLogger(__name__)


class Trainer:
    """
    训练器

    管理完整的模型训练流程，支持三阶段训练策略。

    参数:
        model: 双塔编码器模型（BiEncoder）
        args: 训练参数
        train_dataloader: 训练数据加载器（ThreeStageDataLoader）
        eval_dataloader: 验证数据加载器（可选）
        loss_fn: 损失函数（CombinedLoss）
        tokenizer: HuggingFace tokenizer（Stage 2→3 自动挖掘时需要）
        optimizer: 优化器（None则自动创建AdamW）
        scheduler: 学习率调度器（None则自动创建）
        callbacks: 训练回调列表
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: Any,
        eval_dataloader: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        self.global_step = 0
        self.current_epoch = 0
        self.current_stage = 1

        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.loss_fn is not None:
            self.loss_fn.to(self.device)

        # FP16混合精度
        self.scaler = None
        if args.fp16 and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()

        # 初始化优化器和调度器
        self._setup_optimizer()
        self._setup_scheduler()

    def _setup_optimizer(self) -> None:
        """
        设置优化器

        使用AdamW，BERT backbone层和其他层（投影层等）使用不同学习率：
        - BERT层: args.learning_rate
        - 其他层: args.learning_rate * 10
        """
        if self.optimizer is not None:
            return

        # 区分BERT参数和其他参数
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                'params': backbone_params,
                'lr': self.args.learning_rate,
                'weight_decay': self.args.weight_decay,
            },
            {
                'params': other_params,
                'lr': self.args.learning_rate * 10,
                'weight_decay': 0.0,
            },
        ]

        self.optimizer = AdamW(param_groups)
        logger.info(
            f"优化器已创建: backbone参数 {len(backbone_params)} 个, "
            f"其他参数 {len(other_params)} 个"
        )

    def _setup_scheduler(self) -> None:
        """
        设置学习率调度器

        使用线性预热+线性衰减策略。
        """
        if self.scheduler is not None:
            return

        if self.optimizer is None:
            return

        total_steps = len(self.train_dataloader) * self.args.num_epochs
        warmup_steps = self.args.warmup_steps
        if warmup_steps == 0 and self.args.warmup_ratio > 0:
            warmup_steps = int(total_steps * self.args.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(
            f"学习率调度器已创建: 总步数={total_steps}, "
            f"预热步数={warmup_steps}"
        )

    def train(self) -> Dict[str, float]:
        """
        执行完整训练流程

        最佳模型保存策略（业界标准做法）：
        - 若提供了 eval_dataloader（dev集），则每个epoch末在dev集上评估，
          基于 Recall@5 保存最佳模型权重，防止过拟合。
        - 若未提供 eval_dataloader，则退化为基于 train_loss 保存。
        - 最佳模型同时保存为 checkpoint（含optimizer状态，可恢复训练）
          和 BiEncoder pretrained 格式（可直接加载用于推理）。

        返回:
            训练结果统计字典
        """
        logger.info(f"开始训练，共 {self.args.num_epochs} 个epoch")
        logger.info(f"设备: {self.device}")
        if self.eval_dataloader is not None:
            logger.info("已检测到验证集，将基于 Recall@5 保存最佳模型（防过拟合）")
        else:
            logger.warning(
                "⚠️ 未提供验证集(eval_dataloader=None)，"
                "将基于 train_loss 保存最佳模型。"
                "强烈建议传入 dev 集以获得科学的模型选择。"
            )

        self._call_callbacks("on_train_begin")

        best_metric = 0.0  # Recall@5 越大越好
        best_epoch = 0
        no_improve_count = 0

        for epoch in range(self.args.num_epochs):
            self.current_epoch = epoch

            # 更新训练阶段
            self._update_training_stage(epoch)

            self._call_callbacks("on_epoch_begin", epoch=epoch)

            # 训练一个epoch
            train_metrics = self._train_epoch()

            # 每个epoch末在dev集上验证（若有）
            eval_metrics = {}
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()

            self._call_callbacks(
                "on_epoch_end", epoch=epoch,
                train_metrics=train_metrics, eval_metrics=eval_metrics
            )

            # 确定当前epoch的监控指标（优先使用dev集Recall@5）
            current_metric = eval_metrics.get('eval_recall@5', 0.0)
            metric_source = "dev_recall@5" if eval_metrics else "train_loss"

            # 跳过NaN的epoch，不更新best_metric
            if math.isnan(current_metric) or math.isinf(current_metric):
                no_improve_count += 1
                logger.warning(
                    f"Epoch {epoch}: 监控指标为NaN/Inf，跳过最佳模型更新 "
                    f"(EarlyStopping counter: {no_improve_count})"
                )
            elif current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                no_improve_count = 0

                best_dir = Path(self.args.output_dir) / 'best_model'
                # 保存checkpoint（含optimizer状态，可恢复训练）
                self.save_checkpoint(str(best_dir / 'checkpoint.pt'))
                # 同时保存BiEncoder pretrained格式（可直接用于推理/索引构建）
                if hasattr(self.model, 'save_pretrained'):
                    self.model.save_pretrained(str(best_dir))
                logger.info(
                    f"  ✓ 新的最佳模型已保存 (epoch={epoch}, "
                    f"{metric_source}={current_metric:.4f})"
                )
            else:
                no_improve_count += 1

            logger.info(
                f"Epoch {epoch}/{self.args.num_epochs}: "
                f"阶段{self.current_stage}, "
                f"train_loss={train_metrics.get('loss', 0):.4f}, "
                f"eval_loss={eval_metrics.get('eval_loss', 'N/A')}, "
                f"best={best_metric:.4f}@epoch{best_epoch}"
            )

            # 检查早停（来自 EarlyStoppingCallback 或其他回调）
            should_stop = any(
                getattr(cb, 'should_stop', False) for cb in self.callbacks
            )
            if should_stop:
                logger.info(
                    f"早停触发！在 epoch {epoch} 停止训练，"
                    f"最佳模型来自 epoch {best_epoch}"
                )
                break

        self._call_callbacks("on_train_end")

        logger.info(
            f"训练完成。最佳模型: epoch={best_epoch}, "
            f"{metric_source}={best_metric:.4f}, "
            f"保存位置: {Path(self.args.output_dir) / 'best_model'}"
        )

        return {
            "final_epoch": self.current_epoch,
            "best_epoch": best_epoch,
            "best_recall@5": best_metric,
        }

    def _train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        返回:
            epoch训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        total_infonce = 0.0
        total_distance = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            self._call_callbacks("on_batch_begin", batch_idx=batch_idx)

            loss, metrics = self._train_step(batch)
            if not math.isnan(loss):
                total_loss += loss
                total_infonce += metrics.get('infonce', 0)
                num_batches += 1
            # distance不受NaN影响（仅依赖cos_sim），始终累加
            total_distance += metrics.get('distance', 0)

            self._call_callbacks(
                "on_batch_end", batch_idx=batch_idx,
                loss=loss, metrics=metrics
            )

            # 日志输出
            if self.global_step % self.args.logging_steps == 0:
                logger.info(
                    f"  Step {self.global_step}: loss={loss:.4f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )

            self.global_step += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "infonce": total_infonce / n,
            "distance": total_distance / n,
        }

    def _train_step(self, batch: Dict[str, Any]) -> tuple:
        """
        单步训练

        流程:
        1. 将batch移动到设备
        2. 编码query和document获取向量
        3. 计算联合损失
        4. NaN检测：若loss为NaN，跳过该step的梯度更新
        5. 梯度裁剪 + 反向传播
        6. 更新参数

        参数:
            batch: 包含 query/pos_doc/neg_doc 的输入张量字典

        返回:
            (loss值, 指标字典)
        """
        # 移动到设备
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 前向传播
        use_fp16 = self.args.fp16 and self.device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=use_fp16):
            # 编码query和正例文档
            query_emb = self.model.encode_query(
                input_ids=batch['query_input_ids'],
                attention_mask=batch['query_attention_mask']
            )
            pos_doc_emb = self.model.encode_document(
                input_ids=batch['pos_doc_input_ids'],
                attention_mask=batch['pos_doc_attention_mask']
            )

            # 编码负例文档（如果存在）
            neg_doc_embs = None
            if 'neg_doc_input_ids' in batch:
                neg_ids = batch['neg_doc_input_ids']
                neg_mask = batch['neg_doc_attention_mask']
                batch_size, num_negs, seq_len = neg_ids.shape

                # 展平编码再还原
                neg_ids_flat = neg_ids.view(-1, seq_len)
                neg_mask_flat = neg_mask.view(-1, seq_len)

                neg_doc_embs_flat = self.model.encode_document(
                    input_ids=neg_ids_flat,
                    attention_mask=neg_mask_flat
                )
                neg_doc_embs = neg_doc_embs_flat.view(batch_size, num_negs, -1)

            # 计算损失
            loss, loss_dict = self.loss_fn(query_emb, pos_doc_emb, neg_doc_embs)

        # NaN检测：若loss为NaN/Inf，跳过该step的梯度更新，防止污染模型权重
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"Step {self.global_step}: 检测到loss={loss.item()}，"
                f"跳过该step的梯度更新"
            )
            self.optimizer.zero_grad()
            return float('nan'), loss_dict

        # 梯度缩放（梯度累积）
        loss = loss / self.args.gradient_accumulation_steps

        # 反向传播
        if self.scaler is not None:
            self.scaler.scale(loss).backward()

            if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()

            if (self.global_step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item() * self.args.gradient_accumulation_steps, loss_dict

    def _update_training_stage(self, epoch: int) -> None:
        """
        根据epoch更新三阶段训练的当前阶段

        阶段划分（三阶段只改变负例来源，w全程恒定）：
        - 阶段1: epoch < stage1_epochs（In-Batch负例）
        - 阶段2: stage1_epochs <= epoch < stage1_epochs + stage2_epochs（BM25难负例）
        - 阶段3: epoch >= stage1_epochs + stage2_epochs（模型难负例）

        Stage 2→3 过渡时自动执行难负例挖掘：
        1. 保存 Stage 2 模型
        2. 使用当前模型编码训练数据文档池，挖掘模型难负例
        3. 将挖掘结果加载为新训练数据
        """
        if not self.args.enable_three_stage:
            return

        s1 = self.args.stage1_epochs
        s2 = s1 + self.args.stage2_epochs

        if epoch < s1:
            new_stage = 1
        elif epoch < s2:
            new_stage = 2
        else:
            new_stage = 3

        if new_stage != self.current_stage:
            logger.info(f"训练阶段切换: {self.current_stage} -> {new_stage} (epoch={epoch})")

            # Stage 2→3 过渡：自动挖掘模型难负例
            if self.current_stage == 2 and new_stage == 3:
                self._run_stage3_mining()

            self.current_stage = new_stage

            # 切换数据加载器阶段（负例来源切换）
            if hasattr(self.train_dataloader, 'set_stage'):
                self.train_dataloader.set_stage(new_stage)

    def _run_stage3_mining(self) -> None:
        """
        Stage 2→3 过渡时自动执行难负例挖掘

        流程：
        1. 保存 Stage 2 模型到 {output_dir}/stage2_model/
        2. 对 NQ 和 TriviaQA 训练数据分别执行挖掘
        3. 将挖掘结果加载到 train_dataloader 中
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from scripts.mine_hard_negatives import run_mining

        logger.info("=" * 60)
        logger.info("Stage 2→3 过渡：开始自动挖掘模型难负例")
        logger.info("=" * 60)

        mining_start = time.time()

        # 1. 保存 Stage 2 模型
        stage2_dir = Path(self.args.output_dir) / "stage2_model"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(str(stage2_dir))
        logger.info(f"Stage 2 模型已保存: {stage2_dir}")

        # 2. 切换到 eval 模式执行挖掘
        self.model.eval()

        if self.tokenizer is None:
            # 回退：从 train_dataloader 获取 tokenizer
            self.tokenizer = getattr(self.train_dataloader, 'tokenizer', None)
        if self.tokenizer is None:
            logger.error("无法获取 tokenizer，跳过 Stage 3 挖掘！将继续使用 Stage 2 的 BM25 难负例。")
            self.model.train()
            return

        data_dir = Path(self.args.data_dir)
        mined_dir = Path(self.args.output_dir) / "mined_negatives"
        mined_dir.mkdir(parents=True, exist_ok=True)

        # 定位训练数据文件
        train_files = []
        nq_train = data_dir / "NQ" / "nq-train.json"
        trivia_train = data_dir / "TriviaQA" / "trivia-train.json"

        if nq_train.exists():
            train_files.append((str(nq_train), str(mined_dir / "nq-train-mined.json")))
        if trivia_train.exists():
            train_files.append((str(trivia_train), str(mined_dir / "trivia-train-mined.json")))

        if not train_files:
            logger.error(f"未找到训练数据文件: {data_dir}，跳过 Stage 3 挖掘！")
            self.model.train()
            return

        # 3. 对每个训练文件执行挖掘
        mined_outputs = []
        for train_file, output_file in train_files:
            logger.info(f"挖掘: {train_file} -> {output_file}")
            file_start = time.time()
            output_path = run_mining(
                encoder=self.model,
                tokenizer=self.tokenizer,
                train_file=train_file,
                output_file=output_file,
                batch_size=self.args.mining_batch_size,
                max_doc_length=256,
                max_query_length=64,
                top_k=self.args.mining_top_k,
                num_negatives=self.args.mining_num_negatives,
            )
            mined_outputs.append(output_path)
            logger.info(f"  完成，耗时 {time.time() - file_start:.1f} 秒")

        # 4. 加载挖掘结果替换训练数据
        if hasattr(self.train_dataloader, 'reload_with_mined_data'):
            self.train_dataloader.reload_with_mined_data(mined_outputs)

        # 5. 恢复训练模式
        self.model.train()

        total_time = time.time() - mining_start
        logger.info("=" * 60)
        logger.info(f"Stage 3 难负例挖掘完成！总耗时 {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        logger.info(f"挖掘结果: {mined_outputs}")
        logger.info("=" * 60)

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估模型

        返回:
            评估指标字典
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        recall_hits = 0
        recall_total = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # 编码
                query_emb = self.model.encode_query(
                    input_ids=batch['query_input_ids'],
                    attention_mask=batch['query_attention_mask']
                )
                pos_doc_emb = self.model.encode_document(
                    input_ids=batch['pos_doc_input_ids'],
                    attention_mask=batch['pos_doc_attention_mask']
                )

                # 负例（如果有）
                neg_doc_embs = None
                if 'neg_doc_input_ids' in batch:
                    neg_ids = batch['neg_doc_input_ids']
                    neg_mask = batch['neg_doc_attention_mask']
                    bs, nn_, sl = neg_ids.shape
                    neg_doc_embs = self.model.encode_document(
                        input_ids=neg_ids.view(-1, sl),
                        attention_mask=neg_mask.view(-1, sl)
                    ).view(bs, nn_, -1)

                loss, _ = self.loss_fn(query_emb, pos_doc_emb, neg_doc_embs)

                # 跳过NaN的batch
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

                # Recall@5：候选池 = batch内所有pos_doc + neg_docs（如有）
                batch_size = query_emb.size(0)
                # query 与 batch 内所有 pos_doc 的相似度 [B, B]
                sim_matrix = torch.mm(query_emb, pos_doc_emb.t())

                if neg_doc_embs is not None:
                    # query 与自己的 neg_docs 的相似度 [B, N]
                    sim_neg = torch.bmm(
                        query_emb.unsqueeze(1),
                        neg_doc_embs.transpose(1, 2)
                    ).squeeze(1)
                    # 拼接：[B, B+N]，正例位置仍在 [0..B-1] 列
                    sim_matrix = torch.cat([sim_matrix, sim_neg], dim=1)

                # 对每个 query_i，检查 pos_doc_i（列索引=i）是否在 top-5
                k = min(5, sim_matrix.size(1))
                _, top_indices = sim_matrix.topk(k, dim=1)
                for i in range(batch_size):
                    if i in top_indices[i]:
                        recall_hits += 1
                    recall_total += 1

        avg_loss = total_loss / max(num_batches, 1)
        recall_at_5 = recall_hits / max(recall_total, 1)
        logger.info(
            f"验证完成: eval_loss={avg_loss:.4f}, "
            f"eval_recall@5={recall_at_5:.4f} ({recall_hits}/{recall_total})"
        )
        return {"eval_loss": avg_loss, "eval_recall@5": recall_at_5}

    def save_checkpoint(self, path: str) -> None:
        """
        保存训练检查点

        参数:
            path: 保存路径
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "current_stage": self.current_stage,
            "args": self.args.to_dict(),
        }

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        恢复训练检查点

        参数:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.current_stage = checkpoint.get("current_stage", 1)

        logger.info(f"检查点已恢复: {path}, epoch={self.current_epoch}, step={self.global_step}")

    def _call_callbacks(self, event: str, **kwargs) -> None:
        """调用回调函数"""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(self, **kwargs)