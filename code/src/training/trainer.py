"""
训练器主类

本模块实现完整的训练流程，包括：
- 三阶段训练策略（预热→距离引入→联合优化）
- AdamW优化器（BERT层与其他层可设不同学习率）
- 梯度累积与梯度裁剪
- FP16混合精度训练
- 检查点保存与恢复

论文章节：第4章 4.3节 - 训练流程
"""

import logging
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
        loss_fn: 损失函数（CombinedLoss 或 ScheduledCombinedLoss）
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
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[TrainingCallback]] = None
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss_fn = loss_fn
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
          基于 dev_loss 保存最佳模型权重，防止过拟合。
        - 若未提供 eval_dataloader，则退化为基于 train_loss 保存。
        - 最佳模型同时保存为 checkpoint（含optimizer状态，可恢复训练）
          和 BiEncoder pretrained 格式（可直接加载用于推理）。

        返回:
            训练结果统计字典
        """
        logger.info(f"开始训练，共 {self.args.num_epochs} 个epoch")
        logger.info(f"设备: {self.device}")
        if self.eval_dataloader is not None:
            logger.info("已检测到验证集，将基于 dev_loss 保存最佳模型（防过拟合）")
        else:
            logger.warning(
                "⚠️ 未提供验证集(eval_dataloader=None)，"
                "将基于 train_loss 保存最佳模型。"
                "强烈建议传入 dev 集以获得科学的模型选择。"
            )

        self._call_callbacks("on_train_begin")

        best_metric = float('inf')
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

            # 确定当前epoch的监控指标（优先使用dev_loss）
            current_metric = eval_metrics.get('eval_loss', train_metrics.get('loss', 0))
            metric_source = "dev_loss" if eval_metrics else "train_loss"

            # 基于指标保存最佳模型
            if current_metric < best_metric:
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
            "best_eval_loss": best_metric,
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
            total_loss += loss
            total_infonce += metrics.get('infonce', 0)
            total_distance += metrics.get('distance', 0)
            num_batches += 1

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
        4. 梯度裁剪 + 反向传播
        5. 更新参数

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

        # 更新ScheduledCombinedLoss的权重步数（如果使用）
        if hasattr(self.loss_fn, 'step'):
            self.loss_fn.step()

        return loss.item() * self.args.gradient_accumulation_steps, loss_dict

    def _update_training_stage(self, epoch: int) -> None:
        """
        根据epoch更新三阶段训练的当前阶段

        阶段划分：
        - 阶段1: epoch < stage1_epochs（纯InfoNCE预热）
        - 阶段2: stage1_epochs <= epoch < stage1_epochs + stage2_epochs（引入距离约束）
        - 阶段3: epoch >= stage1_epochs + stage2_epochs（联合优化+动态难负例）
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
            self.current_stage = new_stage

            # 切换数据加载器阶段
            if hasattr(self.train_dataloader, 'set_stage'):
                self.train_dataloader.set_stage(new_stage)

            # 阶段1: 距离权重为0
            if new_stage == 1 and hasattr(self.loss_fn, 'set_distance_weight'):
                self.loss_fn.set_distance_weight(0.0)
            # 阶段2/3: 使用配置的距离权重
            elif new_stage >= 2 and hasattr(self.loss_fn, 'set_distance_weight'):
                self.loss_fn.set_distance_weight(self.args.distance_weight)

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
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"验证完成: eval_loss={avg_loss:.4f}")
        return {"eval_loss": avg_loss}

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