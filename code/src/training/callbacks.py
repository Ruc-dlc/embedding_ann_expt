"""
训练回调（保存、验证等）

本模块实现训练过程中的回调机制，用于：
- 检查点保存
- 模型验证
- 日志记录
- 早停

论文章节：第5章 - 实验实现
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TrainingCallback(ABC):
    """
    训练回调基类
    
    定义训练过程中各个阶段的回调接口。
    """
    
    def on_train_begin(self, trainer: Any, **kwargs) -> None:
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs) -> None:
        """每个epoch开始时调用"""
        pass
    
    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """每个epoch结束时调用"""
        pass
    
    def on_batch_begin(self, trainer: Any, batch_idx: int, **kwargs) -> None:
        """每个batch开始时调用"""
        pass
    
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        loss: float,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """每个batch结束时调用"""
        pass

class CheckpointCallback(TrainingCallback):
    """
    检查点保存回调
    
    定期保存模型检查点。
    
    Args:
        save_dir (str): 检查点保存目录
        save_steps (int): 每N步保存一次
        save_total_limit (int): 最多保存的检查点数量
        save_best_only (bool): 是否只保存最佳模型
        monitor (str): 监控的指标名称
        mode (str): "min" 或 "max"
    """
    
    def __init__(
        self,
        save_dir: str,
        save_steps: int = 1000,
        save_total_limit: int = 3,
        save_best_only: bool = False,
        monitor: str = "eval_recall@5",
        mode: str = "max"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.saved_checkpoints = []
        
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        loss: float,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """根据步数保存检查点"""
        if trainer.global_step % self.save_steps == 0 and trainer.global_step > 0:
            if not self.save_best_only:
                self._save_checkpoint(trainer)
                
    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """epoch结束时检查是否保存最佳模型"""
        if self.save_best_only:
            current_value = eval_metrics.get(self.monitor, train_metrics.get(self.monitor))
            
            if current_value is not None:
                is_best = (self.mode == "min" and current_value < self.best_value) or \
                         (self.mode == "max" and current_value > self.best_value)
                         
                if is_best:
                    self.best_value = current_value
                    self._save_checkpoint(trainer, is_best=True)
                    
    def _save_checkpoint(self, trainer: Any, is_best: bool = False) -> None:
        """保存检查点"""
        if is_best:
            checkpoint_name = "best_model.pt"
        else:
            checkpoint_name = f"checkpoint_step_{trainer.global_step}.pt"
            
        checkpoint_path = self.save_dir / checkpoint_name
        trainer.save_checkpoint(str(checkpoint_path))
        
        if not is_best:
            self.saved_checkpoints.append(checkpoint_path)
            
            # 删除多余的检查点
            while len(self.saved_checkpoints) > self.save_total_limit:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Deleted old checkpoint: {old_checkpoint}")

class ValidationCallback(TrainingCallback):
    """
    验证回调
    
    定期进行模型验证。
    
    Args:
        eval_steps (int): 每N步进行一次验证
        eval_fn: 自定义评估函数
    """
    
    def __init__(
        self,
        eval_steps: int = 500,
        eval_fn: Optional[callable] = None
    ):
        self.eval_steps = eval_steps
        self.eval_fn = eval_fn
        
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        loss: float,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """根据步数进行验证"""
        if trainer.global_step % self.eval_steps == 0 and trainer.global_step > 0:
            if self.eval_fn is not None:
                eval_metrics = self.eval_fn(trainer.model)
            else:
                eval_metrics = trainer.evaluate()
                
            logger.info(f"Step {trainer.global_step} validation: {eval_metrics}")

class LoggingCallback(TrainingCallback):
    """
    日志记录回调
    
    记录训练过程中的各种指标。
    
    Args:
        logging_steps (int): 每N步记录一次
        log_dir (str): 日志目录
        use_tensorboard (bool): 是否使用TensorBoard
        use_wandb (bool): 是否使用Weights & Biases
    """
    
    def __init__(
        self,
        logging_steps: int = 100,
        log_dir: Optional[str] = None,
        use_tensorboard: bool = False,
        use_wandb: bool = False
    ):
        self.logging_steps = logging_steps
        self.log_dir = Path(log_dir) if log_dir else None
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        self.writer = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        # TODO: 初始化TensorBoard和WandB
        
    def on_train_begin(self, trainer: Any, **kwargs) -> None:
        """训练开始时初始化日志"""
        logger.info("Training started")
        logger.info(f"Training arguments: {trainer.args}")
        
    def on_batch_end(
        self,
        trainer: Any,
        batch_idx: int,
        loss: float,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """记录batch级别的日志"""
        if trainer.global_step % self.logging_steps == 0:
            log_msg = f"Step {trainer.global_step}: loss={loss:.4f}"
            
            for key, value in metrics.items():
                log_msg += f", {key}={value:.4f}"
                
            logger.info(log_msg)
            
            # TODO: 写入TensorBoard/WandB
            
    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """记录epoch级别的日志"""
        logger.info(f"Epoch {epoch} completed")
        logger.info(f"  Train metrics: {train_metrics}")
        logger.info(f"  Eval metrics: {eval_metrics}")
        
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """训练结束时关闭日志"""
        logger.info("Training completed")
        
        if self.writer is not None:
            self.writer.close()

class EarlyStoppingCallback(TrainingCallback):
    """
    早停回调
    
    当指标不再改善时提前停止训练。
    
    Args:
        patience (int): 容忍的无改善epoch数
        monitor (str): 监控的指标
        mode (str): "min" 或 "max"
        min_delta (float): 最小改善量
    """
    
    def __init__(
        self,
        patience: int = 3,
        monitor: str = "eval_recall@5",
        mode: str = "max",
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.should_stop = False
        
    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """检查是否应该早停"""
        current_value = eval_metrics.get(self.monitor, train_metrics.get(self.monitor))
        
        if current_value is None:
            return
            
        if self.mode == "min":
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered")