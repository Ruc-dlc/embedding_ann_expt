"""
日志工具

本模块提供统一的日志配置和管理。
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime

def setup_logger(
    name: str = "embedding_ann",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件名
        log_dir: 日志目录
        format_string: 日志格式字符串
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的handler
    logger.handlers.clear()
    
    # 日志格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file or log_dir:
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"{name}_{timestamp}.log"
                
            log_path = log_dir / log_file
        else:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_logger(name: str = "embedding_ann") -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return logging.getLogger(name)

class TrainingLogger:
    """
    训练日志器
    
    专门用于记录训练过程的日志工具。
    
    Args:
        log_dir (str): 日志目录
        experiment_name (str): 实验名称
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=str(self.log_dir)
        )
        
        self.metrics_history = []
        
    def log_metrics(self, metrics: dict, step: int, prefix: str = "") -> None:
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标前缀
        """
        metrics_with_step = {"step": step, **metrics}
        self.metrics_history.append(metrics_with_step)
        
        # 格式化日志消息
        msg_parts = [f"{prefix}Step {step}:"] if prefix else [f"Step {step}:"]
        for key, value in metrics.items():
            if isinstance(value, float):
                msg_parts.append(f"{key}={value:.4f}")
            else:
                msg_parts.append(f"{key}={value}")
                
        self.logger.info(" ".join(msg_parts))
        
    def log_epoch(self, epoch: int, train_metrics: dict, eval_metrics: dict = None) -> None:
        """
        记录epoch信息
        """
        self.logger.info(f"Epoch {epoch} completed")
        self.logger.info(f"  Train: {train_metrics}")
        if eval_metrics:
            self.logger.info(f"  Eval: {eval_metrics}")
            
    def save_history(self) -> None:
        """
        保存历史记录
        """
        import json
        history_path = self.log_dir / f"{self.experiment_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)