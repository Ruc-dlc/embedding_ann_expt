"""
训练参数定义

本模块定义所有训练相关的超参数和配置选项。

论文章节：第5章 5.1节 - 实验设置
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class TrainingArguments:
    """
    训练参数

    包含训练过程中所有可配置的超参数。
    """

    # 输出配置
    output_dir: str = field(
        default="./output",
        metadata={"help": "模型检查点和日志的输出目录"}
    )
    
    # 训练配置
    num_epochs: int = field(
        default=10,
        metadata={"help": "训练轮数"}
    )
    
    batch_size: int = field(
        default=128,
        metadata={"help": "训练批次大小"}
    )
    
    eval_batch_size: int = field(
        default=256,
        metadata={"help": "评估批次大小"}
    )
    
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "初始学习率"}
    )
    
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减系数"}
    )
    
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "学习率预热步数"}
    )
    
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "预热步数占总步数的比例（与warmup_steps二选一）"}
    )
    
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "梯度累积步数"}
    )
    
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪阈值"}
    )
    
    # 混合精度
    fp16: bool = field(
        default=True,
        metadata={"help": "是否使用FP16混合精度训练"}
    )
    
    # 损失函数配置
    temperature: float = field(
        default=0.05,
        metadata={"help": "InfoNCE温度参数τ"}
    )
    
    distance_weight: float = field(
        default=0.6,
        metadata={"help": "距离约束损失权重w"}
    )
    
    # 数据目录
    data_dir: str = field(
        default="data_set/",
        metadata={"help": "数据集根目录（Trainer在Stage 2→3自动挖掘时需要定位训练JSON）"}
    )
    
    # 难负例配置
    num_negatives: int = field(
        default=7,
        metadata={"help": "每个样本的负例数量"}
    )
    
    mining_batch_size: int = field(
        default=256,
        metadata={"help": "Stage 3 难负例挖掘时的编码批次大小"}
    )
    
    mining_top_k: int = field(
        default=200,
        metadata={"help": "Stage 3 挖掘时每个查询检索的候选数量"}
    )
    
    mining_num_negatives: int = field(
        default=50,
        metadata={"help": "Stage 3 挖掘时每个查询保留的难负例数量"}
    )
    
    # 三阶段训练
    enable_three_stage: bool = field(
        default=True,
        metadata={"help": "是否启用三阶段训练"}
    )
    
    stage1_epochs: int = field(
        default=4,
        metadata={"help": "第一阶段（In-Batch负例）轮数"}
    )
    
    stage2_epochs: int = field(
        default=8,
        metadata={"help": "第二阶段（BM25难负例）轮数"}
    )
    
    stage3_epochs: int = field(
        default=12,
        metadata={"help": "第三阶段（模型难负例）轮数"}
    )
    
    # 保存和日志
    save_steps: int = field(
        default=1000,
        metadata={"help": "每N步保存一次检查点"}
    )
    
    save_total_limit: int = field(
        default=3,
        metadata={"help": "最多保存的检查点数量"}
    )
    
    eval_steps: int = field(
        default=500,
        metadata={"help": "每N步进行一次评估"}
    )
    
    logging_steps: int = field(
        default=100,
        metadata={"help": "每N步记录一次日志"}
    )
    
    # 随机性
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    
    # 数据加载
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "数据加载线程数"}
    )
    
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "是否使用pinned memory"}
    )
    
    def __post_init__(self):
        """参数后处理"""
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 验证参数
        self._validate_args()
        
    def _validate_args(self) -> None:
        """验证参数有效性"""
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 < self.temperature <= 1.0, "temperature must be in (0, 1]"
        assert 0 <= self.distance_weight <= 1.0, "distance_weight must be in [0, 1]"
        
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingArguments":
        """从字典创建"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingArguments":
        """从YAML文件加载"""
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config.get('training', config))