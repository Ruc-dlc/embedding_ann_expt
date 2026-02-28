# DACL-DR: 距离感知对比学习稠密检索

> Distance-Aware Contrastive Learning for HNSW-Friendly Dense Retrieval

本项目为论文《面向HNSW友好的稠密检索距离感知对比学习方法研究》的完整代码实现。

## 项目概述

DACL-DR 在标准对比学习（InfoNCE）的基础上，引入余弦距离约束损失，通过三阶段训练策略优化查询-文档向量的空间分布，使其更适配 HNSW 等近似最近邻（ANN）索引的搜索特性，从而在保持检索精度的同时显著降低访问节点数（Visited Nodes）和查询延迟（Latency）。

### 核心公式

```
L_total = L_InfoNCE + w × L_dis

其中:
  L_InfoNCE = -log(exp(sim(q, d+)/τ) / Σ exp(sim(q, d)/τ))
  L_dis = mean(1 - cos(q, d+))
  w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}（论文最优值 w=0.6）
```

### 三阶段训练策略

| 阶段 | 名称 | Epoch数 | 距离权重 | 负例来源 |
|------|------|---------|---------|---------|
| Stage 1 | 热身 | 4 | w=0 | In-Batch Negatives |
| Stage 2 | 引入距离 | 8 | 0→w（线性增长）| BM25 Hard Negatives |
| Stage 3 | 联合优化 | 12 | w | Model Hard Negatives（动态挖掘） |

## 环境配置

### 前置要求

- Miniconda 或 Anaconda
- Python 3.10+
- NVIDIA GPU（推荐 A6000 48GB 或同级别）
- CUDA Driver ≥ 12.1

### 一键安装

```bash
# 创建环境并安装所有依赖（GPU版 PyTorch + faiss-gpu）
conda env create -f environment.yml

# 激活环境
conda activate dacl-dr
```

## 数据准备

本项目使用 DPR 格式的 NQ（Natural Questions）和 TriviaQA 数据集。

### 数据目录结构

```
data_set/
├── NQ/
│   ├── nq-train.json      # NQ训练集
│   ├── nq-dev.json        # NQ验证集
│   └── nq-test.csv        # NQ测试集
├── TriviaQA/
│   ├── trivia-train.json  # TriviaQA训练集
│   ├── trivia-dev.json    # TriviaQA验证集
│   └── trivia-test.csv    # TriviaQA测试集
└── psgs_w100.tsv          # Wikipedia段落语料库（约21M段落）
```

## 运行实验

### 1. BM25 基线评估

```bash
python scripts/run_bm25_eval.py \
    --corpus_file data_set/psgs_w100.tsv \
    --test_file data_set/NQ/nq-dev.json \
    --output_path results/bm25/ \
    --dataset_name NQ
```

### 2. Baseline训练（纯InfoNCE，单阶段）

```bash
python scripts/train_baseline.py \
    --data_dir data_set \
    --output_dir checkpoints/baseline \
    --batch_size 128 \
    --num_epochs 20 \
    --temperature 0.05 \
    --fp16
```

### 3. 距离感知训练（DACL-DR，三阶段）

```bash
# Stage2/3含7个hard neg，物理batch需设为32，通过梯度累积4步达到有效batch=128
python scripts/train_distance_aware.py \
    --data_dir data_set \
    --output_dir checkpoints/distance_aware \
    --distance_weight 0.6 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --stage1_epochs 4 \
    --stage2_epochs 8 \
    --stage3_epochs 12 \
    --temperature 0.05 \
    --fp16
```

### 4. 混合权重消融（w ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0}）

```bash
python scripts/train_ablation.py \
    --weights 0.0 0.2 0.4 0.6 0.8 1.0 \
    --data_dir data_set/ \
    --output_base checkpoints/ \
    --model_name ./local_model_backbone \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --fp16
```

### 5. 消融实验（ABCD 四组模型）

论文 §5.5 消融实验包含四组模型：

| 模型 | 损失函数 | 距离权重 w | 训练策略 |
|------|---------|-----------|---------|
| A (Baseline) | InfoNCE | 0 | 单阶段（In-Batch） |
| B (+L_dis) | InfoNCE + L_dis | 0.6 | 单阶段（In-Batch） |
| C (+Curriculum) | InfoNCE | 0 | 三阶段 |
| D (Full) | InfoNCE + L_dis | 0.6 | 三阶段 |

```bash
python scripts/run_ablation.py \
    --model_dirs \
        A=checkpoints/baseline/final_model \
        B=checkpoints/model_B/final_model \
        C=checkpoints/ablation_w0.0/final_model \
        D=checkpoints/ablation_w0.6/final_model \
    --index_dirs \
        A=indices/baseline \
        B=indices/model_B \
        C=indices/ablation_w0.0 \
        D=indices/ablation_w0.6 \
    --test_file data_set/NQ/nq-dev.json \
    --output_path results/ablation/ \
    --dataset_name NQ
```

### 6. 索引构建

训练完成后，编码全部语料并构建HNSW索引：

```bash
python scripts/build_index.py \
    --encoder_path checkpoints/distance_aware/final_model \
    --corpus_path data_set/psgs_w100.tsv \
    --output_path indices/distance_aware \
    --index_type hnsw \
    --hnsw_m 32 \
    --hnsw_ef_construction 200
```

### 7. 评估

```bash
python scripts/run_evaluation.py \
    --encoder_path checkpoints/distance_aware/final_model \
    --index_path indices/distance_aware/ \
    --test_file data_set/NQ/nq-dev.json \
    --output_path results/ablation_w0.6/ \
    --k_values 1 5 10 20 50 100 \
    --hnsw_ef_search 100 \
    --experiment_name w0.6_nq
```

### 8. 表示空间分析

```bash
python scripts/run_representation_eval.py \
    --encoder_path checkpoints/distance_aware/final_model \
    --dev_file data_set/NQ/nq-dev.json \
    --output_path results/representation/ \
    --num_samples 10000
```

### 9. ANN效率分析（ef_search 敏感度扫描）

```bash
python scripts/run_ef_sweep.py \
    --encoder_path checkpoints/distance_aware/final_model \
    --index_path indices/distance_aware/ \
    --test_file data_set/NQ/nq-dev.json \
    --output_path results/ef_sweep/ \
    --ef_values 16 32 64 128 256 512 \
    --experiment_name w0.6_nq
```

### 10. 生成论文图表

```bash
python scripts/generate_figures.py \
    --results_dir results \
    --output_dir figures \
    --format pdf
```

## 项目结构

```
code/
├── config/                          # 配置文件
│   ├── config.yaml                  # 主配置
│   └── experiment_configs/          # 实验配置
│       ├── baseline.yaml
│       ├── distance_aware.yaml
│       └── ablation.yaml
├── src/                             # 核心源码
│   ├── models/                      # 模型定义
│   │   ├── bi_encoder.py            # 双塔编码器
│   │   ├── query_encoder.py         # 查询编码器
│   │   ├── doc_encoder.py           # 文档编码器
│   │   └── pooling.py               # 池化层
│   ├── losses/                      # 损失函数
│   │   ├── infonce_loss.py          # InfoNCE损失
│   │   ├── distance_loss.py         # 距离约束损失 L_dis
│   │   └── combined_loss.py         # 组合损失 L_total
│   ├── data/                        # 数据处理
│   │   ├── dataset.py               # NQ/TriviaQA数据集
│   │   ├── dataloader.py            # 三阶段数据加载器
│   │   ├── preprocessor.py          # 文本预处理/Tokenization
│   │   └── hard_negative_miner.py   # 难负例挖掘
│   ├── training/                    # 训练
│   │   ├── trainer.py               # 训练循环
│   │   ├── training_args.py         # 训练参数
│   │   ├── scheduler.py             # 学习率调度
│   │   └── callbacks.py             # 训练回调
│   ├── indexing/                     # 索引
│   │   ├── hnsw_index.py            # HNSW索引
│   │   ├── flat_index.py            # Flat索引
│   │   └── faiss_wrapper.py         # FAISS封装
│   ├── retrieval/                   # 检索
│   │   ├── retriever.py             # 检索器
│   │   └── encoder_service.py       # 编码服务
│   └── utils/                       # 工具
│       ├── io_utils.py              # IO工具
│       ├── logger.py                # 日志
│       ├── metrics.py               # 评估指标
│       └── seed.py                  # 随机种子
├── analysis/                        # 分析模块
│   ├── visualization.py             # 可视化工具（t-SNE、距离分布）
│   ├── distribution_analysis.py     # 向量分布分析
│   └── hnsw_simulator.py            # HNSW搜索效率分析（Visited Nodes、Latency）
├── evaluation/                      # 评估模块
│   ├── comprehensive_eval.py        # 综合评估器
│   ├── efficiency_eval.py           # ANN效率评估
│   └── semantic_eval.py             # 语义评估
├── scripts/                         # 入口脚本
│   ├── train_baseline.py            # Baseline训练
│   ├── train_distance_aware.py      # 距离感知训练
│   ├── train_ablation.py            # 混合权重消融训练
│   ├── build_index.py               # 索引构建
│   ├── run_evaluation.py            # 检索评估
│   ├── run_bm25_eval.py             # BM25基线评估
│   ├── run_representation_eval.py   # 表示空间分析
│   ├── run_ef_sweep.py              # ef_search敏感度扫描
│   ├── run_ablation.py              # 消融实验（ABCD四模型）
│   ├── generate_figures.py          # 论文图表生成（12张图）
│   ├── preprocess_dpr.py            # 数据预处理验证
│   └── preprocess_nq.py             # NQ数据预处理
├── tests/                           # 单元测试
│   ├── test_dataset.py
│   ├── test_evaluation.py
│   ├── test_losses.py
│   └── test_retrieval.py
├── data_set/                        # 数据集
│   ├── NQ/                          # Natural Questions
│   ├── TriviaQA/                    # TriviaQA
│   └── psgs_w100.tsv                # Wikipedia语料库
├── environment.yml                  # conda环境配置
└── README.md                        # 项目说明
```

## 评估指标

| 指标 | 说明 | 论文章节 |
|------|------|---------|
| Recall@K | Top-K结果中包含正确答案的查询比例 | §5.2 |
| MRR@K | Top-K结果中正确答案排名的倒数均值 | §5.2 |
| NDCG@K | 归一化折损累积增益 | §5.2 |
| Pos Mean / Pos Var | 正例余弦相似度均值/方差 | §5.3 |
| Alignment | 正样本对的L2距离均值（↓越好） | §5.3 |
| Uniformity | 向量在超球面上的分布均匀性（↓越好） | §5.3 |
| Visited Nodes | HNSW搜索过程中访问的节点数 | §5.4 |
| Latency | 平均查询延迟（ms） | §5.4 |

## 技术栈

- **backbone模型**: BERT-base-uncased (768维)
- **深度学习框架**: PyTorch 2.1+
- **NLP工具**: HuggingFace Transformers
- **向量检索**: FAISS-GPU (HNSW索引 + Flat索引)
- **数据格式**: DPR格式 (JSON)

## 引用

```bibtex
@thesis{dacl_dr_2026,
  title={面向HNSW友好的稠密检索距离感知对比学习方法研究},
  author={dailongchao},
  year={2026}
}
```