# DACL-DR 实验逻辑详解

> 本文以 NQ 数据集为主线，详细讲解从训练到评估的完整实验流程。

---

## 一、整体实验逻辑总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练阶段（Training）                         │
│                                                                     │
│  BERT-base-uncased (backbone)                                       │
│       ↓                                                             │
│  Loss = InfoNCE + w × L_dis    ← 你需要实现的损失函数              │
│       ↓                                                             │
│  三阶段训练策略                 ← 你需要实现的训练pipeline           │
│    Stage1: In-Batch Neg (4ep)                                       │
│    Stage2: BM25 Hard Neg (8ep)                                      │
│    Stage3: Model Hard Neg (12ep)                                    │
│       ↓                                                             │
│  训练完成 → 得到 fine-tuned 编码器                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     编码阶段（Encoding）                            │
│                                                                     │
│  用训练好的编码器，对 21M 段落逐一编码 → 得到 21M 个 768维向量      │
│  序列化存入磁盘（如 .npy 或 FAISS 格式）                           │
│                                                                     │
│  ⚠️ 每个模型（A/B/C/D/w=0.2/0.4/0.6/0.8/1.0）都有不同的编码器    │
│     权重，所以每个模型都需要独立编码一次 21M 段落！                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    索引构建（Index Building）                        │
│                                                                     │
│  用 FAISS 构建 HNSW 索引：                                         │
│    - 参数: M=32, ef_construction=200, L2 距离（向量已归一化，等价于IP排序）│
│    - 输入: 21M 个 768维向量                                        │
│    - 输出: 可查询的 HNSW 索引文件                                  │
│                                                                     │
│  ⚠️ 同样，每个模型的向量不同，索引也要分别构建                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       评估阶段（Evaluation）                        │
│                                                                     │
│  对测试集的每个 query：                                             │
│    1. 用同一编码器编码 query → 768维向量                            │
│    2. 在 HNSW 索引中搜索 top-K 结果                                │
│    3. 与 Ground Truth 对比，计算各项指标                            │
│                                                                     │
│  指标类型：                                                         │
│    - 检索效果: Recall@K, MRR@K, NDCG@K                             │
│    - 表示空间: Alignment, Uniformity, Pos Mean/Var                  │
│    - ANN效率: Visited Nodes, Latency, ef_search敏感度              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、训练阶段详解

### 2.1 你需要实现的核心组件

**（1）损失函数**

```
L_total = L_InfoNCE + w × L_dis
```

- **L_InfoNCE**：标准对比学习损失，softmax over similarities，温度 τ=0.05
- **L_dis**：绝对距离约束项，显式拉近 query 与 positive 的余弦距离
- **w**：混合权重，取值 {0, 0.2, 0.4, 0.6, 0.8, 1.0}

**（2）三阶段训练策略**

```
Stage 1 (In-Batch, 4 epochs):
  负例来源 = batch 内其他 query 的 positive 文档
  损失函数 = L_InfoNCE + w × L_dis（w为实验指定值，全程恒定）
  目的 = 预热模型，学习基本的语义区分能力

Stage 2 (BM25 Hard Neg, 8 epochs):
  负例来源 = 数据集中预处理好的 BM25 难负例
  损失函数 = L_InfoNCE + w × L_dis（w全程恒定）
  目的 = 用词汇匹配的"易混淆"段落增强模型的细粒度判别能力

Stage 3 (Model Hard Neg, 12 epochs):
  负例来源 = 用当前模型从训练数据文档池中挖掘的难负例
  损失函数 = L_InfoNCE + w × L_dis（w全程恒定）
  目的 = 用模型自身认为最难区分的样本，进一步打磨决策边界
```

### 2.2 三阶段负例来源详解（以NQ为例）

**你的 NQ 训练数据（nq-train.json, 58880条）每条记录包含：**

```json
{
  "question": "big little lies season 2 how many episodes",
  "answers": ["seven"],
  "positive_ctxs": [...],          // 9个正例段落（含passage_id）
  "negative_ctxs": [...],          // 50个随机负例（DPR提供）
  "hard_negative_ctxs": [...]      // 92个BM25难负例（DPR用BM25检索提供）
}
```

**各阶段如何使用这些数据：**

| 阶段 | 正例来源 | 负例来源 | 损失函数 | 说明 |
|------|---------|---------|---------|------|
| Stage 1 | `positive_ctxs[0]` | 同一batch中其他query的positive | L_InfoNCE + w × L_dis | 无需额外数据，w全程恒定 |
| Stage 2 | `positive_ctxs[0]` | `hard_negative_ctxs` 中采样1-2个 | L_InfoNCE + w × L_dis | 直接使用现有字段 |
| Stage 3 | `positive_ctxs[0]` | 用当前模型从训练数据文档池中检索得到 | L_InfoNCE + w × L_dis | **需要文档池编码+检索** |

### 2.3 关于DPR预处理数据的论文表述策略

**现实情况：**

数据集是从DPR开源项目获取的，`hard_negative_ctxs` 字段是DPR作者预先用BM25（Lucene/Pyserini）检索挖掘的。`negative_ctxs` 是随机采样的负例。

**论文中的合理表述方式（不需要改论文，现有表述已合适）：**

论文中你的三阶段策略定义为 "In-Batch → BM25 → Model"，这个定义本身是关于**训练策略的设计思想**，而非数据预处理的工程实现。具体来说：

1. **Stage 2 "BM25 Hard Negatives"**：论文描述的是"使用BM25检索得到的难负例进行训练"这一策略思想。至于这些BM25难负例是你自己跑Pyserini生成的，还是使用社区标准数据集中已包含的，这是工程实现细节。学术界的标准做法就是直接使用DPR提供的数据格式（包括BM25难负例），几乎所有DPR后续工作（ANCE、RocketQA、SimANS等）都是如此。

2. **Stage 3 "Model Hard Negatives"**：这一步是你**真正独立完成的工作**——用你自己训练的模型去全库检索挖掘难负例。这是论文中三阶段策略最具技术含量的部分。

3. **论文表述建议**：在实验设置/实现细节中，可以写：
   > "我们遵循DPR的标准数据处理流程。训练数据中的BM25难负例通过Pyserini检索获取。第三阶段的模型动态难负例由第二阶段训练完成的模型在NQ和TriviaQA训练集上检索生成。"

   这样表述既准确又不失诚实。`hard_negative_ctxs`字段本质上就是BM25检索的结果，你说"通过Pyserini检索获取"是对数据来源的准确描述。

**关于预处理代码：**

即使你使用的是DPR提供的数据，你仍然需要编写：
- 数据加载和解析代码（DataLoader）
- 训练样本构造逻辑（从JSON字段中组装正例/负例对）
- Stage 3 的NQ和TriviaQA训练集难负例挖掘代码（这是完全独立的工作）
- BM25基线的运行代码（用Pyserini在测试集上跑BM25）
---

## 三、关于语料库编码的核心问题

### 3.1 每个模型都需要重新编码21M段落吗？

**是的，必须重新编码。**

原因很简单：每个模型（A/B/C/D/w=0.2~1.0）的编码器权重不同，同一段文本经过不同编码器会产生不同的768维向量。向量不同 → HNSW图结构不同 → 检索结果和效率指标都不同。

```
模型A的编码器 → "Aaron is a prophet..." → [0.12, -0.34, 0.87, ...]  // 向量A
模型D的编码器 → "Aaron is a prophet..." → [0.08, -0.41, 0.92, ...]  // 向量D（不同！）
```

### 3.2 编码一次后可以复用吗？

**对于同一个模型：可以。**

对于某个特定模型（比如 w=0.6），21M段落只需要编码一次，序列化到磁盘后，后续所有评估实验（不同ef_search、不同K值、计算各种指标）都可以复用这份向量，无需重新编码。

```
w=0.6模型编码21M段落 → 保存为 embeddings_w06.npy（约60GB）
  ├→ 构建HNSW索引 → hnsw_w06.index
  ├→ 评估 Recall@10/100（不同ef_search）    ← 复用同一索引
  ├→ 评估 Latency                           ← 复用同一索引
  ├→ 评估 Visited Nodes                     ← 复用同一索引
  └→ 计算 Alignment/Uniformity（取子集）    ← 复用向量文件
```

### 3.3 能否先构建一个通用向量数据库给所有实验用？

**不能。**

因为每个模型的向量空间不同，所以不存在"一个通用向量数据库"。但是可以优化流程：

```
优化策略：编码和索引构建采用流水线方式
  模型A训完 → 立即编码21M段落 → 存盘 → 构建索引 → 存盘 → 跑全部评估
  模型B训完 → 立即编码21M段落 → 存盘 → 构建索引 → 存盘 → 跑全部评估
  ...
```

### 3.4 三阶段难负例挖掘需要重新编码全库吗？

**Stage 3 需要一次文档池编码，但只在 Stage 2→Stage 3 的过渡点做一次。**

与编码全部 21M 段落的传统方案不同，我们采用更高效的**训练数据文档池挖掘**策略：
收集训练数据（NQ + TriviaQA）中所有文档（positive_ctxs + hard_negative_ctxs + negative_ctxs），
按 passage_id 去重后构建候选池（约 2~5M 篇文档），用 Stage 2 模型编码后进行检索挖掘。

详细流程如下：

```
Stage 1 (In-Batch, 4 epochs)
  - 不需要文档池编码
  - 负例就是batch内其他query的正例文档
  - 损失函数: L_InfoNCE + w × L_dis（w恒定）
  - 纯GPU训练，很快

       ↓

Stage 2 (BM25 Hard Neg, 8 epochs)
  - 不需要文档池编码
  - 负例直接从 hard_negative_ctxs 字段读取（DPR已提供）
  - 每个query采样1-2个BM25难负例参与训练
  - 损失函数: L_InfoNCE + w × L_dis（w恒定）
  - 纯GPU训练

       ↓  ← 这里需要做一次文档池编码！

  ┌───────────────────────────────────────────────────────────┐
  │ 难负例挖掘步骤（Mining，一次性操作）：                      │
  │                                                            │
  │ 1. 收集训练数据中所有文档                                   │
  │    - positive_ctxs + hard_negative_ctxs + negative_ctxs    │
  │    - 按 passage_id 去重 → 约 2~5M 篇唯一文档              │
  │ 2. 用Stage2训练完的模型编码全部文档池                       │
  │    → ~30-60分钟，得到临时向量文件                           │
  │ 3. 构建临时FAISS Flat索引（精确检索）                       │
  │    → ~10分钟                                               │
  │ 4. 对每个训练query：                                        │
  │    a. 用同一模型编码query                                   │
  │    b. 在Flat索引中检索top-200                               │
  │    c. 去除已知正例（positive_ctxs的passage_id）             │
  │    d. 保留top-50作为Model Hard Negatives                    │
  │ 5. 将挖掘结果保存为新的训练数据文件                         │
  │    → nq-train-mined.json / trivia-train-mined.json          │
  │ 6. 可以删除临时文件(embeddings_temp, flat_index_temp)       │
  └───────────────────────────────────────────────────────────┘

       ↓

Stage 3 (Model Hard Neg, 12 epochs)
  - 不需要再次编码
  - 负例使用上一步挖掘好的 Model Hard Negatives
  - 挖掘结果是固定的，整个Stage 3期间不再更新
  - 损失函数: L_InfoNCE + w × L_dis（w恒定）
  - 纯GPU训练
```

**关键澄清：**
1. **Stage 3 的难负例只挖掘一次**，不是每个epoch都挖掘。这是学术界的标准做法（DPR、ANCE、SimANS等）。
2. **w 在三个阶段保持恒定**。三阶段改变的只是负例来源，不改变损失函数的权重配置。
3. **文档池来源于训练数据**，而非全量 21M 语料库。这样每个 w 模型的挖掘时间从 ~4小时降低到 ~30-60分钟。

### 3.5 编码时间和存储空间估算

```
Stage 3 难负例挖掘 — 编码训练数据文档池（~2-5M篇）× 768维 float32:
  - 时间: ~30-60分钟（A6000, batch_size=256）
  - 存储: 约 3-15 GB（临时文件，挖掘完可删除）
  - 每个 w 模型都需独立挖掘一次（因为 Stage 2 模型不同）

最终评估阶段 — 编码21M段落 × 768维 float32:
  - 时间: ~4小时（A6000, batch_size=256）
  - 存储: 21,015,324 × 768 × 4 bytes ≈ 60.4 GB

HNSW索引 (M=32):
  - 构建时间: ~1.5小时
  - 存储: ~65 GB（向量 + 图结构，FAISS将向量内嵌在索引中）

全部8个模型的总存储需求:
  - 8 × ~65GB索引 ≈ 520 GB（加上checkpoint和数据集，总计约535GB）
  
优化策略:
  - 向量可以用float16存储，空间减半: 21M × 768 × 2 ≈ 30 GB
  - FAISS支持PQ压缩索引（论文中也提到了），可大幅降低索引体积
  - 非核心模型（w=0.2, 0.8, 1.0）评估完后可删除向量，只保留结果
```

---

## 四、评估阶段详解（以NQ测试集为例）

### 4.1 评估数据准备

```
NQ测试集 (nq-test.csv): 3610条
  每条: question \t answers（用\t分隔）
  例如: "who got the first nobel prize in physics" \t "['Wilhelm Conrad Röntgen']"

评估时:
  1. 编码3610个query → 3610个768维向量
  2. 在HNSW索引中搜索 → 每个query得到top-K段落的passage_id
  3. 判断返回的段落是否包含answers中的任一答案（字符串匹配）
     → 这就是Ground Truth的判断逻辑
```

### 4.2 各指标的计算方式

#### （1）Recall@K
```
对每个query:
  retrieved = HNSW搜索返回的top-K个passage_id
  relevant = 包含answer的段落集合（Ground Truth）
  hit = 1 if any(answer in passage.text for passage in retrieved) else 0

Recall@K = sum(hit_i) / len(queries)

实际操作:
  - 在HNSW索引中搜索时，设定ef_search和返回K
  - 对返回的passage_id，查找对应的原文
  - 用字符串匹配检查原文是否包含answer
  - 注意：DPR的标准做法是"答案字符串是否在段落文本中出现"（has_answer检查）
```

#### （2）MRR@K (Mean Reciprocal Rank)
```
对每个query:
  rank = 第一个包含answer的段落在top-K中的排名（1-indexed）
  reciprocal_rank = 1/rank if found else 0

MRR@K = mean(reciprocal_rank_i)
```

#### （3）NDCG@K
```
对每个query:
  对top-K中每个位置i:
    rel_i = 1 if passage_i contains answer else 0
  DCG@K = sum(rel_i / log2(i+1) for i in 1..K)
  IDCG@K = ideal DCG（所有相关结果排在最前面时的DCG）

NDCG@K = DCG@K / IDCG@K
```

#### （4）Alignment 和 Uniformity（表示空间指标）
```
这两个指标不需要HNSW索引，直接用向量计算：

准备：从开发集采样10K对 (query, positive_doc)

Alignment = E[||q - d+||²]    // 正样本对的平均欧氏距离平方
  → 越低越好，表示正样本对越接近

Uniformity = log E[exp(-2||x - y||²)]    // 随机样本对的距离
  → 越低越好，表示分布越均匀

Pos Mean = mean(cos_sim(q, d+))    // 正样本余弦相似度均值
Pos Var  = var(cos_sim(q, d+))     // 正样本余弦相似度方差
Neg Mean = mean(cos_sim(q, d-))    // 负样本余弦相似度均值
Neg Var  = var(cos_sim(q, d-))     // 负样本余弦相似度方差
```

#### （5）ANN效率指标
```
这些指标需要HNSW索引，在不同ef_search设置下测量：

ef_search ∈ {16, 32, 64, 128, 256, 512}

对每个ef_search值，对每个test query:
  1. 设置 index.hnsw.efSearch = ef_search
  2. 记录搜索开始时间
  3. 执行搜索: distances, indices = index.search(query_vec, K)
  4. 记录搜索结束时间
  
  Latency = end_time - start_time (ms)

Visited Nodes:
  FAISS的HNSW实现中可以获取搜索过程中实际访问的节点数
  需要对FAISS做二次封装或hook（你代码中的faiss_wrapper.py就是干这个的）
  
  FAISS C++层有 hnsw_stats 可以记录 visited nodes，利用这个获取，但是注意该值为全局计数，所以每个查询获取到visited nodes以后，需要清零。
 
最终整理成表格:
  | ef_search | w=0 Recall | w=0 Latency | w=0 Visited | w=0.6 Recall | ... |
```

### 4.3 评估流程的完整Pipeline

```
对于每个训练好的模型M:

Step 1: 编码段落库（~4小时）
  for batch in psgs_w100.tsv (batch_size=512):
      embeddings = model_M.encode(batch.texts)
      save_to_disk(embeddings)
  
Step 2: 构建HNSW索引（~1-2小时）
  index = faiss.IndexHNSWFlat(768, M=32)
  index.hnsw.efConstruction = 200
  index.metric_type = faiss.METRIC_INNER_PRODUCT
  index.add(all_embeddings)
  faiss.write_index(index, f"hnsw_{model_name}.index")

Step 3: 编码测试query
  query_embeddings = model_M.encode(test_queries)  # ~几秒

Step 4: 多ef_search评估
  for ef in [16, 32, 64, 128, 256, 512]:
      index.hnsw.efSearch = ef
      for query_vec in query_embeddings:
          start = time.time()
          D, I = index.search(query_vec, K=100)
          latency = time.time() - start
          
          # 记录: retrieved passage_ids, latency, visited_nodes
      
      # 计算该ef_search下的 Recall@10, Recall@100, MRR@10, NDCG@10, 
      #                       avg_latency, avg_visited_nodes

Step 5: 表示空间分析（可并行）
  # 用开发集的query-positive对计算
  query_vecs = model_M.encode(dev_queries)
  pos_vecs = model_M.encode(dev_positive_docs)
  
  alignment = mean(||query_vecs - pos_vecs||²)
  uniformity = log(mean(exp(-2*||x_i - x_j||²)))
  pos_mean = mean(cosine_sim(query_vecs, pos_vecs))
  pos_var = var(cosine_sim(query_vecs, pos_vecs))
```

---

## 五、FAISS索引构建细节

### 5.1 为什么用HNSW而不是Flat

```
Flat索引（精确搜索）:
  - 暴力计算query与所有候选向量的距离
  - 准确率100%，但速度较慢
  - 用途：Stage 3难负例挖掘时使用（对训练数据文档池进行精确检索，保证挖掘质量）

HNSW索引（近似搜索）:
  - 构建图结构，贪心导航搜索
  - 速度快（~毫秒级/query），但有精度损失
  - 用途：最终评估实验中使用（模拟真实部署场景）
  - 这正是论文的核心：你的方法让HNSW在更少访问节点下达到更高召回率
```

### 5.2 索引类型选择

```python
# 论文中的主实验：HNSW Flat（无压缩，768维float32）
# 使用默认L2距离——因向量已L2归一化：||a-b||² = 2(1-⟨a,b⟩)
# L2距离排序与内积/余弦相似度排序完全等价，无需显式设置METRIC_INNER_PRODUCT
index = faiss.IndexHNSWFlat(768, 32)  # dim=768, M=32
index.hnsw.efConstruction = 200

```

### 5.3 关于 Visited Nodes 的获取

```python
# FAISS Python接口中获取visited nodes的标准方法：
# 使用 faiss.cvar.hnsw_stats（需要 FAISS >= 1.9.0，包含 #3309 修复）
# 注意：hnsw_stats 仅在单线程搜索时准确（FAISS官方文档要求）

import faiss

# 临时切换为单线程以确保统计准确
faiss.omp_set_num_threads(1)
faiss.cvar.hnsw_stats.reset()
D, I = index.search(query_vec, k)
visited = faiss.cvar.hnsw_stats.ndis  # 距离计算次数 ≈ 访问节点数

# 你的代码中 scripts/run_evaluation.py 的 search_index() 函数已封装此逻辑
```

---

## 六、推荐的实验执行顺序（结合时间优化）

```
Week 1: 基础框架 + 快速验证
  ├─ Day 1-2: 实现DataLoader、BiEncoder、InfoNCE+L_dis损失
  ├─ Day 3:   训练模型A（单阶段, ~3h）+ 编码(~4h) + 评估
  ├─ Day 4:   训练模型B（单阶段+L_dis, ~3h）+ 编码 + 评估
  └─ Day 5:   对比A和B的结果，验证L_dis是否有效（消融实验的一半）

Week 2: 三阶段训练
  ├─ Day 1-2: 实现三阶段训练pipeline
  ├─ Day 3-4: 训练模型C/w=0（三阶段, ~12h）+ 编码 + 评估
  ├─ Day 4:   Stage 3难负例挖掘代码实现 + 验证
  └─ Day 5:   开始训练w=0.6（最重要的模型）

Week 3: 批量实验 + 可视化
  ├─ Day 1-3: 依次训练 w=0.4, w=0.8, w=0.2, w=1.0
  ├─ Day 4:   绘制所有图表
  └─ Day 5:   t-SNE可视化 + 结果整理
```

---

## 七、关键技术决策总结

| 问题 | 答案 |
|------|------|
| 每个模型都要重新编码21M段落吗？ | **是的**，每个模型权重不同，向量不同 |
| 编码后的向量可以复用吗？ | **同一模型内可以**，不同模型间不行 |
| 能否预先构建一个通用向量数据库？ | **不能**，每个模型需要独立的索引 |
| Stage 3难负例需要每epoch挖掘吗？ | **不需要**，只在Stage 2→3过渡时挖掘一次 |
| BM25难负例需要自己生成吗？ | **不需要**，DPR数据已包含（`hard_negative_ctxs`字段） |
| 论文中能说"使用DPR的数据"吗？ | 可以说"遵循DPR标准数据处理流程"，这是学术界惯例 |
| 存储需求多大？ | 每个模型索引约65GB，8个模型索引约520GB，总计（含数据集+checkpoint）约535GB |
| 能否用float16节省空间？ | **可以**，编码后存float16，加载到FAISS时再转float32 |
| Stage 3挖掘范围和方法？ | **训练数据文档池**（2~5M篇，非全量21M），使用**Flat索引**精确检索 |

---

## 八、NQ数据集完整实验流程示例

以一个完整的模型（w=0.6, 三阶段训练）为例：

```
=== 阶段一：Stage 1 训练 (In-Batch Negatives, 4 epochs) ===

输入：nq-train.json, 58880条
每条取：question + positive_ctxs[0]
负例：batch内其他query的positive（batch_size=128 → 127个负例）
损失：L_InfoNCE + 0.6 × L_dis

训练4个epoch → 保存checkpoint_stage1.pt

=== 阶段二：Stage 2 训练 (BM25 Hard Negatives, 8 epochs) ===

输入：nq-train.json, 58880条
每条取：question + positive_ctxs[0] + hard_negative_ctxs中采样1-2个
负例组成：1-2个BM25难负例 + batch内其他query的positive（混合）
损失：L_InfoNCE + 0.6 × L_dis

从checkpoint_stage1.pt继续训练8个epoch → 保存checkpoint_stage2.pt

=== 过渡：Model Hard Negative Mining ===

用checkpoint_stage2.pt的编码器：
  1. 收集训练数据中所有文档（positive_ctxs + hard_negative_ctxs + negative_ctxs）
     → 按 passage_id 去重，得到约 2~5M 篇唯一文档
  2. 编码全部文档池 → embeddings_temp.npy (~30-60分钟)
  3. 构建Flat索引 → flat_index_temp
  4. 对全部训练query (NQ 58880 + TriviaQA):
     - 编码query
     - Flat索引检索top-200
     - 去除positive_ctxs中的passage_id
     - 保留top-50作为model hard negatives
  5. 保存：nq-train-mined.json / trivia-train-mined.json
  6. 可以删除临时文件(embeddings_temp, flat_index_temp)

=== 阶段三：Stage 3 训练 (Model Hard Negatives, 12 epochs) ===

输入：nq-train-mined.json
每条取：question + positive + mined_hard_negatives中采样1-2个
损失：L_InfoNCE + 0.6 × L_dis

从checkpoint_stage2.pt继续训练12个epoch → 保存checkpoint_final.pt

=== 评估 ===

用checkpoint_final.pt：
  1. 编码21M段落 → embeddings_w06.npy (序列化到磁盘，约60GB)
  2. 构建HNSW索引 → hnsw_w06.index
  3. 编码3610个NQ测试query
  4. 遍历ef_search={16,32,64,128,256,512}:
     - 搜索top-100
     - 计算Recall@10, Recall@100, MRR@10, NDCG@10
     - 记录avg_latency, avg_visited_nodes
  5. 计算表示空间指标（用开发集6515条）:
     - Pos Mean, Pos Var, Neg Mean, Neg Var
     - Alignment, Uniformity

全部结果填入论文表格。
```

---

## 九、关于 psgs_w100.tsv 语料库的补充说明

```
文件大小：11GB
格式：TSV (id \t text \t title)
行数：~21,015,324 + 1（header）
共享：NQ 和 TriviaQA 使用同一语料库

这意味着：
  - 对于同一个模型，NQ和TriviaQA的评估共享同一份段落向量和索引
  - 编码一次、建索引一次，NQ和TriviaQA都能用
  - 区别仅在于查询集不同：NQ用nq-test.csv的query，TriviaQA用trivia-test.csv的query
```