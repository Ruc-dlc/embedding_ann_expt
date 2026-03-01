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


## 三、评估阶段详解（以NQ测试集为例）

### 3.1 评估数据准备

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

### 3.2 各指标的计算方式

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
  FAISS C++层有 hnsw_stats 可以记录 visited nodes，利用这个获取，但是注意该值为全局计数，所以每个查询获取到visited nodes以后，需要清零。
 
最终整理成表格:
  | ef_search | w=0 Recall | w=0 Latency | w=0 Visited | w=0.6 Recall | ... |
```

### 3.3 评估流程的完整Pipeline

```
对于每个训练好的模型M:

Step 1: 编码段落库
  for batch in psgs_w100.tsv (batch_size=512):
      embeddings = model_M.encode(batch.texts)
      save_to_disk(embeddings)
  
Step 2: 构建HNSW索引
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



## 五、NQ数据集完整实验流程示例

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

## 六、关于 psgs_w100.tsv 语料库的补充说明

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
