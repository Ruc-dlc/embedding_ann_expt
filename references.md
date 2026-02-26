1.BM25
标题：Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval
来源网址：https://www.researchgate.net/profile/Stephen-Robertson-11/publication/221299140_Some_Simple_Effective_Approximations_to_the_2-Poisson_Model_for_Probabilistic_Weighted_Retrieval/links/0c960534ff2c4a2c5f000000/Some-Simple-Effective-Approximations-to-the-2-Poisson-Model-for-Probabilistic-Weighted-Retrieval.pdf
简介：基于2-Poisson概率模型，提出并验证了几个简单而有效的词频、文档长度和查询词频加权近似公式，显著提升了信息检索性能，为后来的BM25等经典模型奠定了基础。

2.Transformer
标题：Attention Is All You Need
来源网址：https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
简介：提出了一种名为Transformer的全新神经网络架构，完全摒弃了循环和卷积，仅依靠自注意力和多头注意力机制来高效地建模序列数据，在机器翻译任务上取得了突破性成果，并奠定了现代大语言模型的基础。

3.BERT
标题：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
来源网址：https://aclanthology.org/N19-1423.pdf
简介：BERT 通过掩码语言模型和下一句预测任务进行深度双向预训练，仅需微调即可在多种 NLP 任务上取得当时最先进的性能，开创了“预训练+微调”的新范式。

4.TR-Survey
标题：Dense Text Retrieval Based on Pretrained Language Models: A Survey
来源网址：https://dl.acm.org/doi/pdf/10.1145/3637870
简介：中国人民大学2024年的综述文章，系统性地回顾了基于预训练语言模型的稠密文本检索技术，创新性地从架构、训练、索引和集成四大核心维度组织和总结了该领域的研究进展。

5.Sentence-BERT
标题：Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
来源网址：https://aclanthology.org/D19-1410.pdf
简介：通过在孪生/三元组网络结构中微调 BERT，成功生成了高质量且可高效计算的句子嵌入，解决了原生 BERT 因交叉编码器架构而无法用于大规模语义搜索和聚类的难题，在保持高精度的同时实现了数量级的效率提升。

6.DPR
标题：Dense Passage Retrieval for Open-Domain Question Answering
来源网址：https://aclanthology.org/2020.emnlp-main.550.pdf
简介：提出了一种简单而强大的稠密段落检索方法，通过在双编码器架构下巧妙地利用批内负例和难负例进行微调，仅用标准问答数据就大幅超越了传统 BM25 检索器，并推动了端到端开放域问答系统的性能达到新高度。

7.Simcse
标题：Simcse: Simple contrastive learning of sentence embeddings
来源网址：https://aclanthology.org/2021.emnlp-main.552.pdf
简介：提出了一种极其简单的对比学习框架：其无监督版本通过让句子“预测自身”并利用 Dropout 作为噪声来构建正样本对；其有监督版本则巧妙地利用NLI数据集中的蕴含和矛盾关系。该方法在语义文本相似度任务上大幅刷新了记录，并从对齐-均匀性的角度给出了深刻的理论解释。

8.InfoNCE
标题：Representation Learning with Contrastive Predictive Coding
来源网址：https://arxiv.org/pdf/1807.03748
简介：提出了一种通用的无监督表示学习框架，其核心是通过自回归模型在潜在空间中预测未来，并利用对比损失（InfoNCE）来最大化上下文与未来观测间的互信息。该方法在语音、图像、文本和强化学习四大领域均取得了卓越成果，展示了其作为通用表示学习工具的强大能力。

9.HNSW
标题：Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs
来源网址：https://arxiv.org/pdf/1603.09320
简介：HNSW 通过构建一个多层次的可导航小世界图，并结合智能的邻居选择策略，成功地将近似最近邻搜索的复杂度降至对数级别，在各种数据集上都展现出卓越且鲁棒的性能，成为向量近似检索领域的一个里程碑式工作。

10.Faiss
标题：The faiss library
来源网址：https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11202651
简介：一个为向量相似性搜索而生的强大、灵活且高效的开源库，它通过提供丰富的向量压缩和非穷举搜索工具，并辅以完善的工程实践（如超参调优、过滤搜索），使开发者能够根据具体需求构建出最优的近似最近邻搜索解决方案，并已在工业界得到广泛应用和验证。

11.ColBERT
标题：ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
来源网址：https://dl.acm.org/doi/pdf/10.1145/3397271.3401075
简介：通过创新的“后期交互”架构，将 BERT 的强大语义理解能力与高效的检索性能相结合，实现了既能媲美 BERT 的检索效果，又能达到两个数量级以上的速度提升，并支持端到端的全库神经检索。

12.Colbertv2
标题：ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
来源网址：https://aclanthology.org/2022.naacl-main.272.pdf
简介：通过“去噪监督”和“残差压缩”两大创新，一举解决了原始 ColBERT 存储开销大和监督信号弱的问题，在保持与单向量模型相当的存储效率的同时，实现了顶尖的检索效果和卓越的零样本泛化能力。

13.e5
标题：Text embeddings by weakly-supervised contrastive pre-training
来源网址：https://arxiv.org/pdf/2212.03533
简介：通过构建和利用一个名为 CCPairs 的高质量、大规模弱监督文本对数据集，并结合简洁的对比学习框架，实现了在零样本检索上超越 BM25、在微调后超越数十倍大模型的通用文本嵌入性能。

14.C-pack
标题：C-pack: Packed resources for general chinese embeddings
来源网址：https://dl.acm.org/doi/pdf/10.1145/3626772.3657878
简介：是一个开创性的中文通用文本嵌入资源包，它通过发布最大规模的训练数据（C-MTP）、首个全面的评测基准（C-MTEB）、性能领先的模型家族（BGE）以及完整的训练方案，一举填补了该领域的空白，并迅速成为社区事实标准。

15.RocketQA
标题：RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering
来源网址：https://aclanthology.org/2021.naacl-main.466.pdf
简介：通过一套创新的三重训练策略——跨批次负采样、去噪困难负样本和基于交叉编码器的数据增强——系统性地解决了密集检索模型训练中的核心挑战，在多个基准上实现了当时最先进的检索和问答性能。

16.ANCE
标题：APPROXIMATE NEAREST NEIGHBOR NEGATIVE CONTRASTIVE LEARNING FOR DENSE TEXT RETRIEVAL
来源网址：https://arxiv.org/pdf/2007.00808
简介：通过理论证明了局部负样本是密集检索的瓶颈，并创新性地利用异步更新的 ANN 索引从全库采样全局困难负样本，从而在多个检索任务上实现了SOTA性能，并达到了比传统级联IR流水线高百倍的效率。

17.Alignment_Uniformity
标题：Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
来源网址：https://proceedings.mlr.press/v119/wang20k/wang20k.pdf
简介：本文揭示了对比学习成功的本质在于同时优化特征的对齐性（正样本靠近）和均匀性（整体分布均匀），并证明直接优化这两个简单、可量化的几何性质，就能获得媲美甚至超越传统对比学习的高效表示。

18.SimCLR
标题：A Simple Framework for Contrastive Learning of Visual Representations
来源网址：https://proceedings.mlr.press/v119/chen20j/chen20j.pdf
简介：提出一个简单但高效的自监督对比学习框架（SimCLR），用于学习高质量的视觉表征，无需特殊网络结构或记忆库，通过在同一个图像的不同增强视图之间最大化一致性（即拉近正样本对、推开负样本对），在无需标签的情况下学习到高质量的图像特征。

19.MoCo
标题：Momentum Contrast for Unsupervised Visual Representation Learning
来源网址：https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf
简介：通过队列+动量编码器构建大而一致的动态字典，实现了高性能的无监督视觉表征学习，并在多个下游任务上超越有监督预训练，推动自监督学习的发展。

20.hard_negative_theory
标题：CONTRASTIVE LEARNING WITH HARD NEGATIVE SAMPLES
来源网址：https://arxiv.org/pdf/2010.04592
简介：提出了一种简单高效的无监督困难负样本采样方法（HCL），通过一个可调参数控制负样本难度，在图像、图和文本任务上显著提升了对比学习的下游性能，并给出了其最优表征的理论解释。

21.supervised_contrastive
标题：Supervised Contrastive Learning
来源网址：https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf
简介：

22.batch_size_analysis
标题：Why do We Need Large Batchsizes in Contrastive Learning? A Gradient-Bias Perspective
来源网址：https://proceedings.neurips.cc/paper_files/paper/2022/file/db174d373133dcc6bf83bc98e4b681f8-Paper-Conference.pdf
简介：通过将同一类别的所有样本作为正样本进行对比学习，提出了一种优于交叉熵的监督损失函数，在 ImageNet 上刷新了 ResNet 架构的准确率纪录，并展现出更强的鲁棒性和训练稳定性。

23.hard_negative_mining
标题：Optimizing Dense Retrieval Model Training with Hard Negatives
来源网址：https://dl.acm.org/doi/pdf/10.1145/3404835.3462880
简介：理论分析揭示了DR模型不同负采样策略的本质区别和潜在风险，并据此提出了两种高效且有效的训练算法（STAR和ADORE）。其核心贡献在于阐明了动态硬负采样对于直接优化排名性能的重要性，并提供了切实可行的实现方案。

24.negative_cache
标题：Efficient Training of Retrieval Models Using Negative Cache
来源网址：https://proceedings.neurips.cc/paper/2021/file/2175f8c5cd9604f6b1e576b252d4c86e-Paper.pdf
简介：利用了缓存和近似采样的思想来解决大规模检索模型训练中的效率难题。通过允许负样本嵌入存在一定程度的“过时”，并辅以理论支持的小比例刷新策略，该方法在几乎不牺牲模型性能的前提下，极大地降低了训练所需的计算和内存资源。负缓存和流式缓存框架为高效训练大规模双塔检索模型提供了一个简单、通用且强大的新范式。

25.mixed_negative
标题：Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations
来源网址：https://dl.acm.org/doi/pdf/10.1145/3366424.3386195
简介：针对推荐系统中两塔模型训练的关键环节——负采样，指出了常用批次内负采样的选择偏差问题，并创新性地提出了混合负采样（MNS）策略。通过结合批次内负样本和从全库均匀采样的负样本，MNS有效缓解了偏差，提升了模型对长尾和新商品的检索能力。

26.cross_batch_negative
标题：Cross-Batch Negative Sampling for Training Two-Tower Recommenders
来源网址：https://dl.acm.org/doi/pdf/10.1145/3404835.3463032
简介：提出的CBNS方法是一种优雅且实用的技巧，它巧妙地利用了深度学习模型训练过程中的一个自然现象——嵌入稳定性。通过一个简单的FIFO记忆库，CBNS以极低的额外成本，突破了批内负采样对负样本数量的限制，为训练更强大的两塔推荐模型提供了一种高效的途径。

27.NSW
标题：Approximate nearest neighbor algorithm based on navigable small world graphs
来源网址：https://www.researchgate.net/profile/Yu-Malkov/publication/259126397_Approximate_nearest_neighbor_algorithm_based_on_navigable_small_world_graphs/links/63733c302f4bca7fd06030b8/Approximate-nearest-neighbor-algorithm-based-on-navigable-small-world-graphs.pdf
简介：提出的基于可导航小世界图的ANN算法，以其无与伦比的简洁性、通用性和高效性，成为了ANN领域的一个里程碑。它巧妙地利用了数据插入的自然过程来构建高效的索引，避免了复杂的工程设计。其核心思想直接催生了后来更为流行的HNSW算法（通过引入层级结构进一步优化了性能），并深刻影响了包括FAISS、Milvus、Weaviate等在内的众多现代向量数据库和检索系统。

28.NSG
标题：Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph
来源网址：https://arxiv.org/pdf/1707.00143
简介：通过提出MRNG（提供理论基础）和NSG（提供工程实现），成功解决了十亿级ANNS的效率和可扩展性难题。NSG通过固定导航节点、智能生成候选集和严格的边剪枝策略，在保证极快搜索速度的同时，实现了业界领先的低内存占用，是一个兼具理论创新和巨大工程价值的工作。

29.DiskANN
标题：DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
来源网址：https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf
简介：通过创新的Vamana图算法和精心设计的SSD友好型系统架构（混合存储、重叠分区、BeamSearch、隐式重排序），成功地将高性能、高精度的十亿级ANNS带到了单机环境。这项工作不仅在学术上推动了ANNS领域的发展，也为工业界提供了一种成本效益极高的大规模向量检索解决方案，证明了利用现代SSD的潜力可以突破传统内存限制。

30.graph_ann_survey
标题：A Comprehensive Survey and Experimental Comparison of Graph-Based Approximate Nearest Neighbor Search
来源网址：https://arxiv.org/pdf/2101.12631
简介：论文是一篇全面、深入且实验驱动的综述，旨在系统性地梳理、分类、分析和评估过去十年中涌现的众多基于图的近似最近邻搜索（Graph-based ANNS）算法。其核心目标是为研究者和从业者提供一个清晰的路线图和实用的指导原则。


31.graph_search_theory
标题：Graph-based Nearest Neighbor Search: From Practice to Theory
来源网址：https://proceedings.mlr.press/v119/prokhorenkova20a/prokhorenkova20a.pdf
简介：为实践中极为成功的基于图的近似最近邻搜索（Graph-based ANNS）。作者聚焦于低维（dense regime, d ≪ log n）这一特定但重要的场景，并首次从理论上严格分析了实践中广泛使用的两大核心启发式策略：添加长程连接（shortcut edges）和束搜索（beam search）。

32.gpu_similarity
标题：Billion-scale similarity search with GPUs
来源网址：https://arxiv.org/pdf/1702.08734
简介：解决如何高效利用GPU硬件来处理十亿规模的向量相似性搜索问题。

33.debiased_contrastive
标题：Debiased Contrastive Learning
来源网址：https://proceedings.neurips.cc/paper_files/paper/2020/file/63c3ddcc7b23daa1e42dc41f9a44a873-Paper.pdf
简介：标准对比学习中“负样本采样偏差”（sampling bias）。作者提出了一种新颖的、理论驱动的去偏对比学习（Debiased Contrastive Learning）方法，在无需真实标签的情况下纠正这种偏差，并在多个领域取得了SOTA性能。

34.triplet_loss
标题：FaceNet: A Unified Embedding for Face Recognition and Clustering
来源网址：https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf
简介：Google团队发表的论文提出了FaceNet——一个革命性的、端到端的人脸识别系统。其核心思想是直接学习一个将人脸图像映射到紧凑欧几里得空间的嵌入（embedding），使得该空间中的距离可以直接度量人脸的相似度。这一简洁而强大的范式统一了人脸识别、验证和聚类三大任务，并取得了当时SOTA的性能。

35.knowledge_distill
标题：Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation
来源网址：https://arxiv.org/pdf/2010.02666
简介：该论文的核心贡献是提出了一种跨架构知识蒸馏（Cross-Architecture Knowledge Distillation）方法，旨在显著提升高效神经排序模型的效果，同时完全保留其低延迟的效率优势。其关键在于解决不同模型架构输出分数范围不一致的问题。

36.balanced_sampling
标题：Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling
来源网址：https://dl.acm.org/doi/pdf/10.1145/3404835.3462891
简介：提出了一种名为 TAS-Balanced（Topic-Aware Sampling with Balanced Margins）的高效、低成本训练方法，用于训练高性能的稠密检索（Dense Retrieval, DR）模型。该方法使得在单块消费级GPU上48小时内即可训练出SOTA性能的模型，打破了当时稠密检索模型依赖大规模算力的常规。

37.PQ
标题：Product Quantization for Nearest Neighbor Search
来源网址：https://inria.hal.science/inria-00514462/file/jegou_pq_postprint.pdf
简介：乘积量化（Product Quantization, PQ）的创新方法，用于解决高维空间中的近似最近邻（ANN）。其核心思想是通过一种结构化的量化方式，在极低的内存占用和高效的计算速度下，实现高精度的距离估计。该方法尤其适用于十亿级向量的大规模检索场景。

38.IVF
标题：Searching with quantization: approximate nearest neighbor search using short codes and distance estimators
来源网址：https://inria.hal.science/inria-00410767/document
简介：首次系统地提出了使用乘积量化（Product Quantization, PQ）进行高效近似最近邻搜索的核心思想，并引入了关键的非对称距离计算（Asymmetric Distance Computation, ADC）策略。

39.source_coding
标题：SEARCHING IN ONE BILLION VECTORS: RE-RANK WITH SOURCE CODING
来源网址：https://arxiv.org/pdf/1102.3828
简介：提出了一种基于源编码的重排序（Re-ranking）方法，旨在解决在十亿级向量数据库中进行高效近似最近邻搜索时面临的内存瓶颈问题。其核心思想是利用额外的短量化码来精炼初始检索结果，从而避免从磁盘读取完整的高维向量。

40.billion_scale
标题：Efficient Indexing of Billion-Scale datasets of deep descriptors
来源网址：https://openaccess.thecvf.com/content_cvpr_2016/papers/Babenko_Efficient_Indexing_of_CVPR_2016_paper.pdf
简介：揭示了传统索引方法在深度学习特征上的局限性，并提出了两种新的、更适应深度特征分布的索引结构。同时，它还发布了首个十亿级深度特征公开数据集（DEEP1B），为后续研究奠定了基础。

41.MS MARCO
标题：MS MARCO: A Human Generated MAchine Reading COmprehension Dataset
来源网址：https://arxiv.org/pdf/1611.09268
简介：数据集，来自 Bing 搜索引擎的真实匿名用户查询。由人工标注员根据提供的网页片段撰写的自然语言答案（而非直接从文本中抽取）。

42.NQ
标题：Natural Questions: A Benchmark for Question Answering Research
来源网址：https://watermark02.silverchair.com/tacl_a_00276.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA0kwggNFBgkqhkiG9w0BBwagggM2MIIDMgIBADCCAysGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM--uUscgCCIUBhGlMAgEQgIIC_BLRa_9AQJrVYabNQtrwBKODevq0hP5t8HVvdBwJXDhHebi3SUEUYUcoPXafUPXFFgce9_g8Rw99zz-4a7OIlzHW159GLA9Tv1eeTq0W-u9wf-4tRKiUs8CH2t7ThTzVAZkRyQxC2YeIczxpIHUJ-59YpnIQbnUxtIrfxl4HwprvEHNV4CFArfTUYCk6Ty2eZ8jVjAqQNVHF6qrz--O2jRyq9RtkLJHf-43QB0uKHG4yQoSLLhjhv3w5I5lpgBT8sqSMnvHrOrt7GMmV1xcqNRSEHp50S_Uc3vI75fRotj9kuPtacoOP7TKgOJoBArljaoJDKH0vLk2kUKLzwBWkn8YiVUikrWFPAy-rjg6shj7YsZ6mK_Rrvmv8pa_PLmzX210I_YLQNyp1xSsF9gz6K5hnEdmytY5im82_wmV93Qn2Qnpr0SC_5dCaA1ZdVmc7VUc-hUu1GgFztal4K4R6vLn3LfEGEdaOaTY5_sipt9hM5NTWpl75ItpZr85TYgJ2pAhs5B5WnFXPyKUrSMm8WWWSrkt2k862vIzGRBe_SbbFCGvJtrdBN-optSVlJZQU-sWIMgVIQK8w-1BhVMVET9pJ_ySDtKPy2u83LvRvcnc8zV6S4O9g6DXS4VRkKI7IRNaFe_pVj4WmQlL-hh1dNeeumbk3kdRxiqO48ntCn7vekzRQIMIHKD5bQBT9IxoD0SeEBVVE6b-_GiUh72IUnc6oClbo6FAIjFjuDINnHYaeTPlvlAbfcCrB6ml_o9vn9Bunk1wg-aWmVGM-oaK_LJ2UPKTyVyn_DvqGoFpzzCDcxJB42ypRSYpChTfTZrw_Q8IHAQ49BYb71bpxQNlXSfZR80hE2GSDt--rp3dR_ZatuSoXargZWYCjn4GsmBOBtC16qLWDAsGGTDq0_VvXDQnhjr8Mt456h25Y6Ym_TLzgHobDLJSLOcgb4Lctl4036LJPAm-a2ilJ1Vb2jFBPfnzXU4GgBpFMbBTa1NroxxzoN9gwcMBMYC7D289x
简介：数据集，模拟用户在维基百科上寻找信息的自然过程，答案有明确的粒度。

43.TriviaQA
标题：TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
来源网址：https://aclanthology.org/P17-1147.pdf
简介：数据集，基于 trivia 问答网站（琐事问答），答案是明确的实体，上下文来自高质量的Web资源。

44.BEIR
标题：BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models
来源网址：https://arxiv.org/pdf/2104.08663
简介：提出了 BEIR (Benchmarking-IR)，一个异构的、零样本（zero-shot）用于全面评估信息检索（IR）模型的泛化能力。它旨在解决以往研究中模型只在单一、同质数据集上评估的局限性。

45.MTEB
标题：MTEB: Massive Text Embedding Benchmark
来源网址：https://aclanthology.org/2023.eacl-main.148.pdf
简介：提出了 MTEB (Massive Text Embedding Benchmark)，一个大规模、多任务、多语言的文本嵌入（Text Embedding）模型评估基准。其目标是解决当时文本嵌入领域评估标准单一、不全面的问题，并为从业者和研究者提供一个可靠的模型选择和比较平台。

46.DSSM
标题：Learning deep structured semantic models for web search using clickthrough data
来源网址：https://dl.acm.org/doi/pdf/10.1145/2505515.2505665
简介：解决传统关键词匹配（如 TF-IDF、BM25）在词汇不匹配（lexical gap）场景下的局限性（例如用户搜“汽车”，文档用“轿车”），通过学习查询（query）和文档（document）的深层语义表示，提升 Web 搜索排序效果。

47.youtube_dnn
标题：Deep neural networks for youtube recommendations
来源网址：https://dl.acm.org/doi/pdf/10.1145/2959100.2959190?utm_campaign=Weekly+dose+of+Machine+Learning&utm_medium=email&utm_source=Revue+newsletter
简介：通过深度候选生成（基于多分类）与深度排序（基于观看时长加权），构建了首个成功落地的大规模深度学习推荐系统，其“两阶段”架构、特征工程技巧和目标函数设计成为工业界黄金标准。

48.sampling_bias
标题：Sampling-bias-corrected neural modeling for large corpus item recommendations
来源网址：https://dl.acm.org/doi/pdf/10.1145/3298689.3346996
简介：针对工业界大规模推荐系统中普遍存在的批次内负采样偏差问题，提出了一套完整、实用且经过大规模验证的解决方案。其核心在于实时估计物品频率并在校正后的损失函数中进行训练。

49.facebook_search
标题：Embedding-based retrieval in facebook search
来源网址：https://dl.acm.org/doi/pdf/10.1145/3394486.3403305
简介：系统性地阐述了如何在一个超大规模、高度个性化的社交搜索引擎中，从零开始构建并成功部署基于嵌入的检索系统。其核心贡献在于提出了统一嵌入模型来捕捉社交上下文，并设计了高效的混合检索架构来解决工程落地难题。

50.splade
标题：Splade: Sparse lexical and expansion model for first stage ranking
来源网址：https://dl.acm.org/doi/pdf/10.1145/3404835.3463098
简介：过巧妙的对数饱和激活函数和FLOPS稀疏正则化，成功地将大语言模型的语义理解能力与传统倒排索引的高效精确匹配能力结合起来。其模型简单、有效、高效且可解释，为构建下一代大规模检索系统的第一阶段召回模型提供了一个极具吸引力的解决方案。

51.unsupervised_dense
标题：Unsupervised dense information retrieval with contrastive learning
来源网址：https://arxiv.org/pdf/2112.09118
简介：确立了对比学习作为训练无监督稠密检索器的首选方法。提出的 Contriever 模型不仅在零样本设置下挑战了BM25的统治地位，更作为一个强大的预训练模型，为后续的有监督微调、少样本学习和多语言/跨语言检索任务奠定了坚实的基础。

52.latent_retrieval
标题：Latent retrieval for weakly supervised open domain question answering
来源网址：https://aclanthology.org/P19-1612.pdf
简介：开放域问答领域的一个里程碑。它首次证明了仅从问答对中端到端地学习检索是可行且有效的。通过引入逆完形填空任务（ICT）作为预训练策略，成功地解决了从海量语料中进行潜在检索的学习难题。更重要的是，论文揭示了数据集偏差对模型评估的巨大影响，并明确指出在真实场景下，可学习的检索器相比传统IR系统具有压倒性优势。这项工作为后续大量的稠密检索研究（如DPR, Contriever等）奠定了基础。

53.batch_softmax
标题：Learning dense representations of phrases at scale
来源网址：https://aclanthology.org/2021.acl-long.518.pdf
简介：证明了大规模、纯稠密的短语表示是可行的，并且能构建出高性能、高效率的开放域问答系统。通过结合数据增强、知识蒸馏、创新的负采样以及关键的查询端微调策略，DensePhrases 不仅解决了短语检索模型长期存在的性能瓶颈，还展示了其作为通用稠密知识库的巨大潜力。这项工作为高效、可扩展的知识密集型NLP应用开辟了新的道路。

54.clip
标题：Learning transferable visual models from natural language supervision
来源网址：https://proceedings.mlr.press/v139/radford21a/radford21a.pdf
简介：CLIP 的核心是一个多模态对比学习框架，旨在学习一个共享的视觉-语言嵌入空间。直接从互联网规模的原始文本（即图像的标题、描述等），可以作为一种更丰富、更具扩展性的监督信号，从而学习到更通用、更鲁棒的视觉表示。

55.infonce_kernel
标题：Bridging mini-batch and asymptotic analysis in contrastive learning: From infonce to kernel-based losses
来源网址：https://arxiv.org/pdf/2405.18045
简介：揭示了多种对比学习损失背后共同的优化目标——超球面能量最小化（HEM）。 通过解耦对齐与均匀性项，使优化过程更顺畅；引入核方法，提供了一个批次大小无关且非渐近最优解已知的损失函数。

56.quantization_mips
标题：Accelerating large-scale inference with anisotropic vector quantization
来源网址：https://proceedings.mlr.press/v119/guo20h/guo20h.pdf
简介：指出了传统量化方法在MIPS任务中的根本性不足，并提出了一种优雅且高效的解决方案——各向异性量化。其核心思想是让量化过程“关注重点”，即优先保证对最终检索结果影响最大的那些高分数据点的内积计算精度。

57.nonmetric_mips
标题：Non-metric similarity graphs for maximum inner product search
来源网址：https://proceedings.neurips.cc/paper_files/paper/2018/file/229754d7799160502a143a72f6789927-Paper.pdf
简介：将强大的相似图搜索框架从度量空间推广到了非度量的内积空间，并提出了简单而高效的 ip-NSW 算法。其最深刻的洞见在于指出了并非所有NNS技术都适合通过MIPS-to-NNS转换来解决MIPS问题。

58.hashing_survey
标题：A survey on learning to hash
来源网址：https://arxiv.org/pdf/1606.00185
简介：对学习哈希（Learning to Hash）这一近似最近邻搜索（Approximate Nearest Neighbor Search, ANNS）的核心技术进行全面、系统的梳理和评述。它不仅回顾了经典算法，还特别强调和深入讨论了量化（Quantization）方法，并指出了其优越性。

59.semantic_product_search
标题：Semantic product search
来源网址：https://dl.acm.org/doi/pdf/10.1145/3292500.3330759
简介：提出的三元铰链损失、平均池化+n-gram的组合以及OOV哈希处理，紧密结合产品搜索的实际场景和约束（如短文本、稀疏信号、高QPS、海量数据），提出务实、高效且极其有效的解决方案。

60.t-SNE
标题：Visualizing data using t-SNE
来源网址：https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl
简介：t-SNE 是一种用于高维数据可视化的降维方法，目标是将高维数据（如 1000 维的文本向量、图像特征）映射到 2D 或 3D 空间，使得：
相似的样本在低维空间中距离近；不相似的样本距离远。

61.curriculum_learning
标题：Curriculum learning
来源网址：https://dl.acm.org/doi/pdf/10.1145/1553374.1553380
简介：先学简单概念，再学复杂概念；通过重加权训练样本分布，从易到难逐步引入样本；在视觉（形状识别）和 NLP（语言建模）任务中均取得更快收敛和更好泛化。


