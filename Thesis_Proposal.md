中国人民大学攻读硕士学位研究生开题报告审核表
本表由硕士研究生开题考评专家小组填写，并附“硕士学位论文开题报告（文字部分）”。
院、系：信息学院/计算机系              	  专    业：计算机系统结构
姓  名：戴隆超                         	  研究方向：信息检索
学  号：2023103785                        导师姓名：胡鹤
拟定学位论文题目：基于向量分布的高效稠密检索协同优化方法研究

开题报告考评意见
考    评    项    目


文字报告	1、选题依据
	2、创新性
	3、选题难度和可行性
	4、研究工作方案的合理性
	5、科研工作时间安排的合理性
	6、预期成果
	7、开题报告的文字表达，参考文献引用合理
口头报告	8、条理清晰，层次分明
	9、基本概念清楚、明确选题在学科中的意义
	10、论证严密、逻辑性强
考评小组的意见书：





考评结果	通过		修改后通过		不通过	

组长签名：                         年     月     日
修改后
审核意见	通过		不通过		审核人
签名	
                年     月    日
开题报告考评小组组成
组成	姓名	职称	所在单位	签字
组长（非导师）				


成员				
				
				
				
所在院（系、所、中心）审核意见：

主管领导签字：                       公章

                                                     年   月   日                                               

附：  中国人民大学攻读硕士学位研究生学位论文开题报告
院  系：信息学院/计算机系               专    业：计算机系统结构
姓  名：戴隆超                          研究方向：信息检索
学  号：2023103785                      导师姓名：胡鹤
拟定学位论文题目：基于向量分布的高效稠密检索协同优化方法研究

	选题依据
	选题的理论意义及实践意义
随着搜索引擎和推荐系统的发展，如何在海量数据中快速、准确地检索语义相关内容，成为信息检索领域的重要挑战。近年来，基于深度学习的向量化检索方法迅速成为主流，其中以双塔模型（如SBERT[1]）为代表的稠密向量检索（Dense Retrieval）在开放域问答、网页检索、电子商务、推荐系统等场景中表现出显著优势。典型系统如DPR[2]和SimCSE[3]等成果表明，对比学习方法能够有效提升向量表示的语义判别能力，从而提高检索质量。
然而，现有研究普遍存在一个被忽视的问题：模型训练所产生的向量分布，与实际用于加速检索的近似最近邻搜索（ANN）索引结构之间存在语义与拓扑结构的失配问题，即缺乏协同优化。具体而言，即使双塔模型在语义空间中能够很好地区分正负样本，其向量分布可能不满足不同ANN算法的底层假设：对于基于图的索引（如HNSW[4]），松散的正样本对距离会破坏小世界图的导航效率；对于基于聚类的索引（如IVF[5]），不明显的聚类结构会增加搜索时需要探查的簇数量；对于基于量化的索引（如PQ[6]），高维相关性会降低子空间编码的有效性。这种失配导致ground truth在检索拓扑中分布分散，显著影响检索效率。
在对比学习框架中，主流方法强调"正例更相似、负例更疏离"的相对约束，但缺乏对"绝对距离大小"和"局部拓扑结构"的优化，导致模型可能学习到"排序正确但分布不理想"的嵌入空间。例如，InfoNCE损失[7]仅关注批次内的相对排序，忽略了向量的绝对距离和全局分布特性。对于大规模检索系统，这种分布会显著影响索引构建效率和查询性能。因此，有必要从度量学习和拓扑优化的视角重新思考密集检索模型的训练目标，使向量不仅具备良好的语义表达能力，也更适合被多样化的ANN索引结构高效组织。
基于这一背景，本研究拟在轻量级双塔模型的基础上，通过显式引入拓扑感知的距离约束损失，以更直接地控制query-doc正样本对的绝对距离，并优化向量的局部邻域结构，促使模型生成局部紧致、全局结构化的向量空间。结合多种ANN索引的原理，系统研究向量分布对不同索引结构的影响机制。通过对比优化前后的向量分布、多种索引的构建特性以及跨索引的检索性能（包括召回率、查询延迟、内存占用等），形成一套可工程落地、可扩展的向量检索优化范式。
在实践层面，本研究具有重要意义。大规模检索系统通常根据不同的应用场景选择不同的索引策略：高精度场景可能选用HNSW，内存敏感场景可能选用IVF_PQ等。然而，现有向量表示方法缺乏对这些索引特性的适配。本研究提出的编码空间优化方法能够改善向量在多种索引结构中的适配性，从而在相同硬件条件下提升查询吞吐量（QPS）、降低查询延迟，并减少内存占用。此外，本文基于公开数据集（如NQ[8]、MS MARCO[9]）和开源工具（如Faiss[10]）实现的方法具有高度复现性，可进一步推广至推荐系统、广告检索、多模态检索等实际应用场景。
综上，本研究在理论上推动了"对比学习—向量分布—多种ANN索引结构"这一整体优化范式的发展，首次系统性地研究了向量拓扑特性对多样化索引算法的影响；在实践上面向工业级检索系统的多样性需求，提供了一种提升检索效率与检索质量的可行方案。
	国内外研究现状分析
随着Transformer架构与深度学习技术的迅猛发展，基于稠密向量的语义检索已逐步取代传统基于关键词的稀疏检索方法（如BM25）[11]，成为信息检索领域的主流范式。当前研究体系主要围绕语义向量表示学习、近似最近邻检索结构、以及向量分布与检索效率的协同优化三个关键方向展开。国际上在基础理论与算法创新方面持续引领方向，而国内学术界与工业界近年来则在大规模中文语义理解与实际系统落地方面取得了显著进展。
在语义向量表示学习方面，基于双塔结构与对比学习的方法已成为生成高质量语义向量的标准路径。早期工作中，Oord等人[7]提出的对比预测编码（CPC）首次将InfoNCE损失函数引入表示学习，通过互信息最大化奠定了对比学习的理论基石。Reimers和Gurevych[1]提出的Sentence-BERT（SBERT）采用孪生网络结构对BERT进行改造，实现了句子级别嵌入的高效生成，为后续双塔检索模型奠定了架构基础。Gao等人[3]提出的SimCSE通过简单的Dropout噪声构建正样本，显著改善了向量空间中的各向异性问题，推动无监督句向量学习向前迈进。Wang和Isola[12]则从理论角度将对比学习的目标归纳为“对齐性”（alignment）与“均匀性”（uniformity），为理解和分析表示学习机制提供了清晰的理论框架。在有监督检索任务中，Karpukhin等人[2]提出的DPR首次在开放域问答任务上验证了稠密检索的可行性，推动了语义检索的实用化进程。Qu等人[13]提出的RocketQA通过加强负样本挖掘与训练策略优化，进一步提升了模型在复杂场景下的鲁棒性。近年来，Thakur等人[14]构建的BEIR与Muennighoff等人[15]构建的MTEB等基准的建立，为评估嵌入模型的零样本泛化能力提供了标准体系。国内方面，Xiao等人[16]发布的C-Pack则系统性地整合并提升了中文语义表示的技术水平。然而，现有方法大多侧重于样本间的相对排序约束（如InfoNCE损失），缺乏对向量绝对距离大小和局部拓扑结构的显式控制，这可能导致生成的向量分布过于松散或不规则，进而影响下游索引结构的构建与检索效率。
在近似最近邻检索结构方面，随着数据规模与向量维度的不断攀升，高效的索引机制已成为大规模检索系统的核心组件。其中，基于图结构的ANN算法因其优越的性能表现受到广泛关注。Malkov和Yashunin[4]提出的HNSW通过构建分层可导航小世界图，在检索精度与速度之间实现了良好平衡，已成为Faiss、HNSWlib等主流向量库的核心索引方法。在工程化与系统优化方面，Jayaram Subramanya等人[17]提出的DiskANN创新性地采用固态硬盘与内存混合存储架构，支持百亿级别向量的高效检索，为超大规模应用提供了可行方案。与此同时，基于量化的方法在内存受限场景下展现出独特优势。Jégou等人[5][6]提出的乘积量化及其变体通过将高维向量压缩为短码，实现了内存占用与检索速度的显著优化。Ge等人[18]进一步提出了优化乘积量化方法以提升精度。作为集成了多种索引算法的开源库，Faiss[10, 19]持续推动ANN技术的普及与优化，提供了IVF、PQ、HNSW等算法的统一实现。尽管现有研究在索引结构本身的设计与优化上已取得显著进展，但其普遍将上游模型产出的向量分布视为固定不可优化的先验，未能从根本上解决语义模型输出与下游索引结构之间的适配失配问题。
近年来，越来越多的研究开始关注向量分布对检索效率的影响，尝试从表示学习阶段入手优化嵌入空间的几何特性。Wang和Isola[12]的理论分析表明，对比学习隐式地优化了表征空间的对齐性与均匀性，这为理解向量分布提供了重要视角。Zhan等人[20]通过实验证明，在训练中引入困难负样本可以优化embedding分布，进而提升检索性能。Yang等人[21]提出的混合负采样策略也旨在通过改进采样分布来优化训练过程。在面向特定索引结构的优化方面，Xiong等人[22]提出的近似最近邻负对比学习（ANN-NCL）尝试在训练中引入基于ANN搜索的困难负样本，间接影响了向量分布。然而，这些工作仍主要服务于提升语义召回率（Recall），并未系统性地研究优化后的向量分布对不同索引结构（如HNSW、IVF、PQ）的构建效率、查询速度及内存占用等核心工程指标的影响。向量分布的“拓扑友好性”与索引效率之间的关联机制尚未被充分揭示与量化。
综上所述，现有研究在语义表示与检索算法两个方向上均取得了显著进展，但仍存在明显不足：一方面，以DPR、SimCSE为代表的语义模型侧重于语义对齐质量与排序友好性，缺乏对向量分布紧凑性、聚类性及索引友好性的显式考虑；另一方面，以HNSW、IVF、PQ为代表的检索算法侧重于索引结构本身的优化，却未将上游的向量分布作为可干预、可优化的变量。这种模型训练与索引构建之间的割裂状态，限制了大规模检索系统在“精度-速度-内存”权衡上达到全局最优。针对这一问题，本文提出一种面向拓扑友好性的对比学习向量分布优化方法，通过在训练阶段引入绝对距离约束，显式控制正样本对的向量距离并优化局部邻域结构，从而生成更利于多种ANN索引结构高效组织和检索的向量空间。本研究旨在弥合语义学习与检索效率之间的鸿沟，为构建高性能、低延迟的大规模检索系统提供新的思路与方法。
	参考文献
[1] Reimers N, Gurevych I. Sentence-bert: Sentence embeddings using siamese bert-networks[J]. arXiv preprint arXiv:1908.10084, 2019.
[2] Karpukhin V, Oguz B, Min S, et al. Dense Passage Retrieval for Open-Domain Question Answering[C]//EMNLP (1). 2020: 6769-6781.
[3] Gao T, Yao X, Chen D. Simcse: Simple contrastive learning of sentence embeddings[J]. arXiv preprint arXiv:2104.08821, 2021.
[4] Malkov Y A, Yashunin D A. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs[J]. IEEE transactions on pattern analysis and machine intelligence, 2018, 42(4): 824-836.
[5] Jégou H, Douze M, Schmid C. Searching with quantization: approximate nearest neighbor search using short codes and distance estimators[D]. INRIA, 2009.
[6] Jegou H, Douze M, Schmid C. Product quantization for nearest neighbor search[J]. IEEE transactions on pattern analysis and machine intelligence, 2010, 33(1): 117-128.
[7] Oord A, Li Y, Vinyals O. Representation learning with contrastive predictive coding[J]. arXiv preprint arXiv:1807.03748, 2018.
[8] Lee K, Chang M W, Toutanova K. Latent retrieval for weakly supervised open domain question answering[J]. arXiv preprint arXiv:1906.00300, 2019.
[9] Bajaj P, Campos D, Craswell N, et al. Ms marco: A human generated machine reading comprehension dataset[J]. arXiv preprint arXiv:1611.09268, 2016.
[10] Douze M, Guzhva A, Deng C, et al. The faiss library[J]. IEEE Transactions on Big Data, 2025.
[11] Robertson S E, Walker S. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval[C]//SIGIR’94: Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University. London: Springer London, 1994: 232-241.
[12] Wang T, Isola P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere[C]//International conference on machine learning. PMLR, 2020: 9929-9939.
[13] Qu Y, Ding Y, Liu J, et al. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering[C]//Proceedings of the 2021 conference of the North American chapter of the association for computational linguistics: human language technologies. 2021: 5835-5847.
[14] Thakur N, Reimers N, Rücklé A, et al. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models[J]. arXiv preprint arXiv:2104.08663, 2021.
[15] Muennighoff N, Tazi N, Magne L, et al. Mteb: Massive text embedding benchmark[C]//Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics. 2023: 2014-2037.
[16] Xiao S, Liu Z, Zhang P, et al. C-pack: Packed resources for general chinese embeddings[C]//Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval. 2024: 641-649.
[17] Jayaram Subramanya S, Devvrit F, Simhadri H V, et al. Diskann: Fast accurate billion-point nearest neighbor search on a single node[J]. Advances in neural information processing Systems, 2019, 32.
[18] Ge T, He K, Ke Q, et al. Optimized product quantization for approximate nearest neighbor search[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2013: 2946-2953.
[19] Johnson J, Douze M, Jégou H. Billion-scale similarity search with GPUs[J]. IEEE Transactions on Big Data, 2019, 7(3): 535-547.
[20] Zhan J, Mao J, Liu Y, et al. Optimizing dense retrieval model training with hard negatives[C]//Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021: 1503-1512.
[21] Yang J, Yi X, Zhiyuan Cheng D, et al. Mixed negative sampling for learning two-tower neural networks in recommendations[C]//Companion proceedings of the web conference 2020. 2020: 441-447.
[22] Xiong L, Xiong C, Li Y, et al. Approximate nearest neighbor negative contrastive learning for dense text retrieval[J]. arXiv preprint arXiv:2007.00808, 2020.
[23] Jégou H, Tavenard R, Douze M, et al. Searching in one billion vectors: re-rank with source coding[C]//2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2011: 861-864.
[24] Huang P S, He X, Gao J, et al. Learning deep structured semantic models for web search using clickthrough data[C]//Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013: 2333-2338.
[25] Yi X, Yang J, Hong L, et al. Sampling-bias-corrected neural modeling for large corpus item recommendations[C]//Proceedings of the 13th ACM conference on recommender systems. 2019: 269-277.
[26] Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual representations[C]//International conference on machine learning. PmLR, 2020: 1597-1607.
[27] He K, Fan H, Wu Y, et al. Momentum contrast for unsupervised visual representation learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 9729-9738.
[28] Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//International conference on machine learning. PmLR, 2021: 8748-8763.
[29] Wang J, Zhu J, He X. Cross-batch negative sampling for training two-tower recommenders[C]//Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021: 1632-1636.
[30] Chen C, Zhang J, Xu Y, et al. Why do we need large batchsizes in contrastive learning? a gradient-bias perspective[J]. Advances in Neural Information Processing Systems, 2022, 35: 33860-33875.
[31] Wang L, Yang N, Huang X, et al. Text embeddings by weakly-supervised contrastive pre-training[J]. arXiv preprint arXiv:2212.03533, 2022.
[32] Li Z, Zhang X, Zhang Y, et al. Towards general text embeddings with multi-stage contrastive learning[J]. arXiv preprint arXiv:2308.03281, 2023.
[33] Moiseev F, Abrego G H, Dornbach P, et al. Samtone: Improving contrastive loss for dual encoder retrieval models with same tower negatives[J]. arXiv preprint arXiv:2306.02516, 2023.
[34] Koromilas P, Bouritsas G, Giannakopoulos T, et al. Bridging mini-batch and asymptotic analysis in contrastive learning: From InfoNCE to kernel-based losses[J]. arXiv preprint arXiv:2405.18045, 2024.
[35] Junfan C, Zhang R, Zheng Y, et al. Dualcl: Principled supervised contrastive learning as mutual information maximization for text classification[C]//Proceedings of the ACM Web Conference 2024. 2024: 4362-4371.
	研究方案
	研究内容
本研究围绕“语义模型-向量分布-索引结构”的协同优化问题，旨在构建一个高性能的端到端向量检索系统。具体研究内容包括以下三个层面：
	面向ANN检索友好的损失函数的优化
针对现有对比学习仅关注相对距离而忽视绝对距离分布的问题，本研究将在InfoNCE损失基础上引入绝对距离约束项，构建联合优化目标loss函数。该损失函数在保持语义判别力的同时，强制约束正样本对的余弦相似度向预设目标区间集中，从而产生局部紧致、全局均匀的向量分布，为后续ANN检索奠定基础。
	迭代式难负例样本的挖掘
为提升模型对高迷惑性样本的判别力，本研究拟设计一个三阶段迭代挖掘的可行方案。第一阶段使用批量内负样本进行模型预热；第二阶段引入基于BM25的静态难负例，利用Pyserini等开源工具从词法重叠层面挖掘困难样本；第三阶段进行周期性的模型难负例挖掘：每训练N轮后，用当前最佳模型编码全库文档并构建Faiss索引，为该批次所有查询检索Top-K结果并剔除正例，以此动态更新训练数据中的负例。
	端到端检索系统的实现与验证
为实现面向拓扑友好性的稠密检索闭环验证，本研究将基于Python语言和开源工具链，构建一个系统化、可配置的向量检索验证平台。
具体实施如下：
	模型集成与部署： 将经过拓扑优化（即引入复合损失函数和迭代式难负例挖掘）训练得到的语义模型，集成至检索系统。具备为query和doc生成高维向量能力。
	向量数据库构建与索引实现：借助 FAISS、Pyserini库，构建一个高效的离线/在线向量数据库。系统实现和对比五种代表性索引结构（Flat、IVF、IVF_PQ、HNSW、BM25）。
	系统级性能评估：最终NQ等标准数据集上，对原始模型和优化后模型产出的向量进行全方位、系统性的对比评估。评估维度不仅包含标准的语义召回指标（Recall@k, MRR@k），更将深入索引内部效率指标（例如：HNSW的访问节点数 touch_rate、IVF的探查聚类数nprobe对召回率的影响），从而验证和量化拓扑优化对检索延迟和吞吐量的提升效果。
	研究中所要突破的难题
	损失函数的平衡设计
如何在InfoNCE损失与距离约束损失之间找到最优平衡点，确保在提升分布紧致性的同时不损害模型的语义判别能力，是需要解决的首要难题。这涉及到权重系数的敏感度分析及目标区间的科学设定。
	迭代式挖掘流程的稳定化与效率优化
如何设计高效且鲁棒的多阶段训练流程，确保周期性挖掘到高质量难负例，并避免引入噪声或导致训练震荡，是关键的工程实现难点。这包括挖掘频率、索引更新策略与训练步调的协同。
	有何特色与创新之处
	提出语义-拓扑协同优化新范式
将向量分布优化与索引结构设计纳入统一框架，通过“模型训练时考虑索引效率，索引构建时利用分布特性”的双向优化思路，解决了传统方案中模型与索引割裂的关键问题。
	设计多目标约束的损失函数
基于对比学习，对InfoNCE损失进行增强，使模型训练目标从 “排序正确” 转向 “拓扑友好”，确保产出的向量分布具备高紧致度、低方差的特性，从源头上改善向量分布的ANN检索友好性，从而为HNSW索引优化提供几何基础，进而为稠密检索模型的设计提供新方向。
	构建系统化的索引效率评估框架
系统化地实证分析向量分布对全系列ANN索引（图、量化、聚类、哈希）的深度影响。不仅评估召回率，更深入剖析touch_rate、nprobe等内部效率指标，为工业界针对不同场景（高精度、低内存、高吞吐）的“模型-索引”联合选型提供参数设置手册与依据。
	拟采取的研究方法和时间安排
本研究围绕“模型训练—系统部署—实验评估”三个阶段开展工作。采用理论分析、算法设计、系统实现与性能对比实验相结合的方法，实现一个端到端的稠密检索系统原型。
	系统实现
	本研究将自顶向下设计开发向量数据库原型系统，贯通模型算法的训练、索引的构建与测评工作，实现完整的端到端闭环，具体包含以下三个部分：
	模型训练模块：基于PyTorch框架实现，集成信息噪声对比估计（InfoNCE）损失与绝对距离约束项w*(1-cos⁡(q,d^+ ) )的复合损失函数，支持迭代式难负例挖掘（包括BM25静态挖掘与周期性模型动态挖掘）的训练流程，用于训练DPR、BGE-small双塔语义模型。
	索引核心模块：基于开源库实现，构建并管理七类代表性向量索引（Flat、LSH、PQ、 IVF、IVF_PQ、HNSW、BM25）。该模块负责索引的构建、参数配置、持久化存储及高效检索，并为系统评估提供构建时间、内存占用等底层数据。
	系统评估模块：设计并实现一套完整的自动化评估流水线。该模块集成语义召回评估向量分布、召回与准确率（Recall@k, MRR@k）、查询效率评估（延迟、吞吐量QPS）以及索引内部效率分析（如HNSW的touch_rate、IVF的nprobe分析）功能，最终输出综合性能对比报告。
	理论分析与问题建模
针对现有Dense Retrieval模型与向量索引之间的 语义-拓扑结构失配问题，本研究的核心理论分析在于：
	InfoNCE 的局限性分析：InfoNCE损失仅是一种相对约束，其优化目标是确保正样本对相似度高于负样本对，但无法控制正样本对的绝对距离大小或簇内方差。这种缺乏绝对约束的训练可能导致嵌入空间出现正样本对过于分散、局部紧致度差的问题。对于HNSW等图索引，这种分布会使相关文档在导航图中相距较远，导致查询路径过长，增加访问节点数与查询延迟；对于IVF等聚类索引，则会导致聚类边界模糊、簇内方差过大，使得在检索时需探查更多聚类（更大的nprobe）才能达到目标召回率，同样增加计算开销与响应时间。
	协同优化建模：本研究提出一种协同优化目标，通过在InfoNCE损失中引入绝对距离约束项w*(1-cos⁡(q,d^+ ) )，使模型训练目标从“排序正确”转向“拓扑友好”。该设计旨在从源头上确保产出的向量分布具备高紧致度、低方差的特性。对于HNSW索引，这种局部紧致的分布意味着相关文档在导航图中有更直接的连接，从而显著降低查询延迟与访问节点数（touch_rate）。对于IVF及IVF_PQ索引，优化后的均匀分布特性则有利于形成边界清晰、大小均衡的聚类簇，从而在相同的探查聚类数（nprobe）下，达到更高的召回率或更低的计算复杂度。整体上，通过在训练阶段前置性地优化向量拓扑结构，为下游多样化的ANN索引效率提升奠定了几何基础。
	算法设计与优化
本研究的算法设计分为离线训练与在线检索评估两个阶段。
4.3.1 离线模型训练
损失函数设计：
 
图 1 经典对比学习模型训练范式
损失函数公式：
 
当前基于对比学习的双塔模型训练，普遍采用以InfoNCE为代表的目标函数。其核心优化范式是在一个Batch内，促使查询向量qi与其对应的正文档向量di+的相似度得分（通常位于相似度矩阵的对角线位置），高于qi与本Batch内其他所有文档向量dj (j≠i)的得分。
然而，这一范式存在一个对近似最近邻检索不友好的固有缺陷：它仅优化了样本间的相对排序(Relative Ranking)，而完全忽略了正样本对之间的绝对距离(Absolute Distance)。即只要正例矩阵对角线的值大于同一行的其他值，损失函数即被最小化，认为完成目标，模型训练便视为“完成目标”。这导致即使查询与其正文档的绝对相似度较低（例如余弦相似度仅为0.5），模型也缺乏进一步拉近它们的动力。
这种“宽约束”所产生的向量分布，对后续的ANN检索效率构成严重挑战：
	Ground Truth分散化：语义相关的(q, d+)对被映射到嵌入空间中相距较远的不同区域，使得一个查询的真实答案可能分散在HNSW等图索引的多个不同“簇”中。
	检索范围被迫扩大：为了确保召回这些分散的正样本，检索算法（如HNSW的贪心搜索）不得不扩大搜索范围，访问更多的节点和簇，从而显著增加计算量(Visited Node Count)和查询延迟。
	索引结构低效：基于此类分布构建的索引，其局部连通性与导航路径并非最优，无法充分发挥HNSW等算法的性能潜力。
因此，如何在对模型施加正确语义约束的同时，显式地约束正样本对的绝对距离，促使生成“紧致”且对ANN索引友好的向量分布，成为提升端到端检索系统效率的关键问题。
优化后损失函数公式：
 
在上述训练范式的基础上，我们增加了距离损失，其中，qidi+为query和强正例doc的 cosine 相似度，因此目标是让qidi+尽可能大，于是1-qidi+尽可能小，并通过权重w来控制距离损失干涉的强度。
难负例挖掘策略：
为了提升模型对高迷惑性样本的区分能力，本研究采用一种三阶段迭代挖掘的可行方案：
	阶段一（Warm-up）：
使用批量内随机负样本（In-Batch Negatives）进行模型预热，快速收敛至一个基础语义空间。
	阶段二（BM25 Hard Negatives）：
利用开源Pyserini检索工具基于BM25算法构建倒排索引，从全库中为每个查询挖掘Top-K个困难负例，注入训练数据，使模型学会区分词汇重叠但语义不匹配的文档。
	阶段三（Iterative Model Hard Negatives）：
按照step、epoch周期性地使用当前模型为全库文档编码并构建Faiss索引，然后为每个查询检索Top-K个最近邻非正例文档作为新的困难负例，动态更新训练数据，实现模型自我迭代优化。



模型训练：
1）数据集选型
数据集	描述	备注
NQ	Google搜索真实Query	正例通过官方标注的Gold Passage获取
TriviaQA	网络冷知识问答	BM25检索Top100选取包含答案且排名最高的段落作为正例。
WQ	Google Suggest API	BM25检索Top100选取包含答案且排名最高的段落作为正例。
MS MARCO	Bing搜索日志构建的真实Query	正例通过官方标注获取
2）模型选型
本研究采用轻量级双塔编码器作为基础架构，旨在平衡模型性能与部署效率，为向量分布优化提供清晰的观测窗口。
模型	参数/维度	备注
bge-small-en-v1.5	33M参数，384维	轻量级语义模型SOTA，用于快速验证
BERT-base	220M参数，768维	Dense retrieval经典Baseline
NV-Embed	1B参数，1024维	MTEB榜单SOTA大模型，zero-shot
3）训练框架与技术路线
训练框架与核心算法
	主框架：基于双塔结构构建自定义的稠密检索训练框架，采用 PyTorch 从零实现训练循环，具体包含复合损失函数优化、多阶段难负例挖掘策略、训练器、在线评估等。模型初始化依赖 Hugging Face Transformers，自行编写逻辑实现联合损失函数与迭代式难负例样本挖掘策略。
	核心训练技术：
	联合优化损失：基于余弦相似度矩阵与交叉熵计算实现联合损失函数，在InfoNCE基础上，引入绝对距离约束项，显式地拉近正样本对在向量空间中的绝对距离，公式化引导生成更紧致的向量分布：
L_dis=w*(1-cos⁡(q,d^+ ) )           →         L_total=L_InfoNCE+L_dis
其中 w 为超参数，控制该项在总损失中的权重。
	三阶段迭代式难负例样本挖掘：
为提升模型对困难负样本的判别能力，本文采用三阶段逐步增强的难负例挖掘流程。该策略结合词法难负例与语义难负例，提高模型的决策边界精准度。
阶段一：In-Batch Negatives 预热
使用批量内负样本快速将模型收敛至基础语义空间。此阶段数据仅包含 (q, d+) 对，负例来自同一batch内其他文档。
阶段二：BM25 静态难负例挖掘
利用Pyserini的BM25检索引擎，从全库中为每个查询挖掘词法上相似、语义上不匹配的困难文档，让模型学会区分词汇重叠但语义错误的文档：
(q,d^+,〖doc〗_BM25^hard )
阶段三：模型自监督动态难负例挖掘
每训练 N 个 step/epoch，使用当前模型编码所有文档向量，构建 Faiss 索引，检索每个查询的 Top-K 最近邻文档并剔除正例，用作新的难负例样本，即从模型本身产出难负例样本：
(q,d^+,〖doc〗_model^hard )
3. 训练流程控制
在训练循环中动态切换数据加载器DataLoader进行三阶段训练，定期难负例挖掘脚本，数据集增量更新，按照模型在dev评测集上表现进行快照保存。
4.3.2 在线检索系统端到端部署
该阶段目标是实现一个基于Python生态的向量检索系统原型，用于验证优化后的向量分布对检索效率的实际提升。具体流程如下：
模型服务化：将训练好的模型封装为REST API服务，提供在线向量编码能力。当接收到查询文本时，服务实时调用模型进行推理，获得对应的query查询向量。
向量数据库构建：利用FAISS库，基于优化后的模型编码全量文档，并构建多种索引（Flat、HNSW、IVF、IVF_PQ、PQ、LSH）进行持久化存储，形成离线向量数据库。
端到端检索：在线服务接收到查询向量后，将其传入预先加载的FAISS索引中进行最近邻搜索。由于优化后的向量分布更利于索引结构，查询路径更短，从而实现低延迟、高精度的检索。
系统评估：通过自动化脚本，对系统进行端到端的性能评估，包括召回率、查询延迟、吞吐量以及索引内部效率指标（如HNSW的访问节点数、IVF的探查聚类数等）。
以上设计，本研究将在统一的Python生态下完成从模型训练、索引构建到系统评估的完整闭环，为稠密检索系统的优化提供切实可行的方案和实证依据。
 
图 2 端到端检索系统架构图
4.4 实验章节
4.4.1 实验设计
完整实验一共分为6组，分为3类，基线组、策略组、MTEB榜单当前SOTA模型对比组。
整个评估过程又分为上游任务评估与下游任务评估，上游和下游需要关注的指标有所不同，上游通过由于无下游数据库，故采用穷举法计算候选池中相似度排名前K个文档作为研究对象，考核平均相似度、Topk准确率；下游任务评估包含Recall@k、MRR@k、平均检索延迟、QPS等。
# Baseline组:
1.BM25
2.bge-small-en-v1.5 + InfoNCE Loss
3.bert-base + InfoNCE Loss
# 策略组
4.bge-small-en-v1.5 + InfoNCE Loss + Topology_strategy + 三阶段难负例样本挖掘
5.bert-base + InfoNCE Loss + Topology_strategy + 三阶段难负例样本挖掘
# SOTA model NV-Embed
6.NV-Embed
4.4.2 初步实验结果整理
实验环境：
硬件：
Ubuntu 24.04.3 LTS
NVIDIA RTX A6000 (48GB 显存)
251GB 物理内存（可用 92GB）
7.0TB NVMe 固态硬盘
数据集：nq-dev.json评测集包含6515条查询，psgs_w100.tsv维基百科包含2100w条段落。
在BGE的基础上(base模型为BAAI/bge-small-en-v1.5) fine tuning，具体来说在nq-train.json训练集上训练24个epoch，batch size设置为128，按照在dev评测集上的表现保存获取到最佳的5组checkpoint，weight权值设置分别为0.4、0.6、0.8、1.0，获取上游和下游评估结果。
上游评估：在不建立下游索引的前提下，从nq-dev.json抽取1k条查询，使用psgs_w100.tsv中抽取10w条，计算相似度排序从大到小获取Topk结果。
 
图 3 穷举法bge-small-en-v1.5指标对比

下游评估：下游初步构建HNSW索引，基于nq-dev.json抽取6515条查询,保证查询到正例在候选池的同时，随机补充负例文档数量至20w条段落。正例覆盖率大致约为3%，模拟真实查询场景。
 
图 4 Topk准确率对比
 
图 5 平均倒数排名对比
 
图 6 时延与吞吐量对比
 
图 7 Recall@100召回率对比
	实验总结：通过引入距离约束以及挖掘难负例样本策略，初步研究了对比学习框架下query-doc正例样本对的优化机制，能够在可控的准确率范围下，带来正例相似度提升的收益，针对超参数w的设置，从当前实验来看，w取值为0.6为最佳平衡点，在这样的训练框架下，不同于传统的排序友好的表示学习范式，而转变为拓扑友好的表示空间重构。相似度的提升直接转化为近似最近邻检索效率的改善，能够使得HNSW图索引获取更平滑的表示空间，简化Query在搜索Doc的遍历路径，进而降低平均查询时延，给整体的检索服务吞吐量带来不错的收益。同时相似度的进一步提升，在IVF倒排索引的构建中，也能够使得K-means的簇分布更紧密，提升索引质量。在AI搜索场景下，极低的延迟要求，即使毫秒级别响应优化，也能给用户带来更好的体验。
	基于当前实验结果，后续研究工作将补充证明相似度的提升对IVF倒排索引和其他形式的ANN索引构建和检索带来的影响；根据训练阶段或样本难度自适应调整距离损失权重，结合检索准确率、相似度、多样性等目标进行联合优化；从理论上，深入分析表示空间重构的数学机理。







4.5 时间安排
事项	时间	核心任务	进度
论文调研	25年10月-26年1月	调研ANN索引与对比学习相关工作	持续进行中
双塔模型的训练与实验平台搭建
	25年10月-26年2月	模型训练，收敛好实验中需要的模型组，在向量数据库中集成，搭建端到端检索平台，实验评估及作图脚本的代码编写	目前模型训练流程框架、向量数据库的构建、自动化评估任务脚本代码已基本实现，具备实验的进行能力。后期持续调研相关工作做一些优化
实验验证
	26年12月-26年2月	针对预设计实验组进行实验并记录比较实验结果
 	初步实验已完成，后期完整实验预备中
论文撰写与答辩	26年1月-26年5月	总结实验结论，撰写毕业论文，完成答辩	准备工作中

