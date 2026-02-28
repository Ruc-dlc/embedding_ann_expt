# embedding_ann_expt
毕业论文：dailongchao在线验证实验平台
学生：dailongchao
导师：he hu

# 数据集（分别是训练集、开发集、测试集样本数、文件大小）
实验数据集详情
Dataset                Train       Dev    Test     capacity
Natural Questions  79,168 58,880   8,757  3,610    7GB
TriviaQA           78,785 60,413   8,837  11,313   7GB
corpus                                             10GB
total capacity                                     24GB

Notes：由于数据集过大，无法上传至github，仅提供download command如下
# 下载NQ数据集
python -m dpr.data.download_data --resource data.retriever.nq
# 下载TriviaQA数据集
python -m dpr.data.download_data --resource data.retriever.trivia

# model backbone bert-base-uncased
简介：一款通用的英文文本理解基础模型&预训练模型，未经alignment fine-tuning(SFT、RLHF)过程，适合做downstream task的基座模型
下载link：https://huggingface.co/bert-base-uncased

# 参考文献
references.md（内含论文名称、来源网址、简介）

# experiment source code
完整代码位于./code目录下，内含项目源码README.md，内包含实验步骤、清单、环境配置、项目代码结构等其他项目描述信息
总体上由数据集的预处理部分、模型的训练部分（dataloader加载、联合损失函数与三阶段难负例挖掘pipeline、开发集评估保存最佳权重）、Faiss向量数据库构建、模型训练后的策略评估（指标包括Recall@k、MRR@k、NDCG@k、Recall vs efsearch、Recall vs Visited Nodes、efsearch vs Latency、positve similarity .etc）等其他支持实验modules多个部分所组成

4.Lastest Update time 2026/02/28/10:48










