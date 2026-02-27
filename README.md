# embedding_ann_expt
毕业论文：dailongchao在线验证实验平台
学生：dailongchao
导师：he hu

1.数据集（分别是训练集、开发集、测试集样本数、文件大小）
实验数据集详情
Dataset                Train       Dev    Test     capacity
Natural Questions  79,168 58,880   8,757  3,610    7GB
TriviaQA           78,785 60,413   8,837  11,313   7GB
corpus                                             10GB
total capacity                                     24GB

2.参考文献
references.md（内含论文名称、来源网址、简介）

3.实验源码
完整代码位于./code目录下，内含项目源码README.md，内包含实验步骤、清单、环境配置、项目代码结构等其他项目描述信息
实验总体上开展由数据集的预处理部分、模型的训练部分（dataloader加载、联合损失函数与三阶段难负例挖掘pipeline、开发集评估保存最佳权重）、Faiss向量数据库构建、模型训练后的策略评估（指标包括Recall@k、MRR@k、NDCG@k、Recall vs efsearch、Recall vs Visited Nodes、efsearch vs Latency、positve similarity .etc）等









