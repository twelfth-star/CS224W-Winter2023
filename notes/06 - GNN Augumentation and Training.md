![](assets/Pasted%20image%2020230302193604.png)

# Prediction with GNNs

GNN 的输出是一系列节点的 embedding，即 $\{\textbf{h}_v^{(L)},\forall v\in G\}$，我们通过预测头（prediction head）把它转化为预测。设所需预测为 $k$ -way 的，即输出为一 $k$ 维向量。

* 节点级预测
	* $\widehat{\mathbf{y}}_{\boldsymbol{v}}=\text{Head}_{\text{node}}(\mathbf{h}_{\boldsymbol{v}}^{(L)})=\mathbf{W}^{(H)}\mathbf{h}_{\boldsymbol{v}}^{(L)}$，其中 $\textbf{W}^{(H)}\in\mathbb{R}^{k\times d}$
	* 即直接乘以一可学习的参数矩阵 $\textbf{W}^{(H)}$
* 边级预测
	* $\widehat{\mathbf{y}}_{uv}=\text{Head}_{\text{edge}}(\mathbf{h}_{u}^{(L)},\mathbf{h}_{v}^{(L)})$
	* $\text{Head}_{\text{edge}}(\mathbf{h}_{u}^{(L)},\mathbf{h}_{v}^{(L)})$ 的选择
		* 拼接+线性层： $\widehat{\mathbf{y}}_{uv}=\text{LINEAR}(\text{CONCAT}(\mathbf{h}_u^{(L)},\mathbf{h}_v^{(L)}))$
		* 内积
			* $k=1$：$\widehat{y}_{uv}=(\textbf{h}_{u}^{(L)})^T\textbf{h}_{v}^{(L)}$
			* $k>1$：$\widehat y_{uv}^{(i)}=(\textbf h_u^{(L)})^T\textbf W^{(i)}\textbf h_v^{(L)}$，$\widehat{y}_{uv}=\text{CONCAT}(\widehat{y}_{uv}^{(1)},...,\widehat{y}_{uv}^{(k)})\in\mathbb{R}^k$
* 图级预测
	* $\widehat{\mathbf{y}}_G=\text{Head}_{\text{graph}}(\{\mathbf{h}_v^{(L)}\in\mathbb{R}^d,\forall v\in G\})$
	* $\text{Head}_{\text{graph}}(\{\mathbf{h}_v^{(L)}\in\mathbb{R}^d,\forall v\in G\})$ 的选择
		* 整体平均、最大值、求和池化。只适用于小图。
		* 分层聚合：先对节点根据其 embedding 聚类，每一簇聚合为一节点，反复进行，直到只剩一个节点，以此作为图的输出
		* ![](assets/Pasted%20image%2020230302174119.png)

# Training Graph Neural Networks

训练用的 Ground-truth （即 $y$ ）的来源
* 有监督标签：外部信息，包括分子类型等；
* 无监督（自监督）信号：clustering coefficient, PageRank 等
它们又各自分为节点级 $y_v$、边级 $y_{uv}$ 和图级 $y_G$。

损失函数
* 分类任务：交叉熵（CE），$\text{CE}\left (\boldsymbol{y}^{(i)},\boldsymbol{\widehat{y}^{(i)}}\right)= \sum_{j=1}^{K}\boldsymbol{y}_{j}^{(i)}\log\left (\boldsymbol{\widehat{y}_j^{(i)}}\right)$
* 回归任务：均方误差（MSE），$\text{MSE}\bigl(\boldsymbol{y}^{(i)},\boldsymbol{\widehat{y}}^{(i)}\bigr)=\sum_{j=1}^{K}(\boldsymbol{y}_j^{(i)}-\boldsymbol{\widehat{y}}_j^{(i)})^2$

评价指标
* 回归任务：
	* 均方根误差均方根误差（RMSE），$\sqrt{\sum_{i=1}^N\frac{(\boldsymbol{y}^{(i)}-\boldsymbol{\widehat y}^{(i)})^2}{N}}$
	* 平均绝对值误差（MAE）， $\left.\frac{\sum_{i=1}^N\left|\boldsymbol{y}^{(i)}-\boldsymbol{\widehat{y}^{(i)}}\right|}{N}\right.$
* 分类任务
	* Multi-class 分类（多选一，不是多选多）：$\frac{1\left[\text{argmax}\left(\widehat{\boldsymbol{y}}^{(i)}\right)=\boldsymbol{y}^{(i)}\right]}{N}$
	* 二分类：Accuracy，Precision，Recall，F1-Score，ROC，AUC

# Setting-up GNN Prediction Tasks

数据集划分
* 训练集：最优化 GNN 参数
* 验证集：调整超参数
* 测试集：检验效果
对不同的随机数种子进行随机划分

节点互相连接，互相影响 embedding，训练集中的节点可能和测试集相连
* Solution 1：传导设置（Transductive setting）
	* 所有 splits 都能看到整个图结构，但只能看到自己的节点的 label。
	* 不能泛化到没见过的图，只适用于节点/边预测任务
* Solution 2：归纳设置（Inductive setting）
	* 暴力划分，不管 splits 之间的边。所有 splits 都只能看到自己的节点。
	* 能泛化到没见过的图，适用于节点/边/图预测任务

例子：链接预测 ^d58d0f
* Option 1：归纳设置
	* 把边随机分为 2 类，即消息边和监督边。
	* 直接把图分成 3 个 splits，每个 splits 里的边都有消息边和监督边两种。训练/验证/测试时都只使用自己的节点，用消息边去预测监督边。
* Option 2：传导设置（常用）
	* 把边随机分为 4 类，即 a. 训练消息边、b. 训练监督边、c. 验证边、d. 测试边
	* 训练时：用 a. 预测 b.
	* 验证时：用 a.、b.预测 c.
	* 测试时：用 a.、b.、c.预测 d.
