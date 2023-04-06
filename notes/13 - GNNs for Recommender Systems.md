# Recommender Systems: Task and Evaluation

推荐系统可被自然地被建模为一个用户-物品二分图。边为两者的互动，如点击、购买等。给定过去的边，预测新边。可被视为链接预测。

我们要从大量的物品中选择 $K$ 个（大概在 10~100 的量级）推荐给用户。我们希望在这 $K$ 个中包含更多正物品（即用户在未来会与之互动的）。

评价指标： Recall@K 。令 $P_u$ 为用户 $u$ 的正物品集合，$R_u$ 为用户 $u$ 的推荐集，$|R_u|=K$，则 
$$\text{Recall@}K=\frac{|P_u\cap R_u|}{|P_u|}$$
最终的 Recall@K 取每个用户的 Recall@K 的均值。

# Recommender Systems: Embedding-Based Models

令 $U$ 为用户集，$V$ 为物品集，$E$ 为边集。我们需要一个评分函数 $\text{score}(u,v)$ 来选取评分前 $K$ 高的（且用户没互动过的）物品推荐给用户。

我们使用基于 embedding 的评分函数：$\text{score}(u, v)=f_\theta(\mathbf{u}, \mathbf{v})$，其中 $\mathbf{u},\mathbf{v}\in \mathbb{R}^D$ 。

我们希望 Recall@K 尽可能高，但一个问题是它不可导，无法通过梯度下降法优化。我们考虑以下两个代替损失函数（surrogate loss functions），它们可导。

* 二分损失函数
	* 定义正边集为 $E$，负边集为 $E_\text{neg}$。
	* $$\ell=-\frac{1}{|E|}\sum_{(u,v)\in E}\log\left(\sigma(f_\theta(\mathbf{u},\mathbf{v}))\right)-\frac{1}{|E_\text{neg}|}\sum_{\left(u,v\right)\in E_\text{neg}}\log\left(1-\sigma(f_{\theta}(\mathbf{u},{\mathbf{v}})\right))$$
	* 也就是把正边的分推高，负边得分推低
	* 一个问题是这其实不必要地惩罚了输出，因为我们不需要保证所有正边都高，负边都低，**我们只需要对同一个用户 $u$ 而言，正物品的评分相对高，负物品的评分相对低即可**
* 贝叶斯个性化排序（Bayesian Personalized Ranking，BPR）
	* 对所有用户 $u^*\in U$，定义 $E(u^*)=\{(u^*,v)\mid(u^*,v)\in E\}$，$E_{\text{neg}}(u^*)=\{(u^*,v)\mid(u^*,v)\in E_{\text{neg}}\}$。
	* 对用户 $u^*$：$$\operatorname{Loss}\left(u^{*}\right)=\frac{1}{\left|E\left(u^{*}\right)\right| \cdot\left|E_{\mathrm{neg}}\left(u^{*}\right)\right|} \sum_{\left(u^{*}, v_{\mathrm{pos}}\right) \in E\left(u^{*}\right)} \sum_{\left(u^{*}, v_{\mathrm{neg}}\right)\in E_{\mathrm{neg}}\left(u^{*}\right)}-\log \left(\sigma\left(f_{\theta}\left(\mathbf{u}^{*}, \mathbf{v}_{\text {pos }}\right)-f_{\theta}\left(\mathbf{u}^{*}, \mathbf{v}_{\mathrm{neg}}\right)\right)\right)$$
	* 最终的 BPR 损失函数：$\frac{1}{|U|}\sum_{u^*\in U}\text{Loss}(u^*)$
	* 具体计算中，对于一个小批量 $\mathbf{U}_\text{mini}$，我们先对于每个 $u^*\in\mathbf{U}_\text{mini}$，选取一个正物品 $v_\text{pos}$ 以及一个负物品集 $V_\text{neg}$，计算 $$\frac{1}{\left|\boldsymbol{U}_{\text{mini}}\right|} \sum_{u^{*} \in \boldsymbol{U}_{ \text {mini}}} \frac{1}{\left|\boldsymbol{V}_{\mathrm{neg}}\right|} \sum_{v_{\mathrm{neg} } \in \boldsymbol{V}_{\mathrm{neg}}}-\log \left(\sigma\left(f_{\theta}\left(u^{*}, v_{\mathrm{pos}}\right)-f_{\theta}\left(u^{*}, v_{\mathrm{neg}}\right)\right)\right)$$
为什么 embedding 模型 work？协同滤过：我们为用户推荐与他们相似喜好的用户互动过的物品。而 embedding 模型可以捕获用户的相似性。

# Neural Graph Collaborative Filtering

传统协同滤过基于[浅 encoder](03%20-%20Node%20Embeddings.md#^934fbd)，也就是直接给每个物品和用户指定一个 D 维的 embedding。然后以两者内积为评分函数，通过上面介绍的损失函数训练。这样的模型**只能通过损失函数隐式地捕获一阶图结构**。

我们希望显式地捕获高阶图结构。使用 GNN！

神经网络协同滤过（Neural Graph Collaborative Filtering，NGCF）
* 为每个节点指定可学习的浅 embedding
* 通过多层 GNN 传播 embedding，从而捕获高阶图结构。具体略，参考[04 - Graph Neural Networks (GNN)](04%20-%20Graph%20Neural%20Networks%20(GNN).md)
* 以两者内积作为评分函数

# LightGCN

NGCF 中的浅 embedding 的复杂度为 $O(ND)$，在 $N$ 很大时计算起来非常昂贵。我们需要简化算法。

![](assets/Pasted%20image%2020230312185423.png)
回忆 C&S 中的[扩散矩阵](08%20-%20Label%20Propagation%20on%20Graphs.md#^d06835)。扩散矩阵 $\widetilde{A}=\textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2}$，其中 $\boldsymbol{D}=\text{Diag}(d_1,\ldots,d_N)$ 为度数矩阵。

GCN 中每层的聚合可被写为
$$
\mathbf{E}^{(k+1)}=\text{ReLU}(\tilde{A}\mathbf{E}^{(k)}\mathbf{W}^{(k)})
$$
我们略去非线性函数 ReLU，把 GCN 简化为
$$
\mathbf{E}^{(k+1)}=\tilde{A}\mathbf{E}^{(k)}\mathbf{W}^{(k)}
$$
从而最终的 embedding 为
$$
\mathbf{E}^{(K)}=\tilde{A}^K\mathbf{E}^{(0)}(\mathbf{W}^{(0)}...\mathbf{W}^{(K-1)})=\tilde{A}^K\mathbf{E}\mathbf{W}
$$
非常简便。

我们随后使用多尺度扩散
$$
\mathbf{E}_{\text{final}}=\alpha_0\mathbf{E}^{(0)}+\alpha_1\mathbf{E}^{(1)}+\alpha_2\mathbf{E}^{(2)}+\cdots+\alpha_K\mathbf{E}^{(K)}
$$
其中 $\alpha_i$ 为超参数，LightGCN 为简化计算都取 $\frac{1}{K+1}$。

为什么简化扩散传播 work well？它直接鼓励相似的用户/物品具有相似 embedding。

LightGCN 比基于浅 embedding 的传统协同过滤法多了一个使用 $\tilde{A}$ 的扩散；比 NGCF 少了多层 GNN 中的可学习参数。**但惊人的是，它比两者效果都好。**

# PinSAGE

参考[GraphSAGE](05%20-%20A%20General%20Perspective%20on%20Graph%20Neural%20Networks.md#^bba17e) 。

Pinterest 是一个**超大规模**图片网站，其中的图片被称为 pin。pin 需要被 embed 从而支持相似推荐、搜索、广告等服务。

如何使 GNN 支持百万级推荐系统？我们需要简化计算。

* 共享负样本
	* 在 BPR 中，我们原本对每个用户 $u^*\in \mathbf{U}_\text{mini}$ 都找一个负样本集 $\mathbf{V}_\text{neg}$，但现在我们让小批量中的所有用户都使用同一个负样本集。这样开销可以除以 $|\mathbf{U}_\text{mini}|$。
	* 经验上讲，效果相差不大。
* 困难负样本
	* 工业推荐系统需要细粒度的预测。一共有数百万个物品，但只能推荐给用户 10~100 个。
	* 如果从所有物品中随机选取负样本，这样推荐系统区分起来可能非常容易，因为负样本和正样本相差很大，如正样本为猫咪，负样本为摩托车。
	* 我们希望能够选取困难负样本，如正样本为猫咪，负样本为狗。
	* 困难负样本最好循序渐进地增加，如在第 $n$ 个 epoch 中添加 $n-1$ 个困难负样本。
	* 具体而言，负样本应该离用户很近，但不相连（或者说不特别近）。
	* 具体算法
		1. 为每个用户 $u$ 计算个性化 page rank（PPR）
		2. 把物品根据 PPR 从大到小排序
		3. 选取排名靠前但也没那么前的物品（如 2000 到 5000 名）

