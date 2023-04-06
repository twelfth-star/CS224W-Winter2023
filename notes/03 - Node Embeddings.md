# Encoder and Decoder

手动特征工程构造 feature 太麻烦了，我们可以通过 Graph Representation Learning 自动生成 embeddings $\bf{z}\in \mathbb{R}^d$。

目标在于节点在 embedding 空间里的相似性（比如内积）可以近似表示节点在图中的相似性。即 $\text{similarity}(u, v)\approx \mathbf{z}_v^{\text{T}}\mathbf{z}_u$。

Encoder 将节点映射到 embedding 空间，即 $\mathbf{z}_v=\text{ENC}(v)$。最简单的 Encoder 是 embedding-lookup，或者说浅embedding，就是直接给每个节点 $v$ 指定一个 embedding。即 $\text{ENC}(v)=\mathbf{z}_v=\mathbf{Z}\cdot v$，其中 $\mathbf{Z}\in\mathbb{R}^{d\times |V|}$ 为要学习的参数，$v\in\mathbb{I}^{|V|}$ 为独热编码。 ^934fbd

Decoder 计算 embedding 之间的相似度，比如内积。

现在问题在于如何定义 $\text{similarity}$ 函数（即节点在图中的相似度）。

# Randomo Walk Approaches for Node Embeddings

用从 $u$ 出发的随机游走（walk，图论概念）经过 $v$ 的概率 $P$ 表示节点 $u$ 和 $v$ 的相似度。

兼顾 expressivity（捕获局部和高阶邻域信息）和 efficiency（只需考虑随机游走）。

定义 $N_R(u)$ 为从 $u$ 开始的随机游走策略 $R$ 得到的邻域，$f: u\rightarrow\mathbb{R}^d$ 为节点到 embedding 的映射。则目标为
$$\max\limits_{f}\sum\limits_{u\in V}^{P}\log P(N_{\text{R}}(u)\big|\mathbf{z}_{u})$$
其中 $\log$ 是因为概率要相乘，这里要化乘积为求和。

算法： $N_R(u)$ 初始化为空集，从 $u$ 出发进行定长随机游走，将经过的节点加入 $N_R(u)$。$N_R(u)$ 为一 multiset，多次经过则多次加入。最终的损失函数为
$$
\begin{aligned}\mathcal{L}=\sum_{u\in V}\sum_{v\in N_R(u)}-\log(P(v|\textbf{z}_u))\end{aligned}
$$
其中 $P(v|\textbf{z}_u)$ 由 softmax 得到，因为要突出相似度最大的那个
$$
\begin{aligned}P(v|\mathbf{z}_u)=\frac{\exp(\mathbf{z}_u^{\text{T}}\mathbf{z}_v)}{\sum_{n\in V}\exp(\mathbf{z}_u^{\text{T}}\mathbf{z}_n)}\\ \end{aligned}
$$
**很符合直觉，从 $u$ 出发随机游走，容易遇见的节点 $v$ 在 $N_R(u)$ 中的出现次数就多，权重就大，这就要求要优先最大化 $P(v|\mathbf{z}_u)$，从而让要求优先最大化两者 embedding 的相似度，也即内积 $\mathbf{z}_u^{\text{T}}\mathbf{z}_v$。**

然而这一算法开销太大，复杂度为 $O(|V|^2)$。我们可以通过负采样优化 $\log (\text{softmax})$。
$$
\begin{aligned}\log(\frac{\exp\left(\mathbf{z}_u^{\text{T}}\mathbf{z}_v\right)}{\sum_{n\in V}\exp\left(\mathbf{z}_u^{\text{T}}\mathbf{z}_n\right)})\end{aligned}\approx\log\left(\sigma\left(\textbf{z}_u^\textbf{T}\textbf{z}_v\right)\right)-\sum_{i=1}^k\log\left(\sigma\left(\textbf{z}_u^\textbf{T}\textbf{z}_{n_i}\right)\right),n_i\sim P_V
$$
其中 $\sigma (x)=\frac{1}{1+\exp(-x)}$ 为 sigmoid 函数。这样我们只需按照概率采样 $k$ 个负节点 $n_i$。$k$ 一般取 5~20.

使用随机梯度下降（SGD）作为优化器。
1. 对所有节点 $u$，将 $\mathbf{z}_u$ 初始化为随机值
2. 不断迭代直到收敛：$\mathcal{L}^{(u)}=\sum_{v\in N_R(u)}-\log(P(v|\mathbf{z}_u))$
	1. 随机采样一点 $u$，对所有节点 $v$ 计算导数 $\frac{\partial L^{(u)}}{\partial z_v}$。
	2. 对所有 $v$，更新 $z_v\leftarrow z_v-\eta\frac{\partial{\mathcal{L}^{(u)}}}{\partial z_v}$。

现在我们考虑游走策略 $R$。最简单的显然是定长无偏随机游走，这就是 DeepWalk 算法。但我们也可以使用有偏（biased）随机游走算法以权衡局部和全局信息，比如 node2vec 算法。

![](assets/Pasted%20image%2020230301161248.png)

相对而言，BFS 捕获局部信息，DFS 捕获全局信息。

定义两个参数
* 返回参数 $p$：返回上一节点
* 入出参数 $q$：BFS vs. DFS 的比例
按照参数分配概率。

![](assets/Pasted%20image%2020230301161559.png)

# Embedding Entire Graphs

法一（简单却有效）：将所有节点的 embedding 求和/求平均，即 $\begin{aligned}\mathbf{z}_G=\sum_{v\in G}\mathbf{z}_v\end{aligned}$

法二：引入虚拟节点，它与要求 embedding 的（子）图中的所有节点相连，计算其 embedding

# Matrix Factorization and Node Embeddings

转为矩阵运算就可以放到 GPU 上跑，效率更高。

如果我们定义图中节点相似度为有边连接为 1，无边连接为 0，则 $\mathbf{Z}^\text{T}\mathbf{Z}\approx\mathbf{A}$ 即要求邻接矩阵的分解。

DeepWalk 也可转为矩阵分解 
$$\log\left(vol(G)\left(\dfrac{1}{T}\sum_{r=1}^T(D^{-1}A)^r\right)D^{-1}\right)-\log b
$$
![](assets/Pasted%20image%2020230301162900.png)

随机游走算法的缺陷：
* 无法获得不在训练集的节点的 embedding。需要重新计算所有 embedding。
* 无法捕获结构相似性。
* 无法利用其它点、边、图的 features