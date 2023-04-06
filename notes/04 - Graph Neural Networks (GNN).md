# Basics of Deep Learning

略，参见 [多层感知机（MLP）](Dive%20into%20Deep%20Learning/Dive%20into%20Deep%20Learning.md#多层感知机（MLP）)。

# Deep learning for graphs

定义 $\mathbf{X}\in\mathbb{R}^{|V|\times d}$ 为节点特征矩阵，通常为外部信息比如用户画像、基因功能等，没有外部信息则为独热编码。图 $G=(\mathbf{A},\mathbf{X})$。

**不像图片，图没有标准节点顺序，因此图和节点的 representations 应保证不随节点顺序变化而变化，不然训练出来之后训练集中的数据变一下节点顺序都不对了。**

排列不变性（Permutation Invariance）：函数 $f$ 将图 $G=(\mathbf{A},\mathbf{X})$ 映射为 embedding $f (G)\in \mathbb{R}^d$，若 $f (\mathbf{A},\mathbf{X})=f (P\mathbf{A}P^T, P\mathbf{X})$，其中 $P$ 为排列矩阵，则 $f$ 具有排列不变性。

排列等变性（Permutation Equivariance）：函数 $f$ 将图 $G$ 中的所有节点 $u$ 映射为 embedding $\mathbf{z}_u \in \mathbb{R}^d$，也即将图 $G$ 映射为矩阵 $\mathbb{R}^{m\times d}$，若 $Pf (\mathbf{A},\mathbf{X})=f (P\mathbf{A}P^T, P\mathbf{X})$，其中 $P$ 为排列矩阵，则 $f$ 具有排列等变性。

**排列不变性就是同一图，不同节点顺序做 representation 还一样；排列等变性就是同一节点，以不同节点顺序做 representation 还一样。**

GNN 包含多个满足排列不变性/排列等变性的函数。

# Graph Convolutional Networks

核心思想：从相邻节点中聚合信息。

基本算法：
1. 初始化：$h_{v}^{0}=x_{v}$
2. 迭代：$h^{(k+1)}_v=\sigma(W_k\sum_{u\in N(v)}\dfrac{h^{(k)}}{|N(v)|}+B_kh^{(k)}_v),\forall k\in\{0,\ldots,\mathbb{N}-1\}$，其中 $\sigma$ 为激活函数，$K$ 为迭代次数，$W_k$ 为可学习参数
3. 输出：$\mathbf{z}_v=h_v^{(K)}$

很好理解，节点 $u$ 在第 $k$ 层的编码为参数矩阵 $W_k$ 乘以它的相邻节点在 $k-1$ 层编码的平均，再加上参数矩阵 $B_k$ 乘以它本身在 $k-1$ 层的编码，最后通过激活函数。

显然此法顺序无关，因此满足排列不变性/排列等变性。

我们需要把算法矩阵化以适应 GPU 运算。

1. 令 $H^{(k)}=[h_{1}^{(k)}...h_{|V|}^{(k)}]^{\text{T}}$
2. 注意到 $\sum_{u\in N_v}h_{u}^{(k)}=\mathbf{A}_{v,:}\mathop{\text{H}}^{(k)}$
3. 令 $D_{v,v}=\text{Deg}(v)=|N(v)|$
4. 则 $H^{(k+1)}=D^{-1}\mathbf{A}H^{(k)}$，即 $H^{(k+1)}=\sigma(\tilde{A}H^{(k)}W_k^{\text{T}}+H^{(k)}B_k^{\text{T}})$，其中 $\tilde{A}=D^{-1}A$ 为一稀疏矩阵

关于损失函数：**若为监督学习，则使用相应损失函数**；若为无监督，则需要利用图本身的结构构造损失函数。

$$
\mathcal{L}=\sum_{z_u,z_v}\mathrm{CE}(y_{u,v},\mathrm{DC}(z_u,z_v))
$$
其中 $y_{u,v}$ 若 $u$ 和 $v$ 相似则为 1，否则为 0。是否相似可有随机游走等方法确定，参考 02 节。$\text{CE}$ 为交叉熵损失函数。

由于同一层所有节点参数共享，训练结果可以拓展到未见过的新图。

# GNNs subsume CNNs

![](assets/Pasted%20image%2020230301172116.png)

CNN 可被视为相邻点的尺寸与顺序固定的特殊 GNN。它可以为节点周围与本身共 8 个位置都赋予一个共享参数，因此不满足排列不变性/排列等变性。

参考[卷积神经网络（CNN）](Dive%20into%20Deep%20Learning/Dive%20into%20Deep%20Learning.md#卷积神经网络（CNN）)。