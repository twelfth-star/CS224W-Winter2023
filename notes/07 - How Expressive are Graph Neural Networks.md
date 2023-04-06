# Introduction

GNN 的表达性越强（即产生的 embedding 越能区分不同图结构），效果就越好。

GNN 产生的节点 $u$ 的 embedding 由 $u$ 的计算图决定。计算图本质上是以 $u$ 为根的子树结构。表达性强的 GNN 应该能把不同子树结构映射到不同的 embedding，即应尽可能（在每一层都做到）单射。**进一步说，表达性强的 GNN 应该使用单射的邻域聚合函数**。

# Designing the Most Powerful Graph Neural Network

邻域聚合函数作用域一个多重集（不同节点可能有相同的隐藏表示 $\mathbf{h}$）。
* GCN（平均池化）：无法区分各种隐藏表示比例相同的的不同多重集（如 $\{a, b\}$ 和 $\{a,a,b,b\}$）;
* GraphSAGE（最大值池化）：无法区分隐藏表示种类构成相同的多重集（如 $\{a,b\}$ 和 $\{a,b,b\}$）

如何设计一个表达性最强的 GNN？

* Theorem：任何单射多重集函数都可表示为 $\Phi\left(\sum_{x\in S}f(x)\right)$ ，其中 $\Phi$ 和 $f$ 都是线性函数。
* Theorem：含有 1 个隐藏层且隐藏层维度数充分大、激活函数合适（包括 ReLU 和 sigmoid）的 MLP 可以以任意精度近似任意连续函数。（通用近似定理）

因此，我们可以用 MLP 来近似 $\Phi$ 和 $f$ 。
$$
\text{MLP}_{\Phi}\left(\sum_{x\in S}\text{MLP}_{f}(x)\right)
$$
这被称作图同构网络（Graph Isomorphism Network, GIN）。GIN 的邻域聚合函数是单射的。GIN 是表达性最强的 GNN。

GIN 可以视为 WL Graph Kernel 的发展。

参考[Graph-Level Features and Graph Kernels](02%20-%20Feature%20Engineering%20for%20Machine%20Learning%20in%20Graphs.md#Graph-Level%20Features%20and%20Graph%20Kernels)。WL Graph Kernel 的着色算法：$c^{(k+1)}(v)=\text{HASH}\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u\in N(v)}\right)$，其中 $\text{HASH}$ 为任意哈希函数。

Theorem：以 $\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u\in N(v)}\right)$ 为输入的任意单射函数都可以建模为 $$\text{MLP}_{\Phi}\left ((1+\epsilon)\cdot \text{MLP}_{f}(c^{(k)}(v)))+\sum_{u\in N (v)}\text{MLP}_{f}(c^{(k)}(u))\right)$$其中 $\epsilon$ 是可学习参数。

而如果最初的输入 $c^{(0)}(v)$ 是独热编码形式的，则直接相加就是单射的，不需要 $f$。

因此在 GIN 中
$$
\text{HASH}=\text{GINConv}
$$
$$\text{GINConv}\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u\in N(v)}\right)=\text{MLP}_{\Phi}\left((1+\epsilon)\cdot c^{(k)}(v)+\sum_{u\in N(v)}c^{(k)}(u)\right)$$
即 GIN 的迭代方法是
$$
c^{(k+1)}(v)=\text{GlNConv}\left(\left\{c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u\in N(v)}\right\}\right)
$$
相比于 WL Graph Kernel，GIN 的 embedding 是低维的，粒度更细；而且更新函数的参数可以被学习并用于下游任务。

# When Things Don't Go As Planned

* 对节点 feature 进行正则化
* 使用稳健的 ADAM 作为优化器
* 使用 ReLU 作为激活函数。LeakyReLU、PReLU 也不错。
* 输出层不要放激活函数
* 每一层都要加入偏置项
* embedding 维度数常用 32、64、128
* 仔细进行超参数选择、损失函数选择、权重参数初始化

