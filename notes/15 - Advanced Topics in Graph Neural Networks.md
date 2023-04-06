# Limitation of Graph Neural Network

对于一个完美的 GNN
* 如果两个节点的邻域结构相同，那么 embedding 也相同
	* 未必是好事，有时我们希望有相同邻域结构但位置不同的节点 embedding 不同，如地图
	* Position-aware GNNs
* 如果两个节点的邻域结构不同，那么 embedding 也不同
	* 做不到，无法区分三角形和四边形
	* Identity-aware GNNs

# Position-aware Graph Neural Networks

GNN 在 structure-aware 任务上往往工作得很好，但 position-aware 则不然。

引入锚点（集）。每个节点到各个锚点（集）得距离 characterize 了该节点。

Bourgain 定理：
定义 $f(v)=\left(d_{\text{min}}\big(v,S_{1,1}\big),d_{\text{min}}\big(v,S_{1,2}\big),...,d_{\text{min}}\big(v,S_{\log n,c\log n}\big)\right)\in\mathbb{R}^{c\log^2n}$，其中 $S_{i,j}$ 是从 $V$ 中按照 $\frac{1}{2^{i}}$ 的概率选择节点组成的集合，则 $f(v)$ 定义的 embedding distance 可被证明是接近原本的距离 $(V,d)$ 的。

算法：
1. 随机取样 $O(\log^2 n)$ 个锚点集
2. 通过 $\left(d_{\text{min}}\big(v,S_{1,1}\big),d_{\text{min}}\big(v,S_{1,2}\big),...,d_{\text{min}}\big(v,S_{\log n,c\log n}\big)\right)\in\mathbb{R}^{c\log^2n}$ embed 所有 $v$

一个简单的做法是把这个 embedding 作为增强的节点特征，但并不合理，因为 embedding 的顺序是可以调换的（因为锚点集是随机取样的），但训出来的 NN 显然不能允许节点顺序改变。

需要特殊的 NN 保留排列不变性。具体参考 Position-aware GNN 论文。 

# Identity-Aware Graph Neural Networks

传统 GNN 在点/边/图级的任务上都可能失败。失败的本质是对于不同的输入产生了相同的计算图。

思想：将我们想要 embed 的节点上色，从而计算图就可以被区分了。

![](assets/Pasted%20image%2020230331145523.png)

那么如何实际地对节点上色法进行建模呢？异构信息传递。ID-GNN 对不同颜色的节点使用不同的信息/聚合函数。参考 [09 - Machine Learning with Heterogeneous Graphs](09%20-%20Machine%20Learning%20with%20Heterogeneous%20Graphs.md)。



ID-GNN 的优势在于可以对给定节点所在的环进行计数。一个简化的版本 ID-GNN-Fast 直接将 identity 信息与所在环的个数作为增强节点特征。

# Robustness of Graph Neural Networks

以通过 GCN 进行半监督结点分类任务为例，考察对抗攻击。

定义目标节点 $t$ 为攻击者想改变标签预测的节点，攻击节点集 $S$ 为攻击者可以改变的节点集。

直接攻击：若 $S=\{t\}$，则可以改变 $t$ 的特征、添加/去除 $t$ 与另一节点之间的边。

间接攻击：若 $t \notin S$，则可以改变 $s$ 的特征、添加/去除 $s$ 与另一节点之间的边。

攻击的图操作必须要小（不容易被发现），且能够在最大程度上改变目标节点的预测标签。

记  $A'$ 与 $X'$ 为修改后的邻接矩阵和特征矩阵。应当保证 $(A', X')\approx(A,X)$，且相关图的统计量不改变。记 $Y$ 为节点标签。

记 $\theta^*$ 为通过 $A,X,Y$ 学到的参数，$c_v^\ast$ 为通过 $\theta^\ast$ 预测的 $v$ 的标签。记 $\theta^{\ast\prime}$ 为通过 $A',X',Y$ 学到的参数，$c_v^{\ast\prime}$ 为通过 $\theta^{*'}$ 预测的 $v$ 的标签。目标是 $c_v^\ast\neq c_v^{\ast\prime}$。

对于 $v$ 的预测的改变量：$\Delta (v; A^\prime, X^\prime)=\log f_{\boldsymbol{\theta}^{\ast\prime}}({A}',{X}')_{v, c_{v}^{\ast\prime}}-\log f_ {{{\theta}^{*\prime}} }{({A}^{\prime},{X}^{\prime})}_{v, c_{v}^{\ast}}$

前者为新标签的预测概率，越大越好；后者为原标签的预测概率，越小越好。

故要最大化 $\Delta (v; A^\prime, X^\prime)$。算法：不断地进行最有价值的图操作（即改变 $A$ 或 $X$ 中的某项）。

GCN对对抗性攻击不健壮，但对间接攻击和随机噪声有一定的健壮性。