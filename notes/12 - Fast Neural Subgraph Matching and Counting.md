# Subgraphs and Motifs `

子图分类
* 节点导出子图
	* 节点集的子集+该子集导出的边
	* $V'\subseteq V$，$E'=\{(u,v)\in E\mid u,v\in V'\}$
	* 又称“导出子图”
	* 常用于化学
* 边导出子图
	* 边集的子集+该子集中的边对应的节点
	* $E'\subseteq E$，$V'=\{v\in V\mid (v,u)\in E'\text{ for some }u\}$
	* 又称“非导出子图”
	* 常用于KG

子图同构（subgraph isomorphic）：$G_1$ 的一个子图 $G_1'$ 与 $G_2$ 同构（即存在双射 $f: V_1'\rightarrow V_2$ 使得 $(u, v)\in E_1' \Leftrightarrow (f (u), f (v))\in E_2$）。

motif：重复出现的、具有显著性的互联特征。motif 是一种小的**节点导出子图**。

子图频率
* 记 $G_Q$ 为一小图，$G_T$ 为目标图数据集。
* 图级子图频率定义
	* 记 $V_G'\subseteq V_G$ 为导出图与 $G_Q$ 同构的节点集子集
	* $G_Q$ 在 $G_T$ 中的频率为不同 $V_G'$ 的数量
* 节点级子图频率定义
	* 某个 $G_T$ 的子图与 $G_Q$ 同构，且同构函数把节点 $u$ 映射到 $v$
	*  $G_Q$ 在 $G_T$ 中的频率为不同 $u$ 的数量
	* $(G_Q,v)$ 称为节点固定子图（node-anchored subgraph）

零模型（null model）
* 我们用随机图作为零模型，可以在与零模型的比较中体现某 motif 在某图中的显著性
* Erdős–Rényi (ER) 随机图
	* 可以生成与原图 $G^{\text{real}}$ 节点数一样的随机图 $G^{\text{rand}}$
	* $G_{n,p}$：$n$ 表示节点数，$p$ 表示每个边的出现概率
	* 直接构造即可
* 配置模型（configuration model）
	* 可以生成与原图 $G^{\text{real}}$ 的节点数、边数、度数序列一样的随机图 $G^{\text{rand}}$
	* ![](assets/Pasted%20image%2020230308170009.png)
* 交换模型（switching）
	* 可以生成与原图 $G^{\text{real}}$ 的节点数、边数、度数序列一样的随机图 $G^{\text{rand}}$
	* 不断交换随机两条边的终点 $Q\cdot |E|$ 次（若会导致自环或多重边则跳过）
	* ![](assets/Pasted%20image%2020230308170329.png)
	* $Q$ 为超参数，一般取 100 以保证收敛

如何计算 motif 的显著性？比较它在原图和随即图中的出现频率。记 $N_i^{\text{real}}$ 是第 $i$ 种 motif 在原图 $G^{\text{real}}$ 中的出现频率，而 $N_i^{\text{real}}$ 是第 $i$ 种 motif 在随机图 $G^{\text{rand}}$ 中的出现频率。Z-score 的定义是
$$
Z_i=(N_i^{\text{real}}-\overline{N}_i^{\text{rand}})/\text{std}(N_i^{\text{rand}})
$$
由于在大图中 Z-score 会更高，我们进一步计算显著性评分（significance profile，SP），它是标准化的 Z-score。
$$
\text{}SP_i=Z_i/\sqrt{\sum_j Z_j^2}
$$
Z-score 的负值表示 under-representation（即更罕见），正值表示 over-representation（即更多见）。

![](assets/Pasted%20image%2020230308171238.png)

可见 SP 可以反映不同邻域图的特征。非常神奇。

# Neural Subgraph Representations

如何判断一个查询图是不是目标图的子图？这是一个 NP-hard 的二分类问题。

我们使用 GNN 获得快速且近似的答案。我们希望在 embedding 空间中表征几何形状来捕获子图同构性。

我们使用**节点固定**的定义。预测 $u$ 的领域是否与 $v$ 的邻域同构且 $u$ 与 $v$ 相对应。

对于 $G_T$ 的每个节点 $v$，通过 BFS 得到它的 $k$ 跳邻域 $N(v)$，其中 $k$ 为超参数。计算这个邻域中所有节点的 embedding，最后得到 $v$ 的 embedding 作为这个子图 $G_T'$ 的 embedding $\mathbf{z}_{t}$。类似地得到 $G_Q$ 中节点 $u$ 的 embedding 作为 $G_Q$ 的 embedding $\mathbf{z}_{q}$。

这里的 embedding 空间为顺序 embedding 空间，即 $\mathbf{z}[i]\geq0\quad\forall i$，且若图 $G_1$ 为图 $G_2$ 的子图，则 $\mathbf{z}_{1}[i]\leq\mathbf{z}_{2}[i]\quad\forall i$。**这样的顺序 embedding 空间可以捕获偏序关系（传递性、反对称性、封闭性），因此可以很好地表征子图同构关系。**

![](assets/Pasted%20image%2020230308173025.png)

我们要训练一个 GNN 来把图映射到顺序 embedding 空间，因此需要一个合适的损失函数。这个损失函数应基于顺序约束，即 $\forall_{i=1}^D \mathbf{z}_q[i]\leq \mathbf{z}_t[i]\text{ iff }G_Q\subseteq G_T$。我们使用 max-margin 损失函数。

定义
$$
E\big(G_Q,G_T\big)=\sum_{i=1}^D(\text{max}(0,z_q[i]-z_t[i])\big)^2
$$
为 $G_q,G_t$ 之间的 margin。当 $G_Q\subseteq G_T$ （正样本）时 $E=0$ ，当 $G_Q$ 不是 $G_T$ 的子图时（负样本） $E>0$。

我们构建数据集使得一半的数据中 $G_Q\subseteq G_T$，另一半中则不然。使用这些数据进行训练，最小化 max-margin 损失函数 $\ell$。
正样本：$\ell=E(G_Q,G_T)$，即 $E$ 越小越好
负样本：$\ell=\max(0,\alpha - E(G_Q, G_T))$，即 $E$ 越大越好
这样的损失函数可以防止无限地把 embedding 分得越来越开。

样本的构建
* 随机选取 anchor 节点 $v$，并将 $G$ 中所有与 $v$ 距离在 $K$ 以内的节点构成的节点集合导出的子图作为 $G_T$
* 正样本
	1. 初始化 $S=\{v\}, V=\emptyset$
	2. 令 $N(S)=\{v\in N(u) \mid u \in S\}$ ，从 $N (S)\backslash V$ 中取样 10%放入 $S$ 中，将剩下的放入 $V$ 中
	3. 迭代 $K$ 次后停止，输出 $S$ 导出以 $q$ 为 anchor 的 $G$ 的子图作为 $G_Q$，此时显然 $G_Q\subseteq G_T$
* 负样本
	* 随机在正样本的 $G_Q$ 中增加/减少边/节点从而破坏该样本


# Finding Frequent Subgraphs

如何找到频率最高的大小为 $k$ 的 motif？
1. **枚举**所有大小为 $k$ 的连通子图
2. **计数**这些子图的出现频率
这计算起来极度困难，因为 motif 种数随 $k$ 呈指数增长，且频率计数是 NP-hard 的。

我们可以用 GNN 去预测子图频率，并递增地构建大小为 $k$ 的子图。

对于目标图（数据集）$G_T$，我们希望识别出在所有节点数为 $k$ 的子图中，在 $G_T$ 中出现频率最高的 $r$ 个。这里我们使用节点级定义。


![](assets/Pasted%20image%2020230308185330.png)
SPMiner 算法
1. 把 $G_T$ 划分为多个邻域
	* 即随机在 $G_T$ 取样一组子图（节点固定邻域）$G_{N_i}$
	* 核心思想是通过数有多少个 $G_{N_i}$ 的 embedding $\mathbf{z}_{N_i}$ 满足 $\mathbf{z}_Q\leq \mathbf{z}_{N_i}$ 来估计 $G_Q$ 的频率，速度非常快
2. 将每个邻域 embed 到一个顺序 embedding 空间
3. 随机在图 $G_T$ 中取一个节点 $u$，令 $S=\{u\}$，记以 $u$ 为 anchor 的 $S$ 导出的子图为 $G_Q$。此时显然 $G_Q\subseteq G_{N_i}\quad \forall i$，$\mathbf{z}_Q$ 在左下角。
4. 不断从 $N(S)$ 中选取一个节点添加到 $S$ 中，从而把 $G_Q$ 的大小增加到 $k$ 个。
	* 由于我们要找到频率最高的，因此我们要在选取要添加的节点时尽可能使新的 embedding 留在靠左下的位置，即让淡红色矩形尽可能大。
	* ![](assets/Pasted%20image%2020230308190819.png)
	* 具体而言使用贪心法，添加产生最少的 total violation 的节点。其中 $G$ 的 total violation 指不包含 $G$ 的邻域 $G_{N_i}$ 的数量，即 $|\{G_{N_i}\mid \text{ not }\mathbf{z}_Q \preceq\mathbf{z}_{N_i}\}|$。
