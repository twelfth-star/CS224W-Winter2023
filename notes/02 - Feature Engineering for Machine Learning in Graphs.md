# Node-Level Tasks and Features

* Node degree $k_v$
	* 衡量节点 $v$ 的重要性以及结构特征
	* 与节点 $v$ 相连的边的数量
* Node centrality $c_v$
	* 衡量节点 $v$ 的重要性，有不同的定义
	* Eigenvector centrality
		* 节点 $v$ 的重要性由与其相邻的节点的重要性衡量，$c_v=\frac{1}{\lambda}\sum_\limits{u\in N(v)}c_u$，其中 $\lambda$ 为系数
		* 转化得到 $\lambda\mathbf{c}=\mathbf{A}\mathbf{c}$，其中 $\mathbf{A}$ 是邻接矩阵，$\mathbf{c}$ 是 centrality 向量，$\lambda$ 为特征根
		* 选取最大的 $\lambda$ 对应的 $\mathbf{c}$
	* Betweenness centrality
		* 若节点 $v$ 处在其它很多节点之间的最短路上，则其重要
		* $c_{v}=\sum\limits_{s \neq v \neq t} \frac{\#(\text { shortest paths betwen } s \text { and } t \text { that contain } v \text { ) }}{\#(\text { shortest paths between } s \text { and } t \text { ) }}$
	* Closeness centrality
		* 若节点 $v$ 到其它节点的最短路都很短的话，则其很重要
		* $c_v=\dfrac{1}{\sum_{u\neq v}\text{shortest path length between } u \text{ and } v}$
* Clustering coefficient $e_v$
	* 衡量与 $v$ 相邻的节点之间的连通性
	* $e_v=\frac{\#(\text{edges among neighbourhood nodes})}{\binom{k_v}{2}}=\frac{2\#(\text{triangles through }v)}{k_v(k_v-1)}\in[0,1]$
	* $e_v$ 其实就是 $v$ 的邻域这一集合内真正存在的边的数量除以可以存在的边的数量
* Graphlet Degree Vector (GDV)
	* 以 $v$ 为根节点的不同种的 graphlets 的数量构成的向量
	* 这里 graphlets 指 rooted connected induced non-isomorphic subgraphs
		* induced：包含原图 $G$ 的部分节点以及这些节点间的全部边
		* rooted：如果两个 graphlets 形状相同但 $v$ 在其中的位置有区别，则也算作不同的
* 总结
	*  分类
		* 基于节点的结构特征：Node degree $k_v$，Clustering coefficient $e_v$，Graphlet Degree Vector (GDV)
		* 基于节点的重要性：Node degree $k_v$，Node centrality $c_v$
	* 应用
		* 预测图中有影响力的节点（如社交圈中的名人）
	* 缺陷
		* 无法表征节点离得远近这一特点

# Link Prediction Task and Features

* 讨论的不是边，而是两个节点 $v_1$ 和 $v_2$ 之间的关系
* Distance-Based Features
	* $v_1$，$v_2$ 之间的最短路长度
	* 缺陷：无法衡量两节点的相邻节点重合度
* Local Neighbour Overlap
	* 表征 $v_1$ 和 $v_2$ 的公共相邻节点
	* Common neighbours：$\begin{aligned}|N(v_1)\cap N(v_2)|\end{aligned}$
	* Jaccard's coefficient：$\frac{|N(v_1)\cap N(v_2)|}{|N(v_1)\cup N(v_2)|}$
	* Adamic-Adar index：$\sum_{u\in N(v_1)\cap N(v_2)}\frac{1}{\log(k_u)}$
	* 缺陷：若两节点不相邻则值为 0
* Global Neighbour Overlap
	* Katz index $S_{v_1 v_2}$
		* 为节点 $v_1$ 和 $v_2$ 之间的不同长度的 walks 的数量的加权平均
		* 线代知识：$P_{v_1v_2}^{(k)}=\mathbf{A}_{v_1v_2}^k$，其中 $P_{v_1v_2}^{(k)}$ 表示 $v_1$ 和 $v_2$ 之间长度为 $k$ 的 walk 的数量
		* $S_{v_1v_2}=\sum\limits_{l=1}^{\infty}\beta^l\mathbf{A}^l_{v_1v_2}$，其中 $\beta\in(0,1)$ 为衰减系数
		* 矩阵形式： $\boldsymbol{S}=\sum\limits_{i=1}^\infty\beta^i\boldsymbol{A}^i=(I-\beta\boldsymbol{A})^{-1}_i-I,$

# Graph-Level Features and Graph Kernels

* Graph Kernels $K(G_1, G_2)$
	* 衡量两个图 $G_1$ 和 $G_2$ 的相似度
	* 总体思想：类似于词袋模型，将节点按照某标准分类（如度数），随后各类节点数量组成的向量便是 graph feature vector $\phi(G)$，而 $K (G_1, G_2)=\phi (G_1)^{\text{T}}\phi(G_2)$
	* Graphlet Kernel
		* 分类标准为途中不同 graphlet 的个数
		* 注意：与 node feature 中的 graphlet 不同，这里的 graphlet 不必为 connected，也不必为 rooted
		* 大小为 $k$ 的 graphlet 的列表 $G_k=(g_1,g_2, ..., g_{n_k})$，graphlet 计数向量 $(f_G)_i=\#(g_i\subseteq G)\text{ for }i=1,2,\ldots,n_k$
		* ![](assets/Pasted%20image%2020230228022539.png)
		* 考虑到 $f$ 会随图 $G$ 的大小变化，我们令 $\boldsymbol{h}_G=\dfrac{\boldsymbol{f}_G}{\text{Sum}(\boldsymbol{f}_G)}$，从而 $K(G_1,G_2)=\mathbf{h}_{G_1}^{\text{T}}\mathbf{h}_{G_2}$
		* 缺陷：计算量过大，为 $n^k$，且难以降低
	* Weisfeiler-Lehman Kernel (WL Kernel)
		* color refinement 方法
			* 将所有节点 $v$ 着色为 $c^{(0)}(v)$
			* 不断迭代更新 $c^{(k+1)}(v)=\operatorname{HASH}\left(\left\{c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u\in N(v)}\right\}\right)$，其中 $\text{HASH}$ 只需将不同的输入映射到不同颜色即可
			* 在第 $K$ 次时停止，此时 $c^{(K)}(v)$ 总结了 $K$ 跳相邻节点的结构
		* 最后结果 $\phi(G)$ 为不同颜色的计数向量（颜色在迭代过程中的出现次数也算）
		* 优点：易于计算