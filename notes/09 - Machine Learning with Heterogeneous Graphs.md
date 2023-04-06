# Hetergeneous Graphs

异构图（hetergeneous graph）：有不同种类节点和不同种类的边的图。

异构图可以表示为 $G=(V,E,\tau,\phi)$
* 节点 $v\in V$ 的类型为 $\tau(v)$
* 边 $e=(u, v)\in E$ 的类型为 $\phi(u,v)$
* 边 $e=(u,v)$ 的关系（relation）$r(u,v)=(\tau(u),\phi(u,v),\tau(v))$

有时我们可以直接把节点的类型通过独热编码添加到 feature 里去从而将异构图转化为标准图（同构图）。但更多时候不行。因为不同类型的节点可能连 feature 的数量都不一样，而且不同的关系类型有不同的物理意义。

# Relational GCN (RGCN)

同构图上的 GCN：$\quad\textbf{h}_v^{(l)}=\sigma\left(\textbf{W}^{(l)}\sum\limits_{u\in N(v)}\frac{\textbf{h}_u^{(l-1)}}{|N(v)|}\right)$

如何拓展 GCN 到异构图？

我们对于不同的关系类型 $r_i$ 使用不同的神经网络权重 $W_{r_i}$。

* Relational GCN（RGCN） 
$$\textbf{h}_v^{(l+1)}=\sigma\left(\sum_{r\in R}\sum_{u\in N_v^r}\dfrac{1}{c_{v,r}}\textbf{W}_r^{(l)}\textbf{h}_u^{(l)}+\textbf{W}_0^{(l)}\textbf{h}_v^{(l)}\right)$$
* 其中 $c_{v,r}=|N_v^r|$
* 消息函数
	* 对于关系为给定类型的相邻节点：$\mathbf{m}_{u,r}^{(l)}=\frac{1}{c_{v,r}}\mathbf{W}_{r}^{(l)}\mathbf{h}_{u}^{(l)}$
	* 自环：$\mathbf{m}_{u, r}^{(l)}=\mathbf{W}_0^{(l)}\mathbf{h}_v^{(l)}$
* 聚合函数
	* $\mathbf{h}_{v}^{(l+1)}=\sigma\left(\text{Sum}\left(\left\{\mathbf{m}_{u,r}^{(l)},u\in N(v)\right\}\cup\left\{\mathbf{m}_{v}^{(l)}\right\}\right)\right)$

参数量会随着关系类型数量增长而迅速增长！可能导致过拟合。如何压缩参数量？

* Solution 1：块对角矩阵（Block Diagonal Matrices）
	* 限制 $\mathbf{W}_r$ 为块对角矩阵
	* 若每个对角块的维度为 $B$，则 $\mathbf{W}_r$ 的大小从 $d^{(l+1)}\times d^{(l)}$ 降低到 $B\times\frac{d^{(l+1)}}{B}\times\frac{d^{(l)}}{B}$，其中 $d^{(l)}$ 为第 $l$ 层的隐藏表示的维度
* Solution 2：基学习（Basis Learning）
	* 在不同关系间共享权重
	* 将 $\mathbf{W}_r$ 表示为一系列基矩阵的线性组合，即 $\textbf{W}_r=\sum_{b=1}^B a_{rb}\cdot\textbf{V}_b$

* Example 1：节点分类
	* **这里的节点分类和节点 $v$ 的类型 $\tau(v)$ 不是一个东西！**
	* 如果共有 $k$ 类，直接让最后一层输出 $\textbf{h}^{(L)}\in\mathbb{R}^k$
* Example 2：链接预测
	* **注意：这里的图是有向图**
	* 参考[链接预测](06%20-%20GNN%20Augumentation%20and%20Training.md#^d58d0f)，将边分成四份（训练消息、训练监督、验证和测试），每份中都有所有类型 $r$ 的边
	* RGCN 的最终输出： $\mathbf{h}^{(L)}_v \in \mathbb{R}^d$
	* 特定关系评分函数： $f_{r_i}:\mathbb{R}^d\times \mathbb{R}^d \rightarrow \mathbb{R}$，如 $f_{r_1}(\textbf{h}_E,\textbf{h}_A)=\textbf{h}_E^TW_{r_1}\textbf{h}_A,\textbf{W}_{r_1}\in\mathbb{R}^{d\times d}$
	* 训练
		1. 假设 $(E, r_3, A)$ 是训练监督边，其它是训练消息边
		2. 对训练消息边集用 RGCN 计算 $f_{r_3}(\mathbf{h}_E,\mathbf{h}_A)$
		3. 创建负边（负边不在训练监督边集和是训练消息边集中，**可以认为是不存在的边**），简单地说就是把 $(E,r_3,A)$ 中的目标节点 $A$ 换成别的和 $E$ 无 $r_3$ 关系的节点，这里假设负边有 $(E,r_3,B)$ 和 $(E,r_3,F)$
		4. 计算负边的评分，即 $f_{r_3}(\mathbf{h}_E,\mathbf{h}_B)$ 和 $f_{r_3}(\mathbf{h}_E,\mathbf{h}_F)$
		5. 将交叉熵作为损失函数，从而最大化训练监督边的评分，最小化负边的评分。
			*  这里 $\ell=\text{CE}((1,0,0),(f_{r_3}(\mathbf{h}_E,\mathbf{h}_A),f_{r_3}(\mathbf{h}_E,\mathbf{h}_B),f_{r_3}(\mathbf{h}_E,\mathbf{h}_F)))$
			* 若只有一个负边 $(E,r_3,B)$：$\ell=-\log\sigma\left (f_{r_3}(\mathbf{h}_E,\mathbf{h}_A)\right)-\log \big(1-\sigma\big (f_{r_3}(\mathbf{h}_E,\mathbf{h}_B)\big)\big)$ 
	* 验证
		1. 假设 $(E,r_3,D)$ 为验证边
		2. 对训练消息边集+训练监督边集用 RGCN 计算 $f_{r_3}(\mathbf{h}_E,\mathbf{h}_D)$
		3. 创建负边并计算其评分，这里是 $f_{r_3}(\mathbf{h}_E,\mathbf{h}_B)$ 和 $f_{r_3}(\mathbf{h}_E,\mathbf{h}_F)$
		4. 将以上所有评分排序，记 $\text{RK}$ 为 $(E,r_3,D)$ 的排名
		5. 计算 metrics
			1. $\mathbf{1}[\text{RK} \le k]$：越大越好
			2. $\frac{1}{\text{RK}}$：越大越好

# Heterogeneous Graph Transformer

同构图上的 GAT：$\textbf{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}{\alpha_{vu}}\textbf{W}^{(l)}\textbf{h}_u^{(l-1)})$ ，参考 ![同构图的GAT](05%20-%20A%20General%20Perspective%20on%20Graph%20Neural%20Networks.md#^0ec8bb)
如何扩展 GAT 到异构图？

异构图 Transformer（HGT）使用缩放点积注意力，参考 [缩放点积注意力](计算机/课程/Dive%20into%20Deep%20Learning/Dive%20into%20Deep%20Learning.md#缩放点积注意力)。
$$
\text{Attention}(Q,K,V)=\text{softmax}(\dfrac{QK^T}{\sqrt{d_k}})V
$$
其中 $Q$ 为查询，$K$ 为键，$V$ 为值，三者形状都为 `(batch_size, dim)`。我们通过线性层得到 $Q,K,V$：$T=\text{T-Linear} (X), \quad T\in \{Q, K, V\}$。

我们希望注意力能考虑关系类型 $r$ 这一信息，但又不希望参数量过大：将异构图注意力分解为节点类型相关和边类型相关。

考虑节点 $s,t$ 和边 $e$ 构成的关系 $r(u,v)=(\tau(u),\phi(u,v),\tau(v))$。

$$
\text{ATT-head}^i(s,e,t)=\left(K^i(s)W_{\phi(e)}^{\text{ATT}}Q^i(t)^T\right)
$$
$$
K^i(s)=\operatorname{K-Linear}^i_{\tau(s)}\left(H^{(l-1)}[s]\right)
$$
$$
Q^i(t)=Q\text{-Linear}^i_{\tau(t)}\Big(H^{(l-1)}[t]\Big)
$$
这里我们看到 $\text{K-Linear}$ 只由注意力头编号 $i$ 和节点 $s$ 的类型 $\tau(s)$ 决定，$\text{Q-Linear}$ 也类似。$W_{\phi(e)}^{\text{ATT}}$ 则只由边的类型 $\phi (e)$ 决定。因此注意力机制被分解了。

而整体上
$$
\widetilde{H}^{(l)}[t]=\underset{\forall s\in N(t)}{\bigoplus}\biggr({\textbf{Attention}_{\text{HGT}}(s,e,t)}\cdot\textbf{Message}_{\text{HGT}}(s,e,t)\biggr)
$$
Attention 部分已经求出，现在来看 Message 部分。

与 Attention 一样，HGT 将 Message 部分也分解到节点类型相关和边类型相关。

$$
\textbf{Message}_{\text{HGT}}(s,e,t)=\underset{i\in[1,h]}{||}\text{MSG-head}^i(s,e,t)
$$
$$
\text{MSG-head}^i(s,e,t)=\text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{\text{MSG}}_{\phi(e)}
$$

# Design Space of Heterogeneous GNNs

如何将同构图的 GNN 设计空间扩展到异构图?

* 消息计算
	* 同构图
		* $\textbf{m}_u^{(l)}=\text{MSG}^{(l)}\left(\textbf{h}_u^{(l-1)}\right)$
		* 具体实现：$\textbf{m}_u^{(l)}=\textbf{W}^{(l)}\textbf{h}_u^{(l-1)}$
	* 异构图
		* 不同的关系类型应该有不同的消息函数
		* $\mathbf{m}_{u}^{(l)}=\text{MSG}_{r}^{(l)}\left(\mathbf{h}_{u}^{(l-1)}\right),r=(u,e,v)$
		* 具体实现：$\textbf{m}_u^{(l)}=\textbf{W}_r^{(l)}\textbf{h}_u^{(l-1)}\quad$
* 聚合计算
	* 同构图
		* $\mathbf{h}_v^{(l)}=\text{AGG}^{(l)}\left(\left\{\mathbf{m}_u^{(l)},u\in N(v)\right\}\right)$
		* 具体实现：$\textbf{h}_v^{(l)}=\text{Sum}(\{\textbf{m}_u^{(l)},u\in N(v)\})$
	* 异构图
		* 分两步走：先在同一关系类型内部聚合，再聚合不同关系类型
		* $\mathbf{h}_{v}^{(l)}=\mathrm{AG}_{alll}^{(l)}\left(\mathrm{AGG}_{r}^{(l)}\left(\left\{\mathbf{m}_{u}^{(l)},u\in N_{r}(v)\right\}\right)\right)$
		* 具体实现：$\mathbf{h}_{v}^{(l)}=\text{Concat}\Big(\text{Sum}\Big(\big\{\mathbf{m}_{u}^{(l)},u\in N_{r}(v)\big\}\Big)\Big)$
* 层连通性
	* 同构图
		* 跳过连接、预/后处理
	* 异构图
		* 跳过连接和同构图一样
		* 预/后处理要考虑节点类型：$\textbf{h}_v^{(l)}=\text{MLP}_{\tau(v)}(\textbf{h}_v^{(l)})$
* 图操作
	* 同构图
		* 特征增强、添加虚拟节点/边、邻域取样、子图取样
	* 异构图
		* 根据关系类型进行上述操作
* GNN 预测头
	* 同构图
		* 节点级预测：$\widehat{\mathbf{y}}_v=\text{Head}_{\text{node}}(\mathbf{h}_v^{(L)})=\mathbf{W}^{(H)}\mathbf{h}_v^{(L)}$
		* 边级预测：$\widehat{\mathbf{y}}_{uv}=\text{Head}_{\text{edge}} (\mathbf{h}_{u}^{(L)},\mathbf{h}_{v}^{(L)})=\text{Linear}(\text{Contat}(\mathbf{h}_{u}^{(L)},\mathbf{h}_{v}^{(L)}))$
		* 图级预测：$\widehat{\mathbf{y}}_G=\text{Head}_{\text{graph}}(\{\mathbf{h}_v^{(L)}\in\mathbb{R}^d,\forall v\in G\})$
	* 异构图
		*  节点级预测：$\widehat{\mathbf{y}}_v=\text{Head}_{\text{node},\tau(v)}(\mathbf{h}_v^{(L)})=\mathbf{W}_{\tau(v)}^{(H)}\mathbf{h}_v^{(L)}$
		* 边级预测：$\widehat{\mathbf{y}}_{uv}=\text{Head}_{\text{edge},r} (\mathbf{h}_{u}^{(L)},\mathbf{h}_{v}^{(L)})=\text{Linear}_r(\text{Contat}(\mathbf{h}_{u}^{(L)},\mathbf{h}_{v}^{(L)}))$
		* 图级预测：$\widehat{\mathbf{y}}_G=\text{AGG}(\text{Head}_{\text{graph},i}(\{\mathbf{h}_v^{(L)}\in\mathbb{R}^d,\forall \tau(v)=i\}))$


**本质上就是把不同的关系类型分开来建模。**
