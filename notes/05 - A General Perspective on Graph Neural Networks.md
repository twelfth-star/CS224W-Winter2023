# A Single Layer of a GNN

GNN 层将一组向量压缩为一个向量，主要分消息和聚合两步。

每个节点生成一个消息传递给下一层的节点。$\textbf{m}_u^{(l)}=\text{MSG}^{(l)}\left(\textbf{h}_u^{(l-1)}\right)$，其中 $\text{MSG}$ 为消息函数。对于线性层，消息函数为 $\text{MSG}^{(l)}(\textbf{h}_u^{(l-1)})=\mathbf{W}^{(l)}\textbf{h}_u^{(l-1)}$。

每个节点聚合其相邻节点传来的消息。$\mathbf{h}_{v}^{(l)}=\text{AGG}^{(l)}\left(\left\{\mathbf{m}_{u}^{(l)},u\in N(v)\right\}\right)$ ，其中 $\text{AGG}$ 为聚合函数。聚合函数可以为求和，求平均，求最大值。

但这时节点 $v$ 本身的信息在传递给下一层时可能会丢失。为了突出自身信息，做如下修正。$\textbf{m}_v^{(l)}=\textbf{B}^{(l)}\textbf{h}_v^{(l-1)}$，$\mathbf{h}_v^{(l)}=\text{CONCAT}\left(\text{AGG}\left(\left\{\mathbf{m}_u^{(l)},u\in N(v)\right)\right);\mathbf{m}_v^{(l)}\right)$，其中 $\text{CONCAT}$ 为拼接函数。

具体而言

* Graph Convolutional Networks (GCN)
	* $\textbf{m}_u^{(l)}=\frac{1}{|N(v)|}\textbf{W}^{(l)}\textbf{h}_u^{(l-1)}$
	* $\mathbf{h}_{v}^{(l)}=\sigma\left(\text{Sum}\left(\left\{\mathbf{m}_{u}^{(l)},u\in N(v)\right\}\right)\right)$
* GraphSAGE ^bba17e
	* $\mathbf{h}_{v}^{(l)}=\sigma\left(\mathbf{W}^{(l)}\cdot\text{CONCAT}\left(\mathbf{h}_{v}^{(l-1)},\text{AGG}\left(\left\{\mathbf{h}_{u}^{(l-1)},\forall u\in N(v)\right\}\right)\right)\right)$
	* 分两步：先聚合相邻节点，再和节点 $v$ 本身拼接起来
	* 聚合函数 $\text{AGG}$ 取值
		* 平均：$\text{AGG}=\sum\limits_{u\in N(v)}\frac{\mathbf h_u^{(l-1)}}{|N(v)|}$
		* 池化：$\text{AGG}=\text{Mean}(\{\text{MLP}(\textbf{h}^{(l-1)}u),\forall u\in N(v)\})$
		* LSTM：$\text{AGG}=\text{LSTM}([\mathbf{h}_u^{(l-1)},\forall u\in\pi\big(N(v)\big)])$
	* $l_{2}$ 正则化：$\textbf{h}_v^{(l)}\leftarrow\frac{\textbf{h}_v^{(l)}}{\left\|\textbf{h}_v^{(l)}\right\|_2}\quad\forall v\in V\,$，保证数量级相同
* Graph Attention Networks (GAT) ^0ec8bb
	* $\textbf{h}_v^{(l)}=\sigma(\sum_{u\in N(v)}\alpha_{vu}\textbf{W}^{(l)}\textbf{h}_u^{(l-1)})$
	* 在 GNN 和 GraphSAGE 里，$\alpha_{vu}=\frac{1}{|N(v)|}$，但实际上不同的相邻节点其实重要性不同。我们希望这一权重可以被学习。
	* 定义 $e_{vu}=a(\textbf{W}^{(l)}\textbf{h}_u^{(l-1)},\textbf{W}^{(l)}\textbf{h}^{(l-1)}_v)$，它表征来自 $u$ 的信息对 $v$ 的重要性。其中 $a$ 为注意力机制函数
	* 通过 softmax 把 $e_{vu}$ 转为权重 $\alpha_{vu}=\dfrac{\text{exp}(e_{vu})}{\sum_{k\in N(v)}\text{exp}(e_{vk})}$
	* 一个简单的注意力机制实例： $a(A,B)=\text{LINEAR}(\text{CONCAT}(A,B))$ ，其中 CONCAT 就是简单的向量拼接，LINEAR 的参数是可学习的。
	* 多头注意力机制 
		* $\textbf{h}^{(l)}_v[i]=\sigma (\sum_{u\in N (v)}\alpha^{(i)}_{vu}\textbf{W}^{(l)}\textbf{h}^{(l-1)}_u)\quad i=1,2,...,n_a$
		* $\textbf{h}_v^{(l)}=\text{AGG}(\textbf{h}_v^{(l)}[1],\textbf{h}_v^{(l)}[2],...,\textbf{h}_v^{(l)}[n_a])$
	* 优点
		* 可并行计算
		* 所需储存空间与图大小无关
		* 只注意局部邻域
		* 不依赖全局结构
	* 参考[注意力机制](Dive%20into%20Deep%20Learning/Dive%20into%20Deep%20Learning.md#注意力机制)

# GNN Layers in Practice

一些经典的 NN 技巧在 GNN 里也适用。

* Batch Normalization：先使均值为 0、方差为 1，再做仿射变换（参数可学习）。可稳定网络。
* Dropout：随机将神经元置 0。防止过拟合。主要应用在消息函数的线性层。
* Activation：ReLU、Sigmoid、Parametric ReLU

# Stacking Layers of a GNN

过平滑问题 （over-smoothing problem）
* GNN 常有过平滑问题，即所有节点 embedding 相近/相同。
* 接受域：影响某节点 $v$ 的所有节点集合。在 $K$ 层的 GNN 中，节点 $v$ 的接收域包含所有 $1,2,...,K$ 跳相邻节点。接受域的大小成指数增长。
* $K$ 较大时，不同节点之间接受域高度重合，因此 embedding 也高度相似。
* Lesson 1：谨慎添加 GNN 层
	* 但若 GNN 层数较少，表达能力会降低
	* Solution 1：加强每个 GNN 层的表达能力。比如把 3 层 MLP 作为消息/聚合函数
	* Solution 2：添加不传递消息的层。比如在 GNN 层的前后添加 MLP。
		* 预处理：需要编码节点 feature 时比较重要
		* 后处理：需要节点上的推理/转换时比较重要
* Lesson 2： 为 GNN 添加跳过连接（skip connection）
	* 思路类似 [残差网络（ResNet）](Dive%20into%20Deep%20Learning/Dive%20into%20Deep%20Learning.md#残差网络（ResNet）)，得到 $\mathbf{F}(\mathbf{x})+\mathbf{x}$
	* GNN 的前几层可能能更好地区分节点，因此需要更多地保留
	* 每一层分跳过和不跳过，会得到 $2^N$ 种可能的路径
	* $\textbf{h}_{v}^{(l)}=\sigma\left(\sum_{u\in N(v)}W^{(l)}\frac{\textbf{h}_{u}^{(l-1)}}{|N(v)|}+\textbf{h}_{v}^{(l-1)}\right)$
	* 甚至可以设置跳过多个层
* ![](assets/Pasted%20image%2020230301211559.png)

# Graph Manipulation in GNNs

目前为止，我们都将原始输入的图作为计算图，但实际上这常常并不是最优解。原始的图可能 1. 缺乏 features；2. 太稀疏；3. 太稠密；4. 太大。我们介绍下列解决方法。

* Solution 1：特征增强（feature augmentation）
	* Problem 1：输入图可能没有节点特征，即只有邻接矩阵
		* Approach 1：为节点赋予常量
			* 中表达能力、强归纳能力（即预测没见过的节点的能力）、低计算开支、适用任何图
		* Approach 2：为节点赋予唯一 ID，并转为独热编码
			* 高表达能力、无归纳能力、高计算开支、只适用于小图
	* Problem 2：有些特定结构难以被 GNN 学习
		* 如 GNN 无法学到节点 $v$ 所在环的长度，因为计算图是同一个棵二叉树。此时可以使用独热编码，其中对应所在环长度的位置为 1，其他为 0。
		* [Node-Level Tasks and Features](02%20-%20Feature%20Engineering%20for%20Machine%20Learning%20in%20Graphs.md#Node-Level%20Tasks%20and%20Features) 中所述的 feature 都可以被用作特征增强
* Solution 2：结构增强：添加虚拟边
	* 用虚拟边连接 2 跳相邻节点，即用 $\mathbf{A}+\mathbf{A}^2$ 来计算 GNN。
	* 适用于**二分图**，如作者-论文图中 2 跳虚拟边可以将合作的作者相连。
* Solution 3：结构增强：添加虚拟节点
	* 添加一个和所有节点相连的虚拟节点。
	* 适用于**稀疏图**，原本离得很远的节点之间距离直接缩短为 2，可以大幅提高信息传递。
* Solution 4：节点邻域采样
	* 不从所有邻域内的节点获取消息，而是从中随机采样。
	* 适用于**稠密图**，大幅降低计算开销，而且最终得到的 embedding 不会和计算邻域内的全部节点的结果相差很多。