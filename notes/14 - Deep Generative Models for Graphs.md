# Machine Learning for Graph Generation

两种任务：
* 现实图生成：生成类似于给定图集的图。
* 目标导向的图生成：生成最优化给定目标/符合给定限制的图。
	* 如药品分子生成/优化

这里先讨论**现实图生成**。

第一步：近似

给定来服从概率分布 $p_\text{data}(\mathbf{x})$ 的一些图样本 $\mathbf{x}_i\sim p_\text{data}(\mathbf{x})$，学习概率分布 $p_\text{model}(\mathbf{x};\theta)$ 去近似 $p_\text{data}(\mathbf{x})$，随后再从 $p_\text{model}(\mathbf{x};\theta)$ 中取样。

核心思想：最大似然
$$
\mathbf{\theta}^*=\underset{\mathbf{\theta}}{\operatorname{argmax}}\mathbb E_{x\sim p_\text{data}}\log p_{\text{model}}(\mathbf{x}\mid\mathbf{\theta})
$$
**即找到最可能生成训练样本的模型**。

第二步：取样

先从标准正态分布（简单噪声）中取样
$$
\mathbf{z}_i\sim N(0,1)
$$
随后进行变换
$$
\mathbf{x}_i=f(\mathbf{z}_i;\theta)
$$
其中 $f$ 是深度神经网络。

这里我们使用自回归模型。$p_{\text{model}}(\mathbf{x};\theta)$ 同时被用于密度估计和取样。

链式法则：
$$
p_{\text{model}}(\mathbf{x};\theta)=\prod\limits_{t=1}^n p_{\text{model}}(x_t|x_1,\ldots,x_{t-1};\theta)
$$

# GraphRNN: Generating Realistic Graphs

如何把图转化为序列？把生成图的过程视为不断添加节点和边的过程。

对于图 $G$ 的一个节点排序 $\pi$，序列 $S^\pi=(S^\pi_1, S^\pi_2, ..., S^\pi_{|V|})$ 。

序列 $S^\pi$ 可被视为一个二维数组。第一级是节点级，即 $S^\pi_i$ 表示编号为 $i$ 的节点。第二级是边级，即 $S_i^\pi$ 本身也是一个序列，它长 $i-1$，其中 $S^\pi_{i,j}$ 表示节点 $i$ 与 $j$ 是否相连，1 为相连，0 为不相连。

![](assets/Pasted%20image%2020230313171305.png)

我们使用 RNN 生成序列。

一般的RNN如下图：

![](assets/Pasted%20image%2020230313171943.png)

GraphRNN 包含节点级的 RNN 和边级的 RNN。
* 节点 $v$ 的节点级 RNN 的隐状态作为 $v$ 的边级的 RNN 的初始隐状态
* 节点 $v$ 的边级 RNN 预测这个节点是否与之前的节点相连
* 节点 $v$ 的边级 RNN 的最终的隐状态送入下一个节点的节点级 RNN cell。

特殊token：
* SOS（start of sequence token）：作为初始输入。
* EOS（end of sequence token）：额外输出，若为 0 则继续生成，若为 1 则停止。

![](assets/Pasted%20image%2020230313172605.png)

这很好，但这样的话输出是固定的。如何引入随机性？

**我们从边级 RNN 入手。边级 RNN 是蓝色的。**

我们让输出 $y_t=p_{\text{model}}(x_t\mid x_1,...,x_{t-1};\theta)$，并且让新的输入 $x_{t+1}\sim y_t$。具体而言，就是让 $x_{t+1}$ 服从 $y_t$ 定义的伯努利分布，即有 $y_t$ 的概率 $x_{t+1}=1$，有 $1-y_t$ 的概率 $x_{t+1}=0$。

![](assets/Pasted%20image%2020230313173431.png)

那么给定数据，我们要如何训练呢？强制教学（teacher forcing）。也就是把输入和输出换成训练用的真实数据来做进一步迭代。

![](assets/Pasted%20image%2020230313173545.png)

而真实数据的输出 $y_i^*$ 和模型输出 $y_i\in \{0,1\}$ 可以通过二分类交叉熵损失函数计算损失，从而通过梯度下降优化。
$$
L=-[y_i^*\log(y_i)+(1-y_i^*)\log(1-y_i)]
$$

总结（训练）：
![](assets/Pasted%20image%2020230313174214.png)

![](assets/Pasted%20image%2020230313174238.png)
其中红色的 0 和 1 表示不是由概率分布采样得到的，而是直接使用训练数据。

测试时：
1. 通过从概率分布中采样得到边连通性。
2. 将边级 RNN 的输入改为 GraphGNN 自己的预测

# Scaling Up and Evaluating Graph Generation

GraphRNN的一个问题是当节点数量很大时，预测一个节点与之前的所有节点是否相连开销很大。如何简化这一过程？BFS。

![](assets/Pasted%20image%2020230313185520.png)
# Application of Deep Graph Generative Models to Molecule Generation

我们希望生成的图有如下性质：
* 最优化给定目标，如药品可能性
* 遵守特定规则，如化学有效性规则
* 从给定样本中学习

这个“给定目标”很多时候由物理定理决定，对模型是黑盒。如何把黑盒嵌入到模型中？引入强化学习（reinforcement learning，RL）。

图卷积策略网络（GCPN） = GNN + RL

基于 GNN 节点 embedding 预测链接。

![](assets/Pasted%20image%2020230313190137.png)

分两部分：监督训练与 RL 训练。具体参见强化学习课程，如 CS234。