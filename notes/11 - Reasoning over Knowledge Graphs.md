# Introduction

对于一个 KG，用户会提出一些查询（query），如单跳查询、路径查询和合取查询。

![](assets/Pasted%20image%2020230306181816.png)

一个简单的方法是直接对原 KG 进行查询，但由于很多时候 KG 都是不完整（incomplete）的，即很多实体和边都未被包含，因此可能无法返回所有结果。

但如果对原 KG 进行补全后再查询，这时能得到所有结果，但会非常慢，因为补全后的 KG 是稠密的。如果出度最大为 $d_\text{max}$，查询路径为 $L$，则时间复杂度为 $O(d_\text{max}^L)$。

我们希望有一个方法能隐性地考虑 KG 的不完整性。

![](assets/Pasted%20image%2020230306182706.png)

# Answering Predictive Queries on Knowledge Graphs

对于单跳/路径查询，如“什么蛋白质与 Fulvestrant 引起的不良反应有关？”，我们参考参考 TransE ，embed 查询 $q=(v_a, (r_1,..., r_n))$ 为向量
$$
\mathbf{q}=\mathbf{v}_a+\mathbf{r}_1+...+\mathbf{r}_n
$$
这里使用 TransE 而不是其它的模型是因为它可以自然地处理递推关系。

这里的评分函数是
$$
f_q(t)=-\|\mathbf{q}-\mathbf{t}\|
$$
我们希望 $\mathbf{q}$ 能与真实答案的 embedding $\mathbf{t}$ 尽可能接近。

# Query2Box: Reasoning over KGs Using Box Embeddings

对于更为复杂合取查询，如“什么药品导致气短**且**能够治疗与蛋白质 ESR2 有关的疾病？”，我们需要对多个结果集合取交集。

我们 embed 查询 $q$ 为超矩形（称为boxes）
$$
\mathbf{q}=(\text{Center}(q), \text{Offset}(q))
$$
将所有查询 $q$ 的答案 embed 到这个超矩形内部。

boxes 之间的交集是良定义的！也就是说多个 boxes 取交集之后的结果还是一个 box。

实体 $v$ 被视作无体积的 boxes，关系 $r$ 把一个 box 映射到另一个 box，交集运算符 $f$ 把多个 boxes 映射到一个 box。

投影运算符 $\mathcal{P}:\text{Box}\times\text{Relation}\rightarrow\text{Box}$
$$
\text{Cen}(q')=\text{Cen}(q)+\text{Cen}(r) 
$$
$$
\text{Off}(q')=\text{Off}(q)+\text{Off}(r)
$$
![](assets/Pasted%20image%2020230306190901.png)

几何交集运算符 $\mathcal{I}:\text{Box}\times\cdots\times\text{Box}\to\text{Box}$
$$
\text{Cen}(q_\text{inter})=\sum\limits_i \mathbf{w}_i\odot\text{Cen}(q_i)
$$
$$
\quad\textbf{w}_i=\dfrac{\exp(f_{\text{cen}}(\text{Cen}(q_i)))}{\sum_j\exp(f_{\text{cen}}(\text{Cen}(q_j)))}
$$
其中 $\text{Cen}(q_i)\in\mathbb{R}^d$，$\mathbf{w}_i\in\mathbb{R}^d$，$\odot$ 为 Hadamard 积（即按元素求积）。

这背后的直觉是多个 boxes 的交集的中心应该是在这些 boxes 的中心所围出的区域（红色矩形）内部，因此我们使用这些中心的加权平均（自注意力模型）。
![](assets/Pasted%20image%2020230306190914.png)
另外
$$
\text{Off}(q_\text{inter})=\min(\text{Off}(q_1),...,\text{Off}(q_n))\odot\sigma(f_\text{off}(\text{Off}(q_1),...,\text{Off}(q_n)))
$$
这背后的直觉是 boxes 的交集的边长（offset）应当短于所有 boxes 的边长。$\min$ 和 $\sigma$ 保证了 offset 的收缩。$f_\text{off}$ 提取 boxes 里的表征，使模型更有表达性。

最后我们考虑评分函数 $f_q(v)$。我们首先定义节点 $v$ 到 box 的距离
$$
d_\text{box}(\mathbf{q},\mathbf{v})=d_\text{out}(\mathbf{q},\mathbf{v})+\alpha\cdot d_\text{in}(\mathbf{q},\mathbf{v})
$$
其中 $\alpha\in(0,1)$，其背后的直觉是如果一个点在 box 里，那它的距离的权重应该适当缩小。评分函数依旧为负距离。
$$
f_q(v)=-d_\text{box}(\mathbf{q},\mathbf{v})
$$

我们接下来考虑并集操作（析取），如“什么药品可以治疗乳腺癌**或者**肺癌？”

同时有析取+合取的查询称为 Existential Positive First-order (EPFO) 查询，或者称 AND-OR 查询。

与合取查询不同，我们无法 embed AND-OR 查询到低维空间里。在下图中我们无法用一个矩形 box 同时包括红色点但不包括蓝色点。

![](assets/Pasted%20image%2020230306192708.png)
结论：对于 $M$ 个结果互不重叠的合取查询 $q_1,...,q_M$，我们需要 $O(M)$ 维的 embedding 去处理它们的 OR 查询。在 $M$ 很大时 embedding 会很困难。

解决办法：把取交集的操作放到最后。我们考虑原始查询 $q$ 的析取范式（DNF）
$$
q=q_1\lor q_2\lor ... \lor q_m
$$
取交集的距离函数为
$$
d_\text{box}(\mathbf{q},\mathbf{v})=\min(d_\text{box}(\mathbf{q}_1,\mathbf{v}),...,d_\text{box}(\mathbf{q}_m,\mathbf{v}))
$$
这背后的直觉是若 $\mathbf{v}$ 接近一个 $\mathbf{q}_i$，则它应该接近交集的结果 $\mathbf{q}$。

评分函数依旧为
$$
f_q(v)=-d_\text{box}(\mathbf{q},\mathbf{v})
$$

# How to Train Query2Box

对于给定的查询 $q$ 的 embedding $\mathbf{q}$，我们要最大化答案 $v \in [\![q]\!]$ 的评分 $f_q(v)$，最小化非答案 $v' \notin [\![q]\!]$ 的评分 $f_q(v')$

算法
1. 在图中取样一个查询 $q$、答案 $v \in [\![q]\!]$ 和非答案 $v' \notin [\![q]\!]$ ；
2. 计算 $q$ 的 embedding $\mathbf{q}$；
3. 计算评分 $f_q(v)$ 和 $f_q(v')$；
4. 计算损失函数 $\ell=-\log\sigma\left(f_q(v)\right)-\log(1-\sigma\left(f_q(v')\right))$，并以此最优化 embeddings 和运算符

如何构造查询 $q$？首先我们收集一些查询模板，如((Anchor1, (Rel1, Rel2)), (Anchor2, (Rel3))。
![](assets/Pasted%20image%2020230306194839.png)
我们将此看成一个树，从而可见从 anchor（叶节点）入手不是一个好主意，我们应该从 answer（根节点）入手逆推构造整棵树。

# Example of Query2Box

![](assets/Pasted%20image%2020230306195057.png)
