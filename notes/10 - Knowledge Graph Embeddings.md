# Knowledge Graph Completion

知识图谱（KG）是一种异构图。其中节点称为实体，边称为关系。

知识图谱的特征是数据量大但是容易不完整。如何补全？即给定 (head, relation) 如何预测 tail？

KG 中的边表示为三元组 $(h,r,t)$，其中 $h$ 为头节点，$r$ 为关系，$t$ 为尾节点。

我们给每个实体和关系直接通过 embedding-lookup 方法指定一个 embedding，这被称作[浅 embedding](03%20-%20Node%20Embeddings.md#^934fbd)。我们的目标是 $(h,r)$ 的 embedding 能尽可能接近 $t$ 的 embedding。**注意，这里我们不使用 GNN。**

我们首先讨论 KG 可能具有的一些关系模式（relation pattern）
* 对称性：$r (h, t)\Rightarrow r(t,h)\quad\forall t,h$，如室友关系
* 反对称性：$r (h, t)\Rightarrow \neg r(t,h)\quad\forall t,h$，如上位词关系
* 逆关系：$r_1 (h, t)\Rightarrow r_2(t,h)\quad\forall t,h$，如师生关系
* 复合（传递）关系：$r_1(x,y)\wedge r_2(y,z)\Rightarrow r_3(x,z)\quad\forall x,y,z$，如母亲的丈夫是父亲
* 一对多关系：$r (h, t_1),r (h, t_2),...,r (h, t_n)$ 同时成立，如一个学校可以有多个学生
注意这些关系里的 $\forall t,h$。

KG 可能含有以上的部分或全部关系模式。合适的 KG embedding 模型应当能够反映这些关系模式。

评分函数 $f_r(h,t)$ 描述 $h,t$ 之间有关系 $r$ 的可能性（不等于概率），值越大越好。不同的 KG embedding 模型定义不同的评分函数。

KG embedding 模型

# Knowledge Graph Completion: TransE

最为符合直觉，直接把三者都 embed 到 $d$ 维空间，即 $\textbf{h},\textbf{r},\textbf{t}\in\mathbb{R}^d$。以空间中的距离作为评分函数 $f_r(h,t)=-\|\textbf{h}+\textbf{r}-\textbf{t}\|$。

无法反映对称性和一对多关系。

# Knowledge Graph Completion: TransR

TransE 基础上的提升，先 embed 到 $k$ 维空间，再通过矩阵 $\mathbf{M}_r$ 映射到 $d$ 维空间，即 $\textbf{h,t,r}\in\mathbb{R}^k$，$\mathbf{M}_r\in\mathbb{R}^{d\times k}$，评分函数是 $f_r(h,t)=-\|\boldsymbol{M}_r\textbf{h}+\textbf{r} -\boldsymbol{M}_r\textbf{t}\|$。

解决了无法反映对称性和一对多关系的问题，可以反映全部的关系模式。

# Knowledge Graph Completion: DistMult

使用一种双线性的模型。评分函数是 $f_r(h,t)=<\textbf{h},\textbf{r},\textbf{t}>=\sum_i\textbf{h}_i\cdot\textbf{r}_i\cdot\textbf{t}_i$，其中 $\textbf{h},\textbf{r},\textbf{t}\in\mathbb{R}^k$。**可以视作 $\mathbf{h}*\mathbf{r}$ （指 MATLAB 中的向量按位相乘）和 $\mathbf{t}$ 的 cos 相似度。**

无法反映反对称、逆关系和和传递关系。

# Knowledge Graph Completion: ComplEx

DistMult 基础上的提升，将实体与关系 embed 到复向量空间，即 $\textbf{h},\textbf{r},\textbf{t}\in\mathbb{C}^k$。评分函数为 $f_r(h,t)=\text{Re}(\sum_i\mathbf{h}_i\cdot\mathbf{r}_i\cdot\bar{\mathbf{t}}_i)$。

无法反映传递关系。

# Conclusion

![](assets/Pasted%20image%2020230305214213.png)

* 结合表格选取合适的模型
* 若 KG 没有大量对称关系，可以很快地试着跑一下 TransE
* 随后再使用复杂模型，如ComplEx

