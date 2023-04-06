# Geometric Graphs

若图 $G=(A,S,X)$ 的节点都被嵌入到一个 $d$ 维的**欧氏空间**（通常 $d=3$ ，因为现实世界宏观而言是三维的），则称 $G$ 为几何图。其中邻接矩阵 $A\in \mathbb{R}^{n\times n}$，标量特征 $S\in \mathbb{R}^{n \times f}$，张量特征 $X\in \mathbb{R}^{n\times d}$。

几何图可被用于建模分子，完成性质预测、分子模拟等任务。

在描述几何图时，我们用到了坐标系。对于同一个分子的几何构型，如果使用不同的坐标系，则 $X$ 也不同，如下图中的 (1) 和 (2)。坐标系的变换包括 3D 旋转、平移等，带有**对称性**。对于传统 GNN，不同的 $X$ 得到的结果是完全不同的，哪怕它们可能来自同一个分子的几何构型。

![](assets/Pasted%20image%2020230405230755.png)


有时我们需要的输出也是向量，如预测分子中每个原子上的力。比如对于一个分子及其旋转后的拷贝，对不旋转的分子预测出的力在经过相同的旋转后得到的力与对旋转的分子预测出的力相同。这时我们称模型是对变换**等变**的。

等变（equivariance）的定义：若一个函数 $F: X\rightarrow Y$ 对变换 $\rho$ 满足 $F\circ\rho_X (x)=\rho_Y\circ F(x)$，则它是是等变的。

![](assets/Pasted%20image%2020230405231509.png)

有时我们需要的输出是标量，如预测分子的能量。对于一个分子及其旋转后的拷贝，预测出的结果应当完全相同。这时我们称模型对变换是**不变**的。

不变（invariance）的定义：若一个函数 $F: X\rightarrow Y$ 对变换 $\rho$ 满足 $F\circ\rho_X (x)=F(x)$，则它是是不变的。

不变可被看作是一种特殊的等变，其中 $\rho_Y$ 表示不进行变换。

# Geometric Graph NNs

如果 ML 模型无法处理这种对称性，我们就需要大量的数据增强（输入大量不同变换的结果）。

设计几何 GNN 的另一个好处是可以大量缩小目标函数的范围。

![](assets/Pasted%20image%2020230405233434.png)

一个例子：分子动力学模拟（模拟分子几何结构的稳定结构）。

* 概念
	* 能量：$E(\mathbf{x}_1,\dots,\mathbf{x}_n)$
	* 力：$\mathbf{F}_i (\mathbf{x}_1,\dots, \mathbf{x}_n)=-\frac{\partial E}{\partial r_i}(\mathbf{x}_1,\dots,\mathbf{x}_n)$
	* 位置更新：$X^{(t)}+\mathbf{F}\rightarrow X^{(t+1)}$
* 输入
	* 原子类型 $X=(\mathbf{x}_1,\dots,\mathbf{x}_n)$ ，其中 $\mathbf{x}_i\in\mathbb{R}^d$
	* 原子位置 $R=(\mathbf{x}_1,\dots,\mathbf{x}_n)$ ，其中 $\mathbf{x}_i\in\mathbb{R}^3$
* 预测目标
	* 能量：$E(\mathbf{x}_1,\dots,\mathbf{x}_n)$（不变）
	* 对每个原子的力：$\mathbf{F}_i (\mathbf{x}_1,\dots, \mathbf{x}_n)$（等变）

## Invariant GNNs

不变的 GNN：SchNet。

$$
\mathbf{x}_i^{l+1}=(X^l*W^l)_i=\sum_j\mathbf{x}_j^l\circ W^l(\mathbf{r}_i-\mathbf{r}_j),
$$

其中 $\circ$ 表示 Hadamard 积，$W^l:\mathbb{R}^D\to\mathbb{R}^F$，将原子之间的相对位置映射为与原子特征维度相同的向量（$W^l$ **在这里是指一个映射/函数！不是矩阵！**）。

我们将具体解释 $W^l$ 的计算。

注意到，原子间的距离是旋转/平移不变的，因此我们考虑 $d_{ij}=\|\mathbf{r}_{ij}\|=\|\mathbf{r_i}-\mathbf{r}_j\|$。由于 $d_{ij}$ 是一维的，为了方便训练，我们通过 Radial Basis Functions（RBF）将它升成 300 维 
$$
e_k(\mathbf{r}_i-\mathbf{r}_j)=\exp(-\gamma\|d_{ij}\stackrel{.}{-}\mu_k\stackrel{.}{\|^2})
$$
随后经过一个两层的 MLP，激活函数采用 shifted softplus。最后再与 $W^l$ 做 Hadamard 积。

这个过程称为 cfconv 模块，它聚合原子的成对消息传递。

我们在 cfconv 前后再添加若干个 atom-wise 模块（前向 MLP，$\mathbf{x}^{l+1}_i=W^l\mathbf{x}_i^l+\mathbf{b}^l$），拼成一个 interaction 模块。这个模块带有残差项，即 $\mathbf{x}_i^{l+1} = \mathbf{x}_i^l + \mathbf{v}_i^l$。

最后堆叠 atom-wise 模块和 interaction 模块，预测每个原子的单个标量值，将所有标量相加作为能量的预测值 $\hat{E}$。

具体流程如下图
![](assets/Pasted%20image%2020230406005920.png)

力可以通过计算能量输出与输入坐标的梯度来计算。

损失函数为
$$
\ell(\hat E,(E,\mathbf F_1,\ldots,\mathbf F_n))=\rho\|E-\hat E\|^2+\dfrac{1}{n}\sum_{i=0}^{n}\left\|\mathbf F_i-\left(-\dfrac{\partial\hat E}{\partial\mathbf R_i}\right)\right\|^2
$$

事实上，化学键（图中的边）之间的角度对能量也是有影响的，但 SchNet 没有考虑这一点。

![](assets/Pasted%20image%2020230406011440.png)

DimeNet 基于原子间距离与化学键夹角进行消息交互（两者都是对旋转和平移不变的）。

不变的 GNN 的缺点：必须保证输入特征已经包含了任何必要的等变相互作用。

## Equivariant GNNs

等变的 GNN：PaiNN。PaiNN 依旧使用 $\|\mathbf{r}_{ij}\|$ 来控制信息传递。

但同时，PaiNN 中，每个节点都有两个特征（标量特征 $s_i$ 与向量特征 $v_i$）。

* 初始化
	* $s_i$ 为原子的 embedding，$v_i$ 为 0 向量
* 通过残差神经网络更新
		* $s_i=s_i+\Delta s_i$，$v_i=v_i+\Delta v_i$
		* $\Delta\mathbf{s}_i^m=\left(\boldsymbol{\phi}_s(\mathbf{s})*\mathcal{W}_s\right)_i=\sum\boldsymbol{\phi}_{s}(\mathbf{s}_j)\circ\mathcal{W}_{s}(\lVert\vec{r}_{i j}\rVert)$
			* 类似于 SchNet
			* 不变的消息函数 $\boldsymbol{\phi}_s$、权重 $\mathcal{W}_s$、聚合函数（即直接求和）
		* $\Delta\vec{\mathbf{v}}_{i}^{m}=\sum_{j}\vec{\mathbf{v}}_j\circ\phi_{vv}(\mathbf{s}_j)\circ\mathcal{W}_{vv}(\|\vec{r}_{ij}\|)\\ +\sum_{j}\phi_{vs}(\mathbf{s}_{j})\circ\mathcal{W}{}_{vs}'(\|\vec{\vec{r}}_{ij}\|)\frac{\vec{r}_{ij}}{\|\vec{r}_{ij}\|}$
			* 不同于 SchNet
			* 相对方向 $\vec{r}_{ij}$ 的加权和，保持等变性

可以堆叠多个 PaiNN。最终的 $v_i$ 对输入坐标系是等变的，可以直接用来预测力。

![](assets/Pasted%20image%2020230406013316.png)

# Geometric Generative Models

分子构象（conformation）生成：从 2D 分子图 $\mathbf{G}$ 生成稳定构象 $\mathbf{C}$。

![](assets/Pasted%20image%2020230406001443.png)
需要有对旋转/平移的等变性。

背景知识：扩散模型（Diffusion Models）

扩散模型定义了一个扩散过程，将数据破坏为不同噪声级别的样本，随后学习反向模型，通过去噪生成图片。

![](assets/Pasted%20image%2020230406001634.png)

* 训练
	* 随机取样噪声 $\epsilon$
	* 在每个时间步 $t$ 破坏数据：$x_t=\mu_t x+\sigma_t \epsilon$
		* $\mu_t$，$\sigma_t$ 都是预定义的
		* 随着 $t$ 变大，$\mu_t$ 变小，$\sigma_t$ 变大
	* 学习 $f_\theta(x_t, t)$ 去预测噪声 $\epsilon$
* 取样
	* 随机取样 $x_T\sim N(0,1)$
	* 不断通过预测并减去噪声得到 $x$

## Geometric Diffusion Models

将扩散思想引入分子生成：DeoDiff。

在扩散中不断扰动分子几何构型直到构象被破坏。随后学习反向生成过程——这有点像分子模拟！

分子模拟：学习以预测**力**
扩散模型：学习以预测**噪声**

去噪模型需要对不用太分子坐标系等变：用等变 GNN 去参数化去噪网络。

