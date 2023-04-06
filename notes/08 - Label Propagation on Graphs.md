# How Do We Leverage Node Correlations in Networks?

主要问题：一个图上部分节点有 label，如何以此预测其它节点的 label？

节点 embedding（通过随机游走、GNN ）可以做到，但效果不好。

一个替代方法是标签传播（label propagation）。它基于**节点通过边互相关联**的假设，也就是在图上靠得近得节点 label 也接近。“物以类聚，人以群分”。

# Label Propagation

**标签传播的核心思想：节点 $v$ 的标签 $Y_v$ 为 $c$ 的概率 $P(Y_v=c)$ 是其相邻节点为该类概率的加权平均。**

这里假设任务是做二分类（标签为 0 和 1），以红色表示标签为 0 的节点，以绿色表示标签为 1 的节点，以灰色表示标签未知的节点。

标签传播（LP）算法
1. 初始化
	* 简记 $P(Y_v=1)$ 为 $P_{Y_v}$
	* 有标签的节点 $v$ ：若标签为 1 则 $P^{(0)}_{Y_v}=1$，为 0 则 $P^{(0)}_{Y_v}=0$
	* 无标签的节点 $v$：设置 $P^{(0)}_{Y_v}=0.5$ 
2. 迭代
	* $P^{(t+1)}_{Y_v}=\dfrac{1}{\sum_{(v,u)\in E}A_{v,u}}\sum_{(v,u)\in E}A_{v,u}P^{(t)}_{Y_u}$，直到收敛
	* 收敛判别准则：$\big|P^{(t)}_{Y_v}-P^{(t-1)}_{Y_v}\big|\leq\epsilon\quad\forall v\in V$
3. 输出
	* 最终的 $P_{Y_v}>0.5$ 则标签为 1，$P_{Y_v}<0.5$ 则为 0

缺点
* 收敛慢，甚至可能不收敛
* 未使用节点属性

# Node Classification: Correct & Smooth

* LP：利用节点标签通过边相关连的假设，但不使用节点feature
* GNN：使用节点 feature，但不利用节点标签相关的假设

在节点 feature 非常 predictive 时，GNN 效果很好。但当节点 feature 并不 predictive 时，GNN 会出错。例如有时两个对称的节点会有完全一样的计算树（图中 $v_2$ 和 $v_5$），但它们附近节点的标签不同，这个时候 LP 能很好区分两者但 GNN 完全不行。

![](assets/Pasted%20image%2020230303211137.png)

Correct & Smooth 算法（C&S） ^825a57
1. 训练基预测器（如简单的 MLP 或 GNN）；
2. 使用基预测器预测所有节点的 soft 标签（即节点 $v$ 属于各类的概率向量 $\hat{y}$）；
3. 通过图结构进行后处理（包括 1. 修正和 2. 平滑化）得到最终预测结果。

修正（correct）
* 思想：不同节点有不同程度的预测误差，**相连的节点预测误差大小也相近。**
* 初始化
	* 对于有标签的节点 $v$，计算其初始误差向量 $e_v^{(0)}=y_v-\hat{y}_v$，其中 $y_v$ 为真实标签的独热编码，$\hat{y}_v$ 为预测的概率向量
	* 对于没有标签的节点 $v$，$e_v^{(0)}=\mathbf{0}$
	* 将 $e_v^{(0)}$ 从上往下拼接，得到初始误差矩阵 $\mathbf{E}^{(0)}$
* 迭代
	* 对误差向量进行扩散
	* $\textbf{E}^{(t+1)}\leftarrow(1-\alpha)\cdot\textbf{E}^{(t)}+\alpha\cdot\widetilde{\textbf{A}}\textbf{E}^{(t)}$，其中 $\alpha$ 为超参数
	* $\widetilde{\textbf{A}}$ 为扩散矩阵，$\widetilde{A}=\textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2}$，而 $\boldsymbol{D}=\text{Diag}(d_1,\ldots,d_N)$ 为度数矩阵 ^d06835
	* 若 $i$ 与 $j$ 相连，则 $\widetilde{\textbf{A}}_{ij}=\frac{1}{\sqrt{d_i}\sqrt{d_j}}$，否则为 0
	* 因此如果 $i$ 与 $j$ 都之与彼此相连则 $\widetilde{\textbf{A}}$ 大，两者都还和很多别的节点相连则 $\widetilde{\textbf{A}}$ 小
* 输出
	* $\hat{y}_v \leftarrow s \cdot e_v^{(K)} + \hat{y}_v$，其中 $s$ 为超参数

平滑化（smooth）
* 思想：相邻节点的标签倾向于相同
* 初始化
	* 对于有标签的节点 $v$，其初始标签向量 $z_v^{(0)}=y_v$
	* 对于没有标签的节点 $v$，$z_v^{(0)}=\hat{y}_v$
	* 将 $z_v^{(0)}$ 从上往下拼接，得到初始标签矩阵 $\mathbf{Z}^{(0)}$
* 迭代
	* $\textbf{Z}^{(t+1)}\leftarrow(1-\alpha)\cdot\textbf{Z}^{(t)}+\alpha\cdot\widetilde{\textbf{A}}\textbf{Z}^{(t)}$，其中 $\alpha$ 为超参数
* 输出
	* 平滑化所得结果 $z^{(K)}_v$ 各项之和不为 1，这时取最大值对应的位置作为最终预测
* 平滑化为 LB 的变体

# Masked Label Prediction

另一个显式地包括节点标签信息的办法。从 BERT 中获取灵感。

掩盖标签预测
* 思想：将标签视作 feature
* 算法
	* 将标签矩阵 $Y$ 和 feature 矩阵 $X$ 拼接
	* 记知道标签的节点的标签矩阵为 $\hat{Y}$
	* 随机将 $\hat{Y}$ 的一部分标签向量置为 $\mathbf{0}$（记掩盖标签），将转化后的结果记作 $\tilde{Y}$
	* 用 $[X,\tilde{Y}]$ 去预测被掩盖标签的节点的标签
	* 用 $\hat{Y}$ 去预测不知道标签的节点
* 类似连接预测，参考[链接预测](06%20-%20GNN%20Augumentation%20and%20Training.md#^d58d0f)。
