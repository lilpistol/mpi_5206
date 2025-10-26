这是原始的project instruction:Given a dataset

$$
\mathcal{D}=\left\{\left(x^{(i)}, y^{(i)}\right)\right\}_{i=1}^N, \quad x^{(i)}=\left(x_1^{(i)}, \ldots, x_m^{(i)}\right) \in \mathbb{R}^m, \quad y^{(i)} \in \mathbb{R},
$$

a neural network with one hidden layer approximates the map from $x_i$ to $y_i$ using the following equation:

$$
f(x ; \theta)=\sum_{j=1}^n w_j \sigma\left(\sum_{k=1}^m w_{j k} x_k+w_{j, m+1}\right)+w_{n+1},
$$

where

$$
\begin{gathered}
\theta=\left(w_1, \ldots, w_n, w_{n+1}\right. \\
w_{11}, \ldots, w_{1, m+1} \\
\ldots, \\
\left.w_{n 1}, \ldots, w_{n, m+1}\right)
\end{gathered}
$$

is the set of all parameters in the model and $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ is a nonlinear activation function. The stochastic gradient method aims to solve the following minimization problem:

$$
\min _\theta R(\theta)=\frac{1}{2 N} \sum_{i=1}^N\left|f\left(x^{(i)} ; \theta\right)-y^{(i)}\right|^2 .
$$


To find the optimal value of $\theta$, we start with an initial guess $\theta_0$, and update the solution with

$$
\theta_{k+1}=\theta_k-\eta \widetilde{\nabla_\theta R}\left(\theta_k\right),
$$

where $\eta$ is the learning rate, and $\widetilde{\nabla_\theta R}$ is the approximation of the gradient

$$
\nabla_\theta R(\theta)=\frac{1}{N} \sum_{i=1}^N\left[f\left(x^{(i)} ; \theta\right)-y^{(i)}\right] \nabla_\theta f\left(x^{(i)} ; \theta\right) .
$$


The approximation is done by randomly drawing $M$ distinct integers $\left\{j_1, \ldots, j_M\right\}$ from the set $\{1, \cdots, N\}$ and setting $\widetilde{\nabla_\theta R}$ to

$$
\widetilde{\nabla_\theta R}(\theta)=\frac{1}{M} \sum_{i=1}^M\left[f\left(x^{\left(j_i\right)} ; \theta\right)-y^{\left(j_i\right)}\right] \nabla_\theta f\left(x^{\left(j_i\right)} ; \theta\right) .
$$


The random set $\left\{j_1, \ldots, j_M\right\}$ must be updated for every iteration (1). The iteration terminates when $R\left(\theta_k\right)$ no longer decreases.

This projects requires implementation of the above algorithm using MPI. Any programming language can be used. The implementation must satisfy the following requirements:
- The code should work for any number of processes.
- The dataset is stored nearly evenly among processes, and the algorithm should not send the local dataset to other processes, except when reading the data.
- All processes should compute the stochastic gradient $\widetilde{\nabla_\theta R}$ in parallel.
- Once the solution $\theta$ is found, the code can compute the RMSE in parallel.

Apply your code to the dataset nytaxi2022.csv (the zip file can be found on Canvas). The description of this dataset can be found at https://www.kaggle.com/datasets/diishasiing/  revenue-for-cab-drivers. The requirements include:
- Split the dataset into a training set ( $70 \%$ ) and a test set ( $30 \%$ ).
- Build a machine learning model to predict the total fare amount (column total_amount) using the following columns as features:
- tpep_pickup_datetime
- tpep_dropoff_datetime
- passenger_count
- trip_distance
- RatecodeID
- PULocationID
- DOLocationID
- payment_type
- extra

You may need to preprocess of the data by dropping some incomplete rows and normalizing the data.
- Test your code with at least three activation functions $\sigma$ and at least five batch sizes $M$. Choose a suitable number of neurons in the hidden layer $n$ for each set of parameters. Report the following results:
- The parameters you have chosen.
- Figures showing the training history (the value of $R\left(\theta_k\right)$ versus $k$ ).
- The RMSE of the training data and the test data.
- Training times for different numbers of processes.
- Any efforts you have made to improve the result.


### 1. 环境准备 (必须做)

你需要在你的环境里装好这几个Python库：
* `numpy`: 用于所有数值计算（矩阵乘法、向量等）。
* `pandas`: 用于读取和处理 `.csv` 数据。
* `scikit-learn`: 只用来做数据拆分 (train_test_split) 和数据标准化 (StandardScaler 或 MinMaxScaler)。
* `matplotlib`: 用于最后画图（训练历史）。
* `mpi4py`: MPI的核心库，用于并行。

### 2. 数据预处理 (必须做)

这是模型能跑起来的前提。
1.  **加载数据**: 使用 `pandas.read_csv('nytaxi2022.csv')` 把数据读进来。
2.  **处理特征 (Features)**:
    * **时间**: `tpep_pickup_datetime` 和 `tpep_dropoff_datetime`。你需要把它们转换成一个有用的特征，最简单的方法是计算 **行程时长（分钟或秒）**。
        * `df['pickup'] = pd.to_datetime(df['tpep_pickup_datetime'])`
        * `df['dropoff'] = pd.to_datetime(df['tpep_dropoff_datetime'])`
        * `df['duration'] = (df['dropoff'] - df['pickup']).dt.total_seconds() / 60.0`
    * **其他特征**: `passenger_count`, `trip_distance`, `RatecodeID`, `PULocationID`, `DOLocationID`, `payment_type`, `extra`。
    * **选择列**: 把你处理好的特征（包括 `duration` 和其他数字列）选出来，作为 $X$。
3.  **处理标签 (Target)**:
    * 把 `total_amount` 这一列选出来，作为 $y$。
4.  **清理数据 (必须)**:
    * **丢弃空值**: `df.dropna(inplace=True)`。只要一行里有任何一个 `NaN`，就扔掉它。
    * **丢弃异常值 (建议)**: 比如 `total_amount` 小于0的，或者 `trip_distance` 等于0但 `total_amount` 很大的。60分标准下，可以先不做，但如果模型完全不收敛，回来看看这里。
5.  **拆分数据集 (必须)**:
    * 使用 `sklearn.model_selection.train_test_split` 把你的 $X$ 和 $y$ 拆成70%的训练集和30%的测试集。设置 `random_state` 保证结果可复现。
6.  **数据标准化 (必须)**:
    * 神经网络对数据尺度很敏感。
    * `from sklearn.preprocessing import StandardScaler`
    * `scaler = StandardScaler()`
    * `X_train_scaled = scaler.fit_transform(X_train)` (用训练集计算均值和方差)
    * `X_test_scaled = scaler.transform(X_test)` (用*训练集*的均值和方差来标准化测试集)
    * **注意**: 标签 $y$ （total_amount）通常*不*需要标准化，但如果它的数值特别大，标准化 $y$ 可能有助于训练，记得最后预测时要 `scaler_y.inverse_transform` 转换回来。为了60分，**你可以先不标准化 $y$**。

---

### 3. 神经网络实现 (核心难点)

你需要用 `numpy` 手动实现公式里的所有计算。
1.  **激活函数 (必须)**:
    * 你需要实现 **$\ge 3$** 个激活函数，以及它们各自的**导数**（反向传播用）。
    * **60分选择**:
        * **Sigmoid**: $\sigma(z) = 1 / (1 + e^{-z})$， $\sigma'(z) = \sigma(z) (1 - \sigma(z))$
        * **ReLU**: $\sigma(z) = \max(0, z)$， $\sigma'(z) = 1$ (如果 $z > 0$) 否则 $0$
        * **Tanh**: $\sigma(z) = \tanh(z)$， $\sigma'(z) = 1 - \tanh(z)^2$
2.  **参数初始化**:
    * 你需要一个函数来初始化所有的 $\theta$ (即所有 $w$)。
    * 把所有 $w$ 存在一个**一维长向量** $\theta$ 中，或者一个字典/类里。**建议用一个长向量**，这样更新 $\theta$ 最简单（$\theta = \theta - \eta \cdot \text{gradient}$）。
    * 用小的随机数初始化（比如 `np.random.randn(...) * 0.01`），**不要全初始化为0**。
3.  **前向传播 $f(x ; \theta)$ (必须)**:
    * 写一个函数 `forward(x, theta)`，输入一个**样本** $x$（$m$ 维向量）和**当前所有参数** $\theta$。
    * 它需要计算：
        1.  $z_j = \sum_{k=1}^m w_{j k} x_k+w_{j, m+1}$ (对每个隐藏神经元 $j$)
        2.  $a_j = \sigma(z_j)$ (激活)
        3.  $y_{pred} = \sum_{j=1}^n w_j a_j + w_{n+1}$ (输出)
    * 返回 $y_{pred}$。为了反向传播，最好也**同时返回中间值** $z_j$ 和 $a_j$。
4.  **计算梯度 $\nabla_\theta f(x^{(i)} ; \theta)$ (必须且最难)**:
    * 这是反向传播。你需要计算**单个样本** $(x^{(i)}, y^{(i)})$ 对**所有参数**的梯度。
    * 设 $e = f(x^{(i)}) - y^{(i)}$ (即 $y_{pred} - y_{true}$)
    * 根据链式法则，你需要计算：
        * $\frac{\partial R_i}{\partial w_{n+1}} = e \cdot 1$
        * $\frac{\partial R_i}{\partial w_j} = e \cdot a_j$
        * $\frac{\partial R_i}{\partial w_{j, m+1}} = e \cdot w_j \cdot \sigma'(z_j) \cdot 1$ (隐藏层偏置的梯度)
        * $\frac{\partial R_i}{\partial w_{jk}} = e \cdot w_j \cdot \sigma'(z_j) \cdot x_k^{(i)}$ (隐藏层权重的梯度)
    * 写一个函数 `backward(x, y, theta, intermediate_values)`，返回一个和 $\theta$ **形状完全相同**的梯度向量 $\nabla_\theta R_i$。

---

### 4. MPI 并行化 (项目核心要求)

这里你要用 `mpi4py` 来组织你的训练。
1.  **初始化**:
    * `from mpi4py import MPI`
    * `comm = MPI.COMM_WORLD`
    * `rank = comm.Get_rank()` (当前是第几个进程, 0, 1, 2...)
    * `size = comm.Get_size()` (总共有几个进程)
2.  **数据分发 (要求2)**:
    * **只在 `rank == 0` 的进程上**:
        1.  执行"第2步：数据预处理"的**所有**操作（加载、清理、拆分、标准化）。
        2.  现在 `rank 0` 有了 `X_train_scaled` 和 `y_train`。
    * **把训练数据分给所有进程**:
        * `rank 0` 需要把 `X_train_scaled` 和 `y_train` 切成 `size` 份。
        * 使用 `comm.scatter` (如果每份一样大) 或 `comm.Scatterv` (如果不一样大) 把数据块分发下去。
        * **60分简化**: 用 `numpy.array_split` 切割，然后用 `comm.scatter`。
        * `local_X_train = comm.scatter(split_X_data, root=0)`
        * `local_y_train = comm.scatter(split_y_data, root=0)`
    * **测试集**: 你可以只把测试集 `Bcast` (广播) 给所有进程，或者也 `scatter` 下去。广播更简单。
3.  **并行训练 (要求3)**:
    * **同步参数**: 确保所有进程的 $\theta$ 在**迭代开始前**是完全一样的。
        * `rank 0`: 初始化 $\theta$。
        * `comm.Bcast(theta, root=0)` (把 `rank 0` 的 $\theta$ 广播给所有人)
    * **训练循环 (迭代 $k=1, 2, \dots$)**:
        * **所有进程并行执行**:
            1.  **抽取 $M$ 个索引**:
                * `rank 0` 进程：从**全局** $N$ 个索引 (从 $0$ 到 $N-1$) 中，随机抽取 $M$ 个索引 $\left\{j_1, \ldots, j_M\right\}$。
                * `rank 0` 广播这 $M$ 个索引: `M_indices = comm.Bcast(M_indices, root=0)`
            2.  **计算本地梯度**:
                * 每个进程 $p$ 拿到这 $M$ 个全局索引。
                * 进程 $p$ 检查这 $M$ 个索引里，有哪些是**属于它自己**的 `local_X_train` 里的数据？(这步有点绕，你**必须**知道全局索引 $j_i$ 对应的是哪个进程的哪条本地数据)。
                * **60分简化策略 (更常见)**:
                    * 让 **`rank 0`** 随机抽取 $M$ 个索引。
                    * `rank 0` **`Bcast`** 这 $M$ 个索引给所有人。
                    * **所有进程** 检查这 $M$ 个索引，看哪些索引**落在自己负责的数据范围**内。
                    * 例如：总共 $N=1000$，4个进程。P0 负责 0-249, P1 负责 250-499...
                    * `rank 0` 抽了 $M=32$ 个索引，比如 $\{10, 260, 550, 15\}$。
                    * P0 计算 10 和 15 的梯度。P1 计算 260 的梯度。P2 计算 550 的梯度。P3 啥也不干。
                    * 每个进程 $p$ 计算它负责的索引的梯度，并**求和**，得到 $G_p$。
            3.  **汇总全局梯度 (要求4)**:
                * 使用 `comm.Allreduce(G_p, G_total, op=MPI.SUM)`。
                * `G_total` 现在是所有进程的梯度之和 $\sum G_p$。
            4.  **计算最终梯度**:
                * $\widetilde{\nabla_\theta R} = G_{total} / M$ (注意是除以 $M$，不是 $N$)。
            5.  **更新参数 (同步)**:
                * $\theta = \theta - \eta \cdot \widetilde{\nabla_\theta R}$
                * **重要**: 因为所有进程都执行了 `Allreduce`，它们都拿到了**一样**的 $G_{total}$，所以它们会**独立但相同**地更新 $\theta$。下一轮迭代开始时，它们的 $\theta$ 依然是同步的。
    * **计算全局Loss $R(\theta)$ (用于画图和终止)**:
        * **并行计算**: 每个进程 $p$ 计算它**本地所有数据**的损失平方和 $L_p = \sum_{i \in \text{local}} |f(x^{(i)}) - y^{(i)}|^2$。
        * **汇总**: `comm.Allreduce(L_p, L_total, op=MPI.SUM)`。
        * **计算**: $R(\theta) = L_{total} / (2N)$ (注意 $N$ 是**全局**总样本数)。
        * `rank 0` 负责记录 $R(\theta)$ 的历史，并检查是否停止下降 (终止条件)。
4.  **并行计算RMSE (要求5)**:
    * 训练结束后，用**最终的 $\theta$**。
    * 计算 RMSE 和计算 Loss $R(\theta)$ 几乎一样。
    * `L_total` (全局损失平方和) 已经通过 `Allreduce` 得到了。
    * $MSE = L_{total} / N$ (注意 $N$ 是训练集或测试集的总样本数)
    * $RMSE = \sqrt{MSE}$
    * `rank 0` 负责打印训练集和测试集的最终RMSE。

---

### 5. 实验与报告 (必须做)

你需要提交你的结果。
1.  **参数**:
    * **$\ge 3$ 个激活函数**: $\sigma$ = (Sigmoid, ReLU, Tanh)
    * **$\ge 5$ 个Batch Size**: $M$ = (例如: 32, 64, 128, 256, 512)
    * **$n$ (隐藏层神经元)**: 题目说 "为每组参数选择合适的n"。
        * **60分策略**: 选一个**固定**的 $n$ (比如 $n=32$ 或 $n=64$)，并在报告里说 "我选择 $n=32$ 作为隐藏层大小"。不要浪费时间去调优。
    * **$\eta$ (学习率)**: 选一个能用的，比如 `0.001` 或 `0.0001`。
2.  **报告内容**:
    * **你选择的参数**: $n=32$, $\eta=0.001$, 迭代次数=5000 ...
    * **训练历史图 (必须)**:
        * $R(\theta_k)$ (全局Loss) vs $k$ (迭代次数) 的曲线图。
        * 你需要 $3 \times 5 = 15$ 次实验，但**为了60分**，你可以只展示**最有代表性**的几张图（比如，固定 $M=128$，展示3个激活函数的对比图；固定 $\sigma=\text{ReLU}$，展示5个 $M$ 的对比图）。
    * **RMSE (必须)**:
        * 做一个表格，列出不同 (激活函数, $M$) 组合下的**训练集RMSE**和**测试集RMSE**。
    * **训练时间 (必须)**:
        * 选择你**最好**的一组参数 (e.g., ReLU, M=128)。
        * 分别用 $P=1, 2, 4, 8$ 个进程 (或你机器支持的数量) 跑你的代码。
        * 记录**训练部分**花费的时间。
        * 画一个图或表格，展示 "进程数" vs "训练时间"。
    * **改进努力 (60分)**:
        * 随便写一句，比如 "我尝试了不同的学习率 $\eta$ 来防止梯度爆炸" 或者 "我通过标准化数据来帮助模型收敛"。

### 总结：60分 Checklist

☐ 1. (数据) 成功加载数据，处理了时间特征，`dropna()` 清理了数据。
☐ 2. (数据) 成功拆分了 70/30 训练集/测试集。
☐ 3. (数据) 成功标准化了 $X$ (用 `StandardScaler`)。
☐ 4. (NN) 实现了 Sigmoid, ReLU, Tanh 及其**导数**。
☐ 5. (NN) 实现了 `forward` 函数。
☐ 6. (NN) 实现了 `backward` 函数 (计算 $\nabla_\theta R_i$)。
☐ 7. (MPI) `rank 0` 读数据并 `scatter` 给所有进程。
☐ 8. (MPI) `rank 0` `Bcast` $\theta$ 和 $M$ 个索引。
☐ 9. (MPI) 所有进程并行计算本地梯度 $G_p$。
☐ 10. (MPI) 成功用 `Allreduce` 汇总 $G_total$ 并**同步更新** $\theta$。
☐ 11. (MPI) 成功用 `Allreduce` 汇总全局Loss $R(\theta)$ 和 最终RMSE。
☐ 12. (报告) 跑了 $3 \times 5$ 组实验（或者一个简化的子集）。
☐ 13. (报告) 画出了Loss vs $k$ 的图。
☐ 14. (报告) 提交了RMSE表格。
☐ 15. (报告) 提交了并行加速的时间对比。
