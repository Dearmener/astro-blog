---
title: '深度学习完全指南（二）：数学基础'
description: '掌握深度学习必备的数学知识：线性代数、微积分、概率论与信息论的核心概念'
pubDate: '2024-02-02'
heroImage: ''
category: '技术'
tags: ['深度学习', '数学', '线性代数']
series: '深度学习完全指南'
seriesOrder: 2
---

## 为什么需要数学？

深度学习本质上是一个数学优化问题：通过调整模型参数，最小化损失函数。理解背后的数学原理能帮助你：

- **直觉理解**：知道模型为什么有效（或无效）
- **调参能力**：理解超参数的数学意义
- **论文阅读**：看懂前沿研究
- **创新能力**：设计新的模型和算法

本章覆盖四大数学基础：**线性代数、微积分、概率论、信息论**。

---

## 第一部分：线性代数

线性代数是深度学习的语言，神经网络的核心运算就是**矩阵乘法**。

### 1.1 标量、向量、矩阵、张量

```python
import numpy as np

# 标量 (Scalar): 单个数值
x = 3.14

# 向量 (Vector): 一维数组
v = np.array([1, 2, 3])  # shape: (3,)

# 矩阵 (Matrix): 二维数组
M = np.array([[1, 2], [3, 4], [5, 6]])  # shape: (3, 2)

# 张量 (Tensor): 多维数组
T = np.random.randn(2, 3, 4)  # shape: (2, 3, 4)
```

在深度学习中：
- **标量**：学习率、损失值
- **向量**：偏置项、词嵌入
- **矩阵**：权重矩阵、注意力分数
- **张量**：批量数据、特征图

### 1.2 矩阵运算

#### 矩阵乘法

这是神经网络的核心操作：

$$
C = AB, \quad C_{ij} = \sum_k A_{ik} B_{kj}
$$

```python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[5, 6], [7, 8]])  # (2, 2)

# 矩阵乘法
C = A @ B  # 或 np.matmul(A, B)
# [[19, 22],
#  [43, 50]]

# 逐元素乘法（Hadamard积）
D = A * B
# [[ 5, 12],
#  [21, 32]]
```

#### 广播机制（Broadcasting）

```python
# 矩阵 + 向量（自动广播）
M = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
v = np.array([10, 20, 30])  # (3,)

result = M + v  # v 自动扩展为 (2, 3)
# [[11, 22, 33],
#  [14, 25, 36]]
```

### 1.3 特殊矩阵

| 矩阵类型 | 定义 | 用途 |
|---------|------|------|
| 单位矩阵 $I$ | 对角线为1，其余为0 | 初始化、恒等变换 |
| 对角矩阵 | 只有对角线有非零值 | 缩放变换 |
| 对称矩阵 | $A = A^T$ | 协方差矩阵 |
| 正交矩阵 | $A^T A = I$ | 旋转变换、权重初始化 |

### 1.4 范数（Norm）

范数衡量向量或矩阵的"大小"：

$$
\begin{aligned}
L_1 \text{ 范数}: &\quad \|x\|_1 = \sum_i |x_i| \\
L_2 \text{ 范数}: &\quad \|x\|_2 = \sqrt{\sum_i x_i^2} \\
L_\infty \text{ 范数}: &\quad \|x\|_\infty = \max_i |x_i|
\end{aligned}
$$

```python
x = np.array([3, -4])

l1_norm = np.linalg.norm(x, ord=1)   # 7
l2_norm = np.linalg.norm(x, ord=2)   # 5
linf_norm = np.linalg.norm(x, ord=np.inf)  # 4
```

**应用**：
- L1正则化（Lasso）→ 稀疏性
- L2正则化（Ridge）→ 权重衰减
- 梯度裁剪 → 防止梯度爆炸

### 1.5 特征分解与SVD

#### 特征值分解

对于方阵 $A$：
$$
A = V \Lambda V^{-1}
$$

其中 $\Lambda$ 是特征值对角矩阵，$V$ 是特征向量矩阵。

#### 奇异值分解（SVD）

对于任意矩阵 $A$：
$$
A = U \Sigma V^T
$$

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, Vt = np.linalg.svd(A)

# 低秩近似（数据压缩）
k = 1
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

**应用**：
- PCA降维
- 推荐系统
- 矩阵压缩

---

## 第二部分：微积分

神经网络的训练就是通过**梯度下降**来优化参数，理解导数是关键。

### 2.1 导数与梯度

**导数**衡量函数的变化率：

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

**梯度**是多变量函数的偏导数向量：

$$
\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]
$$

```python
# 数值梯度（有限差分）
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad
```

### 2.2 链式法则

深度学习中最重要的微积分规则：

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

这是**反向传播算法**的数学基础！

```python
# 例：y = sigmoid(wx + b)
# 前向传播
z = w * x + b
y = 1 / (1 + np.exp(-z))

# 反向传播（链式法则）
dy_dz = y * (1 - y)  # sigmoid的导数
dz_dw = x
dz_db = 1

dy_dw = dy_dz * dz_dw  # 链式法则
dy_db = dy_dz * dz_db
```

### 2.3 常见函数的导数

| 函数 | 导数 | 深度学习应用 |
|------|------|-------------|
| $x^n$ | $nx^{n-1}$ | 多项式 |
| $e^x$ | $e^x$ | Softmax |
| $\ln x$ | $1/x$ | 交叉熵损失 |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid激活 |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | 循环网络 |
| $\text{ReLU}(x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | 最常用激活函数 |

### 2.4 雅可比矩阵与海森矩阵

**雅可比矩阵**（一阶导数）：

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

**海森矩阵**（二阶导数）：

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

**应用**：
- 雅可比矩阵：向量对向量的导数
- 海森矩阵：判断极值点类型、二阶优化方法

---

## 第三部分：概率论

深度学习处理的是**不确定性**：数据有噪声、模型有随机性。

### 3.1 基本概念

**概率质量函数（PMF）** - 离散变量：
$$
P(X = x)
$$

**概率密度函数（PDF）** - 连续变量：
$$
P(a \leq X \leq b) = \int_a^b p(x) dx
$$

**条件概率**：
$$
P(A|B) = \frac{P(A \cap B)}{P(B)}
$$

**贝叶斯定理**：
$$
P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}
$$

### 3.2 常见概率分布

#### 伯努利分布（二元分类）

$$
P(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

```python
# 二分类输出层
output = torch.sigmoid(logits)  # 输出概率 p
loss = F.binary_cross_entropy(output, target)
```

#### 高斯分布（正态分布）

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

```python
# 权重初始化
nn.init.normal_(layer.weight, mean=0, std=0.01)

# VAE中的重参数化技巧
z = mu + sigma * torch.randn_like(sigma)
```

#### Softmax分布（多分类）

$$
P(y = k | x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

```python
# 多分类输出层
logits = model(x)
probs = F.softmax(logits, dim=-1)
loss = F.cross_entropy(logits, target)
```

### 3.3 期望与方差

**期望（均值）**：
$$
\mathbb{E}[X] = \sum_x x \cdot P(x) \quad \text{或} \quad \int x \cdot p(x) dx
$$

**方差**：
$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

**应用**：
- Batch Normalization 计算均值和方差
- Dropout 需要在测试时调整期望

### 3.4 最大似然估计（MLE）

给定数据 $D = \{x_1, \ldots, x_n\}$，找到使数据出现概率最大的参数：

$$
\theta^* = \arg\max_\theta \prod_i P(x_i | \theta) = \arg\max_\theta \sum_i \log P(x_i | \theta)
$$

**关键洞察**：最小化交叉熵损失等价于最大化似然！

```python
# 分类问题的交叉熵损失 = 负对数似然
loss = -torch.log(probs[target])  # NLL
loss = F.cross_entropy(logits, target)  # 等价
```

---

## 第四部分：信息论

信息论提供了衡量不确定性和相似性的数学工具。

### 4.1 熵（Entropy）

衡量随机变量的不确定性：

$$
H(X) = -\sum_x P(x) \log P(x)
$$

```python
def entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10))

# 均匀分布熵最大
uniform = np.array([0.25, 0.25, 0.25, 0.25])
print(entropy(uniform))  # 1.386 (log4)

# 确定分布熵为0
certain = np.array([1.0, 0.0, 0.0, 0.0])
print(entropy(certain))  # 0
```

### 4.2 交叉熵（Cross-Entropy）

衡量两个分布的差异，是分类问题最常用的损失函数：

$$
H(P, Q) = -\sum_x P(x) \log Q(x)
$$

```python
# 交叉熵损失
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10))

# PyTorch实现
loss = F.cross_entropy(logits, labels)
```

### 4.3 KL散度（相对熵）

衡量两个分布的"距离"（非对称）：

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)
$$

**性质**：
- $D_{KL}(P \| Q) \geq 0$
- $D_{KL}(P \| Q) = 0$ 当且仅当 $P = Q$
- 非对称：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

**应用**：
- VAE的损失函数
- 知识蒸馏
- 策略梯度算法（PPO）

```python
# VAE损失 = 重构损失 + KL散度
reconstruction_loss = F.binary_cross_entropy(x_recon, x)
kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
total_loss = reconstruction_loss + kl_loss
```

### 4.4 互信息（Mutual Information）

衡量两个变量的依赖程度：

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$

**应用**：
- 对比学习（InfoNCE损失）
- 特征选择
- 表示学习

---

## 实践：手动实现梯度下降

让我们用纯数学实现一个简单的线性回归：

```python
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# 初始化参数
w = np.random.randn(1, 1)
b = np.zeros((1, 1))

# 超参数
lr = 0.1
epochs = 100

# 梯度下降
for epoch in range(epochs):
    # 前向传播
    y_pred = X @ w + b
    
    # 计算损失 (MSE)
    loss = np.mean((y_pred - y) ** 2)
    
    # 计算梯度（手动求导）
    dL_dy_pred = 2 * (y_pred - y) / len(y)  # MSE对预测值的导数
    dL_dw = X.T @ dL_dy_pred  # 链式法则
    dL_db = np.sum(dL_dy_pred)
    
    # 更新参数
    w -= lr * dL_dw
    b -= lr * dL_db
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {w[0,0]:.4f}, b: {b[0,0]:.4f}")

# 最终结果：w ≈ 2, b ≈ 1
```

---

## 总结

| 数学领域 | 核心概念 | 深度学习应用 |
|---------|---------|-------------|
| 线性代数 | 矩阵乘法、范数、SVD | 网络层运算、正则化、降维 |
| 微积分 | 导数、链式法则 | 反向传播、梯度下降 |
| 概率论 | 分布、期望、MLE | 损失函数、生成模型 |
| 信息论 | 熵、交叉熵、KL散度 | 分类损失、VAE、对比学习 |

---

## 下一步

掌握了数学基础，下一篇我们将学习**神经网络的基本结构与反向传播算法**，看看这些数学知识如何组合成强大的学习机器。
