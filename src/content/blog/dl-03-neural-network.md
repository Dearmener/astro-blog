---
title: '深度学习完全指南（三）：神经网络基础与反向传播'
description: '深入理解神经网络的工作原理：从感知机到多层网络，掌握前向传播与反向传播算法'
pubDate: '2024-02-03'
heroImage: ''
category: '技术'
tags: ['深度学习', '神经网络', '反向传播']
series: '深度学习完全指南'
seriesOrder: 3
---

## 从生物神经元到人工神经元

### 生物神经元

大脑中的神经元通过以下方式工作：
1. **树突**：接收来自其他神经元的信号
2. **细胞体**：汇总所有输入信号
3. **轴突**：当信号超过阈值时，向其他神经元发送信号

### 人工神经元（感知机）

人工神经元模拟了这一过程：

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(w^T x + b)
$$

其中：
- $x$：输入特征
- $w$：权重（学习参数）
- $b$：偏置
- $f$：激活函数

```python
import numpy as np

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0
    
    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)
    
    def activation(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid
```

---

## 激活函数详解

激活函数为神经网络引入**非线性**，没有它，多层网络等价于单层。

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

**特点**：
- 输出范围 (0, 1)，可解释为概率
- 梯度消失问题：两端梯度趋近于0
- 输出非零中心化

**应用**：二分类输出层、门控机制（LSTM/GRU）

### Tanh

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)
$$

```python
def tanh(x):
    return np.tanh(x)
```

**特点**：
- 输出范围 (-1, 1)，零中心化
- 仍有梯度消失问题
- 比 Sigmoid 收敛更快

**应用**：RNN隐藏层、归一化输出

### ReLU（Rectified Linear Unit）

$$
\text{ReLU}(x) = \max(0, x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

```python
def relu(x):
    return np.maximum(0, x)
```

**特点**：
- 计算简单高效
- 缓解梯度消失
- Dead ReLU问题：负输入永远输出0

**应用**：CNN和MLP的默认选择

### Leaky ReLU / PReLU

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
```

**特点**：解决 Dead ReLU 问题

### GELU（Gaussian Error Linear Unit）

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)
$$

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

**特点**：
- 平滑非线性
- Transformer中的标准选择（BERT、GPT）

### Swish / SiLU

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

```python
def swish(x):
    return x * sigmoid(x)
```

**特点**：
- 自门控机制
- EfficientNet、Vision Transformer中使用

### 激活函数对比

| 函数 | 范围 | 优点 | 缺点 | 应用场景 |
|------|------|------|------|---------|
| Sigmoid | (0,1) | 概率解释 | 梯度消失 | 二分类输出 |
| Tanh | (-1,1) | 零中心化 | 梯度消失 | RNN |
| ReLU | [0,∞) | 计算快 | Dead ReLU | CNN默认 |
| GELU | (-0.17,∞) | 平滑 | 计算稍复杂 | Transformer |
| Softmax | (0,1) | 多类概率 | - | 多分类输出 |

---

## 多层感知机（MLP）

将多个神经元层叠起来形成深度网络：

```
输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
```

### PyTorch实现

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 创建网络
model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
```

### 从零实现

```python
class MLPFromScratch:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier初始化
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ w + b
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = self.softmax(z)  # 输出层
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

---

## 前向传播

前向传播是从输入到输出的计算过程：

```
x → [Linear → ReLU] → [Linear → ReLU] → [Linear → Softmax] → ŷ
```

数学表示：

$$
\begin{aligned}
z^{(1)} &= W^{(1)}x + b^{(1)} \\
a^{(1)} &= \text{ReLU}(z^{(1)}) \\
z^{(2)} &= W^{(2)}a^{(1)} + b^{(2)} \\
a^{(2)} &= \text{ReLU}(z^{(2)}) \\
z^{(3)} &= W^{(3)}a^{(2)} + b^{(3)} \\
\hat{y} &= \text{Softmax}(z^{(3)})
\end{aligned}
$$

---

## 损失函数

损失函数衡量预测与真实值的差距。

### 均方误差（MSE）- 回归任务

$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# PyTorch
loss = nn.MSELoss()(y_pred, y_true)
```

### 交叉熵损失 - 分类任务

$$
L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

对于 one-hot 编码的标签，简化为：
$$
L = -\log(\hat{y}_{true\_class})
$$

```python
def cross_entropy_loss(y_true, y_pred):
    # y_true: one-hot 或类别索引
    # y_pred: softmax输出
    epsilon = 1e-10
    return -np.sum(y_true * np.log(y_pred + epsilon)) / len(y_true)

# PyTorch（内部包含softmax）
loss = nn.CrossEntropyLoss()(logits, labels)
```

### 二元交叉熵 - 二分类

$$
L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

```python
# PyTorch
loss = nn.BCEWithLogitsLoss()(logits, labels)
```

---

## 反向传播算法

反向传播是深度学习最核心的算法，它高效地计算损失函数对所有参数的梯度。

### 链式法则回顾

如果 $y = f(g(x))$，则：
$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

### 计算图视角

神经网络可以表示为计算图，反向传播就是在图上从输出到输入反向计算梯度。

```
前向: x → z₁ → a₁ → z₂ → a₂ → L
反向: ∂L/∂x ← ∂L/∂z₁ ← ∂L/∂a₁ ← ∂L/∂z₂ ← ∂L/∂a₂ ← ∂L/∂L=1
```

### 详细推导

以两层网络为例，计算 $\frac{\partial L}{\partial W^{(1)}}$：

**Step 1**: 计算输出层梯度
$$
\frac{\partial L}{\partial z^{(2)}} = \hat{y} - y \quad \text{(softmax + cross-entropy)}
$$

**Step 2**: 计算第二层权重梯度
$$
\frac{\partial L}{\partial W^{(2)}} = (a^{(1)})^T \cdot \frac{\partial L}{\partial z^{(2)}}
$$

**Step 3**: 反向传播到第一层
$$
\frac{\partial L}{\partial a^{(1)}} = \frac{\partial L}{\partial z^{(2)}} \cdot (W^{(2)})^T
$$

**Step 4**: 通过ReLU
$$
\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}} \odot \mathbb{1}_{z^{(1)} > 0}
$$

**Step 5**: 计算第一层权重梯度
$$
\frac{\partial L}{\partial W^{(1)}} = x^T \cdot \frac{\partial L}{\partial z^{(1)}}
$$

### 代码实现

```python
class MLPFromScratch:
    # ... 前面的代码 ...
    
    def backward(self, y_true):
        m = y_true.shape[0]
        grads_w = []
        grads_b = []
        
        # 输出层梯度 (softmax + cross-entropy)
        dz = self.activations[-1] - y_true  # (batch, classes)
        
        # 反向遍历每一层
        for i in range(len(self.weights) - 1, -1, -1):
            # 权重和偏置梯度
            dw = self.activations[i].T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            
            if i > 0:
                # 传播到前一层
                da = dz @ self.weights[i].T
                # ReLU的梯度
                dz = da * (self.z_values[i-1] > 0)
        
        return grads_w, grads_b
    
    def update(self, grads_w, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]
```

---

## 完整训练流程

```python
# 数据准备
X_train, y_train = load_mnist()  # (60000, 784), (60000, 10)
X_test, y_test = load_mnist(train=False)

# 创建模型
model = MLPFromScratch([784, 256, 128, 10])

# 训练参数
lr = 0.1
epochs = 10
batch_size = 64

# 训练循环
for epoch in range(epochs):
    # 随机打乱数据
    indices = np.random.permutation(len(X_train))
    
    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        # 获取batch
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        # 前向传播
        y_pred = model.forward(X_batch)
        
        # 计算损失
        loss = cross_entropy_loss(y_batch, y_pred)
        total_loss += loss
        
        # 反向传播
        grads_w, grads_b = model.backward(y_batch)
        
        # 更新参数
        model.update(grads_w, grads_b, lr)
    
    # 评估
    y_pred_test = model.forward(X_test)
    accuracy = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")
```

---

## 梯度问题与解决方案

### 梯度消失

**问题**：深层网络中，梯度在反向传播时指数级衰减，导致浅层参数几乎不更新。

**原因**：
- Sigmoid/Tanh 的导数最大值 < 1
- 连续相乘导致梯度趋近于0

**解决方案**：
- 使用 ReLU 及其变体
- 残差连接（ResNet）
- 批量归一化（BatchNorm）
- 适当的权重初始化

### 梯度爆炸

**问题**：梯度在反向传播时指数级增长，导致参数更新过大。

**解决方案**：
- 梯度裁剪（Gradient Clipping）
- 权重正则化
- 适当的学习率

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 权重初始化

好的初始化可以加速训练并避免梯度问题。

### Xavier/Glorot 初始化

适用于 Tanh、Sigmoid：

$$
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

```python
nn.init.xavier_uniform_(layer.weight)
```

### He/Kaiming 初始化

适用于 ReLU：

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
$$

```python
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

---

## PyTorch 自动微分

PyTorch 的 autograd 自动处理反向传播：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练步骤
for epoch in range(epochs):
    for X_batch, y_batch in dataloader:
        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
```

---

## 总结

| 概念 | 要点 |
|------|------|
| 神经元 | 线性变换 + 非线性激活 |
| 激活函数 | 引入非线性，ReLU是默认选择 |
| 前向传播 | 从输入到输出的计算 |
| 损失函数 | 衡量预测与真实的差距 |
| 反向传播 | 链式法则计算梯度 |
| 梯度问题 | 消失/爆炸，用ReLU、残差、BN解决 |
| 初始化 | Xavier(tanh) / He(ReLU) |

---

## 下一步

掌握了神经网络基础后，下一篇我们将深入学习**卷积神经网络（CNN）**，了解它如何革新了计算机视觉领域。
