---
title: '深度学习完全指南（五）：循环神经网络RNN/LSTM/GRU'
description: '掌握序列建模的核心架构：从基础RNN到LSTM和GRU，理解时序数据处理的关键技术'
pubDate: '2024-02-05'
heroImage: ''
category: '技术'
tags: ['深度学习', 'RNN', 'LSTM', 'NLP']
series: '深度学习完全指南'
seriesOrder: 5
---

## 为什么需要RNN？

CNN擅长处理具有空间结构的数据（图像），但对于**序列数据**（文本、语音、时间序列）却力不从心：

1. **可变长度**：句子长度不固定
2. **顺序依赖**：词的含义依赖于上下文
3. **长距离依赖**："他说他会来"中的两个"他"指的是同一人

RNN通过**循环结构**解决这些问题，让网络拥有"记忆"。

---

## 基础RNN

### 核心思想

RNN在每个时间步接收输入 $x_t$，并维护一个**隐藏状态** $h_t$ 来传递历史信息：

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$
$$
y_t = W_{hy} h_t + b_y
$$

```
x₁ → [RNN] → h₁ → y₁
       ↓
x₂ → [RNN] → h₂ → y₂
       ↓
x₃ → [RNN] → h₃ → y₃
```

### 从零实现

```python
import numpy as np

class RNNCell:
    def __init__(self, input_size, hidden_size):
        # Xavier初始化
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.hidden_size = hidden_size
    
    def forward(self, x, h_prev):
        """
        x: (batch, input_size)
        h_prev: (batch, hidden_size)
        """
        h_next = np.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        return h_next

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = RNNCell(input_size, hidden_size)
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))
        self.hidden_size = hidden_size
    
    def forward(self, x_seq):
        """
        x_seq: (seq_len, batch, input_size)
        """
        batch_size = x_seq.shape[1]
        h = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        hiddens = [h]
        
        for t in range(len(x_seq)):
            h = self.cell.forward(x_seq[t], h)
            y = h @ self.Why + self.by
            outputs.append(y)
            hiddens.append(h)
        
        return np.array(outputs), hiddens
```

### PyTorch实现

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.rnn(x)
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        
        # 取最后一个时间步
        out = self.fc(out[:, -1, :])
        return out

# 使用示例
model = SimpleRNN(input_size=100, hidden_size=256, output_size=10)
x = torch.randn(32, 50, 100)  # batch=32, seq_len=50, input=100
output = model(x)  # (32, 10)
```

---

## RNN的变体

### 双向RNN（Bidirectional RNN）

同时考虑过去和未来的信息：

```python
self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True)
# 输出维度变为 2 * hidden_size
```

```
前向: x₁ → h₁ᶠ → h₂ᶠ → h₃ᶠ
后向: x₃ → h₃ᵇ → h₂ᵇ → h₁ᵇ
输出: [h₁ᶠ; h₁ᵇ], [h₂ᶠ; h₂ᵇ], [h₃ᶠ; h₃ᵇ]
```

### 多层RNN（Stacked RNN）

```python
self.rnn = nn.RNN(input_size, hidden_size, num_layers=3)
```

```
Layer 3: h₁³ → h₂³ → h₃³
Layer 2: h₁² → h₂² → h₃²
Layer 1: h₁¹ → h₂¹ → h₃¹
Input:   x₁    x₂    x₃
```

---

## RNN的致命问题：梯度消失/爆炸

### 问题分析

反向传播通过时间（BPTT）时，梯度需要连乘：

$$
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

由于 $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(z_t)) \cdot W_{hh}$：

- 如果 $||W_{hh}|| < 1$：梯度指数衰减 → **梯度消失**
- 如果 $||W_{hh}|| > 1$：梯度指数增长 → **梯度爆炸**

### 后果

- **梯度消失**：无法学习长距离依赖
- **梯度爆炸**：训练不稳定，参数变成NaN

### 解决方案

| 问题 | 解决方案 |
|------|---------|
| 梯度爆炸 | 梯度裁剪（Gradient Clipping） |
| 梯度消失 | LSTM、GRU（门控机制） |

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

---

## LSTM（Long Short-Term Memory）

LSTM是RNN最重要的变体，通过**门控机制**和**细胞状态**解决了长距离依赖问题。

### 核心结构

LSTM有两条信息流：
- **细胞状态 $C_t$**：长期记忆，信息高速公路
- **隐藏状态 $h_t$**：短期记忆，当前输出

三个门控制信息流动：
- **遗忘门 $f_t$**：决定丢弃哪些旧信息
- **输入门 $i_t$**：决定存储哪些新信息
- **输出门 $o_t$**：决定输出哪些信息

### 数学公式

$$
\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) & \text{遗忘门} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) & \text{输入门} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) & \text{候选记忆} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{更新细胞状态} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) & \text{输出门} \\
h_t &= o_t \odot \tanh(C_t) & \text{隐藏状态}
\end{aligned}
$$

### 直觉理解

想象你在读一本书：
- **细胞状态**：你对整本书的理解
- **遗忘门**：读到新章节时，遗忘不相关的旧信息
- **输入门**：决定当前章节哪些内容值得记住
- **输出门**：根据当前问题，决定回忆哪些信息

### 从零实现

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # 合并所有门的权重
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * 0.01
        self.b = np.zeros((1, 4 * hidden_size))
        self.hidden_size = hidden_size
    
    def forward(self, x, h_prev, c_prev):
        """
        x: (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        """
        # 拼接输入和上一隐藏状态
        combined = np.concatenate([x, h_prev], axis=1)
        
        # 计算所有门
        gates = combined @ self.W + self.b
        
        # 分割为四个门
        i, f, o, g = np.split(gates, 4, axis=1)
        
        # 激活函数
        i = self.sigmoid(i)  # 输入门
        f = self.sigmoid(f)  # 遗忘门
        o = self.sigmoid(o)  # 输出门
        g = np.tanh(g)       # 候选记忆
        
        # 更新细胞状态和隐藏状态
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        
        return h_next, c_next
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

### PyTorch实现

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embed = self.embedding(x)  # (batch, seq_len, embed_size)
        out, (h_n, c_n) = self.lstm(embed)
        # 取最后一个隐藏状态
        out = self.fc(h_n[-1])
        return out
```

---

## GRU（Gated Recurrent Unit）

GRU是LSTM的简化版本，只有两个门，计算更高效。

### 核心结构

- **重置门 $r_t$**：控制如何结合新输入与旧记忆
- **更新门 $z_t$**：控制保留多少旧信息

### 数学公式

$$
\begin{aligned}
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t]) & \text{更新门} \\
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t]) & \text{重置门} \\
\tilde{h}_t &= \tanh(W \cdot [r_t \odot h_{t-1}, x_t]) & \text{候选隐藏} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t & \text{最终隐藏}
\end{aligned}
$$

### 从零实现

```python
class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.hidden_size = hidden_size
    
    def forward(self, x, h_prev):
        combined = np.concatenate([x, h_prev], axis=1)
        
        z = self.sigmoid(combined @ self.Wz)  # 更新门
        r = self.sigmoid(combined @ self.Wr)  # 重置门
        
        combined_reset = np.concatenate([x, r * h_prev], axis=1)
        h_candidate = np.tanh(combined_reset @ self.Wh)
        
        h_next = (1 - z) * h_prev + z * h_candidate
        
        return h_next
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

### PyTorch实现

```python
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embed = self.embedding(x)
        out, h_n = self.gru(embed)
        out = self.fc(h_n[-1])
        return out
```

---

## LSTM vs GRU 对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个（遗忘、输入、输出） | 2个（重置、更新） |
| 状态 | 细胞状态 + 隐藏状态 | 仅隐藏状态 |
| 参数量 | 更多 | 更少（约75%） |
| 训练速度 | 较慢 | 较快 |
| 长序列性能 | 通常更好 | 短序列相当 |
| 适用场景 | 复杂任务、长序列 | 资源受限、短序列 |

**选择建议**：
- 默认先试GRU，速度快
- 如果效果不好，换LSTM
- 现在更推荐Transformer

---

## 应用场景

### 1. 语言模型

预测下一个词：

```python
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        logits = self.fc(out)
        return logits, hidden
    
    def generate(self, start_token, max_len=50, temperature=1.0):
        self.eval()
        tokens = [start_token]
        hidden = None
        
        for _ in range(max_len):
            x = torch.tensor([[tokens[-1]]])
            logits, hidden = self.forward(x, hidden)
            probs = F.softmax(logits[0, -1] / temperature, dim=0)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            if next_token == EOS_TOKEN:
                break
        
        return tokens
```

### 2. 序列分类

情感分析、垃圾邮件检测：

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        # 使用平均池化
        out = out.mean(dim=1)
        return self.fc(out)
```

### 3. 序列到序列（Seq2Seq）

机器翻译、文本摘要：

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        embed = self.embedding(x)
        _, (h, c) = self.lstm(embed)
        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        logits = self.fc(out)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt):
        hidden = self.encoder(src)
        output, _ = self.decoder(tgt, hidden)
        return output
```

### 4. 时间序列预测

股票预测、天气预报：

```python
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 预测下一个时间步
        return out
```

---

## 实战：文本情感分类

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 数据预处理
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

# 模型定义
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

# 训练
model = SentimentLSTM(len(vocab))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for labels, texts in train_loader:
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

---

## RNN的局限与Transformer的崛起

尽管LSTM/GRU解决了梯度问题，RNN仍有根本限制：

| 问题 | 描述 |
|------|------|
| **顺序计算** | 无法并行，训练慢 |
| **长距离依赖** | 虽然改善，但仍有限制 |
| **固定表示** | 编码器输出为固定向量 |

**Transformer**通过**注意力机制**彻底解决了这些问题，成为当前NLP的主流架构。

---

## 总结

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| 基础RNN | 简单，梯度问题严重 | 教学演示 |
| LSTM | 三门控，长距离依赖 | 复杂序列任务 |
| GRU | 两门控，参数少 | 资源受限场景 |
| Bi-RNN | 双向信息 | 需要完整上下文 |

---

## 下一步

RNN家族帮助我们理解了序列建模，但现在**Transformer**已经全面取代了RNN。下一篇我们将深入学习Transformer架构，了解注意力机制如何改变了深度学习。
