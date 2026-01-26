---
title: '深度学习完全指南（六）：Transformer架构详解'
description: '从注意力机制到完整Transformer，深入理解这个改变NLP和CV的革命性架构'
pubDate: '2024-02-06'
heroImage: ''
category: '技术'
tags: ['深度学习', 'Transformer', '注意力机制']
series: '深度学习完全指南'
seriesOrder: 6
---

## Transformer的革命性意义

2017年Google发表的论文"Attention is All You Need"提出了Transformer，彻底改变了深度学习：

- **NLP领域**：BERT、GPT、T5、LLaMA等大模型的基础
- **CV领域**：ViT、DETR、Swin Transformer
- **多模态**：CLIP、DALL-E、Flamingo

**核心创新**：完全基于**注意力机制**，抛弃了RNN的循环结构。

---

## 注意力机制基础

### 什么是注意力？

注意力机制让模型能够**动态关注**输入的不同部分。就像人阅读时，会根据问题关注文章的不同段落。

### Query-Key-Value 框架

注意力可以抽象为：
- **Query (Q)**：当前关注的查询
- **Key (K)**：用于匹配的索引
- **Value (V)**：实际内容

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

```python
def attention(Q, K, V, mask=None):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 可选：应用mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

### 为什么除以 √d_k？

当 $d_k$ 较大时，$QK^T$ 的方差也会变大，导致 softmax 的梯度变小。除以 $\sqrt{d_k}$ 起到缩放作用，稳定训练。

---

## 多头注意力（Multi-Head Attention）

单个注意力只能关注一种模式，多头注意力允许模型同时关注**不同位置**和**不同表示子空间**的信息。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

其中每个头：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 现在: (batch, num_heads, seq_len, d_k)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output, attn_weights
```

---

## 位置编码（Positional Encoding）

Transformer没有循环结构，无法感知序列顺序。位置编码为每个位置添加唯一的向量：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin(pos / 10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i/d_{model}})
\end{aligned}
$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

**为什么用三角函数？**
- 不同位置有唯一编码
- 可以表示相对位置（通过线性变换）
- 可以外推到更长序列

### 其他位置编码方法

| 方法 | 描述 | 使用场景 |
|------|------|---------|
| 正弦位置编码 | 原始Transformer | 通用 |
| 可学习位置编码 | 作为参数学习 | BERT、GPT |
| 相对位置编码 | 编码相对距离 | Transformer-XL |
| RoPE | 旋转位置嵌入 | LLaMA、GLM |
| ALiBi | 注意力线性偏置 | BLOOM |

---

## 前馈神经网络（Feed-Forward Network）

每个位置独立应用的两层全连接网络：

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

通常 $d_{ff} = 4 \times d_{model}$。

### 现代变体

**GLU（Gated Linear Unit）**：
```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## 层归一化（Layer Normalization）

对每个样本的特征进行归一化：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### Pre-LN vs Post-LN

```python
# Post-LN (原始Transformer)
x = x + self.attn(self.norm1(x))

# Pre-LN (更稳定，现代常用)
x = self.norm1(x + self.attn(x))
```

---

## 完整Transformer架构

### Encoder Layer

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
```

### Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 带mask的自注意力（防止看到未来）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力（关注编码器输出）
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x
```

### 完整Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def encode(self, src, src_mask=None):
        x = self.dropout(self.pos_encoding(self.encoder_embed(src) * math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_encoding(self.decoder_embed(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output
```

---

## Mask机制

### Padding Mask

屏蔽填充token：

```python
def create_padding_mask(seq, pad_idx=0):
    # seq: (batch, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    # 输出: (batch, 1, 1, seq_len)
```

### Causal Mask（Look-ahead Mask）

防止decoder看到未来token：

```python
def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask  # 下三角为True
```

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

---

## Transformer变体

### Encoder-only（BERT类型）

用于理解任务：分类、NER、问答

```python
class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.pos_encoding(self.embedding(x))
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.norm(x)
```

### Decoder-only（GPT类型）

用于生成任务：文本生成、代码补全

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        causal_mask = create_causal_mask(x.size(1))
        x = self.pos_encoding(self.embedding(x))
        for layer in self.decoder_layers:
            x = layer(x, causal_mask)
        x = self.norm(x)
        return self.lm_head(x)
```

### Encoder-Decoder（T5类型）

用于序列到序列：翻译、摘要

---

## 高效Transformer

原始自注意力的复杂度是 $O(n^2)$，对长序列不可行。

### 常见优化方法

| 方法 | 思路 | 代表模型 |
|------|------|---------|
| 稀疏注意力 | 只计算部分位置 | Sparse Transformer, BigBird |
| 线性注意力 | 用核函数近似 | Linear Transformer, Performer |
| 局部+全局 | 混合注意力模式 | Longformer, LongT5 |
| 分块计算 | 内存优化 | Flash Attention |

### Flash Attention

通过分块计算和内核融合大幅提升效率：

```python
# 使用Flash Attention（需要安装flash-attn）
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

---

## Vision Transformer (ViT)

将Transformer应用于图像：

```python
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, d_model, num_heads, num_layers):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_model * 4),
            num_layers
        )
        
        self.mlp_head = nn.Linear(d_model, num_classes)
        self.patch_size = patch_size
    
    def forward(self, x):
        # x: (batch, 3, H, W)
        patches = self.patchify(x)  # (batch, num_patches, patch_dim)
        x = self.patch_embed(patches)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        
        # 分类头
        return self.mlp_head(x[:, 0])
    
    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p * C)
        return x
```

---

## 训练技巧

### 学习率调度

Transformer通常使用warmup + 衰减：

```python
class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
```

### Label Smoothing

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_prob = F.log_softmax(pred, dim=-1)
        loss = -(smooth_target * log_prob).sum(dim=-1).mean()
        return loss
```

---

## 总结

| 组件 | 作用 |
|------|------|
| 自注意力 | 建模任意位置间的依赖 |
| 多头注意力 | 捕捉多种关系模式 |
| 位置编码 | 注入序列位置信息 |
| FFN | 逐位置的非线性变换 |
| 残差连接 | 梯度流动，更深网络 |
| 层归一化 | 稳定训练 |

---

## 下一步

Transformer是理解现代AI的关键。下一篇我们将学习**生成模型**，包括GAN、VAE和Diffusion Model，了解AI如何创造新内容。
