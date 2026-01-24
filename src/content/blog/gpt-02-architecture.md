---
title: 'GPT完全指南（二）：架构深度解析'
description: '深入剖析GPT的Decoder-only Transformer架构，包括注意力机制、位置编码、LayerNorm等核心组件'
pubDate: '2024-03-02'
heroImage: ''
category: '技术'
tags: ['GPT', 'Transformer', '注意力机制', '深度学习']
series: 'GPT完全指南'
seriesOrder: 2
---

## GPT架构概览

GPT采用Decoder-only Transformer架构，与原始Transformer的主要区别在于只使用解码器部分，并通过因果掩码实现自回归生成。

### 整体结构

```
Input IDs → Token Embedding + Position Embedding
                    ↓
            ┌─────────────┐
            │ Transformer │ × N layers
            │   Block     │
            └─────────────┘
                    ↓
              Layer Norm
                    ↓
              LM Head → Logits → Next Token
```

### PyTorch完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTConfig:
    """GPT配置类"""
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1,
        bias=True,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias

class GPT(nn.Module):
    """GPT模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.n_positions, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享：embedding和lm_head
        self.transformer.wte.weight = self.lm_head.weight
        
        # 初始化
        self.apply(self._init_weights)
        
        # 计算参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"参数量: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.n_positions, f"序列长度 {t} 超过最大位置 {self.config.n_positions}"
        
        # 位置索引
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Token嵌入 + 位置嵌入
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer块
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        # 计算logits
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时只计算最后一个token
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
```

---

## 核心组件详解

### 1. Token Embedding

将离散的token ID映射为连续的向量表示。

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        # output: (batch_size, seq_len, embed_dim)
        return self.embedding(x)

# 示例
vocab_size = 50257
embed_dim = 768
embedding = TokenEmbedding(vocab_size, embed_dim)

input_ids = torch.tensor([[100, 200, 300]])  # (1, 3)
embeddings = embedding(input_ids)  # (1, 3, 768)
```

### 2. 位置编码

#### 可学习位置编码（GPT使用）

```python
class LearnedPositionalEmbedding(nn.Module):
    """可学习的位置嵌入"""
    def __init__(self, max_positions, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_positions, embed_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.embedding(positions)  # (seq_len, embed_dim)
```

#### 正弦位置编码（原始Transformer）

```python
class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, max_positions, embed_dim):
        super().__init__()
        
        pe = torch.zeros(max_positions, embed_dim)
        position = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.pe[:, :x.size(1), :]
```

#### RoPE旋转位置编码（现代LLM常用）

```python
class RotaryPositionalEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding"""
    def __init__(self, dim, max_positions=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.base = base
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算
        self._set_cos_sin_cache(max_positions)
    
    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x, seq_len=None):
        if seq_len > self.max_positions:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )

def rotate_half(x):
    """将输入分成两半并旋转"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 3. 因果自注意力

GPT的核心：只能看到当前位置及之前的token。

```python
class CausalSelfAttention(nn.Module):
    """因果自注意力"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Q, K, V投影（合并为一个矩阵提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 因果掩码
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 计算Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头格式: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        # att = (q @ k^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 应用因果掩码
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 加权求和
        y = att @ v  # (B, n_head, T, head_dim)
        
        # 合并多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y
```

### 4. Flash Attention（优化版本）

```python
# 使用PyTorch 2.0的Flash Attention
class FlashCausalSelfAttention(nn.Module):
    """使用Flash Attention的因果自注意力"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Flash Attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True  # 因果掩码
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y
```

### 5. MLP（前馈网络）

```python
class MLP(nn.Module):
    """前馈神经网络"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# GELU激活函数
class GELU(nn.Module):
    """Gaussian Error Linear Unit"""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))
```

### 6. GLU变体（现代LLM常用）

```python
class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit (LLaMA使用)"""
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * config.n_embd * 4 / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)  # 对齐到256
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
    
    def forward(self, x):
        # SwiGLU: swish(xW1) * (xW3)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 7. Layer Normalization

```python
class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA使用)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMSNorm: x / sqrt(mean(x^2)) * weight
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### 8. Transformer Block

```python
class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Pre-LN结构（GPT-2开始使用）
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Post-LN结构（原始Transformer）
class PostLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x
```

---

## 多头注意力可视化

```python
def visualize_attention(model, text, tokenizer):
    """可视化注意力权重"""
    import matplotlib.pyplot as plt
    
    # 编码文本
    input_ids = tokenizer.encode(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # 获取注意力权重
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions  # list of (batch, heads, seq, seq)
    
    # 可视化第一层的注意力
    layer_attention = attentions[0][0]  # (heads, seq, seq)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for head_idx, ax in enumerate(axes.flat):
        if head_idx >= layer_attention.size(0):
            break
        
        att = layer_attention[head_idx].numpy()
        im = ax.imshow(att, cmap='viridis')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        ax.set_title(f'Head {head_idx + 1}')
    
    plt.tight_layout()
    plt.show()
```

---

## KV Cache

推理优化的关键技术：缓存已计算的K和V。

```python
class CausalSelfAttentionWithCache(nn.Module):
    """带KV Cache的因果自注意力"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    
    def forward(self, x, layer_past=None, use_cache=False):
        B, T, C = x.size()
        
        # 计算Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 使用缓存
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # 保存缓存
        present = (k, v) if use_cache else None
        
        # 注意力计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 因果掩码（只需要新token的掩码）
        if layer_past is not None:
            # 增量推理：q只有新token，k/v有所有token
            pass  # 不需要掩码，因为q只关注之前的所有k
        else:
            # 首次推理：需要完整的因果掩码
            mask = torch.tril(torch.ones(T, T, device=x.device))
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y, present
```

---

## 模型变体对比

### GPT-2 vs GPT-3 vs LLaMA

| 特性 | GPT-2 | GPT-3 | LLaMA |
|------|-------|-------|-------|
| LayerNorm | Pre-LN | Pre-LN | RMSNorm |
| 位置编码 | 可学习 | 可学习 | RoPE |
| 激活函数 | GELU | GELU | SwiGLU |
| 注意力 | MHA | MHA | GQA |
| FFN维度 | 4x | 4x | 2.67x (with GLU) |

### 完整LLaMA风格实现

```python
class LLaMABlock(nn.Module):
    """LLaMA风格的Transformer Block"""
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)
    
    def forward(self, x, freqs_cis=None, mask=None, cache=None):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, cache)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class GroupedQueryAttention(nn.Module):
    """GQA: Grouped Query Attention"""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_head
        self.n_kv_heads = config.n_kv_heads  # KV头数少于Q头数
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_heads // self.n_kv_heads  # 每个KV头重复次数
        
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
    
    def forward(self, x, freqs_cis=None, mask=None, cache=None):
        B, T, _ = x.shape
        
        # 计算Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # 应用RoPE
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # 处理KV cache
        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        
        # 重复K, V以匹配Q的头数
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        
        # 转置并计算注意力
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(output)
```

---

## 注意力优化技术

### Multi-Query Attention (MQA)

```python
class MultiQueryAttention(nn.Module):
    """MQA: 所有头共享一组K, V"""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.head_dim, bias=False)  # 只有1组
        self.wv = nn.Linear(config.n_embd, self.head_dim, bias=False)  # 只有1组
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
    
    def forward(self, x):
        B, T, _ = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        
        # K, V广播到所有头
        k = k.expand(-1, self.n_heads, -1, -1)
        v = v.expand(-1, self.n_heads, -1, -1)
        
        # 注意力计算...
```

### Sliding Window Attention

```python
class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力（Mistral使用）"""
    def __init__(self, config, window_size=4096):
        super().__init__()
        self.window_size = window_size
        # ... 其他初始化
    
    def forward(self, x):
        B, T, _ = x.shape
        
        # 创建滑动窗口掩码
        mask = torch.ones(T, T, device=x.device)
        mask = torch.triu(mask, diagonal=-self.window_size)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, float('-inf'))
        
        # ... 注意力计算
```

---

## 参数量计算

```python
def count_parameters(config):
    """计算GPT模型参数量"""
    d = config.n_embd
    L = config.n_layer
    V = config.vocab_size
    
    # Token Embedding: V × d
    embedding_params = V * d
    
    # Position Embedding: n_positions × d
    position_params = config.n_positions * d
    
    # 每层Transformer Block
    # Attention: 4 × d × d (Q, K, V, O)
    # MLP: 2 × d × (4d) = 8d²
    # LayerNorm: 2 × 2d = 4d
    block_params = 4 * d * d + 8 * d * d + 4 * d
    total_block_params = L * block_params
    
    # 最终LayerNorm: 2d
    final_ln_params = 2 * d
    
    # LM Head: 与embedding共享，不额外计算
    
    total = embedding_params + position_params + total_block_params + final_ln_params
    
    print(f"Embedding: {embedding_params / 1e6:.2f}M")
    print(f"Position: {position_params / 1e6:.2f}M")
    print(f"Transformer Blocks: {total_block_params / 1e6:.2f}M")
    print(f"Total: {total / 1e6:.2f}M")
    
    return total

# GPT-2 Small
config = GPTConfig(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12)
count_parameters(config)  # ~124M
```

---

## 小结

### GPT架构要点

| 组件 | 作用 | 关键点 |
|------|------|--------|
| Token Embedding | 离散→连续 | 权重共享 |
| Position Encoding | 序列位置信息 | RoPE更优 |
| Causal Attention | 自回归约束 | 掩码实现 |
| MLP | 非线性变换 | GLU变体 |
| LayerNorm | 稳定训练 | Pre-LN/RMSNorm |

### 现代优化

| 技术 | 效果 |
|------|------|
| Flash Attention | 显存↓，速度↑ |
| KV Cache | 推理速度↑ |
| GQA/MQA | KV Cache↓ |
| RoPE | 更好的位置外推 |
| SwiGLU | 更好的表达能力 |

**下一篇**：分词与词表构建，深入理解BPE、WordPiece、SentencePiece等分词算法。
