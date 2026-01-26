---
title: '深度学习完全指南（八）：大语言模型LLM'
description: '从GPT到LLaMA，全面理解大语言模型的架构、训练方法、涌现能力与对齐技术'
pubDate: '2024-02-08'
heroImage: ''
category: '技术'
tags: ['深度学习', 'LLM', 'GPT', 'NLP']
series: '深度学习完全指南'
seriesOrder: 8
---

## 大语言模型的革命

2022年ChatGPT的发布标志着AI进入新纪元。大语言模型（LLM）展现了前所未有的语言理解和生成能力。

### 什么是LLM？

LLM是基于Transformer架构的超大规模语言模型，通过预训练学习人类语言的统计规律。

| 模型 | 参数量 | 发布时间 | 开发者 |
|------|--------|---------|--------|
| GPT-3 | 175B | 2020.06 | OpenAI |
| PaLM | 540B | 2022.04 | Google |
| GPT-4 | ~1.8T(估) | 2023.03 | OpenAI |
| LLaMA 2 | 7B-70B | 2023.07 | Meta |
| Claude 3 | 未公开 | 2024.03 | Anthropic |

---

## 核心架构：Decoder-Only Transformer

### 为什么是Decoder-Only？

| 架构 | 代表 | 适用场景 |
|------|------|---------|
| Encoder-only | BERT | 理解任务 |
| Encoder-Decoder | T5, BART | 翻译、摘要 |
| **Decoder-only** | GPT, LLaMA | 生成任务（最通用） |

Decoder-only通过**自回归**方式生成，更适合生成任务且统一了各种NLP任务的范式。

### GPT架构详解

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.token_embedding.weight = self.lm_head.weight
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 嵌入
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        
        # 计算注意力（带因果mask）
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果mask
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        att = att.masked_fill(causal_mask.to(x.device), float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).reshape(B, T, C)
        
        return self.proj(y)
```

### 现代LLM的改进

#### 1. RMSNorm（替代LayerNorm）

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
```

#### 2. RoPE位置编码

```python
def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """旋转位置嵌入"""
    x_r, x_i = x[..., ::2], x[..., 1::2]
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos
    return torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)
```

#### 3. SwiGLU激活函数

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

#### 4. Grouped Query Attention (GQA)

减少KV cache的显存：

```python
class GQAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads  # 通常 num_kv_heads < num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim)
```

---

## 预训练方法

### 自回归语言建模

预测下一个token：

$$
L = -\sum_{t=1}^{T} \log P(x_t | x_{<t})
$$

```python
def pretrain_step(model, batch, optimizer):
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]
    
    logits, loss = model(input_ids, targets=target_ids)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 训练数据

| 数据集 | 规模 | 内容 |
|--------|------|------|
| Common Crawl | ~60TB | 网页 |
| The Pile | 800GB | 多源混合 |
| RedPajama | 1.2T tokens | 开源复现 |
| 书籍语料 | 数十GB | 文学/技术书籍 |
| 代码数据 | 数百GB | GitHub代码 |

### 训练配置

```python
# LLaMA 7B 参考配置
config = {
    "vocab_size": 32000,
    "d_model": 4096,
    "num_heads": 32,
    "num_layers": 32,
    "max_len": 4096,
    "batch_size": 4 * 1024 * 1024,  # 4M tokens
    "learning_rate": 3e-4,
    "warmup_steps": 2000,
    "total_tokens": 1.4e12,  # 1.4T tokens
}
```

### 分布式训练

```python
# 使用DeepSpeed ZeRO-3
import deepspeed

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config={
        "train_batch_size": 1024,
        "gradient_accumulation_steps": 64,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"}
        }
    }
)
```

---

## 涌现能力（Emergent Abilities）

当模型规模超过某个阈值时，会突然展现出之前没有的能力。

### 典型涌现能力

| 能力 | 描述 | 涌现规模 |
|------|------|---------|
| Few-shot Learning | 从几个例子学习新任务 | ~10B |
| Chain-of-Thought | 逐步推理 | ~100B |
| 指令遵循 | 理解复杂指令 | ~10B |
| 代码生成 | 生成可执行代码 | ~50B |
| 多语言 | 跨语言泛化 | ~10B |

### In-Context Learning

无需微调，通过prompt中的例子学习：

```python
prompt = """
将英文翻译成中文。

English: Hello, how are you?
Chinese: 你好，你好吗？

English: I love programming.
Chinese: 我热爱编程。

English: The weather is nice today.
Chinese:"""

# 模型输出: 今天天气很好。
```

### Chain-of-Thought (CoT)

引导模型逐步推理：

```python
prompt = """
问题：Roger有5个网球。他又买了2罐网球，每罐有3个。他现在有多少个网球？

让我们一步一步思考：
1. Roger开始有5个网球
2. 他买了2罐，每罐3个，所以买了2×3=6个
3. 总共有5+6=11个

答案：11

问题：食堂有23个苹果。如果他们用了20个做午餐，又买了6个，有多少苹果？

让我们一步一步思考："""
```

---

## 指令微调（Instruction Tuning）

### SFT（Supervised Fine-Tuning）

使用高质量指令-响应对进行微调：

```python
# 数据格式
{
    "instruction": "写一首关于春天的诗",
    "input": "",
    "output": "春风轻拂柳枝摇，\n桃花含笑映碧霄。\n燕子归来寻旧巢，\n田间麦浪绿波涛。"
}

# 训练
def sft_step(model, batch):
    # 只在output部分计算损失
    input_ids = batch["input_ids"]
    labels = batch["labels"]  # input部分标记为-100
    
    outputs = model(input_ids, labels=labels)
    return outputs.loss
```

### 数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| FLAN | 1.8K任务 | 多任务指令 |
| Alpaca | 52K | GPT-4生成 |
| ShareGPT | 90K | 真实对话 |
| LIMA | 1K | 高质量精选 |
| OpenOrca | 1M+ | 大规模混合 |

---

## RLHF（人类反馈强化学习）

### 三阶段流程

```
阶段1: SFT → 初始策略模型
阶段2: 训练奖励模型 → 学习人类偏好
阶段3: PPO优化 → 对齐人类价值观
```

### 奖励模型

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward

# 训练：对比学习
def reward_loss(model, chosen_ids, rejected_ids):
    r_chosen = model(chosen_ids)
    r_rejected = model(rejected_ids)
    
    # 偏好损失
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss
```

### PPO优化

```python
def ppo_step(policy, ref_policy, reward_model, prompts):
    # 1. 采样响应
    responses = policy.generate(prompts)
    
    # 2. 计算奖励
    rewards = reward_model(prompts + responses)
    
    # 3. 计算KL惩罚
    log_probs = policy.log_prob(responses)
    ref_log_probs = ref_policy.log_prob(responses)
    kl_penalty = log_probs - ref_log_probs
    
    # 4. PPO损失
    advantages = rewards - kl_penalty * kl_coef
    ratio = torch.exp(log_probs - old_log_probs)
    
    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    loss = -torch.min(loss1, loss2).mean()
    
    return loss
```

### DPO（Direct Preference Optimization）

更简单的对齐方法，无需单独的奖励模型：

```python
def dpo_loss(policy, ref_policy, chosen, rejected, beta=0.1):
    # 策略模型对chosen/rejected的log概率
    pi_chosen = policy.log_prob(chosen)
    pi_rejected = policy.log_prob(rejected)
    
    # 参考模型的log概率
    ref_chosen = ref_policy.log_prob(chosen)
    ref_rejected = ref_policy.log_prob(rejected)
    
    # DPO损失
    log_ratio_chosen = pi_chosen - ref_chosen
    log_ratio_rejected = pi_rejected - ref_rejected
    
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()
    return loss
```

---

## 推理优化

### KV Cache

缓存已计算的Key和Value：

```python
class CachedAttention(nn.Module):
    def forward(self, x, past_kv=None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # 只对新token计算注意力
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # ...
        
        return output, (k, v)  # 返回更新的cache
```

### 量化

减少内存和加速推理：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 推测解码（Speculative Decoding）

用小模型加速大模型推理：

```python
def speculative_decode(large_model, small_model, prompt, K=4):
    # 1. 小模型快速生成K个token
    draft_tokens = small_model.generate(prompt, max_new_tokens=K)
    
    # 2. 大模型并行验证
    logits = large_model(prompt + draft_tokens)
    
    # 3. 接受/拒绝
    accepted = 0
    for i in range(K):
        p_large = softmax(logits[i])
        p_small = small_model.prob(draft_tokens[i])
        
        if random() < min(1, p_large / p_small):
            accepted += 1
        else:
            # 从修正分布采样
            break
    
    return prompt + draft_tokens[:accepted]
```

---

## 开源LLM生态

### 主流开源模型

| 模型 | 参数 | 许可 | 特点 |
|------|------|------|------|
| LLaMA 2 | 7B-70B | 商用许可 | Meta官方 |
| Mistral | 7B | Apache 2.0 | 高效架构 |
| Qwen | 1.8B-72B | 通义千问许可 | 中文优秀 |
| DeepSeek | 7B-67B | MIT | 中国团队 |
| Phi-3 | 3.8B-14B | MIT | 小模型强性能 |

### 使用示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 对话格式
prompt = """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 高效微调：LoRA

无需全量微调，只训练低秩分解矩阵：

$$
W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
$$

```python
from peft import LoraConfig, get_peft_model

# 配置LoRA
lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,          # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用LoRA的层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || 0.062%
```

---

## 总结

| 阶段 | 方法 | 目标 |
|------|------|------|
| 预训练 | 自回归语言建模 | 学习语言知识 |
| 指令微调 | SFT | 学习遵循指令 |
| 对齐 | RLHF/DPO | 符合人类价值观 |
| 部署 | 量化/剪枝/蒸馏 | 高效推理 |

---

## 下一步

LLM是理解语言的基础，但AI的应用远不止文本。下一篇我们将学习**计算机视觉应用**，了解深度学习如何理解和处理图像。
