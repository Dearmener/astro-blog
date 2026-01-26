---
title: 'GPT完全指南（四）：预训练技术详解'
description: '深入解析GPT预训练的核心技术，包括语言建模目标、大规模数据处理、分布式训练策略以及Scaling Laws'
pubDate: '2024-03-04'
heroImage: ''
category: '技术'
tags: ['GPT', '预训练', '分布式训练', 'Scaling Laws', 'LLM']
series: 'GPT完全指南'
seriesOrder: 4
---

预训练是GPT获得强大语言能力的关键阶段。在这个阶段，模型在海量文本数据上学习语言的统计规律、世界知识和推理能力。本文将深入探讨GPT预训练的核心技术，从训练目标到大规模分布式训练的工程实现。

## 语言建模目标

### 自回归语言建模

GPT采用**因果语言建模（Causal Language Modeling, CLM）**作为预训练目标，即根据前文预测下一个token：

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, ..., x_{i-1})$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLMHead(nn.Module):
    """因果语言模型头"""
    
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states, labels=None):
        hidden_states = self.ln(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 移位：用位置i的logits预测位置i+1的token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 交叉熵损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100  # 忽略padding
            )
        
        return logits, loss
```

### 为什么选择自回归？

| 方法 | 优点 | 缺点 |
|------|------|------|
| **自回归（GPT）** | 天然支持生成，因果关系清晰 | 无法看到未来上下文 |
| 掩码语言模型（BERT） | 双向上下文 | 生成需要迭代，速度慢 |
| 排列语言模型（XLNet） | 融合双向优点 | 训练复杂，效率低 |

### Next Token Prediction的本质

```python
def compute_lm_loss(model, input_ids, attention_mask=None):
    """
    计算语言模型损失
    
    input_ids: [batch_size, seq_len]
    """
    # 前向传播
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # 构造标签：将输入右移一位
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # 最后一个位置没有下一个token
    
    # 计算交叉熵
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # 计算困惑度
    perplexity = torch.exp(loss)
    
    return loss, perplexity
```

## 预训练数据处理

### 数据收集与清洗

GPT系列使用的数据集规模：

| 模型 | 训练数据 | Token数量 |
|------|----------|-----------|
| GPT-2 | WebText | 约40GB文本 |
| GPT-3 | CommonCrawl等 | 约3000亿tokens |
| GPT-4 | 多源混合 | 估计>1万亿tokens |

```python
import hashlib
from typing import List, Set
import re

class DataCleaner:
    """预训练数据清洗"""
    
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        
        # 常见的低质量模式
        self.bad_patterns = [
            r'javascript:',
            r'<script.*?</script>',
            r'cookie|Cookie',
            r'privacy policy',
            r'terms of service',
        ]
        
    def clean_text(self, text: str) -> str:
        """清洗单条文本"""
        # 1. 基础清洗
        text = text.strip()
        
        # 2. 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3. 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 4. 移除特殊Unicode字符
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text
    
    def is_quality(self, text: str) -> bool:
        """质量检查"""
        # 长度检查
        if len(text) < 100:
            return False
        
        # 重复检查
        words = text.split()
        if len(words) < 10:
            return False
        
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # 重复词太多
            return False
        
        # 低质量模式检查
        text_lower = text.lower()
        for pattern in self.bad_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def deduplicate(self, text: str) -> bool:
        """去重：返回True表示是新文档"""
        # 使用MinHash或SimHash更高效，这里简化为MD5
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            return False
        
        self.seen_hashes.add(text_hash)
        return True
    
    def process_document(self, text: str) -> str | None:
        """处理单个文档"""
        text = self.clean_text(text)
        
        if not self.is_quality(text):
            return None
        
        if not self.deduplicate(text):
            return None
        
        return text
```

### 数据打包与序列构造

```python
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterator, List
import torch

class PackedDataset(Dataset):
    """
    打包数据集：将多个短文档拼接成固定长度序列
    减少padding，提高GPU利用率
    """
    
    def __init__(
        self, 
        tokenizer, 
        documents: List[str], 
        max_length: int = 2048,
        eos_token_id: int = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id
        
        # 打包所有文档
        self.packed_sequences = self._pack_documents(documents)
    
    def _pack_documents(self, documents: List[str]) -> List[torch.Tensor]:
        """将文档打包成固定长度的序列"""
        packed = []
        current_sequence = []
        
        for doc in documents:
            # 编码文档
            tokens = self.tokenizer.encode(doc)
            tokens.append(self.eos_token_id)  # 添加文档结束符
            
            # 如果单个文档超过最大长度，截断
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # 尝试加入当前序列
            if len(current_sequence) + len(tokens) <= self.max_length:
                current_sequence.extend(tokens)
            else:
                # 当前序列已满，保存并开始新序列
                if current_sequence:
                    # Padding到max_length
                    padded = current_sequence + [self.tokenizer.pad_token_id] * (
                        self.max_length - len(current_sequence)
                    )
                    packed.append(torch.tensor(padded[:self.max_length]))
                current_sequence = tokens
        
        # 保存最后一个序列
        if current_sequence:
            padded = current_sequence + [self.tokenizer.pad_token_id] * (
                self.max_length - len(current_sequence)
            )
            packed.append(torch.tensor(padded[:self.max_length]))
        
        return packed
    
    def __len__(self):
        return len(self.packed_sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.packed_sequences[idx],
            'labels': self.packed_sequences[idx].clone()
        }


class StreamingDataset:
    """
    流式数据集：适用于超大规模数据
    不将所有数据加载到内存
    """
    
    def __init__(
        self, 
        data_files: List[str], 
        tokenizer, 
        max_length: int = 2048,
        shuffle_buffer_size: int = 10000
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
    
    def __iter__(self) -> Iterator[dict]:
        buffer = []
        
        for file_path in self.data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 编码
                    tokens = self.tokenizer.encode(line.strip())
                    if len(tokens) < 10:  # 跳过太短的
                        continue
                    
                    # 截断
                    tokens = tokens[:self.max_length]
                    
                    buffer.append(tokens)
                    
                    # 缓冲区满时shuffle并输出
                    if len(buffer) >= self.shuffle_buffer_size:
                        np.random.shuffle(buffer)
                        for item in buffer[:self.shuffle_buffer_size // 2]:
                            yield self._prepare_sample(item)
                        buffer = buffer[self.shuffle_buffer_size // 2:]
        
        # 输出剩余数据
        for item in buffer:
            yield self._prepare_sample(item)
    
    def _prepare_sample(self, tokens: List[int]) -> dict:
        # Padding
        padded = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(padded[:self.max_length])
        
        # 创建attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[len(tokens):] = 0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }
```

## 分布式训练策略

### 数据并行（Data Parallelism）

最简单的并行策略，每个GPU有完整模型副本：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def train_with_ddp(rank, world_size, model, dataset, epochs=10):
    """使用DDP进行分布式训练"""
    setup_distributed(rank, world_size)
    
    # 模型移到对应GPU并包装DDP
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 分布式采样器
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 确保每个epoch的shuffle不同
        model.train()
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(rank)
            labels = batch['labels'].to(rank)
            
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 只在rank 0打印
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()
```

### 模型并行（Model Parallelism）

当模型太大无法放入单个GPU时：

```python
import torch.nn as nn

class PipelineParallelGPT(nn.Module):
    """
    简化的流水线并行实现
    将模型的不同层放在不同GPU上
    """
    
    def __init__(self, config, num_gpus=4):
        super().__init__()
        self.num_gpus = num_gpus
        layers_per_gpu = config.num_layers // num_gpus
        
        # 嵌入层在第一个GPU
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size).to('cuda:0')
        
        # Transformer层分布到不同GPU
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            gpu_id = i // layers_per_gpu
            gpu_id = min(gpu_id, num_gpus - 1)
            layer = TransformerBlock(config).to(f'cuda:{gpu_id}')
            self.layers.append((layer, gpu_id))
        
        # 输出层在最后一个GPU
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size).to(f'cuda:{num_gpus-1}')
    
    def forward(self, input_ids):
        # 嵌入
        x = self.embed(input_ids.to('cuda:0'))
        
        # 逐层前向，需要在GPU之间传输
        for layer, gpu_id in self.layers:
            x = x.to(f'cuda:{gpu_id}')
            x = layer(x)
        
        # 输出
        x = x.to(f'cuda:{self.num_gpus-1}')
        logits = self.lm_head(x)
        
        return logits


class TransformerBlock(nn.Module):
    """Transformer块（简化版）"""
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_heads,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x
```

### ZeRO优化器

DeepSpeed的ZeRO（Zero Redundancy Optimizer）技术：

```python
import deepspeed

# DeepSpeed配置
ds_config = {
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    
    # ZeRO Stage 2: 分片优化器状态和梯度
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    
    # FP16训练
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    
    # 优化器
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    
    # 学习率调度
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 2000,
            "total_num_steps": 100000
        }
    }
}

def train_with_deepspeed(model, train_dataset):
    """使用DeepSpeed训练"""
    
    # 初始化DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=ds_config
    )
    
    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(model_engine.device)
        labels = batch['labels'].to(model_engine.device)
        
        # 前向传播
        outputs = model_engine(input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播（DeepSpeed自动处理梯度累积）
        model_engine.backward(loss)
        
        # 更新参数
        model_engine.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
```

### FSDP（Fully Sharded Data Parallel）

PyTorch原生的ZeRO-3实现：

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
import functools

def setup_fsdp_model(model, transformer_layer_cls):
    """配置FSDP"""
    
    # 混合精度配置
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    # 自动包装策略：每个Transformer层单独分片
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls}
    )
    
    # 包装模型
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
    )
    
    return model


# 使用示例
class GPTConfig:
    vocab_size = 50257
    hidden_size = 768
    num_layers = 12
    num_heads = 12

# 假设TransformerBlock是我们的层类
model = GPTModel(GPTConfig())
model = setup_fsdp_model(model, TransformerBlock)
```

## Scaling Laws

### 什么是Scaling Laws？

OpenAI发现了模型性能与三个因素之间的幂律关系：

$$L(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C}$$

其中：
- $N$：模型参数量
- $D$：训练数据量（tokens）
- $C$：计算量（FLOPs）
- $L$：测试损失

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_loss_from_params(N, D, C):
    """
    基于Scaling Laws计算预期损失
    参数来自OpenAI论文
    """
    # 临界值（论文中的估计）
    N_c = 8.8e13
    D_c = 5.4e13
    C_c = 1.6e7
    
    # 指数
    alpha_N = 0.076
    alpha_D = 0.095
    alpha_C = 0.050
    
    # 基础损失
    L_inf = 1.69
    
    # 计算损失
    L = L_inf + (N_c / N) ** alpha_N + (D_c / D) ** alpha_D
    
    return L

def compute_optimal_allocation(C_total, ratio=0.5):
    """
    给定总计算预算，计算最优的N和D分配
    Chinchilla论文发现应该1:1分配
    """
    # C ≈ 6 * N * D (每个token约6 FLOPs/参数)
    # 如果 N ∝ D，则 C ∝ N^2
    N_optimal = (C_total / 6) ** 0.5
    D_optimal = C_total / (6 * N_optimal)
    
    return N_optimal, D_optimal

# 可视化Scaling Laws
def plot_scaling_laws():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 模型大小 vs 损失
    N_range = np.logspace(6, 11, 100)  # 1M to 100B
    D_fixed = 1e12  # 固定1T tokens
    losses_N = [compute_loss_from_params(N, D_fixed, N * D_fixed * 6) for N in N_range]
    
    axes[0].loglog(N_range, losses_N)
    axes[0].set_xlabel('Parameters (N)')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Model Size')
    axes[0].grid(True)
    
    # 2. 数据量 vs 损失
    D_range = np.logspace(9, 13, 100)  # 1B to 10T tokens
    N_fixed = 1e9  # 固定1B参数
    losses_D = [compute_loss_from_params(N_fixed, D, N_fixed * D * 6) for D in D_range]
    
    axes[1].loglog(D_range, losses_D)
    axes[1].set_xlabel('Training Tokens (D)')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss vs Data Size')
    axes[1].grid(True)
    
    # 3. 计算量 vs 损失（最优分配）
    C_range = np.logspace(18, 24, 100)
    losses_C = []
    for C in C_range:
        N, D = compute_optimal_allocation(C)
        losses_C.append(compute_loss_from_params(N, D, C))
    
    axes[2].loglog(C_range, losses_C)
    axes[2].set_xlabel('Compute (FLOPs)')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Loss vs Compute (Optimal)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('scaling_laws.png', dpi=150)
    plt.show()
```

### Chinchilla Scaling Laws

DeepMind的Chinchilla论文修正了OpenAI的发现：

| 论文 | 最优比例 N:D | 建议 |
|------|--------------|------|
| GPT-3 (OpenAI) | N >> D | 扩大模型 |
| Chinchilla | N ≈ D | 同比例扩大 |

```python
def chinchilla_optimal(compute_budget):
    """
    Chinchilla最优配置
    N ≈ D，C = 6ND
    """
    # C = 6 * N * D, N = D
    # C = 6 * N^2
    N_optimal = np.sqrt(compute_budget / 6)
    D_optimal = N_optimal
    
    return {
        'params': N_optimal,
        'tokens': D_optimal,
        'compute': compute_budget,
        'ratio': N_optimal / D_optimal
    }

# 对比GPT-3和Chinchilla
print("GPT-3配置:")
print(f"  参数: 175B")
print(f"  Tokens: 300B")
print(f"  比例: {175/300:.2f}")

print("\nChinchilla最优 (相同计算量):")
compute = 6 * 175e9 * 300e9
result = chinchilla_optimal(compute)
print(f"  参数: {result['params']/1e9:.1f}B")
print(f"  Tokens: {result['tokens']/1e9:.1f}B")
print(f"  比例: {result['ratio']:.2f}")
```

## 训练稳定性技术

### 梯度裁剪

```python
def train_step_with_grad_clip(model, optimizer, batch, max_grad_norm=1.0):
    """带梯度裁剪的训练步骤"""
    
    outputs = model(**batch)
    loss = outputs.loss
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_grad_norm
    )
    
    # 检测梯度爆炸
    if total_norm > max_grad_norm * 10:
        print(f"Warning: Large gradient norm {total_norm:.2f}")
    
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item(), total_norm.item()
```

### 学习率调度

```python
import math

class CosineWarmupScheduler:
    """
    带Warmup的余弦退火学习率调度器
    GPT-3使用此策略
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # 线性warmup
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # 余弦退火
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.min_lr + (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            ) / 2

# 使用示例
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_steps=2000,
    total_steps=100000,
    min_lr=1e-5
)
```

### 损失尖峰处理

```python
class LossSpikeDetector:
    """
    检测和处理训练过程中的损失尖峰
    """
    
    def __init__(self, window_size=100, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history = []
    
    def check(self, loss):
        """
        检查当前损失是否异常
        返回True表示正常，False表示尖峰
        """
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.window_size:
            return True
        
        # 计算滑动窗口的均值和标准差
        recent = self.loss_history[-self.window_size:]
        mean = np.mean(recent)
        std = np.std(recent)
        
        # 检查是否超过阈值
        if loss > mean + self.threshold * std:
            print(f"Loss spike detected: {loss:.4f} (mean: {mean:.4f}, std: {std:.4f})")
            return False
        
        return True
    
    def get_stats(self):
        return {
            'mean': np.mean(self.loss_history[-self.window_size:]),
            'std': np.std(self.loss_history[-self.window_size:]),
            'min': np.min(self.loss_history),
            'max': np.max(self.loss_history)
        }
```

## 完整预训练流程

```python
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

class GPTPretrainer:
    """GPT预训练器"""
    
    def __init__(
        self,
        model,
        train_dataset,
        config
    ):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr=config.min_lr
        )
        
        # 混合精度
        self.scaler = GradScaler()
        
        # 数据加载
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 监控
        self.spike_detector = LossSpikeDetector()
        
    def train(self):
        """主训练循环"""
        self.model.train()
        global_step = 0
        
        # 初始化wandb
        wandb.init(project="gpt-pretraining", config=vars(self.config))
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                # 移动数据到GPU
                input_ids = batch['input_ids'].cuda()
                labels = batch['labels'].cuda()
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                
                # 混合精度前向传播
                with autocast():
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    # 梯度累积
                    loss = loss / self.config.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度累积完成后更新
                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # 检查损失尖峰
                    if self.spike_detector.check(loss.item()):
                        self.scaler.step(self.optimizer)
                    else:
                        print("Skipping update due to loss spike")
                    
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # 更新学习率
                    lr = self.scheduler.step()
                    
                    # 日志
                    if global_step % self.config.log_interval == 0:
                        wandb.log({
                            'loss': loss.item() * self.config.gradient_accumulation_steps,
                            'learning_rate': lr,
                            'grad_norm': grad_norm.item(),
                            'step': global_step
                        })
                
                epoch_loss += loss.item()
                global_step += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    'lr': f"{self.scheduler.get_lr():.2e}"
                })
                
                # 保存检查点
                if global_step % self.config.save_interval == 0:
                    self.save_checkpoint(global_step)
                
                if global_step >= self.config.total_steps:
                    break
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch} average loss: {avg_loss:.4f}")
        
        wandb.finish()
    
    def save_checkpoint(self, step):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'step': step
        }
        torch.save(checkpoint, f'checkpoint_{step}.pt')
        print(f"Saved checkpoint at step {step}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        return checkpoint['step']


# 配置类
class PretrainConfig:
    # 模型配置
    vocab_size = 50257
    hidden_size = 768
    num_layers = 12
    num_heads = 12
    
    # 训练配置
    batch_size = 32
    gradient_accumulation_steps = 8
    learning_rate = 6e-4
    min_lr = 6e-5
    weight_decay = 0.1
    max_grad_norm = 1.0
    
    # 调度
    warmup_steps = 2000
    total_steps = 100000
    num_epochs = 10
    
    # 日志和保存
    log_interval = 10
    save_interval = 5000


# 使用示例
if __name__ == "__main__":
    config = PretrainConfig()
    
    # 创建模型
    model = GPTModel(config).cuda()
    
    # 准备数据
    train_dataset = PackedDataset(tokenizer, documents, max_length=2048)
    
    # 训练
    trainer = GPTPretrainer(model, train_dataset, config)
    trainer.train()
```

## 总结

本文详细介绍了GPT预训练的核心技术：

1. **语言建模目标**：自回归因果语言建模的原理和实现
2. **数据处理**：清洗、去重、打包、流式加载
3. **分布式训练**：DDP、模型并行、ZeRO、FSDP
4. **Scaling Laws**：模型大小、数据量、计算量的关系
5. **训练稳定性**：梯度裁剪、学习率调度、损失尖峰处理

### 关键要点

- 预训练的核心是在海量数据上学习下一个token预测
- 数据质量比数量更重要，清洗和去重至关重要
- 分布式训练技术让训练超大模型成为可能
- Scaling Laws指导我们如何分配计算资源
- Chinchilla发现：参数量和数据量应该同比例扩大

下一篇文章，我们将探讨GPT的微调与对齐技术，包括SFT、RLHF和DPO等让模型"听话"的关键技术。
