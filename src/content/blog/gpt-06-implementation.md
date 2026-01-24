---
title: 'GPT完全指南（六）：从零实现miniGPT'
description: '从零开始用PyTorch实现一个完整的GPT模型，包括分词器、模型架构、训练循环和文本生成的全部代码'
pubDate: '2024-03-06'
heroImage: ''
category: '技术'
tags: ['GPT', 'PyTorch', '深度学习', 'Transformer', '实战']
series: 'GPT完全指南'
seriesOrder: 6
---

理论学了再多，不如动手实现一遍。本文将从零开始，用PyTorch实现一个完整的GPT模型——miniGPT。我们将构建所有核心组件，从数据处理到模型架构，从训练循环到文本生成。

## 项目结构

```
minigpt/
├── tokenizer.py      # 字符级分词器
├── model.py          # GPT模型架构
├── dataset.py        # 数据集处理
├── train.py          # 训练循环
├── generate.py       # 文本生成
├── config.py         # 配置文件
└── main.py           # 主程序入口
```

## 配置定义

```python
# config.py
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """GPT模型配置"""
    # 模型架构
    vocab_size: int = 65        # 词表大小（字符级）
    block_size: int = 256       # 上下文窗口大小
    n_layer: int = 6            # Transformer层数
    n_head: int = 6             # 注意力头数
    n_embd: int = 384           # 嵌入维度
    dropout: float = 0.2        # Dropout率
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    
    # 设备
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
```

## 字符级分词器

为了简单起见，我们实现一个字符级分词器：

```python
# tokenizer.py
from typing import List, Dict

class CharTokenizer:
    """简单的字符级分词器"""
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def fit(self, text: str):
        """从文本构建词表"""
        # 获取所有唯一字符
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # 创建映射
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {''.join(chars)}")
        
        return self
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token ids"""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """将token ids解码为文本"""
        return ''.join([self.idx_to_char[i] for i in ids])
    
    def save(self, path: str):
        """保存分词器"""
        import json
        with open(path, 'w') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """加载分词器"""
        import json
        tokenizer = cls()
        with open(path, 'r') as f:
            data = json.load(f)
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        tokenizer.vocab_size = len(tokenizer.char_to_idx)
        return tokenizer


# 使用示例
if __name__ == "__main__":
    text = "Hello, World! 你好，世界！"
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    
    encoded = tokenizer.encode("Hello")
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
```

## 数据集处理

```python
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        Args:
            data: 编码后的文本数据 (token ids)
            block_size: 上下文窗口大小
        """
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取block_size长度的上下文
        x = self.data[idx:idx + self.block_size]
        # 目标是下一个字符
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y


def prepare_data(
    text: str, 
    tokenizer, 
    block_size: int,
    train_split: float = 0.9
) -> Tuple[TextDataset, TextDataset]:
    """准备训练和验证数据集"""
    
    # 编码整个文本
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # 分割数据
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 创建数据集
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    print(f"Training data: {len(train_data):,} tokens")
    print(f"Validation data: {len(val_data):,} tokens")
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")
    
    return train_dataset, val_dataset


def get_batch(
    dataset: TextDataset, 
    batch_size: int, 
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取随机批次"""
    # 随机采样索引
    ix = torch.randint(len(dataset), (batch_size,))
    
    # 获取数据
    x = torch.stack([dataset[i][0] for i in ix])
    y = torch.stack([dataset[i][1] for i in ix])
    
    return x.to(device), y.to(device)
```

## GPT模型架构

现在让我们实现GPT的核心——Transformer架构：

```python
# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig

class CausalSelfAttention(nn.Module):
    """因果自注意力机制"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Q, K, V 投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # 因果掩码：确保只能看到之前的token
        # 使用register_buffer让它成为模型的一部分，但不是参数
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头格式: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        
        # 注意力计算: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        
        # 应用因果掩码
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax + Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 应用到values: (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v
        
        # 重新组合多头: (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer块"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm架构（GPT-2风格）
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT语言模型"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            # Token嵌入
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout
            drop = nn.Dropout(config.dropout),
            # Transformer块
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终Layer Norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # 语言模型头（与token嵌入共享权重）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # 权重绑定
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 特殊初始化：残差投影层
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # 打印参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            idx: 输入token ids, shape (B, T)
            targets: 目标token ids, shape (B, T)，用于计算损失
        
        Returns:
            logits: 预测logits, shape (B, T, vocab_size)
            loss: 交叉熵损失（如果提供了targets）
        """
        device = idx.device
        B, T = idx.size()
        
        assert T <= self.config.block_size, f"序列长度 {T} 超过最大长度 {self.config.block_size}"
        
        # 位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        
        # Token嵌入 + 位置嵌入
        tok_emb = self.transformer.wte(idx)   # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)   # (1, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer块
        for block in self.transformer.h:
            x = block(x)
        
        # 最终Layer Norm
        x = self.transformer.ln_f(x)
        
        # 计算logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # 计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        """
        生成新token
        
        Args:
            idx: 初始上下文, shape (B, T)
            max_new_tokens: 要生成的最大token数
            temperature: 采样温度（越高越随机）
            top_k: Top-K采样
            top_p: Nucleus采样
        """
        for _ in range(max_new_tokens):
            # 如果上下文过长，截断到block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播获取logits
            logits, _ = self(idx_cond)
            
            # 只取最后一个位置的logits
            logits = logits[:, -1, :] / temperature
            
            # Top-K采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-P (Nucleus) 采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def count_parameters(model: nn.Module) -> dict:
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }
```

## 训练循环

```python
# train.py
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

from config import GPTConfig
from model import GPT
from dataset import get_batch, TextDataset

class Trainer:
    """GPT训练器"""
    
    def __init__(
        self,
        model: GPT,
        train_dataset: TextDataset,
        val_dataset: TextDataset,
        config: GPTConfig
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 混合精度训练
        self.scaler = GradScaler()
        self.use_amp = config.device == 'cuda'
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_iters,
            eta_min=config.learning_rate / 10
        )
        
        # 训练状态
        self.iter_num = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self):
        """创建优化器，使用权重衰减"""
        # 分离需要weight decay和不需要的参数
        decay = set()
        no_decay = set()
        
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95)
        )
    
    @torch.no_grad()
    def estimate_loss(self) -> dict:
        """估计训练和验证损失"""
        out = {}
        self.model.eval()
        
        for split, dataset in [('train', self.train_dataset), ('val', self.val_dataset)]:
            losses = torch.zeros(self.config.eval_iters)
            
            for k in range(self.config.eval_iters):
                X, Y = get_batch(dataset, self.config.batch_size, self.config.device)
                
                with autocast(enabled=self.use_amp):
                    logits, loss = self.model(X, Y)
                
                losses[k] = loss.item()
            
            out[split] = losses.mean()
        
        self.model.train()
        return out
    
    def train_step(self) -> float:
        """执行单步训练"""
        # 获取批次
        X, Y = get_batch(
            self.train_dataset,
            self.config.batch_size,
            self.config.device
        )
        
        # 前向传播
        with autocast(enabled=self.use_amp):
            logits, loss = self.model(X, Y)
        
        # 反向传播
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        self.iter_num += 1
        
        return loss.item()
    
    def train(self):
        """主训练循环"""
        print(f"Training on {self.config.device}")
        print(f"Total iterations: {self.config.max_iters}")
        
        pbar = tqdm(range(self.config.max_iters), desc="Training")
        
        for iter_num in pbar:
            # 评估
            if iter_num % self.config.eval_interval == 0 or iter_num == self.config.max_iters - 1:
                losses = self.estimate_loss()
                print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # 保存最佳模型
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint('best_model.pt')
            
            # 训练步骤
            loss = self.train_step()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        print(f"\nTraining complete. Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, f'checkpoints/{filename}')
        print(f"Saved checkpoint to checkpoints/{filename}")
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(f'checkpoints/{filename}')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from checkpoints/{filename}")
```

## 文本生成

```python
# generate.py
import torch
from typing import Optional

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    device: str = 'cuda'
) -> str:
    """
    使用训练好的模型生成文本
    
    Args:
        model: GPT模型
        tokenizer: 分词器
        prompt: 初始提示文本
        max_new_tokens: 生成的最大token数
        temperature: 采样温度
        top_k: Top-K采样参数
        top_p: Nucleus采样参数
        device: 计算设备
    
    Returns:
        生成的完整文本
    """
    model.eval()
    
    # 编码提示
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # 解码
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text


def interactive_generation(model, tokenizer, device: str = 'cuda'):
    """交互式文本生成"""
    print("=" * 50)
    print("Interactive Text Generation")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        prompt = input("\nEnter prompt: ")
        if prompt.lower() == 'quit':
            break
        
        print("\nGenerating...")
        text = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_new_tokens=200,
            temperature=0.8,
            device=device
        )
        
        print("\n" + "-" * 50)
        print(text)
        print("-" * 50)
```

## 主程序

```python
# main.py
import torch
import argparse

from config import GPTConfig
from tokenizer import CharTokenizer
from model import GPT
from dataset import prepare_data
from train import Trainer
from generate import generate_text, interactive_generation

def load_text_data(filepath: str) -> str:
    """加载文本数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters from {filepath}")
    return text

def main():
    parser = argparse.ArgumentParser(description='Train miniGPT')
    parser.add_argument('--data', type=str, default='input.txt', help='Path to training data')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'interactive'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Generation prompt')
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载数据
    text = load_text_data(args.data)
    
    # 创建分词器
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    
    # 创建配置
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        device=device
    )
    
    # 创建模型
    model = GPT(config).to(device)
    
    if args.mode == 'train':
        # 准备数据集
        train_dataset, val_dataset = prepare_data(
            text, 
            tokenizer, 
            config.block_size
        )
        
        # 创建训练器
        trainer = Trainer(model, train_dataset, val_dataset, config)
        
        # 加载检查点（如果有）
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        trainer.save_checkpoint('final_model.pt')
        tokenizer.save('checkpoints/tokenizer.json')
        
    elif args.mode == 'generate':
        # 加载检查点
        if args.checkpoint is None:
            args.checkpoint = 'best_model.pt'
        
        checkpoint = torch.load(f'checkpoints/{args.checkpoint}')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 生成文本
        prompt = args.prompt if args.prompt else text[:50]
        generated = generate_text(
            model, 
            tokenizer, 
            prompt,
            max_new_tokens=500,
            device=device
        )
        print("\nGenerated text:")
        print("=" * 50)
        print(generated)
        
    elif args.mode == 'interactive':
        # 加载检查点
        if args.checkpoint is None:
            args.checkpoint = 'best_model.pt'
        
        checkpoint = torch.load(f'checkpoints/{args.checkpoint}')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 交互式生成
        interactive_generation(model, tokenizer, device)

if __name__ == '__main__':
    main()
```

## 使用示例

### 1. 准备训练数据

下载Shakespeare数据集或使用任何文本文件：

```bash
# 下载Shakespeare数据
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 2. 训练模型

```bash
python main.py --data input.txt --mode train
```

### 3. 生成文本

```bash
python main.py --mode generate --checkpoint best_model.pt --prompt "To be or not to be"
```

### 4. 交互式生成

```bash
python main.py --mode interactive --checkpoint best_model.pt
```

## 训练技巧

### 1. 调整超参数

```python
# 小模型（快速实验）
small_config = GPTConfig(
    n_layer=4,
    n_head=4,
    n_embd=256,
    block_size=128,
    batch_size=32,
    learning_rate=1e-3,
    max_iters=2000
)

# 中等模型
medium_config = GPTConfig(
    n_layer=8,
    n_head=8,
    n_embd=512,
    block_size=256,
    batch_size=64,
    learning_rate=3e-4,
    max_iters=10000
)

# 大模型
large_config = GPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=512,
    batch_size=32,
    learning_rate=1e-4,
    max_iters=50000
)
```

### 2. 监控训练

```python
import wandb

def train_with_logging(trainer, project_name="minigpt"):
    """使用wandb监控训练"""
    wandb.init(project=project_name, config=vars(trainer.config))
    
    for iter_num in range(trainer.config.max_iters):
        loss = trainer.train_step()
        
        wandb.log({
            "train_loss": loss,
            "learning_rate": trainer.scheduler.get_last_lr()[0],
            "iteration": iter_num
        })
        
        if iter_num % trainer.config.eval_interval == 0:
            losses = trainer.estimate_loss()
            wandb.log({
                "val_loss": losses['val'],
                "train_loss_eval": losses['train']
            })
    
    wandb.finish()
```

### 3. 使用更大的数据集

```python
def load_multiple_files(file_paths):
    """加载多个文本文件"""
    all_text = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            all_text.append(f.read())
    return '\n\n'.join(all_text)

# 使用HuggingFace datasets
from datasets import load_dataset

def load_hf_dataset(name="wikitext", subset="wikitext-2-v1"):
    """加载HuggingFace数据集"""
    dataset = load_dataset(name, subset)
    text = '\n'.join(dataset['train']['text'])
    return text
```

## 总结

本文从零实现了一个完整的miniGPT，包括：

1. **字符级分词器**：简单但完整的tokenizer实现
2. **GPT模型架构**：完整的Transformer decoder实现
   - 因果自注意力
   - 前馈网络
   - 位置编码
   - 权重绑定
3. **训练循环**：包含混合精度、梯度裁剪、学习率调度
4. **文本生成**：支持temperature、top-k、top-p采样

### 关键要点

- 理解代码的每一行比复制粘贴更重要
- 从小模型开始实验，确认代码正确后再扩大规模
- 监控训练过程，及时发现问题

### 扩展方向

- 使用BPE分词器替代字符级分词
- 实现Flash Attention提高效率
- 添加LoRA进行参数高效微调
- 使用DeepSpeed进行分布式训练

下一篇文章，我们将探讨GPT的推理优化与部署技术，包括KV Cache、量化和各种推理框架的使用。
