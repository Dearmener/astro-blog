---
title: '深度学习完全指南（十二）：模型训练与优化技巧'
description: '从优化器选择到学习率调度、正则化技术，全面掌握深度学习训练的核心技巧'
pubDate: '2024-02-12'
heroImage: ''
category: '技术'
tags: ['深度学习', '优化器', '学习率', '正则化']
series: '深度学习完全指南'
seriesOrder: 12
---

## 训练流程概述

深度学习训练是一个迭代优化过程，涉及数据准备、模型构建、损失计算、梯度回传和参数更新。

### 标准训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy
```

---

## 优化器详解

### SGD系列

```python
# 基础SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 带动量的SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 带Nesterov动量
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# 带权重衰减（L2正则化）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### Adam系列

```python
# Adam - 自适应学习率
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# AdamW - 解耦权重衰减
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Amsgrad - 修正Adam的收敛问题
optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
```

### 其他优化器

```python
# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# Adadelta
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
```

### 自定义优化器：LAMB

```python
class LAMB(optim.Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # 动量更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adam更新
                adam_update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])
                
                # 权重衰减
                if group['weight_decay'] != 0:
                    adam_update.add_(p, alpha=group['weight_decay'])
                
                # Layer-wise学习率适应
                weight_norm = p.norm()
                update_norm = adam_update.norm()
                
                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = weight_norm / update_norm
                else:
                    trust_ratio = 1.0
                
                p.add_(adam_update, alpha=-group['lr'] * trust_ratio)
```

### 优化器选择指南

| 优化器 | 适用场景 | 推荐学习率 |
|--------|---------|-----------|
| SGD+Momentum | CV任务，需要精调 | 0.01-0.1 |
| Adam | NLP、小数据集 | 1e-4 ~ 1e-3 |
| AdamW | Transformer、BERT | 1e-5 ~ 5e-4 |
| LAMB | 大批量训练 | 1e-3 ~ 1e-2 |

---

## 学习率调度

### 常用调度器

```python
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, 
    CosineAnnealingLR, ReduceLROnPlateau,
    OneCycleLR, CosineAnnealingWarmRestarts
)

# Step衰减
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 多步衰减
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# 指数衰减
scheduler = ExponentialLR(optimizer, gamma=0.95)

# 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 余弦退火带重启
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 自适应衰减
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# One Cycle
scheduler = OneCycleLR(
    optimizer, 
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # warmup占比
    anneal_strategy='cos'
)
```

### Warmup + Cosine Decay

```python
import math

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 线性warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # 余弦衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# 使用
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=100)

for epoch in range(100):
    train_epoch(model, train_loader, criterion, optimizer, device)
    scheduler.step()
    print(f"Epoch {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
```

### 层级学习率

```python
# 不同层使用不同学习率（常用于预训练模型微调）
param_groups = [
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # 预训练层，小学习率
    {'params': model.classifier.parameters(), 'lr': 1e-3}  # 新层，大学习率
]
optimizer = optim.AdamW(param_groups)

# BERT微调常用设置
def get_bert_param_groups(model, lr=2e-5, weight_decay=0.01):
    no_decay = ['bias', 'LayerNorm.weight']
    
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    return param_groups
```

---

## 正则化技术

### L2正则化（权重衰减）

```python
# 方式1：优化器参数
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 方式2：手动添加到损失
def l2_regularization(model, lambda_l2=1e-4):
    l2_reg = torch.tensor(0.)
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    return lambda_l2 * l2_reg

loss = criterion(output, target) + l2_regularization(model)
```

### L1正则化（稀疏化）

```python
def l1_regularization(model, lambda_l1=1e-5):
    l1_reg = torch.tensor(0.)
    for param in model.parameters():
        l1_reg += torch.norm(param, 1)
    return lambda_l1 * l1_reg
```

### Dropout

```python
class ModelWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 训练时随机丢弃，测试时自动缩放
        x = self.fc2(x)
        return x

# 变体
# Dropout2d - 针对卷积特征图，丢弃整个通道
dropout2d = nn.Dropout2d(p=0.2)

# AlphaDropout - 用于SELU激活
alpha_dropout = nn.AlphaDropout(p=0.1)

# DropPath - 用于残差连接
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor
```

### Label Smoothing

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # KL散度
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=-1), dim=-1))

# 使用
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

### Mixup

```python
def mixup_data(x, y, alpha=0.2):
    """
    Mixup数据增强
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 训练循环中使用
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    
    # Mixup
    data, target_a, target_b, lam = mixup_data(data, target, alpha=0.2)
    
    output = model(data)
    loss = mixup_criterion(criterion, output, target_a, target_b, lam)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### CutMix

```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)
    
    # 计算裁剪区域
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    # 混合
    x_clone = x.clone()
    x_clone[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # 调整lambda
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    return x_clone, y, y[index], lam
```

---

## 批归一化与变体

### Batch Normalization

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 运行时统计
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # 计算batch统计量
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # 更新运行统计
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.gamma * x_norm + self.beta
```

### Layer Normalization

```python
# LayerNorm - 跨特征维度归一化，适合NLP
layer_norm = nn.LayerNorm(hidden_size)

# 手动实现
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### Group Normalization

```python
# GroupNorm - 分组归一化，小batch表现好
group_norm = nn.GroupNorm(num_groups=32, num_channels=256)

# 手动实现
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    
    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        
        x = x.view(N, G, C // G, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        
        return self.gamma * x + self.beta
```

### RMSNorm

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (用于LLaMA等)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## 梯度技巧

### 梯度裁剪

```python
# 按范数裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# 自适应梯度裁剪 (AGC)
def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3):
    for p in parameters:
        if p.grad is None:
            continue
        
        param_norm = p.norm()
        grad_norm = p.grad.norm()
        
        max_norm = param_norm * clip_factor
        
        if grad_norm > max_norm:
            p.grad.mul_(max_norm / (grad_norm + eps))
```

### 梯度累积

```python
# 小显存训练大batch
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data.to(device))
    loss = criterion(output, target.to(device))
    loss = loss / accumulation_steps  # 缩放损失
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 梯度检查点

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1024, 1024) for _ in range(100)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # 使用梯度检查点减少显存
            x = checkpoint(layer, x)
        return x
```

---

## 混合精度训练

### 使用AMP

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 缩放损失并反向传播
    scaler.scale(loss).backward()
    
    # 取消缩放并更新
    scaler.step(optimizer)
    scaler.update()
```

### BF16训练

```python
# BFloat16 (更大动态范围)
model = model.to(torch.bfloat16)

with autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
```

---

## 分布式训练

### DataParallel

```python
# 单机多卡 - 简单但效率较低
model = nn.DataParallel(model)
model = model.to(device)
```

### DistributedDataParallel

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # 模型
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 数据
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 确保每个epoch数据不同
        
        for data, target in dataloader:
            # 训练步骤...
            pass
    
    cleanup()

# 启动
import torch.multiprocessing as mp
mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

### FSDP（Fully Sharded Data Parallel）

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
)
```

---

## 早停与模型选择

### 早停

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

# 使用
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss, val_acc = evaluate(...)
    
    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 模型检查点

```python
class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, model, optimizer, epoch, score):
        if self.save_best_only:
            if (self.mode == 'min' and score < self.best_score) or \
               (self.mode == 'max' and score > self.best_score):
                self.best_score = score
                self._save(model, optimizer, epoch)
        else:
            self._save(model, optimizer, epoch)
    
    def _save(self, model, optimizer, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': self.best_score
        }, self.filepath)

# 加载检查点
checkpoint = torch.load(filepath)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 超参数搜索

### 网格搜索

```python
from itertools import product

def grid_search(train_fn, param_grid):
    best_score = float('-inf')
    best_params = None
    
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combination in product(*values):
        params = dict(zip(keys, combination))
        print(f"Testing: {params}")
        
        score = train_fn(**params)
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score

# 使用
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'weight_decay': [1e-4, 1e-3],
    'dropout': [0.1, 0.3, 0.5]
}
best_params, best_score = grid_search(train_and_evaluate, param_grid)
```

### Optuna

```python
import optuna

def objective(trial):
    # 建议超参数
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    
    # 构建模型
    model = MyModel(hidden_dim=hidden_dim, dropout=dropout)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 报告中间结果
        trial.report(val_acc, epoch)
        
        # 剪枝不好的试验
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_acc

# 创建study并优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best params: {study.best_params}")
print(f"Best accuracy: {study.best_value}")
```

---

## 调试技巧

### 过拟合单个batch

```python
def overfit_single_batch(model, dataloader, criterion, optimizer, device, epochs=100):
    """快速验证模型能否学习"""
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    
    model.train()
    for epoch in range(epochs):
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            acc = (output.argmax(1) == target).float().mean()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
```

### 梯度检查

```python
def check_gradients(model):
    """检查梯度是否正常"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 100:
                print(f"Warning: Large gradient in {name}: {grad_norm}")
            elif grad_norm < 1e-7:
                print(f"Warning: Vanishing gradient in {name}: {grad_norm}")
```

### 学习率范围测试

```python
def lr_range_test(model, train_loader, criterion, optimizer, device, 
                  start_lr=1e-7, end_lr=10, num_iter=100):
    """找到最优学习率范围"""
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    
    losses = []
    lrs = []
    
    for i, (data, target) in enumerate(train_loader):
        if i >= num_iter:
            break
        
        # 设置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        lrs.append(lr)
        
        lr *= lr_mult
    
    # 绘图
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()
    
    return lrs, losses
```

---

## 训练最佳实践清单

| 阶段 | 检查项 |
|------|--------|
| **数据** | 数据归一化、数据增强、类别平衡 |
| **模型** | 权重初始化、适当的归一化层、残差连接 |
| **优化器** | AdamW/SGD选择、权重衰减、动量 |
| **学习率** | Warmup、余弦衰减、适当的初始值 |
| **正则化** | Dropout、权重衰减、数据增强 |
| **监控** | 训练/验证曲线、梯度统计、学习率 |
| **调试** | 过拟合单batch、梯度检查、学习率测试 |

---

## 小结

| 技术 | 推荐配置 |
|------|---------|
| 优化器 | AdamW (NLP), SGD+Momentum (CV) |
| 学习率 | Warmup + Cosine Decay |
| 正则化 | Dropout(0.1-0.3) + Weight Decay(0.01) |
| 归一化 | LayerNorm (Transformer), BatchNorm (CNN) |
| 混合精度 | FP16/BF16 + GradScaler |
| 分布式 | DDP > DataParallel |

**下一篇**：深度学习框架对比与实战，PyTorch vs TensorFlow vs JAX。
