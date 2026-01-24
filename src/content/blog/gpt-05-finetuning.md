---
title: 'GPT完全指南（五）：微调与对齐技术'
description: '深入解析GPT模型的微调技术，包括SFT监督微调、RLHF人类反馈强化学习、DPO直接偏好优化等让AI"听话"的关键技术'
pubDate: '2024-03-05'
heroImage: ''
category: '技术'
tags: ['GPT', 'RLHF', 'SFT', 'DPO', '对齐']
series: 'GPT完全指南'
seriesOrder: 5
---

预训练让GPT学会了语言，但要让它真正"有用"且"安全"，还需要微调和对齐。本文深入探讨让GPT从"学会说话"到"说正确的话"的关键技术：监督微调（SFT）、人类反馈强化学习（RLHF）和直接偏好优化（DPO）。

## 为什么需要对齐？

### 预训练模型的问题

预训练后的GPT存在几个关键问题：

1. **不遵循指令**：只会续写，不理解"任务"的概念
2. **输出不可控**：可能生成有害、偏见或错误内容
3. **格式不一致**：无法按要求的格式输出
4. **不知道何时停止**：可能无限续写

```python
# 预训练模型的典型行为
prompt = "写一首关于春天的诗："
# 预训练模型可能输出：
# "写一首关于春天的诗：这是语文课的作业要求..."  ← 续写而非执行任务

# 对齐后的模型：
# "春风拂面暖洋洋，
#  桃花盛开满山岗..."  ← 理解并执行任务
```

### 对齐的三个阶段

| 阶段 | 方法 | 目的 |
|------|------|------|
| 1 | SFT（监督微调） | 学会遵循指令格式 |
| 2 | RM（奖励模型） | 学习人类偏好 |
| 3 | RLHF/DPO | 优化策略以满足偏好 |

## 监督微调（SFT）

### 核心思想

使用高质量的`(指令, 回复)`对进行监督学习，教模型理解并执行各种任务。

### 数据格式

```python
# ChatML格式（GPT-4使用）
def format_chatml(messages):
    """
    ChatML格式化
    messages: [{"role": "system/user/assistant", "content": "..."}]
    """
    formatted = ""
    for msg in messages:
        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return formatted

# 示例数据
sft_example = {
    "messages": [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}
    ]
}

# Alpaca格式
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
```

### SFT数据集构建

```python
import json
from typing import List, Dict
import random

class SFTDatasetBuilder:
    """SFT数据集构建器"""
    
    def __init__(self):
        self.data = []
        
    def add_instruction_data(self, instruction: str, output: str, input_text: str = ""):
        """添加指令数据"""
        self.data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    def add_conversation(self, messages: List[Dict[str, str]]):
        """添加多轮对话"""
        self.data.append({"messages": messages})
    
    def balance_categories(self, category_field: str = "category"):
        """平衡各类别数据"""
        categories = {}
        for item in self.data:
            cat = item.get(category_field, "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)
        
        # 找到最小类别的数量
        min_count = min(len(v) for v in categories.values())
        
        # 对每个类别进行下采样
        balanced = []
        for cat, items in categories.items():
            balanced.extend(random.sample(items, min(len(items), min_count)))
        
        self.data = balanced
        return self
    
    def quality_filter(self, min_output_length: int = 50):
        """质量过滤"""
        filtered = []
        for item in self.data:
            output = item.get("output", "")
            if "messages" in item:
                # 对话格式，检查助手回复
                assistant_msgs = [m for m in item["messages"] if m["role"] == "assistant"]
                if assistant_msgs and len(assistant_msgs[-1]["content"]) >= min_output_length:
                    filtered.append(item)
            elif len(output) >= min_output_length:
                filtered.append(item)
        
        self.data = filtered
        return self
    
    def save(self, path: str):
        """保存数据集"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """加载数据集"""
        builder = cls()
        with open(path, 'r', encoding='utf-8') as f:
            builder.data = json.load(f)
        return builder
```

### SFT训练实现

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class SFTDataset(Dataset):
    """SFT数据集"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 格式化输入
        if "messages" in item:
            text = self._format_conversation(item["messages"])
        else:
            text = self._format_instruction(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels: 只计算回复部分的损失
        labels = self._create_labels(input_ids, item)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _format_conversation(self, messages):
        """格式化对话"""
        text = ""
        for msg in messages:
            if msg["role"] == "system":
                text += f"<|system|>\n{msg['content']}</s>\n"
            elif msg["role"] == "user":
                text += f"<|user|>\n{msg['content']}</s>\n"
            elif msg["role"] == "assistant":
                text += f"<|assistant|>\n{msg['content']}</s>\n"
        return text
    
    def _format_instruction(self, item):
        """格式化指令"""
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        
        if input_text:
            return f"<|user|>\n{instruction}\n\n{input_text}</s>\n<|assistant|>\n{output}</s>"
        else:
            return f"<|user|>\n{instruction}</s>\n<|assistant|>\n{output}</s>"
    
    def _create_labels(self, input_ids, item):
        """创建标签，只在回复部分计算损失"""
        labels = input_ids.clone()
        
        # 找到assistant标记的位置
        # 在assistant标记之前的部分设为-100（忽略）
        assistant_token_id = self.tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]
        
        try:
            assistant_pos = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0][0]
            labels[:assistant_pos + 1] = -100
        except:
            pass  # 如果找不到，整个序列都计算损失
        
        return labels


class SFTTrainer:
    """SFT训练器"""
    
    def __init__(
        self,
        model_name: str,
        train_data: List[Dict],
        val_data: List[Dict] = None,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 使用LoRA进行参数高效微调
        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # 数据集
        self.train_dataset = SFTDataset(train_data, self.tokenizer)
        self.val_dataset = SFTDataset(val_data, self.tokenizer) if val_data else None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
    
    def train(self):
        """训练循环"""
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % 100 == 0:
                        avg_loss = total_loss / 100
                        print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}")
                        total_loss = 0
            
            # 验证
            if self.val_dataset:
                val_loss = self.evaluate()
                print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(f"sft_epoch_{epoch}")
    
    def evaluate(self):
        """评估"""
        self.model.eval()
        total_loss = 0
        
        val_loader = DataLoader(self.val_dataset, batch_size=4)
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        self.model.save_pretrained(f"checkpoints/{name}")
        self.tokenizer.save_pretrained(f"checkpoints/{name}")
```

## 奖励模型（Reward Model）

### 核心思想

训练一个模型来预测人类对回复的偏好打分，用于后续的强化学习。

### 偏好数据格式

```python
# 偏好数据示例
preference_data = {
    "prompt": "解释什么是黑洞",
    "chosen": "黑洞是时空中引力极强的区域...",  # 人类偏好的回复
    "rejected": "黑洞就是一个很黑的洞..."       # 人类不偏好的回复
}
```

### 奖励模型实现

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    奖励模型
    基于预训练语言模型，添加一个标量输出头
    """
    
    def __init__(self, base_model_name: str):
        super().__init__()
        
        # 基础模型（不需要LM头）
        self.backbone = AutoModel.from_pretrained(base_model_name)
        
        # 奖励头：将隐藏状态映射到标量奖励
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用最后一个token的隐藏状态
        last_hidden = outputs.last_hidden_state
        
        # 找到每个序列的最后一个非padding位置
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            last_token_hidden = last_hidden[batch_indices, sequence_lengths]
        else:
            last_token_hidden = last_hidden[:, -1]
        
        # 计算奖励
        reward = self.reward_head(last_token_hidden).squeeze(-1)
        
        return reward


class RewardModelTrainer:
    """奖励模型训练器"""
    
    def __init__(
        self,
        model_name: str,
        train_data: List[Dict],
        learning_rate: float = 1e-5,
        batch_size: int = 4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = RewardModel(model_name).to(self.device)
        
        self.train_data = train_data
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
    
    def compute_loss(self, chosen_rewards, rejected_rewards):
        """
        Bradley-Terry损失
        最大化 P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        """
        return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    
    def train_step(self, batch):
        """单步训练"""
        prompts = batch["prompts"]
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        
        # 编码chosen回复
        chosen_texts = [p + c for p, c in zip(prompts, chosen)]
        chosen_encoding = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        # 编码rejected回复
        rejected_texts = [p + r for p, r in zip(prompts, rejected)]
        rejected_encoding = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)
        
        # 计算奖励
        chosen_rewards = self.model(
            chosen_encoding["input_ids"],
            chosen_encoding["attention_mask"]
        )
        rejected_rewards = self.model(
            rejected_encoding["input_ids"],
            rejected_encoding["attention_mask"]
        )
        
        # 计算损失
        loss = self.compute_loss(chosen_rewards, rejected_rewards)
        
        # 计算准确率
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, accuracy
    
    def train(self, num_epochs: int = 3):
        """训练循环"""
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            num_batches = 0
            
            # 简单的批处理
            for i in range(0, len(self.train_data), self.batch_size):
                batch_data = self.train_data[i:i+self.batch_size]
                batch = {
                    "prompts": [d["prompt"] for d in batch_data],
                    "chosen": [d["chosen"] for d in batch_data],
                    "rejected": [d["rejected"] for d in batch_data]
                }
                
                loss, accuracy = self.train_step(batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_acc += accuracy.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
```

## RLHF（人类反馈强化学习）

### PPO算法原理

RLHF使用PPO（Proximal Policy Optimization）算法优化策略：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是策略比率
- $\hat{A}_t$ 是优势估计
- $\epsilon$ 是裁剪参数

### RLHF实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class RLHFConfig:
    """RLHF配置"""
    # PPO参数
    clip_epsilon: float = 0.2
    value_clip: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    
    # KL散度惩罚
    kl_coef: float = 0.1
    target_kl: float = 0.02
    
    # 训练参数
    batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    learning_rate: float = 1e-6


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        tokenizer,
        config: RLHFConfig
    ):
        self.policy = policy_model
        self.ref_model = ref_model  # 冻结的参考模型，用于KL散度
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        self.device = next(policy_model.parameters()).device
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # 值函数头（可以共享backbone或单独训练）
        self.value_head = nn.Linear(
            policy_model.config.hidden_size, 1
        ).to(self.device)
    
    def generate_responses(self, prompts: List[str], max_length: int = 256):
        """生成回复"""
        self.policy.eval()
        
        responses = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                outputs = self.policy.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].size(1):],
                    skip_special_tokens=True
                )
                responses.append(response)
        
        self.policy.train()
        return responses
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """使用奖励模型计算奖励"""
        texts = [p + r for p, r in zip(prompts, responses)]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            rewards = self.reward_model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
        
        return rewards
    
    def compute_log_probs(self, model, input_ids, attention_mask, response_mask):
        """计算每个token的log概率"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits[:, :-1]  # 预测下一个token
        labels = input_ids[:, 1:]        # 目标token
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取每个位置对应label的log prob
        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 只保留response部分
        token_log_probs = token_log_probs * response_mask[:, 1:]
        
        return token_log_probs
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势估计"""
        advantages = torch.zeros_like(values)
        lastgaelam = 0
        
        for t in reversed(range(values.size(1))):
            if t == values.size(1) - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            # TD误差
            delta = rewards + self.config.gamma * next_value - values[:, t]
            
            # GAE
            advantages[:, t] = lastgaelam = (
                delta + self.config.gamma * self.config.lam * lastgaelam
            )
        
        # 计算returns
        returns = advantages + values
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ):
        """单步PPO更新"""
        # 当前策略的log probs
        new_log_probs = self.compute_log_probs(
            self.policy,
            input_ids,
            attention_mask,
            response_mask
        )
        
        # 参考模型的log probs（用于KL惩罚）
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.ref_model,
                input_ids,
                attention_mask,
                response_mask
            )
        
        # 策略比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL散度惩罚
        kl_div = (old_log_probs - new_log_probs).mean()
        
        # 总损失
        loss = policy_loss + self.config.kl_coef * kl_div
        
        return loss, {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": ratio.mean().item()
        }
    
    def train_step(self, prompts: List[str]):
        """RLHF训练步骤"""
        # 1. 生成回复
        responses = self.generate_responses(prompts)
        
        # 2. 计算奖励
        rewards = self.compute_rewards(prompts, responses)
        
        # 3. 准备训练数据
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # 创建response mask
        prompt_lengths = [
            len(self.tokenizer.encode(p, add_special_tokens=False))
            for p in prompts
        ]
        response_mask = torch.zeros_like(inputs["input_ids"])
        for i, pl in enumerate(prompt_lengths):
            response_mask[i, pl:] = 1
        
        # 4. 计算old log probs
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(
                self.policy,
                inputs["input_ids"],
                inputs["attention_mask"],
                response_mask
            )
            
            # 值函数估计
            outputs = self.policy(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            values = self.value_head(outputs.hidden_states[-1]).squeeze(-1)
        
        # 5. 计算优势
        advantages, returns = self.compute_advantages(
            rewards.unsqueeze(1).expand_as(values),
            values,
            response_mask
        )
        
        # 6. PPO更新
        total_loss = 0
        for _ in range(self.config.ppo_epochs):
            loss, stats = self.ppo_step(
                inputs["input_ids"],
                inputs["attention_mask"],
                response_mask,
                old_log_probs,
                advantages,
                returns
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            "loss": total_loss / self.config.ppo_epochs,
            "reward_mean": rewards.mean().item(),
            **stats
        }
```

## DPO（直接偏好优化）

### DPO的优势

DPO直接从偏好数据学习，无需单独训练奖励模型和使用强化学习：

| 方法 | 步骤 | 复杂度 |
|------|------|--------|
| RLHF | SFT → RM → PPO | 高，需要多个模型 |
| DPO | SFT → DPO | 低，单一训练循环 |

### DPO损失函数

$$L_{DPO}(\theta) = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中：
- $y_w$ 是chosen回复
- $y_l$ 是rejected回复
- $\pi_{ref}$ 是参考策略（SFT模型）
- $\beta$ 是温度参数

### DPO实现

```python
import torch
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DPOConfig:
    """DPO配置"""
    beta: float = 0.1
    learning_rate: float = 5e-7
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024


class DPOTrainer:
    """DPO训练器"""
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: DPOConfig
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        
        self.device = next(model.parameters()).device
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
    
    def get_batch_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算序列的log概率"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits[:, :-1]
        labels = labels[:, 1:]
        
        # 计算每个token的log prob
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取label对应的log prob
        per_token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 创建mask，只保留response部分
        label_mask = (labels != -100).float()
        
        # 序列的总log prob
        return (per_token_log_probs * label_mask).sum(dim=-1)
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> torch.Tensor:
        """计算DPO损失"""
        # 计算log ratio
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO损失
        logits = self.config.beta * (chosen_log_ratios - rejected_log_ratios)
        loss = -F.logsigmoid(logits).mean()
        
        # 计算一些有用的指标
        chosen_rewards = self.config.beta * chosen_log_ratios.detach()
        rejected_rewards = self.config.beta * rejected_log_ratios.detach()
        
        return loss, {
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean().item()
        }
    
    def prepare_batch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """准备训练批次"""
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        
        # 编码chosen
        chosen_texts = [p + c for p, c in zip(prompts, chosen)]
        chosen_encoding = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # 创建chosen labels（只在response部分计算损失）
        chosen_labels = chosen_encoding["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            chosen_labels[i, :prompt_len] = -100
        
        # 编码rejected
        rejected_texts = [p + r for p, r in zip(prompts, rejected)]
        rejected_encoding = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # 创建rejected labels
        rejected_labels = rejected_encoding["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            rejected_labels[i, :prompt_len] = -100
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].to(self.device),
            "chosen_attention_mask": chosen_encoding["attention_mask"].to(self.device),
            "chosen_labels": chosen_labels.to(self.device),
            "rejected_input_ids": rejected_encoding["input_ids"].to(self.device),
            "rejected_attention_mask": rejected_encoding["attention_mask"].to(self.device),
            "rejected_labels": rejected_labels.to(self.device),
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        # 策略模型的log probs
        policy_chosen_logps = self.get_batch_logps(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"]
        )
        
        policy_rejected_logps = self.get_batch_logps(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"]
        )
        
        # 参考模型的log probs
        with torch.no_grad():
            ref_chosen_logps = self.get_batch_logps(
                self.ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"]
            )
            
            ref_rejected_logps = self.get_batch_logps(
                self.ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"]
            )
        
        # 计算DPO损失
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        return loss, metrics
    
    def train(self, train_data: List[Dict], num_epochs: int = 1):
        """训练循环"""
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_metrics = {}
            num_steps = 0
            
            # 简单的批处理
            for i in range(0, len(train_data), self.config.batch_size):
                batch_data = train_data[i:i+self.config.batch_size]
                batch = self.prepare_batch(batch_data)
                
                loss, metrics = self.train_step(batch)
                
                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (num_steps + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                
                num_steps += 1
            
            # 打印epoch统计
            avg_loss = total_loss / num_steps
            avg_metrics = {k: v / num_steps for k, v in total_metrics.items()}
            
            print(f"Epoch {epoch}:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
            print(f"  Reward Margin: {avg_metrics['reward_margin']:.4f}")
```

## 实践建议

### 1. 数据质量至关重要

```python
def validate_preference_data(data: List[Dict]) -> List[Dict]:
    """验证偏好数据质量"""
    valid_data = []
    
    for item in data:
        # 检查必需字段
        if not all(k in item for k in ["prompt", "chosen", "rejected"]):
            continue
        
        # 确保chosen和rejected不同
        if item["chosen"].strip() == item["rejected"].strip():
            continue
        
        # 检查长度
        if len(item["chosen"]) < 10 or len(item["rejected"]) < 10:
            continue
        
        valid_data.append(item)
    
    print(f"Valid data: {len(valid_data)}/{len(data)}")
    return valid_data
```

### 2. 超参数选择

| 参数 | SFT | DPO | 说明 |
|------|-----|-----|------|
| Learning Rate | 2e-5 | 5e-7 | DPO需要更小的学习率 |
| Batch Size | 32-128 | 4-16 | DPO需要更小的batch |
| Epochs | 3-5 | 1-3 | DPO容易过拟合 |
| β (DPO) | - | 0.1-0.5 | 控制偏离参考模型的程度 |

### 3. 评估指标

```python
def evaluate_alignment(model, tokenizer, test_prompts: List[str]):
    """评估对齐效果"""
    results = {
        "helpfulness": [],
        "harmlessness": [],
        "honesty": []
    }
    
    for prompt in test_prompts:
        response = generate(model, tokenizer, prompt)
        
        # 使用GPT-4或人工评估
        # 这里简化为示意
        results["helpfulness"].append(evaluate_helpful(response))
        results["harmlessness"].append(evaluate_safe(response))
        results["honesty"].append(evaluate_honest(response))
    
    return {k: sum(v)/len(v) for k, v in results.items()}
```

## 总结

本文详细介绍了GPT对齐的核心技术：

1. **SFT（监督微调）**：用高质量数据教模型遵循指令
2. **奖励模型**：学习人类偏好的打分系统
3. **RLHF**：使用PPO强化学习优化策略
4. **DPO**：更简单的直接偏好优化方法

### 关键要点

- 对齐是让模型"有用且安全"的关键
- 数据质量比数量更重要
- DPO比RLHF更简单，效果相当
- 选择合适的超参数避免过拟合

下一篇文章，我们将从零实现一个完整的miniGPT，包括模型架构、训练和推理的完整代码。
