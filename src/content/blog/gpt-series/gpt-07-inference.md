---
title: 'GPT完全指南（七）：推理优化与部署'
description: '深入解析GPT模型的推理优化技术，包括KV Cache、量化、Flash Attention、Speculative Decoding以及vLLM、TensorRT-LLM等推理框架'
pubDate: '2024-03-07'
heroImage: ''
category: '技术'
tags: ['GPT', '推理优化', '量化', 'vLLM', '部署']
series: 'GPT完全指南'
seriesOrder: 7
---

训练好的GPT模型要投入实际应用，推理效率至关重要。本文深入探讨GPT推理的各种优化技术，从底层的KV Cache到高级的Speculative Decoding，以及如何使用vLLM、TensorRT-LLM等框架进行高效部署。

## 推理性能瓶颈分析

### 为什么GPT推理慢？

GPT推理的主要瓶颈：

1. **内存带宽受限**：模型参数需要从HBM加载到计算单元
2. **自回归生成**：每次只能生成一个token
3. **注意力计算**：复杂度随序列长度平方增长
4. **批处理困难**：不同请求的序列长度不同

```python
# 推理时间估算
def estimate_inference_time(
    model_params: float,      # 参数量（十亿）
    batch_size: int,
    seq_length: int,
    new_tokens: int,
    memory_bandwidth: float,  # GB/s
    flops: float              # TFLOPS
):
    """
    估算推理时间
    
    两个阶段：
    1. Prefill: 处理输入（计算密集）
    2. Decode: 生成输出（内存密集）
    """
    param_bytes = model_params * 1e9 * 2  # FP16 = 2 bytes
    
    # Prefill阶段：计算密集
    prefill_flops = 2 * model_params * 1e9 * batch_size * seq_length
    prefill_time = prefill_flops / (flops * 1e12)  # seconds
    
    # Decode阶段：内存密集，每个token都要加载全部参数
    decode_time_per_token = param_bytes / (memory_bandwidth * 1e9)
    decode_time = decode_time_per_token * new_tokens
    
    total_time = prefill_time + decode_time
    throughput = new_tokens / total_time  # tokens/s
    
    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "throughput": throughput
    }

# 示例：7B模型在A100上
result = estimate_inference_time(
    model_params=7,
    batch_size=1,
    seq_length=512,
    new_tokens=256,
    memory_bandwidth=2000,  # A100 HBM带宽约2TB/s
    flops=312               # A100 FP16约312 TFLOPS
)
print(f"Throughput: {result['throughput']:.1f} tokens/s")
```

## KV Cache

### 原理

在自回归生成中，每个新token都需要attend到之前所有token。如果不缓存，同样的K和V会被重复计算。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttentionWithCache(nn.Module):
    """带KV Cache的因果自注意力"""
    
    def __init__(self, n_embd, n_head, max_seq_len=2048):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # 预分配KV cache空间
        self.register_buffer(
            'k_cache',
            torch.zeros(1, n_head, max_seq_len, self.head_dim)
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(1, n_head, max_seq_len, self.head_dim)
        )
        self.cache_len = 0
    
    def forward(self, x, use_cache=False):
        B, T, C = x.size()
        
        # 计算Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头格式
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if use_cache:
            # 更新cache
            self.k_cache[:, :, self.cache_len:self.cache_len+T, :] = k
            self.v_cache[:, :, self.cache_len:self.cache_len+T, :] = v
            
            # 使用完整的K, V
            k = self.k_cache[:, :, :self.cache_len+T, :]
            v = self.v_cache[:, :, :self.cache_len+T, :]
            
            self.cache_len += T
        
        # 注意力计算
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(T, k.size(-2), device=x.device, dtype=torch.bool),
            diagonal=k.size(-2) - T + 1
        )
        att = att.masked_fill(causal_mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        # 重组
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    
    def reset_cache(self):
        """重置KV cache"""
        self.cache_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


class GPTWithKVCache(nn.Module):
    """使用KV Cache的GPT推理"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ... 模型定义 ...
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100):
        """使用KV Cache的高效生成"""
        # 重置所有层的cache
        for block in self.transformer.h:
            block.attn.reset_cache()
        
        # Prefill: 处理整个输入
        logits = self.forward(input_ids, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        generated = [next_token]
        
        # Decode: 逐token生成
        for _ in range(max_new_tokens - 1):
            # 只需要处理新token
            logits = self.forward(next_token, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            
            # 检查是否生成EOS
            if next_token.item() == self.config.eos_token_id:
                break
        
        return torch.cat([input_ids] + generated, dim=1)
```

### KV Cache内存计算

```python
def calculate_kv_cache_size(
    batch_size: int,
    seq_length: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # FP16
) -> dict:
    """计算KV Cache内存占用"""
    
    # 每层：2 * batch * seq * heads * head_dim * dtype_bytes
    # 2是因为K和V各一份
    per_layer = 2 * batch_size * seq_length * n_heads * head_dim * dtype_bytes
    total = per_layer * n_layers
    
    return {
        "per_layer_mb": per_layer / 1024 / 1024,
        "total_mb": total / 1024 / 1024,
        "total_gb": total / 1024 / 1024 / 1024
    }

# 示例：LLaMA-7B
cache_size = calculate_kv_cache_size(
    batch_size=1,
    seq_length=4096,
    n_layers=32,
    n_heads=32,
    head_dim=128,
    dtype_bytes=2
)
print(f"KV Cache size: {cache_size['total_gb']:.2f} GB")
```

## Flash Attention

### 原理

Flash Attention通过分块计算和重计算策略，减少HBM访问次数：

```python
import torch
import torch.nn.functional as F

def flash_attention_naive(Q, K, V, block_size=64):
    """
    Flash Attention的简化实现
    实际使用应该用flash_attn库
    """
    B, H, N, D = Q.shape
    
    # 输出和用于online softmax的变量
    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, device=Q.device)  # log-sum-exp
    
    # 分块处理
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        
        # 当前Q块
        Qi = Q[:, :, i:i_end, :]
        Oi = torch.zeros_like(Qi)
        Li = torch.full((B, H, i_end - i), float('-inf'), device=Q.device)
        
        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            
            # 当前K, V块
            Kj = K[:, :, j:j_end, :]
            Vj = V[:, :, j:j_end, :]
            
            # 计算注意力分数
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / (D ** 0.5)
            
            # 应用因果掩码
            if j > i:
                # 完全mask掉
                Sij = torch.full_like(Sij, float('-inf'))
            elif j + block_size > i:
                # 部分mask
                mask = torch.triu(
                    torch.ones(i_end - i, j_end - j, device=Q.device, dtype=torch.bool),
                    diagonal=j - i + 1
                )
                Sij = Sij.masked_fill(mask, float('-inf'))
            
            # Online softmax更新
            mi_new = torch.maximum(Li.unsqueeze(-1), Sij.max(dim=-1, keepdim=True).values)
            
            P = torch.exp(Sij - mi_new)
            
            # 更新输出
            Oi = Oi * torch.exp(Li.unsqueeze(-1) - mi_new) + torch.matmul(P, Vj)
            
            # 更新log-sum-exp
            Li = mi_new.squeeze(-1) + torch.log(
                torch.exp(Li - mi_new.squeeze(-1)) + P.sum(dim=-1)
            )
        
        # 归一化
        O[:, :, i:i_end, :] = Oi / torch.exp(Li).unsqueeze(-1)
    
    return O


# 使用flash_attn库（推荐）
try:
    from flash_attn import flash_attn_func
    
    def efficient_attention(Q, K, V, causal=True):
        """使用Flash Attention 2"""
        # Q, K, V: (batch, seqlen, nheads, headdim)
        return flash_attn_func(Q, K, V, causal=causal)
except ImportError:
    print("flash_attn not installed. Using PyTorch native attention.")
    
    def efficient_attention(Q, K, V, causal=True):
        # PyTorch 2.0+ 的scaled_dot_product_attention
        return F.scaled_dot_product_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            is_causal=causal
        ).transpose(1, 2)
```

## 模型量化

### INT8量化

```python
import torch
import torch.nn as nn

class Int8Linear(nn.Module):
    """INT8量化的线性层"""
    
    def __init__(self, weight, scale, zero_point=None):
        super().__init__()
        # 量化后的权重（INT8）
        self.register_buffer('weight_int8', weight.to(torch.int8))
        # 量化参数
        self.register_buffer('scale', scale)
        if zero_point is not None:
            self.register_buffer('zero_point', zero_point)
        else:
            self.zero_point = None
    
    def forward(self, x):
        # 反量化权重
        if self.zero_point is not None:
            weight = (self.weight_int8.float() - self.zero_point) * self.scale
        else:
            weight = self.weight_int8.float() * self.scale
        
        return F.linear(x, weight)


def quantize_linear_layer(layer: nn.Linear, scheme='symmetric'):
    """量化线性层"""
    weight = layer.weight.data
    
    if scheme == 'symmetric':
        # 对称量化
        scale = weight.abs().max() / 127
        weight_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
        return Int8Linear(weight_int8, scale)
    
    else:  # asymmetric
        # 非对称量化
        w_min, w_max = weight.min(), weight.max()
        scale = (w_max - w_min) / 255
        zero_point = (-w_min / scale).round()
        weight_int8 = ((weight / scale) + zero_point).round().clamp(0, 255).to(torch.int8)
        return Int8Linear(weight_int8, scale, zero_point)


def quantize_model(model, layers_to_quantize=['c_fc', 'c_proj', 'c_attn']):
    """量化整个模型"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(l in name for l in layers_to_quantize):
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                
                quantized_layer = quantize_linear_layer(module)
                setattr(parent, layer_name, quantized_layer)
                print(f"Quantized: {name}")
    
    return model
```

### GPTQ量化

```python
# 使用auto-gptq库进行GPTQ量化
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def quantize_with_gptq(model_name, output_dir, calibration_data):
    """使用GPTQ进行4-bit量化"""
    
    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=4,                    # 4-bit量化
        group_size=128,           # 分组大小
        desc_act=True,            # 激活值感知
        damp_percent=0.1,
    )
    
    # 加载模型并量化
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config
    )
    
    # 使用校准数据进行量化
    model.quantize(calibration_data)
    
    # 保存量化模型
    model.save_quantized(output_dir)
    
    return model


# 使用量化模型
def load_gptq_model(model_path):
    """加载GPTQ量化模型"""
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device="cuda:0",
        use_triton=True  # 使用Triton加速
    )
    return model
```

### AWQ量化

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_with_awq(model_name, output_dir):
    """使用AWQ进行4-bit量化"""
    
    # 加载模型
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 量化配置
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4
    }
    
    # 执行量化
    model.quantize(
        tokenizer,
        quant_config=quant_config
    )
    
    # 保存
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model
```

## Speculative Decoding

### 原理

使用小模型快速生成草稿，大模型验证并接受/拒绝：

```python
import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    """投机解码器"""
    
    def __init__(self, target_model, draft_model, tokenizer, gamma=4):
        """
        Args:
            target_model: 目标大模型
            draft_model: 草稿小模型
            gamma: 每次投机生成的token数
        """
        self.target = target_model
        self.draft = draft_model
        self.tokenizer = tokenizer
        self.gamma = gamma
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
        """投机解码生成"""
        
        generated_tokens = []
        current_input = input_ids
        
        while len(generated_tokens) < max_new_tokens:
            # 1. 使用草稿模型生成gamma个候选token
            draft_tokens = self._draft_generate(current_input, self.gamma, temperature)
            
            # 2. 目标模型并行验证所有候选
            n_accepted = self._verify_and_accept(
                current_input, 
                draft_tokens, 
                temperature
            )
            
            # 3. 更新状态
            accepted_tokens = draft_tokens[:n_accepted]
            generated_tokens.extend(accepted_tokens.tolist())
            
            # 更新输入
            current_input = torch.cat([
                current_input, 
                accepted_tokens.unsqueeze(0)
            ], dim=1)
            
            # 检查是否生成EOS
            if self.tokenizer.eos_token_id in accepted_tokens:
                break
        
        return torch.cat([input_ids, torch.tensor([generated_tokens], device=input_ids.device)], dim=1)
    
    def _draft_generate(self, input_ids, n_tokens, temperature):
        """草稿模型生成"""
        draft_tokens = []
        current = input_ids
        
        for _ in range(n_tokens):
            logits = self.draft(current).logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token.item())
            current = torch.cat([current, next_token], dim=1)
        
        return torch.tensor(draft_tokens, device=input_ids.device)
    
    def _verify_and_accept(self, input_ids, draft_tokens, temperature):
        """验证并接受token"""
        
        # 构造包含所有草稿token的输入
        full_input = torch.cat([
            input_ids,
            draft_tokens.unsqueeze(0)
        ], dim=1)
        
        # 目标模型一次前向传播
        target_logits = self.target(full_input).logits
        
        # 获取草稿模型在每个位置的概率
        draft_input = input_ids
        draft_probs_list = []
        
        for i, token in enumerate(draft_tokens):
            draft_logits = self.draft(draft_input).logits[:, -1, :]
            draft_probs = F.softmax(draft_logits / temperature, dim=-1)
            draft_probs_list.append(draft_probs[0, token].item())
            draft_input = torch.cat([draft_input, token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # 目标模型在每个位置的概率
        n_accepted = 0
        for i, token in enumerate(draft_tokens):
            target_probs = F.softmax(
                target_logits[:, input_ids.size(1) + i - 1, :] / temperature, 
                dim=-1
            )
            target_prob = target_probs[0, token].item()
            draft_prob = draft_probs_list[i]
            
            # 接受概率
            accept_prob = min(1, target_prob / (draft_prob + 1e-10))
            
            if torch.rand(1).item() < accept_prob:
                n_accepted += 1
            else:
                # 拒绝，从这里开始重新采样
                break
        
        # 如果全部接受，还要再采样一个token
        if n_accepted == len(draft_tokens):
            # 从调整后的分布采样
            target_probs = F.softmax(target_logits[:, -1, :] / temperature, dim=-1)
            # ... 采样额外token
        
        return n_accepted


# 使用示例
def speculative_generation_demo():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 加载模型
    target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    draft = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")  # 实际应该用小模型
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    decoder = SpeculativeDecoder(target, draft, tokenizer, gamma=4)
    
    input_text = "The meaning of life is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
    
    output_ids = decoder.generate(input_ids, max_new_tokens=50)
    print(tokenizer.decode(output_ids[0]))
```

## vLLM部署

### 基本使用

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,      # GPU数量
    gpu_memory_utilization=0.9,  # GPU内存使用率
    max_model_len=4096,          # 最大序列长度
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# 批量推理
prompts = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Write a poem about AI:",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 50)
```

### vLLM服务端部署

```python
# 启动服务器
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-2-7b-hf \
#     --port 8000

# 客户端调用
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=256,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 自定义vLLM引擎

```python
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.outputs import RequestOutput

class CustomLLMServer:
    """自定义vLLM推理服务"""
    
    def __init__(self, model_name, **kwargs):
        engine_args = EngineArgs(
            model=model_name,
            **kwargs
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = 0
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        stream: bool = False
    ):
        """生成文本"""
        request_id = str(self.request_counter)
        self.request_counter += 1
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.engine.add_request(request_id, prompt, sampling_params)
        
        if stream:
            return self._stream_generate(request_id)
        else:
            return self._blocking_generate(request_id)
    
    def _blocking_generate(self, request_id):
        """阻塞式生成"""
        while True:
            request_outputs = self.engine.step()
            for output in request_outputs:
                if output.request_id == request_id and output.finished:
                    return output.outputs[0].text
    
    def _stream_generate(self, request_id):
        """流式生成"""
        prev_len = 0
        while True:
            request_outputs = self.engine.step()
            for output in request_outputs:
                if output.request_id == request_id:
                    text = output.outputs[0].text
                    new_text = text[prev_len:]
                    prev_len = len(text)
                    
                    if new_text:
                        yield new_text
                    
                    if output.finished:
                        return
```

## TensorRT-LLM部署

### 模型转换

```python
# TensorRT-LLM模型转换脚本
"""
# 1. 转换模型为TensorRT-LLM格式
python convert_checkpoint.py \
    --model_dir ./llama-7b-hf \
    --output_dir ./llama-7b-trtllm \
    --dtype float16

# 2. 构建TensorRT引擎
trtllm-build \
    --checkpoint_dir ./llama-7b-trtllm \
    --output_dir ./llama-7b-engine \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 512
"""

# Python API使用
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

def run_trtllm_inference():
    """使用TensorRT-LLM进行推理"""
    
    # 加载引擎
    runner = ModelRunner.from_dir(
        engine_dir="./llama-7b-engine",
        rank=0  # GPU rank
    )
    
    # 准备输入
    prompts = ["What is artificial intelligence?"]
    
    # 生成
    outputs = runner.generate(
        prompts,
        max_new_tokens=256,
        end_id=runner.end_id,
        pad_id=runner.pad_id,
        temperature=0.8,
        top_p=0.95
    )
    
    # 解码输出
    for i, output in enumerate(outputs):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output}")
```

### TensorRT-LLM服务部署

```python
# 使用Triton Inference Server部署
"""
# 启动Triton服务器
docker run --gpus all -it --rm \
    -v ./llama-7b-engine:/models/llama \
    nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 \
    tritonserver --model-repository=/models

# 或使用TensorRT-LLM自带的服务
python -m tensorrt_llm.serve \
    --model_dir ./llama-7b-engine \
    --port 8000
"""

import requests
import json

def trtllm_client_request(prompt, max_tokens=256):
    """TensorRT-LLM客户端请求"""
    
    url = "http://localhost:8000/v1/completions"
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }
    
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    return response.json()
```

## 性能对比

```python
import time
import torch

def benchmark_inference(model, tokenizer, prompts, max_new_tokens=256, num_runs=10):
    """推理性能测试"""
    
    # 预热
    for prompt in prompts[:2]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        _ = model.generate(input_ids, max_new_tokens=50)
    
    torch.cuda.synchronize()
    
    # 正式测试
    latencies = []
    total_tokens = 0
    
    for _ in range(num_runs):
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            output = model.generate(input_ids, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append(end - start)
            total_tokens += output.shape[1] - input_ids.shape[1]
    
    avg_latency = sum(latencies) / len(latencies)
    throughput = total_tokens / sum(latencies)
    
    return {
        "avg_latency_s": avg_latency,
        "throughput_tokens_per_s": throughput,
        "total_tokens": total_tokens
    }


# 对比不同框架
def compare_frameworks():
    """对比不同推理框架的性能"""
    prompts = ["Write a story about a robot:"] * 10
    
    results = {}
    
    # 1. HuggingFace原生
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_hf = AutoModelForCausalLM.from_pretrained("...").cuda()
    tokenizer = AutoTokenizer.from_pretrained("...")
    results["huggingface"] = benchmark_inference(model_hf, tokenizer, prompts)
    
    # 2. vLLM
    from vllm import LLM, SamplingParams
    llm = LLM(model="...")
    # ... benchmark vLLM
    
    # 3. TensorRT-LLM
    # ... benchmark TensorRT-LLM
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Latency: {result['avg_latency_s']:.3f}s")
        print(f"  Throughput: {result['throughput_tokens_per_s']:.1f} tokens/s")
```

## 总结

本文详细介绍了GPT推理优化的核心技术：

1. **KV Cache**：避免重复计算，是所有推理优化的基础
2. **Flash Attention**：通过分块计算减少内存访问
3. **量化**：INT8/INT4量化减少内存占用和计算量
4. **Speculative Decoding**：用小模型加速大模型推理
5. **推理框架**：vLLM和TensorRT-LLM的实战部署

### 选择指南

| 场景 | 推荐方案 |
|------|----------|
| 原型开发 | HuggingFace + Flash Attention |
| 生产部署 | vLLM（易用）或 TensorRT-LLM（最快） |
| 移动端/边缘 | 量化（GPTQ/AWQ） |
| 长文本 | Flash Attention + Continuous Batching |

### 性能优化清单

- [ ] 使用KV Cache
- [ ] 启用Flash Attention
- [ ] 根据精度需求选择量化方案
- [ ] 使用Continuous Batching处理并发请求
- [ ] 根据硬件选择合适的推理框架
- [ ] 监控GPU利用率和内存使用

下一篇文章，我们将深入探讨Prompt Engineering的实战技巧，帮助你更好地与GPT模型交互。
