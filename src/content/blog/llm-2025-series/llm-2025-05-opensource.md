---
title: '开源模型崛起：LLaMA 3、Qwen 2.5、DeepSeek'
description: '2025年开源大模型生态，包括 Meta LLaMA 3、阿里 Qwen 2.5、DeepSeek 等模型详解。'
pubDate: 2025-06-30
category: '技术'
tags: ['开源模型', 'LLaMA', 'Qwen', 'DeepSeek']
series: '2025 大模型实战'
seriesOrder: 5
heroImage: 'https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800&h=400&fit=crop'
---

2025 年，开源大模型迎来爆发式增长。LLaMA 3、Qwen 2.5、DeepSeek 等模型在多项基准测试中已接近甚至超越部分闭源模型。本文将深入介绍主流开源模型的特点与使用方法。

## 开源模型格局

### 主要玩家

| 模型 | 厂商 | 参数规模 | 开源协议 | 商用 |
|------|------|----------|----------|------|
| LLaMA 3 | Meta | 8B-405B | Llama License | ✅ |
| Qwen 2.5 | 阿里 | 0.5B-72B | Apache 2.0 | ✅ |
| DeepSeek V3 | DeepSeek | 67B MoE | MIT | ✅ |
| Mixtral | Mistral | 8x22B | Apache 2.0 | ✅ |
| Yi | 零一万物 | 6B-34B | Apache 2.0 | ✅ |

### 能力对比 (基准测试)

```
模型性能对比 (满分 100)

              MMLU    HumanEval   GSM8K   中文能力
LLaMA 3 70B   82.0      81.7      93.0     75
Qwen 2.5 72B  85.3      86.4      91.6     96 ⭐
DeepSeek V3   87.1      82.5      89.2     92
GPT-4 Turbo   86.4      87.1      92.0     88
Claude 3.5    88.7      92.0      96.4     90
```

## LLaMA 3 详解

### 模型版本

| 版本 | 参数 | 上下文 | 显存需求 | 适用场景 |
|------|------|--------|----------|----------|
| LLaMA 3 8B | 80 亿 | 8K | 16GB | 轻量部署 |
| LLaMA 3 70B | 700 亿 | 8K | 140GB | 高性能需求 |
| LLaMA 3 405B | 4050 亿 | 128K | 800GB | 顶级性能 |

### 本地部署

```bash
# 使用 Ollama 部署
ollama pull llama3:70b

# 启动服务
ollama serve

# 使用 API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3:70b",
  "prompt": "解释什么是 Transformer 架构"
}'
```

### Python 调用

```python
import requests

def chat_with_llama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:70b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# 使用 Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "你是一位有帮助的助手"},
    {"role": "user", "content": "介绍一下 Python 的特点"}
]

input_ids = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt"
).to(model.device)

outputs = model.generate(input_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Qwen 2.5 详解

### 模型特点

Qwen 2.5 是目前中文能力最强的开源模型：

```
┌─────────────────────────────────────────────────────┐
│              Qwen 2.5 核心优势                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📚 中文能力第一    超越所有开源和部分闭源模型        │
│  💻 代码能力强      多语言代码生成、调试              │
│  🧮 数学推理好      GSM8K/MATH 分数领先              │
│  📏 多尺寸可选      0.5B 到 72B 满足各种需求         │
│  🔧 工具调用        原生 Function Calling 支持       │
│  📝 长上下文        支持 128K tokens                 │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 快速使用

```python
# 使用 Qwen API (阿里云)
from dashscope import Generation

response = Generation.call(
    model='qwen2.5-72b-instruct',
    messages=[
        {"role": "system", "content": "你是一位专业的技术顾问"},
        {"role": "user", "content": "如何设计一个高并发系统？"}
    ]
)
print(response.output.text)

# 本地部署
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-72B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "请用 Python 实现一个 LRU 缓存"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Qwen-VL 多模态

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

image = Image.open("chart.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "分析这张图表的趋势"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False)
inputs = processor(text=text, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## DeepSeek 详解

### DeepSeek V3 特点

DeepSeek V3 采用 MoE (Mixture of Experts) 架构，实现高性价比：

| 特性 | 数值 |
|------|------|
| 总参数 | 671B |
| 激活参数 | 37B |
| 专家数量 | 256 |
| 上下文 | 128K |
| 训练成本 | 仅 $5.6M |

### API 调用

```python
from openai import OpenAI

# DeepSeek API 兼容 OpenAI 格式
client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一位专业的数据科学家"},
        {"role": "user", "content": "解释梯度下降算法"}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### DeepSeek Coder

```python
# 专业代码模型
response = client.chat.completions.create(
    model="deepseek-coder",
    messages=[
        {"role": "user", "content": """
实现一个 Python 装饰器，功能：
1. 缓存函数结果
2. 支持过期时间
3. 支持最大缓存数量
"""}
    ]
)
```

## 模型选型指南

### 按场景选择

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 中文对话 | Qwen 2.5 | 中文能力最强 |
| 代码生成 | DeepSeek Coder | 代码专项优化 |
| 通用任务 | LLaMA 3 70B | 生态完善 |
| 边缘部署 | Qwen 2.5 7B | 小尺寸高性能 |
| 多模态 | Qwen-VL | 开源最强多模态 |
| 成本敏感 | DeepSeek V3 | 性价比最高 |

### 硬件需求

```
模型显存需求参考 (FP16):

LLaMA 3 8B:    ~16GB  (单卡 4090)
LLaMA 3 70B:   ~140GB (8x A100 40GB)
Qwen 2.5 7B:   ~14GB  (单卡 4090)  
Qwen 2.5 72B:  ~144GB (8x A100 40GB)
DeepSeek V3:   ~64GB  (MoE 激活参数少)

量化后 (INT4):
LLaMA 3 70B:   ~40GB  (2x 4090)
Qwen 2.5 72B:  ~40GB  (2x 4090)
```

## 本地部署方案

### Ollama (推荐)

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen2.5:72b
ollama pull deepseek-v3:latest
ollama pull llama3:70b

# 运行对话
ollama run qwen2.5:72b

# API 调用
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:72b",
  "messages": [{"role": "user", "content": "你好"}]
}'
```

### vLLM 高性能部署

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    tensor_parallel_size=4,  # 4 卡并行
    dtype="bfloat16"
)

# 批量推理
prompts = [
    "解释机器学习",
    "Python 的优点",
    "什么是 Docker"
]

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Text Generation Inference

```bash
# 使用 Docker 部署
docker run --gpus all -p 8080:80 \
  -v /path/to/model:/model \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /model \
  --num-shard 4

# API 调用
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "你好", "parameters": {"max_new_tokens": 100}}'
```

## 微调开源模型

### LoRA 微调示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto"
)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 训练
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qwen-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
```

## 总结

开源模型 2025 年的关键进展：

1. **性能接近闭源**: Qwen 2.5、DeepSeek 在多项测试超越 GPT-4
2. **中文能力领先**: 国产模型在中文场景有明显优势
3. **部署成本降低**: 量化技术让 70B 模型可在消费级硬件运行
4. **生态日益完善**: Ollama、vLLM 等工具简化部署流程

选型建议：
- 中文场景首选 **Qwen 2.5**
- 代码任务首选 **DeepSeek Coder**
- 生态兼容首选 **LLaMA 3**
- 预算有限用 **DeepSeek V3 API**

下一篇将介绍本地部署 LLM 的详细方案：Ollama 与 LM Studio。
