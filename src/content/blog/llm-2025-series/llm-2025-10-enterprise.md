---
title: '企业级大模型部署与微调'
description: '企业环境中大模型的部署架构、安全考量、微调策略与成本优化。'
pubDate: 2025-07-25
category: '技术'
tags: ['企业部署', '微调', 'Fine-tuning', 'LLM']
series: '2025 大模型实战'
seriesOrder: 10
heroImage: 'https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=800&h=400&fit=crop'
---

企业级大模型部署需要考虑性能、安全、成本等多方面因素。本文将详细介绍企业环境中的 LLM 部署架构和微调策略。

## 企业部署架构

### 部署模式选择

| 模式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| API 调用 | 无需运维、快速上线 | 数据外传、成本不可控 | 快速验证 |
| 私有化部署 | 数据安全、完全可控 | 运维成本高 | 金融、医疗 |
| 混合部署 | 灵活平衡 | 架构复杂 | 大型企业 |

### 私有化架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    企业 LLM 部署架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   负载均衡层                                                 │
│   ┌──────────────────────────────────────────────┐          │
│   │              Nginx / Kong API Gateway         │          │
│   └──────────────────────────────────────────────┘          │
│                          │                                   │
│   推理服务层                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│   │  vLLM    │  │  vLLM    │  │  vLLM    │                 │
│   │ Worker 1 │  │ Worker 2 │  │ Worker 3 │                 │
│   └──────────┘  └──────────┘  └──────────┘                 │
│                          │                                   │
│   模型存储层                                                 │
│   ┌──────────────────────────────────────────────┐          │
│   │           MinIO / S3 (模型文件存储)           │          │
│   └──────────────────────────────────────────────┘          │
│                          │                                   │
│   监控层                                                     │
│   ┌──────────────────────────────────────────────┐          │
│   │     Prometheus + Grafana + ELK Stack          │          │
│   └──────────────────────────────────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Kubernetes 部署

```yaml
# llm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model=/models/Qwen2.5-72B-Instruct
          - --tensor-parallel-size=4
          - --max-model-len=32768
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: 320Gi
          requests:
            nvidia.com/gpu: 4
            memory: 256Gi
        volumeMounts:
        - name: model-storage
          mountPath: /models
        ports:
        - containerPort: 8000
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-inference
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

## 高性能推理优化

### vLLM 部署

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    tensor_parallel_size=4,
    dtype="bfloat16",
    max_model_len=32768,
    gpu_memory_utilization=0.9
)

# 批量推理
prompts = ["问题1", "问题2", "问题3"]
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    top_p=0.9
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

### vLLM OpenAI 兼容服务

```bash
# 启动 OpenAI 兼容服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0

# 使用 OpenAI SDK 调用
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[{"role": "user", "content": "你好"}]
)
```

### TGI (Text Generation Inference)

```bash
# 使用 Docker 部署
docker run --gpus all -p 8080:80 \
  -v /path/to/model:/model \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /model \
  --num-shard 4 \
  --max-input-length 4096 \
  --max-total-tokens 8192
```

## 模型微调

### LoRA 微调

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练配置
training_args = TrainingArguments(
    output_dir="./qwen-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch"
)

# 数据集
dataset = load_dataset("json", data_files="train_data.json")

# 训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    max_seq_length=2048
)
trainer.train()

# 保存
model.save_pretrained("./qwen-lora-final")
```

### QLoRA 低资源微调

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# 继续 LoRA 微调...
```

### 数据准备

```python
import json

def prepare_training_data(raw_data: list) -> list:
    """准备训练数据"""
    formatted_data = []
    
    for item in raw_data:
        formatted_data.append({
            "text": f"""<|im_start|>system
你是一位专业的客服助手。<|im_end|>
<|im_start|>user
{item['question']}<|im_end|>
<|im_start|>assistant
{item['answer']}<|im_end|>"""
        })
    
    return formatted_data

# 数据格式示例
raw_data = [
    {"question": "如何退货？", "answer": "退货流程：1. 登录账户..."},
    {"question": "运费怎么算？", "answer": "运费规则：满99免运费..."}
]

train_data = prepare_training_data(raw_data)
with open("train_data.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
```

## 安全与合规

### 数据安全

```python
import hashlib
from cryptography.fernet import Fernet

class DataProtection:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)
    
    def mask_pii(self, text: str) -> str:
        """脱敏 PII 数据"""
        import re
        
        # 手机号脱敏
        text = re.sub(r'1[3-9]\d{9}', lambda m: m.group()[:3] + '****' + m.group()[-4:], text)
        
        # 身份证脱敏
        text = re.sub(r'\d{17}[\dXx]', lambda m: m.group()[:6] + '********' + m.group()[-4:], text)
        
        # 邮箱脱敏
        text = re.sub(r'(\w{2})\w+(@\w+)', r'\1***\2', text)
        
        return text
    
    def encrypt(self, text: str) -> str:
        """加密敏感数据"""
        return self.cipher.encrypt(text.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """解密数据"""
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def audit_log(self, user: str, action: str, data_hash: str):
        """审计日志"""
        import datetime
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user": user,
            "action": action,
            "data_hash": data_hash
        }
        # 写入审计日志
        print(f"Audit: {log_entry}")

# 使用
protection = DataProtection(Fernet.generate_key())
masked_text = protection.mask_pii("联系电话：13812345678")
print(masked_text)  # 联系电话：138****5678
```

### 访问控制

```python
from functools import wraps
import jwt

class AccessControl:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.permissions = {
            "admin": ["read", "write", "delete", "fine-tune"],
            "developer": ["read", "write"],
            "viewer": ["read"]
        }
    
    def verify_token(self, token: str) -> dict:
        """验证 JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def check_permission(self, role: str, action: str) -> bool:
        """检查权限"""
        return action in self.permissions.get(role, [])
    
    def require_permission(self, action: str):
        """权限装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(token: str, *args, **kwargs):
                payload = self.verify_token(token)
                if not self.check_permission(payload["role"], action):
                    raise PermissionError(f"No permission for {action}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

# 使用
ac = AccessControl("secret")

@ac.require_permission("fine-tune")
def run_fine_tuning(model_id: str, dataset: str):
    print(f"Fine-tuning {model_id} with {dataset}")
```

## 成本优化

### 模型量化

```python
from transformers import AutoModelForCausalLM
from awq import AutoAWQForCausalLM

# AWQ 量化
model = AutoAWQForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    device_map="auto"
)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data
)

model.save_quantized("./qwen-72b-awq")
```

### 推理优化

```python
class InferenceOptimizer:
    def __init__(self):
        self.cache = {}
    
    def cache_result(self, prompt_hash: str, result: str, ttl: int = 3600):
        """缓存结果"""
        import time
        self.cache[prompt_hash] = {
            "result": result,
            "expires": time.time() + ttl
        }
    
    def get_cached(self, prompt_hash: str) -> str:
        """获取缓存"""
        import time
        if prompt_hash in self.cache:
            if time.time() < self.cache[prompt_hash]["expires"]:
                return self.cache[prompt_hash]["result"]
        return None
    
    def batch_inference(self, prompts: list, model) -> list:
        """批量推理"""
        # 检查缓存
        results = [None] * len(prompts)
        uncached_indices = []
        uncached_prompts = []
        
        for i, prompt in enumerate(prompts):
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            cached = self.get_cached(prompt_hash)
            if cached:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_prompts.append(prompt)
        
        # 批量处理未缓存的
        if uncached_prompts:
            new_results = model.generate(uncached_prompts)
            for idx, result in zip(uncached_indices, new_results):
                results[idx] = result
                prompt_hash = hashlib.md5(prompts[idx].encode()).hexdigest()
                self.cache_result(prompt_hash, result)
        
        return results
```

### 成本监控

```python
class CostMonitor:
    def __init__(self):
        self.usage = {}
    
    def track_usage(self, user: str, model: str, input_tokens: int, output_tokens: int):
        """记录使用量"""
        if user not in self.usage:
            self.usage[user] = {}
        if model not in self.usage[user]:
            self.usage[user][model] = {"input": 0, "output": 0}
        
        self.usage[user][model]["input"] += input_tokens
        self.usage[user][model]["output"] += output_tokens
    
    def calculate_cost(self, user: str) -> float:
        """计算成本"""
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "qwen-72b": {"input": 0.004, "output": 0.012}
        }
        
        total = 0
        for model, tokens in self.usage.get(user, {}).items():
            if model in pricing:
                total += tokens["input"] * pricing[model]["input"] / 1000
                total += tokens["output"] * pricing[model]["output"] / 1000
        
        return total
    
    def get_report(self) -> dict:
        """生成报告"""
        report = {}
        for user in self.usage:
            report[user] = {
                "usage": self.usage[user],
                "cost": self.calculate_cost(user)
            }
        return report

# 使用
monitor = CostMonitor()
monitor.track_usage("team_a", "qwen-72b", 1000, 500)
print(monitor.get_report())
```

## 监控与运维

### Prometheus 指标

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义指标
REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])
LATENCY = Histogram('llm_request_latency_seconds', 'Request latency', ['model'])
TOKENS_USED = Counter('llm_tokens_total', 'Tokens used', ['model', 'type'])
GPU_MEMORY = Gauge('gpu_memory_usage_bytes', 'GPU memory usage', ['gpu_id'])

def track_request(model: str, latency: float, input_tokens: int, output_tokens: int, success: bool):
    """记录请求指标"""
    REQUEST_COUNT.labels(model=model, status="success" if success else "error").inc()
    LATENCY.labels(model=model).observe(latency)
    TOKENS_USED.labels(model=model, type="input").inc(input_tokens)
    TOKENS_USED.labels(model=model, type="output").inc(output_tokens)

# 启动指标服务器
start_http_server(9090)
```

## 总结

企业级 LLM 部署的关键要点：

1. **架构选择**: 根据数据安全需求选择 API/私有化/混合部署
2. **性能优化**: vLLM/TGI + 量化 + 批处理 + 缓存
3. **微调策略**: LoRA/QLoRA 降低资源需求
4. **安全合规**: 数据脱敏 + 访问控制 + 审计日志
5. **成本控制**: 量化压缩 + 智能路由 + 使用监控

下一步建议：
- 从小规模 POC 开始验证
- 建立完善的监控体系
- 制定清晰的 SLA 标准
- 持续优化成本效率
