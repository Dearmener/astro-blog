---
title: '本地部署 LLM：Ollama 与 LM Studio'
description: '在本地运行大语言模型的完整指南，使用 Ollama 和 LM Studio 实现私有化部署。'
pubDate: 2025-07-05
category: '技术'
tags: ['本地部署', 'Ollama', 'LM Studio', 'LLM']
series: '2025 大模型实战'
seriesOrder: 6
heroImage: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&h=400&fit=crop'
---

本地部署大模型可以确保数据隐私、降低使用成本、实现离线运行。本文将详细介绍使用 Ollama 和 LM Studio 进行本地 LLM 部署的完整方案。

## 为什么选择本地部署

### 优势对比

| 方面 | 云端 API | 本地部署 |
|------|----------|----------|
| 数据隐私 | 数据上传云端 | 完全本地处理 |
| 使用成本 | 按 Token 计费 | 一次投入长期使用 |
| 网络依赖 | 需要联网 | 支持离线 |
| 响应延迟 | 网络延迟 | 本地低延迟 |
| 定制化 | 有限 | 完全可控 |

### 硬件要求

```
推荐配置参考：

入门级 (7B 模型):
- GPU: RTX 4060 8GB / M1 Mac
- RAM: 16GB
- 存储: 50GB SSD

标准级 (13-14B 模型):
- GPU: RTX 4080 16GB / M2 Pro Mac
- RAM: 32GB
- 存储: 100GB SSD

进阶级 (70B 模型):
- GPU: RTX 4090 24GB x2 / M3 Max Mac
- RAM: 64GB
- 存储: 200GB SSD
```

## Ollama 完全指南

### 安装配置

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (使用 WSL2 或原生安装)
# 下载安装包: https://ollama.com/download

# 验证安装
ollama --version
```

### 基础使用

```bash
# 拉取模型
ollama pull llama3:8b           # 基础版
ollama pull llama3:70b          # 高性能版
ollama pull qwen2.5:7b          # 中文优化
ollama pull deepseek-coder:33b  # 代码专用
ollama pull mistral:7b          # 高效能

# 运行对话
ollama run qwen2.5:7b

# 查看已下载模型
ollama list

# 删除模型
ollama rm llama3:8b

# 查看模型信息
ollama show qwen2.5:7b
```

### API 调用

```python
import requests
import json

# 基础调用
def chat(prompt: str, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# 流式调用
def chat_stream(prompt: str, model: str = "qwen2.5:7b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                yield data["response"]

# 使用示例
print(chat("什么是机器学习？"))

for chunk in chat_stream("写一首关于编程的诗"):
    print(chunk, end="", flush=True)
```

### Chat API (推荐)

```python
# 对话格式 API
def chat_completion(messages: list, model: str = "qwen2.5:7b") -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# 多轮对话
messages = [
    {"role": "system", "content": "你是一位 Python 专家"},
    {"role": "user", "content": "如何实现单例模式？"}
]
print(chat_completion(messages))

messages.append({"role": "assistant", "content": "..."})
messages.append({"role": "user", "content": "用装饰器实现呢？"})
print(chat_completion(messages))
```

### 自定义模型 (Modelfile)

```dockerfile
# Modelfile - 创建自定义模型
FROM qwen2.5:7b

# 设置系统提示
SYSTEM """你是一位专业的代码审查专家。
审查代码时请：
1. 检查代码规范
2. 发现潜在 bug
3. 提出优化建议
"""

# 调整参数
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

# 设置停止词
PARAMETER stop "<|endoftext|>"
PARAMETER stop "```"
```

```bash
# 创建自定义模型
ollama create code-reviewer -f Modelfile

# 使用自定义模型
ollama run code-reviewer
```

### OpenAI 兼容 API

```python
from openai import OpenAI

# Ollama 兼容 OpenAI API
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 任意值
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "你是一位有帮助的助手"},
        {"role": "user", "content": "解释什么是 Docker"}
    ]
)

print(response.choices[0].message.content)
```

## LM Studio 完全指南

### 安装与界面

```
下载地址: https://lmstudio.ai/

支持平台:
- Windows (x64)
- macOS (Intel/Apple Silicon)
- Linux (AppImage)

特点:
- 图形化界面
- 内置模型下载
- 一键启动服务器
- 实时性能监控
```

### 模型下载

```
推荐模型 (按场景):

通用对话:
- TheBloke/Qwen2.5-7B-Instruct-GGUF
- TheBloke/Llama-3-8B-Instruct-GGUF

中文优化:
- TheBloke/Qwen2.5-14B-Instruct-GGUF
- THUDM/glm-4-9b-chat-GGUF

代码生成:
- TheBloke/DeepSeek-Coder-33B-Instruct-GGUF
- TheBloke/CodeLlama-34B-Instruct-GGUF

量化版本选择:
- Q4_K_M: 平衡质量和大小 (推荐)
- Q5_K_M: 更高质量
- Q8_0: 最高质量
- Q2_K: 最小体积
```

### API 使用

```python
# LM Studio 本地服务器 (默认端口 1234)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# 对话
response = client.chat.completions.create(
    model="local-model",  # LM Studio 中加载的模型
    messages=[
        {"role": "user", "content": "你好，介绍一下自己"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "写一段 Python 代码"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 性能优化

### GPU 加速配置

```bash
# Ollama GPU 设置
export OLLAMA_GPU_LAYERS=35  # GPU 加载层数

# 多 GPU
export CUDA_VISIBLE_DEVICES=0,1

# Apple Silicon 优化
# 自动使用 Metal，无需配置
```

### 量化模型

```python
# 使用量化减少显存占用

# GGUF 量化格式 (推荐)
# Q4_K_M: 4-bit 量化，保持较好质量
# 70B 模型: ~40GB → ~40GB (Q4) → ~35GB (Q4_K_M)

# 在 Ollama 中使用量化模型
ollama pull llama3:70b-instruct-q4_K_M
```

### 上下文长度优化

```bash
# 增加上下文长度
ollama run qwen2.5:7b --ctx-size 16384

# 在 Modelfile 中设置
PARAMETER num_ctx 16384
```

## 实战案例

### 本地 RAG 系统

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 初始化本地模型
llm = Ollama(model="qwen2.5:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 加载文档
from langchain_community.document_loaders import TextLoader
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 创建 RAG 链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 查询
result = qa_chain.invoke("文档中提到了什么技术？")
print(result)
```

### 本地代码助手

```python
import subprocess
import requests

class LocalCodeAssistant:
    def __init__(self, model="deepseek-coder:33b"):
        self.model = model
        self.base_url = "http://localhost:11434/api"
    
    def generate_code(self, description: str) -> str:
        """根据描述生成代码"""
        prompt = f"""请根据以下需求生成 Python 代码：

需求：{description}

要求：
- 代码要有详细注释
- 包含错误处理
- 遵循 PEP 8 规范

代码："""
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
    
    def review_code(self, code: str) -> str:
        """审查代码"""
        prompt = f"""请审查以下代码：

```python
{code}
```

请指出：
1. 潜在的 bug
2. 性能问题
3. 代码风格问题
4. 改进建议"""
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
    
    def explain_code(self, code: str) -> str:
        """解释代码"""
        prompt = f"""请详细解释以下代码的功能：

```python
{code}
```"""
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]

# 使用
assistant = LocalCodeAssistant()
code = assistant.generate_code("实现一个线程安全的单例模式")
print(code)
```

### 本地翻译服务

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text")
    source = data.get("source", "auto")
    target = data.get("target", "en")
    
    prompt = f"""将以下{source}文本翻译成{target}:

{text}

只输出翻译结果，不要添加解释。"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False
        }
    )
    
    return jsonify({
        "translation": response.json()["response"]
    })

if __name__ == "__main__":
    app.run(port=5000)
```

## Ollama vs LM Studio 对比

| 特性 | Ollama | LM Studio |
|------|--------|-----------|
| 界面 | 命令行 | 图形化 |
| 易用性 | 需要命令行基础 | 小白友好 |
| 模型管理 | 命令行管理 | 可视化管理 |
| API | 内置服务器 | 需手动开启 |
| 自定义 | Modelfile 灵活 | 界面配置 |
| 资源占用 | 较低 | 稍高 |
| 适合人群 | 开发者 | 普通用户 |

## 总结

本地部署 LLM 的关键要点：

1. **选择合适的模型**: 根据硬件和需求选择参数量
2. **使用量化版本**: Q4_K_M 平衡质量和性能
3. **善用工具**: Ollama 适合开发，LM Studio 适合体验
4. **优化配置**: GPU 层数、上下文长度按需调整

推荐组合：
- 开发环境: Ollama + Qwen 2.5 7B
- 代码任务: Ollama + DeepSeek Coder 33B
- 日常体验: LM Studio + 各种模型切换

下一篇将介绍 RAG 检索增强生成的详细实现方案。
