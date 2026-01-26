---
title: 'Gemini 2.0 与 Google AI 生态'
description: 'Google Gemini 2.0 多模态能力与 Google AI 生态系统完整指南。'
pubDate: 2025-06-25
category: '技术'
tags: ['Gemini', 'Google AI', '多模态', 'LLM']
series: '2025 大模型实战'
seriesOrder: 4
heroImage: 'https://images.unsplash.com/photo-1573804633927-bfcbcd909acd?w=800&h=400&fit=crop'
---

Google Gemini 2.0 是当前多模态能力最强的大模型之一，拥有 200 万 tokens 的超长上下文和原生多模态理解能力。本文将深入介绍 Gemini 2.0 的使用方法。

## Gemini 模型家族

### 模型版本

| 模型 | 上下文 | 特点 | 定价 (输入/输出) |
|------|--------|------|------------------|
| Gemini 2.0 Flash | 1M | 速度快、性价比高 | $0.075 / $0.30 |
| Gemini 2.0 Pro | 2M | 能力最强 | $1.25 / $5.00 |
| Gemini 1.5 Flash | 1M | 稳定版本 | $0.075 / $0.30 |

### 核心优势

```
┌─────────────────────────────────────────────────────┐
│              Gemini 2.0 核心能力                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  🔤 文本理解     200万 tokens 超长上下文             │
│  👁️ 视觉理解     图片、图表、文档、屏幕截图          │
│  🎵 音频处理     语音识别、音乐理解                   │
│  🎬 视频分析     直接理解视频内容                     │
│  💻 代码生成     多语言代码生成与解释                 │
│  🔧 工具使用     Function Calling + Google 搜索      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## 快速开始

### 环境配置

```bash
# 安装 SDK
pip install google-generativeai

# 设置 API Key
export GOOGLE_API_KEY="your-api-key"
```

### 基础调用

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

# 创建模型实例
model = genai.GenerativeModel('gemini-2.0-flash')

# 简单对话
response = model.generate_content("解释什么是机器学习")
print(response.text)
```

### 流式输出

```python
# 流式响应
response = model.generate_content(
    "写一篇关于人工智能发展历史的文章",
    stream=True
)

for chunk in response:
    print(chunk.text, end="")
```

## 多模态能力

### 图片理解

```python
import PIL.Image

# 加载图片
image = PIL.Image.open("diagram.png")

# 图片分析
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content([
    "请详细分析这张架构图：",
    image
])
print(response.text)

# 多图对比
image1 = PIL.Image.open("before.png")
image2 = PIL.Image.open("after.png")

response = model.generate_content([
    "对比这两张图片的差异：",
    image1,
    image2
])
```

### 视频分析

```python
# 上传视频文件
video_file = genai.upload_file("meeting.mp4")

# 等待处理完成
import time
while video_file.state.name == "PROCESSING":
    time.sleep(5)
    video_file = genai.get_file(video_file.name)

# 分析视频
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content([
    "总结这段视频的主要内容，列出关键时间点：",
    video_file
])
print(response.text)
```

### 音频理解

```python
# 上传音频
audio_file = genai.upload_file("podcast.mp3")

# 音频转写和分析
response = model.generate_content([
    "请转写这段音频，并总结主要观点：",
    audio_file
])
```

### PDF 文档分析

```python
# 直接分析 PDF
pdf_file = genai.upload_file("research_paper.pdf")

response = model.generate_content([
    """请分析这篇论文：
    1. 研究背景和动机
    2. 主要方法和创新点
    3. 实验结果
    4. 局限性和未来方向""",
    pdf_file
])
```

## 200 万 Token 长上下文

### 整本书分析

```python
# Gemini 2.0 Pro 支持 200 万 tokens
# 约等于 1500 页书籍

model = genai.GenerativeModel('gemini-2.0-pro')

with open("entire_book.txt", "r") as f:
    book_content = f.read()

response = model.generate_content(f"""
请分析这本书的完整内容：

{book_content}

请提供：
1. 全书摘要（500字）
2. 章节结构分析
3. 主要人物/概念关系图
4. 核心论点和论据
5. 写作风格评价
""")
```

### 代码库分析

```python
import os

def read_codebase(directory):
    code_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.js', '.ts')):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    code_files.append(f"=== {path} ===\n{f.read()}")
    return "\n\n".join(code_files)

codebase = read_codebase("./my_project")

response = model.generate_content(f"""
分析以下代码库：

{codebase}

请提供：
1. 项目架构概述
2. 主要模块功能
3. 代码质量评估
4. 潜在问题和改进建议
5. 技术栈分析
""")
```

## Google AI Studio

### 在线测试

Google AI Studio 提供免费的在线测试环境：

```
https://aistudio.google.com/

功能特点：
- 即时测试各版本模型
- 可视化 Prompt 调试
- 多模态内容上传
- API 代码导出
- 使用量监控
```

### Prompt 模板

```python
# 使用预设模板
from google.generativeai import protos

# 创建带模板的模型
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction="""你是一位专业的数据分析师。
分析数据时请：
1. 先检查数据质量
2. 提供统计摘要
3. 发现异常值
4. 给出可视化建议
5. 提出业务洞察"""
)

response = model.generate_content("分析以下销售数据...")
```

## Vertex AI 企业集成

### 配置 Vertex AI

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# 初始化 Vertex AI
vertexai.init(project="your-project-id", location="us-central1")

# 使用企业版 Gemini
model = GenerativeModel("gemini-2.0-flash")

response = model.generate_content("企业级任务...")
```

### 安全与合规

```python
# 配置安全设置
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
}

model = genai.GenerativeModel(
    'gemini-2.0-flash',
    safety_settings=safety_settings
)
```

## Function Calling

### 定义函数

```python
# 定义可调用的函数
def get_weather(city: str, unit: str = "celsius") -> dict:
    """获取城市天气"""
    # 实际调用天气 API
    return {"city": city, "temp": 25, "unit": unit}

def search_flights(origin: str, destination: str, date: str) -> list:
    """搜索航班"""
    return [{"flight": "CA123", "price": 1500}]

# 创建工具
tools = [
    genai.Tool(function_declarations=[
        genai.FunctionDeclaration(
            name="get_weather",
            description="获取指定城市的天气信息",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        ),
        genai.FunctionDeclaration(
            name="search_flights",
            description="搜索航班信息",
            parameters={
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["origin", "destination", "date"]
            }
        )
    ])
]

model = genai.GenerativeModel('gemini-2.0-flash', tools=tools)
```

### 处理函数调用

```python
chat = model.start_chat()

response = chat.send_message("北京明天天气怎么样？")

# 检查是否有函数调用
for part in response.parts:
    if hasattr(part, 'function_call'):
        fc = part.function_call
        print(f"调用函数: {fc.name}")
        print(f"参数: {dict(fc.args)}")
        
        # 执行函数
        if fc.name == "get_weather":
            result = get_weather(**dict(fc.args))
        
        # 返回结果
        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )]
            )
        )
        print(response.text)
```

## 实战案例：智能文档助手

```python
class DocumentAssistant:
    def __init__(self):
        self.model = genai.GenerativeModel(
            'gemini-2.0-pro',
            system_instruction="""你是一位智能文档助手。
能够：
1. 理解各种格式的文档（PDF、图片、视频）
2. 回答关于文档内容的问题
3. 提取关键信息和数据
4. 生成摘要和报告"""
        )
        self.chat = self.model.start_chat()
        self.uploaded_files = []
    
    def upload_document(self, file_path: str):
        """上传文档"""
        file = genai.upload_file(file_path)
        self.uploaded_files.append(file)
        return f"已上传: {file_path}"
    
    def ask(self, question: str) -> str:
        """提问"""
        content = [question] + self.uploaded_files
        response = self.chat.send_message(content)
        return response.text
    
    def summarize(self) -> str:
        """生成摘要"""
        return self.ask("请为所有上传的文档生成一份综合摘要")
    
    def extract_data(self, data_type: str) -> str:
        """提取特定数据"""
        return self.ask(f"请从文档中提取所有的{data_type}")

# 使用示例
assistant = DocumentAssistant()
assistant.upload_document("report.pdf")
assistant.upload_document("chart.png")

print(assistant.ask("这份报告的主要结论是什么？"))
print(assistant.extract_data("财务数据"))
print(assistant.summarize())
```

## 与其他模型对比

| 能力 | Gemini 2.0 | GPT-5 | Claude 4 |
|------|------------|-------|----------|
| 上下文长度 | 2M ⭐ | 256K | 500K |
| 多模态 | 原生支持 ⭐ | 支持 | 支持 |
| 视频理解 | 原生支持 ⭐ | 支持 | 不支持 |
| 搜索整合 | Google 搜索 ⭐ | Bing | 无 |
| 代码能力 | 强 | 最强 | 很强 |
| 中文能力 | 良好 | 良好 | 很好 |

## 总结

Gemini 2.0 的核心优势：

1. **超长上下文**: 200 万 tokens 可处理整本书
2. **原生多模态**: 图片、视频、音频一体化
3. **Google 生态**: 与搜索、云服务深度整合
4. **性价比高**: Flash 版本价格实惠

适用场景：
- 长文档分析（合同、论文、书籍）
- 视频内容理解
- 多模态应用开发
- 需要搜索增强的场景

下一篇将介绍开源模型生态：LLaMA 3、Qwen 2.5、DeepSeek。
