---
title: 'Claude 3.5/4 深度使用指南'
description: 'Anthropic Claude 系列模型的完整使用指南，包括 API、提示技巧与最佳实践。'
pubDate: 2025-06-15
category: '技术'
tags: ['Claude', 'Anthropic', 'LLM', 'API']
series: '2025 大模型实战'
seriesOrder: 2
heroImage: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&h=400&fit=crop'
---

Claude 是 Anthropic 公司推出的大语言模型系列，以安全性、长上下文和代码能力著称。本文将深入介绍 Claude 3.5/4 的使用技巧。

## Claude 模型家族

### 当前模型版本

| 模型 | 上下文 | 特点 | 适用场景 |
|------|--------|------|----------|
| Claude 3.5 Sonnet | 200K | 性价比最高 | 日常开发 |
| Claude 3.5 Haiku | 200K | 速度最快 | 实时应用 |
| Claude 3 Opus | 200K | 能力最强 | 复杂任务 |
| Claude 4 | 500K | 下一代旗舰 | 企业级应用 |

### 定价策略

```
Claude 3.5 Sonnet:
  输入: $3 / 百万 tokens
  输出: $15 / 百万 tokens

Claude 3.5 Haiku:
  输入: $0.25 / 百万 tokens
  输出: $1.25 / 百万 tokens

Claude 3 Opus:
  输入: $15 / 百万 tokens
  输出: $75 / 百万 tokens
```

## API 快速入门

### 环境配置

```bash
# 安装 SDK
pip install anthropic

# 设置 API Key
export ANTHROPIC_API_KEY="sk-ant-xxx"
```

### 基础调用

```python
from anthropic import Anthropic

client = Anthropic()

# 简单对话
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "解释什么是量子计算"}
    ]
)

print(message.content[0].text)
```

### 流式输出

```python
# 流式响应 - 适合长文本生成
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    messages=[{"role": "user", "content": "写一篇关于 AI 的文章"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## System Prompt 最佳实践

### 角色定义

```python
system_prompt = """你是一位资深的 Python 开发专家，具有以下特点：

1. 专业背景：
   - 10 年 Python 开发经验
   - 精通 Django、FastAPI、asyncio
   - 熟悉设计模式和架构原则

2. 回答风格：
   - 代码优先，先给出可运行的示例
   - 解释关键概念，不假设用户已知
   - 指出常见陷阱和最佳实践

3. 输出格式：
   - 使用 Markdown 格式
   - 代码块标注语言类型
   - 复杂逻辑添加注释"""

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    system=system_prompt,
    messages=[
        {"role": "user", "content": "如何实现一个异步任务队列？"}
    ]
)
```

### 输出格式控制

```python
# 强制 JSON 输出
system_prompt = """你是一个数据提取助手。
请始终以 JSON 格式输出，包含以下字段：
- entities: 提取的实体列表
- relations: 实体间的关系
- summary: 简短摘要

示例输出：
{
  "entities": ["苹果公司", "iPhone"],
  "relations": [{"from": "苹果公司", "to": "iPhone", "type": "生产"}],
  "summary": "苹果公司是 iPhone 的制造商"
}"""
```

## 高级功能

### Vision 图片理解

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

# 分析图片
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encode_image("diagram.png")
                    }
                },
                {
                    "type": "text",
                    "text": "请详细描述这张架构图，并指出可能的优化点"
                }
            ]
        }
    ]
)
```

### Tool Use 工具调用

```python
# 定义工具
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_web",
        "description": "搜索网页获取最新信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
    }
]

# 调用带工具的对话
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ]
)

# 处理工具调用
for block in message.content:
    if block.type == "tool_use":
        print(f"调用工具: {block.name}")
        print(f"参数: {block.input}")
```

### Computer Use 计算机控制

```python
# Claude 可以控制计算机执行任务
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    tools=[
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080
        },
        {
            "type": "bash_20241022",
            "name": "bash"
        },
        {
            "type": "text_editor_20241022",
            "name": "str_replace_editor"
        }
    ],
    messages=[
        {
            "role": "user",
            "content": "帮我创建一个 Python 项目，包含虚拟环境和 requirements.txt"
        }
    ]
)
```

## Prompt Engineering 技巧

### 1. 分步骤思考

```python
prompt = """请分析以下代码的性能问题：

```python
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates
```

请按以下步骤分析：
1. 首先，分析当前代码的时间复杂度
2. 然后，找出性能瓶颈
3. 接着，提出优化方案
4. 最后，给出优化后的代码"""
```

### 2. Few-shot 示例

```python
prompt = """将自然语言转换为 SQL 查询。

示例 1：
问题：查找所有年龄大于 30 的用户
SQL：SELECT * FROM users WHERE age > 30;

示例 2：
问题：统计每个部门的员工数量
SQL：SELECT department, COUNT(*) as count FROM employees GROUP BY department;

示例 3：
问题：找出购买金额最高的前 10 个客户
SQL：SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 10;

现在请转换：
问题：查找 2024 年注册且订单数超过 5 的用户"""
```

### 3. XML 标签组织

```python
prompt = """<task>
分析以下产品评论的情感倾向
</task>

<reviews>
1. 这个产品质量很好，物流也快，非常满意！
2. 包装破损，客服态度差，再也不买了
3. 一般般吧，没有想象中那么好
</reviews>

<output_format>
请以 JSON 格式输出，每条评论包含：
- review_id: 评论编号
- sentiment: positive/negative/neutral
- confidence: 0-1 的置信度
- keywords: 关键情感词
</output_format>"""
```

## 长上下文处理

### 文档分析

```python
def analyze_long_document(doc_path: str):
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Claude 支持 200K tokens，约 15 万字
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": f"""请分析以下文档：

<document>
{content}
</document>

请提供：
1. 文档摘要（200字以内）
2. 主要观点（列表形式）
3. 关键数据和结论
4. 潜在问题或不足"""
            }
        ]
    )
    return message.content[0].text
```

### 多轮对话记忆

```python
class ClaudeChat:
    def __init__(self, system_prompt: str = None):
        self.client = Anthropic()
        self.system = system_prompt
        self.messages = []
    
    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=self.system,
            messages=self.messages
        )
        
        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def clear_history(self):
        self.messages = []

# 使用示例
chat = ClaudeChat(system_prompt="你是一位友好的 AI 助手")
print(chat.chat("你好！"))
print(chat.chat("刚才我说了什么？"))  # Claude 记得上文
```

## 实战案例：代码审查助手

```python
class CodeReviewer:
    def __init__(self):
        self.client = Anthropic()
        self.system = """你是一位严格的代码审查专家。
审查时请关注：
1. 代码质量：可读性、命名规范、注释
2. 潜在 Bug：边界条件、空值处理、类型安全
3. 性能问题：算法复杂度、内存使用
4. 安全漏洞：注入攻击、敏感数据暴露
5. 最佳实践：设计模式、SOLID 原则"""
    
    def review(self, code: str, language: str = "python") -> dict:
        prompt = f"""请审查以下 {language} 代码：

```{language}
{code}
```

请以 JSON 格式输出审查结果：
{{
  "score": 1-10 的评分,
  "issues": [
    {{"severity": "high/medium/low", "line": 行号, "description": "问题描述", "suggestion": "修改建议"}}
  ],
  "positives": ["代码的优点"],
  "refactored_code": "优化后的完整代码"
}}"""

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=self.system,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# 使用
reviewer = CodeReviewer()
result = reviewer.review("""
def process(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result
""")
print(result)
```

## 总结

Claude 3.5/4 是当前最强大的 LLM 之一，特别适合：

- **代码开发**: 代码生成、审查、重构
- **长文档处理**: 200K 上下文支持整本书分析
- **多模态任务**: 图片理解、图表分析
- **企业应用**: 安全可控、合规友好

使用建议：
1. 善用 System Prompt 定义角色和输出格式
2. 使用 XML 标签组织复杂 Prompt
3. 利用 Tool Use 扩展能力边界
4. 长文档直接输入，无需分块处理

下一篇我们将介绍 GPT-5 与 ChatGPT 的最新功能。
