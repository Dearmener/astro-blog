---
title: 'Function Calling 与 Tool Use'
description: '大模型函数调用与工具使用能力详解，构建能够执行实际操作的 AI 应用。'
pubDate: 2025-07-15
category: '技术'
tags: ['Function Calling', 'Tool Use', 'API', 'LLM']
series: '2025 大模型实战'
seriesOrder: 8
heroImage: 'https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&h=400&fit=crop'
---

Function Calling 让大模型能够调用外部函数，从"聊天助手"进化为"执行代理"。本文将详细介绍各主流模型的工具调用能力和实现方法。

## Function Calling 基础

### 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                  Function Calling 流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  用户: "北京今天天气怎么样？"                                  │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │    LLM      │  识别需要调用工具                           │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼ 返回工具调用请求                                   │
│  {                                                          │
│    "name": "get_weather",                                   │
│    "arguments": {"city": "北京"}                            │
│  }                                                          │
│         │                                                    │
│         ▼ 应用执行函数                                       │
│  ┌─────────────┐                                            │
│  │ get_weather │ ──► {"temp": 25, "weather": "晴"}         │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼ 将结果返回 LLM                                     │
│  ┌─────────────┐                                            │
│  │    LLM      │  生成自然语言回答                           │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  "北京今天天气晴朗，气温 25°C"                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 各平台对比

| 平台 | 功能名称 | 并行调用 | 强制调用 |
|------|----------|----------|----------|
| OpenAI | Function Calling | ✅ | ✅ |
| Anthropic | Tool Use | ✅ | ✅ |
| Google | Function Calling | ✅ | ✅ |
| 阿里云 | 工具调用 | ✅ | ✅ |

## OpenAI Function Calling

### 定义函数

```python
from openai import OpenAI

client = OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索商品信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food", "books"]
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### 处理调用

```python
import json

def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟天气 API"""
    return {"city": city, "temp": 25, "unit": unit, "weather": "晴"}

def search_products(query: str, category: str = None, max_price: float = None) -> list:
    """模拟商品搜索"""
    return [{"name": f"{query}商品", "price": 99.9}]

# 函数映射
function_map = {
    "get_weather": get_weather,
    "search_products": search_products
}

def chat_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    # 第一次调用：让 LLM 决定是否需要工具
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    
    # 检查是否有工具调用
    if message.tool_calls:
        messages.append(message)
        
        # 执行所有工具调用
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            # 执行函数
            result = function_map[func_name](**func_args)
            
            # 添加工具结果
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
        
        # 第二次调用：生成最终回答
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return message.content

# 使用
print(chat_with_tools("北京天气怎么样？"))
print(chat_with_tools("帮我找一款 500 元以内的蓝牙耳机"))
```

### 并行调用

```python
# GPT-4 支持并行调用多个函数
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京和上海今天的天气分别怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"
)

# 可能返回多个 tool_calls
for tool_call in response.choices[0].message.tool_calls:
    print(f"调用: {tool_call.function.name}")
    print(f"参数: {tool_call.function.arguments}")
```

### 强制调用

```python
# 强制使用特定函数
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)

# 强制不使用工具
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=tools,
    tool_choice="none"
)
```

## Claude Tool Use

### 定义工具

```python
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "获取城市天气",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "执行数学计算",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }
    }
]
```

### 处理调用

```python
def chat_with_claude_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # 检查是否需要工具调用
    while response.stop_reason == "tool_use":
        # 处理工具调用
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # 执行工具
                if block.name == "get_weather":
                    result = get_weather(**block.input)
                elif block.name == "calculate":
                    result = {"result": eval(block.input["expression"])}
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })
        
        # 继续对话
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
    
    # 提取文本回答
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    
    return ""

print(chat_with_claude_tools("北京天气如何？"))
print(chat_with_claude_tools("计算 (123 + 456) * 2"))
```

## 实战案例

### 智能客服系统

```python
class CustomerServiceBot:
    def __init__(self):
        self.client = OpenAI()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_order",
                    "description": "查询订单状态",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "订单号"}
                        },
                        "required": ["order_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_return",
                    "description": "创建退货申请",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["order_id", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "transfer_human",
                    "description": "转接人工客服",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_type": {"type": "string"},
                            "summary": {"type": "string"}
                        },
                        "required": ["issue_type"]
                    }
                }
            }
        ]
        self.system_prompt = """你是一位专业的客服代表。
请礼貌地回答客户问题，并在需要时使用工具查询信息或执行操作。"""
    
    def _execute_function(self, name: str, args: dict) -> dict:
        if name == "query_order":
            return {"order_id": args["order_id"], "status": "已发货", "eta": "3天后"}
        elif name == "create_return":
            return {"return_id": "RT123456", "status": "已受理"}
        elif name == "transfer_human":
            return {"queue_position": 3, "estimated_wait": "5分钟"}
    
    def chat(self, user_message: str, history: list = None) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                result = self._execute_function(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            final_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return final_response.choices[0].message.content
        
        return message.content

# 使用
bot = CustomerServiceBot()
print(bot.chat("我的订单 ORD123456 到哪了？"))
print(bot.chat("我想退货，商品有质量问题"))
```

### 数据分析助手

```python
import pandas as pd

class DataAnalysisAssistant:
    def __init__(self):
        self.client = OpenAI()
        self.df = None
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "load_data",
                    "description": "加载 CSV 数据文件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_data",
                    "description": "获取数据基本统计信息",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_data",
                    "description": "使用 Pandas 查询数据",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Pandas 查询表达式"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_chart",
                    "description": "生成图表",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chart_type": {"type": "string", "enum": ["bar", "line", "pie", "scatter"]},
                            "x_column": {"type": "string"},
                            "y_column": {"type": "string"},
                            "title": {"type": "string"}
                        },
                        "required": ["chart_type", "x_column", "y_column"]
                    }
                }
            }
        ]
    
    def _execute(self, name: str, args: dict) -> str:
        if name == "load_data":
            self.df = pd.read_csv(args["file_path"])
            return f"已加载数据，共 {len(self.df)} 行，{len(self.df.columns)} 列"
        
        elif name == "describe_data":
            if self.df is None:
                return "请先加载数据"
            return self.df.describe().to_string()
        
        elif name == "query_data":
            if self.df is None:
                return "请先加载数据"
            try:
                result = eval(f"self.df.{args['query']}")
                return str(result)
            except Exception as e:
                return f"查询错误: {e}"
        
        elif name == "plot_chart":
            return f"已生成 {args['chart_type']} 图表: {args.get('title', '未命名')}"
    
    def analyze(self, instruction: str) -> str:
        messages = [
            {"role": "system", "content": "你是一位数据分析专家，帮助用户分析数据。"},
            {"role": "user", "content": instruction}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools
        )
        
        # 处理多轮工具调用...
        return response.choices[0].message.content

# 使用
assistant = DataAnalysisAssistant()
print(assistant.analyze("加载 sales.csv 并分析销售趋势"))
```

## 最佳实践

### 1. 函数描述要清晰

```python
# ✅ 好的描述
{
    "name": "search_products",
    "description": "在商品数据库中搜索商品。支持关键词搜索，可按类别和价格范围筛选。返回匹配的商品列表，包含名称、价格、库存等信息。",
    "parameters": {...}
}

# ❌ 差的描述
{
    "name": "search",
    "description": "搜索",
    "parameters": {...}
}
```

### 2. 参数要有示例

```python
"parameters": {
    "type": "object",
    "properties": {
        "date": {
            "type": "string",
            "description": "日期，格式为 YYYY-MM-DD，例如：2025-01-15"
        }
    }
}
```

### 3. 错误处理

```python
def safe_execute(func_name: str, args: dict) -> dict:
    try:
        result = function_map[func_name](**args)
        return {"success": True, "data": result}
    except KeyError:
        return {"success": False, "error": f"未知函数: {func_name}"}
    except TypeError as e:
        return {"success": False, "error": f"参数错误: {e}"}
    except Exception as e:
        return {"success": False, "error": f"执行错误: {e}"}
```

## 总结

Function Calling 的核心要点：

1. **清晰的函数定义**: 详细的描述和参数说明
2. **正确的调用流程**: 请求 → 执行 → 返回结果 → 生成回答
3. **错误处理**: 优雅处理各种异常情况
4. **安全考虑**: 验证参数，限制敏感操作

下一篇将介绍多模态大模型的应用开发。
