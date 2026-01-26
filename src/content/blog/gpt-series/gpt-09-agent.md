---
title: 'GPT完全指南（九）：GPT Agent开发'
description: '深入学习GPT Agent的开发技术，包括Function Calling、工具使用、LangChain框架、多Agent协作以及实战项目开发'
pubDate: '2024-03-09'
heroImage: ''
category: '技术'
tags: ['GPT', 'Agent', 'LangChain', 'Function Calling', 'AI']
series: 'GPT完全指南'
seriesOrder: 9
---

GPT Agent是让大语言模型从"问答机器"进化为"智能助手"的关键技术。通过赋予GPT使用工具、访问外部信息和执行操作的能力，我们可以构建真正有用的AI应用。本文将深入探讨GPT Agent的开发技术。

## Agent基础概念

### 什么是Agent？

Agent是一个能够感知环境、做出决策并采取行动的智能体：

```
观察 (Observe) → 思考 (Think) → 行动 (Act) → 观察 ...
```

在GPT Agent中：
- **观察**：接收用户输入和工具返回结果
- **思考**：LLM进行推理和规划
- **行动**：调用工具或生成最终回复

### Agent架构

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def run(self, **kwargs) -> str:
        """执行工具"""
        raise NotImplementedError


@dataclass
class AgentAction:
    """Agent动作"""
    tool: str
    tool_input: Dict[str, Any]
    log: str  # 思考过程


@dataclass
class AgentFinish:
    """Agent完成"""
    return_values: Dict[str, Any]
    log: str


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    @abstractmethod
    def plan(self, input: str, intermediate_steps: List) -> AgentAction | AgentFinish:
        """规划下一步动作"""
        pass
    
    def run(self, input: str, max_iterations: int = 10) -> str:
        """执行Agent"""
        intermediate_steps = []
        
        for i in range(max_iterations):
            # 规划
            output = self.plan(input, intermediate_steps)
            
            # 检查是否完成
            if isinstance(output, AgentFinish):
                return output.return_values.get("output", "")
            
            # 执行动作
            tool = self.tools[output.tool]
            observation = tool.run(**output.tool_input)
            
            # 记录中间步骤
            intermediate_steps.append((output, observation))
        
        return "达到最大迭代次数"
```

## Function Calling

### OpenAI Function Calling

```python
import openai
import json
from typing import Callable

class FunctionCallingAgent:
    """基于Function Calling的Agent"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.functions = []
        self.function_map = {}
    
    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict,
        func: Callable
    ):
        """注册函数"""
        self.functions.append({
            "name": name,
            "description": description,
            "parameters": parameters
        })
        self.function_map[name] = func
    
    def run(self, user_message: str, max_iterations: int = 5) -> str:
        """运行Agent"""
        messages = [{"role": "user", "content": user_message}]
        
        for _ in range(max_iterations):
            # 调用LLM
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                functions=self.functions,
                function_call="auto"
            )
            
            message = response.choices[0].message
            
            # 检查是否需要调用函数
            if message.get("function_call"):
                # 执行函数
                func_name = message.function_call.name
                func_args = json.loads(message.function_call.arguments)
                
                func = self.function_map[func_name]
                result = func(**func_args)
                
                # 添加函数调用和结果到消息历史
                messages.append(message)
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": str(result)
                })
            else:
                # 返回最终回复
                return message.content
        
        return "达到最大迭代次数"


# 使用示例
agent = FunctionCallingAgent()

# 注册天气查询函数
agent.register_function(
    name="get_weather",
    description="获取指定城市的天气信息",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"]
    },
    func=lambda city: f"{city}今天晴，温度25°C"
)

# 注册计算函数
agent.register_function(
    name="calculate",
    description="执行数学计算",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4'"
            }
        },
        "required": ["expression"]
    },
    func=lambda expression: str(eval(expression))
)

# 运行
result = agent.run("北京今天天气怎么样？另外帮我算一下 15 * 23 + 7")
print(result)
```

### 工具定义最佳实践

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class SearchInput(BaseModel):
    """搜索工具输入"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")

class CodeExecutionInput(BaseModel):
    """代码执行工具输入"""
    code: str = Field(description="要执行的Python代码")
    timeout: int = Field(default=30, description="超时时间（秒）")

def create_tool_schema(model: BaseModel) -> Dict:
    """从Pydantic模型创建工具schema"""
    schema = model.schema()
    return {
        "type": "object",
        "properties": schema["properties"],
        "required": schema.get("required", [])
    }


# 工具定义装饰器
def tool(name: str, description: str):
    """工具装饰器"""
    def decorator(func):
        func._tool_name = name
        func._tool_description = description
        
        # 从函数签名推断参数
        import inspect
        sig = inspect.signature(func)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                # 简单类型映射
                type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
                param_type = type_map.get(param.annotation, "string")
                parameters["properties"][param_name] = {"type": param_type}
                
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)
        
        func._tool_parameters = parameters
        return func
    return decorator


# 使用装饰器
@tool("search_web", "在网上搜索信息")
def search_web(query: str, max_results: int = 5) -> str:
    """搜索网页"""
    # 实际实现...
    return f"搜索 '{query}' 的结果..."

@tool("send_email", "发送电子邮件")
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    return f"邮件已发送到 {to}"
```

## LangChain框架

### 基本使用

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool, tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import Type
from pydantic import BaseModel, Field

# 创建自定义工具
class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "用于数学计算。输入数学表达式，返回计算结果。"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)


# 使用装饰器创建工具
@tool
def search_wikipedia(query: str) -> str:
    """搜索Wikipedia获取信息。输入搜索词，返回相关内容摘要。"""
    # 实际实现应该调用Wikipedia API
    return f"Wikipedia关于'{query}'的信息: ..."


# 初始化Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [
    CalculatorTool(),
    search_wikipedia,
]

# 添加记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 运行
response = agent.run("帮我计算一下圆周率乘以100，然后查一下圆周率的历史")
print(response)
```

### 自定义Agent

```python
from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import ChatPromptTemplate
from typing import List, Tuple, Union

class CustomAgent(BaseSingleActionAgent):
    """自定义Agent实现"""
    
    llm: Any
    tools: List[BaseTool]
    prompt: ChatPromptTemplate
    
    @property
    def input_keys(self):
        return ["input"]
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步动作"""
        
        # 构建历史
        history = ""
        for action, observation in intermediate_steps:
            history += f"行动: {action.tool}[{action.tool_input}]\n"
            history += f"观察: {observation}\n"
        
        # 构建提示
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])
        
        messages = self.prompt.format_messages(
            tools=tools_desc,
            history=history,
            input=kwargs["input"]
        )
        
        # 调用LLM
        response = self.llm.predict_messages(messages)
        
        # 解析响应
        return self._parse_response(response.content)
    
    def _parse_response(self, response: str) -> Union[AgentAction, AgentFinish]:
        """解析LLM响应"""
        if "最终答案:" in response:
            answer = response.split("最终答案:")[-1].strip()
            return AgentFinish(
                return_values={"output": answer},
                log=response
            )
        
        # 解析工具调用
        import re
        match = re.search(r"行动: (\w+)\[(.*?)\]", response)
        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            return AgentAction(
                tool=tool_name,
                tool_input=tool_input,
                log=response
            )
        
        return AgentFinish(
            return_values={"output": response},
            log=response
        )
    
    async def aplan(self, intermediate_steps, **kwargs):
        return self.plan(intermediate_steps, **kwargs)
```

### Agent链和工作流

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

class AgentWorkflow:
    """Agent工作流"""
    
    def __init__(self, llm):
        self.llm = llm
        self.chains = {}
    
    def add_chain(self, name: str, prompt_template: str, output_key: str):
        """添加处理链"""
        prompt = PromptTemplate(
            input_variables=self._extract_variables(prompt_template),
            template=prompt_template
        )
        self.chains[name] = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_key=output_key
        )
    
    def _extract_variables(self, template: str) -> List[str]:
        """提取模板变量"""
        import re
        return re.findall(r'\{(\w+)\}', template)
    
    def build_sequential(self, chain_names: List[str]) -> SequentialChain:
        """构建顺序链"""
        chains = [self.chains[name] for name in chain_names]
        
        # 收集所有输入和输出变量
        input_vars = set()
        output_vars = []
        
        for chain in chains:
            input_vars.update(chain.input_keys)
            output_vars.append(chain.output_key)
        
        # 移除中间变量
        final_input_vars = input_vars - set(output_vars)
        
        return SequentialChain(
            chains=chains,
            input_variables=list(final_input_vars),
            output_variables=output_vars
        )


# 使用示例：文章生成工作流
workflow = AgentWorkflow(llm)

# 1. 大纲生成
workflow.add_chain(
    "outline",
    "根据主题'{topic}'生成文章大纲:\n",
    "outline"
)

# 2. 内容扩展
workflow.add_chain(
    "expand",
    "根据以下大纲扩展完整文章:\n{outline}\n\n完整文章:",
    "article"
)

# 3. 润色优化
workflow.add_chain(
    "polish",
    "请润色以下文章，改进语言表达:\n{article}\n\n润色后的文章:",
    "final_article"
)

# 构建并运行
pipeline = workflow.build_sequential(["outline", "expand", "polish"])
result = pipeline({"topic": "人工智能的未来"})
print(result["final_article"])
```

## 多Agent协作

### Agent团队

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

class Role(Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"

@dataclass
class Message:
    """Agent间的消息"""
    sender: str
    receiver: str
    content: str
    message_type: str = "normal"

class MultiAgentSystem:
    """多Agent系统"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents: Dict[str, 'SpecializedAgent'] = {}
        self.message_queue: List[Message] = []
        self.conversation_history: List[Message] = []
    
    def add_agent(self, agent: 'SpecializedAgent'):
        """添加Agent"""
        self.agents[agent.name] = agent
        agent.system = self
    
    def broadcast(self, sender: str, content: str):
        """广播消息给所有Agent"""
        for name in self.agents:
            if name != sender:
                self.send_message(sender, name, content)
    
    def send_message(self, sender: str, receiver: str, content: str):
        """发送消息"""
        message = Message(sender=sender, receiver=receiver, content=content)
        self.message_queue.append(message)
        self.conversation_history.append(message)
    
    def run(self, task: str, max_rounds: int = 10) -> str:
        """运行多Agent协作"""
        
        # 协调者分发任务
        coordinator = self.agents.get("coordinator")
        if coordinator:
            subtasks = coordinator.decompose_task(task)
            for agent_name, subtask in subtasks.items():
                self.send_message("coordinator", agent_name, subtask)
        else:
            # 没有协调者，直接广播任务
            self.broadcast("system", task)
        
        # 执行轮次
        for round_num in range(max_rounds):
            if not self.message_queue:
                break
            
            # 处理所有待处理消息
            current_messages = self.message_queue.copy()
            self.message_queue.clear()
            
            for message in current_messages:
                agent = self.agents.get(message.receiver)
                if agent:
                    response = agent.process_message(message)
                    if response:
                        self.send_message(
                            message.receiver,
                            response["receiver"],
                            response["content"]
                        )
        
        # 汇总结果
        return self._summarize_results()
    
    def _summarize_results(self) -> str:
        """汇总所有Agent的结果"""
        results = []
        for name, agent in self.agents.items():
            if hasattr(agent, 'result') and agent.result:
                results.append(f"[{name}]: {agent.result}")
        return "\n\n".join(results)


class SpecializedAgent:
    """专业化Agent"""
    
    def __init__(self, name: str, role: Role, llm, system_prompt: str):
        self.name = name
        self.role = role
        self.llm = llm
        self.system_prompt = system_prompt
        self.system: Optional[MultiAgentSystem] = None
        self.result: Optional[str] = None
        self.memory: List[Message] = []
    
    def process_message(self, message: Message) -> Optional[Dict]:
        """处理收到的消息"""
        self.memory.append(message)
        
        # 构建提示
        prompt = f"""
{self.system_prompt}

你收到了来自 {message.sender} 的消息:
{message.content}

请根据你的角色处理这个消息。如果需要与其他Agent协作，请明确说明。
"""
        
        response = self.llm.generate(prompt)
        
        # 解析响应，决定下一步动作
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """解析响应"""
        # 检查是否需要发送消息给其他Agent
        if "@" in response:
            # 简单的@mention解析
            import re
            match = re.search(r"@(\w+): (.+)", response)
            if match:
                return {
                    "receiver": match.group(1),
                    "content": match.group(2)
                }
        
        # 保存结果
        self.result = response
        return None
    
    def decompose_task(self, task: str) -> Dict[str, str]:
        """分解任务（协调者专用）"""
        prompt = f"""
作为协调者，请将以下任务分解为子任务，分配给不同的Agent。

可用的Agent:
- researcher: 负责信息收集和研究
- writer: 负责撰写内容
- reviewer: 负责审核和改进

任务: {task}

请输出JSON格式的任务分配:
{{"agent_name": "子任务描述", ...}}
"""
        response = self.llm.generate(prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return {"researcher": task}


# 使用示例
def create_writing_team(llm):
    """创建写作团队"""
    system = MultiAgentSystem(llm)
    
    # 协调者
    coordinator = SpecializedAgent(
        name="coordinator",
        role=Role.COORDINATOR,
        llm=llm,
        system_prompt="你是团队协调者，负责分配任务和协调工作。"
    )
    
    # 研究员
    researcher = SpecializedAgent(
        name="researcher",
        role=Role.RESEARCHER,
        llm=llm,
        system_prompt="你是研究员，负责收集和分析信息。输出结构化的研究笔记。"
    )
    
    # 作家
    writer = SpecializedAgent(
        name="writer",
        role=Role.WRITER,
        llm=llm,
        system_prompt="你是专业作家，根据研究材料撰写高质量内容。"
    )
    
    # 审核员
    reviewer = SpecializedAgent(
        name="reviewer",
        role=Role.REVIEWER,
        llm=llm,
        system_prompt="你是审核员，检查内容质量并提出改进建议。"
    )
    
    system.add_agent(coordinator)
    system.add_agent(researcher)
    system.add_agent(writer)
    system.add_agent(reviewer)
    
    return system

# 运行
team = create_writing_team(llm)
result = team.run("写一篇关于量子计算的科普文章")
```

## 实战项目：智能助手

### 完整的个人助手Agent

```python
import os
import json
from datetime import datetime
from typing import List, Dict, Any

class PersonalAssistant:
    """智能个人助手"""
    
    def __init__(self, llm, user_profile: Dict = None):
        self.llm = llm
        self.user_profile = user_profile or {}
        self.tools = self._initialize_tools()
        self.conversation_history = []
        self.context = {}
    
    def _initialize_tools(self) -> Dict[str, callable]:
        """初始化工具"""
        return {
            "search_web": self._search_web,
            "get_weather": self._get_weather,
            "manage_calendar": self._manage_calendar,
            "send_email": self._send_email,
            "take_notes": self._take_notes,
            "set_reminder": self._set_reminder,
            "calculate": self._calculate,
            "translate": self._translate,
        }
    
    def _get_tools_description(self) -> str:
        """获取工具描述"""
        descriptions = {
            "search_web": "搜索网页获取信息。参数: query (搜索词)",
            "get_weather": "获取天气信息。参数: city (城市名)",
            "manage_calendar": "管理日历。参数: action (add/list/delete), event (事件描述), time (时间)",
            "send_email": "发送邮件。参数: to (收件人), subject (主题), body (内容)",
            "take_notes": "记录笔记。参数: title (标题), content (内容)",
            "set_reminder": "设置提醒。参数: message (提醒内容), time (提醒时间)",
            "calculate": "数学计算。参数: expression (表达式)",
            "translate": "翻译文本。参数: text (原文), target_lang (目标语言)",
        }
        return "\n".join([f"- {k}: {v}" for k, v in descriptions.items()])
    
    def chat(self, user_input: str) -> str:
        """处理用户输入"""
        
        # 添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # 构建系统提示
        system_prompt = self._build_system_prompt()
        
        # 决定是否需要使用工具
        response = self._process_with_tools(user_input, system_prompt)
        
        # 添加到对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        user_info = ""
        if self.user_profile:
            user_info = f"\n用户信息: {json.dumps(self.user_profile, ensure_ascii=False)}"
        
        recent_history = self._get_recent_history(5)
        
        return f"""
你是一个智能个人助手，名叫小智。你的职责是帮助用户完成各种任务。

{user_info}

当前时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}

可用工具:
{self._get_tools_description()}

最近对话:
{recent_history}

回复指南:
1. 首先判断是否需要使用工具
2. 如果需要工具，输出: [使用工具: 工具名(参数)]
3. 根据工具结果或直接知识回答用户
4. 保持友好、专业的语气
5. 如果不确定，诚实地说明
"""
    
    def _get_recent_history(self, n: int = 5) -> str:
        """获取最近的对话历史"""
        recent = self.conversation_history[-n*2:] if len(self.conversation_history) > n*2 else self.conversation_history
        return "\n".join([f"{h['role']}: {h['content']}" for h in recent])
    
    def _process_with_tools(self, user_input: str, system_prompt: str) -> str:
        """处理输入，可能使用工具"""
        
        max_iterations = 5
        current_input = user_input
        tool_results = []
        
        for _ in range(max_iterations):
            # 调用LLM
            response = self.llm.generate(
                system_prompt + f"\n\n用户: {current_input}" + 
                (f"\n\n工具结果: {tool_results}" if tool_results else "")
            )
            
            # 检查是否需要使用工具
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                tool_name, params = tool_call
                if tool_name in self.tools:
                    result = self.tools[tool_name](**params)
                    tool_results.append({
                        "tool": tool_name,
                        "params": params,
                        "result": result
                    })
                    continue
            
            # 没有工具调用，返回响应
            return self._clean_response(response)
        
        return "抱歉，处理过程中遇到了问题。请稍后再试。"
    
    def _parse_tool_call(self, response: str) -> tuple:
        """解析工具调用"""
        import re
        match = re.search(r'\[使用工具: (\w+)\((.*?)\)\]', response)
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # 解析参数
            params = {}
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key.strip()] = value.strip().strip('"\'')
            
            return tool_name, params
        return None
    
    def _clean_response(self, response: str) -> str:
        """清理响应，移除工具调用标记"""
        import re
        return re.sub(r'\[使用工具:.*?\]', '', response).strip()
    
    # 工具实现
    def _search_web(self, query: str) -> str:
        """网页搜索"""
        # 实际应该调用搜索API
        return f"搜索'{query}'的结果: [模拟搜索结果]"
    
    def _get_weather(self, city: str) -> str:
        """获取天气"""
        # 实际应该调用天气API
        return f"{city}今天晴，温度22-28°C，空气质量良好"
    
    def _manage_calendar(self, action: str, event: str = "", time: str = "") -> str:
        """管理日历"""
        if action == "add":
            return f"已添加日程: {event} @ {time}"
        elif action == "list":
            return "今天的日程: 10:00 团队会议, 14:00 客户电话"
        return "操作完成"
    
    def _send_email(self, to: str, subject: str, body: str) -> str:
        """发送邮件"""
        return f"邮件已发送给 {to}"
    
    def _take_notes(self, title: str, content: str) -> str:
        """记录笔记"""
        return f"笔记已保存: {title}"
    
    def _set_reminder(self, message: str, time: str) -> str:
        """设置提醒"""
        return f"提醒已设置: {time} - {message}"
    
    def _calculate(self, expression: str) -> str:
        """计算"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except:
            return "计算错误"
    
    def _translate(self, text: str, target_lang: str) -> str:
        """翻译"""
        # 实际应该调用翻译API
        return f"翻译结果 ({target_lang}): [模拟翻译]"


# 使用示例
assistant = PersonalAssistant(
    llm=llm,
    user_profile={
        "name": "张三",
        "timezone": "Asia/Shanghai",
        "preferences": {"language": "zh-CN"}
    }
)

# 对话
print(assistant.chat("今天北京天气怎么样？"))
print(assistant.chat("帮我设置一个下午3点的会议提醒"))
print(assistant.chat("计算一下 15% 的小费，消费是 328 元"))
```

## 总结

本文详细介绍了GPT Agent的开发技术：

1. **Agent基础**：观察-思考-行动循环
2. **Function Calling**：OpenAI原生的工具调用能力
3. **LangChain**：流行的Agent开发框架
4. **多Agent协作**：团队协作和任务分解
5. **实战项目**：完整的智能助手实现

### Agent开发清单

- [ ] 明确Agent的能力边界
- [ ] 设计清晰的工具接口
- [ ] 实现可靠的错误处理
- [ ] 添加对话记忆和上下文管理
- [ ] 考虑安全性和权限控制
- [ ] 监控Agent的行为和性能

### 最佳实践

1. **工具设计**：工具功能要单一、明确
2. **提示工程**：清晰定义Agent的角色和约束
3. **错误处理**：优雅处理工具调用失败
4. **迭代限制**：防止无限循环
5. **日志记录**：记录Agent的决策过程

下一篇文章，我们将探讨GPT在实际应用中的案例和最佳实践，包括RAG系统、代码生成和企业应用等。
