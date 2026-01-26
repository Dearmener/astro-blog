---
title: 'AI Agent 自动化实战'
description: '构建自主执行任务的 AI Agent，实现复杂业务流程的智能自动化。'
pubDate: 2025-05-30
category: '技术'
tags: ['AI自动化', 'Agent', 'AutoGPT', '工作流']
series: 'AI 自动化工作流'
seriesOrder: 7
heroImage: 'https://images.unsplash.com/photo-1531746790731-6c087fecd65a?w=800&h=400&fit=crop'
---

AI Agent 是 2025 年最热门的 AI 应用形态。与传统工作流不同，Agent 能够自主规划、执行和调整任务。本文将深入探讨 AI Agent 的构建和应用。

## Agent 基础概念

### Agent vs 传统工作流

| 特性 | 传统工作流 | AI Agent |
|------|------------|----------|
| 执行逻辑 | 预定义流程 | 自主决策 |
| 适应性 | 固定路径 | 动态调整 |
| 错误处理 | 预设分支 | 自我修正 |
| 任务范围 | 特定任务 | 开放目标 |
| 复杂度 | 线性/树状 | 循环/递归 |

### Agent 核心组件

```
┌─────────────────────────────────────────────────┐
│                  AI Agent                        │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────┐     ┌─────────────┐            │
│  │    LLM      │ ←── │   Memory    │            │
│  │   (大脑)    │     │   (记忆)    │            │
│  └──────┬──────┘     └─────────────┘            │
│         │                                        │
│         ▼                                        │
│  ┌─────────────┐     ┌─────────────┐            │
│  │  Planning   │ ←── │   Tools     │            │
│  │  (规划器)   │     │   (工具)    │            │
│  └──────┬──────┘     └─────────────┘            │
│         │                                        │
│         ▼                                        │
│  ┌─────────────┐                                │
│  │  Executor   │                                │
│  │  (执行器)   │                                │
│  └─────────────┘                                │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 构建 Agent

### 使用 LangChain Agent

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool, StructuredTool
from langchain import hub
from pydantic import BaseModel, Field

# 定义工具的输入 Schema
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")

class EmailInput(BaseModel):
    to: str = Field(description="收件人邮箱")
    subject: str = Field(description="邮件主题")
    body: str = Field(description="邮件正文")

# 定义工具
async def search_web(query: str, max_results: int = 5) -> str:
    """在网上搜索信息"""
    # 实现搜索逻辑
    return f"找到 {max_results} 条关于 '{query}' 的结果..."

async def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    # 实现邮件发送
    return f"已发送邮件到 {to}"

async def create_calendar_event(title: str, time: str, attendees: list) -> str:
    """创建日历事件"""
    return f"已创建事件: {title} at {time}"

async def query_database(sql: str) -> str:
    """执行数据库查询"""
    # 安全检查和执行
    return "查询结果: ..."

# 创建工具列表
tools = [
    StructuredTool.from_function(
        func=search_web,
        name="search",
        description="搜索网页信息",
        args_schema=SearchInput
    ),
    StructuredTool.from_function(
        func=send_email,
        name="email",
        description="发送邮件",
        args_schema=EmailInput
    ),
    Tool(
        name="calendar",
        func=create_calendar_event,
        description="创建日历事件"
    ),
    Tool(
        name="database",
        func=query_database,
        description="查询数据库"
    ),
]

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    return_intermediate_steps=True
)

# 执行复杂任务
result = await agent_executor.ainvoke({
    "input": """
    帮我完成以下任务：
    1. 搜索下周北京的天气预报
    2. 如果有雨，给团队发邮件提醒带伞
    3. 创建一个下周三上午 10 点的户外团建会议
    """
})
```

### 带记忆的 Agent

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# 使用 Redis 持久化记忆
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    session_id="agent-session-001"
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    chat_memory=message_history,
    return_messages=True,
    k=10  # 保留最近 10 轮对话
)

# 创建带记忆的 Agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Agent 会记住之前的对话
result1 = await agent_executor.ainvoke({"input": "我叫张三，我是产品经理"})
result2 = await agent_executor.ainvoke({"input": "我的名字是什么？我做什么工作？"})
```

## 高级 Agent 模式

### ReAct Agent (推理+行动)

```python
from langchain.agents import create_react_agent

react_prompt = """
你是一个智能助手，可以使用工具来完成任务。

可用工具:
{tools}

使用以下格式：

思考：我需要思考如何完成这个任务
行动：要使用的工具名称
行动输入：工具的输入
观察：工具返回的结果
...（可以重复多次思考/行动/观察）
思考：我现在知道最终答案了
最终答案：给用户的最终回复

开始！

问题：{input}
{agent_scratchpad}
"""

react_agent = create_react_agent(llm, tools, react_prompt)
```

### Plan-and-Execute Agent

```python
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)

# 创建规划器
planner = load_chat_planner(llm)

# 创建执行器
executor = load_agent_executor(llm, tools, verbose=True)

# 组合
plan_execute_agent = PlanAndExecute(
    planner=planner,
    executor=executor
)

# 执行复杂任务
result = plan_execute_agent.run("""
创建一份市场分析报告：
1. 收集竞争对手信息
2. 分析市场趋势
3. 生成报告并发送给管理层
""")
```

### Multi-Agent 协作

```python
from langchain.agents import AgentExecutor
from typing import List, Dict

class MultiAgentSystem:
    """多 Agent 协作系统"""
    
    def __init__(self):
        self.agents: Dict[str, AgentExecutor] = {}
        self.coordinator = self._create_coordinator()
    
    def register_agent(self, name: str, agent: AgentExecutor, description: str):
        """注册专业 Agent"""
        self.agents[name] = {
            "executor": agent,
            "description": description
        }
    
    def _create_coordinator(self):
        """创建协调 Agent"""
        coordinator_prompt = """
        你是一个任务协调员，负责将复杂任务分配给专业 Agent。
        
        可用的专业 Agent：
        {agent_descriptions}
        
        分析用户任务，决定：
        1. 需要哪些 Agent 参与
        2. 执行顺序
        3. 如何整合结果
        
        任务：{task}
        """
        return ...  # 实现协调逻辑
    
    async def execute(self, task: str) -> str:
        """执行多 Agent 任务"""
        # 1. 协调器分析任务
        plan = await self.coordinator.analyze(task)
        
        # 2. 按顺序执行各 Agent
        results = []
        for step in plan.steps:
            agent = self.agents[step.agent_name]
            result = await agent["executor"].ainvoke({
                "input": step.task,
                "context": results  # 传递之前的结果
            })
            results.append(result)
        
        # 3. 整合结果
        final = await self.coordinator.synthesize(results)
        return final

# 使用
system = MultiAgentSystem()

# 注册专业 Agent
system.register_agent(
    "researcher",
    researcher_agent,
    "负责信息搜索和研究"
)
system.register_agent(
    "analyst",
    analyst_agent,
    "负责数据分析和洞察"
)
system.register_agent(
    "writer",
    writer_agent,
    "负责内容撰写和报告"
)

# 执行复杂任务
result = await system.execute("""
创建一份 2025 年 AI 行业分析报告，包括：
- 市场规模和增长趋势
- 主要玩家分析
- 技术发展方向
- 投资建议
""")
```

## 实战案例

### 自动化销售助手

```python
class SalesAgent:
    """自动化销售助手"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
    
    def _setup_tools(self):
        return [
            Tool(
                name="search_leads",
                func=self.search_leads,
                description="搜索潜在客户"
            ),
            Tool(
                name="get_company_info",
                func=self.get_company_info,
                description="获取公司详细信息"
            ),
            Tool(
                name="draft_email",
                func=self.draft_email,
                description="起草销售邮件"
            ),
            Tool(
                name="schedule_meeting",
                func=self.schedule_meeting,
                description="安排会议"
            ),
            Tool(
                name="update_crm",
                func=self.update_crm,
                description="更新 CRM 记录"
            ),
        ]
    
    async def prospect(self, criteria: str):
        """自动寻找潜在客户"""
        return await self.agent.ainvoke({
            "input": f"""
            根据以下条件寻找潜在客户并进行初步接触：
            
            目标客户标准：{criteria}
            
            执行步骤：
            1. 搜索符合条件的公司
            2. 获取关键决策者信息
            3. 研究公司近期动态
            4. 起草个性化的开发信
            5. 更新 CRM 记录
            """
        })

# 使用
sales_agent = SalesAgent()
result = await sales_agent.prospect("""
- 行业：科技/金融
- 规模：100-1000人
- 地区：北京、上海
- 正在进行数字化转型
""")
```

### 自动化数据分析师

```python
class DataAnalystAgent:
    """自动化数据分析师"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.tools = [
            Tool(
                name="query_database",
                func=self.safe_query,
                description="执行 SQL 查询"
            ),
            Tool(
                name="create_chart",
                func=self.create_chart,
                description="创建图表"
            ),
            Tool(
                name="statistical_analysis",
                func=self.run_statistics,
                description="运行统计分析"
            ),
            Tool(
                name="generate_report",
                func=self.generate_report,
                description="生成分析报告"
            ),
        ]
    
    async def analyze(self, question: str):
        """分析数据并回答问题"""
        return await self.agent.ainvoke({
            "input": f"""
            作为数据分析师，请分析以下问题：
            
            {question}
            
            要求：
            1. 首先理解问题需要什么数据
            2. 编写并执行必要的 SQL 查询
            3. 对结果进行统计分析
            4. 创建可视化图表
            5. 生成分析报告和建议
            """
        })
```

## 安全考虑

```python
class SafeAgent:
    """安全增强的 Agent"""
    
    # 工具白名单
    ALLOWED_TOOLS = ["search", "email", "calendar"]
    
    # 敏感操作需要确认
    CONFIRM_REQUIRED = ["delete", "payment", "admin"]
    
    async def execute_with_safety(self, task: str):
        """带安全检查的执行"""
        
        # 1. 输入净化
        sanitized_task = self.sanitize_input(task)
        
        # 2. 意图分析
        intent = await self.analyze_intent(sanitized_task)
        
        # 3. 风险评估
        risk_level = self.assess_risk(intent)
        
        if risk_level == "high":
            # 需要人工确认
            return {
                "status": "pending_approval",
                "task": sanitized_task,
                "risk_factors": intent.risk_factors
            }
        
        # 4. 执行限制
        result = await self.agent.ainvoke(
            {"input": sanitized_task},
            config={
                "max_iterations": 5,
                "timeout": 60,
                "allowed_tools": self.ALLOWED_TOOLS
            }
        )
        
        # 5. 输出审计
        self.audit_log(task, result)
        
        return result
```

## 总结

AI Agent 自动化的关键：

1. **明确目标** - 清晰定义 Agent 要完成的任务
2. **合适工具** - 提供必要且充足的工具
3. **安全边界** - 设置适当的限制和审计
4. **人机协作** - 关键决策保留人工确认

下一篇，我们将总结企业级 AI 工作流最佳实践。
