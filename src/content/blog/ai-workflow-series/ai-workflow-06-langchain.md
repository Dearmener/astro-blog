---
title: 'LangChain 工作流构建'
description: '使用 LangChain 构建复杂的 AI 工作流，实现灵活的 Agent 和 Chain 编排。'
pubDate: 2025-05-20
category: '技术'
tags: ['AI自动化', 'LangChain', 'Python', 'Agent']
series: 'AI 自动化工作流'
seriesOrder: 6
heroImage: 'https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=800&h=400&fit=crop'
---

LangChain 是最流行的 LLM 应用开发框架，提供了构建复杂 AI 工作流所需的所有工具。本文将介绍如何用 LangChain 构建生产级工作流。

## LangChain 基础

### 安装配置

```bash
# 安装核心包
pip install langchain langchain-openai langchain-community

# 可选：向量数据库
pip install chromadb pinecone-client

# 可选：文档加载
pip install pypdf unstructured
```

### 核心概念

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# 1. Model - 语言模型
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Prompt Template - 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}"),
    ("human", "{question}")
])

# 3. Chain - 链式调用
chain = prompt | llm

# 4. 执行
result = chain.invoke({
    "role": "数据分析师",
    "question": "分析这份销售数据的趋势"
})
```

## LCEL (LangChain Expression Language)

### 链式组合

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 基础链
basic_chain = prompt | llm | StrOutputParser()

# 并行执行
parallel_chain = RunnableParallel(
    summary=prompt | llm | StrOutputParser(),
    keywords=keyword_prompt | llm | StrOutputParser(),
)

# 条件分支
def route(input):
    if input["type"] == "technical":
        return technical_chain
    else:
        return general_chain

branching_chain = RunnableLambda(route)

# 组合复杂工作流
workflow = (
    RunnablePassthrough.assign(
        context=retriever | format_docs
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

### 流式输出

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 流式 LLM
streaming_llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 异步流式
async for chunk in chain.astream({"question": "解释量子计算"}):
    print(chunk.content, end="", flush=True)
```

## RAG 工作流

### 完整 RAG 实现

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = PyPDFLoader("knowledge_base.pdf")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "，", " "]
)
splits = text_splitter.split_documents(documents)

# 3. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 多样性检索
    search_kwargs={"k": 5, "fetch_k": 10}
)

# 5. RAG 提示词
rag_prompt = ChatPromptTemplate.from_template("""
基于以下上下文回答问题。如果上下文中没有相关信息，请诚实说明。

上下文：
{context}

问题：{question}

回答：
""")

# 6. 格式化函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 7. 构建 RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("什么是机器学习？")
```

### 高级 RAG 策略

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_cohere import CohereRerank

# 1. 压缩检索 - 提取相关部分
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 2. Rerank - 重排序
reranker = CohereRerank(top_n=3)
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)

# 3. 多查询检索 - 扩展查询
from langchain.retrievers import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# 4. 父文档检索 - 保持上下文
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=text_splitter,
)
```

## Agent 构建

### ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

# 定义工具
def search_web(query: str) -> str:
    """搜索网页获取信息"""
    # 实现搜索逻辑
    return f"搜索结果：{query}"

def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

def get_weather(city: str) -> str:
    """获取天气信息"""
    # 调用天气 API
    return f"{city}的天气：晴，25°C"

tools = [
    Tool(name="Search", func=search_web, description="搜索网页信息"),
    Tool(name="Calculator", func=calculate, description="数学计算"),
    Tool(name="Weather", func=get_weather, description="查询天气"),
]

# 获取 ReAct 提示词
prompt = hub.pull("hwchase17/react")

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# 执行
result = agent_executor.invoke({
    "input": "北京今天天气怎么样？如果温度超过30度，计算需要喝多少水（每度0.1升）"
})
```

### 自定义 Agent

```python
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str):
        # 检查是否完成
        if "最终答案:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("最终答案:")[-1].strip()},
                log=llm_output,
            )
        
        # 解析动作
        match = re.search(r"动作:\s*(.*?)\n动作输入:\s*(.*)", llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        
        raise ValueError(f"无法解析输出: {llm_output}")
```

## 生产级工作流

### 完整业务工作流示例

```python
from langchain.callbacks import get_openai_callback
from langchain_core.runnables import RunnableConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerServiceWorkflow:
    """智能客服工作流"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.classifier = self._create_classifier()
        self.responder = self._create_responder()
        self.escalation_handler = self._create_escalation_handler()
    
    def _create_classifier(self):
        """创建意图分类器"""
        prompt = ChatPromptTemplate.from_template("""
        分析以下客户消息，返回 JSON：
        {{
            "intent": "inquiry|complaint|support|other",
            "urgency": "low|medium|high",
            "sentiment": "positive|neutral|negative",
            "topic": "产品|物流|售后|其他"
        }}
        
        消息：{message}
        """)
        return prompt | self.llm | JsonOutputParser()
    
    def _create_responder(self):
        """创建响应生成器"""
        prompt = ChatPromptTemplate.from_template("""
        你是专业的客服代表。
        
        客户消息：{message}
        意图分析：{classification}
        相关知识：{context}
        
        请生成友好、专业的回复。
        """)
        return prompt | self.llm | StrOutputParser()
    
    def _create_escalation_handler(self):
        """创建升级处理器"""
        prompt = ChatPromptTemplate.from_template("""
        此问题需要人工处理。
        
        客户消息：{message}
        分析结果：{classification}
        
        生成工单摘要，包含：
        - 问题类型
        - 紧急程度
        - 建议处理方式
        """)
        return prompt | self.llm | StrOutputParser()
    
    async def process(self, message: str, config: RunnableConfig = None):
        """处理客户消息"""
        try:
            with get_openai_callback() as cb:
                # 1. 分类
                classification = await self.classifier.ainvoke(
                    {"message": message},
                    config=config
                )
                logger.info(f"分类结果: {classification}")
                
                # 2. 检查是否需要升级
                if (classification.get("urgency") == "high" or 
                    classification.get("sentiment") == "negative"):
                    
                    escalation = await self.escalation_handler.ainvoke({
                        "message": message,
                        "classification": classification
                    }, config=config)
                    
                    return {
                        "type": "escalation",
                        "response": "您的问题已转交专员处理，请稍候。",
                        "ticket": escalation,
                        "tokens_used": cb.total_tokens
                    }
                
                # 3. 检索知识库
                context = await self.retriever.ainvoke(message)
                
                # 4. 生成回复
                response = await self.responder.ainvoke({
                    "message": message,
                    "classification": classification,
                    "context": format_docs(context)
                }, config=config)
                
                return {
                    "type": "auto_response",
                    "response": response,
                    "classification": classification,
                    "tokens_used": cb.total_tokens
                }
                
        except Exception as e:
            logger.error(f"处理失败: {e}")
            return {
                "type": "error",
                "response": "抱歉，系统繁忙，请稍后再试。",
                "error": str(e)
            }

# 使用
workflow = CustomerServiceWorkflow()
result = await workflow.process("我的订单三天了还没收到，太慢了！")
```

### 监控和追踪

```python
from langchain.callbacks import LangChainTracer
from langsmith import Client

# LangSmith 追踪
client = Client()
tracer = LangChainTracer(project_name="customer-service")

# 在调用时使用
result = chain.invoke(
    {"question": "..."},
    config={"callbacks": [tracer]}
)

# 自定义指标
from langchain.callbacks.base import BaseCallbackHandler

class MetricsCallback(BaseCallbackHandler):
    def __init__(self):
        self.tokens = 0
        self.latency = 0
    
    def on_llm_end(self, response, **kwargs):
        self.tokens += response.llm_output.get("token_usage", {}).get("total_tokens", 0)
```

## 总结

LangChain 工作流的优势：

1. **灵活性最高** - 代码级控制
2. **功能完整** - 覆盖所有 AI 场景
3. **生态丰富** - 集成众多工具
4. **生产就绪** - 监控、追踪完善

下一篇，我们将探索 AI Agent 自动化实战。
