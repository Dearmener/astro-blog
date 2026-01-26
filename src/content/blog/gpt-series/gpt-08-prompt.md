---
title: 'GPT完全指南（八）：Prompt Engineering实战'
description: '系统学习Prompt Engineering的核心技巧，包括Few-shot、Chain-of-Thought、ReAct等高级技术，以及结构化输出和实际应用案例'
pubDate: '2024-03-08'
heroImage: ''
category: '技术'
tags: ['GPT', 'Prompt Engineering', 'Few-shot', 'CoT', 'LLM']
series: 'GPT完全指南'
seriesOrder: 8
---

Prompt Engineering（提示工程）是与GPT等大语言模型高效交互的关键技能。一个好的prompt可以让模型输出质量提升数倍。本文将系统介绍Prompt Engineering的核心技巧，从基础到高级，帮助你充分发挥GPT的潜力。

## 基础原则

### 1. 清晰明确

```
❌ 不好的prompt:
"帮我写点东西"

✅ 好的prompt:
"请用专业但易懂的语言，写一篇500字左右的文章，介绍什么是机器学习，目标读者是没有技术背景的商业人士。"
```

### 2. 提供上下文

```
❌ 缺少上下文:
"翻译这段话"

✅ 提供上下文:
"你是一位专业的技术文档翻译。请将以下英文技术文档翻译成中文，保持技术术语的准确性，必要时在括号中保留英文原文。

原文：
Machine learning is a subset of artificial intelligence..."
```

### 3. 指定输出格式

```python
prompt = """
分析以下产品评论的情感倾向。

评论: "这个手机太棒了！电池续航很长，拍照效果也很好。唯一的缺点是价格有点贵。"

请按以下JSON格式输出：
{
    "overall_sentiment": "正面/负面/中性",
    "score": 0-100的数字,
    "positive_aspects": ["方面1", "方面2"],
    "negative_aspects": ["方面1"],
    "summary": "一句话总结"
}
"""
```

## 核心技术

### Zero-shot Prompting

直接给出任务描述，不提供示例：

```python
def zero_shot_classification(text, categories):
    prompt = f"""
将以下文本分类到最合适的类别中。

类别选项: {', '.join(categories)}

文本: "{text}"

只输出类别名称，不要其他内容。
"""
    return prompt

# 使用
categories = ["科技", "体育", "娱乐", "财经", "健康"]
text = "苹果公司今日发布了新款iPhone"
prompt = zero_shot_classification(text, categories)
# 输出: 科技
```

### Few-shot Prompting

提供少量示例，帮助模型理解任务：

```python
def few_shot_sentiment_analysis(text):
    prompt = """
判断以下评论的情感倾向（正面/负面/中性）。

示例1:
评论: "这家餐厅的菜品非常美味，服务也很周到！"
情感: 正面

示例2:
评论: "等了两个小时才上菜，太慢了。"
情感: 负面

示例3:
评论: "味道一般，价格适中。"
情感: 中性

现在请判断:
评论: "{text}"
情感: """
    return prompt

# Few-shot示例选择策略
def select_examples(query, example_pool, k=3):
    """
    选择与查询最相关的示例
    使用语义相似度或其他方法
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_embedding = model.encode([query])[0]
    example_embeddings = model.encode([e['text'] for e in example_pool])
    
    # 计算相似度
    similarities = np.dot(example_embeddings, query_embedding)
    
    # 选择top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return [example_pool[i] for i in top_k_indices]
```

### Chain-of-Thought (CoT)

引导模型逐步推理：

```python
# 标准CoT
def chain_of_thought_prompt(question):
    prompt = f"""
问题: {question}

让我们一步一步来思考这个问题：
"""
    return prompt

# Zero-shot CoT
def zero_shot_cot(question):
    prompt = f"""
问题: {question}

让我们一步一步来思考。
"""
    return prompt

# Few-shot CoT
def few_shot_cot_math():
    prompt = """
问题: 一个农场有23只鸡和17只鸭。农场主又买了15只鸡。现在农场总共有多少只家禽？

让我们一步一步思考：
1. 原来的鸡: 23只
2. 原来的鸭: 17只
3. 新买的鸡: 15只
4. 现在的鸡: 23 + 15 = 38只
5. 总家禽: 38 + 17 = 55只

答案: 55只

问题: 小明有120元。他花了35元买了一本书，又花了48元买了一个书包。他还剩多少钱？

让我们一步一步思考：
"""
    return prompt


# Self-Consistency CoT
def self_consistency_cot(question, num_samples=5):
    """
    多次采样，取多数票
    """
    import openai
    
    answers = []
    
    for _ in range(num_samples):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": chain_of_thought_prompt(question)}
            ],
            temperature=0.7  # 使用较高温度增加多样性
        )
        
        # 提取答案
        answer = extract_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # 多数票
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    
    return most_common
```

### ReAct (Reasoning + Acting)

结合推理和行动的框架：

```python
def react_prompt(question, tools_description):
    prompt = f"""
你可以使用以下工具来回答问题：

{tools_description}

请按以下格式回答问题：

Thought: 我需要思考如何解决这个问题
Action: 工具名称[参数]
Observation: 工具返回的结果
... (重复Thought/Action/Observation直到得到答案)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

问题: {question}

Thought: """
    return prompt


class ReActAgent:
    """ReAct代理实现"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tools_description = self._format_tools()
    
    def _format_tools(self):
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def run(self, question, max_steps=10):
        """执行ReAct循环"""
        
        prompt = react_prompt(question, self.tools_description)
        history = []
        
        for step in range(max_steps):
            # 获取LLM响应
            response = self.llm.generate(prompt + "".join(history))
            
            # 解析响应
            thought, action = self._parse_response(response)
            
            if action is None:
                # 到达最终答案
                return self._extract_final_answer(response)
            
            # 执行动作
            tool_name, tool_input = self._parse_action(action)
            observation = self.tools[tool_name].run(tool_input)
            
            # 更新历史
            history.append(f"\nThought: {thought}")
            history.append(f"\nAction: {action}")
            history.append(f"\nObservation: {observation}")
            history.append(f"\nThought: ")
        
        return "达到最大步数，未能得出答案"
    
    def _parse_response(self, response):
        """解析LLM响应"""
        lines = response.strip().split('\n')
        thought = ""
        action = None
        
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Final Answer:"):
                return thought, None
        
        return thought, action
    
    def _parse_action(self, action_str):
        """解析动作字符串"""
        import re
        match = re.match(r"(\w+)\[(.*)\]", action_str)
        if match:
            return match.group(1), match.group(2)
        return action_str, ""
    
    def _extract_final_answer(self, response):
        """提取最终答案"""
        for line in response.split('\n'):
            if line.startswith("Final Answer:"):
                return line[13:].strip()
        return response
```

### Tree of Thoughts (ToT)

探索多个推理路径：

```python
class TreeOfThoughts:
    """思维树实现"""
    
    def __init__(self, llm, evaluator):
        self.llm = llm
        self.evaluator = evaluator
    
    def solve(self, problem, num_branches=3, max_depth=3):
        """
        使用BFS探索思维树
        """
        from collections import deque
        
        # 初始状态
        initial_state = {"problem": problem, "thoughts": [], "depth": 0}
        queue = deque([initial_state])
        best_solution = None
        best_score = -float('inf')
        
        while queue:
            state = queue.popleft()
            
            if state["depth"] >= max_depth:
                # 评估最终状态
                score = self.evaluator.evaluate(state)
                if score > best_score:
                    best_score = score
                    best_solution = state
                continue
            
            # 生成候选思维
            candidates = self._generate_thoughts(state, num_branches)
            
            # 评估并筛选
            scored_candidates = []
            for candidate in candidates:
                score = self.evaluator.evaluate_intermediate(candidate)
                scored_candidates.append((score, candidate))
            
            # 选择top-k继续探索
            scored_candidates.sort(reverse=True)
            for score, candidate in scored_candidates[:num_branches]:
                if score > 0.3:  # 阈值过滤
                    new_state = {
                        "problem": problem,
                        "thoughts": state["thoughts"] + [candidate],
                        "depth": state["depth"] + 1
                    }
                    queue.append(new_state)
        
        return best_solution
    
    def _generate_thoughts(self, state, num_branches):
        """生成候选思维"""
        prompt = f"""
问题: {state['problem']}

当前思路:
{self._format_thoughts(state['thoughts'])}

请提出{num_branches}个不同的下一步思路，每个思路用[思路]标记：
"""
        
        response = self.llm.generate(prompt)
        thoughts = self._parse_thoughts(response)
        
        return thoughts[:num_branches]
    
    def _format_thoughts(self, thoughts):
        if not thoughts:
            return "（尚未开始）"
        return "\n".join([f"{i+1}. {t}" for i, t in enumerate(thoughts)])
    
    def _parse_thoughts(self, response):
        """解析生成的思路"""
        import re
        thoughts = re.findall(r'\[思路\](.*?)(?=\[思路\]|$)', response, re.DOTALL)
        return [t.strip() for t in thoughts if t.strip()]
```

## 高级技巧

### 角色扮演

```python
def expert_role_prompt(domain, question):
    roles = {
        "医学": "你是一位经验丰富的医学专家，拥有20年临床经验。",
        "法律": "你是一位资深律师，精通中国法律法规。",
        "金融": "你是一位华尔街资深分析师，专注于股票和投资分析。",
        "编程": "你是一位有15年经验的高级软件工程师，精通多种编程语言。"
    }
    
    role = roles.get(domain, "你是一位该领域的专家。")
    
    prompt = f"""
{role}

请根据你的专业知识回答以下问题。回答时：
1. 使用专业但易懂的语言
2. 如果问题超出你的专业范围，请明确说明
3. 如果信息可能过时，请提醒用户确认最新情况

问题: {question}
"""
    return prompt
```

### 结构化输出

```python
import json
from typing import List, Optional
from pydantic import BaseModel

class ProductReview(BaseModel):
    """产品评论分析结果"""
    sentiment: str  # positive, negative, neutral
    score: float  # 0-1
    key_points: List[str]
    suggestions: Optional[List[str]]

def structured_output_prompt(review_text):
    schema = ProductReview.schema_json(indent=2)
    
    prompt = f"""
分析以下产品评论，并按指定的JSON格式输出结果。

评论: "{review_text}"

输出格式必须严格遵循以下JSON Schema:
{schema}

请输出有效的JSON:
"""
    return prompt


# 使用OpenAI的Function Calling
def function_calling_example():
    import openai
    
    functions = [
        {
            "name": "analyze_sentiment",
            "description": "分析文本的情感倾向",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "情感倾向"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "置信度，0-1之间"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "关键情感词"
                    }
                },
                "required": ["sentiment", "confidence"]
            }
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "分析这条评论: '这个产品太棒了，完全超出预期！'"}
        ],
        functions=functions,
        function_call={"name": "analyze_sentiment"}
    )
    
    # 解析函数调用结果
    function_args = json.loads(
        response.choices[0].message.function_call.arguments
    )
    
    return function_args
```

### 自我一致性检查

```python
def self_consistency_check(question, num_samples=5):
    """
    通过多次采样和一致性检查提高可靠性
    """
    import openai
    from collections import Counter
    
    answers = []
    reasonings = []
    
    for i in range(num_samples):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"""
{question}

请一步一步思考，然后给出最终答案。
格式：
推理过程: ...
最终答案: ...
"""}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # 提取答案和推理
        answer = extract_final_answer(content)
        reasoning = extract_reasoning(content)
        
        answers.append(answer)
        reasonings.append(reasoning)
    
    # 统计答案
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]
    
    # 计算一致性
    consistency = count / num_samples
    
    return {
        "answer": most_common_answer,
        "consistency": consistency,
        "all_answers": answers,
        "all_reasonings": reasonings
    }
```

### Prompt模板管理

```python
from string import Template
from typing import Dict, Any
import yaml

class PromptTemplate:
    """Prompt模板管理"""
    
    def __init__(self, template: str, input_variables: list):
        self.template = Template(template)
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        # 检查必需变量
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return self.template.safe_substitute(**kwargs)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'PromptTemplate':
        """从YAML文件加载模板"""
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls(
            template=config['template'],
            input_variables=config.get('input_variables', [])
        )
    
    def save_yaml(self, path: str):
        """保存模板到YAML文件"""
        config = {
            'template': self.template.template,
            'input_variables': self.input_variables
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)


class PromptLibrary:
    """Prompt库管理"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
    
    def register(self, name: str, template: PromptTemplate):
        """注册模板"""
        self.templates[name] = template
    
    def get(self, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        return self.templates[name]
    
    def format(self, name: str, **kwargs) -> str:
        """格式化指定模板"""
        return self.get(name).format(**kwargs)
    
    def load_from_directory(self, directory: str):
        """从目录加载所有模板"""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith('.yaml'):
                name = filename[:-5]  # 移除.yaml后缀
                path = os.path.join(directory, filename)
                self.templates[name] = PromptTemplate.from_yaml(path)


# 使用示例
library = PromptLibrary()

# 注册模板
library.register("sentiment", PromptTemplate(
    template="""
分析以下文本的情感倾向。

文本: "$text"

情感（正面/负面/中性）:
""",
    input_variables=["text"]
))

library.register("summarize", PromptTemplate(
    template="""
请将以下文章总结为$length字以内的摘要。

文章:
$article

摘要:
""",
    input_variables=["article", "length"]
))

# 使用模板
prompt = library.format("sentiment", text="这个产品太棒了！")
```

## 实战案例

### 1. 代码生成与审查

```python
def code_generation_prompt(task, language, requirements=None):
    """代码生成prompt"""
    
    req_text = ""
    if requirements:
        req_text = f"\n要求:\n" + "\n".join([f"- {r}" for r in requirements])
    
    prompt = f"""
你是一位专业的{language}开发者。请根据以下需求编写代码。

任务: {task}
{req_text}

请提供：
1. 完整的、可运行的代码
2. 必要的注释
3. 简单的使用示例

代码:
```{language}
"""
    return prompt


def code_review_prompt(code, language):
    """代码审查prompt"""
    
    prompt = f"""
你是一位资深的{language}代码审查专家。请审查以下代码并提供反馈。

代码:
```{language}
{code}
```

请从以下方面进行审查：

1. **代码质量**: 
   - 可读性
   - 命名规范
   - 代码结构

2. **潜在问题**:
   - Bug风险
   - 安全漏洞
   - 性能问题

3. **改进建议**:
   - 具体的优化方案
   - 最佳实践建议

4. **总体评分**: (1-10分)

请按以上格式输出审查结果。
"""
    return prompt
```

### 2. 文档问答（RAG增强）

```python
def rag_qa_prompt(question, context_chunks):
    """RAG问答prompt"""
    
    context = "\n\n---\n\n".join([
        f"[来源 {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    prompt = f"""
基于以下参考资料回答用户的问题。

参考资料:
{context}

---

用户问题: {question}

回答要求:
1. 只使用参考资料中的信息回答
2. 如果参考资料不包含答案，请明确说明"根据提供的资料无法回答"
3. 引用信息时标注来源（如 [来源 1]）
4. 保持回答简洁、准确

回答:
"""
    return prompt


class RAGSystem:
    """简单的RAG系统"""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def answer(self, question, top_k=3):
        # 检索相关文档
        docs = self.retriever.search(question, top_k=top_k)
        
        # 构造prompt
        prompt = rag_qa_prompt(question, docs)
        
        # 生成回答
        response = self.llm.generate(prompt)
        
        return {
            "answer": response,
            "sources": docs
        }
```

### 3. 多轮对话管理

```python
class ConversationManager:
    """多轮对话管理"""
    
    def __init__(self, system_prompt, max_history=10):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history = []
    
    def add_message(self, role, content):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})
        
        # 保持历史长度
        if len(self.history) > self.max_history * 2:
            # 保留系统消息和最近的对话
            self.history = self.history[-self.max_history * 2:]
    
    def get_messages(self):
        """获取完整的消息列表"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        return messages
    
    def summarize_history(self, llm):
        """总结历史对话以节省token"""
        if len(self.history) < 6:
            return
        
        # 总结前面的对话
        old_history = self.history[:-4]
        
        summary_prompt = f"""
请总结以下对话的关键信息，保留重要的上下文：

{self._format_history(old_history)}

总结（100字以内）:
"""
        
        summary = llm.generate(summary_prompt)
        
        # 用总结替换旧历史
        self.history = [
            {"role": "system", "content": f"之前的对话总结: {summary}"}
        ] + self.history[-4:]
    
    def _format_history(self, history):
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in history
        ])


# 使用示例
conversation = ConversationManager(
    system_prompt="你是一个友好的AI助手，专门帮助用户解答编程问题。"
)

# 多轮对话
conversation.add_message("user", "Python如何读取文件？")
response1 = llm.chat(conversation.get_messages())
conversation.add_message("assistant", response1)

conversation.add_message("user", "如果文件很大怎么办？")
response2 = llm.chat(conversation.get_messages())
```

## 评估与优化

### Prompt评估框架

```python
class PromptEvaluator:
    """Prompt评估器"""
    
    def __init__(self, llm, test_cases):
        self.llm = llm
        self.test_cases = test_cases
    
    def evaluate(self, prompt_template, metrics=['accuracy', 'consistency']):
        """评估prompt性能"""
        results = {
            'total_cases': len(self.test_cases),
            'metrics': {}
        }
        
        predictions = []
        
        for case in self.test_cases:
            prompt = prompt_template.format(**case['input'])
            response = self.llm.generate(prompt)
            
            predictions.append({
                'input': case['input'],
                'expected': case.get('expected'),
                'actual': response
            })
        
        # 计算各项指标
        if 'accuracy' in metrics and all(p['expected'] for p in predictions):
            results['metrics']['accuracy'] = self._compute_accuracy(predictions)
        
        if 'consistency' in metrics:
            results['metrics']['consistency'] = self._compute_consistency(predictions)
        
        if 'format_compliance' in metrics:
            results['metrics']['format_compliance'] = self._check_format(predictions)
        
        results['predictions'] = predictions
        
        return results
    
    def _compute_accuracy(self, predictions):
        """计算准确率"""
        correct = sum(
            1 for p in predictions
            if self._normalize(p['expected']) == self._normalize(p['actual'])
        )
        return correct / len(predictions)
    
    def _compute_consistency(self, predictions):
        """计算一致性（多次运行相同输入）"""
        # 简化实现：假设每个case运行多次
        pass
    
    def _check_format(self, predictions):
        """检查输出格式合规性"""
        pass
    
    def _normalize(self, text):
        """归一化文本用于比较"""
        return text.strip().lower()


# A/B测试
def ab_test_prompts(prompt_a, prompt_b, test_cases, llm, num_runs=3):
    """A/B测试两个prompt"""
    
    evaluator = PromptEvaluator(llm, test_cases)
    
    results_a = []
    results_b = []
    
    for _ in range(num_runs):
        results_a.append(evaluator.evaluate(prompt_a))
        results_b.append(evaluator.evaluate(prompt_b))
    
    # 统计分析
    avg_accuracy_a = sum(r['metrics']['accuracy'] for r in results_a) / num_runs
    avg_accuracy_b = sum(r['metrics']['accuracy'] for r in results_b) / num_runs
    
    return {
        'prompt_a': {
            'avg_accuracy': avg_accuracy_a,
            'all_results': results_a
        },
        'prompt_b': {
            'avg_accuracy': avg_accuracy_b,
            'all_results': results_b
        },
        'winner': 'A' if avg_accuracy_a > avg_accuracy_b else 'B'
    }
```

## 总结

本文详细介绍了Prompt Engineering的核心技术：

1. **基础原则**：清晰明确、提供上下文、指定格式
2. **核心技术**：Zero-shot、Few-shot、Chain-of-Thought、ReAct
3. **高级技巧**：角色扮演、结构化输出、自我一致性检查
4. **实战应用**：代码生成、RAG问答、多轮对话
5. **评估优化**：评估框架、A/B测试

### Prompt Engineering清单

- [ ] 明确定义任务和期望输出
- [ ] 提供必要的上下文信息
- [ ] 使用Few-shot示例（如果需要）
- [ ] 对复杂问题使用CoT推理
- [ ] 指定输出格式（JSON、Markdown等）
- [ ] 设置合适的角色/身份
- [ ] 测试和迭代优化
- [ ] 监控实际效果

下一篇文章，我们将探讨如何构建GPT Agent，让AI能够使用工具并完成复杂任务。
