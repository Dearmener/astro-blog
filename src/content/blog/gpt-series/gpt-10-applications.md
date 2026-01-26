---
title: 'GPT完全指南（十）：应用案例与最佳实践'
description: '探索GPT在实际应用中的案例，包括RAG系统、代码助手、内容创作、企业应用等，以及生产环境部署的最佳实践'
pubDate: '2024-03-10'
heroImage: ''
category: '技术'
tags: ['GPT', 'RAG', '应用开发', '最佳实践', 'LLM']
series: 'GPT完全指南'
seriesOrder: 10
---

经过前面九篇文章的学习，我们已经掌握了GPT的原理和技术。本文将聚焦于实际应用，探讨如何在生产环境中构建基于GPT的应用，分享成功案例和最佳实践。

## RAG系统

### 什么是RAG？

RAG（Retrieval-Augmented Generation）通过检索外部知识来增强GPT的回答能力：

```
用户问题 → 检索相关文档 → 构造增强提示 → GPT生成回答
```

### 完整RAG实现

```python
import os
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    """文档类"""
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class RAGSystem:
    """完整的RAG系统实现"""
    
    def __init__(
        self,
        embedding_model,
        llm,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        self.documents: List[Document] = []
        self.embeddings_index = None
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """添加文档到知识库"""
        if metadata is None:
            metadata = [{}] * len(texts)
        
        for text, meta in zip(texts, metadata):
            # 分块
            chunks = self._split_text(text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    metadata={**meta, "chunk_index": i}
                )
                self.documents.append(doc)
        
        # 计算嵌入
        self._build_index()
    
    def _split_text(self, text: str) -> List[str]:
        """文本分块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在句子边界切分
            if end < len(text):
                # 向后找句号
                for i in range(min(50, len(text) - end)):
                    if text[end + i] in '.。！？!?\n':
                        end = end + i + 1
                        break
            
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
        
        return chunks
    
    def _build_index(self):
        """构建向量索引"""
        texts = [doc.content for doc in self.documents]
        embeddings = self.embedding_model.encode(texts)
        
        for doc, emb in zip(self.documents, embeddings):
            doc.embedding = emb
        
        # 构建简单的向量索引（实际应用可用FAISS等）
        self.embeddings_index = np.array(embeddings)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """检索相关文档"""
        if top_k is None:
            top_k = self.top_k
        
        # 计算查询嵌入
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 计算相似度
        similarities = np.dot(self.embeddings_index, query_embedding)
        
        # 获取top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def generate(self, query: str, system_prompt: str = None) -> Dict[str, Any]:
        """生成回答"""
        # 检索
        relevant_docs = self.retrieve(query)
        
        # 构造上下文
        context = "\n\n---\n\n".join([
            f"[文档 {i+1}]\n{doc.content}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        # 构造提示
        if system_prompt is None:
            system_prompt = """你是一个知识助手。请根据提供的参考资料回答用户问题。
规则:
1. 只使用参考资料中的信息
2. 如果资料不足以回答，请说明
3. 引用时注明来源"""
        
        prompt = f"""
{system_prompt}

参考资料:
{context}

用户问题: {query}

回答:
"""
        
        # 生成
        response = self.llm.generate(prompt)
        
        return {
            "answer": response,
            "sources": relevant_docs,
            "query": query
        }


# 高级RAG技术
class HybridRAG(RAGSystem):
    """混合检索RAG"""
    
    def __init__(self, *args, bm25_weight: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_weight = bm25_weight
        self.bm25_index = None
    
    def _build_index(self):
        """构建混合索引"""
        super()._build_index()
        
        # 构建BM25索引
        from rank_bm25 import BM25Okapi
        
        tokenized_docs = [doc.content.split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """混合检索"""
        if top_k is None:
            top_k = self.top_k
        
        # 向量检索得分
        query_embedding = self.embedding_model.encode([query])[0]
        vector_scores = np.dot(self.embeddings_index, query_embedding)
        vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
        
        # BM25得分
        bm25_scores = self.bm25_index.get_scores(query.split())
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        # 混合得分
        hybrid_scores = (1 - self.bm25_weight) * vector_scores + self.bm25_weight * bm25_scores
        
        # 获取top-k
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]


class ReRankRAG(RAGSystem):
    """带重排序的RAG"""
    
    def __init__(self, *args, reranker, rerank_top_k: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """检索并重排序"""
        if top_k is None:
            top_k = self.top_k
        
        # 首先获取更多候选
        candidates = super().retrieve(query, top_k=self.rerank_top_k)
        
        # 重排序
        pairs = [(query, doc.content) for doc in candidates]
        rerank_scores = self.reranker.compute_score(pairs)
        
        # 按重排序分数排序
        sorted_indices = np.argsort(rerank_scores)[::-1][:top_k]
        
        return [candidates[i] for i in sorted_indices]
```

### RAG评估

```python
class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, rag_system, eval_llm=None):
        self.rag = rag_system
        self.eval_llm = eval_llm  # 用于自动评估的LLM
    
    def evaluate(self, test_cases: List[Dict]) -> Dict[str, float]:
        """评估RAG系统"""
        results = {
            "retrieval_precision": [],
            "answer_relevance": [],
            "faithfulness": [],
            "answer_correctness": []
        }
        
        for case in test_cases:
            query = case["question"]
            expected_answer = case.get("answer")
            relevant_doc_ids = case.get("relevant_docs", [])
            
            # 获取RAG回答
            response = self.rag.generate(query)
            
            # 评估检索精度
            if relevant_doc_ids:
                retrieved_ids = [doc.metadata.get("id") for doc in response["sources"]]
                precision = len(set(retrieved_ids) & set(relevant_doc_ids)) / len(retrieved_ids)
                results["retrieval_precision"].append(precision)
            
            # 评估答案相关性（使用LLM）
            if self.eval_llm:
                relevance = self._evaluate_relevance(query, response["answer"])
                results["answer_relevance"].append(relevance)
                
                # 评估忠实度（答案是否基于检索的文档）
                faithfulness = self._evaluate_faithfulness(
                    response["answer"],
                    [doc.content for doc in response["sources"]]
                )
                results["faithfulness"].append(faithfulness)
            
            # 评估答案正确性
            if expected_answer:
                correctness = self._evaluate_correctness(
                    response["answer"],
                    expected_answer
                )
                results["answer_correctness"].append(correctness)
        
        # 计算平均分
        return {k: np.mean(v) if v else None for k, v in results.items()}
    
    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """评估答案与问题的相关性"""
        prompt = f"""
评估以下答案与问题的相关程度。

问题: {question}
答案: {answer}

请给出1-5的评分:
1: 完全不相关
2: 略微相关
3: 部分相关
4: 大部分相关
5: 完全相关

只输出数字:
"""
        score = self.eval_llm.generate(prompt).strip()
        try:
            return int(score) / 5.0
        except:
            return 0.5
    
    def _evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """评估答案是否忠实于上下文"""
        context_str = "\n".join(contexts)
        prompt = f"""
评估答案是否完全基于提供的上下文信息。

上下文:
{context_str}

答案: {answer}

评分标准:
1: 答案包含上下文中没有的信息（幻觉）
3: 答案部分基于上下文
5: 答案完全基于上下文

只输出数字:
"""
        score = self.eval_llm.generate(prompt).strip()
        try:
            return int(score) / 5.0
        except:
            return 0.5
    
    def _evaluate_correctness(self, generated: str, expected: str) -> float:
        """评估答案正确性"""
        prompt = f"""
比较生成的答案与标准答案的一致性。

标准答案: {expected}
生成答案: {generated}

评分:
1: 完全不一致
3: 部分一致
5: 完全一致

只输出数字:
"""
        score = self.eval_llm.generate(prompt).strip()
        try:
            return int(score) / 5.0
        except:
            return 0.5
```

## 代码助手

### 智能代码生成

```python
class CodeAssistant:
    """代码助手"""
    
    def __init__(self, llm, supported_languages: List[str] = None):
        self.llm = llm
        self.supported_languages = supported_languages or [
            "python", "javascript", "typescript", "java", "go", "rust"
        ]
        self.conversation_history = []
    
    def generate_code(
        self,
        task: str,
        language: str,
        context: str = None,
        style_guide: str = None
    ) -> Dict[str, str]:
        """生成代码"""
        
        prompt = f"""
你是一位专业的{language}开发者。请根据需求编写高质量代码。

任务描述: {task}
"""
        
        if context:
            prompt += f"\n现有代码上下文:\n```{language}\n{context}\n```\n"
        
        if style_guide:
            prompt += f"\n代码风格指南: {style_guide}\n"
        
        prompt += f"""
请提供:
1. 完整的、可运行的代码
2. 详细的注释
3. 使用示例
4. 可能的边缘情况处理

代码:
```{language}
"""
        
        response = self.llm.generate(prompt)
        
        # 解析响应
        code = self._extract_code(response, language)
        explanation = self._extract_explanation(response)
        
        return {
            "code": code,
            "explanation": explanation,
            "language": language
        }
    
    def explain_code(self, code: str, language: str, level: str = "intermediate") -> str:
        """解释代码"""
        
        level_prompts = {
            "beginner": "像解释给编程新手一样",
            "intermediate": "像解释给有一定经验的开发者一样",
            "expert": "像解释给高级开发者一样，包括性能和设计模式分析"
        }
        
        prompt = f"""
{level_prompts.get(level, level_prompts["intermediate"])}解释以下{language}代码:

```{language}
{code}
```

请解释:
1. 代码的整体功能
2. 关键部分的作用
3. 使用的算法或设计模式
4. 潜在的改进点
"""
        
        return self.llm.generate(prompt)
    
    def review_code(self, code: str, language: str) -> Dict[str, Any]:
        """代码审查"""
        
        prompt = f"""
请对以下{language}代码进行专业的代码审查:

```{language}
{code}
```

请从以下方面进行分析，并以JSON格式输出:

{{
    "overall_score": 1-10的评分,
    "issues": [
        {{
            "severity": "critical/major/minor/suggestion",
            "line": 行号（如果适用）,
            "description": "问题描述",
            "suggestion": "改进建议"
        }}
    ],
    "positive_aspects": ["优点1", "优点2"],
    "security_concerns": ["安全问题（如果有）"],
    "performance_issues": ["性能问题（如果有）"],
    "refactoring_suggestions": ["重构建议"],
    "summary": "总结评价"
}}
"""
        
        response = self.llm.generate(prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return {"raw_response": response}
    
    def fix_bug(self, code: str, error_message: str, language: str) -> Dict[str, str]:
        """修复Bug"""
        
        prompt = f"""
以下{language}代码出现了错误，请帮助修复:

代码:
```{language}
{code}
```

错误信息:
```
{error_message}
```

请:
1. 分析错误原因
2. 提供修复后的完整代码
3. 解释修复的内容

分析:
"""
        
        response = self.llm.generate(prompt)
        
        fixed_code = self._extract_code(response, language)
        
        return {
            "analysis": response,
            "fixed_code": fixed_code
        }
    
    def refactor_code(
        self,
        code: str,
        language: str,
        goals: List[str] = None
    ) -> Dict[str, str]:
        """重构代码"""
        
        goals = goals or ["提高可读性", "改善性能", "增强可维护性"]
        goals_str = "\n".join([f"- {g}" for g in goals])
        
        prompt = f"""
请重构以下{language}代码，目标是:
{goals_str}

原始代码:
```{language}
{code}
```

请提供:
1. 重构后的代码
2. 重构说明（做了哪些改动，为什么）
3. 重构前后的对比

重构后的代码:
```{language}
"""
        
        response = self.llm.generate(prompt)
        refactored_code = self._extract_code(response, language)
        
        return {
            "original_code": code,
            "refactored_code": refactored_code,
            "explanation": response
        }
    
    def generate_tests(self, code: str, language: str, framework: str = None) -> str:
        """生成测试用例"""
        
        framework_hint = f"使用{framework}测试框架" if framework else ""
        
        prompt = f"""
为以下{language}代码生成全面的单元测试{framework_hint}:

```{language}
{code}
```

请生成:
1. 正常情况的测试
2. 边缘情况的测试
3. 错误处理的测试
4. 必要的mock和fixture

测试代码:
```{language}
"""
        
        return self._extract_code(self.llm.generate(prompt), language)
    
    def _extract_code(self, response: str, language: str) -> str:
        """从响应中提取代码"""
        import re
        pattern = rf'```{language}\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # 尝试通用代码块
        pattern = r'```\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return response
    
    def _extract_explanation(self, response: str) -> str:
        """提取解释部分"""
        import re
        # 移除代码块
        explanation = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        return explanation.strip()
```

## 内容创作平台

### 智能写作助手

```python
class ContentCreator:
    """内容创作助手"""
    
    def __init__(self, llm):
        self.llm = llm
        self.style_profiles = {
            "professional": "专业、正式、客观",
            "casual": "轻松、友好、口语化",
            "academic": "学术、严谨、引用规范",
            "creative": "创意、生动、富有想象力",
            "marketing": "吸引人、有说服力、行动导向"
        }
    
    def generate_article(
        self,
        topic: str,
        style: str = "professional",
        word_count: int = 1000,
        target_audience: str = None,
        keywords: List[str] = None,
        outline: List[str] = None
    ) -> Dict[str, str]:
        """生成文章"""
        
        style_desc = self.style_profiles.get(style, style)
        
        prompt = f"""
请撰写一篇关于"{topic}"的文章。

要求:
- 风格: {style_desc}
- 字数: 约{word_count}字
"""
        
        if target_audience:
            prompt += f"- 目标读者: {target_audience}\n"
        
        if keywords:
            prompt += f"- 关键词（需自然融入）: {', '.join(keywords)}\n"
        
        if outline:
            prompt += f"- 大纲:\n" + "\n".join([f"  {i+1}. {item}" for i, item in enumerate(outline)])
        
        prompt += "\n请开始撰写文章:"
        
        article = self.llm.generate(prompt)
        
        # 生成元信息
        meta = self._generate_meta(topic, article)
        
        return {
            "article": article,
            "title": meta.get("title"),
            "summary": meta.get("summary"),
            "tags": meta.get("tags"),
            "word_count": len(article)
        }
    
    def _generate_meta(self, topic: str, article: str) -> Dict:
        """生成文章元信息"""
        prompt = f"""
为以下文章生成元信息（JSON格式）:

文章主题: {topic}
文章内容: {article[:500]}...

请生成:
{{
    "title": "吸引人的标题",
    "summary": "100字以内的摘要",
    "tags": ["标签1", "标签2", "标签3"]
}}
"""
        
        response = self.llm.generate(prompt)
        
        import json
        try:
            return json.loads(response)
        except:
            return {}
    
    def rewrite(
        self,
        content: str,
        target_style: str = None,
        target_length: str = None,
        preserve_meaning: bool = True
    ) -> str:
        """改写内容"""
        
        instructions = []
        
        if target_style:
            instructions.append(f"改为{target_style}风格")
        
        if target_length:
            instructions.append(f"调整为{target_length}")
        
        if preserve_meaning:
            instructions.append("保持原意不变")
        
        prompt = f"""
请改写以下内容:

原文:
{content}

改写要求:
{chr(10).join(['- ' + i for i in instructions])}

改写后:
"""
        
        return self.llm.generate(prompt)
    
    def generate_social_posts(
        self,
        content: str,
        platforms: List[str] = None
    ) -> Dict[str, str]:
        """生成社交媒体文案"""
        
        platforms = platforms or ["twitter", "linkedin", "weibo"]
        
        platform_specs = {
            "twitter": "280字符以内，简洁有力，可使用话题标签",
            "linkedin": "专业、有深度，适合职场人士",
            "weibo": "140字以内，可使用表情和话题",
            "instagram": "视觉化描述，使用相关标签",
            "facebook": "友好互动，鼓励评论分享"
        }
        
        results = {}
        
        for platform in platforms:
            spec = platform_specs.get(platform, "适合该平台的风格")
            
            prompt = f"""
将以下内容改编为{platform}平台的发帖文案。

原内容:
{content}

平台要求: {spec}

文案:
"""
            
            results[platform] = self.llm.generate(prompt)
        
        return results
    
    def generate_email(
        self,
        purpose: str,
        recipient: str,
        key_points: List[str],
        tone: str = "professional"
    ) -> Dict[str, str]:
        """生成邮件"""
        
        prompt = f"""
请撰写一封{tone}风格的邮件。

目的: {purpose}
收件人: {recipient}
关键点:
{chr(10).join(['- ' + p for p in key_points])}

请生成邮件主题和正文:
"""
        
        response = self.llm.generate(prompt)
        
        # 解析主题和正文
        lines = response.strip().split('\n')
        subject = ""
        body = response
        
        for i, line in enumerate(lines):
            if line.startswith("主题:") or line.startswith("Subject:"):
                subject = line.split(":", 1)[1].strip()
                body = '\n'.join(lines[i+1:]).strip()
                break
        
        return {
            "subject": subject,
            "body": body
        }
```

## 企业应用

### 智能客服系统

```python
class CustomerServiceBot:
    """智能客服系统"""
    
    def __init__(self, llm, knowledge_base: RAGSystem):
        self.llm = llm
        self.kb = knowledge_base
        self.conversations = {}
        self.escalation_keywords = ["人工", "投诉", "经理", "退款"]
    
    def handle_message(
        self,
        user_id: str,
        message: str,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """处理用户消息"""
        
        # 获取或创建会话
        if user_id not in self.conversations:
            self.conversations[user_id] = {
                "history": [],
                "context": {},
                "sentiment_scores": []
            }
        
        conv = self.conversations[user_id]
        
        # 意图识别
        intent = self._classify_intent(message)
        
        # 情感分析
        sentiment = self._analyze_sentiment(message)
        conv["sentiment_scores"].append(sentiment)
        
        # 检查是否需要转人工
        if self._should_escalate(message, sentiment, conv):
            return {
                "response": "正在为您转接人工客服，请稍候...",
                "action": "escalate",
                "reason": "用户请求或情绪检测"
            }
        
        # 根据意图处理
        if intent == "faq":
            response = self._handle_faq(message)
        elif intent == "order_inquiry":
            response = self._handle_order_inquiry(message, metadata)
        elif intent == "complaint":
            response = self._handle_complaint(message, conv)
        else:
            response = self._handle_general(message, conv)
        
        # 更新会话历史
        conv["history"].append({"role": "user", "content": message})
        conv["history"].append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "intent": intent,
            "sentiment": sentiment,
            "action": None
        }
    
    def _classify_intent(self, message: str) -> str:
        """意图分类"""
        prompt = f"""
将以下客户消息分类为一个意图:
- faq: 常见问题咨询
- order_inquiry: 订单查询
- complaint: 投诉抱怨
- technical_support: 技术支持
- general: 一般咨询

消息: "{message}"

意图:
"""
        
        intent = self.llm.generate(prompt).strip().lower()
        
        valid_intents = ["faq", "order_inquiry", "complaint", "technical_support", "general"]
        return intent if intent in valid_intents else "general"
    
    def _analyze_sentiment(self, message: str) -> float:
        """情感分析，返回-1到1的分数"""
        prompt = f"""
分析以下消息的情感倾向，输出一个-1到1之间的数字:
-1: 非常负面
0: 中性
1: 非常正面

消息: "{message}"

分数:
"""
        
        try:
            score = float(self.llm.generate(prompt).strip())
            return max(-1, min(1, score))
        except:
            return 0
    
    def _should_escalate(self, message: str, sentiment: float, conv: Dict) -> bool:
        """判断是否需要转人工"""
        
        # 关键词触发
        if any(kw in message for kw in self.escalation_keywords):
            return True
        
        # 持续负面情绪
        recent_scores = conv["sentiment_scores"][-3:]
        if len(recent_scores) >= 3 and all(s < -0.5 for s in recent_scores):
            return True
        
        # 当前消息极度负面
        if sentiment < -0.8:
            return True
        
        return False
    
    def _handle_faq(self, message: str) -> str:
        """处理FAQ"""
        result = self.kb.generate(message)
        return result["answer"]
    
    def _handle_order_inquiry(self, message: str, metadata: Dict) -> str:
        """处理订单查询"""
        # 实际应该调用订单系统API
        prompt = f"""
用户询问订单相关问题: {message}

用户信息: {metadata}

请生成一个友好的回复，告知用户如何查询订单或需要提供什么信息:
"""
        return self.llm.generate(prompt)
    
    def _handle_complaint(self, message: str, conv: Dict) -> str:
        """处理投诉"""
        prompt = f"""
用户发起投诉: {message}

历史对话:
{self._format_history(conv["history"][-4:])}

请:
1. 表示歉意和理解
2. 询问具体情况（如果需要）
3. 提供可能的解决方案

回复:
"""
        return self.llm.generate(prompt)
    
    def _handle_general(self, message: str, conv: Dict) -> str:
        """处理一般咨询"""
        # 先尝试从知识库获取答案
        kb_result = self.kb.generate(message)
        
        # 如果知识库没有相关信息，使用对话模式
        if "无法回答" in kb_result["answer"] or "没有相关" in kb_result["answer"]:
            prompt = f"""
你是一位友好的客服代表。请回答用户的问题。

历史对话:
{self._format_history(conv["history"][-4:])}

用户: {message}

回复:
"""
            return self.llm.generate(prompt)
        
        return kb_result["answer"]
    
    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        return "\n".join([
            f"{h['role']}: {h['content']}"
            for h in history
        ])
```

## 生产环境最佳实践

### 1. 错误处理和重试

```python
import time
from functools import wraps
from typing import Callable

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1,
    max_delay: float = 60,
    exponential_base: float = 2,
    errors: tuple = (Exception,)
):
    """指数退避重试装饰器"""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    if retry == max_retries:
                        raise
                    
                    # 计算下次延迟
                    delay = min(delay * exponential_base, max_delay)
                    # 添加抖动
                    jitter = delay * 0.1 * (2 * np.random.random() - 1)
                    
                    print(f"重试 {retry + 1}/{max_retries}，等待 {delay + jitter:.2f}s")
                    time.sleep(delay + jitter)
            
            return None
        
        return wrapper
    return decorator


class RobustLLMClient:
    """健壮的LLM客户端"""
    
    def __init__(self, primary_client, fallback_client=None):
        self.primary = primary_client
        self.fallback = fallback_client
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0
        }
    
    @retry_with_exponential_backoff(max_retries=3)
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本，带重试和降级"""
        self.metrics["total_requests"] += 1
        
        try:
            result = self.primary.generate(prompt, **kwargs)
            self.metrics["successful_requests"] += 1
            return result
        except Exception as e:
            if self.fallback:
                print(f"主服务失败，使用备用服务: {e}")
                self.metrics["fallback_used"] += 1
                result = self.fallback.generate(prompt, **kwargs)
                self.metrics["successful_requests"] += 1
                return result
            else:
                self.metrics["failed_requests"] += 1
                raise
```

### 2. 成本控制

```python
class CostTracker:
    """成本追踪器"""
    
    # 价格（每1K tokens）
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self, budget_limit: float = 100.0):
        self.budget_limit = budget_limit
        self.total_cost = 0.0
        self.usage_log = []
    
    def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: str = None
    ):
        """记录使用量"""
        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        
        cost = (
            (input_tokens / 1000) * pricing["input"] +
            (output_tokens / 1000) * pricing["output"]
        )
        
        self.total_cost += cost
        
        self.usage_log.append({
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "request_id": request_id
        })
        
        # 检查预算
        if self.total_cost > self.budget_limit:
            raise BudgetExceededError(f"预算超限: ${self.total_cost:.2f} > ${self.budget_limit:.2f}")
        
        return cost
    
    def get_summary(self) -> Dict:
        """获取使用摘要"""
        return {
            "total_cost": self.total_cost,
            "remaining_budget": self.budget_limit - self.total_cost,
            "total_requests": len(self.usage_log),
            "total_input_tokens": sum(u["input_tokens"] for u in self.usage_log),
            "total_output_tokens": sum(u["output_tokens"] for u in self.usage_log)
        }


class BudgetExceededError(Exception):
    pass
```

### 3. 监控和日志

```python
import logging
from datetime import datetime

class LLMLogger:
    """LLM请求日志记录器"""
    
    def __init__(self, log_file: str = "llm_requests.log"):
        self.logger = logging.getLogger("llm")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def log_request(
        self,
        prompt: str,
        response: str,
        model: str,
        latency: float,
        tokens_used: int,
        metadata: Dict = None
    ):
        """记录请求"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "latency_ms": latency * 1000,
            "tokens": tokens_used,
            "metadata": metadata
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def log_error(self, error: Exception, context: Dict = None):
        """记录错误"""
        self.logger.error(f"Error: {str(error)}, Context: {context}")
```

## 总结

本文介绍了GPT在实际应用中的案例和最佳实践：

1. **RAG系统**：检索增强生成，让GPT能够利用外部知识
2. **代码助手**：代码生成、审查、修复和重构
3. **内容创作**：文章生成、改写、社交媒体文案
4. **企业应用**：智能客服系统的完整实现
5. **生产实践**：错误处理、成本控制、监控日志

### 关键要点

| 方面 | 建议 |
|------|------|
| 架构设计 | 模块化、可扩展、易维护 |
| 错误处理 | 重试机制、优雅降级、用户友好提示 |
| 成本控制 | 预算限制、使用追踪、模型选择优化 |
| 质量保证 | 输出验证、人工审核流程、持续评估 |
| 安全性 | 输入清洗、敏感信息过滤、访问控制 |

### GPT应用开发清单

- [ ] 明确定义应用场景和目标
- [ ] 选择合适的模型和架构
- [ ] 实现健壮的错误处理
- [ ] 设置成本控制机制
- [ ] 建立监控和日志系统
- [ ] 进行充分的测试
- [ ] 部署前的安全审查
- [ ] 上线后的持续优化

恭喜你完成了GPT完全指南的学习！希望这个系列能够帮助你深入理解GPT技术，并在实际项目中成功应用。记住，技术在不断发展，保持学习的热情是最重要的。

祝你在AI应用开发的道路上取得成功！
