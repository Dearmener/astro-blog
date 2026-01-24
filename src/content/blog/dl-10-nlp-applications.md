---
title: '深度学习完全指南（十）：自然语言处理应用'
description: '从文本分类到机器翻译、问答系统，全面掌握NLP的核心技术与实战应用'
pubDate: '2024-02-10'
heroImage: ''
category: '技术'
tags: ['深度学习', 'NLP', '自然语言处理', 'Transformer']
series: '深度学习完全指南'
seriesOrder: 10
---

## NLP任务全景

自然语言处理（NLP）是深度学习最活跃的应用领域之一。从BERT到GPT，预训练语言模型彻底改变了NLP的研究范式。

### NLP任务分类

| 类别 | 任务 | 应用场景 |
|------|------|---------|
| **文本分类** | 情感分析、垃圾邮件检测 | 舆情监控、内容审核 |
| **序列标注** | NER、词性标注 | 信息抽取、知识图谱 |
| **文本匹配** | 语义相似度、问答匹配 | 搜索引擎、客服 |
| **文本生成** | 摘要、翻译、对话 | 自动写作、翻译 |
| **信息抽取** | 关系抽取、事件抽取 | 知识图谱构建 |
| **问答系统** | 阅读理解、知识问答 | 智能助手 |

---

## 文本预处理

### Tokenization

```python
from transformers import AutoTokenizer

# 加载预训练tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

text = "深度学习正在改变世界"
tokens = tokenizer.tokenize(text)
print(f"分词结果: {tokens}")
# ['深', '度', '学', '习', '正', '在', '改', '变', '世', '界']

# 编码
encoded = tokenizer(text, return_tensors='pt')
print(f"input_ids: {encoded['input_ids']}")
print(f"attention_mask: {encoded['attention_mask']}")

# 批量编码（自动padding）
texts = ["你好", "深度学习正在改变世界"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

### 自定义词表

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 训练BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 从文件训练
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# 保存
tokenizer.save("custom_tokenizer.json")
```

---

## 文本分类

### BERT文本分类

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-chinese', dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 使用示例
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertClassifier(num_classes=3)

text = "这部电影非常好看，强烈推荐！"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    pred = torch.argmax(logits, dim=1)
    print(f"预测类别: {pred.item()}")
```

### 使用Hugging Face Trainer

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 加载数据集
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})

# 加载模型和tokenizer
model_name = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 数据预处理
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(preprocess, batched=True)

# 评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro')
    }

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 多标签分类

```python
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        return {'loss': loss, 'logits': logits}

# 推理时使用sigmoid
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    probs = torch.sigmoid(outputs['logits'])
    predictions = (probs > 0.5).int()  # 多标签预测
```

---

## 命名实体识别（NER）

### BiLSTM-CRF

```python
import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           num_layers=2, bidirectional=True, 
                           dropout=dropout, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, tags=None, mask=None):
        embeddings = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # 训练：返回负对数似然
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # 推理：Viterbi解码
            return self.crf.decode(emissions, mask=mask)

# BIO标签
TAGS = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
```

### BERT-NER

```python
from transformers import BertForTokenClassification, BertTokenizerFast

class BERT_NER:
    def __init__(self, model_path, labels):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(
            model_path, num_labels=len(labels)
        )
        self.labels = labels
        self.id2label = {i: l for i, l in enumerate(labels)}
    
    def predict(self, text):
        # 分词
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            return_offsets_mapping=True,
            truncation=True
        )
        offset_mapping = inputs.pop('offset_mapping')[0]
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
        
        # 解析实体
        entities = []
        current_entity = None
        
        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            if offset[0] == offset[1]:  # 特殊token
                continue
            
            label = self.id2label[pred.item()]
            char = text[offset[0]:offset[1]]
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'type': label[2:],
                    'text': char,
                    'start': offset[0].item(),
                    'end': offset[1].item()
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += char
                current_entity['end'] = offset[1].item()
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

# 使用
ner = BERT_NER('bert-ner-chinese', ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'])
text = "马云在杭州创办了阿里巴巴"
entities = ner.predict(text)
# [{'type': 'PER', 'text': '马云', 'start': 0, 'end': 2},
#  {'type': 'LOC', 'text': '杭州', 'start': 3, 'end': 5},
#  {'type': 'ORG', 'text': '阿里巴巴', 'start': 8, 'end': 12}]
```

---

## 文本匹配

### 双塔模型（Bi-Encoder）

```python
class BiEncoder(nn.Module):
    """双塔模型：独立编码，向量检索"""
    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
    
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = (token_embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1)
        return sum_embeddings / sum_mask
    
    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        query_emb = self.encode(query_ids, query_mask)
        doc_emb = self.encode(doc_ids, doc_mask)
        
        # 余弦相似度
        query_emb = F.normalize(query_emb, p=2, dim=1)
        doc_emb = F.normalize(doc_emb, p=2, dim=1)
        
        similarity = torch.mm(query_emb, doc_emb.t())
        return similarity

# 对比学习训练
def contrastive_loss(similarity, temperature=0.05):
    """InfoNCE损失"""
    labels = torch.arange(similarity.size(0), device=similarity.device)
    return F.cross_entropy(similarity / temperature, labels)
```

### 交叉编码器（Cross-Encoder）

```python
class CrossEncoder(nn.Module):
    """交叉编码器：联合编码，更精确"""
    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 输入格式: [CLS] query [SEP] doc [SEP]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        score = self.classifier(outputs.pooler_output)
        return score.squeeze(-1)

# 使用示例
def match_score(query, document):
    inputs = tokenizer(
        query, document,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        score = model(**inputs)
    return torch.sigmoid(score).item()
```

### Sentence-BERT

```python
from sentence_transformers import SentenceTransformer, util

# 加载预训练模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 编码句子
sentences = [
    "深度学习是人工智能的核心技术",
    "机器学习是AI的重要组成部分",
    "今天天气真好"
]
embeddings = model.encode(sentences)

# 计算相似度
similarity = util.cos_sim(embeddings[0], embeddings[1:])
print(f"句子1与句子2的相似度: {similarity[0][0]:.4f}")
print(f"句子1与句子3的相似度: {similarity[0][1]:.4f}")

# 语义搜索
query = "人工智能技术"
query_embedding = model.encode(query)

# 找最相似的句子
scores = util.cos_sim(query_embedding, embeddings)[0]
top_idx = torch.argmax(scores).item()
print(f"最相似: {sentences[top_idx]}")
```

---

## 机器翻译

### Seq2Seq with Attention

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                         dropout=dropout, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # 合并双向hidden
        hidden = torch.tanh(self.fc(
            torch.cat([hidden[-2], hidden[-1]], dim=1)
        ))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, src_len, hidden_dim * 2)
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, 
                         num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))
        
        attn_weights = self.attention(hidden, encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        output = self.fc(torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1))
        
        return output, hidden.squeeze(0), attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        mask = (src != 0)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        
        return outputs
```

### Transformer翻译模型

```python
from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, src_lang='zh', tgt_lang='en'):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    
    def translate(self, texts, max_length=128):
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

# 使用
translator = Translator('zh', 'en')
results = translator.translate([
    "深度学习正在改变世界",
    "人工智能是未来的发展方向"
])
print(results)
```

---

## 文本摘要

### 抽取式摘要

```python
from transformers import BertModel, BertTokenizer
import numpy as np
from sklearn.cluster import KMeans

class ExtractiveSummarizer:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_sentence_embeddings(self, sentences):
        embeddings = []
        for sent in sentences:
            inputs = self.tokenizer(sent, return_tensors='pt', 
                                   truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS] token作为句子表示
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])
        return np.array(embeddings)
    
    def summarize(self, text, num_sentences=3):
        # 分句
        sentences = text.replace('。', '。\n').split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # 获取句子向量
        embeddings = self.get_sentence_embeddings(sentences)
        
        # K-means聚类选择代表句
        kmeans = KMeans(n_clusters=num_sentences, random_state=42)
        kmeans.fit(embeddings)
        
        # 每个cluster选最近的句子
        selected_indices = []
        for i in range(num_sentences):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            distances = np.linalg.norm(
                embeddings[cluster_indices] - kmeans.cluster_centers_[i], axis=1
            )
            selected_indices.append(cluster_indices[np.argmin(distances)])
        
        # 按原顺序排列
        selected_indices.sort()
        summary = '。'.join([sentences[i] for i in selected_indices]) + '。'
        
        return summary
```

### 生成式摘要

```python
from transformers import BartForConditionalGeneration, BertTokenizer

class AbstractiveSummarizer:
    def __init__(self, model_name='fnlp/bart-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
    
    def summarize(self, text, max_length=150, min_length=50):
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            max_length=1024, 
            truncation=True
        )
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
```

---

## 问答系统

### 阅读理解（Extractive QA）

```python
from transformers import BertForQuestionAnswering, BertTokenizer

class ExtractiveQA:
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
    
    def answer(self, question, context):
        inputs = self.tokenizer(
            question, context,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取答案位置
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)
        
        # 解码答案
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens)
        
        # 计算置信度
        start_prob = F.softmax(outputs.start_logits, dim=1)[0][start_idx].item()
        end_prob = F.softmax(outputs.end_logits, dim=1)[0][end_idx].item()
        confidence = (start_prob + end_prob) / 2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'start': start_idx.item(),
            'end': end_idx.item()
        }

# 使用
qa = ExtractiveQA()
context = "阿里巴巴集团由马云在1999年于杭州创立，是一家中国的跨国科技公司。"
question = "阿里巴巴是谁创立的？"

result = qa.answer(question, context)
print(f"答案: {result['answer']}, 置信度: {result['confidence']:.2f}")
```

### 检索增强生成（RAG）

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAG:
    def __init__(self, retriever_model='paraphrase-multilingual-MiniLM-L12-v2'):
        self.encoder = SentenceTransformer(retriever_model)
        self.documents = []
        self.index = None
    
    def build_index(self, documents):
        """构建向量索引"""
        self.documents = documents
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # 使用FAISS构建索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        
        # 归一化后内积等于余弦相似度
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'document': self.documents[idx],
                'score': score
            })
        
        return results
    
    def generate(self, query, generator, top_k=3):
        """检索增强生成"""
        # 检索相关文档
        retrieved = self.retrieve(query, top_k)
        
        # 构造prompt
        context = "\n".join([r['document'] for r in retrieved])
        prompt = f"""基于以下信息回答问题：

{context}

问题：{query}
答案："""
        
        # 生成答案
        answer = generator.generate(prompt)
        
        return {
            'answer': answer,
            'sources': retrieved
        }
```

---

## 对话系统

### 多轮对话模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class DialogueSystem:
    def __init__(self, model_name='THUDM/chatglm-6b'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).half().cuda()
        self.history = []
    
    def chat(self, user_input, max_length=2048):
        response, self.history = self.model.chat(
            self.tokenizer, 
            user_input, 
            history=self.history,
            max_length=max_length
        )
        return response
    
    def reset(self):
        """清空对话历史"""
        self.history = []

# 使用
chatbot = DialogueSystem()

# 多轮对话
print(chatbot.chat("你好，请介绍一下深度学习"))
print(chatbot.chat("它有哪些主要应用？"))
print(chatbot.chat("能详细说说NLP方向吗？"))
```

### 意图识别 + 槽位填充

```python
class JointIntentSlot(nn.Module):
    """联合意图识别和槽位填充"""
    def __init__(self, model_name, num_intents, num_slots):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 意图分类（使用[CLS]）
        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        
        # 槽位标注（使用所有token）
        self.slot_classifier = nn.Linear(hidden_size, num_slots)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        
        # 意图识别
        intent_logits = self.intent_classifier(outputs.pooler_output)
        
        # 槽位填充
        slot_logits = self.slot_classifier(outputs.last_hidden_state)
        
        return intent_logits, slot_logits

# 训练示例
# 输入: "帮我订一张明天从北京到上海的机票"
# 意图: book_flight
# 槽位: O O O O O B-date O B-from_city O B-to_city O O O
```

---

## 关系抽取

```python
class RelationExtractor(nn.Module):
    """实体关系抽取"""
    def __init__(self, model_name, num_relations):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 使用实体位置的特殊标记
        self.entity_start_linear = nn.Linear(hidden_size, hidden_size)
        self.relation_classifier = nn.Linear(hidden_size * 2, num_relations)
    
    def forward(self, input_ids, attention_mask, head_positions, tail_positions):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 获取头实体和尾实体的表示
        batch_size = input_ids.size(0)
        head_repr = sequence_output[torch.arange(batch_size), head_positions]
        tail_repr = sequence_output[torch.arange(batch_size), tail_positions]
        
        # 拼接实体表示进行关系分类
        concat_repr = torch.cat([head_repr, tail_repr], dim=-1)
        relation_logits = self.relation_classifier(concat_repr)
        
        return relation_logits

# 示例
# 输入: "[CLS] [E1] 马云 [/E1] 创立了 [E2] 阿里巴巴 [/E2] [SEP]"
# 输出: "创始人" (关系类型)
```

---

## 评估指标

### 文本分类指标

```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification(y_true, y_pred, labels):
    print(classification_report(y_true, y_pred, target_names=labels))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_true, y_pred))
```

### NER评估（严格匹配）

```python
from seqeval.metrics import classification_report as ner_report
from seqeval.metrics import f1_score as ner_f1

def evaluate_ner(y_true, y_pred):
    """
    y_true, y_pred: List[List[str]]
    例: [['O', 'B-PER', 'I-PER', 'O'], ...]
    """
    print(ner_report(y_true, y_pred))
    print(f"F1: {ner_f1(y_true, y_pred):.4f}")
```

### 翻译评估（BLEU）

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import sacrebleu

def calculate_bleu(references, hypotheses):
    # 使用sacrebleu（更标准）
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU: {bleu.score:.2f}")
    return bleu.score
```

### 摘要评估（ROUGE）

```python
from rouge import Rouge

def calculate_rouge(references, hypotheses):
    rouge = Rouge()
    scores = rouge.get_scores(hypotheses, references, avg=True)
    
    print(f"ROUGE-1: {scores['rouge-1']['f']:.4f}")
    print(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
    print(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")
    
    return scores
```

---

## 实战项目：智能客服系统

```python
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss

class CustomerServiceBot:
    def __init__(self):
        # 意图识别模型
        self.intent_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.intent_model = BertClassifier(num_classes=10)
        
        # 检索模型
        self.retriever = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # FAQ知识库
        self.faqs = self.load_faqs()
        self.faq_index = self.build_faq_index()
        
        # 对话历史
        self.history = []
    
    def load_faqs(self):
        return {
            "如何退款": "您可以在订单详情页点击申请退款...",
            "配送时间": "一般商品1-3天送达...",
            "修改收货地址": "在订单发货前可以修改地址..."
        }
    
    def build_faq_index(self):
        questions = list(self.faqs.keys())
        embeddings = self.retriever.encode(questions)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        return index, questions
    
    def classify_intent(self, text):
        """意图识别"""
        inputs = self.intent_tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            logits = self.intent_model(**inputs)
            intent = torch.argmax(logits, dim=1).item()
        return intent
    
    def search_faq(self, query, threshold=0.8):
        """FAQ检索"""
        index, questions = self.faq_index
        
        query_emb = self.retriever.encode([query])
        faiss.normalize_L2(query_emb)
        
        scores, indices = index.search(query_emb.astype('float32'), 1)
        
        if scores[0][0] > threshold:
            question = questions[indices[0][0]]
            return self.faqs[question]
        return None
    
    def respond(self, user_input):
        """生成回复"""
        # 1. 尝试FAQ检索
        faq_answer = self.search_faq(user_input)
        if faq_answer:
            return faq_answer
        
        # 2. 意图识别
        intent = self.classify_intent(user_input)
        
        # 3. 根据意图路由
        if intent == 0:  # 问候
            return "您好！有什么可以帮助您的吗？"
        elif intent == 1:  # 咨询
            return "让我帮您查询一下..."
        elif intent == 2:  # 投诉
            return "非常抱歉给您带来不便，我们会尽快处理..."
        else:
            return "抱歉，我没有理解您的问题，请问您想咨询什么？"

# 使用
bot = CustomerServiceBot()
print(bot.respond("我想退款"))
print(bot.respond("发货需要几天"))
```

---

## 小结

本文涵盖了NLP的核心应用：

| 任务 | 主流方法 | 推荐模型 |
|------|---------|---------|
| 文本分类 | Fine-tune PLM | BERT, RoBERTa |
| NER | BiLSTM-CRF / BERT | BERT-NER |
| 文本匹配 | 双塔/交叉编码 | Sentence-BERT |
| 机器翻译 | Transformer | MarianMT, mBART |
| 文本摘要 | Seq2Seq | BART, T5 |
| 问答系统 | RAG | BERT-QA + Retriever |
| 对话系统 | 生成式LLM | ChatGLM, LLaMA |

**下一篇**：强化学习基础，从MDP到DQN、PPO等核心算法。
