---
title: 'n8n 入门与实战'
description: '从零开始学习 n8n 自动化平台，掌握工作流创建、AI 集成和实战案例。'
pubDate: 2025-04-10
category: '技术'
tags: ['AI自动化', 'n8n', '工作流', '开源']
series: 'AI 自动化工作流'
seriesOrder: 2
heroImage: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=800&h=400&fit=crop'
---

n8n 是一个强大的开源工作流自动化平台，支持自托管和丰富的 AI 集成。本文将带你从零开始掌握 n8n。

## n8n 简介

### 为什么选择 n8n

- **开源免费** - MIT 许可，可自托管
- **可视化编辑** - 拖拽式创建工作流
- **400+ 集成** - 丰富的节点生态
- **AI 原生** - 内置 LangChain 支持
- **代码能力** - JavaScript/Python 自定义

### 安装 n8n

```bash
# Docker 安装（推荐）
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n

# npm 安装
npm install n8n -g
n8n start

# 访问界面
# http://localhost:5678
```

## 核心概念

### 工作流结构

```
┌──────────────────────────────────────────┐
│               n8n 工作流                  │
├──────────────────────────────────────────┤
│                                           │
│  ┌─────────┐                              │
│  │ Trigger │  触发节点（工作流入口）        │
│  └────┬────┘                              │
│       │                                   │
│       ▼                                   │
│  ┌─────────┐                              │
│  │  Node   │  处理节点（数据转换/API调用）  │
│  └────┬────┘                              │
│       │                                   │
│    ┌──┴──┐                                │
│    │ IF  │  条件节点（分支逻辑）           │
│    └┬───┬┘                                │
│     │   │                                 │
│     ▼   ▼                                 │
│  ┌────┐ ┌────┐                            │
│  │Node│ │Node│  并行处理                  │
│  └────┘ └────┘                            │
│                                           │
└──────────────────────────────────────────┘
```

### 数据流动

```javascript
// n8n 中的数据结构
// 每个节点接收和输出的都是数组
[
  {
    "json": {
      "name": "John",
      "email": "john@example.com"
    },
    "binary": {} // 可选：文件数据
  },
  {
    "json": {
      "name": "Jane", 
      "email": "jane@example.com"
    }
  }
]
```

## 实战案例

### 案例 1: 智能邮件分类

创建一个工作流，自动分类收到的邮件并打标签。

```yaml
工作流设计：
1. 触发器：每 5 分钟检查新邮件
2. 获取邮件：IMAP 节点
3. AI 分类：OpenAI 分析邮件内容
4. 条件分支：根据分类结果
5. 执行动作：移动到对应文件夹/打标签
```

**工作流 JSON 配置**:

```json
{
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "parameters": {
        "rule": {
          "interval": [{ "field": "minutes", "minutesInterval": 5 }]
        }
      },
      "position": [250, 300]
    },
    {
      "name": "Get Emails",
      "type": "n8n-nodes-base.emailReadImap",
      "parameters": {
        "mailbox": "INBOX",
        "options": { "unseen": true }
      },
      "position": [450, 300]
    },
    {
      "name": "AI Classify",
      "type": "@n8n/n8n-nodes-langchain.openAi",
      "parameters": {
        "model": "gpt-4o-mini",
        "messages": {
          "values": [
            {
              "role": "system",
              "content": "你是邮件分类助手。根据邮件内容返回分类：work/personal/promotion/spam"
            },
            {
              "role": "user", 
              "content": "主题：{{$json.subject}}\n内容：{{$json.text}}"
            }
          ]
        }
      },
      "position": [650, 300]
    },
    {
      "name": "Switch",
      "type": "n8n-nodes-base.switch",
      "parameters": {
        "dataType": "string",
        "value1": "={{$json.message.content}}",
        "rules": {
          "rules": [
            { "value2": "work" },
            { "value2": "personal" },
            { "value2": "promotion" },
            { "value2": "spam" }
          ]
        }
      },
      "position": [850, 300]
    }
  ]
}
```

### 案例 2: 客户反馈处理管道

```javascript
// Function 节点：处理 AI 响应
const aiResponse = $input.first().json;

// 解析 AI 返回的 JSON
let analysis;
try {
  analysis = JSON.parse(aiResponse.message.content);
} catch (e) {
  analysis = {
    sentiment: 'unknown',
    category: 'other',
    urgency: 'medium',
    summary: aiResponse.message.content
  };
}

// 添加处理时间戳
analysis.processedAt = new Date().toISOString();
analysis.originalFeedback = $input.first().json.feedback;

return [{ json: analysis }];
```

### 案例 3: 自动化内容发布

```yaml
工作流：内容多平台发布
触发器：Notion 数据库更新

步骤：
1. Notion Trigger - 监听"待发布"状态
2. 获取内容详情
3. AI 优化标题和摘要
4. 并行发布：
   - Twitter 节点
   - LinkedIn 节点  
   - 微信公众号（HTTP Request）
5. 更新 Notion 状态为"已发布"
6. 发送 Slack 通知
```

**代码实现**:

```javascript
// Code 节点：生成各平台适配内容
const content = $input.first().json;

// 原始内容
const originalTitle = content.properties.Title.title[0].plain_text;
const originalContent = content.properties.Content.rich_text[0].plain_text;

// 生成各平台版本
const platforms = {
  twitter: {
    // Twitter 限制 280 字符
    text: originalContent.slice(0, 250) + '...',
    hashtags: content.properties.Tags.multi_select.map(t => '#' + t.name).join(' ')
  },
  linkedin: {
    title: originalTitle,
    text: originalContent,
    // LinkedIn 适合长文
  },
  wechat: {
    title: originalTitle,
    digest: originalContent.slice(0, 120),
    content: originalContent
  }
};

return Object.entries(platforms).map(([platform, data]) => ({
  json: { platform, ...data }
}));
```

## AI 节点详解

### OpenAI 节点

```javascript
// 系统提示设计
const systemPrompt = `
你是一个专业的数据分析助手。
任务：分析用户提供的数据，返回结构化 JSON。

输出格式：
{
  "insights": ["洞察1", "洞察2"],
  "recommendations": ["建议1", "建议2"],
  "riskLevel": "low|medium|high"
}

注意：只返回 JSON，不要其他文字。
`;
```

### LangChain 节点

n8n 内置了 LangChain 支持，可以创建更复杂的 AI 工作流：

```yaml
LangChain 节点类型：
- Chat Model：对话模型
- Embedding：向量嵌入
- Vector Store：向量存储
- Retriever：检索器
- Chain：处理链
- Agent：AI Agent
- Tool：工具调用
```

### 使用 RAG 的工作流

```javascript
// 创建 RAG 工作流
// 1. 文档加载节点
// 2. 文本分割节点
// 3. 嵌入节点
// 4. 向量存储节点（Pinecone/Qdrant）
// 5. 检索增强生成

// 查询时：
// 1. 用户问题 → 嵌入
// 2. 向量检索相关文档
// 3. 文档 + 问题 → LLM
// 4. 返回答案
```

## 高级技巧

### 错误处理

```javascript
// 在 Function 节点中处理错误
try {
  const result = await someOperation();
  return [{ json: { success: true, data: result } }];
} catch (error) {
  // 记录错误但不中断工作流
  return [{
    json: {
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    }
  }];
}
```

### 并行处理

```javascript
// Split In Batches 节点配置
{
  "batchSize": 10,
  "options": {
    "reset": false
  }
}

// 处理后用 Merge 节点合并结果
```

### 环境变量

```bash
# .env 文件
OPENAI_API_KEY=sk-xxx
DATABASE_URL=postgres://...
SLACK_WEBHOOK=https://hooks.slack.com/...
```

```javascript
// 在节点中使用
const apiKey = $env.OPENAI_API_KEY;
```

## 部署最佳实践

### Docker Compose 生产部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  n8n:
    image: n8nio/n8n
    restart: always
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_ENCRYPTION_KEY=${ENCRYPTION_KEY}
      - DB_TYPE=postgresdb
      - DB_POSTGRESDB_HOST=postgres
      - DB_POSTGRESDB_DATABASE=n8n
      - DB_POSTGRESDB_USER=n8n
      - DB_POSTGRESDB_PASSWORD=${DB_PASSWORD}
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    restart: always
    environment:
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=n8n
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  n8n_data:
  postgres_data:
```

## 总结

n8n 的核心优势：

1. **开源自托管** - 完全控制数据
2. **AI 原生** - LangChain 深度集成
3. **灵活性** - 代码节点无限可能
4. **社区活跃** - 丰富的模板和支持

下一篇，我们将学习 Make (Integromat) —— 可视化最强的自动化平台。
