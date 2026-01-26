---
title: 'Make (Integromat) 深度教程'
description: '掌握 Make 自动化平台的高级功能，创建复杂的 AI 驱动工作流。'
pubDate: 2025-04-20
category: '技术'
tags: ['AI自动化', 'Make', 'Integromat', '工作流']
series: 'AI 自动化工作流'
seriesOrder: 3
heroImage: 'https://images.unsplash.com/photo-1551434678-e076c223a692?w=800&h=400&fit=crop'
---

Make（原 Integromat）以其强大的可视化编辑器和灵活的数据处理能力著称。本文将深入探讨 Make 的高级功能和 AI 集成。

## Make 平台特色

### 与其他平台对比

| 特性 | Make | Zapier | n8n |
|------|------|--------|-----|
| 可视化程度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 数据操作 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 价格性价比 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 学习曲线 | 中等 | 简单 | 较陡 |
| AI 集成 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 核心概念

```
Make 术语对照：
- Scenario = 工作流
- Module = 节点
- Connection = 连接/认证
- Bundle = 数据包
- Operation = 执行次数
```

## 场景创建基础

### 模块类型

```yaml
触发模块 (Triggers):
  - Instant: 实时触发（Webhook）
  - Scheduled: 定时触发
  - Polling: 轮询触发

动作模块 (Actions):
  - Search: 搜索数据
  - Create: 创建记录
  - Update: 更新记录
  - Delete: 删除记录

流程控制:
  - Router: 条件分支
  - Iterator: 循环处理
  - Aggregator: 数据聚合
  - Repeater: 重复执行
```

### 数据映射

Make 的数据映射非常强大：

```javascript
// 文本函数
{{lower(1.name)}}  // 转小写
{{upper(1.email)}} // 转大写
{{trim(1.text)}}   // 去除空格

// 数组操作
{{first(1.items)}}           // 第一个元素
{{last(1.items)}}            // 最后一个元素
{{length(1.items)}}          // 数组长度
{{map(1.items; "name")}}     // 提取所有 name

// 条件判断
{{if(1.status = "active"; "启用"; "禁用")}}

// 日期函数
{{formatDate(now; "YYYY-MM-DD")}}
{{addDays(1.date; 7)}}
```

## AI 集成实战

### OpenAI 模块配置

```yaml
OpenAI 模块设置:
  Connection: 选择或创建 OpenAI 连接
  Model: gpt-4o / gpt-4o-mini
  Messages:
    - Role: system
      Content: |
        你是一个专业的数据分析助手。
        请分析用户输入并返回 JSON 格式。
    - Role: user
      Content: "{{1.input_text}}"
  
  Advanced Settings:
    Temperature: 0.3
    Max Tokens: 1000
    Response Format: json_object
```

### 实战案例：智能表单处理

```yaml
场景：自动处理网站询盘表单

流程设计：
1. Webhook 接收表单数据
2. OpenAI 分析客户意图和紧急程度
3. 根据分析结果分流：
   - 高意向 → 立即通知销售 + CRM 创建
   - 一般咨询 → 发送自动回复
   - 垃圾信息 → 标记并忽略
4. 所有数据记录到 Google Sheets
```

**详细配置**:

```json
{
  "name": "智能表单处理",
  "modules": [
    {
      "id": 1,
      "module": "gateway:CustomWebHook",
      "parameters": {
        "hook": "form_submission"
      }
    },
    {
      "id": 2,
      "module": "openai:CreateCompletion",
      "parameters": {
        "model": "gpt-4o-mini",
        "messages": [
          {
            "role": "system",
            "content": "分析表单提交，返回JSON：{\"intent\": \"high/medium/low/spam\", \"category\": \"product/service/partnership/other\", \"summary\": \"简要描述\", \"suggested_response\": \"建议回复\"}"
          },
          {
            "role": "user",
            "content": "姓名: {{1.name}}\n邮箱: {{1.email}}\n公司: {{1.company}}\n消息: {{1.message}}"
          }
        ]
      }
    },
    {
      "id": 3,
      "module": "builtin:BasicRouter",
      "routes": [
        {
          "name": "高意向",
          "filter": "{{2.choices[].message.content.intent}} = \"high\""
        },
        {
          "name": "一般咨询",
          "filter": "{{2.choices[].message.content.intent}} = \"medium\""
        },
        {
          "name": "低意向/垃圾",
          "filter": "{{2.choices[].message.content.intent}} in [\"low\", \"spam\"]"
        }
      ]
    }
  ]
}
```

### 实战案例：内容审核流水线

```yaml
场景：用户生成内容 (UGC) 自动审核

流程：
1. 监听新内容提交
2. 并行处理：
   a. 文本审核（OpenAI Moderation）
   b. 图片审核（Google Vision / AWS Rekognition）
3. 聚合审核结果
4. 决策：
   - 全部通过 → 自动发布
   - 部分问题 → 人工审核队列
   - 严重违规 → 自动拒绝 + 通知
5. 更新内容状态
```

## 高级功能

### Iterator + Aggregator 模式

处理批量数据的常用模式：

```javascript
// 输入：一个包含多个项目的数组
// Bundle: { items: [{id: 1, name: "A"}, {id: 2, name: "B"}, ...] }

// Iterator 将数组拆分为多个 bundle
// 每个 bundle: {id: 1, name: "A"}, {id: 2, name: "B"}, ...

// 中间处理：对每个 bundle 进行 AI 处理

// Aggregator 将结果合并
// 输出：一个包含所有处理结果的数组
```

### Error Handler 配置

```yaml
错误处理策略：
  
  Break:
    - 停止场景执行
    - 记录错误详情
    
  Resume:
    - 继续执行下一个 bundle
    - 跳过出错的数据
    
  Commit:
    - 保存已处理的数据
    - 回滚当前 bundle
    
  Rollback:
    - 回滚所有操作
    - 重置到初始状态

推荐配置：
  - API 调用失败 → Resume + 记录日志
  - 数据验证失败 → Resume + 通知管理员
  - 关键操作失败 → Break + 告警
```

### Data Store 使用

Make 内置的数据存储功能：

```yaml
Data Store 用途：
  - 缓存 API 响应
  - 存储配置信息
  - 跨场景共享数据
  - 去重处理
  - 状态管理

示例：API 响应缓存
1. 检查 Data Store 是否有缓存
2. 如果有且未过期 → 使用缓存
3. 如果没有或过期 → 调用 API → 存入 Data Store
```

### HTTP 模块高级配置

```yaml
HTTP Request 设置：
  URL: https://api.example.com/v1/analyze
  Method: POST
  
  Headers:
    Authorization: Bearer {{1.api_key}}
    Content-Type: application/json
  
  Body:
    type: raw
    content: |
      {
        "text": "{{1.content}}",
        "options": {
          "language": "zh",
          "detailed": true
        }
      }
  
  Parse Response: Yes
  
  Advanced:
    Timeout: 30
    Follow Redirects: Yes
    Reject Self-Signed Certificates: No
```

## 性能优化

### 减少操作数

```yaml
优化策略：

1. 使用 Array Aggregator 减少循环
   优化前：100个项目 = 100次操作
   优化后：1次聚合 = 1次操作

2. 条件过滤提前
   在 Iterator 之前过滤数据
   减少不必要的处理

3. 批量 API 调用
   合并多个请求为一个
   使用支持批量的 API

4. 缓存重复数据
   使用 Data Store 缓存
   设置合理的过期时间
```

### 执行时间优化

```yaml
时间优化：

1. 使用 Webhooks 替代 Polling
   - 即时响应 vs 定时检查
   - 减少空转操作

2. 并行处理
   - Router 分支并行执行
   - 善用 Make 的并行能力

3. 精简数据传递
   - 只传递需要的字段
   - 避免大文件直接传递
```

## 定价与成本控制

### 操作计算方式

```yaml
操作计费规则：
  - 每个模块执行 = 1次操作
  - Iterator 每个 bundle = 1次操作
  - Router 只计算执行的分支
  - 错误重试也计算操作

示例计算：
  场景：处理 100 条表单数据
  - Webhook: 1 次
  - Iterator: 100 次
  - AI 处理: 100 次
  - Router: 100 次（假设都走同一分支）
  - 创建记录: 100 次
  总计：401 次操作
```

### 省钱技巧

```yaml
1. 合理使用过滤器
   在数据源就过滤掉不需要的数据

2. 减少不必要的模块
   合并可以合并的操作

3. 使用 Aggregator
   批量处理代替逐条处理

4. 选择合适的触发频率
   不需要实时的用定时触发
```

## 总结

Make 的核心优势：

1. **可视化最佳** - 复杂流程一目了然
2. **数据处理强** - 内置丰富的函数
3. **性价比高** - 操作单价相对便宜
4. **易于调试** - 每步都可查看数据

下一篇，我们将探索 Zapier 的 AI 功能。
