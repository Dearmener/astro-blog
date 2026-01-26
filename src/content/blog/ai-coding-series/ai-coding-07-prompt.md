---
title: 'Prompt Engineering for Coding'
description: '掌握面向编程的 Prompt 工程技巧，让 AI 生成更精准的代码。'
pubDate: 2025-04-12
category: '技术'
tags: ['AI编程', 'Prompt Engineering', 'ChatGPT', 'Claude']
series: 'AI Coding 完全指南'
seriesOrder: 7
heroImage: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800&h=400&fit=crop'
---

Prompt Engineering 是与 AI 高效协作的核心技能。针对编程场景，有特定的技巧和模式可以大幅提升 AI 输出质量。

## Prompt 基础框架

### COSTAR 框架

```markdown
**C**ontext (上下文): 项目背景和技术栈
**O**bjective (目标): 具体要完成的任务
**S**tyle (风格): 代码风格和规范要求
**T**one (语调): 正式/简洁/详细
**A**udience (受众): 代码的使用者
**R**esponse (响应): 期望的输出格式
```

### 实际应用

```markdown
# Context
我正在开发一个 React + TypeScript 项目，使用 Zustand 做状态管理，
TanStack Query 做数据获取。项目使用 ESLint + Prettier。

# Objective
创建一个购物车状态管理模块，支持：
- 添加/删除商品
- 更新数量
- 计算总价
- 持久化到 localStorage

# Style
- 使用 TypeScript 严格模式
- 函数式编程风格
- 遵循 React Hooks 最佳实践

# Tone
代码简洁，注释精炼

# Audience
中级前端开发者

# Response
提供完整代码文件，包含类型定义和使用示例
```

## 编程 Prompt 模式

### 1. 规格先行模式

先定义接口，再生成实现：

```markdown
请实现以下 TypeScript 接口：

\`\`\`typescript
interface RateLimiter {
  // 检查是否允许请求
  tryAcquire(key: string): Promise<boolean>;
  
  // 获取剩余配额
  getRemainingQuota(key: string): Promise<number>;
  
  // 重置某个 key 的限制
  reset(key: string): Promise<void>;
}

// 配置选项
interface RateLimiterConfig {
  maxRequests: number;  // 最大请求数
  windowMs: number;     // 时间窗口（毫秒）
  storage: 'memory' | 'redis';
}
\`\`\`

要求：
1. 实现滑动窗口算法
2. 支持内存和 Redis 两种存储
3. 线程安全
4. 包含单元测试
```

### 2. 示例驱动模式

通过输入输出示例引导：

```markdown
实现一个日期格式化函数，行为如下：

\`\`\`typescript
// 相对时间格式化
formatRelative(new Date()) // "刚刚"
formatRelative(Date.now() - 30000) // "30秒前"
formatRelative(Date.now() - 3600000) // "1小时前"
formatRelative(Date.now() - 86400000) // "昨天"
formatRelative(Date.now() - 172800000) // "2天前"
formatRelative(Date.now() - 604800000) // "1周前"
formatRelative(Date.now() - 2592000000) // "1个月前"
formatRelative(Date.now() - 31536000000) // "1年前"

// 未来时间
formatRelative(Date.now() + 3600000) // "1小时后"
formatRelative(Date.now() + 86400000) // "明天"

// 绝对时间（超过1年）
formatRelative(Date.now() - 63072000000) // "2023-01-15"
\`\`\`
```

### 3. 约束显式化模式

明确说明限制条件：

```markdown
实现一个缓存类，要求：

## 必须满足
- 最大缓存 1000 个条目
- LRU 淘汰策略
- O(1) 时间复杂度的 get/set
- 支持 TTL 过期

## 禁止使用
- 任何外部库
- ES6 Map 以外的数据结构
- setInterval 定时清理

## 代码约束
- 单文件实现
- 不超过 100 行
- 必须有 TypeScript 类型
```

### 4. 渐进式细化模式

逐步添加复杂度：

```markdown
# 第一轮
实现一个基础的 HTTP 客户端，支持 GET/POST

# 第二轮
添加请求拦截器和响应拦截器

# 第三轮  
添加自动重试机制（指数退避）

# 第四轮
添加请求取消功能（AbortController）

# 第五轮
添加请求缓存（带 TTL）
```

## 高级技巧

### 1. 角色设定

```markdown
你是一位有 15 年经验的 TypeScript 专家，专注于：
- 类型体操和高级类型
- 函数式编程模式
- 性能优化

请以代码审查者的角度，改进以下代码...
```

### 2. 思维链引导

```markdown
请按以下步骤实现功能：

1. 首先，分析需求并列出核心数据结构
2. 然后，设计公共 API 接口
3. 接着，实现核心算法（附上复杂度分析）
4. 最后，添加错误处理和边界情况

每一步请先简要说明思路，再给出代码。
```

### 3. 对比请求

```markdown
请提供两种实现方案：

方案 A：性能优先（可以牺牲一些可读性）
方案 B：可读性优先（可以牺牲一些性能）

并比较两者的优缺点。
```

### 4. 错误预防

```markdown
实现时请注意避免以下常见错误：

1. 忘记处理空数组/null/undefined
2. 浮点数精度问题
3. 异步操作的竞态条件
4. 内存泄漏（事件监听器未移除）
5. 时区问题

对于每个可能的错误点，请在代码中添加注释说明如何处理。
```

## 场景化 Prompt 模板

### 数据库查询优化

```markdown
我有以下 SQL 查询，在数据量大时性能很差：

\`\`\`sql
SELECT u.*, 
  (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count,
  (SELECT SUM(amount) FROM orders WHERE user_id = u.id) as total_amount
FROM users u
WHERE u.created_at > '2024-01-01'
ORDER BY u.created_at DESC
LIMIT 100;
\`\`\`

表结构：
- users: 100万行，有 created_at 索引
- orders: 500万行，有 user_id 索引

请：
1. 分析性能问题
2. 提供优化后的查询
3. 建议需要添加的索引
4. 估算优化后的性能提升
```

### API 设计

```markdown
设计一个 RESTful API，要求：

资源：用户的收藏列表

功能：
- 获取收藏列表（支持分页、筛选、排序）
- 添加收藏
- 移除收藏
- 批量操作
- 检查是否已收藏

请提供：
1. API 端点设计（URL、HTTP 方法）
2. 请求/响应格式（JSON Schema）
3. 错误码设计
4. OpenAPI 3.0 规范文档
```

### 组件开发

```markdown
创建一个 React 组件：可编辑数据表格

功能要求：
- 支持内联编辑单元格
- 支持添加/删除行
- 支持列排序
- 支持多选行
- 支持撤销/重做
- 键盘导航（方向键、Tab、Enter）

技术要求：
- TypeScript
- 使用 React 18 特性
- 无外部 UI 库
- 可访问性（ARIA）
- 单元测试

请先给出组件 API 设计，我确认后再给出完整实现。
```

## Prompt 优化清单

```markdown
## 检查清单

### 明确性
- [ ] 任务目标是否清晰？
- [ ] 输入输出是否明确？
- [ ] 约束条件是否完整？

### 上下文
- [ ] 技术栈是否说明？
- [ ] 项目背景是否提供？
- [ ] 现有代码是否相关？

### 格式
- [ ] 是否指定输出格式？
- [ ] 是否要求代码注释？
- [ ] 是否需要使用示例？

### 质量
- [ ] 是否要求错误处理？
- [ ] 是否要求类型安全？
- [ ] 是否要求测试代码？
```

## 常见错误与修正

| 错误 Prompt | 问题 | 改进版本 |
|-------------|------|----------|
| "写一个登录功能" | 太模糊 | "用 React + TypeScript 实现登录表单，包含邮箱/密码验证，调用 /api/auth/login 接口" |
| "优化这段代码" | 无目标 | "优化这段代码的性能，当前处理 10000 条数据需要 5 秒，目标是 < 500ms" |
| "修复 bug" | 无上下文 | "这个函数在 items 为空时抛出异常，请添加空数组处理并返回空结果" |

## 总结

高质量 Prompt 的要素：

1. **具体**: 避免模糊描述
2. **完整**: 提供必要上下文
3. **结构化**: 使用清晰的格式
4. **可验证**: 明确成功标准

下一篇，我们将探讨 AI 辅助调试与测试的技巧。
