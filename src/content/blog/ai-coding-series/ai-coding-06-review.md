---
title: 'AI 代码审查与重构'
description: '学习如何利用 AI 工具进行高效的代码审查和安全重构，提升代码质量。'
pubDate: 2025-04-05
category: '技术'
tags: ['AI编程', '代码审查', '重构', '代码质量']
series: 'AI Coding 完全指南'
seriesOrder: 6
heroImage: 'https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=800&h=400&fit=crop'
---

AI 不仅能帮我们写代码，更能帮我们审查和改进代码。本文将介绍如何利用 AI 工具进行系统性的代码审查和安全重构。

## AI 代码审查的优势

### 为什么需要 AI 审查

| 人工审查 | AI 审查 |
|----------|---------|
| 容易疲劳 | 始终如一 |
| 主观偏见 | 客观一致 |
| 时间有限 | 即时响应 |
| 知识受限 | 博学多才 |

### AI 审查的适用场景

- ✅ 安全漏洞扫描
- ✅ 性能问题检测
- ✅ 代码风格统一
- ✅ 重复代码识别
- ⚠️ 业务逻辑验证（需人工）
- ⚠️ 架构决策（需讨论）

## 代码审查实战

### 安全审查 Prompt

```markdown
请审查以下代码的安全性问题，关注：

1. **注入攻击**
   - SQL 注入
   - XSS
   - 命令注入

2. **认证与授权**
   - 硬编码凭证
   - 权限检查缺失
   - Session 管理

3. **数据处理**
   - 敏感信息泄露
   - 不安全的反序列化
   - 路径遍历

4. **加密与随机数**
   - 弱加密算法
   - 不安全的随机数
   - 明文存储敏感数据

请用以下格式输出：
- 🔴 严重 (必须修复)
- 🟡 中等 (建议修复)
- 🟢 低危 (可选优化)
```

### 实际审查示例

```typescript
// 待审查代码
class UserService {
  async login(username: string, password: string) {
    const query = `SELECT * FROM users WHERE username = '${username}'`;
    const user = await db.query(query);
    
    if (user && user.password === password) {
      const token = Math.random().toString(36);
      return { token, user };
    }
    throw new Error('Invalid credentials');
  }

  async updateProfile(userId: string, data: any) {
    await db.query(`UPDATE users SET ${Object.entries(data).map(([k,v]) => `${k}='${v}'`).join(',')} WHERE id='${userId}'`);
  }
}
```

**AI 审查结果**:

```markdown
## 代码审查报告

### 🔴 严重问题

#### 1. SQL 注入漏洞 (第 3 行, 第 13 行)
**风险**: 攻击者可以执行任意 SQL 命令

```typescript
// ❌ 当前代码
const query = `SELECT * FROM users WHERE username = '${username}'`;

// ✅ 修复方案
const query = 'SELECT * FROM users WHERE username = ?';
const user = await db.query(query, [username]);
```

#### 2. 明文密码比较 (第 6 行)
**风险**: 密码应该经过哈希处理后比较

```typescript
// ❌ 当前代码
if (user.password === password)

// ✅ 修复方案
import bcrypt from 'bcrypt';
const isValid = await bcrypt.compare(password, user.passwordHash);
```

### 🟡 中等问题

#### 3. 不安全的随机数 (第 7 行)
**风险**: Math.random() 不适合生成安全令牌

```typescript
// ❌ 当前代码
const token = Math.random().toString(36);

// ✅ 修复方案
import crypto from 'crypto';
const token = crypto.randomBytes(32).toString('hex');
```

#### 4. 无类型校验 (第 11 行)
**风险**: 任意字段注入

```typescript
// ❌ 当前代码
async updateProfile(userId: string, data: any)

// ✅ 修复方案
interface ProfileUpdate {
  name?: string;
  email?: string;
  avatar?: string;
}
async updateProfile(userId: string, data: ProfileUpdate)
```

### 修复后代码

```typescript
import bcrypt from 'bcrypt';
import crypto from 'crypto';
import { z } from 'zod';

const ProfileUpdateSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  email: z.string().email().optional(),
  avatar: z.string().url().optional(),
});

class UserService {
  async login(username: string, password: string) {
    const query = 'SELECT * FROM users WHERE username = ?';
    const user = await db.query(query, [username]);
    
    if (user && await bcrypt.compare(password, user.passwordHash)) {
      const token = crypto.randomBytes(32).toString('hex');
      const { passwordHash, ...safeUser } = user;
      return { token, user: safeUser };
    }
    throw new Error('Invalid credentials');
  }

  async updateProfile(userId: string, data: unknown) {
    const validated = ProfileUpdateSchema.parse(data);
    const updates = Object.entries(validated)
      .filter(([, v]) => v !== undefined)
      .map(([k]) => `${k} = ?`);
    const values = Object.values(validated).filter(v => v !== undefined);
    
    await db.query(
      `UPDATE users SET ${updates.join(', ')} WHERE id = ?`,
      [...values, userId]
    );
  }
}
```
```

## AI 辅助重构

### 重构策略

```markdown
## 重构原则

1. **小步重构**: 每次只做一个改动
2. **测试先行**: 确保有测试覆盖
3. **保持功能**: 重构不改变行为
4. **及时提交**: 每个成功的重构都提交
```

### 常见重构模式

#### 1. 提取函数

```typescript
// 重构前
async function processOrder(order: Order) {
  // 验证订单 - 30行
  if (!order.items.length) throw new Error('Empty order');
  if (order.total < 0) throw new Error('Invalid total');
  // ... 更多验证
  
  // 计算折扣 - 20行
  let discount = 0;
  if (order.coupon) {
    // 复杂计算...
  }
  
  // 处理支付 - 40行
  // ...
}

// AI 重构后
async function processOrder(order: Order) {
  validateOrder(order);
  const discount = calculateDiscount(order);
  await processPayment(order, discount);
}

function validateOrder(order: Order): void {
  if (!order.items.length) throw new Error('Empty order');
  if (order.total < 0) throw new Error('Invalid total');
  // ...
}

function calculateDiscount(order: Order): number {
  if (!order.coupon) return 0;
  // ...
}

async function processPayment(order: Order, discount: number): Promise<void> {
  // ...
}
```

#### 2. 消除重复

```typescript
// 重构前 - 多处类似代码
async function getUsers() {
  try {
    const response = await fetch('/api/users');
    if (!response.ok) throw new Error('Failed');
    return response.json();
  } catch (error) {
    logger.error('API Error:', error);
    throw error;
  }
}

async function getProducts() {
  try {
    const response = await fetch('/api/products');
    if (!response.ok) throw new Error('Failed');
    return response.json();
  } catch (error) {
    logger.error('API Error:', error);
    throw error;
  }
}

// AI 重构后
async function fetchApi<T>(endpoint: string): Promise<T> {
  try {
    const response = await fetch(endpoint);
    if (!response.ok) {
      throw new ApiError(`Request failed: ${response.status}`, response.status);
    }
    return response.json();
  } catch (error) {
    logger.error(`API Error [${endpoint}]:`, error);
    throw error;
  }
}

const getUsers = () => fetchApi<User[]>('/api/users');
const getProducts = () => fetchApi<Product[]>('/api/products');
```

#### 3. 简化条件

```typescript
// 重构前
function getShippingCost(order: Order): number {
  if (order.total > 100) {
    if (order.isPremium) {
      return 0;
    } else {
      return 5;
    }
  } else {
    if (order.isPremium) {
      return 5;
    } else {
      if (order.total > 50) {
        return 8;
      } else {
        return 12;
      }
    }
  }
}

// AI 重构后
function getShippingCost(order: Order): number {
  const { total, isPremium } = order;
  
  // 使用查找表简化逻辑
  const shippingRules = [
    { condition: () => total > 100 && isPremium, cost: 0 },
    { condition: () => total > 100, cost: 5 },
    { condition: () => isPremium, cost: 5 },
    { condition: () => total > 50, cost: 8 },
    { condition: () => true, cost: 12 },
  ];
  
  return shippingRules.find(rule => rule.condition())!.cost;
}
```

## 集成到工作流

### PR 审查自动化

```yaml
# .github/workflows/ai-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: AI Review
        uses: coderabbitai/ai-pr-reviewer@latest
        with:
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          review_comment_lgtm: false
          path_filters: |
            !**/test/**
            !**/*.test.ts
```

### 本地审查脚本

```bash
#!/bin/bash
# review.sh - 本地 AI 审查脚本

# 获取改动的文件
changed_files=$(git diff --name-only HEAD~1)

# 对每个文件进行审查
for file in $changed_files; do
  echo "Reviewing: $file"
  claude -p "审查这个文件的代码质量和安全性: $(cat $file)"
done
```

## 审查清单模板

```markdown
## AI 代码审查清单

### 安全性
- [ ] 无 SQL 注入风险
- [ ] 无 XSS 漏洞
- [ ] 敏感数据已加密
- [ ] 权限检查完整

### 性能
- [ ] 无 N+1 查询
- [ ] 大数据集有分页
- [ ] 无内存泄漏风险

### 可维护性
- [ ] 函数长度 < 50 行
- [ ] 圈复杂度 < 10
- [ ] 有必要的注释
- [ ] 命名清晰准确

### 测试
- [ ] 核心逻辑有测试
- [ ] 边界条件有覆盖
- [ ] 测试可读性好
```

## 总结

AI 代码审查的最佳实践：

1. **结合人工**: AI 发现问题，人工决策
2. **持续集成**: 自动化到 CI/CD
3. **建立标准**: 统一的审查清单
4. **及时修复**: 发现问题立即处理

下一篇，我们将深入探讨 Prompt Engineering for Coding —— 如何写出让 AI 更好理解的指令。
