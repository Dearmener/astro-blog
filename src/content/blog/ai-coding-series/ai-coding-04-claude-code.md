---
title: 'Claude Code (Claude CLI) 实战'
description: '掌握 Anthropic 官方 AI 编程工具 Claude Code，学习 Agent 模式和复杂项目开发。'
pubDate: 2025-03-22
category: '技术'
tags: ['AI编程', 'Claude', 'CLI', 'Anthropic']
series: 'AI Coding 完全指南'
seriesOrder: 4
heroImage: 'https://images.unsplash.com/photo-1629654297299-c8506221ca97?w=800&h=400&fit=crop'
---

Claude Code 是 Anthropic 官方发布的命令行 AI 编程工具，它将 Claude 的强大能力带入终端，支持复杂的多文件操作和 Agent 自主执行。

## Claude Code 简介

### 核心特点

- **终端原生**: 在你熟悉的终端环境中工作
- **Agent 模式**: 自主规划和执行复杂任务
- **多文件操作**: 一次性修改整个项目
- **Git 集成**: 自动提交和版本管理
- **MCP 支持**: 连接外部工具和数据源

### 安装与配置

```bash
# 通过 npm 安装
npm install -g @anthropic-ai/claude-code

# 或使用 Homebrew (macOS)
brew install claude-code

# 登录认证
claude login

# 验证安装
claude --version
```

## 基础用法

### 启动会话

```bash
# 在当前目录启动
claude

# 指定项目目录
claude --project ./my-project

# 使用特定模型
claude --model claude-3.5-sonnet
```

### 基本命令

```bash
# 在 Claude Code 会话中

> 帮我创建一个 Express API 项目，包含用户 CRUD

Claude: 我来为你创建项目结构...

📁 创建文件:
  - src/index.ts
  - src/routes/users.ts
  - src/models/User.ts
  - src/middleware/auth.ts
  - package.json
  - tsconfig.json
  
> /run npm install

> /run npm run dev
```

### 常用斜杠命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助信息 |
| `/clear` | 清除对话历史 |
| `/compact` | 压缩上下文 |
| `/run <cmd>` | 执行 shell 命令 |
| `/git` | Git 操作 |
| `/add <file>` | 添加文件到上下文 |
| `/web <url>` | 抓取网页内容 |

## Agent 模式详解

### 自主任务执行

```bash
> 帮我把这个 JavaScript 项目迁移到 TypeScript，包括：
> 1. 安装必要的依赖
> 2. 创建 tsconfig.json
> 3. 重命名所有 .js 文件为 .ts
> 4. 添加类型注解
> 5. 修复所有类型错误
> 6. 确保所有测试通过

Claude Agent:
🔍 分析项目结构...
   - 发现 23 个 JavaScript 文件
   - 发现 15 个测试文件
   - 使用 Jest 作为测试框架

📦 安装依赖...
   npm install -D typescript @types/node @types/jest ts-jest

📝 创建配置文件...
   ✓ tsconfig.json
   ✓ jest.config.js 更新

🔄 转换文件...
   [=================>] 23/23 完成

🔧 修复类型错误...
   - src/utils/date.ts: 添加返回类型
   - src/services/api.ts: 修复泛型参数
   - src/models/user.ts: 添加接口定义
   ...

🧪 运行测试...
   ✓ 47 tests passed

✅ 迁移完成！生成迁移报告...
```

### 任务分解

Claude Code 会自动将复杂任务分解为步骤：

```bash
> 为这个 React 项目添加完整的认证系统

Claude 规划:
┌─────────────────────────────────────┐
│ 任务分解                              │
├─────────────────────────────────────┤
│ 1. 创建认证相关类型定义                  │
│ 2. 实现 JWT 工具函数                    │
│ 3. 创建 AuthContext                   │
│ 4. 实现登录/注册组件                    │
│ 5. 添加路由保护 HOC                     │
│ 6. 集成到现有路由                       │
│ 7. 编写单元测试                        │
│ 8. 更新文档                           │
└─────────────────────────────────────┘

是否继续执行？[Y/n]
```

## 实战案例

### 案例 1: 代码审查

```bash
> /add src/api/payment.ts
> 帮我审查这个支付处理代码，关注安全性问题

Claude:
## 代码审查报告

### 🔴 严重问题

1. **SQL 注入风险** (第 45 行)
   ```typescript
   // 当前代码
   const query = `SELECT * FROM orders WHERE id = ${orderId}`;
   
   // 建议修复
   const query = 'SELECT * FROM orders WHERE id = ?';
   const result = await db.query(query, [orderId]);
   ```

2. **敏感信息日志** (第 78 行)
   ```typescript
   // 当前代码 - 记录了完整卡号
   logger.info(`Processing card: ${cardNumber}`);
   
   // 建议修复
   logger.info(`Processing card: ****${cardNumber.slice(-4)}`);
   ```

### 🟡 中等问题

3. **缺少金额验证** (第 23 行)
   添加金额范围和精度验证...

### 🟢 建议改进

4. **添加幂等性检查**
   防止重复支付...

是否自动应用修复？[Y/n]
```

### 案例 2: 性能优化

```bash
> 分析 src/components/DataGrid.tsx 的性能问题

Claude:
## 性能分析报告

### 问题 1: 不必要的重渲染
位置: 第 34-56 行

```tsx
// 当前代码 - 每次渲染都创建新对象
const columns = [
  { field: 'id', headerName: 'ID' },
  { field: 'name', headerName: 'Name' },
];

// 优化后 - 使用 useMemo
const columns = useMemo(() => [
  { field: 'id', headerName: 'ID' },
  { field: 'name', headerName: 'Name' },
], []);
```

### 问题 2: 大数据集未虚拟化
位置: 第 78 行

建议使用 react-window 或 react-virtualized...

### 问题 3: 过度的 useEffect 依赖
位置: 第 92 行

```tsx
// 当前 - 每次 data 变化都重新排序
useEffect(() => {
  setSortedData(sortData(data));
}, [data, sortConfig]);

// 优化 - 使用 useMemo 替代
const sortedData = useMemo(
  () => sortData(data, sortConfig),
  [data, sortConfig]
);
```

预计优化后性能提升: ~40%
```

### 案例 3: 测试生成

```bash
> /add src/services/orderService.ts
> 为 OrderService 生成完整的单元测试，覆盖所有边界情况

Claude:
```typescript
// src/services/__tests__/orderService.test.ts

import { OrderService } from '../orderService';
import { mockDb, mockPaymentGateway } from '../../__mocks__';

describe('OrderService', () => {
  let service: OrderService;

  beforeEach(() => {
    service = new OrderService(mockDb, mockPaymentGateway);
    jest.clearAllMocks();
  });

  describe('createOrder', () => {
    it('should create order with valid items', async () => {
      const items = [{ productId: '1', quantity: 2 }];
      const order = await service.createOrder('user123', items);
      
      expect(order.status).toBe('pending');
      expect(order.items).toHaveLength(1);
    });

    it('should throw error for empty items', async () => {
      await expect(service.createOrder('user123', []))
        .rejects.toThrow('Order must have at least one item');
    });

    it('should throw error for invalid quantity', async () => {
      const items = [{ productId: '1', quantity: 0 }];
      
      await expect(service.createOrder('user123', items))
        .rejects.toThrow('Quantity must be positive');
    });

    it('should handle out of stock items', async () => {
      mockDb.products.findById.mockResolvedValue({ stock: 0 });
      const items = [{ productId: '1', quantity: 1 }];
      
      await expect(service.createOrder('user123', items))
        .rejects.toThrow('Product 1 is out of stock');
    });
  });

  describe('processPayment', () => {
    it('should process payment and update order status', async () => {
      mockPaymentGateway.charge.mockResolvedValue({ success: true });
      
      const result = await service.processPayment('order123', {
        method: 'card',
        token: 'tok_xxx'
      });
      
      expect(result.status).toBe('paid');
      expect(mockPaymentGateway.charge).toHaveBeenCalledTimes(1);
    });

    it('should handle payment failure gracefully', async () => {
      mockPaymentGateway.charge.mockResolvedValue({ 
        success: false, 
        error: 'Insufficient funds' 
      });
      
      const result = await service.processPayment('order123', {
        method: 'card',
        token: 'tok_xxx'
      });
      
      expect(result.status).toBe('payment_failed');
      expect(result.error).toBe('Insufficient funds');
    });
  });
});
```

测试覆盖率: 94%
```

## 高级功能

### MCP (Model Context Protocol)

连接外部工具和数据源：

```bash
# 配置 MCP 服务器
claude mcp add github --token $GITHUB_TOKEN
claude mcp add postgres --connection-string $DATABASE_URL

# 在会话中使用
> 查看 main 分支最近的 PR，分析代码变更

> 查询 users 表中最近注册的 100 个用户
```

### 自定义 Skills

创建 `~/.claude/skills/code-review/SKILL.md`:

```markdown
---
name: code-review
description: 执行代码审查时使用此技能
---

## 审查清单

1. **安全性**
   - SQL 注入
   - XSS 攻击
   - 敏感信息泄露

2. **性能**
   - N+1 查询
   - 内存泄漏
   - 不必要的渲染

3. **代码质量**
   - 命名规范
   - 函数复杂度
   - 测试覆盖

## 输出格式

使用 emoji 标记严重程度:
- 🔴 严重
- 🟡 中等
- 🟢 建议
```

### Git 集成

```bash
> /git status

> 帮我创建一个原子提交，分离出登录功能的改动

Claude:
正在分析变更...

建议分为 3 个提交:
1. feat(auth): add login form component
2. feat(auth): implement JWT authentication service  
3. feat(auth): add protected route wrapper

是否执行？[Y/n]
```

## 配置文件

### ~/.claude/config.yaml

```yaml
model: claude-3.5-sonnet
temperature: 0.2

auto_commit: false
auto_run_commands: false  # 安全起见默认关闭

themes:
  dark: true
  
keybindings:
  accept: enter
  cancel: ctrl+c
  
ignore_patterns:
  - node_modules
  - .git
  - "*.log"
  - dist
```

## 与其他工具对比

| 特性 | Claude Code | Cursor | Copilot CLI |
|------|-------------|--------|-------------|
| 终端原生 | ✅ | ❌ | ✅ |
| 多文件操作 | ✅ | ✅ | ❌ |
| Agent 模式 | ✅ | ✅ | ❌ |
| Git 集成 | ✅ | ⚠️ | ⚠️ |
| MCP 支持 | ✅ | ❌ | ❌ |

## 总结

Claude Code 的核心优势：

1. **终端友好** - 符合 Unix 哲学
2. **Agent 能力** - 自主完成复杂任务
3. **Git 原生** - 自动管理版本
4. **可扩展** - MCP 连接一切

下一篇，我们将对比 Windsurf 与 Codeium，看看这个免费替代品的实力。
