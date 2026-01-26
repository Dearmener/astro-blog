---
title: 'Cursor IDE 完全攻略'
description: '深入探索 Cursor IDE 的 AI 功能，包括 Composer、Agent 模式和项目级代码生成。'
pubDate: 2025-03-15
category: '技术'
tags: ['AI编程', 'Cursor', 'IDE', 'AI Agent']
series: 'AI Coding 完全指南'
seriesOrder: 3
heroImage: 'https://images.unsplash.com/photo-1607799279861-4dd421887fb3?w=800&h=400&fit=crop'
---

Cursor 是第一个专为 AI 编程设计的 IDE，基于 VS Code 构建但提供了远超普通扩展的 AI 能力。2025 年，Cursor 已成为全栈开发者的首选工具。

## Cursor vs VS Code + Copilot

| 特性 | Cursor | VS Code + Copilot |
|------|--------|-------------------|
| 代码补全 | ✅ 更智能 | ✅ 标准 |
| Chat 对话 | ✅ 支持多模型 | ✅ 仅 GPT |
| 项目级编辑 | ✅ Composer | ❌ 不支持 |
| Agent 模式 | ✅ 自动执行 | ❌ 不支持 |
| 代码库索引 | ✅ 内置 | ⚠️ 需配置 |
| 多文件重构 | ✅ 原生支持 | ⚠️ 手动操作 |

## 核心功能详解

### 1. Inline Edit (Cmd+K)

在代码中直接编辑，无需切换上下文：

```typescript
// 选中以下代码，按 Cmd+K，输入 "添加错误重试机制，最多重试3次"

async function fetchData(url: string) {
  const response = await fetch(url);
  return response.json();
}

// Cursor 生成 ↓

async function fetchData(url: string, maxRetries = 3): Promise<any> {
  let lastError: Error | null = null;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      lastError = error as Error;
      console.warn(`Attempt ${attempt}/${maxRetries} failed:`, error);
      
      if (attempt < maxRetries) {
        // 指数退避
        await new Promise(resolve => 
          setTimeout(resolve, Math.pow(2, attempt) * 1000)
        );
      }
    }
  }
  
  throw new Error(`Failed after ${maxRetries} attempts: ${lastError?.message}`);
}
```

### 2. Chat (Cmd+L)

智能对话，支持代码引用：

```
用户: @UserService.ts 中的 createUser 方法，
      帮我添加邮箱验证功能，发送验证邮件后返回 pending 状态

Cursor: 我来帮你修改 createUser 方法...

[自动生成代码并解释改动]
```

**Chat 引用语法**：
- `@filename` - 引用文件
- `@symbol` - 引用函数/类
- `@folder` - 引用文件夹
- `@docs` - 引用外部文档
- `@web` - 搜索网页

### 3. Composer (Cmd+I)

项目级代码生成，Cursor 最强大的功能：

```markdown
## Composer 指令示例

创建一个完整的用户认证模块：

1. 创建 src/auth/types.ts - 定义 User, Session, AuthToken 类型
2. 创建 src/auth/jwt.ts - JWT 生成和验证
3. 创建 src/auth/middleware.ts - Express 认证中间件
4. 创建 src/auth/routes.ts - 登录、注册、刷新 token 路由
5. 更新 src/app.ts - 集成认证路由
6. 创建 src/auth/__tests__/auth.test.ts - 单元测试

要求：
- 使用 TypeScript 严格模式
- 密码使用 bcrypt 加密
- JWT 包含 accessToken 和 refreshToken
- 遵循现有代码风格
```

Composer 会：
1. 分析现有代码结构
2. 创建所有文件
3. 保持代码风格一致
4. 处理导入导出关系

### 4. Agent 模式

让 Cursor 自动执行任务：

```
用户: 帮我把所有 API 路由从 callback 风格改成 async/await，
      并添加统一的错误处理

Cursor Agent:
✓ 扫描项目找到 15 个路由文件
✓ 分析现有错误处理模式
✓ 逐个转换路由
✓ 添加 asyncHandler 包装器
✓ 运行测试确认无破坏
✓ 生成变更摘要
```

## 高级配置

### .cursorrules 文件

在项目根目录创建 `.cursorrules`：

```markdown
# Cursor Rules for This Project

## Code Style
- Use TypeScript strict mode
- Prefer functional components with hooks
- Use named exports over default exports
- Maximum line length: 100 characters

## Naming Conventions
- Components: PascalCase (UserProfile.tsx)
- Hooks: camelCase with 'use' prefix (useAuth.ts)
- Utils: camelCase (formatDate.ts)
- Constants: SCREAMING_SNAKE_CASE

## Architecture Patterns
- Services in src/services/
- API routes in src/api/
- Shared types in src/types/
- React components in src/components/

## Testing
- Test files next to source: Component.test.tsx
- Use React Testing Library
- Prefer integration tests over unit tests

## Forbidden
- No any types
- No console.log in production code
- No inline styles in React components
```

### 模型选择策略

```json
// settings.json
{
  "cursor.chat.defaultModel": "claude-3.5-sonnet",
  "cursor.composer.defaultModel": "claude-3.5-sonnet",
  "cursor.autocomplete.model": "cursor-fast",
  
  "cursor.modelPreferences": {
    "complex-refactoring": "claude-3.5-sonnet",
    "quick-fixes": "gpt-4o-mini",
    "documentation": "gpt-4o"
  }
}
```

## 实战工作流

### 场景：添加新功能

```bash
# 1. 用自然语言描述需求
Cmd+I (Composer)
"添加用户头像上传功能，支持裁剪和压缩，保存到 S3"

# 2. 审查生成的代码
Cursor 创建:
- src/components/AvatarUploader.tsx
- src/services/imageService.ts
- src/services/s3Service.ts
- src/api/upload.ts

# 3. 针对具体文件优化
Cmd+K on AvatarUploader.tsx
"添加拖拽上传支持"

# 4. 生成测试
Cmd+L
"@AvatarUploader.tsx 生成完整的组件测试"

# 5. 更新文档
Cmd+L
"根据新增的上传功能更新 README"
```

### 场景：Debug 复杂问题

```typescript
// 选中报错代码，Cmd+L

用户: 这段代码在生产环境偶发 "Maximum call stack size exceeded"，
      但本地无法复现，帮我分析可能的原因

Cursor: 我分析了代码，发现几个潜在问题：

1. **循环引用检测缺失**
   在 deepMerge 函数中，如果对象存在循环引用...
   
2. **递归终止条件不完整**
   当 source 为 null 但 target 为对象时...

[提供修复代码和测试用例]
```

## 快捷键速查

| 功能 | Mac | Windows |
|------|-----|---------|
| Inline Edit | `Cmd+K` | `Ctrl+K` |
| Chat | `Cmd+L` | `Ctrl+L` |
| Composer | `Cmd+I` | `Ctrl+I` |
| 接受建议 | `Tab` | `Tab` |
| 拒绝建议 | `Esc` | `Esc` |
| 下一个建议 | `Cmd+]` | `Ctrl+]` |
| 添加到 Chat | `Cmd+Shift+L` | `Ctrl+Shift+L` |

## Cursor 订阅方案

| Plan | 价格 | 特点 |
|------|------|------|
| Free | $0 | 2000 补全/月，50 慢速请求 |
| Pro | $20/月 | 无限补全，500 快速请求 |
| Business | $40/月 | 团队功能，优先支持 |

## 最佳实践

### 1. 分步指令

```
❌ "重构整个项目，添加 TypeScript，修复所有 bug，添加测试"

✅ 第一步: "将 src/utils/ 下的文件转换为 TypeScript"
   第二步: "为转换后的文件添加单元测试"
   第三步: "修复 TypeScript 报告的类型错误"
```

### 2. 提供上下文

```
❌ "修复这个 bug"

✅ "这个函数在 items 为空数组时返回 undefined 而不是空数组，
    这导致下游的 map 调用报错。请修复并添加边界条件测试"
```

### 3. 验证生成代码

```typescript
// Cursor 生成后，始终验证：
// 1. 类型是否正确
// 2. 边界条件是否处理
// 3. 错误处理是否完善
// 4. 是否符合项目规范
```

## 总结

Cursor 的核心优势：

1. **Composer** - 项目级代码生成
2. **Agent 模式** - 自动化复杂任务
3. **深度索引** - 理解整个代码库
4. **多模型支持** - 根据任务选择最佳模型

下一篇，我们将探索 Claude Code (Claude CLI) —— Anthropic 官方的命令行编程工具。
