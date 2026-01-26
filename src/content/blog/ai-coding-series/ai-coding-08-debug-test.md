---
title: 'AI 辅助调试与测试'
description: '学习如何利用 AI 工具进行高效的 Bug 调试和自动化测试生成。'
pubDate: 2025-04-19
category: '技术'
tags: ['AI编程', '调试', '测试', '自动化']
series: 'AI Coding 完全指南'
seriesOrder: 8
heroImage: 'https://images.unsplash.com/photo-1504639725590-34d0984388bd?w=800&h=400&fit=crop'
---

调试和测试是开发中最耗时的环节之一。AI 工具可以大幅提升这两个环节的效率。本文将介绍 AI 辅助调试和测试的实战技巧。

## AI 辅助调试

### 错误分析 Prompt 模板

```markdown
我遇到了以下错误：

**错误信息**:
\`\`\`
TypeError: Cannot read property 'map' of undefined
    at UserList (UserList.tsx:15:23)
    at renderWithHooks (react-dom.development.js:14985:18)
\`\`\`

**相关代码**:
\`\`\`typescript
// UserList.tsx
function UserList({ users }) {
  return (
    <ul>
      {users.map(user => <li key={user.id}>{user.name}</li>)}
    </ul>
  );
}
\`\`\`

**上下文**:
- 组件首次渲染时报错
- users 数据从 API 获取
- 使用 React 18 + TypeScript

请分析：
1. 错误原因
2. 修复方案
3. 如何防止类似问题
```

### 复杂 Bug 调试流程

```typescript
// 场景：内存泄漏调试

// 1. 描述问题
/*
问题：React 应用运行一段时间后内存持续增长
环境：React 18, Chrome 120
复现步骤：
1. 打开页面
2. 在列表页和详情页之间反复切换
3. 观察 Chrome DevTools Memory 面板
*/

// 2. 提供可疑代码
function useWebSocket(url: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      setMessages(prev => [...prev, JSON.parse(event.data)]);
    };
    
    // 可疑：没有清理函数
  }, [url]);
  
  return messages;
}

// 3. AI 分析结果
/*
AI 诊断：

🔴 问题 1：WebSocket 未关闭
当组件卸载时，WebSocket 连接仍然保持，导致：
- 连接累积
- 事件监听器累积
- 闭包持有旧状态

🔴 问题 2：消息数组无限增长
messages 数组只增不减，长时间运行会占用大量内存

修复代码：
*/

function useWebSocket(url: string, maxMessages = 100) {
  const [messages, setMessages] = useState<Message[]>([]);
  
  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onmessage = (event) => {
      setMessages(prev => {
        const newMessages = [...prev, JSON.parse(event.data)];
        // 限制消息数量
        return newMessages.slice(-maxMessages);
      });
    };
    
    // ✅ 清理函数
    return () => {
      ws.close();
    };
  }, [url, maxMessages]);
  
  return messages;
}
```

### 性能问题诊断

```markdown
**问题描述**:
React 列表组件在数据量大时渲染卡顿

**性能数据**:
- 100 条数据：渲染时间 50ms ✓
- 1000 条数据：渲染时间 800ms ✗
- 10000 条数据：渲染时间 12s ✗

**代码**:
\`\`\`tsx
function ProductList({ products }) {
  const [filter, setFilter] = useState('');
  
  const filtered = products.filter(p => 
    p.name.toLowerCase().includes(filter.toLowerCase())
  );
  
  return (
    <div>
      <input onChange={e => setFilter(e.target.value)} />
      {filtered.map(product => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
}
```

**AI 优化方案**:

```tsx
import { useMemo, useCallback, memo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useDeferredValue } from 'react';

// 1. 使用 memo 避免子组件重渲染
const ProductCard = memo(function ProductCard({ product }) {
  return <div>{/* ... */}</div>;
});

function ProductList({ products }) {
  const [filter, setFilter] = useState('');
  // 2. 使用 useDeferredValue 延迟非紧急更新
  const deferredFilter = useDeferredValue(filter);
  
  // 3. 使用 useMemo 缓存计算结果
  const filtered = useMemo(() => {
    const lowerFilter = deferredFilter.toLowerCase();
    return products.filter(p => 
      p.name.toLowerCase().includes(lowerFilter)
    );
  }, [products, deferredFilter]);
  
  // 4. 使用虚拟列表
  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 80,
  });
  
  return (
    <div>
      <input onChange={e => setFilter(e.target.value)} />
      <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
        <div style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map(virtualRow => (
            <ProductCard
              key={filtered[virtualRow.index].id}
              product={filtered[virtualRow.index]}
              style={{
                position: 'absolute',
                top: virtualRow.start,
                height: virtualRow.size,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
```

## AI 生成测试

### 单元测试生成

```markdown
为以下函数生成完整的单元测试：

\`\`\`typescript
function parseQueryString(url: string): Record<string, string | string[]> {
  const params: Record<string, string | string[]> = {};
  const queryString = url.split('?')[1];
  
  if (!queryString) return params;
  
  queryString.split('&').forEach(pair => {
    const [key, value] = pair.split('=').map(decodeURIComponent);
    
    if (params[key]) {
      if (Array.isArray(params[key])) {
        (params[key] as string[]).push(value);
      } else {
        params[key] = [params[key] as string, value];
      }
    } else {
      params[key] = value;
    }
  });
  
  return params;
}
\`\`\`

要求：
1. 使用 Jest
2. 覆盖所有分支
3. 包含边界情况
4. 覆盖率 > 95%
```

**AI 生成的测试**:

```typescript
describe('parseQueryString', () => {
  describe('基础功能', () => {
    it('should parse simple query string', () => {
      const result = parseQueryString('https://example.com?name=john&age=30');
      expect(result).toEqual({ name: 'john', age: '30' });
    });

    it('should handle single parameter', () => {
      const result = parseQueryString('https://example.com?key=value');
      expect(result).toEqual({ key: 'value' });
    });
  });

  describe('数组参数', () => {
    it('should handle duplicate keys as array', () => {
      const result = parseQueryString('https://example.com?tag=a&tag=b&tag=c');
      expect(result).toEqual({ tag: ['a', 'b', 'c'] });
    });

    it('should handle mixed single and duplicate keys', () => {
      const result = parseQueryString('https://example.com?name=john&tag=a&tag=b');
      expect(result).toEqual({ name: 'john', tag: ['a', 'b'] });
    });
  });

  describe('URL 编码', () => {
    it('should decode URL encoded values', () => {
      const result = parseQueryString('https://example.com?name=John%20Doe&city=%E5%8C%97%E4%BA%AC');
      expect(result).toEqual({ name: 'John Doe', city: '北京' });
    });

    it('should decode URL encoded keys', () => {
      const result = parseQueryString('https://example.com?user%20name=john');
      expect(result).toEqual({ 'user name': 'john' });
    });
  });

  describe('边界情况', () => {
    it('should return empty object for URL without query string', () => {
      expect(parseQueryString('https://example.com')).toEqual({});
    });

    it('should return empty object for URL with empty query string', () => {
      expect(parseQueryString('https://example.com?')).toEqual({});
    });

    it('should handle empty value', () => {
      const result = parseQueryString('https://example.com?key=');
      expect(result).toEqual({ key: '' });
    });

    it('should handle key without value', () => {
      const result = parseQueryString('https://example.com?key');
      expect(result).toEqual({ key: undefined });
    });

    it('should handle special characters', () => {
      const result = parseQueryString('https://example.com?q=a+b&x=1%2B1');
      expect(result).toEqual({ q: 'a b', x: '1+1' });
    });
  });

  describe('复杂场景', () => {
    it('should handle fragment identifier', () => {
      const result = parseQueryString('https://example.com?key=value#section');
      // 注意：当前实现可能需要处理 hash
      expect(result.key).toContain('value');
    });

    it('should handle multiple question marks', () => {
      const result = parseQueryString('https://example.com?q=what?');
      expect(result).toEqual({ q: 'what?' });
    });
  });
});
```

### 集成测试生成

```markdown
为以下 API 端点生成集成测试：

\`\`\`typescript
// POST /api/orders
// 创建订单

// 请求体
interface CreateOrderRequest {
  items: Array<{ productId: string; quantity: number }>;
  shippingAddress: Address;
  paymentMethod: 'card' | 'paypal';
}

// 响应
interface CreateOrderResponse {
  orderId: string;
  status: 'pending' | 'confirmed';
  total: number;
  estimatedDelivery: string;
}
\`\`\`

测试要求：
- 使用 supertest
- 测试成功和失败场景
- Mock 外部支付服务
- 验证数据库状态
```

**AI 生成的集成测试**:

```typescript
import request from 'supertest';
import { app } from '../app';
import { db } from '../database';
import { paymentService } from '../services/payment';

jest.mock('../services/payment');

describe('POST /api/orders', () => {
  const validOrder = {
    items: [
      { productId: 'prod_1', quantity: 2 },
      { productId: 'prod_2', quantity: 1 },
    ],
    shippingAddress: {
      street: '123 Main St',
      city: 'San Francisco',
      zip: '94102',
    },
    paymentMethod: 'card',
  };

  beforeEach(async () => {
    await db.seed();
    jest.clearAllMocks();
  });

  afterAll(async () => {
    await db.cleanup();
  });

  describe('成功场景', () => {
    it('should create order with valid data', async () => {
      (paymentService.authorize as jest.Mock).mockResolvedValue({ 
        success: true, 
        transactionId: 'txn_123' 
      });

      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer valid_token')
        .send(validOrder)
        .expect(201);

      expect(response.body).toMatchObject({
        orderId: expect.any(String),
        status: 'confirmed',
        total: expect.any(Number),
        estimatedDelivery: expect.any(String),
      });

      // 验证数据库状态
      const order = await db.orders.findById(response.body.orderId);
      expect(order).not.toBeNull();
      expect(order.items).toHaveLength(2);
    });

    it('should calculate correct total', async () => {
      (paymentService.authorize as jest.Mock).mockResolvedValue({ success: true });

      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer valid_token')
        .send(validOrder)
        .expect(201);

      // prod_1: $10 x 2 = $20, prod_2: $15 x 1 = $15
      expect(response.body.total).toBe(35);
    });
  });

  describe('验证错误', () => {
    it('should reject empty items', async () => {
      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer valid_token')
        .send({ ...validOrder, items: [] })
        .expect(400);

      expect(response.body.error).toBe('Order must have at least one item');
    });

    it('should reject invalid quantity', async () => {
      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer valid_token')
        .send({
          ...validOrder,
          items: [{ productId: 'prod_1', quantity: 0 }],
        })
        .expect(400);

      expect(response.body.error).toContain('quantity');
    });

    it('should reject non-existent product', async () => {
      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer valid_token')
        .send({
          ...validOrder,
          items: [{ productId: 'invalid_id', quantity: 1 }],
        })
        .expect(404);

      expect(response.body.error).toBe('Product not found');
    });
  });

  describe('支付失败', () => {
    it('should handle payment authorization failure', async () => {
      (paymentService.authorize as jest.Mock).mockResolvedValue({
        success: false,
        error: 'Insufficient funds',
      });

      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer valid_token')
        .send(validOrder)
        .expect(402);

      expect(response.body.error).toBe('Payment failed: Insufficient funds');

      // 验证订单状态为 pending
      const orders = await db.orders.findByUser('test_user');
      const pendingOrder = orders.find(o => o.status === 'payment_failed');
      expect(pendingOrder).toBeDefined();
    });
  });

  describe('认证', () => {
    it('should reject unauthenticated requests', async () => {
      await request(app)
        .post('/api/orders')
        .send(validOrder)
        .expect(401);
    });

    it('should reject invalid token', async () => {
      await request(app)
        .post('/api/orders')
        .set('Authorization', 'Bearer invalid_token')
        .send(validOrder)
        .expect(401);
    });
  });
});
```

## 测试覆盖率分析

```markdown
请分析以下覆盖率报告，指出需要补充测试的地方：

\`\`\`
File           | % Stmts | % Branch | % Funcs | Uncovered Lines
---------------|---------|----------|---------|----------------
orderService   |   78.5  |    62.5  |    90   | 45-52, 78-85
paymentService |   85.2  |    70.0  |   100   | 23-28
userService    |   92.0  |    88.5  |   100   | 156
\`\`\`
```

## 总结

AI 辅助调试与测试的关键：

1. **提供完整上下文** - 错误信息、代码、环境
2. **明确期望行为** - 描述正确的输出
3. **迭代优化** - 基于 AI 建议逐步改进
4. **人工验证** - AI 生成的测试需要审查

下一篇，我们将介绍 AI 编程最佳实践与工作流。
