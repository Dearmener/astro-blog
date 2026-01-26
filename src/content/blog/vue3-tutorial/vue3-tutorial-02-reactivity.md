---
title: 'Vue3 入门教程（二）：响应式系统'
description: '深入理解 Vue3 的响应式原理，掌握 ref 和 reactive 的正确使用方法'
pubDate: '2024-01-16'
heroImage: ''
category: '教程'
tags: ['Vue3', '前端', '教程']
series: 'Vue3 入门教程'
seriesOrder: 2
---

## 响应式基础

Vue3 的响应式系统是其核心特性之一。当响应式数据发生变化时，视图会自动更新。

### ref - 基本类型的响应式

`ref` 适用于基本类型（string、number、boolean）：

```ts
import { ref } from 'vue'

const count = ref(0)
const message = ref('Hello Vue3')

// 在 JS 中需要 .value
console.log(count.value) // 0
count.value++

// 在 template 中自动解包
// <span>{{ count }}</span>
```

### reactive - 对象的响应式

`reactive` 适用于对象和数组：

```ts
import { reactive } from 'vue'

const user = reactive({
  name: '张三',
  age: 25,
  hobbies: ['coding', 'reading']
})

// 直接访问，无需 .value
console.log(user.name) // 张三
user.age = 26
user.hobbies.push('gaming')
```

## ref vs reactive 如何选择？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 基本类型 | `ref` | 只能用 ref |
| 对象/数组 | `ref` 或 `reactive` | ref 更统一 |
| 需要替换整个对象 | `ref` | reactive 不能替换 |
| 解构使用 | `ref` | reactive 解构会丢失响应性 |

**最佳实践**：统一使用 `ref`，保持代码一致性。

## computed - 计算属性

计算属性会自动追踪依赖，并在依赖变化时重新计算：

```ts
import { ref, computed } from 'vue'

const firstName = ref('张')
const lastName = ref('三')

// 只读计算属性
const fullName = computed(() => {
  return firstName.value + lastName.value
})

// 可写计算属性
const fullNameWritable = computed({
  get: () => firstName.value + lastName.value,
  set: (val) => {
    firstName.value = val[0]
    lastName.value = val.slice(1)
  }
})
```

## watch - 侦听器

当需要在数据变化时执行副作用（如 API 请求），使用 `watch`：

```ts
import { ref, watch } from 'vue'

const searchQuery = ref('')

// 侦听单个 ref
watch(searchQuery, async (newVal, oldVal) => {
  console.log(`搜索词从 ${oldVal} 变为 ${newVal}`)
  // 调用搜索 API
  const results = await searchAPI(newVal)
})

// 立即执行 + 深度侦听
watch(
  searchQuery,
  (val) => { /* ... */ },
  { immediate: true, deep: true }
)
```

## 下一步

在下一章中，我们将学习 Vue3 的生命周期钩子和组件通信方式。
