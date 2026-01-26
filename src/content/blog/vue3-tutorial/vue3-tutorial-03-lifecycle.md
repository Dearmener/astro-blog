---
title: 'Vue3 入门教程（三）：生命周期与组件通信'
description: '学习 Vue3 组件的生命周期钩子以及父子组件之间的通信方式'
pubDate: '2024-01-17'
heroImage: ''
category: '教程'
tags: ['Vue3', '前端', '教程']
series: 'Vue3 入门教程'
seriesOrder: 3
---

## 生命周期钩子

Vue3 组件从创建到销毁会经历一系列生命周期阶段：

```ts
import { 
  onBeforeMount,
  onMounted,
  onBeforeUpdate,
  onUpdated,
  onBeforeUnmount,
  onUnmounted
} from 'vue'

// 组件挂载前
onBeforeMount(() => {
  console.log('DOM 还未创建')
})

// 组件挂载后（最常用）
onMounted(() => {
  console.log('DOM 已创建，可以操作 DOM 或发起请求')
})

// 组件更新前
onBeforeUpdate(() => {
  console.log('数据已变化，DOM 即将更新')
})

// 组件更新后
onUpdated(() => {
  console.log('DOM 已更新')
})

// 组件卸载前
onBeforeUnmount(() => {
  console.log('清理定时器、取消订阅等')
})

// 组件卸载后
onUnmounted(() => {
  console.log('组件已销毁')
})
```

## 组件通信

### Props - 父传子

父组件通过 props 向子组件传递数据：

```vue
<!-- Parent.vue -->
<script setup lang="ts">
import Child from './Child.vue'
import { ref } from 'vue'

const message = ref('Hello from parent')
</script>

<template>
  <Child :msg="message" :count="42" />
</template>
```

```vue
<!-- Child.vue -->
<script setup lang="ts">
// 使用 defineProps 声明 props
const props = defineProps<{
  msg: string
  count: number
}>()

console.log(props.msg) // Hello from parent
</script>

<template>
  <p>{{ msg }} - {{ count }}</p>
</template>
```

### Emits - 子传父

子组件通过 emit 向父组件发送事件：

```vue
<!-- Child.vue -->
<script setup lang="ts">
const emit = defineEmits<{
  (e: 'update', value: string): void
  (e: 'delete', id: number): void
}>()

const handleClick = () => {
  emit('update', 'new value')
}
</script>

<template>
  <button @click="handleClick">更新</button>
</template>
```

```vue
<!-- Parent.vue -->
<script setup lang="ts">
const handleUpdate = (value: string) => {
  console.log('收到子组件更新:', value)
}
</script>

<template>
  <Child @update="handleUpdate" />
</template>
```

### v-model - 双向绑定

`v-model` 是 props + emit 的语法糖：

```vue
<!-- Parent.vue -->
<template>
  <CustomInput v-model="searchText" />
  <!-- 等价于 -->
  <CustomInput 
    :modelValue="searchText" 
    @update:modelValue="searchText = $event" 
  />
</template>
```

```vue
<!-- CustomInput.vue -->
<script setup lang="ts">
const props = defineProps<{ modelValue: string }>()
const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()
</script>

<template>
  <input 
    :value="modelValue" 
    @input="emit('update:modelValue', ($event.target as HTMLInputElement).value)"
  />
</template>
```

## 总结

通过本系列教程，你已经学习了：

1. **Vue3 基础** - 环境搭建和第一个组件
2. **响应式系统** - ref、reactive、computed、watch
3. **生命周期与通信** - 钩子函数、props、emit、v-model

接下来可以继续深入学习：
- Vue Router 路由管理
- Pinia 状态管理
- 组合式函数（Composables）

祝你学习愉快！🎉
