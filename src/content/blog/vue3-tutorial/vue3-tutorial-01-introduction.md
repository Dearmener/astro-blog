---
title: 'Vue3 入门教程（一）：初识 Vue3'
description: '从零开始学习 Vue3，本章介绍 Vue3 的核心特性和开发环境搭建'
pubDate: '2024-01-15'
heroImage: ''
category: '教程'
tags: ['Vue3', '前端', '教程']
series: 'Vue3 入门教程'
seriesOrder: 1
---

## 为什么选择 Vue3？

Vue3 是 Vue.js 的最新主要版本，带来了许多令人兴奋的新特性：

- **Composition API** - 更灵活的代码组织方式
- **更好的 TypeScript 支持** - 从底层重新设计的类型系统
- **性能提升** - 更快的渲染速度和更小的包体积
- **Teleport 组件** - 轻松处理模态框等场景

## 开发环境搭建

### 1. 安装 Node.js

首先确保你的电脑安装了 Node.js（推荐 v18+）：

```bash
node --version
npm --version
```

### 2. 创建 Vue3 项目

使用官方脚手架 `create-vue` 创建项目：

```bash
npm create vue@latest my-vue3-app
cd my-vue3-app
npm install
npm run dev
```

### 3. 项目结构

```
my-vue3-app/
├── src/
│   ├── components/    # 组件目录
│   ├── views/         # 页面目录
│   ├── App.vue        # 根组件
│   └── main.ts        # 入口文件
├── package.json
└── vite.config.ts
```

## 第一个 Vue3 组件

```vue
<script setup lang="ts">
import { ref } from 'vue'

const count = ref(0)
const increment = () => count.value++
</script>

<template>
  <button @click="increment">
    点击次数: {{ count }}
  </button>
</template>

<style scoped>
button {
  padding: 0.5rem 1rem;
  font-size: 1rem;
}
</style>
```

## 下一步

在下一章中，我们将深入学习 Vue3 的响应式系统和 `ref`、`reactive` 的使用方法。
