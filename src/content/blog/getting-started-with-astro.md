---
title: '开始使用 Astro 构建你的博客'
description: '学习如何使用 Astro 框架快速搭建一个现代化的个人博客，包括内容管理、样式配置等基础知识。'
pubDate: 2024-01-15
category: '技术'
tags: ['Astro', 'Web开发', '教程']
heroImage: 'https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=800&h=400&fit=crop'
---

欢迎来到我的博客！这是使用 Astro 构建的第一篇文章。

## 为什么选择 Astro？

Astro 是一个现代化的静态网站生成器，它有以下优点：

- **零 JavaScript 默认**：默认情况下，Astro 不会发送任何 JavaScript 到客户端
- **组件岛屿**：只在需要的地方加载 JavaScript
- **多框架支持**：可以使用 React、Vue、Svelte 等任何框架
- **内容优先**：内置对 Markdown 和 MDX 的支持

## 代码示例

下面是一个简单的 Astro 组件示例：

```astro
---
const greeting = 'Hello, World!';
---

<h1>{greeting}</h1>

<style>
  h1 {
    color: purple;
  }
</style>
```

## 开始使用

1. 安装依赖：`npm install`
2. 启动开发服务器：`npm run dev`
3. 开始编写你的内容！

> Astro 让构建内容驱动的网站变得前所未有的简单。

希望这篇文章能帮助你开始使用 Astro 构建自己的博客！
