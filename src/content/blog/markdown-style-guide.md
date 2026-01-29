---
title: 'Markdown 语法指南'
description: '一份完整的 Markdown 语法指南，帮助你更好地撰写博客文章。'
pubDate: 2024-01-10
category: '教程'
tags: ['Markdown', '写作', '指南']
heroImage: 'https://images.unsplash.com/photo-1542831371-29b0f74f9713?w=800&h=400&fit=crop'
---

这篇文章展示了一些基本的 Markdown 语法，可以在 Astro 的 Markdown 文件中使用。


## 标题

以下 HTML `<h2>` 到 `<h6>` 元素代表五个级别的节标题。

### H3 标题
#### H4 标题
##### H5 标题
###### H6 标题

## 段落

这是一个普通的段落。Markdown 使写作变得简单而高效。

这是另一个段落，段落之间用空行分隔。

## 列表

### 无序列表

- 项目一
- 项目二
- 项目三
  - 子项目 A
  - 子项目 B

### 有序列表

1. 第一步
2. 第二步
3. 第三步

## 引用

> 这是一段引用文本。引用可以用来强调重要的内容或者引用他人的话。

## 代码

### 行内代码

使用 `const greeting = "Hello"` 定义一个常量。

### 代码块

```javascript
function greet(name) {
  return `Hello, ${name}!`;
}

console.log(greet('World'));
```

## 链接和图片

这是一个 [链接示例](https://astro.build)。

## 表格

| 名称 | 描述 |
| --- | --- |
| Astro | 静态网站生成器 |
| Tailwind | CSS 框架 |
| Markdown | 标记语言 |

## 强调

**粗体文本** 和 *斜体文本* 可以用来强调内容。

---

以上就是常用的 Markdown 语法，希望对你有帮助！
