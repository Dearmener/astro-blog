---
title: '我做的 3 个 AI 辅助开发开源项目'
description: '介绍 yt-dlp TUI、AI Chat TOC Sidebar 和 File Manager 三个开源项目，以及我如何用 LLM 和 Vibe Coding 把想法快速落地。'
pubDate: 2026-04-25
category: '技术'
tags:
  - AI 编程
  - 开源项目
  - Vibe Coding
  - Agent
  - RAG
heroImage: 'https://images.unsplash.com/photo-1515879218367-8466d910aaa4?w=800&h=400&fit=crop'
---

过去一段时间，我一边做研究和实习，一边持续把一些高频需求做成真正可用的工具。它们不只是“练手项目”，而是围绕我自己的实际使用场景展开：下载和整理内容、浏览长对话、提升文件管理效率。

这些项目有一个共同点：我都大量使用了 LLM 辅助开发。对我来说，AI 不只是代码补全工具，更像一个可以协助做需求拆解、方案讨论、异常定位、模块重构和交互迭代的工程伙伴。下面就集中介绍一下这 3 个项目。

## 为什么会做这几个项目

我做项目时通常有两个判断标准：

1. 这个问题是不是我自己真的高频会遇到。
2. 这个问题能不能通过 AI 辅助开发，被更快地做成一个完整产品。

这三个项目分别对应了三个不同方向：

- `yt-dlp TUI`：解决多平台视频下载和终端交互体验问题。
- `AI Chat TOC Sidebar`：解决长篇 AI 对话难以浏览和跳转的问题。
- `File Manager`：解决文件整理规则复杂、重复操作多的问题。

从结果看，它们也分别锻炼了我在命令行工具、浏览器扩展和多入口桌面效率工具上的设计与工程实现能力。

## 1. yt-dlp TUI：多平台视频下载工具

GitHub: [https://github.com/Dearmener/yt_download](https://github.com/Dearmener/yt_download)

`yt-dlp TUI` 是一个基于 `LLM + Vibe Coding` 独立完成的多平台视频下载工具，支持 YouTube、Bilibili、Twitter 等 10+ 平台。相比直接调用下载脚本，我更希望把它做成一个真正“能长期使用”的工具，所以重点放在了终端交互体验、下载流程拆分和并发处理能力上。

这个项目里，我主要做了几件事：

- 支持多平台链接解析和播放列表处理。
- 增加并行元数据获取，减少等待时间。
- 设计多线程下载流程，提升批量任务效率。
- 将复杂逻辑拆成可维护模块，避免脚本越写越乱。

这个项目给我最大的收获，不是“把下载功能做出来”，而是如何在 AI 辅助开发的环境下，把一个原本容易写成一次性脚本的东西，做成结构相对清晰、可持续迭代的工具。

## 2. AI Chat TOC Sidebar：给长对话加目录

GitHub: [https://github.com/Dearmener/ai-toc-extension](https://github.com/Dearmener/ai-toc-extension)

随着我越来越多地使用 ChatGPT、Claude、Gemini 这类产品，一个非常明显的问题就是：对话一长，信息就开始失控。虽然内容本身有价值，但定位、回看、跳转都很低效。

于是我做了 `AI Chat TOC Sidebar`。这是一个浏览器扩展，面向 AI 对话场景自动生成目录，帮助用户在长上下文中快速浏览内容结构，并进行跳转。

这个项目里，我重点关注的不是“能不能做出目录”，而是“目录是不是足够好用”。因此我围绕真实使用体验做了不少交互设计：

- 自动识别对话结构并生成侧边目录。
- 支持拖拽和缩放，减少对原页面布局的侵入。
- 支持快速跳转，提高长对话内容定位效率。
- 持续迭代目录生成和交互逻辑，适配不同 AI 产品界面。

这个项目很能体现我理解 AI 产品的一个方式：很多需求并不是“大而全”的平台，而是围绕一个明确痛点，做一个足够锋利的效率增强工具。

## 3. File Manager：支持多入口的智能文件整理工具

GitHub: [https://github.com/Dearmener/file_manager](https://github.com/Dearmener/file_manager)

文件整理是一个典型的“人人都烦，但很少有人认真做工具”的问题。很多时候并不是整理本身困难，而是规则各不相同，用户又希望既自动化，又可控。

所以我做了 `File Manager`。这是一个基于 Python 和 LLM 辅助开发完成的文件整理工具，支持 `TUI`、`CLI JSON` 接口和 `Raycast` 扩展，尽量覆盖不同使用入口下的整理需求。

核心能力包括：

- 支持前缀、后缀、扩展名和正则规则匹配。
- 支持 `Dry Run`，先预览结果再真正执行。
- 支持撤销，降低自动整理带来的风险。
- 支持监控自动整理，提高持续使用价值。

这个项目的重点不是技术炫技，而是把“规则引擎 + 多入口交互 + 可控自动化”组合起来，做成一个真正能服务日常工作流的工具。

## 我是怎么用 AI 来做这些项目的

很多人会把 AI 编程理解成“让模型帮忙写代码”，但我自己的实践更接近下面这套流程：

1. 用自然语言先拆需求，确定最小可用版本。
2. 让模型协助生成初版代码和模块边界。
3. 针对报错、异常路径和边界条件做多轮修正。
4. 在功能逐步稳定后，再让模型帮助做重构和抽象。
5. 最后由我自己统一做取舍、验收和工程落地。

在这个过程中，AI 最有价值的不是替我“完成项目”，而是显著降低了我在信息检索、样板代码、接口对接和局部重构上的时间消耗，让我能把更多精力放在真正重要的问题上：

- 用户到底要解决什么问题。
- 产品的核心交互是否合理。
- 模块边界是否清晰。
- 工程实现是否足够稳定。

这也是为什么我越来越倾向于把这种方式理解为 `AI-assisted building`，而不是简单的“让 AI 帮我写代码”。

## 这 3 个项目背后反映出的能力

如果从能力结构上总结，这 3 个项目分别对应了我比较关注的几个方向：

- AI 辅助开发能力：能借助 LLM 完成需求拆解、代码生成、异常处理和模块重构。
- Agent 式任务拆解能力：能把复杂需求分解成可执行的小模块，再逐步集成。
- 工具型产品意识：关注真实使用场景，而不是只做 demo。
- 工程落地能力：能把一个想法做成真正能运行、能迭代、能复用的产品。

这也是我后续继续做 AI 应用、Agent 系统和智能工具产品时最看重的一组能力组合。

## 项目链接汇总

- yt-dlp TUI: [https://github.com/Dearmener/yt_download](https://github.com/Dearmener/yt_download)
- AI Chat TOC Sidebar: [https://github.com/Dearmener/ai-toc-extension](https://github.com/Dearmener/ai-toc-extension)
- File Manager: [https://github.com/Dearmener/file_manager](https://github.com/Dearmener/file_manager)
- GitHub 主页: [https://github.com/Dearmener](https://github.com/Dearmener)

如果你也在做 AI 应用、Agent 工作流或者 AI 辅助开发相关项目，欢迎交流。
