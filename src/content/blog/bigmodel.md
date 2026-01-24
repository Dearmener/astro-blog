---
title: "使用 Agent Skills 扩展 Claude 功能"
pubDate: 2026-01-15
description: "介绍如何通过 Agent Skills 扩展 Claude 智能体的功能，实现模块化能力。"
category: "技术"
tags: ["Claude", "Agent Skills", "AI 扩展"]
heroImage: "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=800&h=400&fit=crop"
---
## Skills 是什么？

Agent Skills 就是用来扩展智能体功能的模块化能力。通过一个文件夹 + 一个 **SKILL.md** 文件告诉 Claude：“在某些场景下，我希望你用这个能力做某件事”

Agent Skills 的关键是渐进式披露，分三层加载：

- **层级 1：技能发现** -- AI 先读取所有技能的元数据（name 和 description），判断任务是否相关，这些元数据始终在系统提示中。
- **层级 2：加载核心指令** -- 如果相关，AI 自动读取 SKILL.md 的正文内容，获取详细指导。
- **层级 3：加载资源文件** -- 只在需要时读取额外文件（如脚本、示例），或通过工具执行脚本。

## Skills 格式要求

Skills 存放在  **~/.claude/skills/** （个人全局）或项目目录下的  **.claude/skills/** （项目专用）。

```yaml
my-skill/
├── SKILL.md              ← 必须的
├── scripts/              ← 可选（脚本等）
├── references/           ← 可选（资源/规范文档）
└── assets/               ← 可选（模板/附加资源）
```

SKILL.md是 SKills 中唯一必须的

```md
---
name: your-skill-name
description: What it does and when Claude should use it
---

# Skill Title

## Instructions
Clear, concrete, actionable rules.

## Examples
- Example usage 1
- Example usage 2

## Guidelines
- Guideline 1
- Guideline 2
```

SKill.md的元数据字段要求:

| 字段           | 必需 | 作用                  |
| ---------------- | ------ | ----------------------- |
| <span data-type="text" style="background-color: var(--b3-card-error-background); color: var(--b3-card-error-color);">name</span>               | 是   | 唯一标识，小写+连字符 |
| <span data-type="text" style="background-color: var(--b3-card-error-background); color: var(--b3-card-error-color);">description</span>               | 是   | **触发条件（最重要）**                      |
| allowed-tools  | 否   | 限制可用工具          |
| model          | 否   | 指定模型              |
| context        | 否   | `fork` \= 独立上下文     |
| agent          | 否   | fork 时使用的子代理   |
| hooks          | 否   | Skill 生命周期钩子    |
| user-invocable | 否   | 是否显示在 `/` 菜单      |

## 推荐目录结构

为避免上下文膨胀：

- 核心规则 → `SKILL.md`
- 详细资料 → 单独文件
- 实用逻辑 → 脚本执行（不加载）

推荐结构:

```
my-skill/
├── SKILL.md
├── reference.md
├── examples.md    # 存放示例文件
└── scripts/
    └── helper.py
```

## 第一个 Skill

让我们暂时忘掉复杂的创建过程，先从 **使用一个现成的 Skill** 开始，感受它带来的便利。

### 创建 Skill 目录

Skills 存放在  **~/.claude/skills/** （个人全局）或项目目录下的  **.claude/skills/** （项目专用）。

本章节再项目目录下测试，先创建个目录 claude-test:

```
mkdir claude-test
```

进入该目录，创建 skills 的目录与文件：

```
mkdir -p .claude/skills/python-naming-standard
```

### 编写配置文件 SKILL.md

在目录下创建 SKILL.md，这是 Skill 的大脑 ，告诉 Claude 什么时候用它。

```
---
name: Python 内部命名规范技能
description: 当用户要求重构、审查或编写 Python 代码时，请参考此规范。
---

## 指令
1. 所有的内部辅助函数必须以 `_internal_` 前缀命名。
2. 如果发现不符合此规则的代码，请自动提出修改建议。
3. 在执行 `claude commit` 前，必须检查此规范。

## 参考示例
- 正确：`def _internal_calculate_risk():`
- 错误：`def _calculate_risk():`
```

字段要求：

- **name**：必须仅使用小写字母、数字和连字符（最多 64 个字符）
- **description**：Skill 的简要描述及其使用时机（最多 1024 个字符）

创建完后文件结构如下：

![](https://www.runoob.com/wp-content/uploads/2026/01/7d0592b4-61f8-4170-a639-6e83f6740cb6.png)

你的项目现在看起来应该是这样的：

```
my-project/
├─ src/
│  └─ test.py              # 项目源码
├─ .claude/
│  ├─ skills/
│  │  └─ hello-world/
│  │     ├─ skill.md       # Skill 定义（YAML + Instructions，机器可执行）
│  │     └─ README.md      # Skill 说明（人类阅读，可选）
│  └─ config.yml           # Claude 项目级配置（可选）
├─ .gitignore
└─ README.md               # 项目整体说明
```

接下来我们再终端执行以下命令启动 Claude Code：

```
claude
```

输入任务：

```
帮我写一个计算用户折扣的函数
```

Claude 就会会扫描已安装的 Skills，发现你的请求涉及 "Python 代码编写"，匹配了 python-naming-standard。

![](https://www.runoob.com/wp-content/uploads/2026/01/1527ed0b-d1a9-420a-8f25-c71231e23c05.png)

它会根据 SKILL.md 中的要求，生成如下代码：

```
def _internal_get_discount(user_score):
    # 计算逻辑...
    return discount
```

### 添加资源文件（可选）

另外我们可以在  **.claude/skills/**  下添加以下目录：

在同一文件夹添加：

- `examples/`：存放示例文件。
- `references/`：存放参考文档。
- `scripts/`：存放可执行脚本（例如 Python 处理 PDF）。

然后在 SKILL.md 中引用：

```
查看示例 commit：./examples/good-commit.txt
运行脚本：使用工具执行 ./scripts/process.py
```
