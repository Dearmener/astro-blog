---
title: 'Python 数据科学环境搭建'
description: '从零搭建 Python 机器学习开发环境，包括 Anaconda、Jupyter、常用库配置。'
pubDate: 2025-08-05
category: '技术'
tags: ['Python', 'Anaconda', 'Jupyter', '环境配置']
series: '机器学习完全指南'
seriesOrder: 2
heroImage: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=800&h=400&fit=crop'
---

良好的开发环境是机器学习项目成功的基础。本文将详细介绍如何搭建专业的 Python 数据科学环境。

## 环境选择

### 方案对比

| 方案 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| Anaconda | 开箱即用、包管理强 | 体积大 | 初学者 |
| Miniconda | 轻量灵活 | 需手动安装包 | 有经验者 |
| venv + pip | 原生支持 | 科学计算库安装麻烦 | 简单项目 |
| Docker | 环境隔离完美 | 学习成本 | 生产部署 |

## Anaconda 安装

### 下载安装

```bash
# macOS/Linux
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh

# 或使用 Miniconda (更轻量)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 配置镜像源

```bash
# 配置清华镜像 (国内用户)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

### 创建虚拟环境

```bash
# 创建环境
conda create -n ml python=3.11

# 激活环境
conda activate ml

# 查看环境列表
conda env list

# 删除环境
conda env remove -n ml
```

## 核心库安装

### 数据处理

```bash
# NumPy - 数值计算
conda install numpy

# Pandas - 数据分析
conda install pandas

# 验证安装
python -c "import numpy; print(numpy.__version__)"
python -c "import pandas; print(pandas.__version__)"
```

### 数据可视化

```bash
# Matplotlib - 基础绑图
conda install matplotlib

# Seaborn - 统计可视化
conda install seaborn

# Plotly - 交互式可视化
pip install plotly
```

### 机器学习

```bash
# Scikit-learn - 经典机器学习
conda install scikit-learn

# XGBoost - 梯度提升
conda install xgboost

# LightGBM - 轻量级梯度提升
conda install lightgbm
```

### 深度学习

```bash
# PyTorch (GPU 版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 或 CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Jupyter 环境

### 安装配置

```bash
# 安装 JupyterLab
conda install jupyterlab

# 安装扩展
pip install jupyterlab-git
pip install jupyterlab-lsp

# 启动
jupyter lab
```

### Jupyter 快捷键

| 快捷键 | 功能 |
|--------|------|
| Shift + Enter | 运行并跳转下一格 |
| Ctrl + Enter | 运行当前格 |
| A | 上方插入单元格 |
| B | 下方插入单元格 |
| DD | 删除单元格 |
| M | 转为 Markdown |
| Y | 转为代码 |

### 常用魔法命令

```python
# 计时
%time result = slow_function()
%timeit fast_function()

# 内存分析
%load_ext memory_profiler
%memit large_data_operation()

# 显示图表
%matplotlib inline

# 自动重载模块
%load_ext autoreload
%autoreload 2

# 查看变量
%whos
```

## VS Code 配置

### 必装扩展

| 扩展 | 功能 |
|------|------|
| Python | Python 语言支持 |
| Jupyter | Notebook 支持 |
| Pylance | 智能提示 |
| Python Indent | 自动缩进 |
| autoDocstring | 自动文档 |

### settings.json 配置

```json
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/ml/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "jupyter.askForKernelRestart": false
}
```

## 项目结构

### 标准目录结构

```
ml_project/
├── data/
│   ├── raw/           # 原始数据
│   ├── processed/     # 处理后数据
│   └── external/      # 外部数据
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature.ipynb
│   └── 03_model.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py        # 数据处理
│   ├── features.py    # 特征工程
│   ├── model.py       # 模型训练
│   └── utils.py       # 工具函数
├── models/            # 保存的模型
├── reports/           # 报告和图表
├── tests/             # 测试代码
├── requirements.txt
├── environment.yml
└── README.md
```

### requirements.txt

```txt
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

### environment.yml

```yaml
name: ml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - pip
  - pip:
    - xgboost
    - lightgbm
```

## 验证环境

### 测试脚本

```python
# test_environment.py
import sys

def check_import(module_name):
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {module_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {module_name}: not installed")
        return False

print(f"Python: {sys.version}")
print("-" * 40)

modules = [
    'numpy', 'pandas', 'sklearn', 
    'matplotlib', 'seaborn', 
    'torch', 'xgboost', 'lightgbm'
]

results = [check_import(m) for m in modules]
print("-" * 40)
print(f"Passed: {sum(results)}/{len(results)}")
```

### 运行测试

```bash
python test_environment.py
```

## GPU 环境配置

### CUDA 安装

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 安装 CUDA Toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit

# 验证
nvcc --version
```

### PyTorch GPU 验证

```python
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # 简单测试
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU computation: OK")
```

## 常见问题

### 包冲突解决

```bash
# 查看冲突
conda list --revisions

# 回滚到之前版本
conda install --revision 2

# 或重建环境
conda env remove -n ml
conda create -n ml python=3.11
```

### pip vs conda

```bash
# 优先使用 conda
conda install package_name

# conda 没有再用 pip
pip install package_name

# 不要混用，可能导致冲突
```

## 总结

搭建数据科学环境的关键步骤：

1. **选择工具**: Anaconda/Miniconda
2. **创建虚拟环境**: 隔离项目依赖
3. **安装核心库**: NumPy, Pandas, Scikit-learn
4. **配置 Jupyter**: 交互式开发
5. **GPU 支持**: CUDA + PyTorch

下一篇将介绍数据预处理与特征工程。
