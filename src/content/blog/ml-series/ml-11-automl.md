---
title: 'AutoML 与自动化机器学习'
description: '探索自动化机器学习工具，包括 Auto-sklearn、TPOT、H2O AutoML，实现高效的模型自动选择与调优'
pubDate: 2025-09-20
category: '技术'
tags: ['机器学习', 'AutoML', 'Auto-sklearn', 'TPOT', 'H2O']
series: '机器学习完全指南'
seriesOrder: 11
heroImage: 'https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&h=400&fit=crop'
---

## AutoML 简介

AutoML（自动化机器学习）旨在自动化机器学习流程中的关键步骤，包括特征工程、模型选择和超参数调优，让非专家也能构建高质量的机器学习模型。

### 为什么需要 AutoML

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 传统 ML 流程 vs AutoML
"""
传统机器学习:
1. 数据预处理（手动）
2. 特征工程（手动）
3. 模型选择（试错）
4. 超参数调优（Grid/Random Search）
5. 模型评估
6. 迭代优化

AutoML:
1. 提供数据
2. 设置目标和约束
3. 自动完成上述所有步骤
4. 获得最优模型
"""

# 创建示例数据集
X, y = make_classification(
    n_samples=2000, n_features=20,
    n_informative=15, n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
```

## Auto-sklearn

Auto-sklearn 是基于 Scikit-learn 的自动化机器学习库，使用贝叶斯优化进行模型选择和超参数调优。

### 安装与基础使用

```python
# pip install auto-sklearn

import autosklearn.classification
from sklearn.metrics import accuracy_score, classification_report

# 创建 AutoML 分类器
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=300,  # 总时间限制（秒）
    per_run_time_limit=30,        # 单次运行时间限制
    n_jobs=-1,
    memory_limit=4096,            # 内存限制 MB
    seed=42
)

# 训练
automl.fit(X_train, y_train)

# 查看结果
print("=== Auto-sklearn 排行榜 ===")
print(automl.leaderboard())

# 预测与评估
y_pred = automl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

### 查看最佳模型

```python
# 查看模型配置
print("最佳模型配置:")
print(automl.show_models())

# 获取统计信息
print("\n运行统计:")
print(automl.sprint_statistics())
```

### Auto-sklearn 回归任务

```python
import autosklearn.regression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# 创建回归数据
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 回归任务
automl_reg = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=300,
    per_run_time_limit=30,
    n_jobs=-1,
    seed=42
)

automl_reg.fit(X_train_reg, y_train_reg)

# 评估
y_pred_reg = automl_reg.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")
```

## TPOT

TPOT（Tree-based Pipeline Optimization Tool）使用遗传算法自动构建和优化机器学习管道。

### TPOT 基础使用

```python
# pip install tpot

from tpot import TPOTClassifier, TPOTRegressor

# 创建 TPOT 分类器
tpot = TPOTClassifier(
    generations=5,           # 遗传算法代数
    population_size=50,      # 种群大小
    cv=5,                    # 交叉验证折数
    random_state=42,
    verbosity=2,
    n_jobs=-1,
    max_time_mins=5          # 最大运行时间（分钟）
)

# 训练
tpot.fit(X_train, y_train)

# 评估
tpot_score = tpot.score(X_test, y_test)
print(f"TPOT 测试集准确率: {tpot_score:.4f}")

# 导出最佳管道代码
tpot.export('tpot_best_pipeline.py')
print("最佳管道已导出到 tpot_best_pipeline.py")
```

### 自定义搜索空间

```python
# 自定义配置
tpot_config = {
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgboost.XGBClassifier': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'sklearn.preprocessing.StandardScaler': {},
    'sklearn.preprocessing.MinMaxScaler': {}
}

tpot_custom = TPOTClassifier(
    generations=5,
    population_size=30,
    config_dict=tpot_config,
    cv=5,
    random_state=42,
    verbosity=2
)

tpot_custom.fit(X_train, y_train)
```

### TPOT 导出代码示例

```python
# tpot_best_pipeline.py 生成的代码示例
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 最佳管道
exported_pipeline = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42
    )
)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
"""
```

## H2O AutoML

H2O AutoML 是企业级自动化机器学习平台，支持大规模数据处理。

### H2O 环境设置

```python
# pip install h2o

import h2o
from h2o.automl import H2OAutoML

# 初始化 H2O 集群
h2o.init(max_mem_size="4G")

# 转换数据为 H2O Frame
df_train = pd.DataFrame(X_train)
df_train['target'] = y_train
df_test = pd.DataFrame(X_test)
df_test['target'] = y_test

train_h2o = h2o.H2OFrame(df_train)
test_h2o = h2o.H2OFrame(df_test)

# 设置特征和目标
features = [str(i) for i in range(20)]
target = 'target'

# 转换目标为因子（分类任务）
train_h2o[target] = train_h2o[target].asfactor()
test_h2o[target] = test_h2o[target].asfactor()
```

### H2O AutoML 训练

```python
# 创建 AutoML 实例
aml = H2OAutoML(
    max_models=20,           # 最多训练 20 个模型
    max_runtime_secs=300,    # 最大运行时间
    seed=42,
    balance_classes=True,    # 处理类别不平衡
    sort_metric='AUC'        # 排序指标
)

# 训练
aml.train(x=features, y=target, training_frame=train_h2o)

# 查看排行榜
lb = aml.leaderboard
print("=== H2O AutoML 排行榜 ===")
print(lb.head(10))

# 最佳模型
best_model = aml.leader
print(f"\n最佳模型: {best_model.model_id}")
```

### H2O 模型评估与解释

```python
# 测试集评估
performance = best_model.model_performance(test_h2o)
print(f"测试集 AUC: {performance.auc():.4f}")

# 特征重要性
varimp = best_model.varimp(use_pandas=True)
print("\n特征重要性 Top 10:")
print(varimp.head(10))

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(varimp['variable'][:10][::-1], varimp['relative_importance'][:10][::-1])
plt.xlabel('相对重要性')
plt.title('H2O AutoML 特征重要性')
plt.tight_layout()
plt.show()

# 保存模型
h2o.save_model(best_model, path='./h2o_model', force=True)

# 关闭集群
h2o.cluster().shutdown()
```

## PyCaret - 低代码 AutoML

PyCaret 提供了最简洁的 AutoML 接口，适合快速原型开发。

### PyCaret 快速入门

```python
# pip install pycaret

from pycaret.classification import *

# 准备数据
df = pd.DataFrame(X)
df['target'] = y

# 设置实验
clf_setup = setup(
    data=df,
    target='target',
    train_size=0.8,
    session_id=42,
    normalize=True,
    fold=5,
    verbose=False
)

# 比较所有模型
best_models = compare_models(n_select=5)

# 查看最佳模型
print("=== PyCaret 最佳模型 ===")
print(best_models)
```

### PyCaret 完整工作流

```python
# 调优最佳模型
tuned_model = tune_model(best_models[0])

# 集成模型
bagged = ensemble_model(tuned_model, method='Bagging')
boosted = ensemble_model(tuned_model, method='Boosting')

# 混合模型
blended = blend_models(top3=best_models[:3])

# 堆叠模型
stacked = stack_models(estimator_list=best_models[:3])

# 评估模型
evaluate_model(tuned_model)

# 预测
predictions = predict_model(tuned_model)
print(predictions.head())

# 最终化模型
final_model = finalize_model(tuned_model)

# 保存模型
save_model(final_model, 'pycaret_model')
```

## AutoML 工具对比

| 工具 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| Auto-sklearn | 成熟稳定，集成 sklearn | 速度较慢 | 中小规模数据 |
| TPOT | 可导出代码，可定制 | 耗时较长 | 需要可解释管道 |
| H2O AutoML | 企业级，支持大数据 | 需要 Java 环境 | 大规模生产环境 |
| PyCaret | 代码简洁，功能全面 | 灵活性较低 | 快速原型开发 |

## 最佳实践

```python
# AutoML 使用建议

# 1. 合理设置时间限制
"""
- 小数据集（<10k）: 5-15 分钟
- 中等数据集（10k-100k）: 15-60 分钟
- 大数据集（>100k）: 1-4 小时
"""

# 2. 数据质量仍然重要
"""
AutoML 不能自动修复:
- 数据泄露
- 标签错误
- 严重的类别不平衡
"""

# 3. 保留测试集
"""
永远不要让 AutoML 看到测试集
"""

# 4. 模型解释
import shap

# 使用 SHAP 解释 AutoML 模型
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 5. 生产部署考虑
"""
- 模型大小
- 推理延迟
- 依赖管理
"""
```

## 总结

AutoML 的核心价值：

- **降低门槛**：让非专家也能构建高质量模型
- **提高效率**：自动化繁琐的调参过程
- **模型发现**：可能发现人类忽视的模型组合
- **基准建立**：快速建立性能基准

**但 AutoML 不能替代**：
- 领域知识
- 数据质量控制
- 特征工程洞察
- 模型可解释性需求

下一篇我们将学习机器学习模型部署与 MLOps。
