---
title: '模型评估与调优'
description: '全面掌握机器学习模型评估指标、交叉验证方法、超参数调优技术，避免过拟合与欠拟合'
pubDate: 2025-09-05
category: '技术'
tags: ['机器学习', '模型评估', '交叉验证', '超参数调优', '过拟合']
series: '机器学习完全指南'
seriesOrder: 8
heroImage: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=400&fit=crop'
---

## 模型评估的重要性

模型评估是机器学习工作流中最关键的环节之一。一个好的评估策略能帮助我们选择最佳模型，避免在测试集或真实场景中表现不佳。

### 训练集、验证集与测试集

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 创建数据集
X, y = make_classification(n_samples=10000, n_features=20, 
                           n_informative=15, random_state=42)

# 三分法：训练集 60%，验证集 20%，测试集 20%
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
```

## 分类评估指标

### 混淆矩阵与基础指标

```python
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵')
plt.show()

# 各项指标
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred):.4f}")
print(f"召回率 (Recall): {recall_score(y_test, y_pred):.4f}")
print(f"F1 分数: {f1_score(y_test, y_pred):.4f}")

# 完整分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))
```

### ROC 曲线与 AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# 获取预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('ROC 曲线')
plt.legend(loc="lower right")
plt.show()

print(f"ROC-AUC 分数: {roc_auc_score(y_test, y_prob):.4f}")
```

### PR 曲线（适用于不平衡数据）

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'PR 曲线 (AP = {ap:.4f})')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('Precision-Recall 曲线')
plt.legend()
plt.show()
```

## 回归评估指标

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)

# 创建回归数据
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                               noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 训练回归模型
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

# 评估指标
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.4f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test_reg, y_pred_reg):.4f}")
```

## 交叉验证

交叉验证是评估模型泛化能力的金标准方法。

### K-Fold 交叉验证

```python
from sklearn.model_selection import cross_val_score, KFold

# 基础交叉验证
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, scoring='accuracy'
)
print(f"5-Fold CV 准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 自定义 K-Fold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_10 = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=kfold, scoring='accuracy'
)
print(f"10-Fold CV 准确率: {cv_scores_10.mean():.4f} (+/- {cv_scores_10.std()*2:.4f})")
```

### 分层 K-Fold（适用于不平衡数据）

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_strat = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=skfold, scoring='f1'
)
print(f"Stratified 5-Fold F1: {cv_scores_strat.mean():.4f}")
```

### 多指标交叉验证

```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, scoring=scoring, return_train_score=True
)

for metric in scoring:
    train_score = cv_results[f'train_{metric}'].mean()
    test_score = cv_results[f'test_{metric}'].mean()
    print(f"{metric}: 训练 {train_score:.4f}, 测试 {test_score:.4f}")
```

## 超参数调优

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
print(f"测试集分数: {grid_search.score(X_test, y_test):.4f}")
```

### Random Search（大搜索空间推荐）

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 30),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,  # 随机尝试 50 种组合
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)
print(f"最佳参数: {random_search.best_params_}")
print(f"最佳分数: {random_search.best_score_:.4f}")
```

### Optuna 贝叶斯优化

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0)
    }
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    return score

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"最佳参数: {study.best_params}")
print(f"最佳分数: {study.best_value:.4f}")
```

## 过拟合与欠拟合

### 学习曲线诊断

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='训练分数', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                 alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, label='验证分数', color='orange')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                 alpha=0.1, color='orange')
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.title('学习曲线')
plt.legend()
plt.grid(True)
plt.show()
```

### 验证曲线

```python
from sklearn.model_selection import validation_curve

param_range = [1, 2, 5, 10, 20, 50, 100]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, param_name='max_depth',
    param_range=param_range, cv=5, scoring='accuracy', n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_scores.mean(axis=1), label='训练分数', color='blue')
plt.semilogx(param_range, val_scores.mean(axis=1), label='验证分数', color='orange')
plt.xlabel('max_depth')
plt.ylabel('准确率')
plt.title('验证曲线')
plt.legend()
plt.grid(True)
plt.show()
```

## 防止过拟合的策略

```python
# 1. 正则化
from sklearn.linear_model import LogisticRegression

# L1 正则化（稀疏）
l1_model = LogisticRegression(penalty='l1', solver='saga', C=0.1)

# L2 正则化（平滑）
l2_model = LogisticRegression(penalty='l2', C=0.1)

# 2. Early Stopping（以 XGBoost 为例）
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=1000, random_state=42)
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=20,
              verbose=False)

# 3. Dropout（深度学习中常用）
# 4. 数据增强
# 5. 减少模型复杂度
```

## 评估最佳实践

| 场景 | 推荐方法 |
|------|----------|
| 数据量充足 | 训练/验证/测试三分法 + K-Fold |
| 数据量有限 | K-Fold 交叉验证 (K=5 或 10) |
| 不平衡数据 | Stratified K-Fold + F1/AUC |
| 时间序列 | 时序交叉验证 |
| 超参数调优 | RandomSearch + Optuna |

## 总结

模型评估与调优的核心要点：

- 选择合适的评估指标（准确率不是万能的）
- 使用交叉验证获得可靠的性能估计
- 结合 Grid/Random/Bayesian Search 进行超参数调优
- 通过学习曲线诊断过拟合/欠拟合
- 保留独立测试集作为最终评估

下一篇我们将进行完整的 Scikit-learn 实战项目。
