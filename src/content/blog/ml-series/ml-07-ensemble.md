---
title: '集成学习：Random Forest 与 XGBoost'
description: '深入理解集成学习方法，掌握 Bagging、Boosting 原理，实战 Random Forest、XGBoost 和 LightGBM 算法'
pubDate: 2025-08-30
category: '技术'
tags: ['机器学习', '集成学习', 'Random Forest', 'XGBoost', 'LightGBM']
series: '机器学习完全指南'
seriesOrder: 7
heroImage: 'https://images.unsplash.com/photo-1509228627152-72ae9ae6848d?w=800&h=400&fit=crop'
---

## 集成学习概述

集成学习（Ensemble Learning）是机器学习中的重要范式，通过组合多个基学习器来提升模型性能。其核心思想是"三个臭皮匠，顶个诸葛亮"。

### 为什么集成学习有效

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 创建分类数据集
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=15, n_redundant=5,
                           random_state=42)

# 单棵决策树
single_tree = DecisionTreeClassifier(random_state=42)
single_scores = cross_val_score(single_tree, X, y, cv=5)
print(f"单棵决策树准确率: {single_scores.mean():.4f} (+/- {single_scores.std():.4f})")

# 随机森林（100棵树）
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=5)
print(f"随机森林准确率: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")
```

## Bagging：Bootstrap Aggregating

Bagging 通过对训练数据进行有放回采样，训练多个基学习器，然后通过投票或平均进行预测。

### Bagging 原理实现

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# 手动实现 Bagging 概念
class SimpleBagging:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap 采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # 训练基学习器
            model = DecisionTreeClassifier()
            model.fit(X_boot, y_boot)
            self.models.append(model)
        return self
    
    def predict(self, X):
        # 投票
        predictions = np.array([m.predict(X) for m in self.models])
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), 
            axis=0, arr=predictions
        )

# 使用 sklearn 的 BaggingClassifier
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,  # 每次采样 80% 的数据
    max_features=0.8,  # 每次采样 80% 的特征
    bootstrap=True,
    random_state=42
)
bagging.fit(X, y)
print(f"Bagging 准确率: {cross_val_score(bagging, X, y, cv=5).mean():.4f}")
```

## Random Forest 详解

随机森林是 Bagging 的扩展，不仅对样本进行采样，还对特征进行随机选择。

### Random Forest 核心参数

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 创建随机森林分类器
rf = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_depth=None,        # 树的最大深度
    min_samples_split=2,   # 内部节点再划分所需最小样本数
    min_samples_leaf=1,    # 叶子节点最少样本数
    max_features='sqrt',   # 每棵树使用的特征数
    bootstrap=True,        # 是否使用 bootstrap 采样
    oob_score=True,        # 是否使用袋外数据评估
    n_jobs=-1,             # 并行数
    random_state=42
)

rf.fit(X, y)
print(f"OOB Score: {rf.oob_score_:.4f}")

# 特征重要性
feature_importance = rf.feature_importances_
print("Top 5 重要特征:", np.argsort(feature_importance)[-5:][::-1])
```

### Random Forest 超参数调优

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.4f}")
```

## Boosting：提升方法

Boosting 是一种串行集成方法，每个新模型都专注于修正前一个模型的错误。

### AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 弱分类器
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME',
    random_state=42
)

ada.fit(X, y)
ada_scores = cross_val_score(ada, X, y, cv=5)
print(f"AdaBoost 准确率: {ada_scores.mean():.4f}")
```

## XGBoost 实战

XGBoost（eXtreme Gradient Boosting）是目前最流行的梯度提升库，在 Kaggle 竞赛中广泛使用。

### XGBoost 基础使用

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建 DMatrix（XGBoost 优化的数据结构）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 参数设置
params = {
    'max_depth': 6,
    'eta': 0.3,              # 学习率
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',   # 使用直方图加速
    'device': 'cpu'
}

# 训练模型
evallist = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100,
                  evals=evallist, early_stopping_rounds=10,
                  verbose_eval=20)

# 预测
y_pred = model.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)
```

### XGBoost Scikit-learn API

```python
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,          # 样本采样比例
    colsample_bytree=0.8,   # 特征采样比例
    reg_alpha=0,            # L1 正则化
    reg_lambda=1,           # L2 正则化
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

xgb_clf.fit(X_train, y_train, 
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False)

print(f"XGBoost 准确率: {xgb_clf.score(X_test, y_test):.4f}")

# 特征重要性可视化
xgb.plot_importance(xgb_clf, max_num_features=10)
```

## LightGBM 高效训练

LightGBM 是微软开发的梯度提升框架，以其高效性著称，特别适合大规模数据。

### LightGBM 基础使用

```python
import lightgbm as lgb

# 创建数据集
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 参数设置
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# 训练
gbm = lgb.train(params, lgb_train, num_boost_round=200,
                valid_sets=[lgb_eval],
                callbacks=[lgb.early_stopping(stopping_rounds=20)])

# 预测
y_pred = gbm.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
```

### LightGBM vs XGBoost 对比

```python
import time

# XGBoost 训练时间
start = time.time()
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - start

# LightGBM 训练时间
start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start

print(f"XGBoost 训练时间: {xgb_time:.3f}s")
print(f"LightGBM 训练时间: {lgb_time:.3f}s")
print(f"LightGBM 速度提升: {xgb_time/lgb_time:.2f}x")
```

## 实战：信用卡欺诈检测

```python
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# 创建不平衡数据集模拟欺诈检测
X, y = make_classification(n_samples=10000, n_features=20,
                           weights=[0.97, 0.03],  # 3% 正样本
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 使用 SMOTE 处理不平衡
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 训练 XGBoost
xgb_fraud = XGBClassifier(
    n_estimators=100,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42
)
xgb_fraud.fit(X_train_balanced, y_train_balanced)

# 评估
y_pred = xgb_fraud.predict(X_test)
y_prob = xgb_fraud.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

## 模型选择建议

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 小数据集 | Random Forest | 稳定，不易过拟合 |
| 大数据集 | LightGBM | 训练速度快 |
| Kaggle 竞赛 | XGBoost | 性能优异，调参灵活 |
| 类别特征多 | CatBoost | 原生支持类别特征 |
| 需要解释性 | Random Forest | 特征重要性直观 |

## 总结

集成学习是提升模型性能的有效方法：

- **Bagging** 通过并行训练减少方差
- **Boosting** 通过串行训练减少偏差
- **Random Forest** 是 Bagging 的经典实现
- **XGBoost/LightGBM** 是 Boosting 的高效实现

下一篇我们将学习如何正确评估和调优这些模型。
