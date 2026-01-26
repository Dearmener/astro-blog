---
title: '监督学习 - 分类算法详解'
description: '逻辑回归、决策树、SVM、KNN 等分类算法的原理、实现与应用。'
pubDate: 2025-08-20
category: '技术'
tags: ['分类', '监督学习', 'SVM', '决策树']
series: '机器学习完全指南'
seriesOrder: 5
heroImage: 'https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=800&h=400&fit=crop'
---

分类是监督学习中预测离散类别的任务。本文将详细介绍常用分类算法的原理和实现。

## 分类任务概述

### 类型

- **二分类**: 两个类别（垃圾邮件检测）
- **多分类**: 多个类别（图像分类）
- **多标签**: 一个样本多个标签

### 评估指标

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

def evaluate_classifier(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
```

## 逻辑回归

### 原理

通过 Sigmoid 函数将线性输出映射到概率：

```
P(y=1|x) = 1 / (1 + e^(-wx-b))
```

### 代码实现

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

evaluate_classifier(y_test, y_pred)
```

## 决策树

### 原理

通过树形结构进行决策，每个节点基于特征划分数据。

### 代码实现

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 训练
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
evaluate_classifier(y_test, y_pred)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=iris.feature_names, 
          class_names=iris.target_names, filled=True)
plt.show()

# 特征重要性
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

## 支持向量机 (SVM)

### 原理

寻找最大间隔的决策边界，支持非线性分类。

### 代码实现

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 标准化 (SVM 对尺度敏感)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性核
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_scaled, y_train)

# RBF 核 (高斯核)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)

# 多项式核
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train_scaled, y_train)

# 评估
for name, model in [('Linear', svm_linear), ('RBF', svm_rbf), ('Poly', svm_poly)]:
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} SVM Accuracy: {acc:.4f}")
```

## K 近邻 (KNN)

### 原理

基于距离，找到 K 个最近邻居进行投票。

### 代码实现

```python
from sklearn.neighbors import KNeighborsClassifier

# 不同 K 值对比
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k}: Accuracy = {acc:.4f}")

# 最佳 KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)
```

## 朴素贝叶斯

### 原理

基于贝叶斯定理，假设特征条件独立。

### 代码实现

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# 高斯朴素贝叶斯 (连续特征)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"Gaussian NB Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 多项式朴素贝叶斯 (文本分类)
# mnb = MultinomialNB()
# mnb.fit(X_train_counts, y_train)
```

## 随机森林

### 原理

集成多棵决策树，通过投票或平均得到最终结果。

### 代码实现

```python
from sklearn.ensemble import RandomForestClassifier

# 训练
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

evaluate_classifier(y_test, y_pred)

# 特征重要性
importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

## 梯度提升

### XGBoost

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

## 模型对比

```python
from sklearn.model_selection import cross_val_score

models = {
    'Logistic': LogisticRegression(max_iter=200),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

## 算法选择指南

| 场景 | 推荐算法 |
|------|----------|
| 线性可分、可解释性 | 逻辑回归 |
| 特征重要性 | 决策树、随机森林 |
| 小样本、高维 | SVM |
| 快速原型 | KNN |
| 文本分类 | 朴素贝叶斯 |
| 最高性能 | XGBoost、LightGBM |

## 总结

分类算法的关键要点：

1. **数据预处理**: 标准化对 SVM、KNN 很重要
2. **模型选择**: 根据数据特点选择算法
3. **评估指标**: 不平衡数据关注 F1、AUC
4. **超参数调优**: 网格搜索、随机搜索
5. **集成方法**: 通常性能最好

下一篇将介绍无监督学习中的聚类与降维。
