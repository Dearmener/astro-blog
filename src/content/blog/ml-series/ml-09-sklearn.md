---
title: 'Scikit-learn 实战项目'
description: '从数据加载到模型部署的完整机器学习项目实战，掌握 Pipeline、特征工程、模型训练与评估全流程'
pubDate: 2025-09-10
category: '技术'
tags: ['机器学习', 'Scikit-learn', '实战项目', 'Pipeline', '特征工程']
series: '机器学习完全指南'
seriesOrder: 9
heroImage: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=400&fit=crop'
---

## 项目概述：客户流失预测

本文将完成一个端到端的机器学习项目：电信客户流失预测。我们将使用 Scikit-learn 的 Pipeline 构建完整的机器学习工作流。

### 数据准备

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 模拟电信客户数据
np.random.seed(42)
n_samples = 5000

data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 100, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                        'Bank transfer', 'Credit card'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'online_security': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n_samples),
    'senior_citizen': np.random.randint(0, 2, n_samples),
    'partner': np.random.choice(['Yes', 'No'], n_samples),
    'dependents': np.random.choice(['Yes', 'No'], n_samples),
}

# 生成目标变量（流失与否）
df = pd.DataFrame(data)
churn_prob = 0.2 + 0.3 * (df['contract'] == 'Month-to-month').astype(int) \
           + 0.1 * (df['tenure'] < 12).astype(int) \
           - 0.2 * (df['contract'] == 'Two year').astype(int)
churn_prob = np.clip(churn_prob, 0.05, 0.95)
df['churn'] = np.random.binomial(1, churn_prob)

print(f"数据集形状: {df.shape}")
print(f"流失率: {df['churn'].mean():.2%}")
print(df.head())
```

### 探索性数据分析

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 数值特征分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(['tenure', 'monthly_charges', 'total_charges']):
    sns.histplot(data=df, x=col, hue='churn', ax=axes[i], kde=True)
    axes[i].set_title(f'{col} 分布')
plt.tight_layout()
plt.show()

# 类别特征与流失关系
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cat_cols = ['contract', 'payment_method', 'internet_service', 
            'online_security', 'tech_support', 'partner']
for i, col in enumerate(cat_cols):
    ax = axes[i//3, i%3]
    df.groupby(col)['churn'].mean().plot(kind='bar', ax=ax)
    ax.set_title(f'{col} 流失率')
    ax.set_ylabel('流失率')
plt.tight_layout()
plt.show()
```

## 构建 Pipeline

### 定义特征处理流程

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 定义特征类型
numeric_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract', 'payment_method', 'internet_service',
                       'online_security', 'tech_support', 'partner', 'dependents']
binary_features = ['senior_citizen']

# 数值特征处理流程
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别特征处理流程
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, 
                            handle_unknown='ignore'))
])

# 组合预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

print("预处理器构建完成")
```

### 完整 Pipeline 与模型

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 创建完整 Pipeline
def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# 准备数据
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 定义多个模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, 
                             use_label_encoder=False, eval_metric='logloss')
}
```

## 模型训练与评估

### 交叉验证比较

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    pipeline = create_pipeline(model)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, 
                            scoring='roc_auc', n_jobs=-1)
    results[name] = scores
    print(f"{name}: AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# 可视化比较
plt.figure(figsize=(10, 6))
plt.boxplot([results[name] for name in models.keys()], labels=models.keys())
plt.ylabel('ROC-AUC')
plt.title('模型交叉验证比较')
plt.xticks(rotation=15)
plt.show()
```

### 详细评估最佳模型

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)

# 选择最佳模型训练
best_pipeline = create_pipeline(XGBClassifier(
    n_estimators=100, random_state=42,
    use_label_encoder=False, eval_metric='logloss'
))
best_pipeline.fit(X_train, y_train)

# 预测
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

# 分类报告
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=['留存', '流失']))

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['留存', '流失'], yticklabels=['留存', '流失'])
plt.xlabel('预测')
plt.ylabel('实际')
plt.title('混淆矩阵')
plt.show()

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC 曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC 曲线')
plt.legend()
plt.show()
```

## 超参数调优

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# 定义参数空间（注意 Pipeline 参数命名）
param_distributions = {
    'classifier__n_estimators': randint(50, 300),
    'classifier__max_depth': randint(3, 15),
    'classifier__learning_rate': uniform(0.01, 0.3),
    'classifier__subsample': uniform(0.6, 0.4),
    'classifier__colsample_bytree': uniform(0.6, 0.4)
}

# 随机搜索
pipeline = create_pipeline(XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric='logloss'
))

random_search = RandomizedSearchCV(
    pipeline, param_distributions,
    n_iter=30, cv=5, scoring='roc_auc',
    n_jobs=-1, random_state=42, verbose=1
)

random_search.fit(X_train, y_train)

print(f"\n最佳参数: {random_search.best_params_}")
print(f"最佳交叉验证 AUC: {random_search.best_score_:.4f}")
print(f"测试集 AUC: {random_search.score(X_test, y_test):.4f}")
```

## 特征重要性分析

```python
# 获取特征名称
feature_names = (numeric_features + 
                list(best_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features)) +
                binary_features)

# 获取特征重要性
importances = best_pipeline.named_steps['classifier'].feature_importances_

# 可视化
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][-15:], importance_df['importance'][-15:])
plt.xlabel('重要性')
plt.title('Top 15 特征重要性')
plt.tight_layout()
plt.show()
```

## 模型持久化

```python
import joblib

# 保存模型
joblib.dump(random_search.best_estimator_, 'churn_model.joblib')
print("模型已保存到 churn_model.joblib")

# 加载模型
loaded_model = joblib.load('churn_model.joblib')

# 验证加载的模型
y_pred_loaded = loaded_model.predict(X_test)
print(f"加载模型预测准确率: {(y_pred_loaded == y_test).mean():.4f}")
```

## 预测新数据

```python
# 模拟新客户数据
new_customer = pd.DataFrame({
    'tenure': [6],
    'monthly_charges': [85.5],
    'total_charges': [513.0],
    'contract': ['Month-to-month'],
    'payment_method': ['Electronic check'],
    'internet_service': ['Fiber optic'],
    'online_security': ['No'],
    'tech_support': ['No'],
    'senior_citizen': [0],
    'partner': ['No'],
    'dependents': ['No']
})

# 预测
churn_prob = loaded_model.predict_proba(new_customer)[0, 1]
churn_pred = loaded_model.predict(new_customer)[0]

print(f"客户流失概率: {churn_prob:.2%}")
print(f"预测结果: {'流失' if churn_pred == 1 else '留存'}")

# 批量预测
def predict_churn(df, model):
    """批量预测客户流失"""
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return pd.DataFrame({
        'churn_prediction': predictions,
        'churn_probability': probabilities
    })

# 对测试集进行批量预测
test_predictions = predict_churn(X_test, loaded_model)
print(f"\n高风险客户数量（概率 > 70%）: {(test_predictions['churn_probability'] > 0.7).sum()}")
```

## 自定义 Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class TenureGroupTransformer(BaseEstimator, TransformerMixin):
    """将 tenure 转换为分组"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['tenure_group'] = pd.cut(X['tenure'], 
                                   bins=[0, 12, 24, 48, 72],
                                   labels=['0-1年', '1-2年', '2-4年', '4年+'])
        return X

class ChargeRatioTransformer(BaseEstimator, TransformerMixin):
    """计算费用比率特征"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['charge_ratio'] = X['total_charges'] / (X['tenure'] + 1)
        return X

# 使用自定义 Transformer
from sklearn.pipeline import FeatureUnion

custom_features = FeatureUnion([
    ('tenure_group', TenureGroupTransformer()),
    ('charge_ratio', ChargeRatioTransformer())
])
```

## 项目总结

### 完整工作流程

```
1. 数据加载与探索 (EDA)
   ↓
2. 特征工程（数值/类别处理）
   ↓
3. 构建 Pipeline（预处理 + 模型）
   ↓
4. 交叉验证与模型选择
   ↓
5. 超参数调优（RandomSearch/Optuna）
   ↓
6. 最终评估（测试集）
   ↓
7. 模型持久化与部署
```

### 关键代码模板

```python
# 标准 ML Pipeline 模板
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. 定义预处理器
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 2. 构建 Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 3. 训练与评估
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)

# 4. 保存模型
joblib.dump(pipeline, 'model.joblib')
```

下一篇我们将学习时间序列分析与预测。
