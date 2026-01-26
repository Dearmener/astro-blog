---
title: '监督学习 - 回归算法详解'
description: '线性回归、多项式回归、岭回归、Lasso 回归等回归算法原理与实战。'
pubDate: 2025-08-15
category: '技术'
tags: ['回归', '监督学习', '线性回归', 'Scikit-learn']
series: '机器学习完全指南'
seriesOrder: 4
heroImage: 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800&h=400&fit=crop'
---

回归是监督学习中预测连续数值的任务。本文将详细介绍各种回归算法的原理和应用。

## 线性回归

### 原理

线性回归假设特征与目标之间存在线性关系：

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

目标是找到最优的权重 w 和偏置 b，使预测值与真实值的差距最小。

### 代码实现

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 生成示例数据
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100) * 0.5

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MSE | Σ(y-ŷ)²/n | 均方误差 |
| RMSE | √MSE | 均方根误差 |
| MAE | Σ\|y-ŷ\|/n | 平均绝对误差 |
| R² | 1 - SS_res/SS_tot | 决定系数 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

evaluate_regression(y_test, y_pred)
```

## 多项式回归

### 原理

处理非线性关系，通过添加多项式特征：

```
y = w₀ + w₁x + w₂x² + w₃x³ + ...
```

### 代码实现

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# 创建多项式回归管道
def create_poly_regression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])

# 生成非线性数据
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 + X.ravel() + np.random.randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 对比不同度数
for degree in [1, 2, 3, 5]:
    model = create_poly_regression(degree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Degree {degree}: R² = {r2:.4f}")
```

## 正则化回归

### 岭回归 (Ridge)

L2 正则化，防止过拟合：

```python
from sklearn.linear_model import Ridge, RidgeCV

# 固定 alpha
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# 交叉验证选择最佳 alpha
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
ridge_cv.fit(X_train, y_train)
print(f"最佳 alpha: {ridge_cv.alpha_}")
```

### Lasso 回归

L1 正则化，可以进行特征选择：

```python
from sklearn.linear_model import Lasso, LassoCV

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# 查看非零系数
non_zero = np.sum(lasso.coef_ != 0)
print(f"非零系数数量: {non_zero}")

# 交叉验证
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
print(f"最佳 alpha: {lasso_cv.alpha_}")
```

### 弹性网络 (Elastic Net)

结合 L1 和 L2 正则化：

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)

# 交叉验证
elastic_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5)
elastic_cv.fit(X_train, y_train)
print(f"最佳 alpha: {elastic_cv.alpha_}")
print(f"最佳 l1_ratio: {elastic_cv.l1_ratio_}")
```

## 梯度提升回归

### XGBoost

```python
import xgboost as xgb

# 训练
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# 特征重要性
importance = model.feature_importances_
```

### LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 实战案例：房价预测

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

# 加载数据
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 数据预处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 模型对比
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    results[name] = scores.mean()
    print(f"{name}: R² = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# 最佳模型训练
best_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
print(f"\n测试集 R²: {r2_score(y_test, y_pred):.4f}")
```

## 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
```

## 总结

回归算法选择指南：

| 场景 | 推荐算法 |
|------|----------|
| 线性关系、可解释性 | 线性回归 |
| 多重共线性 | 岭回归 |
| 特征选择 | Lasso |
| 非线性关系 | XGBoost、LightGBM |
| 小样本 | 正则化回归 |

下一篇将介绍监督学习中的分类算法。
