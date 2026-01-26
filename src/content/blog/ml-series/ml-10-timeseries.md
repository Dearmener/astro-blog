---
title: '时间序列分析与预测'
description: '掌握时间序列数据处理、ARIMA 模型、Prophet 预测以及 LSTM 深度学习方法进行时序预测'
pubDate: 2025-09-15
category: '技术'
tags: ['机器学习', '时间序列', 'ARIMA', 'Prophet', 'LSTM']
series: '机器学习完全指南'
seriesOrder: 10
heroImage: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800&h=400&fit=crop'
---

## 时间序列基础

时间序列是按时间顺序排列的数据点序列，广泛应用于股票预测、销量预测、天气预报等场景。

### 时间序列的组成

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 生成模拟时间序列数据
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=730, freq='D')

# 构建包含趋势、季节性和噪声的数据
trend = np.linspace(100, 200, 730)  # 趋势
seasonal = 30 * np.sin(2 * np.pi * np.arange(730) / 365)  # 年度季节性
weekly = 10 * np.sin(2 * np.pi * np.arange(730) / 7)  # 周季节性
noise = np.random.normal(0, 10, 730)  # 噪声

sales = trend + seasonal + weekly + noise
df = pd.DataFrame({'date': dates, 'sales': sales})
df.set_index('date', inplace=True)

# 可视化
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
axes[0].plot(df.index, sales, label='原始数据')
axes[0].set_title('原始时间序列')
axes[1].plot(df.index, trend, label='趋势', color='orange')
axes[1].set_title('趋势分量')
axes[2].plot(df.index, seasonal + weekly, label='季节性', color='green')
axes[2].set_title('季节性分量')
axes[3].plot(df.index, noise, label='噪声', color='red')
axes[3].set_title('随机噪声')
plt.tight_layout()
plt.show()
```

### 时间序列分解

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# 分解时间序列
decomposition = seasonal_decompose(df['sales'], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposition.observed.plot(ax=axes[0], title='观测值')
decomposition.trend.plot(ax=axes[1], title='趋势')
decomposition.seasonal.plot(ax=axes[2], title='季节性')
decomposition.resid.plot(ax=axes[3], title='残差')
plt.tight_layout()
plt.show()
```

## 平稳性检验

ARIMA 等模型要求时间序列是平稳的，需要先进行平稳性检验。

### ADF 检验

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    """ADF 平稳性检验"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'=== {name} ADF 检验 ===')
    print(f'ADF 统计量: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'临界值:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.4f}')
    print(f'结论: {"平稳" if result[1] < 0.05 else "非平稳"}')
    return result[1] < 0.05

# 原始数据检验
adf_test(df['sales'], '原始数据')

# 差分后检验
df['sales_diff'] = df['sales'].diff()
adf_test(df['sales_diff'].dropna(), '一阶差分')
```

### 差分处理

```python
# 一阶差分
df['diff_1'] = df['sales'].diff(1)

# 季节性差分（年度）
df['diff_seasonal'] = df['sales'].diff(365)

# 可视化差分效果
fig, axes = plt.subplots(3, 1, figsize=(14, 8))
axes[0].plot(df['sales'])
axes[0].set_title('原始数据')
axes[1].plot(df['diff_1'])
axes[1].set_title('一阶差分')
axes[2].plot(df['diff_seasonal'].dropna())
axes[2].set_title('季节性差分')
plt.tight_layout()
plt.show()
```

## ARIMA 模型

ARIMA（自回归积分滑动平均）是经典的时间序列预测方法。

### ACF 和 PACF 分析

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['sales_diff'].dropna(), ax=axes[0], lags=40)
axes[0].set_title('自相关函数 (ACF)')
plot_pacf(df['sales_diff'].dropna(), ax=axes[1], lags=40)
axes[1].set_title('偏自相关函数 (PACF)')
plt.tight_layout()
plt.show()
```

### 自动 ARIMA 参数选择

```python
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 手动网格搜索最优参数
def find_best_arima(data, p_range, d_range, q_range):
    best_aic = float('inf')
    best_order = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
    
    return best_order, best_aic

# 搜索最优参数
best_order, best_aic = find_best_arima(
    df['sales'], 
    p_range=range(0, 4), 
    d_range=range(0, 2), 
    q_range=range(0, 4)
)
print(f"最优参数: {best_order}, AIC: {best_aic:.2f}")
```

### ARIMA 训练与预测

```python
# 划分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df['sales'][:train_size], df['sales'][train_size:]

# 训练 ARIMA 模型
model = ARIMA(train, order=best_order)
fitted_model = model.fit()
print(fitted_model.summary())

# 预测
forecast = fitted_model.forecast(steps=len(test))

# 可视化
plt.figure(figsize=(14, 6))
plt.plot(train.index, train, label='训练数据')
plt.plot(test.index, test, label='实际值')
plt.plot(test.index, forecast, label='预测值', color='red')
plt.legend()
plt.title('ARIMA 预测结果')
plt.show()

# 评估
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
```

## Prophet 预测

Prophet 是 Facebook 开发的时间序列预测库，特别适合有明显季节性和节假日效应的数据。

### Prophet 基础使用

```python
from prophet import Prophet

# 准备 Prophet 格式的数据
prophet_df = df.reset_index()
prophet_df.columns = ['ds', 'y']
prophet_df = prophet_df[['ds', 'y']]

# 划分数据
train_prophet = prophet_df[:train_size]
test_prophet = prophet_df[train_size:]

# 创建和训练模型
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive'
)
model.fit(train_prophet)

# 创建未来日期框架
future = model.make_future_dataframe(periods=len(test_prophet))

# 预测
forecast = model.predict(future)

# 可视化
fig = model.plot(forecast)
plt.title('Prophet 预测结果')
plt.show()

# 组件分解
fig = model.plot_components(forecast)
plt.show()
```

### Prophet 高级配置

```python
# 添加节假日效应
holidays = pd.DataFrame({
    'holiday': 'chinese_new_year',
    'ds': pd.to_datetime(['2020-01-25', '2021-02-12', '2022-02-01']),
    'lower_window': -3,
    'upper_window': 7,
})

# 高级配置
model = Prophet(
    growth='linear',  # 'linear' 或 'logistic'
    changepoint_prior_scale=0.05,  # 趋势变化点灵活性
    seasonality_prior_scale=10,  # 季节性强度
    holidays=holidays,
    yearly_seasonality=10,  # 傅里叶阶数
    weekly_seasonality=3
)

# 添加自定义季节性
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# 添加额外回归变量
# model.add_regressor('temperature')

model.fit(train_prophet)
```

### Prophet 交叉验证

```python
from prophet.diagnostics import cross_validation, performance_metrics

# 交叉验证
df_cv = cross_validation(
    model, 
    initial='365 days',  # 初始训练期
    period='30 days',    # 每次移动的间隔
    horizon='90 days'    # 预测范围
)

# 性能指标
df_metrics = performance_metrics(df_cv)
print(df_metrics[['horizon', 'mse', 'rmse', 'mae', 'mape']].tail())
```

## LSTM 时间序列预测

使用深度学习方法进行时间序列预测。

### 数据预处理

```python
from sklearn.preprocessing import MinMaxScaler

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['sales'].values.reshape(-1, 1))

# 创建序列数据
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 30  # 使用过去30天预测
X, y = create_sequences(scaled_data, seq_length)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 重塑为 LSTM 输入格式 [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
```

### 构建 LSTM 模型

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 构建模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 可视化训练过程
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.title('LSTM 训练过程')
plt.show()
```

### LSTM 预测与评估

```python
# 预测
y_pred = model.predict(X_test)

# 反归一化
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# 评估
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"LSTM - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# 可视化
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv, label='实际值')
plt.plot(y_pred_inv, label='预测值')
plt.legend()
plt.title('LSTM 预测结果')
plt.show()
```

## 模型对比

```python
# 对比各模型性能
results = pd.DataFrame({
    '模型': ['ARIMA', 'Prophet', 'LSTM'],
    'RMSE': [15.23, 12.45, 10.87],  # 示例数据
    'MAE': [12.10, 10.23, 8.92]
})

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results['模型']))
width = 0.35
ax.bar(x - width/2, results['RMSE'], width, label='RMSE')
ax.bar(x + width/2, results['MAE'], width, label='MAE')
ax.set_xticks(x)
ax.set_xticklabels(results['模型'])
ax.legend()
ax.set_title('模型性能对比')
plt.show()
```

## 实战建议

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 小数据量 | ARIMA | 简单高效 |
| 有季节性 | Prophet | 自动处理季节性 |
| 复杂模式 | LSTM | 强大的非线性拟合 |
| 多变量 | LSTM/Prophet | 支持外部变量 |
| 需要解释 | Prophet | 组件分解清晰 |

## 总结

时间序列预测的关键步骤：

1. **数据探索**：理解趋势、季节性、周期性
2. **平稳性检验**：ADF 检验确定差分阶数
3. **模型选择**：根据数据特点选择合适模型
4. **参数调优**：网格搜索或自动调参
5. **评估验证**：使用时序交叉验证

下一篇我们将学习 AutoML 自动化机器学习。
