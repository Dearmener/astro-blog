---
title: '数据预处理与特征工程'
description: '机器学习数据预处理全流程：清洗、转换、特征选择与特征构造。'
pubDate: 2025-08-10
category: '技术'
tags: ['数据预处理', '特征工程', 'Pandas', '数据清洗']
series: '机器学习完全指南'
seriesOrder: 3
heroImage: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=400&fit=crop'
---

数据预处理和特征工程是机器学习项目中最重要的环节，直接决定模型性能。本文将详细介绍完整的数据处理流程。

## 数据预处理流程

```
原始数据 → 数据清洗 → 数据转换 → 特征工程 → 特征选择 → 模型训练
```

## 数据加载与探索

### 加载数据

```python
import pandas as pd
import numpy as np

# 读取 CSV
df = pd.read_csv('data.csv')

# 读取 Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 读取 JSON
df = pd.read_json('data.json')

# 基本信息
print(df.shape)
print(df.info())
print(df.describe())
```

### 探索性数据分析 (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 查看前几行
df.head()

# 数据类型
df.dtypes

# 缺失值统计
df.isnull().sum()

# 数值列分布
df.describe()

# 分布可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 直方图
df['age'].hist(ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

# 箱线图
df.boxplot(column='salary', ax=axes[0, 1])
axes[0, 1].set_title('Salary Boxplot')

# 相关性热力图
sns.heatmap(df.corr(), annot=True, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Matrix')

# 类别分布
df['category'].value_counts().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Category Distribution')

plt.tight_layout()
plt.show()
```

## 数据清洗

### 处理缺失值

```python
# 检查缺失值
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'count': missing, 'percent': missing_pct}))

# 删除缺失值
df_dropped = df.dropna()  # 删除任何有缺失的行
df_dropped = df.dropna(subset=['important_col'])  # 只看特定列

# 填充缺失值
df['age'].fillna(df['age'].mean(), inplace=True)  # 均值填充
df['category'].fillna(df['category'].mode()[0], inplace=True)  # 众数填充
df['salary'].fillna(df['salary'].median(), inplace=True)  # 中位数填充

# 前向/后向填充 (时序数据)
df['value'].fillna(method='ffill', inplace=True)
df['value'].fillna(method='bfill', inplace=True)

# 插值填充
df['value'].interpolate(method='linear', inplace=True)
```

### 处理重复值

```python
# 检查重复
duplicates = df.duplicated().sum()
print(f"Duplicates: {duplicates}")

# 删除重复
df = df.drop_duplicates()

# 基于特定列去重
df = df.drop_duplicates(subset=['id', 'date'], keep='first')
```

### 处理异常值

```python
# IQR 方法检测异常值
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

outliers, lower, upper = detect_outliers_iqr(df, 'salary')
print(f"Outliers: {len(outliers)}")

# 处理异常值
# 方法1: 删除
df = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

# 方法2: 截断
df['salary'] = df['salary'].clip(lower=lower, upper=upper)

# 方法3: 替换为边界值
df.loc[df['salary'] < lower, 'salary'] = lower
df.loc[df['salary'] > upper, 'salary'] = upper
```

## 数据转换

### 数值型特征处理

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化 (Z-score)
scaler = StandardScaler()
df['salary_scaled'] = scaler.fit_transform(df[['salary']])

# 归一化 (Min-Max)
scaler = MinMaxScaler()
df['age_normalized'] = scaler.fit_transform(df[['age']])

# 鲁棒缩放 (对异常值不敏感)
scaler = RobustScaler()
df['income_robust'] = scaler.fit_transform(df[['income']])

# 对数变换 (处理偏斜分布)
df['salary_log'] = np.log1p(df['salary'])

# Box-Cox 变换
from scipy import stats
df['salary_boxcox'], _ = stats.boxcox(df['salary'] + 1)
```

### 类别型特征处理

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (有序类别)
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# One-Hot Encoding (无序类别)
df_encoded = pd.get_dummies(df, columns=['city', 'gender'], drop_first=True)

# 使用 sklearn
ohe = OneHotEncoder(sparse=False, drop='first')
encoded = ohe.fit_transform(df[['city']])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

# 目标编码 (Target Encoding)
def target_encode(df, column, target):
    means = df.groupby(column)[target].mean()
    return df[column].map(means)

df['city_target_encoded'] = target_encode(df, 'city', 'salary')
```

### 日期特征处理

```python
# 转换为日期类型
df['date'] = pd.to_datetime(df['date'])

# 提取日期特征
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# 计算时间差
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

## 特征工程

### 特征构造

```python
# 数值组合
df['price_per_sqft'] = df['price'] / df['area']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# 多项式特征
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['age', 'income']])

# 分箱 (Binning)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], 
                         labels=['少年', '青年', '中年', '中老年', '老年'])

# 等频分箱
df['income_quantile'] = pd.qcut(df['income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# 交互特征
df['age_income'] = df['age'] * df['income']
```

### 文本特征

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 词频统计
vectorizer = CountVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(df['description'])

# TF-IDF
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(df['description'])

# 基本文本统计
df['text_length'] = df['description'].str.len()
df['word_count'] = df['description'].str.split().str.len()
```

## 特征选择

### 过滤法

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# 方差阈值
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

# 卡方检验 (分类)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 互信息
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 相关性过滤
correlation_matrix = df.corr()
high_corr = (correlation_matrix.abs() > 0.8) & (correlation_matrix != 1)
```

### 包装法

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 递归特征消除
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
selected_features = X.columns[rfe.support_]
```

### 嵌入法

```python
from sklearn.ensemble import RandomForestClassifier

# 基于模型的特征重要性
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 特征重要性
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))

# L1 正则化选择
from sklearn.linear_model import LassoCV
lasso = LassoCV()
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0]
```

## 数据管道

### Pipeline 构建

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 定义预处理管道
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'education']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 完整管道
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# 训练
full_pipeline.fit(X_train, y_train)

# 预测
predictions = full_pipeline.predict(X_test)
```

## 总结

数据预处理的关键步骤：

1. **数据探索**: 了解数据分布和质量
2. **数据清洗**: 处理缺失值、异常值、重复值
3. **数据转换**: 标准化、编码、变换
4. **特征工程**: 构造有意义的新特征
5. **特征选择**: 选择最相关的特征
6. **Pipeline**: 构建可复用的处理流程

下一篇将详细介绍监督学习中的回归算法。
