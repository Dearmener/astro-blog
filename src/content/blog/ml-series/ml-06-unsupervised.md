---
title: '无监督学习 - 聚类与降维'
description: 'K-Means、DBSCAN、层次聚类与 PCA、t-SNE 降维技术详解。'
pubDate: 2025-08-25
category: '技术'
tags: ['无监督学习', '聚类', 'PCA', 'K-Means']
series: '机器学习完全指南'
seriesOrder: 6
heroImage: 'https://images.unsplash.com/photo-1509228468518-180dd4864904?w=800&h=400&fit=crop'
---

无监督学习从无标签数据中发现隐藏的结构和模式。本文将介绍聚类和降维两大核心技术。

## 聚类概述

### 应用场景

- 客户细分
- 图像分割
- 异常检测
- 文档聚类

## K-Means 聚类

### 原理

1. 随机初始化 K 个中心点
2. 将每个点分配到最近的中心
3. 更新中心点为簇内均值
4. 重复直到收敛

### 代码实现

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')
plt.show()
```

### 选择最佳 K 值

```python
# 肘部法则
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 轮廓系数
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.4f}")
```

## DBSCAN

### 原理

基于密度的聚类，能发现任意形状的簇，自动识别噪声点。

### 代码实现

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 生成半月形数据
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()

# 统计
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

## 层次聚类

### 代码实现

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 层次聚类
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = agg.fit_predict(X)

# 树状图
Z = linkage(X, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```

## 聚类评估

```python
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, 
    davies_bouldin_score, adjusted_rand_score
)

def evaluate_clustering(X, labels, y_true=None):
    print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
    print(f"Calinski-Harabasz: {calinski_harabasz_score(X, labels):.4f}")
    print(f"Davies-Bouldin: {davies_bouldin_score(X, labels):.4f}")
    
    if y_true is not None:
        print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, labels):.4f}")

evaluate_clustering(X, y_pred, y_true)
```

## 降维概述

### 目的

- 可视化高维数据
- 减少计算复杂度
- 去除噪声和冗余

## PCA 主成分分析

### 原理

找到数据方差最大的方向（主成分），进行线性变换。

### 代码实现

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 加载手写数字数据
digits = load_digits()
X = digits.data  # 64 维
y = digits.target

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"原始维度: {X.shape[1]}")
print(f"降维后: {X_pca.shape[1]}")
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Digits Dataset')
plt.show()
```

### 选择主成分数量

```python
# 累计解释方差
pca_full = PCA()
pca_full.fit(X)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumsum)+1), cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title('PCA - Explained Variance')
plt.show()

# 保留 95% 方差
n_components = np.argmax(cumsum >= 0.95) + 1
print(f"保留 95% 方差需要 {n_components} 个主成分")
```

## t-SNE

### 原理

非线性降维，保持局部结构，适合可视化。

### 代码实现

```python
from sklearn.manifold import TSNE

# t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE of Digits Dataset')
plt.show()
```

### 参数调优

```python
# 不同 perplexity 对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, perp in zip(axes, [5, 30, 50]):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X[:500])
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:500], cmap='tab10', s=10)
    ax.set_title(f'Perplexity = {perp}')

plt.tight_layout()
plt.show()
```

## UMAP

### 代码实现

```python
import umap

# UMAP 降维
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10')
plt.colorbar(scatter)
plt.title('UMAP of Digits Dataset')
plt.show()
```

## PCA vs t-SNE vs UMAP

| 方法 | 类型 | 速度 | 保持结构 | 可逆 |
|------|------|------|----------|------|
| PCA | 线性 | 快 | 全局 | 是 |
| t-SNE | 非线性 | 慢 | 局部 | 否 |
| UMAP | 非线性 | 中等 | 局部+全局 | 否 |

## 实战：客户聚类

```python
import pandas as pd

# 模拟客户数据
np.random.seed(42)
n_customers = 1000

data = {
    'age': np.random.randint(18, 70, n_customers),
    'income': np.random.exponential(50000, n_customers),
    'spending_score': np.random.randint(1, 100, n_customers),
    'purchase_frequency': np.random.poisson(10, n_customers)
}
df = pd.DataFrame(data)

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 分析各簇特征
cluster_summary = df.groupby('cluster').mean()
print(cluster_summary)

# 可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.title('Customer Segments')
plt.show()
```

## 总结

无监督学习要点：

1. **聚类选择**: K-Means 快速，DBSCAN 处理噪声
2. **K 值确定**: 肘部法则 + 轮廓系数
3. **降维选择**: PCA 快速，t-SNE/UMAP 可视化
4. **数据预处理**: 标准化很重要

下一篇将介绍集成学习方法。
