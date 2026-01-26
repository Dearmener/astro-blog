---
title: '机器学习模型部署与 MLOps'
description: '全面掌握机器学习模型部署方法，包括 Flask/FastAPI API 服务、Docker 容器化、MLflow 实验管理和 CI/CD 自动化'
pubDate: 2025-09-25
category: '技术'
tags: ['机器学习', 'MLOps', 'Docker', 'MLflow', 'FastAPI']
series: '机器学习完全指南'
seriesOrder: 12
heroImage: 'https://images.unsplash.com/photo-1667372393119-3d4c48d07fc9?w=800&h=400&fit=crop'
---

## MLOps 概述

MLOps（Machine Learning Operations）是将机器学习模型从实验环境可靠地部署到生产环境的实践方法，结合了 DevOps 的理念与机器学习的特殊需求。

### MLOps 生命周期

```
数据准备 → 特征工程 → 模型训练 → 模型评估 → 模型部署 → 监控反馈
    ↑                                                      ↓
    ←←←←←←←←←←←←←← 持续迭代 ←←←←←←←←←←←←←←←←←←←←←←←←
```

## 模型序列化

部署前首先需要将模型保存为可移植的格式。

### Joblib / Pickle

```python
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 训练模型
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 使用 Joblib 保存（推荐用于 sklearn 模型）
joblib.dump(model, 'model.joblib')

# 使用 Pickle 保存
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载模型
loaded_model = joblib.load('model.joblib')
print(f"加载模型准确率: {loaded_model.score(X_test, y_test):.4f}")
```

### ONNX 格式（跨平台）

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 转换为 ONNX 格式
initial_type = [('float_input', FloatTensorType([None, 20]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# 保存 ONNX 模型
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# 使用 ONNX Runtime 进行推理
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession('model.onnx')
input_name = sess.get_inputs()[0].name
pred = sess.run(None, {input_name: X_test.astype(np.float32)})[0]
print(f"ONNX 预测结果: {pred[:5]}")
```

## FastAPI 部署

FastAPI 是现代高性能的 Python Web 框架，非常适合部署机器学习模型。

### 基础 API 服务

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# 创建应用
app = FastAPI(
    title="ML Model API",
    description="机器学习模型预测服务",
    version="1.0.0"
)

# 加载模型
model = joblib.load('model.joblib')

# 定义请求体
class PredictionRequest(BaseModel):
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.1, 0.2, 0.3, 0.4, 0.5] * 4
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 预测接口
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 批量预测
class BatchPredictionRequest(BaseModel):
    instances: List[List[float]]

@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    features = np.array(request.instances)
    predictions = model.predict(features).tolist()
    probabilities = model.predict_proba(features).tolist()
    
    return {
        "predictions": predictions,
        "probabilities": probabilities
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 运行与测试

```bash
# 安装依赖
pip install fastapi uvicorn

# 运行服务
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 测试 API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]}'
```

## Docker 容器化

使用 Docker 将模型服务容器化，确保环境一致性。

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY app.py .
COPY model.joblib .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.txt

```text
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
joblib==1.3.2
numpy==1.26.2
pydantic==2.5.2
```

### 构建与运行

```bash
# 构建镜像
docker build -t ml-model-api:v1 .

# 运行容器
docker run -d -p 8000:8000 --name ml-api ml-model-api:v1

# 查看日志
docker logs ml-api

# 测试
curl http://localhost:8000/health
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model.joblib
    volumes:
      - ./models:/app/models
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## MLflow 实验管理

MLflow 是开源的机器学习生命周期管理平台。

### 实验跟踪

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# 设置实验
mlflow.set_experiment("customer-churn-prediction")

# 开始运行
with mlflow.start_run(run_name="random_forest_v1"):
    # 参数
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    }
    
    # 记录参数
    mlflow.log_params(params)
    
    # 训练模型
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 记录指标
    mlflow.log_metrics({
        "accuracy": accuracy,
        "f1_score": f1
    })
    
    # 记录模型
    mlflow.sklearn.log_model(model, "model")
    
    # 记录额外文件
    # mlflow.log_artifact("feature_importance.png")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
```

### 模型注册

```python
# 注册模型到 Model Registry
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
mlflow.register_model(model_uri, "ChurnPredictionModel")

# 加载已注册的模型
model = mlflow.sklearn.load_model("models:/ChurnPredictionModel/1")

# 或加载最新生产版本
model = mlflow.sklearn.load_model("models:/ChurnPredictionModel/Production")
```

### MLflow 服务部署

```bash
# 启动 MLflow 服务器
mlflow server --host 0.0.0.0 --port 5000

# 使用 MLflow 模型服务
mlflow models serve -m "models:/ChurnPredictionModel/Production" -p 8001

# 测试
curl -X POST http://localhost:8001/invocations \
     -H "Content-Type: application/json" \
     -d '{"inputs": [[0.1, 0.2, ...]]}'
```

## CI/CD 自动化

### GitHub Actions 工作流

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Train model
        run: python train.py
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: model
          path: model.joblib

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: model
      
      - name: Build and push Docker image
        run: |
          docker build -t ml-model:${{ github.sha }} .
          docker push registry.example.com/ml-model:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/ml-api \
            ml-api=registry.example.com/ml-model:${{ github.sha }}
```

## 模型监控

### 性能监控

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# 定义指标
PREDICTION_COUNT = Counter(
    'ml_predictions_total', 
    'Total number of predictions',
    ['model_version', 'prediction']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds'
)

# 在预测函数中使用
@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()
    
    prediction = model.predict(features)[0]
    
    # 记录指标
    PREDICTION_COUNT.labels(
        model_version="v1",
        prediction=str(prediction)
    ).inc()
    
    PREDICTION_LATENCY.observe(time.time() - start_time)
    
    return {"prediction": prediction}

# 启动 Prometheus 指标服务
start_http_server(9090)
```

### 数据漂移检测

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 定义列映射
column_mapping = ColumnMapping(
    target='target',
    numerical_features=['feature_1', 'feature_2', ...],
    categorical_features=['cat_feature_1', ...]
)

# 创建数据漂移报告
report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=train_data,
    current_data=production_data,
    column_mapping=column_mapping
)

# 保存报告
report.save_html('drift_report.html')

# 检查是否存在漂移
drift_detected = report.as_dict()['metrics'][0]['result']['dataset_drift']
if drift_detected:
    print("警告：检测到数据漂移，可能需要重新训练模型！")
```

## 部署架构示例

```
                    ┌─────────────┐
                    │   用户请求    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  负载均衡器   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │ API Pod │       │ API Pod │       │ API Pod │
    │ (模型)  │       │ (模型)  │       │ (模型)  │
    └────┬────┘       └────┬────┘       └────┬────┘
         │                 │                 │
         └─────────────────┼─────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Redis 缓存  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   MLflow    │
                    │ Model Store │
                    └─────────────┘
```

## 最佳实践

| 阶段 | 最佳实践 |
|------|----------|
| 开发 | 使用 MLflow 跟踪实验 |
| 打包 | 使用 Docker 容器化 |
| 部署 | 使用 Kubernetes 编排 |
| 监控 | 使用 Prometheus + Grafana |
| 版本 | 模型版本与代码版本关联 |

## 总结

MLOps 的核心要点：

1. **可复现性**：代码、数据、模型版本化
2. **自动化**：CI/CD 流水线
3. **监控**：性能指标、数据漂移
4. **可扩展性**：容器化、微服务架构
5. **协作**：实验跟踪、模型注册

恭喜你完成了机器学习完全指南系列！现在你已经掌握了从数据预处理到模型部署的完整机器学习流程。
