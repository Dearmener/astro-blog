---
title: '深度学习完全指南（十四）：部署与工程化'
description: '从模型导出到生产部署，全面掌握ONNX、TensorRT、模型压缩与MLOps工程实践'
pubDate: '2024-02-14'
heroImage: ''
category: '技术'
tags: ['深度学习', '模型部署', 'ONNX', 'MLOps']
series: '深度学习完全指南'
seriesOrder: 14
---

## 部署概述

将深度学习模型从研究环境部署到生产环境，需要考虑性能、延迟、资源消耗等多方面因素。

### 部署场景

| 场景 | 特点 | 技术选型 |
|------|------|---------|
| 云端服务 | 高吞吐、弹性扩展 | TF Serving, Triton |
| 边缘设备 | 低功耗、离线 | TFLite, ONNX Runtime |
| 移动端 | 包体小、低延迟 | CoreML, TFLite |
| 浏览器 | 无需后端 | TF.js, ONNX.js |
| 嵌入式 | 极致优化 | TensorRT, OpenVINO |

### 部署流程

```
训练模型 → 模型导出 → 模型优化 → 推理引擎 → 服务化 → 监控运维
```

---

## ONNX：跨框架模型格式

### PyTorch导出ONNX

```python
import torch
import torch.onnx

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.fc = torch.nn.Linear(64 * 224 * 224, 1000)
    
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleModel()
model.eval()

# 导出
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("ONNX模型导出成功")
```

### TensorFlow导出ONNX

```python
import tf2onnx
import tensorflow as tf

# 加载SavedModel
model = tf.saved_model.load('saved_model_dir')

# 转换为ONNX
input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)

# 保存
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
```

### ONNX模型验证

```python
import onnx
import onnxruntime as ort
import numpy as np

# 验证模型
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
print("模型验证通过")

# 打印模型信息
print(f"IR版本: {model.ir_version}")
print(f"Opset版本: {model.opset_import[0].version}")
print("输入:")
for input in model.graph.input:
    print(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
print("输出:")
for output in model.graph.output:
    print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
```

### ONNX Runtime推理

```python
import onnxruntime as ort
import numpy as np

# 创建推理会话
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)

# 获取输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 推理
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
print(f"输出形状: {result[0].shape}")

# 批量推理
class ONNXInference:
    def __init__(self, model_path, device='cuda'):
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def predict(self, input_data):
        return self.session.run([self.output_name], {self.input_name: input_data})[0]
    
    def predict_batch(self, inputs, batch_size=32):
        results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            results.append(self.predict(batch))
        return np.concatenate(results)
```

---

## TensorRT：NVIDIA GPU优化

### ONNX转TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 动态batch
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
    config.add_optimization_profile(profile)
    
    # 构建引擎
    engine = builder.build_engine(network, config)
    
    # 保存
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    return engine

# 使用trtexec命令行工具
# trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=1024
```

### TensorRT推理

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配host和device内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def predict(self, input_data):
        # 复制输入到GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 复制输出到CPU
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host'].copy()

# 使用
trt_model = TRTInference('model.trt')
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = trt_model.predict(input_data)
```

---

## 模型压缩

### 量化（Quantization）

#### PyTorch动态量化

```python
import torch

# 动态量化 (推理时量化权重)
model_fp32 = MyModel()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# 比较大小
import os
torch.save(model_fp32.state_dict(), 'model_fp32.pt')
torch.save(model_int8.state_dict(), 'model_int8.pt')
print(f"FP32: {os.path.getsize('model_fp32.pt') / 1e6:.2f} MB")
print(f"INT8: {os.path.getsize('model_int8.pt') / 1e6:.2f} MB")
```

#### PyTorch静态量化

```python
import torch
from torch.quantization import get_default_qconfig, prepare, convert

# 准备量化
model = MyModel()
model.eval()
model.qconfig = get_default_qconfig('fbgemm')  # x86 CPU

# 准备：插入观察者
model_prepared = prepare(model)

# 校准：用代表性数据
with torch.no_grad():
    for data, _ in calibration_loader:
        model_prepared(data)

# 转换
model_quantized = convert(model_prepared)
```

#### 量化感知训练（QAT）

```python
import torch
from torch.quantization import prepare_qat, convert

model = MyModel()
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# 准备QAT
model_qat = prepare_qat(model)

# 正常训练
optimizer = torch.optim.Adam(model_qat.parameters())
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model_qat(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 转换为量化模型
model_qat.eval()
model_quantized = convert(model_qat)
```

### 剪枝（Pruning）

```python
import torch
import torch.nn.utils.prune as prune

model = MyModel()

# 非结构化剪枝：按权重大小
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# 结构化剪枝：按通道
prune.ln_structured(model.conv1, name='weight', amount=0.2, n=2, dim=0)

# 全局剪枝
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

# 移除剪枝重参数化
for module, name in parameters_to_prune:
    prune.remove(module, name)

# 计算稀疏度
def compute_sparsity(model):
    zeros = 0
    total = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            zeros += (param == 0).sum().item()
            total += param.numel()
    return zeros / total

print(f"稀疏度: {compute_sparsity(model):.2%}")
```

### 知识蒸馏（Knowledge Distillation）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        # 软标签损失 (KL散度)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # 组合损失
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# 训练
teacher_model = TeacherModel().eval()
student_model = StudentModel()
criterion = DistillationLoss(temperature=4.0, alpha=0.3)
optimizer = torch.optim.Adam(student_model.parameters())

for data, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher_model(data)
    
    student_logits = student_model(data)
    loss = criterion(student_logits, teacher_logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 模型服务化

### FastAPI服务

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io

app = FastAPI()

# 加载模型
model = torch.jit.load('model_scripted.pt')
model.eval()

class PredictionResponse(BaseModel):
    class_id: int
    confidence: float
    class_name: str

CLASSES = ['cat', 'dog', 'bird', ...]

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    return image

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess(image_bytes)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, class_id = torch.max(probabilities, dim=1)
    
    return PredictionResponse(
        class_id=class_id.item(),
        confidence=confidence.item(),
        class_name=CLASSES[class_id.item()]
    )

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 运行: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Triton Inference Server

```python
# model_repository/
# └── my_model/
#     ├── config.pbtxt
#     └── 1/
#         └── model.onnx

# config.pbtxt
"""
name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}
"""

# 客户端
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# 准备输入
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

# 推理
outputs = [httpclient.InferRequestedOutput("output")]
result = client.infer(model_name="my_model", inputs=inputs, outputs=outputs)
output_data = result.as_numpy("output")
```

### TensorFlow Serving

```bash
# Docker部署
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models,target=/models/my_model \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

```python
import requests
import numpy as np

# REST API调用
data = {"instances": np.random.randn(1, 224, 224, 3).tolist()}
response = requests.post(
    "http://localhost:8501/v1/models/my_model:predict",
    json=data
)
predictions = response.json()["predictions"]

# gRPC调用
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.inputs['input'].CopyFrom(tf.make_tensor_proto(input_data))

result = stub.Predict(request)
```

---

## 移动端部署

### TensorFlow Lite

```python
import tensorflow as tf

# 转换
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Python推理
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

```java
// Android推理
import org.tensorflow.lite.Interpreter;

// 加载模型
Interpreter tflite = new Interpreter(loadModelFile());

// 推理
float[][] output = new float[1][1000];
tflite.run(inputBuffer, output);
```

### CoreML (iOS)

```python
import coremltools as ct
import torch

# PyTorch转CoreML
model = torch.jit.load('model_scripted.pt')
model.eval()

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, 224, 224))],
    outputs=[ct.TensorType(name="output")]
)

mlmodel.save('model.mlmodel')
```

```swift
// Swift推理
import CoreML
import Vision

let model = try! VNCoreMLModel(for: MyModel().model)
let request = VNCoreMLRequest(model: model) { request, error in
    guard let results = request.results as? [VNClassificationObservation] else { return }
    print(results.first?.identifier ?? "Unknown")
}

let handler = VNImageRequestHandler(ciImage: image)
try! handler.perform([request])
```

---

## MLOps实践

### 实验追踪：MLflow

```python
import mlflow
import mlflow.pytorch

# 开始实验
mlflow.set_experiment("image_classification")

with mlflow.start_run():
    # 记录参数
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    
    # 训练
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        
        # 记录指标
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        }, step=epoch)
    
    # 保存模型
    mlflow.pytorch.log_model(model, "model")
    
    # 记录artifact
    mlflow.log_artifact("config.yaml")
```

### 模型注册与版本管理

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 注册模型
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "ImageClassifier")

# 转换到生产
client.transition_model_version_stage(
    name="ImageClassifier",
    version=mv.version,
    stage="Production"
)

# 加载生产模型
model = mlflow.pytorch.load_model("models:/ImageClassifier/Production")
```

### CI/CD Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/
      
      - name: Train model
        run: python train.py
      
      - name: Evaluate model
        run: python evaluate.py
      
      - name: Export model
        run: python export_onnx.py
      
      - name: Upload artifact
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: model.onnx

  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - name: Download artifact
        uses: actions/download-artifact@v2
        with:
          name: model
      
      - name: Deploy to production
        run: |
          # 部署逻辑
```

### 监控与告警

```python
from prometheus_client import Counter, Histogram, start_http_server
import time

# 定义指标
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['model', 'status'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
PREDICTION_CONFIDENCE = Histogram('prediction_confidence', 'Prediction confidence scores')

class MonitoredModel:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    
    def predict(self, input_data):
        start_time = time.time()
        
        try:
            output = self.model(input_data)
            confidence = float(torch.max(torch.softmax(output, dim=1)))
            
            # 记录指标
            PREDICTION_COUNT.labels(model=self.model_name, status='success').inc()
            PREDICTION_LATENCY.observe(time.time() - start_time)
            PREDICTION_CONFIDENCE.observe(confidence)
            
            return output
            
        except Exception as e:
            PREDICTION_COUNT.labels(model=self.model_name, status='error').inc()
            raise e

# 启动metrics服务器
start_http_server(8080)
```

### 数据漂移检测

```python
import numpy as np
from scipy import stats

class DataDriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, new_data):
        """使用KS检验检测数据漂移"""
        drift_detected = {}
        
        for feature in range(self.reference_data.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.reference_data[:, feature],
                new_data[:, feature]
            )
            
            drift_detected[feature] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift': p_value < self.threshold
            }
        
        return drift_detected

# 使用
detector = DataDriftDetector(training_data)
drift_report = detector.detect_drift(production_data)

if any(f['drift'] for f in drift_report.values()):
    print("警告：检测到数据漂移！")
```

---

## 性能优化

### 批处理优化

```python
import asyncio
from collections import deque
import time

class BatchingServer:
    def __init__(self, model, max_batch_size=32, max_wait_time=0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.lock = asyncio.Lock()
    
    async def predict(self, input_data):
        future = asyncio.Future()
        
        async with self.lock:
            self.queue.append((input_data, future))
            
            if len(self.queue) >= self.max_batch_size:
                await self._process_batch()
        
        # 等待结果或超时
        await asyncio.wait_for(future, timeout=1.0)
        return future.result()
    
    async def _process_batch(self):
        batch_items = []
        while self.queue and len(batch_items) < self.max_batch_size:
            batch_items.append(self.queue.popleft())
        
        if not batch_items:
            return
        
        # 批量推理
        inputs = torch.stack([item[0] for item in batch_items])
        outputs = self.model(inputs)
        
        # 返回结果
        for i, (_, future) in enumerate(batch_items):
            future.set_result(outputs[i])
```

### 模型缓存

```python
from functools import lru_cache
import hashlib

class CachedModel:
    def __init__(self, model, cache_size=1000):
        self.model = model
        self.cache_size = cache_size
        self._cache = {}
    
    def _hash_input(self, input_data):
        return hashlib.md5(input_data.tobytes()).hexdigest()
    
    def predict(self, input_data):
        key = self._hash_input(input_data)
        
        if key in self._cache:
            return self._cache[key]
        
        result = self.model(input_data)
        
        if len(self._cache) >= self.cache_size:
            # LRU淘汰
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[key] = result
        return result
```

---

## 部署清单

| 阶段 | 检查项 |
|------|--------|
| **导出** | 模型格式、动态batch、精度验证 |
| **优化** | 量化、剪枝、图优化 |
| **测试** | 精度对比、性能基准、边界用例 |
| **服务** | 负载均衡、健康检查、超时处理 |
| **监控** | 延迟、吞吐、错误率、资源使用 |
| **运维** | 版本回滚、A/B测试、灰度发布 |

---

## 小结

| 技术 | 适用场景 | 性能提升 |
|------|---------|---------|
| ONNX | 跨框架部署 | 10-30% |
| TensorRT | NVIDIA GPU | 2-5x |
| 量化 | 边缘设备 | 2-4x |
| 剪枝 | 模型压缩 | 1.5-3x |
| 知识蒸馏 | 小模型 | - |
| 批处理 | 高吞吐 | 2-10x |

**系列完结**：本系列从深度学习基础概念到生产部署，涵盖了深度学习的完整知识体系。希望能帮助你建立系统的深度学习知识框架，在实践中不断深入探索！
