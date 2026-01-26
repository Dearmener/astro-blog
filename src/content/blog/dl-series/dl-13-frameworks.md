---
title: '深度学习完全指南（十三）：深度学习框架对比与实战'
description: 'PyTorch、TensorFlow、JAX三大框架深度对比，掌握各框架特点与最佳实践'
pubDate: '2024-02-13'
heroImage: ''
category: '技术'
tags: ['深度学习', 'PyTorch', 'TensorFlow', 'JAX']
series: '深度学习完全指南'
seriesOrder: 13
---

## 框架概览

深度学习框架是构建和训练神经网络的基础设施。目前三大主流框架各有特色。

### 框架对比

| 特性 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| 开发者 | Meta | Google | Google |
| 计算图 | 动态 | 动态/静态 | 函数式变换 |
| 主要用户 | 研究、学术 | 工业部署 | 前沿研究 |
| 学习曲线 | 平缓 | 中等 | 陡峭 |
| 调试 | 简单 | 中等 | 较难 |
| 部署 | TorchScript, ONNX | TF Serving, TFLite | 需转换 |
| 生态 | Hugging Face, timm | TFHub, Keras | Flax, Haiku |

### 使用趋势

- **PyTorch**: 研究论文首选，Hugging Face生态核心
- **TensorFlow**: 生产部署成熟，移动端/边缘设备强
- **JAX**: 函数式编程，TPU优化，前沿研究

---

## PyTorch深入

### 核心概念

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 张量基础
x = torch.randn(3, 4, requires_grad=True)
y = x * 2 + 1
z = y.sum()
z.backward()  # 自动求导
print(x.grad)  # 梯度

# 设备管理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
model = model.to(device)

# 张量操作
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 矩阵乘法
c = torch.matmul(a, b)  # 或 a @ b

# 广播
d = a + torch.tensor([1, 2])  # 自动广播

# 形状操作
x = torch.randn(2, 3, 4)
x.view(2, 12)          # 改变形状
x.permute(0, 2, 1)     # 交换维度
x.unsqueeze(0)         # 增加维度
x.squeeze()            # 去除大小为1的维度
```

### 模型定义

```python
# 方式1: nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

# 方式2: 继承nn.Module
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 方式3: nn.ModuleList / nn.ModuleDict
class DynamicMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1])
            for i in range(len(layer_dims) - 1)
        ])
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)
```

### 数据加载

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# 自定义Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,  # GPU加速
    drop_last=True    # 丢弃不完整batch
)

# 图像数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 训练循环

```python
def train(model, train_loader, val_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_acc = correct / len(val_loader.dataset)
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, '
              f'Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model
```

### PyTorch Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class LitModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

# 训练
model = LitModel(784, 256, 10)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='gpu',
    devices=1,
    callbacks=[
        ModelCheckpoint(monitor='val_acc', mode='max'),
        EarlyStopping(monitor='val_loss', patience=3)
    ],
    precision=16,  # 混合精度
)

trainer.fit(model, train_loader, val_loader)
```

---

## TensorFlow/Keras深入

### 核心概念

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 张量基础
x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
y = tf.Variable([[1.0, 2.0]], trainable=True)

# 自动求导
with tf.GradientTape() as tape:
    z = tf.reduce_sum(x * y)
grads = tape.gradient(z, y)

# 设备管理
with tf.device('/GPU:0'):
    result = tf.matmul(a, b)

# 查看GPU
print(tf.config.list_physical_devices('GPU'))
```

### 模型定义

```python
# 方式1: Sequential API
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 方式2: Functional API
inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# 方式3: 子类化
class CustomModel(keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense2 = layers.Dense(output_dim)
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

# 多输入多输出
input_a = keras.Input(shape=(64,), name='input_a')
input_b = keras.Input(shape=(32,), name='input_b')

x = layers.concatenate([input_a, input_b])
x = layers.Dense(128, activation='relu')(x)

output_1 = layers.Dense(10, name='output_1')(x)
output_2 = layers.Dense(1, name='output_2')(x)

model = keras.Model(
    inputs=[input_a, input_b],
    outputs=[output_1, output_2]
)
```

### 数据管道

```python
# tf.data API
def create_dataset(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 图像数据增强
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# TFRecord读取
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed['image'])
    image = tf.image.resize(image, [224, 224])
    return image, parsed['label']

dataset = tf.data.TFRecordDataset('data.tfrecord')
dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
```

### 训练方式

```python
# 方式1: model.fit()
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    keras.callbacks.TensorBoard(log_dir='./logs')
]

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=callbacks
)

# 方式2: 自定义训练循环
@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# 方式3: 自定义train_step
class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
```

### 分布式训练

```python
# 镜像策略 (单机多卡)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(...)

model.fit(train_dataset, epochs=10)

# 多机多卡
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# TPU策略
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
```

---

## JAX深入

### 核心概念

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# JAX数组 (不可变)
x = jnp.array([1.0, 2.0, 3.0])
y = x.at[0].set(10.0)  # 创建新数组

# 自动求导
def f(x):
    return jnp.sum(x ** 2)

grad_f = grad(f)
print(grad_f(jnp.array([1.0, 2.0, 3.0])))  # [2., 4., 6.]

# JIT编译
@jit
def slow_f(x):
    return x @ x.T

# 向量化 (vmap)
def single_example_loss(params, x, y):
    pred = model(params, x)
    return (pred - y) ** 2

batch_loss = vmap(single_example_loss, in_axes=(None, 0, 0))

# 并行化 (pmap)
@jax.pmap
def parallel_train_step(params, batch):
    return train_step(params, batch)
```

### 函数式变换

```python
from jax import value_and_grad, hessian

# 同时计算函数值和梯度
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

loss, grads = value_and_grad(loss_fn)(params, x, y)

# 高阶导数
def f(x):
    return x ** 3

df = grad(f)      # 一阶导数
ddf = grad(df)    # 二阶导数
print(ddf(2.0))   # 12.0

# Hessian矩阵
H = hessian(loss_fn)(params, x, y)

# 雅可比矩阵
from jax import jacfwd, jacrev
J_forward = jacfwd(f)(x)   # 前向模式
J_reverse = jacrev(f)(x)   # 反向模式
```

### Flax神经网络

```python
import flax.linen as nn
from flax.training import train_state
import optax

class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# 初始化
model = MLP(hidden_dim=256, output_dim=10)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones([1, 784]))

# 前向传播
output = model.apply(params, x, training=True, rngs={'dropout': key})
```

### 训练循环

```python
import optax
from flax.training import train_state

def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones([1, 784]))
    tx = optax.adamw(learning_rate, weight_decay=0.01)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn(params, batch['image'], training=True, rngs={'dropout': rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label'])
        return loss.mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn(state.params, batch['image'], training=False)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])
    return accuracy

# 训练
rng = jax.random.PRNGKey(0)
state = create_train_state(rng, model, learning_rate=1e-3)

for epoch in range(num_epochs):
    for batch in train_loader:
        rng, step_rng = jax.random.split(rng)
        state, loss = train_step(state, batch, step_rng)
    
    # 评估
    accuracies = [eval_step(state, batch) for batch in val_loader]
    print(f"Epoch {epoch+1}, Val Accuracy: {np.mean(accuracies):.4f}")
```

### JAX高级特性

```python
# 自定义VJP (反向模式自动微分)
from jax import custom_vjp

@custom_vjp
def clip_gradient(x):
    return x

def clip_gradient_fwd(x):
    return clip_gradient(x), None

def clip_gradient_bwd(_, g):
    return (jnp.clip(g, -1.0, 1.0),)

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

# Scan (循环的函数式表达)
from jax.lax import scan

def rnn_cell(carry, x):
    h = carry
    new_h = jnp.tanh(jnp.dot(x, W_x) + jnp.dot(h, W_h))
    return new_h, new_h

final_h, all_h = scan(rnn_cell, initial_h, inputs)

# Checkpoint (节省显存)
from jax.checkpoint import checkpoint

@checkpoint
def expensive_layer(x):
    # 重新计算而不是存储中间结果
    return heavy_computation(x)
```

---

## 框架实战对比

### 相同模型的三种实现

**任务**: MNIST分类

#### PyTorch版本

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 训练
model = Net().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

#### TensorFlow版本

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 模型
model = keras.Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# 数据
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 训练
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=5)
```

#### JAX/Flax版本

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from tensorflow.keras.datasets import mnist

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# 数据
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 训练
model = CNN()
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones([1, 28, 28, 1]))
tx = optax.adam(1e-3)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

# 训练循环
batch_size = 64
for epoch in range(5):
    for i in range(0, len(x_train), batch_size):
        x_batch = jnp.array(x_train[i:i+batch_size])
        y_batch = jnp.array(y_train[i:i+batch_size])
        state, loss = train_step(state, x_batch, y_batch)
```

---

## 模型保存与加载

### PyTorch

```python
# 保存完整模型
torch.save(model, 'model.pt')
model = torch.load('model.pt')

# 只保存参数 (推荐)
torch.save(model.state_dict(), 'model_weights.pt')
model.load_state_dict(torch.load('model_weights.pt'))

# 保存检查点
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# TorchScript导出
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# ONNX导出
torch.onnx.export(model, dummy_input, 'model.onnx', 
                  input_names=['input'], output_names=['output'])
```

### TensorFlow

```python
# SavedModel格式 (推荐)
model.save('saved_model')
model = keras.models.load_model('saved_model')

# HDF5格式
model.save('model.h5')
model = keras.models.load_model('model.h5')

# 只保存权重
model.save_weights('weights.h5')
model.load_weights('weights.h5')

# TFLite转换
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### JAX/Flax

```python
from flax.training import checkpoints

# 保存检查点
checkpoints.save_checkpoint(
    ckpt_dir='checkpoints',
    target=state,
    step=epoch
)

# 加载检查点
state = checkpoints.restore_checkpoint(
    ckpt_dir='checkpoints',
    target=state
)

# 使用orbax (新标准)
import orbax.checkpoint as ocp

checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
checkpointer.save('checkpoint', state)
state = checkpointer.restore('checkpoint')
```

---

## 调试与性能分析

### PyTorch调试

```python
# 检测异常
torch.autograd.set_detect_anomaly(True)

# 梯度检查
torch.autograd.gradcheck(func, inputs, eps=1e-6, atol=1e-4, rtol=1e-3)

# 性能分析
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        train_step(data, target)
        prof.step()
```

### TensorFlow调试

```python
# Eager模式调试
tf.config.run_functions_eagerly(True)

# 检查数值稳定性
tf.debugging.enable_check_numerics()

# 性能分析
tf.profiler.experimental.start('logdir')
# 运行代码...
tf.profiler.experimental.stop()
```

### JAX调试

```python
# 禁用JIT以便调试
with jax.disable_jit():
    result = my_function(x)

# 检查中间值
from jax import debug
def my_fn(x):
    y = x * 2
    debug.print("y = {}", y)
    return y + 1

# 检查形状
jax.debug.print("Shape: {}", x.shape)
```

---

## 生态系统

### PyTorch生态

| 库 | 用途 |
|-----|------|
| torchvision | 计算机视觉 |
| torchaudio | 音频处理 |
| torchtext | 自然语言处理 |
| PyTorch Lightning | 训练框架 |
| Hugging Face | 预训练模型 |
| timm | 图像模型库 |

### TensorFlow生态

| 库 | 用途 |
|-----|------|
| TensorFlow Hub | 预训练模型 |
| TFX | MLOps管道 |
| TF Lite | 移动端部署 |
| TF.js | 浏览器运行 |
| TF Serving | 生产部署 |

### JAX生态

| 库 | 用途 |
|-----|------|
| Flax | 神经网络 |
| Haiku | 神经网络 (DeepMind) |
| Optax | 优化器 |
| Orbax | 检查点 |
| Equinox | 函数式神经网络 |

---

## 选择建议

| 场景 | 推荐框架 |
|------|---------|
| 学术研究 | PyTorch |
| 快速原型 | PyTorch / Keras |
| 生产部署 | TensorFlow |
| 移动端 | TensorFlow Lite |
| 大规模训练 | JAX (TPU) |
| NLP/LLM | PyTorch + Hugging Face |
| 函数式编程 | JAX |

---

## 小结

| 框架 | 优势 | 劣势 |
|------|------|------|
| PyTorch | 易用、灵活、研究友好 | 部署相对复杂 |
| TensorFlow | 部署成熟、生态完整 | API变化大 |
| JAX | 性能优秀、函数式 | 学习曲线陡峭 |

**下一篇**：深度学习模型部署与工程化，包括ONNX、TensorRT、模型压缩与MLOps。
