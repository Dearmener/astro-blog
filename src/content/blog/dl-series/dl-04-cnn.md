---
title: '深度学习完全指南（四）：卷积神经网络CNN详解'
description: '从卷积操作到经典架构，全面掌握CNN在图像处理中的核心原理与实现'
pubDate: '2024-02-04'
heroImage: ''
category: '技术'
tags: ['深度学习', 'CNN', '计算机视觉']
series: '深度学习完全指南'
seriesOrder: 4
---

## 为什么需要CNN？

传统全连接网络处理图像存在严重问题：

1. **参数爆炸**：224×224×3的图像展平后有150,528个输入，一层就需要数千万参数
2. **忽略空间结构**：图像的像素位置关系被破坏
3. **无法平移不变**：同一物体在不同位置需要重新学习

CNN通过**局部连接**和**权重共享**解决了这些问题。

---

## 卷积操作

### 什么是卷积？

卷积是一种数学操作，用一个小的**卷积核（kernel/filter）**在输入上滑动，计算局部区域的加权和。

$$
(I * K)[i,j] = \sum_m \sum_n I[i+m, j+n] \cdot K[m, n]
$$

```python
import numpy as np

def conv2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    
    return output

# 边缘检测示例
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
```

### 卷积的关键概念

#### 步幅（Stride）

卷积核每次移动的像素数：

```python
# stride=1: 每次移动1像素
# stride=2: 每次移动2像素，输出尺寸减半
```

#### 填充（Padding）

在输入边缘填充值（通常为0）：

- **Valid padding**：不填充，输出变小
- **Same padding**：填充使输出尺寸与输入相同

```python
# PyTorch
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # same
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)  # valid
```

#### 输出尺寸计算

$$
O = \frac{I - K + 2P}{S} + 1
$$

其中 I=输入尺寸，K=卷积核大小，P=填充，S=步幅

### 多通道卷积

实际图像有多个通道（RGB），卷积核也是3D的：

```python
# 输入: (batch, in_channels, H, W)
# 卷积核: (out_channels, in_channels, kH, kW)
# 输出: (batch, out_channels, H', W')

conv = nn.Conv2d(
    in_channels=3,      # RGB
    out_channels=64,    # 64个滤波器
    kernel_size=3,
    stride=1,
    padding=1
)
```

---

## 池化层（Pooling）

池化用于**降低空间维度**，减少计算量并增强平移不变性。

### 最大池化（Max Pooling）

取局部区域的最大值：

```python
# 2x2最大池化，stride=2
nn.MaxPool2d(kernel_size=2, stride=2)

# 输入: 4x4 → 输出: 2x2
```

### 平均池化（Average Pooling）

取局部区域的平均值：

```python
nn.AvgPool2d(kernel_size=2, stride=2)
```

### 全局平均池化（GAP）

将整个特征图压缩为一个值，常用于分类网络的最后一层：

```python
nn.AdaptiveAvgPool2d(1)  # 输出: (batch, channels, 1, 1)
```

---

## CNN的基本架构

典型CNN结构：

```
输入 → [Conv → ReLU → Pool] × N → Flatten → FC → Softmax
```

### PyTorch实现

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 经典CNN架构演进

### LeNet-5 (1998)

手写数字识别的开山之作：

```
Input(32x32) → Conv(6@5x5) → Pool → Conv(16@5x5) → Pool → FC → FC → Output(10)
```

```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

### AlexNet (2012)

ImageNet竞赛的转折点，开启深度学习时代：

**创新点**：
- 使用ReLU激活函数
- Dropout正则化
- 数据增强
- GPU训练

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
```

### VGGNet (2014)

证明了**深度**的重要性，使用统一的3×3卷积核：

```
VGG-16架构：
[Conv3-64]×2 → Pool →
[Conv3-128]×2 → Pool →
[Conv3-256]×3 → Pool →
[Conv3-512]×3 → Pool →
[Conv3-512]×3 → Pool →
FC-4096 → FC-4096 → FC-1000
```

**设计哲学**：
- 小卷积核（3×3）堆叠可以增加感受野同时减少参数
- 两个3×3卷积 ≈ 一个5×5卷积的感受野

### GoogLeNet/Inception (2014)

引入**Inception模块**，同时使用多种尺度的卷积核：

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        
        # 1x1 卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, 1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 → 3x3 卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 → 5x5 卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 池化 → 1x1 卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
```

### ResNet (2015)

引入**残差连接**，解决了深层网络的退化问题：

$$
y = F(x) + x
$$

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out
```

**为什么残差连接有效？**
- 梯度可以直接回传，缓解梯度消失
- 学习残差比学习完整映射更容易
- 允许网络变得更深（152层甚至1000+层）

### DenseNet (2017)

每一层与之前所有层都连接：

$$
x_l = H_l([x_0, x_1, ..., x_{l-1}])
$$

```python
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                self._make_layer(in_channels + i * growth_rate, growth_rate)
            )
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)
```

### EfficientNet (2019)

系统性地研究了网络**深度、宽度、分辨率**的最优缩放：

$$
\text{depth}: d = \alpha^\phi, \quad
\text{width}: w = \beta^\phi, \quad
\text{resolution}: r = \gamma^\phi
$$

其中 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

```python
# 使用 timm 库加载
import timm

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

---

## 重要技术组件

### 批量归一化（Batch Normalization）

加速训练，允许更大的学习率：

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad
y = \gamma \hat{x} + \beta
$$

```python
nn.BatchNorm2d(num_features)
```

### 1×1卷积

用于**通道变换**和**降维**：

```python
# 降维：256通道 → 64通道
nn.Conv2d(256, 64, kernel_size=1)
```

### 深度可分离卷积（Depthwise Separable Conv）

大幅减少计算量，MobileNet的核心：

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise: 每个通道单独卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 
                                    padding=1, groups=in_channels)
        # Pointwise: 1x1卷积混合通道
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

**计算量对比**：
- 标准卷积：$D_K^2 \cdot M \cdot N \cdot D_F^2$
- 深度可分离：$D_K^2 \cdot M \cdot D_F^2 + M \cdot N \cdot D_F^2$
- 节省约 $\frac{1}{N} + \frac{1}{D_K^2}$ 的计算

---

## 现代CNN架构

### ConvNeXt (2022)

将Transformer的设计理念引入CNN：

```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + x
```

---

## 架构对比

| 模型 | 年份 | 层数 | Top-1准确率 | 参数量 | 特点 |
|------|------|------|-------------|--------|------|
| AlexNet | 2012 | 8 | 57.1% | 60M | 开创性工作 |
| VGG-16 | 2014 | 16 | 71.5% | 138M | 小卷积核 |
| GoogLeNet | 2014 | 22 | 74.8% | 6.8M | Inception模块 |
| ResNet-50 | 2015 | 50 | 76.1% | 25M | 残差连接 |
| DenseNet-121 | 2017 | 121 | 74.4% | 8M | 密集连接 |
| EfficientNet-B0 | 2019 | - | 77.1% | 5.3M | 复合缩放 |
| ConvNeXt-T | 2022 | - | 82.1% | 29M | 现代化设计 |

---

## 实战：图像分类

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# 使用预训练模型
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)  # 修改最后一层

# 训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## 总结

| 概念 | 要点 |
|------|------|
| 卷积操作 | 局部连接、权重共享、特征提取 |
| 池化 | 降维、增强平移不变性 |
| 经典架构 | VGG(深度)、Inception(多尺度)、ResNet(残差) |
| 关键技术 | BatchNorm、1×1卷积、深度可分离卷积 |
| 设计趋势 | 更深、更宽、自动化搜索 |

---

## 下一步

CNN革新了计算机视觉，但它难以处理**序列数据**。下一篇我们将学习**循环神经网络（RNN）**，了解如何处理文本、时间序列等顺序数据。
