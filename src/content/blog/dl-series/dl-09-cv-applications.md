---
title: '深度学习完全指南（九）：计算机视觉应用'
description: '从图像分类到目标检测、语义分割、人脸识别，全面掌握计算机视觉的核心技术与实战应用'
pubDate: '2024-02-09'
heroImage: ''
category: '技术'
tags: ['深度学习', '计算机视觉', 'CV', '目标检测']
series: '深度学习完全指南'
seriesOrder: 9
---

## 计算机视觉概述

计算机视觉（Computer Vision, CV）是深度学习最成功的应用领域之一。从2012年AlexNet在ImageNet上的突破开始，深度学习彻底改变了CV的发展轨迹。

### CV任务分类

| 任务类型 | 输入 | 输出 | 典型应用 |
|---------|------|------|---------|
| 图像分类 | 图像 | 类别标签 | 产品识别、医学诊断 |
| 目标检测 | 图像 | 边界框+类别 | 自动驾驶、安防监控 |
| 语义分割 | 图像 | 像素级标签 | 医学图像、自动驾驶 |
| 实例分割 | 图像 | 实例级掩码 | 机器人视觉 |
| 姿态估计 | 图像 | 关键点坐标 | 动作捕捉、健身AI |
| 人脸识别 | 图像 | 身份ID | 门禁、支付 |
| OCR | 图像 | 文字 | 文档数字化 |

---

## 图像分类进阶

### 经典架构演进

```
LeNet (1998) → AlexNet (2012) → VGG (2014) → GoogLeNet (2014)
     → ResNet (2015) → DenseNet (2017) → EfficientNet (2019)
          → ViT (2020) → Swin Transformer (2021) → ConvNeXt (2022)
```

### 现代分类网络：EfficientNet

EfficientNet通过复合缩放（Compound Scaling）平衡网络的深度、宽度和分辨率：

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# 加载预训练EfficientNet
model = models.efficientnet_b0(weights='IMAGENET1K_V1')

# 修改分类头用于自定义任务
num_classes = 10
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Vision Transformer (ViT)

ViT将Transformer引入图像分类，开创了CV的新范式：

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """将图像分割为patches并嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 可学习的cls token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 添加cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # 分类
        x = self.norm(x[:, 0])  # 取cls token
        x = self.head(x)
        
        return x

# 创建模型
model = ViT(num_classes=100)
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

---

## 目标检测

目标检测需要同时完成定位（Localization）和分类（Classification）。

### 检测范式演进

| 范式 | 代表模型 | 特点 |
|------|---------|------|
| 两阶段 | R-CNN系列 | 先提取候选框，再分类 |
| 单阶段 | YOLO, SSD | 端到端，速度快 |
| Anchor-free | FCOS, CenterNet | 无需预设锚框 |
| Transformer | DETR, DINO | 端到端，无NMS |

### YOLO系列

YOLO（You Only Look Once）是最流行的实时检测框架：

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # nano版本，最快

# 推理
results = model('image.jpg')

# 显示结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        print(f"类别: {model.names[cls]}, 置信度: {conf:.2f}, 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

# 训练自定义数据集
model.train(data='custom_dataset.yaml', epochs=100, imgsz=640)

# 导出模型
model.export(format='onnx')
```

### DETR: 端到端目标检测

DETR使用Transformer实现端到端检测，无需NMS后处理：

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_queries=100):
        super().__init__()
        
        # CNN backbone
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=8, 
            num_encoder_layers=6, num_decoder_layers=6
        )
        
        # 可学习的查询
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # 位置编码
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
        # 预测头
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no object
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # cx, cy, w, h
        )
    
    def forward(self, x):
        # Backbone特征
        features = self.backbone(x)  # (B, 2048, H/32, W/32)
        h = self.conv(features)  # (B, hidden_dim, H, W)
        
        B, C, H, W = h.shape
        
        # 位置编码
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # (H*W, 1, C)
        
        # Transformer
        src = h.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, C)
        
        hs = self.transformer(src + pos, query)  # (num_queries, B, C)
        hs = hs.permute(1, 0, 2)  # (B, num_queries, C)
        
        # 预测
        class_logits = self.class_head(hs)  # (B, num_queries, num_classes+1)
        bbox_pred = self.bbox_head(hs).sigmoid()  # (B, num_queries, 4)
        
        return class_logits, bbox_pred
```

### 匈牙利匹配算法

DETR使用匈牙利算法进行预测与GT的最优匹配：

```python
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def hungarian_matching(pred_logits, pred_boxes, gt_labels, gt_boxes):
    """
    计算预测与GT的最优匹配
    """
    num_queries = pred_logits.shape[0]
    num_gts = len(gt_labels)
    
    # 分类代价
    pred_probs = F.softmax(pred_logits, dim=-1)
    class_cost = -pred_probs[:, gt_labels]  # (num_queries, num_gts)
    
    # 边界框L1代价
    bbox_cost = torch.cdist(pred_boxes, gt_boxes, p=1)  # (num_queries, num_gts)
    
    # GIoU代价
    giou_cost = -generalized_box_iou(pred_boxes, gt_boxes)  # (num_queries, num_gts)
    
    # 总代价
    cost_matrix = class_cost + 5 * bbox_cost + 2 * giou_cost
    
    # 匈牙利算法求解
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
    
    return pred_indices, gt_indices
```

---

## 语义分割

语义分割为每个像素分配类别标签。

### 主流架构

| 模型 | 年份 | 特点 |
|------|------|------|
| FCN | 2015 | 首个端到端分割网络 |
| U-Net | 2015 | 跳跃连接，医学图像 |
| DeepLab | 2017 | 空洞卷积，ASPP |
| PSPNet | 2017 | 金字塔池化 |
| Mask2Former | 2022 | 统一分割 |

### U-Net实现

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        
        # Encoder (下采样)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (上采样)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder + Skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out(d1)

# 创建模型
model = UNet(num_classes=21)
x = torch.randn(2, 3, 256, 256)
out = model(x)
print(f"输出形状: {out.shape}")  # (2, 21, 256, 256)
```

### DeepLab v3+ with ASPP

```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # 1x1卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积，不同膨胀率
        rates = [6, 12, 18]
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        
        # 全局平均池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # 多尺度特征
        feat1 = self.conv1(x)
        feat2 = self.atrous_convs[0](x)
        feat3 = self.atrous_convs[1](x)
        feat4 = self.atrous_convs[2](x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        
        # 拼接融合
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)
```

### 分割损失函数

```python
class DiceLoss(nn.Module):
    """Dice Loss，适合类别不平衡"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        
        # One-hot编码
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        # 计算Dice
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    """Focal Loss，关注难样本"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 组合损失
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)
```

---

## 实例分割

实例分割 = 目标检测 + 语义分割，需要区分同类别的不同实例。

### Mask R-CNN

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# 加载预训练模型
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 推理
image = torch.randn(3, 800, 800)
with torch.no_grad():
    predictions = model([image])

# 解析结果
boxes = predictions[0]['boxes']      # 边界框
labels = predictions[0]['labels']    # 类别
scores = predictions[0]['scores']    # 置信度
masks = predictions[0]['masks']      # 实例掩码 (N, 1, H, W)

# 自定义训练
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # 修改分类头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # 修改mask头
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    return model
```

---

## 人脸识别

人脸识别系统包括：人脸检测 → 人脸对齐 → 特征提取 → 特征比对。

### ArcFace损失

ArcFace是目前最流行的人脸识别损失函数：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # 缩放因子
        self.m = m  # 角度margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, features, labels):
        # 归一化
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 处理边界情况
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # 只对正确类别添加margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

# 使用示例
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        # Backbone
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, embedding_dim)
        
        # ArcFace head
        self.arcface = ArcFace(embedding_dim, num_classes)
    
    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = F.normalize(features)
        
        if labels is not None:
            return self.arcface(features, labels)
        return features
```

### 人脸检测：RetinaFace

```python
from retinaface import RetinaFace

# 检测人脸
faces = RetinaFace.detect_faces("group_photo.jpg")

for face_id, face_info in faces.items():
    facial_area = face_info["facial_area"]  # [x1, y1, x2, y2]
    landmarks = face_info["landmarks"]       # 5个关键点
    confidence = face_info["score"]
    
    print(f"人脸 {face_id}: 置信度={confidence:.2f}")
    print(f"  位置: {facial_area}")
    print(f"  左眼: {landmarks['left_eye']}")
    print(f"  右眼: {landmarks['right_eye']}")
```

### 完整人脸识别流程

```python
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = MTCNN(keep_all=True)
        self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
        self.database = {}  # {name: embedding}
    
    def register(self, name, image):
        """注册人脸"""
        faces = self.detector(image)
        if faces is None:
            return False
        
        embedding = self.encoder(faces[0].unsqueeze(0))
        self.database[name] = embedding.detach().numpy()
        return True
    
    def recognize(self, image, threshold=0.6):
        """识别人脸"""
        faces = self.detector(image)
        if faces is None:
            return []
        
        results = []
        for face in faces:
            embedding = self.encoder(face.unsqueeze(0)).detach().numpy()
            
            # 与数据库比对
            best_match = None
            best_similarity = 0
            
            for name, db_embedding in self.database.items():
                similarity = np.dot(embedding.flatten(), db_embedding.flatten())
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            
            if best_similarity > threshold:
                results.append((best_match, best_similarity))
            else:
                results.append(("Unknown", best_similarity))
        
        return results
```

---

## 姿态估计

### OpenPose关键点

人体姿态估计检测人体关键点（关节位置）：

```python
import torch
import torchvision.models as models

# 使用torchvision的KeypointRCNN
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = torch.randn(3, 800, 600)

with torch.no_grad():
    predictions = model([image])

# 解析关键点
# COCO格式：17个关键点
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

keypoints = predictions[0]['keypoints']  # (N, 17, 3) - x, y, visibility
scores = predictions[0]['keypoints_scores']
```

### 轻量级姿态估计：MoveNet

```python
import tensorflow as tf
import tensorflow_hub as hub

# 加载MoveNet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

def detect_pose(image):
    """检测单人姿态"""
    # 预处理
    input_image = tf.cast(image, dtype=tf.int32)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)
    input_image = tf.expand_dims(input_image, axis=0)
    
    # 推理
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    
    return keypoints  # (17, 3) - y, x, confidence
```

---

## OCR文字识别

### 基于深度学习的OCR流程

```
文本检测 → 文本方向矫正 → 文本识别 → 后处理
```

### 使用PaddleOCR

```python
from paddleocr import PaddleOCR

# 初始化
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 识别
result = ocr.ocr('document.jpg', cls=True)

for line in result[0]:
    box = line[0]      # 文本框坐标
    text = line[1][0]  # 识别文字
    conf = line[1][1]  # 置信度
    print(f"文字: {text}, 置信度: {conf:.2f}")
```

### CRNN文字识别模型

```python
class CRNN(nn.Module):
    """CNN + RNN + CTC"""
    def __init__(self, img_height, num_chars, hidden_size=256):
        super().__init__()
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.ReLU()
        )
        
        # 计算CNN输出特征维度
        self.feature_height = img_height // 16 - 1
        
        # 双向LSTM
        self.rnn = nn.LSTM(512 * self.feature_height, hidden_size, 
                          num_layers=2, bidirectional=True, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)  # +1 for blank
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # (B, 512, H', W')
        
        # 转换为序列
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)  # (B, W', C*H')
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # (B, W', hidden*2)
        
        # 输出
        output = self.fc(rnn_out)  # (B, W', num_chars+1)
        
        return output.permute(1, 0, 2)  # CTC要求 (T, B, C)

# CTC损失
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
```

---

## 数据增强

数据增强是CV训练的关键技术：

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 训练时增强
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    A.GaussNoise(p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 检测任务增强
detection_transform = A.Compose([
    A.RandomResizedCrop(640, 640, scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# MixUp增强
def mixup(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    
    return mixed_images, labels_a, labels_b, lam

# CutMix增强
def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = images.size()
    index = torch.randperm(batch_size)
    
    # 计算裁剪区域
    cut_ratio = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    return images, labels, labels[index], lam
```

---

## 评估指标

### 分类指标

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Top-K准确率
def topk_accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k / batch_size)
    
    return results
```

### 检测指标：mAP

```python
def calculate_ap(recalls, precisions):
    """计算单类别AP"""
    # 11点插值法
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    return ap

def calculate_iou(box1, box2):
    """计算IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)
```

### 分割指标：mIoU

```python
def calculate_miou(pred, target, num_classes):
    """计算mIoU"""
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.mean(ious)
```

---

## 实战项目：车牌识别系统

```python
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO

class LicensePlateRecognizer:
    def __init__(self):
        # 车牌检测模型
        self.detector = YOLO('license_plate_detector.pt')
        
        # 字符识别模型
        self.chars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
        self.recognizer = CRNN(32, len(self.chars))
        self.recognizer.load_state_dict(torch.load('crnn_license.pt'))
        self.recognizer.eval()
    
    def detect(self, image):
        """检测车牌"""
        results = self.detector(image)
        plates = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_img = image[y1:y2, x1:x2]
            plates.append({
                'bbox': (x1, y1, x2, y2),
                'image': plate_img,
                'confidence': box.conf[0].item()
            })
        
        return plates
    
    def recognize(self, plate_img):
        """识别车牌号"""
        # 预处理
        img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 32))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            output = self.recognizer(img)
        
        # CTC解码
        _, preds = output.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        
        # 去重复和blank
        text = []
        prev = 0
        for p in preds:
            if p != 0 and p != prev:
                text.append(self.chars[p - 1])
            prev = p
        
        return ''.join(text)
    
    def process(self, image):
        """完整处理流程"""
        plates = self.detect(image)
        
        for plate in plates:
            plate['text'] = self.recognize(plate['image'])
        
        return plates

# 使用示例
recognizer = LicensePlateRecognizer()
image = cv2.imread('car.jpg')
results = recognizer.process(image)

for r in results:
    print(f"车牌: {r['text']}, 置信度: {r['confidence']:.2f}")
```

---

## 小结

本文涵盖了计算机视觉的核心应用：

| 任务 | 主流方法 | 推荐模型 |
|------|---------|---------|
| 图像分类 | CNN/ViT | EfficientNet, Swin |
| 目标检测 | 单阶段/Transformer | YOLOv8, DINO |
| 语义分割 | Encoder-Decoder | U-Net, DeepLab |
| 实例分割 | 两阶段 | Mask R-CNN |
| 人脸识别 | 度量学习 | ArcFace |
| 姿态估计 | 关键点检测 | MoveNet, HRNet |
| OCR | 检测+识别 | PaddleOCR |

**下一篇**：深度学习在自然语言处理中的应用，包括文本分类、命名实体识别、机器翻译等。
