---
title: '深度学习完全指南（七）：生成模型GAN/VAE/Diffusion'
description: '深入理解三大生成模型范式：GAN的对抗训练、VAE的变分推断、Diffusion的去噪扩散'
pubDate: '2024-02-07'
heroImage: ''
category: '技术'
tags: ['深度学习', 'GAN', 'VAE', 'Diffusion', '生成模型']
series: '深度学习完全指南'
seriesOrder: 7
---

## 什么是生成模型？

生成模型学习数据的分布 $p(x)$，然后从中**采样生成新数据**。

### 判别模型 vs 生成模型

| 类型 | 学习目标 | 应用 |
|------|---------|------|
| 判别模型 | $p(y|x)$ | 分类、回归 |
| 生成模型 | $p(x)$ 或 $p(x|z)$ | 图像生成、数据增强 |

### 生成模型的三大范式

1. **GAN**：通过对抗博弈隐式学习分布
2. **VAE**：通过变分推断显式建模潜在空间
3. **Diffusion**：通过逐步去噪生成数据

---

## 第一部分：GAN（生成对抗网络）

### 核心思想

GAN由两个网络组成：
- **生成器 G**：从噪声生成假数据
- **判别器 D**：区分真假数据

两者进行**零和博弈**：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

```
噪声 z → [Generator] → 假图像 → [Discriminator] → 真/假
                              ↑
真实图像 ─────────────────────→
```

### 基础GAN实现

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)
```

### 训练循环

```python
# 初始化
generator = Generator(latent_dim=100, img_shape=(1, 28, 28))
discriminator = Discriminator(img_shape=(1, 28, 28))
adversarial_loss = nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        
        # 真假标签
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)
        
        # ---------------------
        # 训练判别器
        # ---------------------
        optimizer_D.zero_grad()
        
        # 真实图像的损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        
        # 生成假图像
        z = torch.randn(batch_size, latent_dim)
        gen_imgs = generator(z)
        
        # 假图像的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # ---------------------
        # 训练生成器
        # ---------------------
        optimizer_G.zero_grad()
        
        # 生成器希望判别器认为假图是真的
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        g_loss.backward()
        optimizer_G.step()
```

### GAN的变体

#### DCGAN（深度卷积GAN）

使用卷积/反卷积替代全连接：

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, channels=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # channels x 64 x 64
        )
```

#### WGAN（Wasserstein GAN）

使用Wasserstein距离替代JS散度，更稳定：

```python
# WGAN的判别器（称为critic）没有sigmoid
class Critic(nn.Module):
    def __init__(self):
        # ...
        # 最后一层不用sigmoid
        self.fc = nn.Linear(256, 1)

# 损失函数
def wgan_loss(real_validity, fake_validity):
    return -torch.mean(real_validity) + torch.mean(fake_validity)

# 梯度惩罚 (WGAN-GP)
def gradient_penalty(critic, real_imgs, fake_imgs, device):
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolates.requires_grad_(True)
    
    d_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

#### StyleGAN

高质量人脸生成，引入样式控制：

```python
# StyleGAN的核心：样式注入
class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.style_scale = nn.Linear(style_dim, out_channels)
        self.style_bias = nn.Linear(style_dim, out_channels)
    
    def forward(self, x, style):
        x = self.conv(x)
        scale = self.style_scale(style).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(style).unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias
```

#### 条件GAN（cGAN）

根据条件（如类别标签）生成：

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        # 将标签嵌入与噪声拼接
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            # ...
        )
    
    def forward(self, z, labels):
        label_embedding = self.label_embed(labels)
        gen_input = torch.cat([z, label_embedding], dim=-1)
        return self.model(gen_input)
```

### GAN的问题与技巧

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| 模式崩塌 | 只生成少数样本 | MinibatchDiscrimination, WGAN |
| 训练不稳定 | G和D难以平衡 | 谱归一化, 梯度惩罚 |
| 评估困难 | 无明确指标 | FID, IS |

---

## 第二部分：VAE（变分自编码器）

### 核心思想

VAE通过**变分推断**学习数据的潜在表示：
- **编码器**：将数据映射到潜在分布 $q(z|x)$
- **解码器**：从潜在变量重构数据 $p(x|z)$

$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

这个下界称为**ELBO（Evidence Lower Bound）**。

### VAE架构

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧：使采样可微"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
```

### VAE损失函数

```python
def vae_loss(recon_x, x, mu, log_var):
    # 重构损失（BCE或MSE）
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度（解析解）
    # D_KL(N(mu, sigma) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss
```

### 训练循环

```python
vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(epochs):
    for x, _ in dataloader:
        x = x.view(-1, 784)
        
        recon_x, mu, log_var = vae(x)
        loss = vae_loss(recon_x, x, mu, log_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 生成新样本
with torch.no_grad():
    z = torch.randn(64, 20)
    samples = vae.decode(z)
```

### VAE变体

#### β-VAE

增强解耦表示：

```python
def beta_vae_loss(recon_x, x, mu, log_var, beta=4.0):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss
```

#### VQ-VAE

使用离散潜在空间：

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
    
    def forward(self, z):
        # 找最近的嵌入向量
        distances = torch.cdist(z.flatten(2).transpose(1, 2), self.embedding.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.embedding(indices).transpose(1, 2).view_as(z)
        
        # 直通梯度
        z_q = z + (z_q - z).detach()
        
        return z_q, indices
```

### GAN vs VAE

| 特性 | GAN | VAE |
|------|-----|-----|
| 训练方式 | 对抗博弈 | 最大化ELBO |
| 生成质量 | 更清晰 | 可能模糊 |
| 训练稳定性 | 不稳定 | 稳定 |
| 潜在空间 | 无明确结构 | 连续、有结构 |
| 模式覆盖 | 可能模式崩塌 | 覆盖更完整 |

---

## 第三部分：Diffusion Model（扩散模型）

### 核心思想

Diffusion Model通过两个过程生成数据：
1. **前向扩散**：逐步向数据添加噪声，直到变成纯噪声
2. **反向去噪**：学习从噪声逐步恢复原始数据

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

### 数学框架

#### 前向过程

给定噪声调度 $\beta_1, ..., \beta_T$：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

#### 反向过程

学习预测噪声 $\epsilon_\theta(x_t, t)$：

$$
L = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

### 简化实现

```python
class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model  # U-Net或Transformer
        self.timesteps = timesteps
        
        # 噪声调度
        self.betas = self.linear_beta_schedule(timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def q_sample(self, x_0, t, noise=None):
        """前向扩散：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, t, noise=None):
        """计算训练损失"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        
        return F.mse_loss(noise, predicted_noise)
```

### U-Net架构

Diffusion常用U-Net作为去噪网络：

```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 下采样
        self.down1 = DownBlock(in_channels, 64, time_dim)
        self.down2 = DownBlock(64, 128, time_dim)
        self.down3 = DownBlock(128, 256, time_dim)
        
        # 中间层
        self.mid = MidBlock(256, time_dim)
        
        # 上采样
        self.up1 = UpBlock(256 + 256, 128, time_dim)
        self.up2 = UpBlock(128 + 128, 64, time_dim)
        self.up3 = UpBlock(64 + 64, 64, time_dim)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # 下采样路径
        d1 = self.down1(x, t_emb)
        d2 = self.down2(d1, t_emb)
        d3 = self.down3(d2, t_emb)
        
        # 中间
        m = self.mid(d3, t_emb)
        
        # 上采样路径（带skip连接）
        u1 = self.up1(torch.cat([m, d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([u1, d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([u2, d1], dim=1), t_emb)
        
        return self.out(u3)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings
```

### DDPM采样

```python
@torch.no_grad()
def p_sample(model, x_t, t, t_index):
    """单步去噪"""
    betas_t = model.betas[t][:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = model.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    sqrt_recip_alphas_t = torch.sqrt(1.0 / model.alphas[t])[:, None, None, None]
    
    # 预测噪声
    predicted_noise = model.model(x_t, t)
    
    # 计算均值
    model_mean = sqrt_recip_alphas_t * (
        x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        posterior_variance_t = betas_t
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, shape):
    """完整采样过程"""
    device = next(model.parameters()).device
    
    # 从纯噪声开始
    x = torch.randn(shape, device=device)
    
    # 逐步去噪
    for t in reversed(range(model.timesteps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, t)
    
    return x
```

### DDIM加速采样

DDIM允许跳过步骤，加速采样：

```python
@torch.no_grad()
def ddim_sample(model, shape, ddim_steps=50, eta=0.0):
    """DDIM采样（更快）"""
    device = next(model.parameters()).device
    
    # 选择子集时间步
    times = torch.linspace(0, model.timesteps - 1, ddim_steps, dtype=torch.long, device=device)
    times = times.flip(0)
    
    x = torch.randn(shape, device=device)
    
    for i, t in enumerate(times[:-1]):
        t_next = times[i + 1]
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # 预测噪声
        predicted_noise = model.model(x, t_batch)
        
        # 计算x_0预测
        alpha_t = model.alphas_cumprod[t]
        alpha_t_next = model.alphas_cumprod[t_next]
        
        x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        # 计算方向
        dir_xt = torch.sqrt(1 - alpha_t_next) * predicted_noise
        
        x = torch.sqrt(alpha_t_next) * x0_pred + dir_xt
    
    return x
```

### 条件生成：Classifier-Free Guidance

```python
def classifier_free_guidance_sample(model, shape, condition, guidance_scale=7.5):
    """无分类器引导"""
    # 同时预测条件和无条件噪声
    noise_cond = model(x_t, t, condition)
    noise_uncond = model(x_t, t, None)  # 或用特殊token
    
    # 引导
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    
    return noise_pred
```

### 现代Diffusion模型

| 模型 | 特点 | 应用 |
|------|------|------|
| DDPM | 原始框架 | 基础 |
| Stable Diffusion | 潜在空间扩散 | 文生图 |
| DALL-E 2/3 | CLIP引导 | 文生图 |
| Midjourney | 艺术风格 | 文生图 |
| Imagen | 大语言模型文本编码 | 文生图 |
| Sora | 视频生成 | 文生视频 |

### Stable Diffusion架构

```
文本 → CLIP Text Encoder → Cross-Attention
                               ↓
噪声 → U-Net (with attention) → 去噪潜在
                               ↓
                          VAE Decoder → 图像
```

---

## 三大范式对比

| 特性 | GAN | VAE | Diffusion |
|------|-----|-----|-----------|
| 训练目标 | 对抗损失 | ELBO | 去噪损失 |
| 训练稳定性 | 不稳定 | 稳定 | 非常稳定 |
| 生成质量 | 高 | 中等 | 最高 |
| 采样速度 | 快（单次前向） | 快 | 慢（多步迭代） |
| 模式覆盖 | 可能崩塌 | 好 | 非常好 |
| 似然估计 | 不可行 | 近似 | 可计算 |
| 当前主流 | StyleGAN系列 | VQ-VAE | Stable Diffusion |

---

## 总结

| 模型 | 核心思想 | 代表应用 |
|------|---------|---------|
| GAN | 对抗训练 | StyleGAN人脸生成 |
| VAE | 变分推断 | 图像压缩、表示学习 |
| Diffusion | 逐步去噪 | Stable Diffusion文生图 |

---

## 下一步

生成模型是AI创造力的基础。下一篇我们将学习**大语言模型（LLM）**，了解GPT、LLaMA等模型如何理解和生成语言。
