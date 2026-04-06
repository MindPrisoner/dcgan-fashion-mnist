# FashionMNIST GAN Project

这是一个在 FashionMNIST 数据集上进行生成式对抗网络实验的项目，覆盖了 DCGAN、WGAN 和 WGAN-GP 三种生成对抗训练思路。

当前代码主流程实际采用的是 WGAN-GP，用于提升训练稳定性并改善生成图像质量。

## 项目目标

- 在 FashionMNIST 上训练生成模型
- 对比 DCGAN、WGAN 和 WGAN-GP 的训练表现
- 观察 GAN 训练中的不稳定性、振荡和收敛特征
- 保存训练过程中的中间生成结果

## 目录结构

```text
dcgan_project/
├── models/
│   ├── generator.py        # 生成器
│   └── discriminator.py   # Critic / 判别器
├── outputs/               # 生成结果保存目录
├── utils/
│   └── save_images.py     # 图片保存工具
├── train.py               # 训练入口
├── requirements.txt       # 依赖列表
└── README.md
```

## 数据集

项目使用 `torchvision.datasets.FashionMNIST`，输入图像为灰度图，尺寸为 28x28。

预处理流程：

- `ToTensor()`
- `Normalize((0.5,), (0.5,))`

## 模型结构

### 生成器

`models/generator.py` 中的生成器使用反卷积逐步上采样，将 100 维噪声向量映射为单通道服饰图像。

### Critic

`models/discriminator.py` 中当前启用的是 `Critic`，配合 WGAN-GP 使用。

Critic 中使用了：

- 卷积层
- `LeakyReLU`
- `InstanceNorm2d`

当前实现没有使用 Sigmoid，而是输出 Wasserstein critic score。

## 训练流程

运行方式：

```bash
pip install -r requirements.txt
python train.py
```

训练脚本会：

1. 下载并加载 FashionMNIST
2. 构建生成器和 critic
3. 使用 WGAN-GP 进行对抗训练
4. 定期保存生成图像到 `outputs/`
5. 训练结束后保存模型权重到 `checkpoints/`

## 关键超参数

当前主流程中的设置如下：

- `latent_dim = 100`
- `batch_size = 128`
- `epochs = 30`
- `lr = 1e-4`
- `n_critic = 5`
- `lambda_gp = 10`

这些参数是当前实验中稳定性和生成质量之间的折中配置。

## 训练版本说明

这个项目保留了三种训练思路的注释实现：

- DCGAN
- WGAN
- WGAN-GP

最终采用 WGAN-GP 作为主实验方案，原因很直接：

- 训练更稳定
- critic 的约束更平滑
- 生成图像的视觉质量更好

## 实验观察

从实验过程来看，GAN 的典型特征非常明显：

- 训练过程会出现振荡
- 图像质量不会单调提升
- 不同版本的损失函数会明显改变训练稳定性

具体结论如下：

- DCGAN 的对抗训练波动较大，生成结果更依赖训练阶段
- WGAN 的训练动态更稳定
- WGAN-GP 在视觉效果和稳定性上最均衡

## 输出结果

训练过程中会在 `outputs/` 目录中保存阶段性生成图像，例如：

- `epoch_002.png`
- `epoch_010.png`
- `epoch_020.png`
- `epoch_030.png`

这些图像用于观察模型在不同训练阶段的生成质量变化。

## 注意事项

- 训练时需要联网下载 FashionMNIST 数据集
- 当前代码使用 `cuda` 时会自动切换到 GPU，否则回退到 CPU
- 生成器和 critic 的权重会在训练结束后保存到 `checkpoints/`
- 项目中的部分 DCGAN 和 WGAN 代码以注释形式保留，作为实验对比记录

