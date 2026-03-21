import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, feature_g=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_g * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.Conv2d(feature_g, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
