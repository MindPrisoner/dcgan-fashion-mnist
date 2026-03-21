# DCGAN
# import torch.nn as nn
#
#
# class Discriminator(nn.Module):
#     def __init__(self, img_channels=1, feature_d=64):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(feature_d * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(feature_d * 2, feature_d * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(feature_d * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x).view(-1)
# WGAN
# import torch.nn as nn
#
#
# class Critic(nn.Module):
#     def __init__(self, img_channels=1, feature_d=64):
#         super().__init__()
#
#         self.net = nn.Sequential(
#             nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(feature_d * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(feature_d * 2, feature_d * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(feature_d * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False)
#         )
#
#     def forward(self, x):
#         return self.net(x).view(-1)


import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, img_channels=1, feature_d=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_d * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(feature_d * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.net(x).view(-1)