import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),

            ResidualBlock(64, 128),
            nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)),  # 16x16 -> 8x8
            nn.InstanceNorm2d(128),  # Replacing LayerNorm
            nn.LeakyReLU(0.2, inplace=True),

            ResidualBlock(128, 256),
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)),  # 8x8 -> 4x4
            nn.InstanceNorm2d(256),  # Replacing LayerNorm
            nn.LeakyReLU(0.2, inplace=True),

            SelfAttention(256),  # Adding self-attention

            ResidualBlock(256, 512),
            nn.utils.spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1)
        )

    def forward(self, img):
        return self.model(img)
