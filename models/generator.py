import torch.nn as nn
from diffusers import UNet2DModel

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=128,  # Image size
            in_channels=1,  # Grayscale images
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t):
        return self.model(x, t).sample

