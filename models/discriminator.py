import torch
import torch.nn as nn
from models.vit import ViT

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_shape = (1, 128, 128)
        self.patch_res = 16
        self.embed_dim = 1024
        self.feature_extractor = ViT(self.img_shape, self.patch_res, self.embed_dim, 8, 5)
        self.disc = nn.Linear(self.embed_dim, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(img)
        return self.disc(features)
