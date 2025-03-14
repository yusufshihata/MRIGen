import torch
from models.model import DViTGAN
from tqdm import tqdm

def train(model: DViTGAN, trainloader: torch.utils.data.DataLoader, epochs: int = 100):
    loop = tqdm(trainlaoder, unit="batch")
    for epoch in range(epochs):
        for imgs, _ in tdqm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True):
            imgs = imgs.to(model.device)

            model.generator_step(imgs)
            
            model.discriminator_step(imgs)

