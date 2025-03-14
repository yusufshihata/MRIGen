import torch
from models.model import DViTGAN
from tqdm import tqdm

import torch
import torch.optim as optim
from tqdm import tqdm

def train_dvitgan(model, dataloader, num_epochs=100, device="cuda", save_every=10):
    model.to(device)
    
    for epoch in range(1, num_epochs + 1):
        g_loss_epoch = 0
        d_loss_epoch = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            
            # Train discriminator
            model.discriminator_step(real_imgs)
            d_loss = model.disc(real_imgs).mean().item()
            d_loss_epoch += d_loss

            # Train generator
            model.generator_step(real_imgs)
            g_loss = -model.disc(model.gen(model._generate_noise(real_imgs))).mean().item()
            g_loss_epoch += g_loss

            loop.set_postfix(G_Loss=g_loss, D_Loss=d_loss)

        # Compute average losses for the epoch
        g_loss_epoch /= len(dataloader)
        d_loss_epoch /= len(dataloader)

        print(f"Epoch [{epoch}/{num_epochs}] - G_Loss: {g_loss_epoch:.4f}, D_Loss: {d_loss_epoch:.4f}")

        # Save model checkpoint
        if epoch % save_every == 0:
            torch.save(model.gen.state_dict(), f"generator_epoch_{epoch}.pth")
            torch.save(model.disc.state_dict(), f"discriminator_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")


