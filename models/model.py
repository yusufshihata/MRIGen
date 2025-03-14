import torch
import torch.nn as nn
import torch.optim as optim
from models.modules.generator import Generator
from models.modules.discriminator import Discriminator
from typing import Optional

class DViTGAN:
    def __init__(self, device: Optional[str] = "cuda"):
        super(DViTGAN, self).__init__()
        self.gen = Generator()
        self.disc = Discriminator()
        self.device = device
        
        self._update_model_optimizers()

    def generator_step(self, data):
        self.gen.train()
        self.disc.eval()

        self.Goptim.zero_grad()

        noise = self._generate_noise(data)

        fake_imgs = self.gen(noise)
        fake_logits = self.disc(fake_imgs)

        loss = - fake_logits.mean().view(-1)

        loss.backward()
        self.Goptim.step()

    def discriminator_step(self, data):
        self.gen.eval()
        self.disc.train()

        self.Doptim.zero_grad()

        noise = self._generate_noise(data)

        fake_imgs = self.gen(noise)
        
        fake_logits = self.disc(fake_imgs)
        real_logits = self.disc(data)

        loss = fake_logits.mean() - real_logits.mean()

        loss += self._compute_gp(data, fake_imgs)

        loss.backward()
        self.Doptim.step()

    def _generate_noise(self, data):
        return torch.randn(data.shape)

    def _compute_gp(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        alpha = alpha.expand_as(real_data)

        interpolation = alpha * real_data + (1 - alpha) * fake_data

        interpolated_logits = self.disc(interpolation)
        grad_outputs = torch.ones_like(interpolated_logits)

        gradients = torch.autograd.grad(
            outputs = interpolated_logits,
            inputs = interpolation,
            grad_outputs = grad_outputs,
            create_graph = True,
            retain_graph = True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

    def _update_model_optimizers(self):
        self.gen = self.gen.to(self.device)
        self.disc = self.disc.to(self.device)
        
        self.Goptim = optim.Adam(self.gen.parameters(), lr=1e-4)
        self.Doptim = optim.Adam(self.disc.parameters(), lr=5e-4)

