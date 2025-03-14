import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vit import PatchEmbedding

class ViTTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.patch_res = 16
        self.img_shape = (1, 128, 128)
        self.patch_size = ((self.img_shape[1] * self.img_shape[2]) // self.patch_res**2)
        self.latent_size = 512
        self.img = torch.randn(self.batch_size, *self.img_shape)

    def test_patch_embedding(self):
        patch_embedding = PatchEmbedding(self.patch_res, self.img_shape, self.latent_size)
        output = patch_embedding(self.img)
        self.assertEqual(output.shape, torch.Size([self.batch_size, self.patch_size, self.latent_size]))

if __name__ == "__main__":
    unittest.main()

