import torch
from torch import optim, distributions, nn
from gan import Discriminator, Generator
from datasets.data_utils import ListDatasets
from torch.utils.data import DataLoader
import numpy as np


def get_normal_sampler(mu, sigma, device):
    return lambda batch_size, feature_size: torch.Tensor(
        np.random.normal(mu, sigma, (batch_size, feature_size)), device=device)  # Gaussian


def get_uniform_sampler(device):
    return lambda batch_size, feature_size: torch.rand(batch_size, feature_size, device=device)


class GAN_Vanilla(object):
    def __init__(self, batch_size=16, D_lr=1e-3, G_lr=1e-3,
                 r_dim=3072, z_dim=100, h_size=512, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.batch_size = batch_size
        self.z_dim = z_dim

        self.D = Discriminator(r_dim, h_size).to(device=self.device)
        self.G = Generator(z_dim, r_dim, h_size).to(device=self.device)

        self.criterion = nn.BCELoss(size_average=True)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=D_lr)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=G_lr)

        root = '/home/lhq/PycharmProjects/gan.pytorch/datasets/data/test/'
        text = '/home/lhq/PycharmProjects/gan.pytorch/datasets/data/labels.txt'

        dataset = ListDatasets(root=root, fname_list=text)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.label_real = torch.ones(batch_size, 1, device=self.device)
        self.label_fake = torch.zeros(batch_size, 1, device=self.device)

    def train(self, epoch, step_D, step_G, print_every):
        for ep in range(epoch):
            d_loss, g_loss = 0, 0
            for i, data in enumerate(self.dataloader):
                data = data.view(self.batch_size, -1)
                if i%step_D == 0:
                    real_output = self.D(data)
                    d_loss_1 = self.criterion(real_output, self.label_real)

                    z = get_normal_sampler(0, 0.1, self.device)(self.batch_size, self.z_dim)
                    fake_image = self.G(z).detach()

                    fake_output = self.D(fake_image)
                    d_loss_2 = self.criterion(fake_output, self.label_fake)
                    d_loss = d_loss_1 + d_loss_2

                    self.D_optimizer.zero_grad()
                    d_loss.backward()
                    self.D_optimizer.step()
                if i%step_G == 0:
                    z = get_normal_sampler(0, 0.1, self.device)(self.batch_size, self.z_dim)
                    fake_image = self.G(z)

                    fake_output = self.D(fake_image)
                    g_loss = self.criterion(fake_output, self.label_real)

                    self.G_optimizer.zero_grad()
                    g_loss.backward()
                    self.G_optimizer.step()
                print('{} epoch: D_loss: {}, G_loss: {}'.format(ep, d_loss[0].item(), g_loss[0].item()))
            if ep % print_every == 0:
                print('{} epoch: D_loss: {}, G_loss: {}'.format(ep, d_loss[0].item(), g_loss[0].item()))


if __name__ == '__main__':
    gan = GAN_Vanilla()
    gan.train(100, 1, 5, 1)
