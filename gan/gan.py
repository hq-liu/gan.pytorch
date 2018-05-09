from torch import nn
from torch.nn import functional as F
import torch


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, h_size):
        super(Generator, self).__init__()
        self.input_layer = nn.Linear(in_features=input_dim, out_features=h_size)
        self.hidden_layer = nn.Linear(in_features=h_size, out_features=h_size)
        self.output_layer = nn.Linear(in_features=h_size, out_features=output_dim)

    def forward(self, x):
        x = F.relu6(self.input_layer(x))
        x = F.relu6(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, h_size):
        super(Discriminator, self).__init__()
        self.input_layer = nn.Linear(in_features=input_dim, out_features=h_size)
        self.hidden_layer = nn.Linear(in_features=h_size, out_features=h_size)
        self.output_layer = nn.Linear(in_features=h_size, out_features=1)

    def forward(self, x):
        x = F.relu6(self.input_layer(x))
        x = F.relu6(self.hidden_layer(x))
        x = F.sigmoid(self.output_layer(x))
        return x


if __name__ == '__main__':
    a = torch.randn(1, 1024)
    z = torch.randn(1, 100)
    D = Discriminator(1024, 512)
    G = Generator(100, 1024, 512)
    a_ = D(a)
    z_ = G(z)
    print(a_, z_.size())
