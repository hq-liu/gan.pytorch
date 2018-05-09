import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, z_dim, n):
        super(Decoder, self).__init__()
        self.n = n
        self.fc = nn.Linear(z_dim, 8*8*n)
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, padding=1, stride=1)
        )
        self.stage_2 = nn.Sequential(
            nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, padding=1, stride=1)
        )
        self.stage_3 = nn.Sequential(
            nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding=1, stride=1)
        )
        self.conv_last = nn.Conv2d(in_channels=n, out_channels=3, kernel_size=3, padding=1, stride=1)

    @staticmethod
    def up_sampling(input_tensor):
        feature_size = input_tensor.size(2)
        return nn.Upsample(size=(feature_size*2, feature_size*2), mode="bilinear")(input_tensor)

    def forward(self, h):
        h = self.fc(h)
        N = h.size(0)
        h = h.view(N, self.n, 8, 8)
        h = F.elu(self.stage_1(h))
        h = self.up_sampling(h)
        h = F.elu(self.stage_2(h))
        h = self.up_sampling(h)
        h = F.elu(self.stage_3(h))
        return self.conv_last(h)


class Encoder(nn.Module):
    def __init__(self, n, z_dim):
        super(Encoder, self).__init__()
        self.n = n
        self.conv = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=3, padding=1, stride=1)
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=n, out_channels=n, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=n, out_channels=2*n, kernel_size=3, padding=1, stride=1)
        )
        self.stage_2 = nn.Sequential(
            nn.Conv2d(in_channels=2*n, out_channels=2*n, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=2*n, out_channels=3*n, kernel_size=3, padding=1, stride=1)
        )
        self.stage_3 = nn.Sequential(
            nn.Conv2d(in_channels=3*n, out_channels=3*n, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            nn.Conv2d(in_channels=3*n, out_channels=3*n, kernel_size=3, padding=1, stride=1)
        )
        self.fc = nn.Linear(in_features=8*8*3*n, out_features=z_dim)

    @staticmethod
    def sub_sampling(input_tensor):
        channels = input_tensor.size(1)
        return nn.Conv2d(in_channels=channels, out_channels=channels,
                         kernel_size=3, stride=2, padding=1)(input_tensor)

    def forward(self, x):
        x = F.elu(self.conv(x))
        x = F.elu(self.stage_1(x))
        x = self.sub_sampling(x)
        x = F.elu(self.stage_2(x))
        x = self.sub_sampling(x)
        x = F.elu(self.stage_3(x))
        x = x.view(x.size(0), self.n*3*8*8)
        return self.fc(x)


if __name__ == '__main__':
    encoder = Encoder(10, 100)
    decoder = Decoder(100, 10)
    a = torch.randn(1, 100)
    b = decoder(a)
    c = encoder(b)
    print(b.size(), c.size())

