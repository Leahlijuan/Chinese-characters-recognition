import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 100, 1),
            nn.BatchNorm2d(100),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(100, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.net(input)
        #print(x.size())
        return x

    def init_param(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                param_shape = layer.weight.shape
                layer.weight.data = torch.tensor(np.random.normal(0.0, 0.02, size=param_shape), dtype=torch.float)
                layer.bias.data.zero_()
            if isinstance(layer, nn.ConvTranspose2d):
                param_shape = layer.weight.shape
                layer.weight.data = torch.tensor(np.random.normal(0.0, 0.02, size=param_shape), dtype=torch.float)
                layer.bias.data.zero_()
            if isinstance(layer, nn.BatchNorm2d):
                param_shape = layer.weight.shape
                layer.weight.data = torch.tensor(np.random.normal(1.0, 0.02, size=param_shape), dtype=torch.float)
                layer.bias.data.zero_()


class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.net = nn.Sequential(
            # 3*32*128
            nn.Conv2d(3,64,4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            # 64*16*64
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64*8*32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128*4*16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 256*2*8
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512*1*4
            nn.Conv2d(512, 1, (1, 4)),
            nn.Sigmoid(),
            # 100*1*1
        )

    def forward(self, input):
        x = self.net(input)
        bs = x.size()[0]
        #print(x.size())
        x = x.view((bs, 1))
        return x

    def init_param(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                param_shape = layer.weight.shape
                layer.weight.data = torch.tensor(np.random.normal(0.0, 0.02, size=param_shape), dtype=torch.float)
                layer.bias.data.zero_()
            if isinstance(layer, nn.ConvTranspose2d):
                param_shape = layer.weight.shape
                layer.weight.data = torch.tensor(np.random.normal(0.0, 0.02, size=param_shape), dtype=torch.float)
                layer.bias.data.zero_()
            if isinstance(layer, nn.BatchNorm2d):
                param_shape = layer.weight.shape
                layer.weight.data = torch.tensor(np.random.normal(1.0, 0.02, size=param_shape), dtype=torch.float)
                layer.bias.data.zero_()
