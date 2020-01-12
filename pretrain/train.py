from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from model_new import netG, netD

#writer = SummaryWriter(log_dir='/kaggel/working/')

train_img_path = "train_imgs/"

dataset = dset.ImageFolder(root=train_img_path,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                           shuffle=True, num_workers=2)

G = netG()
G.init_param()
D = netD()
D.init_param()

criterion_BCE = nn.BCELoss()
criterion_MSE = nn.MSELoss()

optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

wtl2 = 0.998
if torch.cuda.is_available():
    D.cuda()
    G.cuda()
    criterion_BCE.cuda()
    criterion_MSE.cuda()
    print("cuda is available!")


def train(G, D, train_loader, criterion_BCE, criterion_MSE, optimizer_G, optimizer_D, epoch):
    real_gt = 1
    fake_gt = 0
    for i, cpu_images in enumerate(train_loader, 0):
        bs = cpu_images[0].size()[0]
        # print(bs)
        labels = Variable(torch.rand(bs, dtype=torch.float))
        if torch.cuda.is_available():
            v_images = Variable(cpu_images[0]).cuda()
        else:
            v_images = Variable(cpu_images[0])
        # TO DO
        crop_images = v_images.clone()
        # print(crop_images.size())
        crop_images[:, 0, 8:25, 15:32] = 2 * 117.0 / 255.0 - 1.0
        crop_images[:, 0, 8:25, 55:72] = 2 * 117.0 / 255.0 - 1.0
        crop_images[:, 0, 8:25, 95:112] = 2 * 117.0 / 255.0 - 1.0

        crop_images[:, 1, 8:25, 15:32] = 2 * 104.0 / 255.0 - 1.0
        crop_images[:, 1, 8:25, 55:72] = 2 * 104.0 / 255.0 - 1.0
        crop_images[:, 1, 8:25, 95:112] = 2 * 104.0 / 255.0 - 1.0

        crop_images[:, 2, 8:25, 15:32] = 2 * 123.0 / 255.0 - 1.0
        crop_images[:, 2, 8:25, 55:72] = 2 * 123.0 / 255.0 - 1.0
        crop_images[:, 2, 8:25, 95:112] = 2 * 123.0 / 255.0 - 1.0
        # print(crop_images[:, 0, 8:25, 15:32])

        D.zero_grad()
        predict = D(v_images)
        labels.data.fill_(real_gt)
        if torch.cuda.is_available():
            errorD_real = criterion_BCE(predict, labels.cuda())
        else:
            errorD_real = criterion_BCE(predict, labels)
        errorD_real.backward(retain_graph=True)

        G.zero_grad()
        fake = G(crop_images)
        predict = D(fake)
        labels.data.fill_(fake_gt)
        if torch.cuda.is_available():
            errorD_fake = criterion_BCE(predict, labels.cuda())
        else:
            errorD_fake = criterion_BCE(predict, labels)
        errorD_fake.backward(retain_graph=True)
        optimizer_D.step()

        errorD = errorD_fake + errorD_real

        # errorG_MSE = criterion_MSE(fake, v_images)
        penalty_weight = v_images.clone()
        penalty_weight.data.fill_(wtl2*10)
        penalty_weight.data[:, 4:29, 11:36, :].fill_(wtl2)
        penalty_weight.data[:, 4:29, 51:76, :].fill_(wtl2)
        penalty_weight.data[:, 4:29, 91:116, :].fill_(wtl2)
        errorG_MSE = (fake - v_images).pow(2)
        errorG_MSE = errorG_MSE * penalty_weight
        errorG_MSE = errorG_MSE.mean()
        labels.data.fill_(real_gt)
        if torch.cuda.is_available():
            errorG_D = criterion_BCE(predict, labels.cuda())
        else:
            errorG_D = criterion_BCE(predict, labels)
        errorG = (1 - wtl2) * errorG_D + wtl2 * errorG_MSE
        errorG.backward()
        optimizer_G.step()

    print("==============", epoch)
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f'
          % (epoch, 500,
             errorD.data, errorG_D.data, errorG.data))
    vutils.save_image(cpu_images[0],
                      'result_new/real_samples_epoch_%03d.png' % (epoch))
    vutils.save_image(crop_images.data,
                      'result_new/real_crop_epoch_%03d.png' % (epoch))
    vutils.save_image(fake.data,
                      'result_new/fake_samples_epoch_%03d.png' % (epoch))

            #     writer.add_scalar('Tain_netD/Loss', terrorD.data, epoch)
            #     writer.add_scalar('Tain_netG/Loss', errorG.data, epoch)

    torch.save({'epoch': epoch,
                'state_dict': G.state_dict()},
                'result_new/netG_' + str(epoch) + '.pth')
    torch.save({'epoch': epoch,
                'state_dict': D.state_dict()},
                'result_new/netD_' + str(epoch) + '.pth')


for epoch in range(500):
    print("start training")
    train(G, D, train_loader, criterion_BCE, criterion_MSE, optimizer_G, optimizer_D, epoch)




