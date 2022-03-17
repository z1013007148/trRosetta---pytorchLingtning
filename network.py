import torch
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR


def d():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def elu():
    return nn.ELU(inplace=True)


def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)


def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)


class trRosettaNetworkModule(LightningModule):
    def __init__(self, filters=64, kernel=3, num_layers=61):
        super().__init__()

        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers

        self.first_block = nn.Sequential( # in:526 out: 64 kernel:1
            conv2d(526, filters, 1),
            instance_norm(filters),
            elu()
        )

        # stack of residual blocks with dilations
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)] # [ 64个[1,2,4,8,16] ]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters), # 归一化
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations]) # 不同间距的卷积核的循环

        self.activate = elu()

        # conv to anglegrams and distograms
        self.to_prob_theta = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
        self.to_prob_phi = nn.Sequential(conv2d(filters, 13, 1), nn.Softmax(dim=1))
        self.to_distance = nn.Sequential(conv2d(filters, 37, 1), nn.Softmax(dim=1))
        # self.to_prob_bb = nn.Sequential(conv2d(filters, 3, 1), nn.Softmax(dim=1))
        self.to_prob_omega = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.first_block(x)

        for layer in self.layers: # 61层
            x = self.activate(x + layer(x)) # +上层直连，elu激活

        prob_theta = self.to_prob_theta(x)  # anglegrams for theta
        prob_phi = self.to_prob_phi(x)  # anglegrams for phi

        x = 0.5 * (x + x.permute((0, 1, 3, 2)))  # symmetrize 对称化

        prob_distance = self.to_distance(x)  # distograms
        # prob_bb = self.to_prob_bb(x)            # beta-strand pairings (not used)
        prob_omega = self.to_prob_omega(x)  # anglegrams for omega

        return prob_phi, prob_theta, prob_omega, prob_distance

    def training_step(self, batch, batch_idx):
        x, y1, y2, y3, y4 = batch

        logits = self(x)  # forward(x) （batch,channel,height,width）
        loss1 = F.cross_entropy(logits[0], y1.long())
        loss2 = F.cross_entropy(logits[1], y2.long())
        loss3 = F.cross_entropy(logits[2], y3.long())
        loss4 = F.cross_entropy(logits[3], y4.long())
        loss = loss1 + loss2 + loss3 + loss4
        torch.cuda.empty_cache()
        return loss # -->training_step_end()

    # def validation_step(self, *args, **kwargs): # 定义了step，对应的loader也会使用，功能就开启了，fit里可以不用填
    #     pass
    #
    # def test_step(self, *args, **kwargs):
    #     pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # scheduler = StepLR(optimizer, step_size=1)
        return optimizer#, scheduler
