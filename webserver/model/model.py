import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import multiprocessing as mp
     
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x, pool_dim=2):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, pool_dim)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, pre_enc=False, nImageChannels=3):
        super(DenseNet, self).__init__()   

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = growthRate
        self.conv1 = nn.Conv2d(nImageChannels, nChannels, kernel_size=3, padding=1,
                               bias=False)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
#         print(nChannels, nOutChannels)
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels#+2*growthRate
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*0.8))
#         print(nChannels, nOutChannels)
        self.trans2 = Transition(nChannels, nOutChannels)
        
        nChannels = nOutChannels#+2*growthRate
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*1.0))
#         print(nChannels)
        self.trans3 = Transition(nChannels, nOutChannels)
        
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.fc = nn.Linear(nOutChannels*4*4, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.conv1(x) # out - 32 x 256 x 256
#         print(out.size())
        out = self.trans1(self.dense1(out)) # out - 64 x 128 x 128
#         print(out.size())
        out = self.trans2(self.dense2(out)) # out - 128 x 64 x 64
#         print(out.size())
        out1 = self.trans3(self.dense3(out)) # out - 224 x 32 x 32
#         print(out1.size())
        
        out_f = F.avg_pool2d(F.relu(self.bn1(out1)), 32)
        to_cls = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out1)), 8))
        cls_f = self.fc(to_cls.view(x.size(0), -1))
        
        return out_f, cls_f

class SimDenseNet2(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck, pre_enc=False, nImageChannels=3):
        super(SimDenseNet2, self).__init__()   

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = growthRate
        self.conv1 = nn.Conv2d(nImageChannels, nChannels, kernel_size=3, padding=1,
                               bias=False)

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
#         print(nChannels, nOutChannels)
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels#+2*growthRate
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*0.8))
#         print(nChannels, nOutChannels)
        self.trans2 = Transition(nChannels, nOutChannels)
        
        nChannels = nOutChannels#+2*growthRate
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*1.0))
#         print(nChannels)
        self.trans3 = Transition(nChannels, nOutChannels)
        
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.fc = nn.Linear(nOutChannels*4*4, nClasses)
        
        # Non-Linear Layer for Feature embedding
        self.feature_conv = nn.Sequential(
            nn.Linear(224*4*4, 224*2*2),
            nn.ReLU(),
            nn.Linear(224*2*2, 224),
            nn.ReLU(),
            nn.Linear(224, 100))
           
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.conv1(x) # out - 32 x 256 x 256
        out = self.trans1(self.dense1(out)) # out - 64 x 128 x 128
        out = self.trans2(self.dense2(out)) # out - 128 x 64 x 64
        out1 = self.trans3(self.dense3(out)) # out - 224 x 32 x 32
        
        to_f = F.avg_pool2d(F.relu(self.bn1(out1)), 8).view(x.size(0), -1)
        out_f = self.feature_conv(to_f)
        
        to_cls = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out1)), 8))
        cls_f = self.fc(to_cls.view(x.size(0), -1))
        
        return out_f, cls_f    


    
if __name__ == '__main__':
    ##DONE
    # tmp = torch.ones((2, 9, 256, 256))
    tmp = torch.ones((2, 3, 256, 256))
    model = DenseNet(growthRate=24, depth=20, reduction=0.5,
                      bottleneck=True, nClasses=3)
    print(model(tmp)[0].size())
    print(model(tmp)[1].size())
#     print(model)
#     print(model(tmp)[0].size())
#     print(model(tmp)[1].size())
    #print(model(tmp)[2].size())
## END

