'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple
# from utils.api import _log_api_usage_once

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1_1 = nn.BatchNorm2d(in_planes)
        self.conv1_1 = nn.Conv2d(in_planes, 2*growth_rate, kernel_size=1, stride=1,bias=False)
        self.bn1_2 = nn.BatchNorm2d(in_planes)
        self.conv1_2 = nn.Conv2d(in_planes, growth_rate, kernel_size=3,stride=1,padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(in_planes)
        self.conv1_3 = nn.Conv2d(in_planes, growth_rate, kernel_size=5,stride=1,padding=2 ,bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out1 = self.conv1_1(F.relu(self.bn1_1(x)))
        out2 = self.conv1_2(F.relu(self.bn1_2(x)))
        out3 = self.conv1_3(F.relu(self.bn1_3(x)))
        out = torch.cat((out1,out2,out3), 1)
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        # _log_api_usage_once(self)
        self.growth_rate = growth_rate
        self.drop_path_prob=0.0  #修改

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        # self.linear = nn.Linear(num_planes, num_classes)
        self.linear =nn.Sequential(nn.Linear(num_planes,num_classes),
		 				  nn.LogSoftmax(dim=1))

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = F.log_softmax(self.linear(out), dim=1)

        # out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


def DenseNet121(num_class,pretrained,weights, **kwargs: Any):
    model=DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_class, **kwargs)
    if pretrained==True:
        # model =torch.load(weights,map_location='cuda')
        net_weights = model.state_dict()
        state_dict=torch.load(weights,map_location='cuda') 
        pre_dict =  {k: v for k, v in state_dict.items() if net_weights[k].numel() == v.numel()}
        model.load_state_dict(pre_dict,strict=False)
        # in_channel = model.bn.in_features
        # model.linear = nn.Linear(in_channel, num_class)
    return model

def DenseNet169(num_class):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_classes=num_class)

def DenseNet201(num_class):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_classes=num_class)

def DenseNet161(num_class):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, num_classes=num_class)

def densenet_cifar(num_class):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=num_class)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
