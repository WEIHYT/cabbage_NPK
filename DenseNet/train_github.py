import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

from models.densenet_model import *
# from models.densenet_msff import *
from dataloaders import cfg_cabbage as cfg
from dataloaders import make_data_loader
from tqdm import tqdm
from torch import Tensor
from models.focalloss import *
import torchvision
# from torchvision.models import densenet121


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)  #64
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='use which gpu to train, must be a \
                                                                     comma-separated list of integers only (default=0)')  #0，1
    parser.add_argument('--save',action='store_true', default='work/cabbage_201', help='save results')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--dataset', type=str, default='oled_data', choices=['oled_data'],
                                                                   help='dataset name (default: pascal)')
    parser.add_argument('--img_size', type=int, default=(84,84), help='train and val image resize')   #32*32 84*84
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'],
                                                                          help='loss func type (default: ce)')
    parser.add_argument('--pretrained', action='store_true', default=False, help='True加载预训练权重')
    parser.add_argument('--freeze', action='store_true', default=False, help='True冻结特征提取层')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    # setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed)  
        
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    if args.loss_type =='focal':
        args.criterion =FocalLoss()
        print('focal')
    else:
        args.criterion = nn.CrossEntropyLoss()
        print('ce')

    # normMean = [0.49139968, 0.48215827, 0.44653124]
    # normStd = [0.24703233, 0.24348505, 0.26158768]
    # normTransform = transforms.Normalize(normMean, normStd)

    # trainTransform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normTransform
    # ])
    # testTransform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normTransform
    # ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    #...
    # trainLoader = DataLoader(
    #     dset.CIFAR10(root='cifar', train=True, download=True,
    #                  transform=trainTransform),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # testLoader = DataLoader(
    #     dset.CIFAR10(root='cifar', train=True, download=True,
    #                  transform=testTransform),
    #     batch_size=args.batch_size, shuffle=False, **kwargs)
    #...
    trainLoader, testLoader, _ ,_= make_data_loader(args, **kwargs)

    # net = DenseNet(growthRate=12, depth=100, reduction=0.5,bottleneck=True, nClasses=10)
    net = DenseNet201(4)
    # net = DenseNet121(cfg.NUM_CLASSES,pretrained=args.pretrained,weights='/root/cabbage/work/Plant_Village_MSFF/latest.pth')
    #冷冻层 False冻结
    if args.freeze==True:
        # for param in net.features.parameters():
        # # param.requires_grad = True  #一起训练
        #     param.requires_grad = False   #迁移学习，固定他的特征提取层，优化他的全连接分类层

        for name,param in net.named_parameters():
            # param.requires_grad=False
            if (name != 'bn.weight') and (name != 'bn.bias') and (name != 'linear.0.weight') and (name != 'linear.0.bias'):
                param.requires_grad = False
                # if 'dense4.15' in name:
                #     param.requires_grad = True

        params_to_upate=[]
        print("params to learn:")
        for name,pa in net.named_parameters():
            if pa.requires_grad==True:
                params_to_upate.append(pa)
                print('\t',name)

    # print(net)
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.epochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))  #net.state_dict()
        # os.system('/home/xu519/HYT/cabbage/DenseNet/plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    total=0.0
    correct=0.0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):  #trainLoader
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)  修改
        optimizer.zero_grad()
        output = net(data)
        # loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss = args.criterion(output, target)
        loss.requires_grad_(True)  #新加
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        # incorrect = pred.ne(target.data).cpu().sum()
        # err = 100.*incorrect/len(data)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        Acc=100.*correct/total
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), Acc))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), Acc))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    total=0.0
    correct=0.0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        # test_loss += F.nll_loss(output, target).item()
        test_loss = args.criterion(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        # incorrect += pred.ne(target.data).cpu().sum()
        total += target.size(0)
        correct += pred.eq(target).sum().item()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    # err = 100.*incorrect/nTotal
    Acc =100.*correct/total
    print('\nTest set: Average loss: {:.4f}, Acc: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, Acc))

    testF.write('{},{},{}\n'.format(epoch, test_loss, Acc))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__=='__main__':
    main()