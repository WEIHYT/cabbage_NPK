#!/usr/bin/python
# import sys

# 添加模块搜索路径
# sys.path.append('./Classify_ModelZoo')
# sys.path.append('./Classify_ModelZoo/ResearchTools')
# sys.path.append('/home/nas928/chenmengzhen/Classify_ModelZoo/ResearchTools')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import os.path as osp

from models import *
from util.utils import progress_bar
from util.ios import mkdir_if_missing

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()



best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def whitebox_data_transform(data, nquery=100, direction='forward'):
    channel_mean = (0.4914, 0.4822, 0.4465)
    channel_std = (0.2023, 0.1994, 0.2010)
    if direction == 'forward':
        for i in range(nquery):
            for j in range(3):
                data[i][j] = (data[i][j] - channel_mean[j]) / channel_std[j]
        return data
    elif direction == 'backward':
        for i in range(nquery):
            for j in range(3):
                data[i][j] = (data[i][j] * channel_mean[j]) + channel_std[j]
        return data
    else:
        raise ValueError('Data transformation direction is either forward or backward.')
    
def load_model(modelname, num_class):
    print('==> Building model..')
    # ResNet
    if(modelname == 'ResNet18'):
        net = ResNet18(num_class)
    elif(modelname == 'ResNet34'):
        net = ResNet34(num_class)
    elif(modelname == 'ResNet50'):
        net = ResNet50(num_class)
    elif(modelname == 'ResNet101'):
        net = ResNet101(num_class)
    elif(modelname == 'ResNet152'):
        net = ResNet152(num_class)
    # VGG
    elif(modelname == 'VGG11'):
        net = VGG('VGG11',num_class)
    elif(modelname == 'VGG13'):
        net = VGG('VGG13',num_class)
    elif(modelname == 'VGG16'):
        net = VGG('VGG16',num_class)
    elif(modelname == 'VGG19'):
        net = VGG('VGG19',num_class)
    # DenseNet
    elif(modelname == 'DenseNet121'):
        net = DenseNet121(num_class)
    elif(modelname == 'DenseNet161'):
        net = DenseNet161(num_class)
    elif(modelname == 'DenseNet169'):
        net = DenseNet169(num_class)
    elif(modelname == 'DenseNet201'):
        net = DenseNet201(num_class)
    # MobileNet
    elif(modelname == 'MobileNet'):
        net = MobileNet(num_class)
    elif(modelname == 'MobileNetV2'):
        net = MobileNetV2(num_class)  
    # EfficientNet
    elif(modelname == 'EfficientNetB0'):
        net = EfficientNetB0(num_class)  
    # ShuffleNetV2
    elif(modelname == 'Shufflenet_v2_x0_5'):
        net = ShuffleNetV2(net_size=0.5, num_class = num_class)
    elif(modelname == 'Shufflenet_v2_x1_0'):
        net = ShuffleNetV2(net_size=1, num_class = num_class)
    elif(modelname == 'Shufflenet_v2_x1_5'):
        net = ShuffleNetV2(net_size=1.5, num_class = num_class)
    elif(modelname == 'Shufflenet_v2_x2_0'):
        net = ShuffleNetV2(net_size=2, num_class = num_class)
    else:
        raise ValueError('Unable to find the model.')
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = ResNeXt29_2x64d()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = RegNetX_200MF()
    # net = SimpleDLA()

    return net

def load_dataloader(dataset):
    print('==> Preparing data..')
    if(dataset == 'cifar10'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
        
        return trainloader, testloader
    
    elif(dataset == 'cifar100'):
        transform_train = transforms.Compose([transforms.RandomResizedCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test = transforms.Compose([transforms.Resize((32, 32)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
        train_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=True,
                                                 download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            train_dataset,batch_size=64,shuffle=True,pin_memory=True,num_workers=2)
 
        test_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100', train=False,
                                               download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            test_dataset,batch_size=64,shuffle=False,pin_memory=True,num_workers=2)
        
        return trainloader, testloader
    
    elif(dataset == 'military'):
        raise ValueError('Unable to support the dataset.')
    else: 
        raise ValueError('Unable to support the dataset.')
    
def main(whitebox_config): 
    global start_epoch
    global best_acc
    start_epoch = 0
    best_acc = 0
    # Data
    trainloader, testloader = load_dataloader(whitebox_config['dataset'])
    # Model
    net = load_model(whitebox_config['model_name'],whitebox_config['num_class'])
    torch.manual_seed(whitebox_config['seed'])
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = whitebox_config['device']
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        torch.cuda.manual_seed(whitebox_config['seed'])
    modeldir = osp.join('checkpoint', whitebox_config['exp_phase'],whitebox_config['dataset'], whitebox_config['model_name'])
    mkdir_if_missing(modeldir)
    save_path = os.path.join(modeldir, 'ckpt.pth')

    if whitebox_config['resume']:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(modeldir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(save_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=whitebox_config['lr'],
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Testing
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'modelName':whitebox_config['model_name'],
            }
            torch.save(state, save_path)
            best_acc = acc
 
    for epoch in range(start_epoch, start_epoch+whitebox_config['epochs']):
        train(epoch)
        test(epoch)
        scheduler.step()

if __name__ == "__main__":
    main(dict(
            image_size = (3, 84, 84),
            exp_phase = 'Classify_ModelZoo',
            num_class = 10,
            model_name = 'ResNet18',
            seed = 1,
            device = 'cuda',
            lr = 0.1,
            resume = False,
            attrinfer_size =(100,10),   # (query number, num_class)
            epochs = 200,
            dataset = 'cifar10',
        )
    )