'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.utils.tensorboard import SummaryWriter
from net_methods import train, test


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lp_norm', default=2, type=int, help='lp norm')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    print('pre')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print('post')
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # tensorboard
    writer = SummaryWriter()

    # Data
    print('==> Preparing data..')
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

    if device == 'cuda':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=0)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    nets = []
    net = VGG('VGG11', lp_norm=args.lp_norm, device=device)
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)

    # Training
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_acc = train(net, trainloader, criterion, optimizer, epoch, device)
        test_loss, test_acc, best_acc = test(net, testloader, criterion, epoch, device, best_acc)

        writer.add_scalars('Loss', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'train_acc': train_acc, 'test_acc': test_acc}, epoch)

    writer.close()


def main_nets():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--lp_norm', default=2, type=int, help='lp norm')
    parser.add_argument('--epochs', default=5, type=int, help='epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # tensorboard
    writer = SummaryWriter()

    # Data
    print('==> Preparing data..')
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

    if device == 'cuda':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=0)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    nets = []
    norms = [1, 2, 3, 4, 5, 10, 100, 1000]
    nets.append(VGG('VGG11', norm_type='ST', device=device).to(device))
    nets.append(VGG('VGG11', norm_type='BN', device=device).to(device))
    for i in range(len(norms)):
        nets.append(VGG('VGG11', norm_type='LP', lp_norm=norms[i], device=device).to(device))

    for i in range(len(nets)):
        if device == 'cuda':
            nets[i] = torch.nn.DataParallel(nets[i])
            cudnn.benchmark = True

    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = net.to(device)


    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizers = []
    for i in range(len(nets)):
        optimizers.append(optim.SGD(nets[i].parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4))

    # Training
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss_train_dict = {}
        loss_test_dict = {}
        acc_train_dict = {}
        acc_test_dict = {}

        for i in range(len(nets)):
            train_loss, train_acc = train(nets[i], trainloader, criterion, optimizers[i], epoch, device)
            test_loss, test_acc, best_acc = test(nets[i], testloader, criterion, epoch, device, best_acc)

            loss_train_dict['l'+str(norms[i])] = train_loss
            loss_test_dict['l'+str(norms[i])] = test_loss
            acc_train_dict['l' + str(norms[i])] = train_acc
            acc_test_dict['l' + str(norms[i])] = test_acc

        # writer.add_scalars('Loss_'+str(i), {'train_loss_'+str(i): train_loss, 'test_loss_'+str(i): test_loss}, epoch)
        # writer.add_scalars('Accuracy_'+str(i), {'train_acc_'+str(i): train_acc, 'test_acc_'+str(i): test_acc}, epoch)
        writer.add_scalars('Loss_train', loss_train_dict, epoch)
        writer.add_scalars('Loss_test', loss_test_dict, epoch)
        writer.add_scalars('Acc_train', acc_train_dict, epoch)
        writer.add_scalars('Acc_test', acc_test_dict, epoch)

    writer.close()

if __name__ == '__main__':
    main_nets()