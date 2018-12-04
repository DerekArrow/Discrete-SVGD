from __future__ import print_function
import argparse
from utils import SVGD, Cal_Acc, paint
from Net import Net, ConvNet, FConvNet, FNet
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
# from models.binarized_modules import  Binarize,Ternarize,Ternarize2,Ternarize3,Ternarize4,HingeLoss
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--particles', type=int, default=1, metavar='LR',
                    help='number of particles (default: 1)')
parser.add_argument('--fullpre', type=int, default=0, metavar='LR',
                    help='full precision (default: 0)')
parser.add_argument('--ensemble', type=int, default=0, metavar='LR',
                    help='whether to do ensemble (default: 0)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', default='M',
                    help='dataset: mnist(M) or cifar(C)')
parser.add_argument('--gpus', default=0,
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Construct dataset
kwargs = {'num_workers': 8, 'pin_memory': True}
if args.dataset=='M':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/data/data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size,shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/data/data/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
elif args.dataset=='C':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/data/fan/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/data/fan/', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    print("Error! No Dataset!")

#####   Define Model  ####
if args.fullpre:
    if args.dataset=='M':
        model = FNet()
    elif args.dataset=='C':
        model = FConvNet()
    else:
        prnt("Error! no Models!")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
elif args.ensemble:
    models = []
    optimizers = []
    for i in range(args.particles):
        if args.dataset=='M':
            model = Net()
        elif args.dataset=='C':
            model = ConvNet()
        else:
            print("Error! no Models!")
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        models.append(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizers.append(optimizer)
    
criterion = nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizer.zero_grad()
        loss.backward()

        if args.fullpre:
            optimizer.step()
        else:
            for (name,p) in list(model.named_parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion1(output, target).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().float()

    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))

    return acc.numpy()

def Ensembletrain(epoch):
    for model in models:
        model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        idx = int(batch_idx%args.particles)
        # print('idx:',idx)
        optimizers[idx].zero_grad()
        output = models[idx](data)
        loss = criterion(output, target)
        accp_s = Cal_Acc(output,target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        optimizers[idx].zero_grad()
        loss.backward()

        for p in list(models[idx].parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizers[idx].step()
        for p in list(models[idx].parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccp: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), float(accp_s)))

def Ensembletest():
    for model in models:
        model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
                data, target = Variable(data), Variable(target)
        
        for i in range(args.particles):
            output = models[i](data)
            if i==0:
                Toutputs = output
            else:
                Toutputs += output
            test_loss += criterion1(output, target).item() # sum up batch loss
        pred = Toutputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().float()

    test_loss /= (len(test_loader.dataset)*args.particles)
    acc = (100.0 * correct) / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))

    return acc.numpy()

accs = []
xs = []
for epoch in range(1, args.epochs + 1):
    if args.ensemble:
        Ensembletrain(epoch)
        acc = Ensembletest()
    else:
        train(epoch)
        acc = test()

    accs.append(acc)
    xs.append(epoch)
    if args.fullpre:
        paint(xs, accs, 'epochs', 'accuracy(%)', 'fullpreaccuracy', './figs/{}FullPrecision.jpg'.format(args.dataset))
        scio.savemat('./figs/mat/{}FullPrecision.mat'.format(args.dataset),{'acc':np.array(accs)})
    elif args.ensemble:
        paint(xs, accs, 'epochs', 'accuracy(%)', 'Ensembleaccuracy', './figs/{}Ensemble{}.jpg'.format(args.dataset, args.particles))
        scio.savemat('./figs/mat/{}Ensemble{}.mat'.format(args.dataset, args.particles),{'acc':np.array(accs)})

### CUDA_VISIBLE_DEVICES=2 python toy_main.py --fullpre 0 --ensemble 1 --particles 1 --epochs 100  --dataset C
