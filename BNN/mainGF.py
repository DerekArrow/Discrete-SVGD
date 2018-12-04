from __future__ import print_function
import argparse
from utils import SVGD, Cal_Acc, paint
from Net import GFConvNet, GFNet
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
parser.add_argument('--particles', type=int, default=5, metavar='LR',
                    help='number of particles (default: 1)')
parser.add_argument('--naive', type=int, default=0, metavar='LR',
                    help='whether to let particles independent (default: 0)')
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

#### Construct Dataset
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

######  Define Model ###
criterion = nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss(size_average=False)
models=[]
optimizers = [] 
for i in range(args.particles):
    if args.dataset=='M':
        model = GFNet()
    elif args.dataset=='C':
        model = GFConvNet()
    else:
        print("Error! no Models!")
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    model = model.cuda()
    models.append(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizers.append(optimizer)
 

def train(epoch):
    for model in models:
        model.train()
    w_s = torch.ones(args.particles).cuda()
    logits = []
    for batch_idx, (data, target) in enumerate(train_loader):
        idx = int(batch_idx%args.particles)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizers[idx].zero_grad()   #  clear previous gradient
        anneal = 3 + epoch//8
        output = models[idx](data, 1, anneal)
        if args.naive==0:
            lossphat = -criterion(output, target)
        else:
            lossphat = criterion(output, target)
        lossphat.backward()

        if args.naive==0:
            logits.append(output.data)

            output = models[idx](data, 0)
            accp_s = Cal_Acc(output,target)
            lossp = -criterion(output, target)

            w_s[idx] = torch.exp(lossphat).item() / torch.exp(lossp).item()

        if (idx+1)%args.particles==0:
            #   store each parameters and its gradients, then change to sign
            if args.naive==0:
                #####  use logits to define kernel   #####
                # for i in range(args.particles):
                #     temp = logits[i].view(logits[i].numel()).cuda().unsqueeze(0)
                #     if i==0:
                #         x_s_s = temp
                #     else:
                #         x_s_s = torch.cat((x_s_s,temp),0)
                ###########
                logits = []

                # to calculate kernels use x_s_s
                gradient = SVGD(models, args, w_s)
                #  optimize parameters
            if epoch%40==0:
                for i in range(args.particles):
                    optimizers[i].param_groups[0]['lr']=optimizers[i].param_groups[0]['lr']*0.1

            ########## update the particles############
            if args.naive==0:
                for i in range(args.particles):
                    optimizers[i].zero_grad()
                    F = 0
                    for p in list(models[i].parameters()):
                        T = F + np.cumprod(p.data.shape)[-1]
                        p.grad.data = gradient[i][F:T].reshape(p.data.shape).float()/args.particles
                        F = T
            #########################################
            #    check if the gradient is right
            # _, g_s_s1 = Grad(models, args)
            # scio.savemat('g_s_s1.mat',{'g_s_s1':g_s_s1.cpu().numpy()})
            #######
            for i in range(args.particles):
                optimizers[i].step()
                for p in list(models[i].parameters()):
                    p.data.clamp_(-1,1)
                    if hasattr(p,'org'):
                        p.org.copy_(p.data)

        if batch_idx % args.log_interval == 0 and args.naive==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossp: {:.6f}\tAccp: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lossp, float(accp_s)))

def test(epoch):
    for model in models:
        model.eval()
    total_test_loss = 0
    total_correct = 0

    for i in range(args.particles):
        for p in list(models[i].parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org.sign())

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        test_loss = 0
        Toutputs = 0
        for i in range(args.particles):
            output = models[i](data, 0)
            if i==0:
                Toutputs = output
            else:
                Toutputs += output
            test_loss += criterion1(output, target).item() # sum up batch loss
        pred = Toutputs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        total_correct += pred.eq(target.data.view_as(pred)).cpu().sum().float()

        test_loss /= (len(test_loader.dataset)*args.particles)
        total_test_loss += test_loss

    accp = 100.0 * total_correct / len(test_loader.dataset)
    print('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, total_test_loss, total_correct, len(test_loader.dataset),
        accp))

    for i in range(args.particles):
        for p in list(models[i].parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)

    return accp

accs = []
xs = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    acc = test(epoch)

    accs.append(acc)
    xs.append(epoch)
    if args.naive==0:
        paint(xs, accs, 'epochs', 'accuracy(%)', 'accuracyp', './figs/{}GFout{}.jpg'.format(args.dataset, args.particles))
    else:
        paint(xs, accs, 'epochs', 'accuracy(%)', 'accuracyp', './figs/{}GFnavieout{}.jpg'.format(args.dataset, args.particles))

accs = np.array(accs)
if args.naive==0:
    scio.savemat('./figs/mat/{}GFout{}.mat'.format(args.dataset, args.particles),{'acc':accs})
else:
    scio.savemat('./figs/mat/{}GFnavieout{}.mat'.format(args.dataset, args.particles),{'acc':accs})


### CUDA_VISIBLE_DEVICES=2 python toy_mainGF.py --naive 0 --particles 1 --epochs 100  --dataset C