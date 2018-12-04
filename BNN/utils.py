import os
import torch
import logging.config
import scipy.io as scio
import scipy.stats as st
import shutil
import pandas as pd
import numpy as np
import numpy.matlib as nm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
from scipy.spatial.distance import pdist, squareform

def l2distance(theta):
    XY = torch.mm(theta, theta.t())
    X2 = torch.sum(theta**2, 1, keepdim=True)
    pairwise_dists = X2 + X2.t() - 2*XY
    return pairwise_dists

def SVGD(models, args, w_s, h = -1):
    for i in range(args.particles):
        flag = 0
        for p in list(models[i].parameters()):
            if flag==0:
                flag = 1
                x_s = p.data.view(p.data.numel()).cuda()
                g_s = p.grad.data.view(p.grad.data.numel()).cuda()
            else:
                x_s = torch.cat((x_s, p.data.view(p.data.numel()).cuda()),0)
                g_s = torch.cat((g_s, p.grad.data.view(p.grad.data.numel()).cuda()),0)
        # p.data = torch.sign(p.data)
        # print('grad',p.grad.data.shape)
        # print('data',p.data.shape)
        x_s = x_s.unsqueeze(0)
        g_s = g_s.unsqueeze(0)
        if i==0:
            x_s_s = x_s
            g_s_s = g_s
        else:
            x_s_s = torch.cat((x_s_s,x_s),0)
            g_s_s = torch.cat((g_s_s,g_s),0)
    # print(g_s_s.shape)
    theta = x_s_s
    w_s = w_s/w_s.sum()
    if theta.shape[0]>1:
        pairwise_dists = l2distance(theta)
        if h < 0: # if h < 0, using median trick
            h = torch.median(pairwise_dists)  
            h = torch.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        Kxy = torch.exp( -pairwise_dists / h**2 / 2)
        # print('sq_dist:',sq_dist,'\tpairwise_dist:',pairwise_dists,'\th:',h,'\tKxy:',Kxy)
        # print('w_s:',w_s)
        weighted_Kxy = Kxy                         #  every xj's weights 
        for row in range(Kxy.shape[0]):
            weighted_Kxy[row,:] = torch.mul(Kxy[row,:], w_s)  # weighted kernel
        dxkxy = -torch.matmul(weighted_Kxy, theta)
        sumkxy = torch.sum(weighted_Kxy, dim=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + torch.mul(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
    else:
        weighted_Kxy = torch.ones((1,1)).cuda()
        dxkxy = torch.zeros(theta.shape).cuda()
    ## calculate the gradient 
    gradient = - (torch.mm(weighted_Kxy, g_s_s) + dxkxy)
    return gradient



def Grad(models, args):
    for i in range(args.particles):
        flag = 0
        for p in list(models[i].parameters()):
            if hasattr(p,'org'):
                if flag==0:
                    flag = 1
                    x_s = p.org.view(p.org.numel()).cuda()
                    g_s = p.grad.data.view(p.grad.data.numel()).cuda()
                else:
                    x_s = torch.cat((x_s, p.org.view(p.org.numel()).cuda()),0)
                    g_s = torch.cat((g_s, p.grad.data.view(p.grad.data.numel()).cuda()),0)
            else:
                if flag==0:
                    flag = 1
                    x_s = p.data.view(p.data.numel()).cuda()
                    g_s = p.grad.data.view(p.grad.data.numel()).cuda()
                else:
                    x_s = torch.cat((x_s, p.data.view(p.data.numel()).cuda()),0)
                    g_s = torch.cat((g_s, p.grad.data.view(p.grad.data.numel()).cuda()),0)
        # p.data = torch.sign(p.data)
        # print('grad',p.grad.data.shape)
        # print('data',p.data.shape)
        x_s = x_s.unsqueeze(0)
        g_s = g_s.unsqueeze(0)
        if i==0:
            x_s_s = x_s
            g_s_s = g_s
        else:
            x_s_s = torch.cat((x_s_s,x_s),0)
            g_s_s = torch.cat((g_s_s,g_s),0)
    # print(g_s_s.shape)
    return x_s_s, g_s_s

def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    #def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
    #    self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    # kernel_img = model.features[0][0].kernel.data.clone()
    # kernel_img.add_(-kernel_img.min())
    # kernel_img.mul_(255 / kernel_img.max())
    # save_image(kernel_img, 'kernel%s.jpg' % epoch)

def Cal_Acc(output, target):
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.data.view_as(pred)).cpu().float().mean()
    return correct.item()

def paint(x, y, xlabel, ylabel, title, savefig):
    plt.figure() 
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefig)