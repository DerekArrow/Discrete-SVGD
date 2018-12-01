import numpy as np
import numpy.matlib as nm
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform

def GFSVGD(theta, w_s, h = -1):
    if theta.shape[0]>1:
        XY = torch.mm(theta, theta.t())
        X2 = torch.sum(theta**2, 1, keepdim=True)
        pairwise_dists = X2 + X2.t() - 2*XY
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        Kxy = torch.exp( -pairwise_dists / h**2 / 2).float()
        # print('pairwise_dist:',pairwise_dists,'\th:',h,'\tKxy:',Kxy)
        # print('w_s:',w_s)
        weighted_Kxy = Kxy                      #  every xj's weights 
        for row in range(Kxy.shape[0]):
            weighted_Kxy[row,:] = torch.mul(Kxy[row,:], w_s)  # weighted kernel
        dxkxy = -torch.matmul(weighted_Kxy, theta)
        sumkxy = torch.sum(weighted_Kxy, dim=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + torch.mul(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
    else:
        weighted_Kxy = torch.ones((1,1))
        dxkxy = torch.zeros(theta.shape)
    return (weighted_Kxy, dxkxy)
    

def update(theta, cMRF, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
    # Check input
    if theta is None:
        raise ValueError('theta cannot be None!')
    
    # adagrad with momentum
    fudge_factor = 1e-6
    historical_grad = 0
    for iter in range(n_iter):
        if debug and (iter+1) % 1000 == 0:
            pass
        # print ('iter ' + str(iter+1))
        # print('theta', theta)
        # lamda = 0.3 - iter/(n_iter*10)
        lamda = 5 + (20*iter)//n_iter
        w_s, lnpgrad = cMRF(theta, lamda)
        # calculating the kernel matrix
        kxy, dxkxy = GFSVGD(theta, w_s , h = -1)  
        # print('kxy:',kxy)
        # print('dxkxy:',dxkxy)
        grad_theta = (torch.mm(kxy, lnpgrad) + dxkxy) / theta.shape[0]  
        
        # adagrad 
        if iter == 0:
            historical_grad = historical_grad + grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = torch.div(grad_theta, fudge_factor+torch.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad 
        
    return theta