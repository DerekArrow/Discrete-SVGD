import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

import torch.nn._functions as tnnf

def xup(tensor, quant):
    tup = (-(quant*tensor)).exp()
    outup = (1-tup) / (1+tup)
    return  outup

def xdown(tensor, quant):
    tdown = (quant*tensor).exp()
    outdown = (tdown-1) / (tdown+1)
    return outdown

def Sigmoid(tensor, quant=10):
    out = torch.where(tensor>=0, xup(tensor,quant), xdown(tensor,quant))
    # out = tensor
    # out = xup(tensor)
    return out

class GFBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(GFBinarizeLinear, self).__init__(*kargs, **kwargs)
        self.weight.org=self.weight.data.clone()
        self.bias = None

    def forward(self, input, isGF, anneal):

        if input.size(1) != 3*32*32:
            if isGF==0:
                input=Binarize(input)
            else:
                # print('Linear Input')
                input=Sigmoid(input, anneal)
        # if not hasattr(self.weight,'org'):
        #     self.weight.org=self.weight.data.clone()
        # self.weight.data=torch.clamp_(self.weight.org,-1,1)
        if isGF==0:
            out = nn.functional.linear(input, self.weight)
        else:
            # print("linear")
            out = nn.functional.linear(input, Sigmoid(self.weight, anneal))
        # if not self.bias is None:
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out

class GFBinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(GFBinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.weight.org=self.weight.data.clone()
        self.bias = None
        
    def forward(self, input, isGF, anneal):
        if input.size(1) != 3:
            if isGF==0:
                input.data = Binarize(input.data)
            else:
                # print("conv input")
                input = Sigmoid(input, anneal)
        # if not hasattr(self.weight,'org'):
        #     self.weight.org=self.weight.data.clone()
        # self.weight.data=torch.clamp_(self.weight.org,-1,1)
        if isGF==0:
            out = nn.functional.conv2d(input, Binarize(self.weight), None, self.stride,
                                   self.padding, self.dilation, self.groups)
        else:
            # print("conv")
            out = nn.functional.conv2d(input, Sigmoid(self.weight, anneal), None, self.stride,
                                   self.padding, self.dilation, self.groups)
        # if not self.bias is None:
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
