import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from binarized_modules import BinarizeLinear,BinarizeConv2d
from GFSVBNbinarized_modules import  GFBinarizeLinear,GFBinarizeConv2d

# for mnist
def MBinarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def MSigmoid(tensor, quant=4):
    temp = torch.exp(torch.neg(quant*tensor))
    return (1-temp) / (1+temp) 

class MBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(MBinarizeLinear, self).__init__(*kargs, **kwargs)
        self.weight.org=self.weight.data.clone().cuda()
        self.bias = None

    def forward(self, input):

        if input.size(1) != 784:
            input.data=MBinarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=MBinarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        # if not self.bias is None:
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out

class MGFBinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(MGFBinarizeLinear, self).__init__(*kargs, **kwargs)
        # self.weight.org = 0
        self.weight.org=self.weight.data.clone()
        self.bias = None

    def forward(self, input, isGF, anneal): 

        if input.size(1) != 784:
            if isGF==0:
                input=MBinarize(input)
            else:
                input=MSigmoid(input, anneal)
        # if not hasattr(self.weight,'org'):
        #     self.weight.org=self.weight.data.clone()
        # self.weight.data=torch.clamp_(self.weight.org,-1,1)
        if isGF==0:
            out = nn.functional.linear(input, MBinarize(self.weight))
        else:
            out = nn.functional.linear(input, MSigmoid(self.weight, anneal))
        # if not self.bias is None:
        #     self.bias.org=self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out

class Net(nn.Module):
    def __init__(self, num=28):
        super(Net, self).__init__()
        self.num = num
        self.infl_ratio=3
        self.fc1 = MBinarizeLinear(num*num, num*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(num*self.infl_ratio)
        self.fc4 = MBinarizeLinear(num*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.num * self.num)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc4(x)
        return x

class GFNet(nn.Module):
    def __init__(self, num=28):
        super(GFNet, self).__init__()
        self.num = num
        self.infl_ratio=3
        self.fc1 = MGFBinarizeLinear(num*num, num*self.infl_ratio)
        
        self.bn1 = nn.BatchNorm1d(num*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.sgmd1 = nn.Sigmoid()
        self.fc2 = MGFBinarizeLinear(num*self.infl_ratio, 10)
        
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.drop=nn.Dropout(0.5)

    def forward(self, x, isGF=0, anneal=1):
        x = x.view(-1, self.num * self.num)
        x1 = self.fc1(x, isGF, anneal)
        x = self.bn1(x1)
        # x = self.htanh1(x)
        # x = self.sgmd1(x)
        x2 = self.fc2(x, isGF, anneal)
        return x2

class FNet(nn.Module):
    def __init__(self, num=28):
        super(FNet, self).__init__()
        self.num = num
        self.infl_ratio=3
        self.fc1 = nn.Linear(num*num, num*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(num*self.infl_ratio)
        self.fc4 = nn.Linear(num*self.infl_ratio, 10)
        self.logsoftmax=nn.LogSoftmax(dim=1)
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, self.num * self.num)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc4(x)
        return x

######  for cifar #######
class ConvNet(nn.Module):
    def __init__(self, num=32, num_classes=10):
        super(ConvNet, self).__init__()
        self.ratioInfl=128
        self.features = nn.Sequential(
            BinarizeConv2d(3, int(self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(int(self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(self.ratioInfl), int(self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(self.ratioInfl), int(2*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(int(2*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(2*self.ratioInfl), int(2*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(2*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(2*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(int(4*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(16 * 4 * self.ratioInfl, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Dropout(0.5),
            BinarizeLinear(1024, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * self.ratioInfl)
        x = self.classifier(x)
        return x

class GFConvNet(nn.Module):
    def __init__(self, num=32, num_classes=10):
        super(GFConvNet, self).__init__()
        self.ratioInfl=128
        
        self.conv1 = GFBinarizeConv2d(3, int(self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(int(self.ratioInfl))

        self.conv2 = GFBinarizeConv2d(int(self.ratioInfl), int(self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(int(self.ratioInfl))

        self.conv3 = GFBinarizeConv2d(int(self.ratioInfl), int(2*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(int(2*self.ratioInfl))

        self.conv4 = GFBinarizeConv2d(int(2*self.ratioInfl), int(2*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(int(2*self.ratioInfl))

        self.conv5 = GFBinarizeConv2d(int(2*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False)
        self.bn5 = nn.BatchNorm2d(int(4*self.ratioInfl))

        self.conv6 = GFBinarizeConv2d(int(4*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(int(4*self.ratioInfl))
        
        self.fc1 = GFBinarizeLinear(16 * 4 * self.ratioInfl, 1024, bias=False)
        self.fcbn1 = nn.BatchNorm1d(1024)
        self.fc2 = GFBinarizeLinear(1024, num_classes)
        self.fcbn2 = nn.BatchNorm1d(num_classes)

    def forward(self, x, isGF=0, anneal=1):
        x = self.conv1(x, isGF, anneal)
        x = self.bn1(x)
        x = self.conv2(x, isGF, anneal)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x, isGF, anneal)
        x = self.bn3(x)
        x = self.conv4(x, isGF, anneal)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.conv5(x, isGF, anneal)
        x = self.bn5(x)
        x = self.conv6(x, isGF, anneal)
        x = self.pool6(x)
        x = self.bn6(x)
        x = x.view(-1, 16 * 4 * self.ratioInfl)
        
        x = self.fc1(x, isGF, anneal)
        x = self.fcbn1(x)
        nn.Dropout(0.5)
        x = self.fc2(x, isGF, anneal)
        x = self.fcbn2(x)
        return x

class FConvNet(nn.Module):
    def __init__(self, num=32, num_classes=10):
        super(FConvNet, self).__init__()
        self.ratioInfl=128
        self.features = nn.Sequential(
            nn.Conv2d(3, int(self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(int(self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(int(self.ratioInfl), int(self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(int(self.ratioInfl), int(2*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(int(2*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(int(2*self.ratioInfl), int(2*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(2*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(int(2*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(int(4*self.ratioInfl), int(4*self.ratioInfl), kernel_size=3, stride=1, padding=1, bias = False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(int(4*self.ratioInfl)),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * self.ratioInfl, 1024, bias = False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes, bias = False),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 4 * self.ratioInfl)
        x = self.classifier(x)
        return x