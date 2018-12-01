import numpy as np
from loaduai import readUai
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class Model(nn.Module):
	def __init__(self, dim):
		super(Model, self).__init__()
		self.dim = dim
		for i in range(self.dim):
			setattr(self, 'fc'+str(i+1), nn.Linear(self.dim, 1, bias=True))

	def forward(self, x):
		out = torch.FloatTensor([0])
		for i in range(self.dim):
			out += torch.exp(getattr(self, 'fc'+str(i+1))(x))

		return torch.log(out) 


def construct(tab):
	data = np.argwhere(tab!=-1)
	target = tab.flatten()
	return data, target

def convex_learning(var, tab):
	epochs = 500
	dim = len(var)
	model = Model(dim)
	optimizer = optim.Adam(model.parameters(), lr=0.1)
	model.train()
	data, target = construct(tab)
	terms = target.size
	data = torch.from_numpy(data).float()
	target = torch.from_numpy(target).float()
	target = torch.log(target)
	for epoch in range(epochs):
		for i in range(terms):
			x = data[i,:]
			y = target[i]
			x = Variable(x)
			y = Variable(y)
			optimizer.zero_grad()
			output = model(x)
			loss = (output-y)**2
			loss.backward()
			optimizer.step()
	print('loss:', loss.item())

	for i, pa in enumerate(model.parameters()):
		p = pa.data
		# print(p.shape)
		if i==0:
			a = p
		elif i==1:
			b = p
		elif i%2==0:
			a = torch.cat((a,p))
		else:
			b = torch.cat((b,p))
	return [a, b]

def convert(factors):
	paras = []
	for factor in factors:
		var = factor.v
		tab = factor.t
		print('var:', var)
		print('tab:', tab)
		para = convex_learning(var, tab)
		paras.append(para)
	return paras

if __name__=='__main__':
	factors, dims = readUai('uai1.txt')
	paras = convert(factors)
	for a,b in paras:
		print(a,b)
