import numpy as np
from loaduai import readUai
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from convert import convert
import matplotlib.pyplot as plt
import numpy.matlib as nm
import pandas as pd
from scipy import stats
from numpy.linalg import inv
import scipy.io as scio
from tools import paint
from GF_SVGD import GFSVGD, update
import scipy.stats as st
import argparse
import copy
import os 

class CMRF(nn.Module):
	def __init__(self, factors, paras, samples, dims, Z):
		super(CMRF,self).__init__()
		self.paras = copy.deepcopy(paras)
		self.factors = factors
		self.dims = dims
		self.Z = Z
		self.samples = samples
		self.bb = []

	def discrete(self, a):
		b = (torch.sign(a.data-0.5) + 1) / 2
		return b

	def sigmoid(self, x, lamda):
		expx = torch.exp(-lamda*x)
		X = 1/(1+expx)
		DX = (lamda*expx) / (1+expx)**2
		return X, DX

	def extend(self):
		self.bb = []
		for _,b in self.paras:
			bb = b.unsqueeze_(1)
			for _ in range(self.samples-1):
				bb = torch.cat((bb,b),1)
			self.bb.append(bb.float())

	def forward(self, x, lamda):
		x_discrete = self.discrete(x)
		sigx, dx = self.sigmoid(x, lamda)
		Phat = torch.ones(self.samples)
		P = torch.ones(self.samples)
		Dx = torch.zeros(x.shape)	
		for i in range(len(self.factors)):
			# print('i=',i)
			var = self.factors[i].v
			a, _ = self.paras[i]
			bb = self.bb[i]
			vlabel = [v.label for v in var]
			# print(a,bb,vlabel)
			expterm = (torch.mm(a,sigx[:,vlabel].t().float()) + bb).exp()
			Ph = torch.sum(expterm, 0)
			# print('Ph:',Ph)
			Phat *= Ph
			P *= torch.sum((torch.mm(a,x_discrete[:,vlabel].t()) + bb).exp(), 0)
			for k in range(self.samples):
				for j in range(len(vlabel)):
					jj = vlabel[j]
					Dx[k,jj] += torch.sum(a[:,j] * expterm[:,k]) / Ph[k]

		# G0 = st.norm.pdf(x, 0, lamda)
		# G1 = st.norm.pdf(x, 1, lamda)
		# Phat = torch.from_numpy(G0+G1).prod(1).float().mul(Phat)
		# Var = np.square(lamda)
		# dG0 = torch.exp(-(x**2)/(2*Var))
		# dG1 = torch.exp(-((x-1)**2)/(2*Var))
		# dx = - (x*dG0/Var + (x-1)*dG1/Var) / (dG0+dG1)
		Dx *= dx
		w_s = Phat/P
		w_s = w_s/torch.sum(w_s)
		# print('G0+G1:',G0+G1)
		# print('Phat:', Phat)
		# print('P:',P)
		# print('dx:',dx)
		# print('Dx:',Dx)
		# print('w_s:',w_s)

		return w_s, Dx

def gen_value(dim):
    value = []
    def ten2two(x):
        a=[]
        for i in bin(x)[2:]:
            a.append(int(i))
        a = [ 0 for _ in range(dim - len(a)) ] + a
        return np.array(a)

    for k in range(np.power(2,dim)):
        value.append(ten2two(k))
    return np.array(value)

def marignal(tab, idxs):
	for i in range(len(idxs)):
		if i==len(idxs)-1:
			tab = tab[idxs[i]]
		else:
			tab = tab[idxs[i],...]
	return tab

def get_prob(factors, term, idx, dims):
	p1=1
	p0=1
	for factor in factors:
		interm = copy.deepcopy(term)
		var = factor.v
		tab = factor.t
		vlabel = [v.label for v in var]
		if idx in vlabel:
			# p1
			interm[idx] = 1
			idxs = interm[vlabel]
			partp1 = marignal(tab, idxs)
			p1 = p1*partp1
			# p0
			interm[idx] = 0
			idxs = interm[vlabel]
			partp0 = marignal(tab, idxs)
			p0 = p0*partp0
	p = p1/(p1+p0)
	return p

def Gibbs_Sampling(factors, samples, dims, Z):
	x = np.zeros((samples+50, dims))
	term = np.random.binomial(1, 0.5, dims)# initialize one sample

	for i in range(samples+50):
		for idx in range(dims):
			p = get_prob(factors, term, idx, dims)
			ele = np.random.binomial(1, p, 1)  # only 0,1 two states can be right
			term[idx] = ele[0]
		x[i,:] = term

	return x[50:,:] 

def GF_SVGD_Estimate(cMRF, factors, paras, samples, dims, Z):
	x = np.random.normal(0.5, 0.2, [samples, dims])
	x = torch.from_numpy(x).float() # continue x
	x = update(x, cMRF, n_iter=10000, stepsize=0.0001)
	x = cMRF.discrete(x)
	return x.numpy()

def Exact_Mean(factors, dims):
	probs = np.zeros((np.power(2,dims), dims))
	elements = gen_value(dims)
	for e in range(elements.shape[0]):
		p=1
		for factor in factors:
			var = factor.v
			vlabel = [v.label for v in var]
			tab = factor.t
			idxs = elements[e,:][vlabel]
			partp = marignal(tab, idxs)
			p = p*partp
		probs[e,:] = p
	Z = np.sum(probs[:,0])
	probs = probs / Z
	print('probs:',probs)
	print('Z:',Z)
	mean_value = np.sum(np.multiply(elements, probs),0)

	return mean_value, Z

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Discrete MRF')
	parser.add_argument('--file',default='uai1')
	args = parser.parse_args()
	file = args.file
	factors, dims = readUai(file + '.txt')
	paras = convert(factors)
	# print('paras:',paras)
	num_samples = [2,4,8,16,32,64,128]
	if os.path.exists(file + '_Z.mat'):
		print("Loading Mean and Normalizer Z")
		mean_value = scio.loadmat(file + '_mean_value.mat')['mean']
		Z = scio.loadmat(file + '_Z.mat')['Z']	
	else:
		print("Calculate ture Mean and Normalizer Z")
		mean_value, Z = Exact_Mean(factors, dims)
		scio.savemat(file + '_mean_value.mat', {'mean':mean_value})
		scio.savemat(file + '_Z.mat', {'Z':Z})
	print('True Mean is:',mean_value)

	MSES_Gibbs = []
	MSES_GFSVGD = []
	for samples in num_samples:
		print('#samples=' + str(samples))
		MSE_Gibbs = 0
		for j in range(1000):
			x_Gibbs = Gibbs_Sampling(factors, samples, dims, Z)
			MSE_Gibbs += np.mean((np.mean(x_Gibbs, 0) - mean_value)**2)/1000
		MSES_Gibbs.append(MSE_Gibbs)
		print('MSE_Gibbs:', MSE_Gibbs)

		MSE_GFSVGD = 0
		cMRF = CMRF(factors, paras, samples, dims, Z)
		cMRF.extend()
		times = 1
		for j in range(times):
			x_GFSVGD = GF_SVGD_Estimate(cMRF, factors, paras, samples, dims, Z)
			MSE_GFSVGD += np.mean((np.mean(x_GFSVGD, 0) - mean_value)**2)/times
		MSES_GFSVGD.append(MSE_GFSVGD)
		print('MSE_GFSVGD:', MSE_GFSVGD)

	MSES_Gibbs = np.log10(np.array(MSES_Gibbs))
	MSES_GFSVGD = np.log10(np.array(MSES_GFSVGD))
	scio.savemat(file + '_gibbs.mat', {'gibbs':MSES_Gibbs})
	scio.savemat(file + '_GFSVGD.mat', {'GFSVGD':MSES_GFSVGD})

	# MSES_Gibbs = scio.loadmat(file + '_gibbs.mat')['gibbs']
	# MSES_GFSVGD = scio.loadmat(file + '_GFSVGD.mat')['GFSVGD']	
	# paint(MSES_Gibbs, MSES_GFSVGD, 7)
