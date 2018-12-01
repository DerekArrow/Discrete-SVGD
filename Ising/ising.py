import numpy as np
import numpy.matlib as nm
from GFsvgd import SVGD
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from numpy.linalg import inv, eig
import scipy.io as scio

def gibbs(n, p, dim):
    samples = np.zeros((n+50, dim))
    x0 = np.random.binomial(1, p[0], dim)

    for i in range(n+50):
        for idx in range(dim):
            x = np.random.binomial(1, p[idx], 1)
            x0[idx] = x[0]
        samples[i, :] = x0

    return samples[50:,:]

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
        # value = self.gen_value()
        # pdf = st.multivariate_normal.pdf(value, self.mu, inv(self.A)) 
        # self.muTrue = np.sum(np.multiply(value, nm.repmat(pdf[:,np.newaxis], 1, value.shape[1])), 0)
        # self.pTrue = (self.muTrue+1)/2
        self.muTrue = np.zeros(mu.shape[0])
        self.pTrue = self.muTrue + 0.5
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)-theta
    
    def cal_w_s(self, x0):
        w_s = np.ones(x0.shape[0])
        for i in range(x0.shape[0]):
            phat = np.exp(np.matmul(np.matmul(x0[i,:]-self.mu, self.A), x0[i,:]-self.mu))
            p    = np.exp(np.matmul(np.matmul(np.sign(x0[i,:])-self.mu, self.A), np.sign(x0[i,:])-self.mu))
            w_s[i] = 1.0*phat/p

        w_s = (w_s/np.sum(w_s)).squeeze()
        return w_s

    def gen_value(self):
        dim = self.mu.shape[0]
        value = []
        def ten2two(x):
            a=[]
            for i in bin(x)[2:]:
                a.append(2*int(i)-1)
            a = [ -1 for _ in range(dim - len(a)) ] + a
            return np.array(a)

        for k in range(np.power(2,dim)):
            value.append(ten2two(k))
        return np.array(value)

    
if __name__ == '__main__':
    dim = 10
    y = np.linspace(1,7,7).astype(int)
    size = np.power(2,y).astype(int)

    A = np.ones((dim,dim))*0.0001
    A = (A + A.T)/2
    for i in range(dim):
        A[i][i] = 0.003 + np.sum(A[i,:]) - A[i][i]
    mu = np.zeros(dim)

    DataGuassian = np.random.multivariate_normal(mu, inv(A), 1000)
    if mu.shape[0]==2:
        DataGuassian = pd.DataFrame(DataGuassian, columns=["X", "Y"])

    model = MVN(mu, A)
    # print('pTrue:',model.pTrue)
    # print('muTrue',model.muTrue)
    ising = []
    ising2 = []
    ising3 = []
    
    for n in size:
        print('n=',n)
        # GF-SVGD
        theta1 = 0
        for j in range(30):
            x0 = np.random.normal(5,10,[n,dim])
            theta = SVGD().update(model, x0, n_iter=2000, stepsize=0.1)
            theta1 += np.mean((np.mean(theta,0)-model.muTrue)**2)/30

        ising.append(theta1)
        # Monte Carlo
        theta2 = 0
        for j in range(1000):
            for k in range(dim):
                theta = np.random.binomial(1, model.pTrue[k], n)
                theta[np.where(theta==0)]=-1
                theta2 += ((np.mean(theta)-model.muTrue[k])**2)/(1000*dim)

        ising2.append(theta2)
        # Gibbs Sampling
        theta3 = 0
        for j in range(1000):
            theta = gibbs(n, model.pTrue, dim)
            theta[np.where(theta==0)]=-1
            theta3 += np.mean((np.mean(theta,0)-model.muTrue)**2)/1000

        ising3.append(theta3)

        print('GF-SVGD',theta1)
        print('Monte-Carlo',theta2)
        print('Gibbs',theta3)
    
    ising = np.log10(np.array(ising))
    ising2 = np.log10(np.array(ising2))
    ising3 = np.log10(np.array(ising3))
    # scio.savemat('ising.mat', {'ising':ising})
    # scio.savemat('ising2.mat', {'ising2':ising2})
    # scio.savemat('ising3.mat', {'ising3':ising3})
    # ising = scio.loadmat('ising.mat')['ising'].squeeze()
    # ising2 = scio.loadmat('ising2.mat')['ising2'].squeeze()
    # ising3 = scio.loadmat('ising3.mat')['ising3'].squeeze()
    plt.figure(figsize=(8,6))
    x = np.linspace(1,7,7).astype(int)
    plt.plot(x, ising2, 'r-s', label='Exact Monte Carlo',linewidth=3, markersize=18)
    plt.plot(x, ising3, 'b-D', label='Gibbs Sampling',linewidth=3, markersize=18)
    plt.plot(x, ising, 'g-o', label='Gradient Free SVGD',linewidth=3, markersize=18)
    plt.xlabel('#samples',fontsize=30)
    plt.ylabel('log10 MSE',fontsize=30)
    # plt.xlim([1,7])
    plt.xticks([1, 4, 7], ['1', '10','100'],fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=16)
    plt.savefig('ising.pdf')
    plt.show()
    # scio.savemat('ising.mat', {'ising':ising})
    # scio.savemat('ising2.mat', {'ising2':ising2})
    # scio.savemat('ising3.mat', {'ising3':ising3})
