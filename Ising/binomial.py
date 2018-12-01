import numpy as np
import numpy.matlib as nm
from GFsvgd import SVGD
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from numpy.linalg import inv
import scipy.io as scio

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
        self.f1 = np.exp(np.multiply(-A/2,(1-mu)**2))
        self.f_1 = np.exp(np.multiply(-A/2,(-1-mu)**2))
        self.pTrue = self.f1/(self.f1+self.f_1)
        self.muTrue = 2*self.pTrue-1
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), np.diag(self.A))-theta
    
    def cal_w_s(self, x0):
        w_s = np.ones(x0.shape[0])
        for i in range(x0.shape[0]):
            phat = np.exp(np.matmul(np.matmul(x0[i,:]-self.mu, np.diag(self.A)), x0[i,:]-self.mu))
            p    = np.exp(np.matmul(np.matmul(np.sign(x0[i,:])-self.mu, np.diag(self.A)), np.sign(x0[i,:])-self.mu))
            w_s[i] = 1.0*phat/p

        w_s = (w_s/np.sum(w_s)).squeeze()
        return w_s
    
if __name__ == '__main__':
    dim = 2
    y = np.linspace(1,7,7).astype(int)
    size = np.power(2,y).astype(int)

    A = np.ones(dim)/40.0
    mu = np.zeros(dim)+6

    DataGuassian = np.random.multivariate_normal(mu, np.diag(1/A), 1000)
    if mu.shape[0]==2:
        DataGuassian = pd.DataFrame(DataGuassian, columns=["X", "Y"])

    model = MVN(mu, A)
    print('pTrue:',model.pTrue)
    print('muTrue',model.muTrue)
    errors = []
    errors2 = []

    for n in size:
        print('n=',n)
        theta1 = 0
        for j in range(30):
            x0 = np.random.normal(10,30,[n,dim])
            theta = SVGD().update(model, x0, n_iter=2000, stepsize=0.1)
            theta1 += np.mean((np.mean(theta,0)-model.muTrue)**2)/30

        errors.append(theta1)

        theta2 = 0
        for j in range(1000):
            for k in range(dim):
                theta = np.random.binomial(1, model.pTrue[k], n)
                theta[np.where(theta==0)]=-1
                theta2 += ((np.mean(theta)-model.muTrue[k])**2)/(1000*dim)

        errors2.append(theta2)
        print('GF-SVGD',theta1)
        print('Monte-Carlo',theta2)
    
    errors = np.log(np.array(errors))
    errors2 = np.log(np.array(errors2))
    # scio.savemat('errors.mat', {'errors':errors})
    # scio.savemat('errors2.mat', {'errors2':errors2})
    # errors = scio.loadmat('errors.mat')['errors'].squeeze()
    # errors2 = scio.loadmat('errors2.mat')['errors2'].squeeze()
    plt.figure(figsize=(8,6))
    x = np.linspace(1,7,7).astype(int)
    plt.plot(x, errors, 'g--o', label='Gradient Free SVGD',linewidth=3, markersize=18)
    plt.plot(x, errors2, 'r--s', label='Exact Monte Carlo',linewidth=3, markersize=18)
    plt.xlabel('#samples',fontsize=30)
    plt.ylabel('log10 MSE',fontsize=30)
    # plt.xlim([1,7])
    plt.xticks([1, 4, 7], ['1', '10','100'],fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=16)
    # plt.savefig('GFSVGD.pdf')
    plt.show()
    # scio.savemat('errors.mat', {'errors':errors})
    # scio.savemat('errors2.mat', {'errors2':errors2})
        
        
