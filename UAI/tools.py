import numpy as np 
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio

def paint(MSE_Gibbs, MSE_GFSVGD, log2_samples):
    plt.figure(figsize=(8,6))
    x = np.linspace(1,log2_samples,log2_samples).astype(int)
    plt.plot(x, MSE_GFSVGD, 'g--o', label='Gradient Free SVGD',linewidth=3, markersize=18)
    plt.plot(x, MSE_Gibbs, 'r--s', label='Gibbs Sampling',linewidth=3, markersize=18)
    plt.xlabel('#samples',fontsize=30)
    plt.ylabel('log10 MSE',fontsize=30)
    # plt.xlim([1,7])
    plt.xticks([1, 4, 7], ['1', '10','100'],fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=16)
    # plt.savefig('GFSVGD.pdf')
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Discrete MRF')
    parser.add_argument('--file',default='test')
    args = parser.parse_args()
    file = args.file

    MSES_Gibbs = scio.loadmat(file + '_gibbs.mat')['gibbs'].squeeze()
    MSES_GFSVGD = scio.loadmat(file + '_GFSVGD.mat')['GFSVGD'].squeeze()
    paint(MSES_Gibbs, MSES_GFSVGD, 7)