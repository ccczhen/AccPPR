import sys
sys.path.append('../')
from alg import forward_push
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
csrs={
    'webs':'../datasets/directed_web-stanford_csr_matrix.npz',
    'lj':'../datasets/directed_lj_csr_matrix.npz',
    'pokec':'../datasets/directed_pokec_csr_matrix.npz'
}
indexes=['A','B','C']
datasets=['webs','lj','pokec']
alpha=.15
fig6={}
s=1000
for i,dataset in tqdm(enumerate(datasets),total=3):
    adj_m=sp.load_npz(csrs[dataset])
    eps=1e-3
    ppr,num_oper_list,l1_error=forward_push(adj_m,s,eps,1-alpha)
    fig6[dataset]={'oper':num_oper_list,
    'error':l1_error}

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fontsize=25
plt.rc('xtick',labelsize=15)

datasets=['webs','lj','pokec']
names={'webs':'web-Stanford',
'lj':'livejournal',
'pokec':'pokec'}

for i,dataset in enumerate(datasets):
    cur=fig6[dataset]
    num_oper_list=cur['oper']
    l1_error=cur['error']
    n=len(num_oper_list)
    print(n)

    index=np.linspace(0.1*n,0.9*n,100,dtype=int)
    axs[i].plot(num_oper_list[index],np.log10(l1_error[index]),linestyle='dotted',linewidth=5)
    axs[i].set_ylim(-3.1,-.5)
    if i==0:
        axs[i].set_yticks([-1,-2,-3,],[r'$10^{-1}$',r'$10^{-2}}$',r'$10^{-3}$'],fontsize=fontsize)
        axs[i].set_ylabel(r'$\Vert x^t-x^*\Vert$',fontsize=fontsize)
    else:
        axs[i].set_yticks([-1,-2,-3,],[],fontsize=fontsize)
    axs[i].set_xlabel(r'#updates',fontsize=20)
    
    axs[i].grid()
    axs[i].set_title(f'({indexes[i]}) {names[dataset]}',y=-0.3,fontsize=fontsize,
            fontfamily="Times New Roman"
            )
fig.savefig(f'figs/linear.pdf', dpi=600, bbox_inches='tight', pad_inches=0,format='pdf')