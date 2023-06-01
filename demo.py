from alg import (forward_push_mean,
forward_push,
power_iteration,
heavy_ball,
nag,
power_push,
forward_push_sor,
power_push_sor)

import scipy.sparse as sp
import numpy as np

method:dict={'forward_push':forward_push,
             'forward_push_mean':forward_push_mean,
             'power_iteration':power_iteration,
             'power_push':power_push,
             'forward_push_sor':forward_push_sor,
             'power_push_sor':power_push_sor,
             'heavy_ball':heavy_ball,
             'nag':nag
             }
             
node_id=0
alpha=0.15
adj_matrix=sp.load_npz('./datasets/directed_web-stanford_csr_matrix.npz')

true_ppr=power_iteration(adj_matrix,node_id,1e-10,1-alpha)[0]
for k,v in method.items():
    eps=1e-8
    ppr,_,_=v(adj_matrix,node_id,eps,1-alpha)
    print(k,sum(np.abs(ppr-true_ppr)))