import argparse
import multiprocessing
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm
import random
import json
import time
import sys
sys.path.append('../')

from alg import (
forward_push,
power_iteration,
heavy_ball,
nag,
power_push,
forward_push_sor,
power_push_sor)

parser = argparse.ArgumentParser()
parser.add_argument('-d',type=str)
parser.add_argument('-a',type=float)

args = parser.parse_args()
dataset= args.d
alpha=args.a
methods:dict={}
if dataset in ['dblp','orkut','products']:
    methods:dict={
                'forward_push':forward_push,
                'power_iteration':power_iteration,
                'heavy_ball':heavy_ball,
                'nag':nag,
                'power_push':power_push,
                'forward_push_sor':forward_push_sor,
                'power_push_sor':power_push_sor
                }
elif dataset in ['lj','pokec','webs']:
    methods:dict={
                
                'power_push':power_push,
                'forward_push':forward_push,
                'power_iteration':power_iteration,
                'forward_push_sor':forward_push_sor,
                'power_push_sor':power_push_sor,
                }

csr='error! please check your dataset name'
if dataset=='dblp':
    csr='../datasets/undirected_snap-com-dblp_csr-mat.npz'
elif dataset=='orkut':
    csr='../datasets/undirected_orkut_csr_matrix.npz'
elif dataset=='products':
    csr='../datasets/undirected_products_csr_matrix.npz'
elif dataset=='lj':
    csr='../datasets/directed_lj_csr_matrix.npz'
elif dataset=='pokec':
    csr='../datasets/directed_pokec_csr_matrix.npz'
elif dataset=='webs':
    csr='../datasets/directed_web-stanford_csr_matrix.npz'

adj_m=sp.load_npz(csr)

def single_ppr(para):
    dataset,adj_m,alpha,method,s,eps=para
    ppr_method=methods[method]
    start=time.time()
    ppr,num_oper_list,l1_error=ppr_method(adj_m,s,eps,1-alpha)
    num_time=time.time()-start
    points_list=[]
    for i in range(len(num_oper_list)):
        cur_time=num_oper_list[i]/num_oper_list[-1]*num_time
        points_list.append((num_oper_list[i],cur_time,l1_error[i]))
    with open(f'./result/{dataset}_{alpha}.json','a') as f:
        json.dump({method:points_list},f)
        f.write('\n')

random.seed(0)
n=adj_m.shape[0]
para_space=[]
nodes_num=50
random_nodes=random.sample(range(n), nodes_num)
for method in methods.keys():
    for node in random_nodes:
        if method=='power_push_sor':
            EPS=1e-5
        else:
            EPS=1e-8
        para_space.append((dataset,adj_m,alpha,method,node,EPS))

if __name__ == '__main__':
    cpu_num=60
    pool = multiprocessing.Pool(cpu_num)
    for _ in tqdm(pool.imap_unordered(single_ppr, para_space),total=len(para_space)):
        pass
    pool.close() 
    pool.join()