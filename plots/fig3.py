import numpy as np
import time
from scipy.sparse import csr_matrix
from algo import _forward_push_dummy
from algo import _forward_push_mean
import pickle as pkl
import multiprocessing

def single_run(para):
    alpha, num_nodes, eps, dataset, seed = para
    dblp = np.load(f'../datasets/{dataset}')
    dblp_a = csr_matrix((dblp['data'], dblp['indices'], dblp['indptr']), shape=dblp['shape'])
    degree = dblp_a.sum(1).A.flatten()

    indices = dblp_a.indices
    indptr = dblp_a.indptr
    n, _ = dblp_a.shape

    y1 = []
    x1 = []
    y2 = []
    x2 = []
    np.random.seed(seed)
    nodes = np.random.permutation(n)[:num_nodes]
    for s in nodes:
        start_time = time.time()
        re = _forward_push_mean(indptr, indices, degree, alpha, eps, s, debug=False)
        run_time = time.time() - start_time
        freq_com, num_oper2, sum_r, epoch_num = re[:n], re[n], re[n + 1], re[n + 2]
        y2.append(num_oper2)
        x2.append(run_time)
        print(f"finish in {run_time:.3e} with operations: "
              f"{num_oper2:.0f} r-sum:{sum_r:.3e} ep-num: {epoch_num:.0f}")
        start_time = time.time()
        re = _forward_push_dummy(indptr, indices, degree, alpha, eps, s)
        run_time = time.time() - start_time
        freq_com, num_oper1, sum_r, epoch_num = re[:n], re[n], re[n + 1], re[n + 2]
        print(f"finish in {run_time:.3e} with operations: "
              f"{num_oper1:.0f} r-sum:{sum_r:.3e} ep-num: {epoch_num:.0f} "
              f"{((num_oper1 - num_oper2) / num_oper1) * 100}")
        y1.append(num_oper1)
        x1.append(run_time)
    return dataset, seed, x1, y1, x2, y2

def test_ppr_mean():
    num_cpus = 60
    # input parameters
    eps = np.float64(1e-6)
    para_space = []
    num_nodes = 1000
    alpha = 0.2
    seed = 0
    for dataset in ['directed_lj_csr_matrix.npz', 
                    'directed_pokec_csr_matrix.npz',
                    'directed_web-stanford_csr_matrix.npz', 
                    'undirected_dblp_csr_matrix.npz',
                    'undirected_orkut_csr_matrix.npz', 
                    'undirected_products_csr_matrix.npz'
                    ]:
        para = (alpha, num_nodes, eps, dataset, seed)
        para_space.append(para)
        seed += 1
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=single_run, iterable=para_space)
    pool.close()
    pool.join()
    pkl.dump(results, open(f'results/forward_push_mean.pkl', 'wb'))
    
def draw_figure_ppr_mean():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rc('font', family='serif', serif='Times New Roman')
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 20}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    sns.set()
    sns.set_theme(style='white')
    clrs = sns.color_palette()
    colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    # lines=['dashed','dashdot',(0, (1, 1)),(0, (3, 2, 1, 5)),(0, (5, 10)),(0, (3, 5, 1, 5, 1, 5))]
    lines={'undirected_dblp_csr_matrix.npz':'dashed',
          'directed_web-stanford_csr_matrix.npz': (0, (3, 1, 1, 1)),
          'directed_lj_csr_matrix.npz':(0, (5, 1)),
          'undirected_products_csr_matrix.npz':'dotted',
          'directed_pokec_csr_matrix.npz':(0, (3, 2, 1, 5)),
          'undirected_orkut_csr_matrix.npz':(0, (3, 1, 1, 1, 1, 1))}


    dataset_list = ['directed_lj_csr_matrix.npz',   
                    'directed_pokec_csr_matrix.npz',
                    'directed_web-stanford_csr_matrix.npz', 'undirected_dblp_csr_matrix.npz',
                    'undirected_orkut_csr_matrix.npz', 'undirected_products_csr_matrix.npz']
    dataset_label = ["livejournal", "pokec", "web-Standford", "dblp", "orkut", "products"]
    results = pkl.load(open(f'results/forward_push_mean.pkl', 'rb'))
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    import matplotlib.ticker as mtick

    for dataset, seed, rt_1, operations_1, rt_2, operations_2 in results:
        ii = dataset_list.index(dataset)
        operations_1 = np.asarray(operations_1)
        operations_2 = np.asarray(operations_2)
        y = ((operations_1 - operations_2) / operations_1) * 100.
        ax.plot(sorted(y)[50:], label=dataset_label[ii],
                linewidth=2.8,color=colors[ii],linestyle=lines
                [dataset])

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim([0, 50])
    ax.set_xlabel(r"Nodes $s$", fontsize=18)
    ax.set_ylabel("Percentage of operations reduced", fontsize=18)
    ax.legend(loc='center right', fontsize=13.5,
                    bbox_to_anchor=(1, 1.1),ncols=3,frameon=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax.grid()
    plt.show()
    fig.savefig(f"new_figs/fig-ppr-mean-operations1.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0, format='pdf')

    plt.close()
if __name__=='__main__':
test_ppr_mean()
draw_figure_ppr_mean()