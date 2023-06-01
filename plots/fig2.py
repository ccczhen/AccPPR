import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix


from algo import _forward_push_dummy_bound
import pickle as pkl
import multiprocessing
def single_run_bound(para):
    alpha, num_nodes, dataset, seed = para
    dblp = np.load(f'../datasets/{dataset}')
    csr_mat = csr_matrix((dblp['data'], dblp['indices'], dblp['indptr']), shape=dblp['shape'])
    degree = csr_mat.sum(1).A.flatten()

    indices = csr_mat.indices
    indptr = csr_mat.indptr
    n, _ = csr_mat.shape
    m = len(csr_mat.data)

    np.random.seed(seed)
    nodes = np.random.permutation(n)[:num_nodes]
    start = np.log10(0.01 / m)
    end = np.log10(100 / m)
    eps_list = np.logspace(start, end, num=10, base=10.)
    list_bounds = []
    for eps in eps_list:
        bounds = []
        opers = []
        for s in nodes:
            results = _forward_push_dummy_bound(indptr, indices, degree, alpha, eps, s)
            sum_r, aver_active_vol, aver_ratio, t, estimate_t, num_operations, theoretical_bound, ppr_vec = results
            bounds.append(theoretical_bound)
            opers.append(num_operations)
        bound1 = (m / alpha) * np.log(1 / (eps * m)) + m
        bound2 = 1. / (eps * alpha)
        bound3 = np.mean(bounds)

        list_bounds.append([np.mean(opers), bound1, bound2, bound3])
        print(list_bounds[-1])
    return dataset, seed, n, m, eps_list, list_bounds

def test_ppr_dummy_bounds():
    num_cpus = 60
    # input parameters
    para_space = []
    num_nodes = 50
    alpha = 0.85
    seed = 0
    for dataset in ['undirected_dblp_csr_matrix.npz']:
        para = (alpha, num_nodes, dataset, seed)
        para_space.append(para)
        seed += 1
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=single_run_bound, iterable=para_space)
    pool.close()
    pool.join()
    pkl.dump(results, open(f'results/forward_theoretical_bounds-{alpha}.pkl', 'wb'))
    
def draw_figure_ppr_bounds():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rc('font', family='serif', serif='Times New Roman')
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 25}
    plt.rcParams["font.weight"] = "bold"
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = False
    plt.rc('text', usetex=False)
    sns.set()
    sns.set_theme(style='white')
    colors=['tab:blue','tab:orange','tab:green','tab:red']
    dataset_list = [
                    'undirected_dblp_csr_matrix.npz',
                    ]
    dataset_label = [
          "DBLP", 
          ]
    results1 = pkl.load(open(f'results/forward_theoretical_bounds-0.15.pkl', 'rb'))
    results2 = pkl.load(open(f'results/forward_theoretical_bounds-0.5.pkl', 'rb'))
    results3 = pkl.load(open(f'results/forward_theoretical_bounds-0.85.pkl', 'rb'))
    list_titles = [r"Real", r"$B_1$", r"$B_2$", r"Ours"]
    marker_list = ["s", "D", "^", 'o']
    list_results = [results1, results2, results3]
    for jj, data in enumerate(dataset_list):
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.3))
        for kk in range(3):
            for dataset, seed, n, m, eps_list, list_bounds in list_results[kk]:
                if data != dataset:
                    continue
                for qq, title in enumerate(list_titles):
                    yy = [_[qq] for _ in list_bounds if _[qq] > 0.]
                    ax[kk].loglog(eps_list[:len(yy)], yy, label=title, linewidth=1.5,markerfacecolor='white',fillstyle='none',
                                  marker=marker_list[qq],
                                   color=colors[qq],
                                     markersize=8.)
            ax[kk].set_xlabel(r"$\epsilon$", fontsize=25)

            ax[0].set_ylabel("Number of Operations", fontsize=25)
            ax[kk].tick_params(axis='both', which='major', labelsize=16)
            ax[kk].tick_params(axis='both', which='minor', labelsize=16)
            ax[kk].grid()

        ax[0].legend(fontsize=19)
        plt.subplots_adjust(wspace=0.15, hspace=0.1)
        plt.show()
        fig.savefig(f"figs/fig-ppr-operations-bounds-{dataset_label[jj]}.pdf", dpi=300,
                    bbox_inches='tight', pad_inches=0, format='pdf')
        plt.close()

if __name__ == '__main__':
    test_ppr_dummy_bounds()
    draw_figure_ppr_bounds()