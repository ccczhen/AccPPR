# AccPPR

## Overview

Welcome to the repository of Accelerating Personalized PageRank Vector Computation! We build this repository to make our proposed algorithm publicly available and reproduce main experimental results shown in our KDD2023 paper.

## Preparation

### Datasets 

We provide DBLP and Web-Stanford in ./dataset/. \
Products is available on https://ogb.stanford.edu/docs/nodeprop/#ogbn-products.
The rest datasets are available on https://snap.stanford.edu/data/. 

### Installation Instructions

Create a virtual environment and install the dependencies via the following command:

```shell
conda env create -f environment.yml
```

## Experiment Results

### Figure 2, Figure 3, Figure 6

To draw Figure2, Figure 3 and Figure 6, run the following code:

```shell
cd plots
python fig2.py
python fig3.py
python fig6.py
```

### Figure 4, Figure 5

First compute the ppr vectors, the time cost and the operation cost of each dataset:

```shell
cd plots
python cal_ppr.py -d dblp -a .2
```

The above code computes the ppr vectors of 50 random selected nodes in the dblp dataset when $\alpha$=0.2. To draw the plot, you should run the code on the total 6 datasets. Then run the following code:

```shell
python fig45.py
```

## Algorithms

We provide a demo to show how to use our proposed algorithms.

```shell
python demo.py
```

To use our algorithms on your own graph, you should provide the adjacent matrix of the graph in the form of [Compressed Sparse Row matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) and ensure that there is no self-loop in the graph, and each node has at least 1 out-degree.   
