import networkx as nx
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
import pygsp as pg


def create_pygsp_graph(G_n):
    """Convert a NetworkX graph to a PyGSP graph and compute its Laplacian."""
    adjacency_matrix = nx.to_numpy_array(G_n)
    G = pg.graphs.Graph(adjacency_matrix)
    G.compute_laplacian()
    return G


def heat_kernels_topk(graph, t_values, k=50):
    """
    Compute heat kernels using only the largest k eigenpairs.

    Parameters:
        graph (pg.graphs.Graph): PyGSP graph
        t_values (list): diffusion times
        k (int): number of largest eigenvalues to compute
    """
    # Convert Laplacian to sparse format
    L_sparse = sp.csr_matrix(graph.L)

    # Get k largest eigenvalues/eigenvectors
    evals, evecs = eigsh(L_sparse, k=k, which='LA')

    kernels = {}
    for t in t_values:
        exp_evals = np.exp(-t * evals)
        kernels[t] = evecs @ np.diag(exp_evals) @ evecs.T
    return kernels


def diffusion_distance_matrices(hk_dict):
    """
    Compute diffusion distance matrices for each t.

    Parameters:
        hk_dict (dict): Heat kernel matrices keyed by t.

    Returns:
        dict: Diffusion distance matrices for each t.
    """
    D_dict = {}
    for t, hk in hk_dict.items():
        n = hk.shape[0]
        D = np.zeros((n, n))
        for u in range(n):
            for v in range(n):
                diff = hk[u, :] - hk[v, :]
                D[u, v] = 0.5 * (np.sum(diff ** 2) / n)
        D_dict[t] = D
    return D_dict


def Q_matrices(D):
    """
    Compute Q_w matrices from a diffusion distance matrix.

    Parameters:
        D (np.ndarray): Diffusion distance matrix.

    Returns:
        list: List of Q_w matrices for each vertex w.
    """
    n = D.shape[0]
    Q_list = []
    for w in range(n):
        col_w = D[:, w].reshape(-1, 1)
        Q_w = col_w @ col_w.T
        Q_list.append(Q_w)
    return Q_list
