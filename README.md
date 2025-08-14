# heat_kernel_package

**Heat kernel and diffusion distance computations on graphs**

This Python package provides tools to compute heat kernels, diffusion distance matrices, and related matrices (`Q_w`) on graphs, with support for large graphs using truncated eigenpair computations.

---

## Features

- Convert a NetworkX graph into a PyGSP graph with Laplacian.
- Compute heat kernels for multiple diffusion times.
- Efficiently compute heat kernels for large graphs using only the top-k eigenpairs.
- Compute diffusion distance matrices from heat kernels.
- Compute `Q_w` matrices for each vertex.
- Easy-to-use API for research and experimentation in graph analysis, signal processing on graphs, and network diffusion studies.

---

## Installation

### From GitHub (public repository)
```bash
pip install git+https://github.com/ajmaths/heat_kernel.git
```
##Requirements

-Python ≥ 3.7
-numpy
-scipy
-networkx
-matplotlib
-pygsp

##Usage
```
import networkx as nx
from heat_kernel_package import (
    create_pygsp_graph,
    heat_kernels_topk,
    diffusion_distance_matrices,
    Q_matrices
)

# Example: create a graph
G_n = nx.path_graph(10)

# Convert to PyGSP graph
G = create_pygsp_graph(G_n)

# Set diffusion times
t_values = [0.1, 0.5, 1.0]

# Compute heat kernels using top-k eigenpairs
hk_dict = heat_kernels_topk(G, t_values, k=5)

# Compute diffusion distance matrices
D_dict = diffusion_distance_matrices(hk_dict)

# Compute Q_w matrices for each t
for t in t_values:
    D = D_dict[t]
    Q_list = Q_matrices(D)
    print(f"t = {t}, Q_matrices count: {len(Q_list)}")
```
##Contributing

Contributions and improvements are welcome! Please submit pull requests or issues on GitHub.

