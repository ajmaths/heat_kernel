import networkx as nx
from heat_kernel_package.heat_kernel import create_pygsp_graph, heat_kernels, diffusion_distance_matrices, Q_matrices

# Example graph
G_n = nx.path_graph(5)  # Change to your own graph

# Create PyGSP graph
G = create_pygsp_graph(G_n)

# Compute heat kernels
t_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
hk_dict = heat_kernels(G, t_values)

# Compute diffusion distance matrices
D_dict = diffusion_distance_matrices(hk_dict)

# Get Q_w matrices for each t
for t in t_values:
    D = D_dict[t]
    Q_list = Q_matrices(D)
    print(f"t = {t}, Q_matrices count: {len(Q_list)}")
