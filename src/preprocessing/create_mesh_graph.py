import networkx as nx
import pickle

# read the mesh parent to child file and create the graph
file = '../../data/bioasq_dataset/MeSH_parent_child_mapping_2017.txt'
with open(file, 'r') as fread:
	lines = fread.readlines()

lines = [line.strip() for line in lines]
edges = [line.split(' ') for line in lines]

nodes = sorted(list(set([node for edge in edges for node in edge])))
mesh_to_idx = {k: v for v, k in enumerate(nodes)}
edges = [[mesh_to_idx[edge[0]], mesh_to_idx[edge[1]]] for edge in edges]


G = nx.DiGraph()
G.add_edges_from(edges)
G = G.to_undirected()
with open("../../data/mesh_graph.adjlist",'wb') as fwrite:
	nx.write_adjlist(G, fwrite)

with open("../../data/mesh_to_idx.pkl", 'wb') as fwrite:
	pickle.dump(mesh_to_idx, fwrite)
