import networkx as nx
import collections
import pickle


# read the mesh parent to child file and create the graph
def create_ontology_tree(file='../data/bioasq_dataset/MeSH_parent_child_mapping_2017.txt'):
	with open(file, 'r') as fread:
		lines = fread.readlines()

	lines = [line.strip() for line in lines]
	edges = [line.split(' ') for line in lines]

	# created a sorted nodes list 
	nodes = sorted(list(set([node for edge in edges for node in edge])))
	# mesh_to_idx = {k: v for v, k in enumerate(nodes)}
	edges = [[edge[0], edge[1]] for edge in edges]

	# Create directed graph from the networkx library
	G = nx.DiGraph()
	G.add_edges_from(edges)
	
	return G

def get_path_from_root(G, node, root='MeSH'):
	path = list(nx.all_simple_paths(G, root, node))[0]
	return path


def get_masking_info(G, mesh_labels, mesh_to_idx, max_seq_len):
	mesh_seqs = []
	for mesh_label in mesh_labels:
		mesh_seqs.append(get_path_from_root(G, mesh_label))

	# with open('../../data/mesh_to_idx.pkl', 'rb') as fread:
	# 	mesh_to_idx = pickle.load(fread)

	node_desc = collections.defaultdict(list)
	for mesh_seq in mesh_seqs:
		for i in range(len(mesh_seq)-1):
			node_desc[mesh_to_idx[mesh_seq[i]]].append(mesh_to_idx[mesh_seq[i+1]])

	node_desc = dict(node_desc)
	
	mesh_idx_seqs = []
	for mesh_seq in mesh_seqs:
		mesh_idx_seqs.append([mesh_to_idx[mesh] for mesh in mesh_seq])

	
	target_list = []
	mask_list = []
	for j, mesh_idx_seq in enumerate(mesh_idx_seqs):
		target = [0]*max_seq_len
		mask = [[] for i in range(max_seq_len)]
		for i, mesh_idx in enumerate(mesh_idx_seq):
			target[i] = mesh_idx
			if i-1 >= 0:
				mask[i] += node_desc[mesh_idx_seq[i-1]]

		mask = [list(set(mask_i)) for mask_i in mask]
		mask = [[v for v in mask[i] if v!= target[i]] for i in range(len(mask))]

		target_list.append(target)
		mask_list.append(mask)

	return target_list, mask_list

def get_semantic_dist(undirG, mesh1, mesh2):
	try:
		return nx.shortest_path_length(G, mesh1, mesh2)
	except NetworkXNoPath:
		return sys.maxsize


def get_legitimate_active_children(G, mesh_term):
	all_children = list(G.successor(mesh_term))

	return all_children



