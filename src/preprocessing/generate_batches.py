import sys
import os
import json
import pickle
from utils.mesh_operations import *
from utils.mesh_tree_operations import *

# Some global variables
max_seq_len = 20
file_size = 5000
max_batches_in_dir = 60000
n_docs_training = 12000000

# write to disk all files at once
# def write_files_to_disk(record_list, folder_names, file_names):
# 	for i, record in enumerate(record_list):
# 		folder = folder_names[i]
# 		file = file_names[i]
# 		with open('../data/bioasq_dataset/train_batches/'+folder+"/"+file+'.json','w') as f:
# 			json.dump(record, f)

# 	return None

# generate and pre-store path to every node from the root
def all_paths_from_root(G):
	paths_from_root = {}
	mesh_labels = list(G.nodes())
	
	leaf_nodes = [x for x in mesh_labels if G.out_degree(x)==0]

	for node in leaf_nodes:
		path = get_path_from_root(G, node)
		
		for i in range(1, len(path)):
			paths_from_root[path[i]] = path[0:i+1]

		# paths_from_root[mesh_label] = get_path_from_root(G, mesh_label)

	return paths_from_root


# generatie batches and write them to a file
def generate_batches():

	with open('../data/bioasq_dataset/allMeSH_2017.json', 'r', encoding="utf8", errors='ignore') as f:
		training_docs = json.load(f)['articles']
		# training_docs = training_docs
	
	G = create_ontology_tree()
	
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	name_to_mesh_id = mesh_name_to_id_mapping()	

	# get all paths from root to all nodes
	if os.path.isfile('../data/paths_from_root.pkl'):
		paths_from_root = pickle.load(open('../data/paths_from_root.pkl','rb'))
	else:
		paths_from_root = all_paths_from_root(G)
		pickle.dump(paths_from_root, open('../data/paths_from_root.pkl','wb'))

	print ("Done loading paths to every node from the root node in the Ontology")

	abs_batch = []
	tgt_batch = []
	masks_batch = []
	labels_batch = [] 
	
	for i, doc in enumerate(training_docs):
		abstract = doc['abstractText']
		mesh_labels = [name_to_mesh_id[mesh_name] for mesh_name in doc['meshMajor']]

		# remove mesh terms whose children have alread been included
		mesh_labels_cleaned = remove_redundant_mesh_terms(G, mesh_labels, paths_from_root)

		targets, masks = get_masking_info(G, mesh_labels_cleaned, mesh_to_idx, max_seq_len, paths_from_root) 

		abs_batch.append(abstract)
		tgt_batch.append(targets)
		masks_batch.append(masks)
		labels_batch.append(mesh_labels)

		# folder on which to save the files
		folder = str(i//(file_size*max_batches_in_dir))

		# write the dara into file
		if (i+1) % file_size == 0:
			with open('../data/bioasq_dataset/train_batches/'+str((i+1)//file_size)+'.json','w') as f:
				data = {'abs': abs_batch, 'tgt':tgt_batch, 'mask': masks_batch, 'labels':labels_batch}
				json.dump(data, f)
				print (i, "  abstracts written: ", len(abs_batch))
				abs_batch = []
				tgt_batch = []
				masks_batch = []
				labels_batch = []
			# record_list.append(data)
			# file_names.append(str((i+1)//file_size))
			# folder_names.append(folder)

		# if (i+1)%10 == 0:
		# 	write_files_to_disk(record_list, folder_names, file_names)
		# 	record_list = []
		# 	file_names = []
		# 	folder_names = []

	# write the left over data into file
	if len(abs_batch) > 0:
		with open('../data/bioasq_dataset/train_batches/'+str(((i+1)//file_size) + 1) +'.json','w') as f:
			data = {'abs': abs_batch, 'tgt':tgt_batch, 'mask': masks_batch, 'labels':labels_batch}
			json.dump(data, f)
			print ("abstracts written: ", len(abs_batch))


# def generate_train_batch_folders():
# 	n_dirs_needed = n_docs_training//(max_batches_in_dir*file_size)
# 	# create folders to store the batches
# 	for i in range(n_dirs_needed+1):
# 		# Path to be created
# 		path = "../data/bioasq_dataset/train_batches/" + str(i)
# 		if not os.path.exists(path):
# 			os.mkdir(path)


if __name__ == '__main__':
	# generate_train_batch_folders()
	generate_batches()
