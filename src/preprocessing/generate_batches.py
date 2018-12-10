import sys
sys.path.append('/Users/gauravsc/Code/seq-to-tree/src')

import json
from utils.mesh_operations import *
from utils.mesh_tree_operations import *


max_seq_len = 20
batch_size = 200


# generatie batches and write them to a file
def generate_batches():

	with open('../data/bioasq_dataset/toyMeSH_2017.json', 'r') as f:
		training_docs = json.load(f)['articles']
	
	G = create_ontology_tree()
	
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	name_to_mesh_id = mesh_name_to_id_mapping()	

	abs_batch = []
	tgt_batch = []
	masks_batch = []
	labels_batch = [] 
	for i, doc in enumerate(training_docs):
		abstract = doc['abstractText']
		mesh_labels = [name_to_mesh_id[mesh_name] for mesh_name in doc['meshMajor']]

		# remove mesh terms whose children have alread been included
		mesh_labels_cleaned = remove_redundant_mesh_terms(G, mesh_labels)

		targets, masks = get_masking_info(G, mesh_labels_cleaned, mesh_to_idx, max_seq_len) 

		abs_batch.append(abstract)
		tgt_batch.append(targets)
		masks_batch.append(masks)
		labels_batch.append(mesh_labels)

		# write the dara into file
		if (i+1) % batch_size == 0:
			with open('../data/bioasq_dataset/train_batches/'+str((i+1)//batch_size)+'.json','w') as f:
				data = {'abs': abs_batch, 'tgt':tgt_batch, 'mask': masks_batch, 'labels':labels_batch}
				json.dump(data, f)
				print (i, "  abstracts written: ", len(abs_batch))
				abs_batch = []
				tgt_batch = []
				masks_batch = []
				labels_batch = []

	# write the left over data into file
	if len(abs_batch) > 0:
		with open('../data/bioasq_dataset/train_batches/'+str(((i+1)//batch_size) + 1) +'.json','w') as f:
			data = {'abs': abs_batch, 'tgt':tgt_batch, 'mask': masks_batch, 'labels':labels_batch}
			json.dump(data, f)
			print ("abstracts written: ", len(abs_batch))



if __name__ == '__main__':
	generate_batches()
