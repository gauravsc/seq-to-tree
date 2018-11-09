import json
from utils.mesh_operations import *
from utils.mesh_tree_operations import *


max_seq_len = 20
batch_size = 2

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
	for i, doc in enumerate(training_docs):
		abstract = doc['abstractText']
		mesh_labels = [name_to_mesh_id[mesh_name] for mesh_name in doc['meshMajor']]
		targets, masks = get_masking_info(G, mesh_labels, mesh_to_idx, max_seq_len) 

		abs_batch.append(abstract)
		tgt_batch.append(targets)
		masks_batch.append(masks)

		# write the dara into file
		if (i+1) % batch_size == 0:
			with open('../data/bioasq_dataset/train_batches/'+str((i+1)//batch_size)+'.json','w') as f:
				data = {'abs': abs_batch, 'tgt':tgt_batch, 'mask': masks_batch}
				json.dump(data, f)
				print (i, "  abstracts written: ", len(abs_batch))
				abs_batch = []
				tgt_batch = []
				masks_batch = []


	# write the left over data into file
	if len(abs_batch) > 0:
		with open('../data/bioasq_dataset/train_batches/'+str(((i+1)//batch_size) + 1) +'.json','w') as f:
			data = {'abs': abs_batch, 'tgt':tgt_batch, 'mask': masks_batch}
			json.dump(data, f)
			print ("abstracts written: ", len(abs_batch))



if __name__ == '__main__':
	generate_batches()
