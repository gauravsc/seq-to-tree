import json
import pickle
from collections import defaultdict
import os

# read the mesh to index files
with open('../data/mesh_to_idx.pkl', 'rb') as fread:
	mesh_to_idx = pickle.load(fread)

# mesh vocab i.e. idx to mesh mapping
mesh_vocab = [" "] * len(mesh_to_idx)
for mesh, idx in mesh_to_idx.items():
	mesh_vocab[idx] = mesh




# getting the training files	
files = os.listdir('../data/bioasq_dataset/train_batches/')

inverted_idx = defaultdict(set)
list_of_abstracts = []
ndoc = 0
for fi in files:
	with open('../data/bioasq_dataset/train_batches/'+fi, 'r', encoding="utf8", errors='ignore') as f:
		data = json.load(f)
	
	abstracts = data['abs']
	list_of_labels = data['labels']
	
	n_abstracts = len(abstracts)
	for i in range(n_abstracts):
		list_of_abstracts.append(abstracts[i])
		print ("# of labels : ",len(list_of_labels[i]))
		for label in list_of_labels[i]:
			inverted_idx[label].add(ndoc)
		ndoc += 1




print (len(list_of_abstracts))
print (inverted_idx)






