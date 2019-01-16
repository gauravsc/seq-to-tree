import json
import pickle
from collections import defaultdict
import os

# # read the mesh to index files
# with open('../data/mesh_to_idx.pkl', 'rb') as fread:
# 	mesh_to_idx = pickle.load(fread)

# dictionary for keeping label counts
label_cnt = {}
# getting the training files	
files = os.listdir('../data/bioasq_dataset/train_batches/')
for fi in files:
	print ("File read:", fi)
	with open('../data/bioasq_dataset/train_batches/'+fi, 'r', encoding="utf8", errors='ignore') as f:
		data = json.load(f)
	
	abstracts = data['abs']
	list_of_labels = data['labels']
	n_abstracts = len(abstracts)

	for labels in list_of_labels:
		for label in labels:
			if label in label_cnt:
				label_cnt[label] += 1
			else:
				label_cnt[label] = 1

n_labels = sum(list(label_cnt.values()))

for label in label_cnt.keys():
	label_cnt[label] = label_cnt[label]/float(n_labels)


thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
cnts = np.array(list(label_cnt.values()))

for threshold in thresholds:
	n_labels_threshold = np.sum(cnts>threshold)
	print ("# of labels with frequency above ", threshold, " are: ", n_labels_threshold)




