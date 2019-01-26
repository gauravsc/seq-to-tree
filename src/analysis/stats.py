import json
import pickle
from collections import defaultdict
import os
import numpy as np
import pickle

# # read the mesh to index files
# with open('../data/mesh_to_idx.pkl', 'rb') as fread:
# 	mesh_to_idx = pickle.load(fread)

# dictionary for keeping label counts
label_cnt = {}
seq_cnt = 0
# getting the training files	
files = os.listdir('../data/bioasq_dataset/train_batches/')
for fi in files:
	print ("File read:", fi)
	with open('../data/bioasq_dataset/train_batches/'+fi, 'r', encoding="utf8", errors='ignore') as f:
		data = json.load(f)
	
	abstracts = data['abs']
	list_of_labels = data['labels']
	seq_cnt += np.sum([len(seq) for seq in data['tgt']])
	n_abstracts = len(abstracts)

	for labels in list_of_labels:
		for label in labels:
			if label in label_cnt:
				label_cnt[label] += 1
			else:
				label_cnt[label] = 1

n_labels = sum(list(label_cnt.values()))

print ("Total # of labels: ", n_labels, "Max # of any one label: ", np.max(list(label_cnt.values())), "# of sequences: ", seq_cnt)

for label in label_cnt.keys():
	label_cnt[label] = label_cnt[label]/float(n_labels)

thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
cnts = np.array(list(label_cnt.values()))

for threshold in thresholds:
	n_labels_threshold = np.sum(cnts>threshold)
	print ("# of labels with frequency above ", threshold, " are: ", n_labels_threshold)

pickle.dump(label_cnt, open('../data/label_cnt.pkl','wb'))

label_cnt_fractions = sorted(np.array(list(label_cnt.values())), reverse=True)
label_cnt_cumsum = np.cumsum(label_cnt_fractions)

pickle.dump(label_cnt_cumsum, open('../data/label_cnt_cumsum.pkl','wb'))

print ("Cumulative sum of label counts:")
for i in label_cnt_cumsum:
	print (i)



