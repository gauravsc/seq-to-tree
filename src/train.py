import sys
sys.path.append('/Users/gauravsc/Code/seq-to-tree/src')

import numpy as np
import json 
from utils.mesh_tree_operations import *
from model.transformer.Models import *
from utils.embedding_operations import *
import pickle

# global vparameters
vocab_size = 150000
src_max_seq_len = 1000
tgt_max_seq_len = 20

def get_vocab(data_file):
	data = json.load(open(data_file,'r'))
	records = data['articles']
	
	abstracts = []
	for record in records:
		abstracts.append(record['abstractText'])

	abstracts = [abstract.lower() for abstract in abstracts]
	abstracts = [abstract.split(' ') for abstract in abstracts]

	word_count = {}
	for abstract in abstracts:
		for word in abstract:
			if word in word_count:
				word_count[word] += 1
			else:
				word_count[word] = 0

	vocab = [k for k in sorted(word_count, key=word_count.get, reverse=True)]
	# reduce the size of vocab to max size
	vocab = vocab[:vocab_size]
	# add the unknown token to the vocab
	vocab = ['$PAD$'] + vocab + ['unk']

	word_to_ind = {}
	for i, word in enumerate(vocab):
		word_to_ind[word] = i

	return vocab, word_to_ind


def prepare_train_data(minibatch_data, word_to_idx, mesh_to_idx, ontology_idx_tree, root):
	abstracts = minibatch_data['abs']
	tgts = minibatch_data['tgt']
	masks = minibatch_data['mask']

	src_seq_list = []
	src_pos_list = []
	tgt_seq_list = []
	tgt_pos_list = []
	mask_mat_list = []


	for i, abst in enumerate(abstracts):
		word_seq = abst.lower().strip().split(' ')
		idx_seq = [word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in word_seq]
		src_seq = np.zeros(src_max_seq_len, dtype=int)
		src_seq[:len(idx_seq)] = idx_seq
		src_pos = np.zeros(src_max_seq_len, dtype=int)
		src_pos[:len(idx_seq)] = range(1, len(idx_seq)+1)

		for j, tgt in enumerate(tgts[i]):
			tgt_seq = np.zeros(tgt_max_seq_len, dtype=int)
			tgt_seq[:len(tgt)] = tgt
			tgt_pos = np.zeros(tgt_max_seq_len, dtype=int)
			tgt_pos[:len(tgt)] = range(1, len(tgt)+1)
			
			# list of tgt sequences 
			tgt_seq_list.append(tgt_seq)
			tgt_pos_list.append(tgt_pos)

			# list of src sequences
			src_seq_list.append(src_seq)
			src_pos_list.append(src_pos)

			mask = masks[i][j]
			mask_mat = np.zeros((len(mesh_to_idx), tgt_max_seq_len))
			
			# create the masking matrix that only computes loss on legitimate children
			mask_mat[root, 0] = 1
			for k in range(1, len(tgt)):
				child_nodes = list(ontology_idx_tree.successors(tgt[k-1]))
				mask_mat[child_nodes, k] = 1

			# mask the active sibling nodes
			for k in range(1, len(tgt)):
				if len(mask[k]) > 0:
					mask_mat[mask[k], k] = 0

			
			# create list of masking matrices	
			mask_mat_list.append(mask_mat)

	src_seq = np.vstack(src_seq_list)
	src_pos = np.vstack(src_pos_list)
	tgt_seq = np.vstack(tgt_seq_list)
	tgt_pos = np.vstack(tgt_pos_list)


	print (src_seq.shape, src_pos.shape, tgt_seq.shape, tgt_pos.shape)
	print (src_seq[0], tgt_seq[0])

	return src_seq, src_pos, tgt_seq, tgt_pos, mask_mat_list




def main():
	
	data_file = '../data/bioasq_dataset/toyMeSH_2017.json'

	# create the vocabulary for the input 
	src_vocab, word_to_idx = get_vocab(data_file)
	print("vocabulary of size: ", len(src_vocab))

	# create the vocabulary of mesh terms
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	mesh_vocab = [" "] * len(mesh_to_idx)
	for mesh, idx in mesh_to_idx.items():
		mesh_vocab[idx] = mesh


	# read source word embedding matrix
	src_emb = read_embeddings('../data/embeddings/word.processed.embeddings', src_vocab)

	# read mesh embedding matrix
	tgt_emb = read_embeddings('../data/embeddings/mesh.processed.embeddings', mesh_vocab)

	# create the ontology tree with nodes as idx of the mesh terms
	ontology_idx_tree = create_idx_ontology_tree()
	root = [n for n,d in ontology_idx_tree.in_degree() if d==0] [0]
	print ("Ontolgy tree created with # Nodes: ", len(ontology_idx_tree.nodes()))



	for i in range(1,3):
		minibatch_data = json.load(open('../data/bioasq_dataset/train_batches/'+str(i)+'.json','r'))
		src_seq, src_pos, tgt_seq, tgt_pos, masks = prepare_train_data(minibatch_data, word_to_idx, mesh_to_idx, ontology_idx_tree, root)



	


	# create the transformer model










if __name__ == '__main__':
	main()