import sys
import os
import os.path
import random as rd
import numpy as np
import json 
from utils.mesh_tree_operations import *
from model.transformer.Models import *
from utils.embedding_operations import *
import pickle
import torch
import torch.nn as nn
from eval.eval import * 

# global vparameters
vocab_size = 150000
src_max_seq_len = 1000
tgt_max_seq_len = 20
learning_rate = 0.005
threshold = 0.0
n_train_iterations = 1400
save_model = True
load_model = True

def get_vocab(data_file):

	if os.path.isfile('../data/english_vocab.pkl'):
		vocab, word_to_ind = pickle.load(open('../data/english_vocab.pkl','r'))
		return vocab, word_to_ind

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

	pickle.dump((vocab, word_to_ind), open('../data/english_vocab.pkl','w'))

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
			tgt_pos[:len(tgt)+1] = range(1, len(tgt)+1+1)
			
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
			for k in range(1, len(tgt)+1):
				child_nodes = list(ontology_idx_tree.successors(tgt[k-1]))
				
				# positive + negative sampling 
				if k < len(tgt):
					child_nodes = [tgt[k]]+rd.sample(child_nodes, min(5, len(child_nodes)))

				mask_mat[child_nodes, k] = 1.0

			# mask the active sibling nodes
			for k in range(1, len(tgt)):
				if len(mask[k]) > 0:
					mask_mat[mask[k], k] = 0.0

			
			# create list of masking matrices	
			mask_mat_list.append(mask_mat)

	src_seq = np.vstack(src_seq_list)
	src_pos = np.vstack(src_pos_list)
	tgt_seq = np.vstack(tgt_seq_list)
	tgt_pos = np.vstack(tgt_pos_list)
	mask_tensor = np.stack(mask_mat_list, axis=0)
	
	return src_seq, src_pos, tgt_seq, tgt_pos, mask_tensor


def batch_one_hot_encode(vector_list, vocab_size):
	encoded_mat_list = []
	for vector in vector_list:
		encoded_mat = np.zeros((vocab_size, len(vector)))
		encoded_mat[vector, range(len(vector))] = 1.
		encoded_mat_list.append(encoded_mat)

	encoded_tensor = np.stack(encoded_mat_list, axis=0)

	return encoded_tensor


def train(transformer, loss_criterion, optimizer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root):
	# Finally preperation for the training data e.g. source, target and mask
	# Get a list of all the batch files 
	files = os.listdir('../data/bioasq_dataset/train_batches/')
	for i in range(1,n_train_iterations):
		file = rd.choice(files)
		minibatch_data = json.load(open('../data/bioasq_dataset/train_batches/'+file,'r'))
		src_seq, src_pos, tgt_seq, tgt_pos, mask_tensor = prepare_train_data(minibatch_data, word_to_idx, mesh_to_idx, ontology_idx_tree, root)
		src_seq = torch.tensor(src_seq)
		src_pos = torch.tensor(src_pos)
		tgt_seq = torch.tensor(tgt_seq)
		tgt_pos = torch.tensor(tgt_pos)
		mask_tensor = torch.tensor(mask_tensor, dtype=torch.float)
		# print(src_seq.shape, src_pos.shape, tgt_seq.shape, tgt_pos.shape)

		target = torch.tensor(batch_one_hot_encode(tgt_seq, len(mesh_vocab)), dtype=torch.float)

		# shift the sequence of target nodes by one right
		zero_col = torch.ones(tgt_seq.shape[0], 1, dtype=torch.long)
		tgt_seq = torch.cat((zero_col, tgt_seq), dim=1)

		# print (tgt_seq[0, :])
		# print (tgt_pos[0, :])

		output = transformer(src_seq, src_pos, tgt_seq, tgt_pos)

		loss = loss_criterion(output, target)
		loss = loss*mask_tensor
		loss = torch.sum(loss)/torch.sum(mask_tensor)

		print ("loss: ", loss)

		# idx_to_monitor = np.where(mask_tensor[0, :, 3]==1)[0]

		# # print (tgt_seq[0, :])
		# print (target[0, idx_to_monitor, 3])
		# print (output[0, idx_to_monitor, 3])

		# back-propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# output = transformer(src_seq, src_pos, tgt_seq, tgt_pos)
		# print (output[0, idx_to_monitor, 3])


	return transformer


def recursive_decoding(transformer, mesh_idx_seq, src_seq, src_pos, ontology_idx_tree):
	# obtain the next state based on the previous state of the sequence
	tgt_seq = np.zeros(tgt_max_seq_len+1, dtype=int)
	tgt_seq[:len(mesh_idx_seq)] = mesh_idx_seq
	tgt_pos = np.zeros(tgt_max_seq_len, dtype=int)
	tgt_pos[:len(mesh_idx_seq)] = range(1, len(mesh_idx_seq)+1)

	tgt_seq = torch.tensor(tgt_seq.reshape((1,-1)))
	tgt_pos = torch.tensor(tgt_pos.reshape((1,-1)))

	# print (mesh_idx_seq)

	# # shift the sequence of target nodes by one right
	# shift_col = torch.ones(tgt_seq.shape[0], 1, dtype=torch.long)
	# tgt_seq = torch.cat((shift_col, tgt_seq), dim=1)

	# print (tgt_seq.shape, tgt_seq)
	# print (tgt_pos.shape, tgt_pos)

	# get output of the model 
	output = transformer(src_seq, src_pos, tgt_seq, tgt_pos)

	# get the activated children of the current node
	legit_children = list(ontology_idx_tree.successors(mesh_idx_seq[-1]))
	act_nodes = output[0, legit_children, len(mesh_idx_seq)-1]
	print("active nodes: ", act_nodes)
	nxt_nodes = [legit_children[i] for i in range(len(legit_children))  if act_nodes[i] > threshold]
	
	if len(nxt_nodes) == 0:
		return [mesh_idx_seq[-1]]

	pred_mesh_idx = []
	for node in nxt_nodes:
		pred_mesh_idx += recursive_decoding(transformer, mesh_idx_seq+[node], src_seq, src_pos, ontology_idx_tree)

	return pred_mesh_idx


def predict(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root, test_data):
	test_data = test_data['documents']
	
	# iterate over one document at a time
	pred_mesh_idx = []
	for doc in test_data:
		word_seq = doc['abstractText'].lower().strip().split(' ')
		word_idx_seq = [word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in word_seq]
		src_seq = np.zeros(src_max_seq_len, dtype=int)
		src_seq[:len(word_idx_seq)] = word_idx_seq
		src_pos = np.zeros(src_max_seq_len, dtype=int)
		src_pos[:len(word_idx_seq)] = range(1, len(word_idx_seq)+1)

		# reshape to create batch of size 1
		src_seq = torch.tensor(src_seq.reshape((1,-1)))
		src_pos = torch.tensor(src_pos.reshape((1,-1)))

		mesh_idx_seq = [1, root]
		pred_mesh_idx.append(recursive_decoding(transformer, mesh_idx_seq, src_seq, src_pos, ontology_idx_tree))

	# print(pred_mesh_idx)

	return pred_mesh_idx


def main():
	
	# location of the toy dataset ---> this needs to be replaced with the final dataset
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


	# # read source word embedding matrix
	# src_emb = read_embeddings('../data/embeddings/word.processed.embeddings', src_vocab)

	# # read mesh embedding matrix
	# tgt_emb = read_embeddings('../data/embeddings/mesh.processed.embeddings', mesh_vocab)

	# create the ontology tree with nodes as idx of the mesh terms
	ontology_idx_tree = create_idx_ontology_tree()
	root = [n for n,d in ontology_idx_tree.in_degree() if d==0] [0]
	print ("Ontolgy tree created with # Nodes: ", len(ontology_idx_tree.nodes()))


	# create the model, criterior, optimizer etc
	transformer = Transformer(len(src_vocab), len(mesh_vocab), src_max_seq_len, tgt_max_seq_len, d_word_vec=128, d_model=128, d_inner=256,
            n_layers=2, n_head=8, d_k=128, d_v=128, dropout=0.1)
	# mseloss = nn.MSELoss(reduction="none")
	# loss_criterion= nn.BCELoss(reduction="none")
	loss_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
	optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005)

	# train the model 
	transformer = train(transformer, loss_criterion, optimizer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root)
	
	# save the learned model
	if save_model:
		torch.save(transformer.state_dict(), '../saved_models/model.pt')


	# load the saved model
	if load_model:
		transformer.load_state_dict(torch.load('../saved_models/model.pt'))
	
	# validate the model
	val_data_files = ['../data/bioasq_dataset/val_batches/1.json', '../data/bioasq_dataset/val_batches/2.json']
	transformer = transformer.eval()
	pred_mesh_idx = []
	true_mesh_idx = []
	for data_file in val_data_files:
		data = json.load(open(data_file,'r'))
		abstracts = data['abs']
		tgts = data['tgt']
		val_data = {"documents":[]}
		for i in range(len(abstracts)):
			abstract_text = abstracts[i]
			mesh_idxs = [seq[-1] for seq in tgts[i]] 
			val_data['documents'].append({'abstractText': abstract_text})
			true_mesh_idx.append(mesh_idxs)

		pred_mesh_idx += predict(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root, val_data)


	pred_mesh_idx = [list(set(pred_labels))  for pred_labels in pred_mesh_idx]
	print (true_mesh_idx)
	print (pred_mesh_idx)

	f1_scores_list = []
	print ("\n Evaluation on the dev set")
	for true_labels, pred_labels in zip(true_mesh_idx, pred_mesh_idx):
		f1_scores_list.append(f1_score(true_labels, pred_labels))


	print ("f1 score: ", np.mean(f1_scores_list))



	# # test the model
	# test_data_files = ['../data/bioasq_dataset/Task5a-Batch1-Week1_raw.json']
	# transformer = transformer.eval()
	# for data_file in test_data_files:
	# 	pred_mesh_terms = predict(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root, data_file)



	





if __name__ == '__main__':
	main()