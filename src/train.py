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
import torch.nn.functional as F
from eval.eval import * 

# global vparameters
vocab_size = 250000
src_max_seq_len = 1000
tgt_max_seq_len = 20
learning_rate = 0.005
threshold = 0.5
n_train_iterations = 1400
save_model = True
load_model = True
train_model = True
batch_size = 240
clip_norm = 5.0
max_batch_size = 16
in_device = torch.device("cuda:0")
out_device = torch.device("cuda:0")
n_epochs = 100
# set random seed
rd.seed(9001)

def get_vocab(data_file):

	if os.path.isfile('../data/english_vocab.pkl'):
		vocab, word_to_ind = pickle.load(open('../data/english_vocab.pkl','rb'))
		return vocab, word_to_ind

	with open('../data/bioasq_dataset/allMeSH_2017.json', 'r', encoding="utf8", errors='ignore') as f:
		records = json.load(f)['articles']
	
	word_count = {}
	for record in records:
		words_in_abstract = record['abstractText'].lower().split(' ')
		for word in words_in_abstract:
			if word in word_count:
				word_count[word] += 1
			else:
				word_count[word] = 1

	vocab = [k for k in sorted(word_count, key=word_count.get, reverse=True)]
	
	# reduce the size of vocab to max size
	vocab = vocab[:vocab_size]

	# add the unknown token to the vocab
	vocab = ['$PAD$'] + vocab + ['unk']

	word_to_ind = {}
	for i, word in enumerate(vocab):
		word_to_ind[word] = i

	pickle.dump((vocab, word_to_ind), open('../data/english_vocab.pkl','wb'))

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
		
		# only copy till max length
		src_seq[:len(idx_seq)] = idx_seq[:src_max_seq_len]
		src_pos = np.zeros(src_max_seq_len, dtype=int)
		src_pos[: min(src_max_seq_len, len(idx_seq))] = range(1, min(src_max_seq_len, len(idx_seq))+1)

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


def train(transformer, loss_criterion, optimizer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root, files):
	# Final preperation for the training data e.g. source, target and mask
	print ("training on set of files:", files)

	# Get the count of # of files
	files_cnt = len(files)

	for ep in range(n_epochs):
		list_losses = [] 
		for it, file in enumerate(files):

			file_data = json.load(open('../data/bioasq_dataset/train_batches/'+file,'r'))
			file_size = len(file_data['abs'])
			
			i = 0
			while i < file_size:
				minibatch_data = {'abs': file_data['abs'][i:i+batch_size], 
				'tgt': file_data['tgt'][i:i+batch_size], 'mask':file_data['mask'][i:i+batch_size]}
				
				src_seq, src_pos, tgt_seq, tgt_pos, mask_tensor = prepare_train_data(minibatch_data, word_to_idx, mesh_to_idx, ontology_idx_tree, root)
				
				src_seq_full = torch.tensor(src_seq)
				src_pos_full = torch.tensor(src_pos)
				tgt_seq_full = torch.tensor(tgt_seq)
				tgt_pos_full = torch.tensor(tgt_pos)
				mask_tensor_full = torch.tensor(mask_tensor, dtype=torch.float)

				for rs in range(int((batch_size*15)/max_batch_size)):

					# fix the max size of the batch 
					ind = np.random.choice(src_seq_full.shape[0], max_batch_size, replace=False)
					src_seq = src_seq_full[ind]
					src_pos = src_pos_full[ind]
					tgt_seq = tgt_seq_full[ind]
					tgt_pos = tgt_pos_full[ind]
					mask_tensor = mask_tensor_full[ind]

					# print(src_seq.shape, src_pos.shape, tgt_seq.shape, tgt_pos.shape)
					target = torch.tensor(batch_one_hot_encode(tgt_seq, len(mesh_vocab)), dtype=torch.float)

					# shift the sequence of target nodes by one right
					zero_col = torch.ones(tgt_seq.shape[0], 1, dtype=torch.long)
					tgt_seq = torch.cat((zero_col, tgt_seq), dim=1)

					# print (tgt_seq[0, :])
					# print (tgt_pos[0, :])

					# copy tensors to the gpu device where the model is located
					src_seq = src_seq.to(in_device)
					src_pos = src_pos.to(in_device)
					tgt_seq = tgt_seq.to(in_device)
					tgt_pos = tgt_pos.to(in_device)
					target = target.to(out_device)
					mask_tensor = mask_tensor.to(out_device)

					output = transformer(src_seq, src_pos, tgt_seq, tgt_pos)

					loss = loss_criterion(output, target)
					loss = loss*mask_tensor
					loss = torch.sum(loss)/torch.sum(mask_tensor)

					print("loss: ", loss)
					# mask_tensor_copy = mask_tensor.data.cpu().numpy()
					# idx_ = 6
					# idx_to_monitor = np.where(mask_tensor_copy[0, :, idx_]==1)[0]

					# # print (tgt_seq[0, :])
					# print (target[0, idx_to_monitor, idx_])
					# print (output[0, idx_to_monitor, idx_])

					# back-propagation
					optimizer.zero_grad()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(transformer.parameters(), clip_norm)
					optimizer.step()

					list_losses.append(loss.data.cpu().numpy())

					del loss, output, mask_tensor

				i += batch_size

				# output = transformer(src_seq, src_pos, tgt_seq, tgt_pos)
				# print (output[0, idx_to_monitor, idx_])

		print("Epochs: ", str(ep)+"/"+str(n_epochs), "loss: ", np.mean(list_losses))
		

		if ep >= 2:
			# validate the  model
			validate_model(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root)

		# save the learned model
		if save_model:
			torch.save(transformer.state_dict(), '../saved_models/model.pt')

					
				
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

	# copy the tensors to gpu
	tgt_seq = tgt_seq.to(in_device)
	tgt_pos = tgt_pos.to(in_device)

	# get output of the model 
	output = transformer(src_seq, src_pos, tgt_seq, tgt_pos)

	# apply sigmoid to the output
	output = F.sigmoid(output)

	# get the activated children of the current node
	legit_children = list(ontology_idx_tree.successors(mesh_idx_seq[-1]))
	act_nodes = output[0, legit_children, len(mesh_idx_seq)-1]
	# print("active nodes: ", act_nodes)
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
	for i, doc in enumerate(test_data):
		print (str(i)+"/"+str(len(test_data)))
		word_seq = doc['abstractText'].lower().strip().split(' ')
		word_idx_seq = [word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in word_seq]
		# print ("word idx sequence: ", word_idx_seq)
		src_seq = np.zeros(src_max_seq_len, dtype=int)
		src_seq[:len(word_idx_seq)] = word_idx_seq[:src_max_seq_len]
		src_pos = np.zeros(src_max_seq_len, dtype=int)
		src_pos[:min(src_max_seq_len, len(word_idx_seq))] = range(1, min(src_max_seq_len, len(word_idx_seq))+1)

		# reshape to create batch of size 1
		src_seq = torch.tensor(src_seq.reshape((1,-1)))
		src_pos = torch.tensor(src_pos.reshape((1,-1)))

		# copy the tensor to gpu  
		src_seq = src_seq.to(in_device)
		src_pos = src_pos.to(in_device)

		mesh_idx_seq = [1, root]
		pred_for_one_doc = recursive_decoding(transformer, mesh_idx_seq, src_seq, src_pos, ontology_idx_tree)
		pred_mesh_idx.append(pred_for_one_doc)

		print(set(pred_for_one_doc))

	return pred_mesh_idx


def validate_model(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root):
	path = '../data/bioasq_dataset/val_batches/'
	val_data_files = os.listdir(path)
	transformer = transformer.eval()
	pred_mesh_idx = []
	true_mesh_idx = []
	
	for file in val_data_files:
		data = json.load(open(path+file,'r'))
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
	# print (true_mesh_idx)
	# print (pred_mesh_idx)

	f1_scores_list = []
	print ("\n Evaluation on the val set")
	for true_labels, pred_labels in zip(true_mesh_idx, pred_mesh_idx):
		f1_scores_list.append(f1_score(true_labels, pred_labels))

	print ("f1 score: ", np.mean(f1_scores_list))



def main():
	
	# location of the toy dataset ---> this needs to be replaced with the final dataset
	data_file = '../data/bioasq_dataset/allMeSH_2017.json'

	# create the vocabulary for the input 
	src_vocab, word_to_idx = get_vocab(data_file)
	print("vocabulary of size: ", len(src_vocab))

	# create the vocabulary of mesh terms
	with open('../data/mesh_to_idx.pkl', 'rb') as fread:
		mesh_to_idx = pickle.load(fread)
	
	mesh_vocab = [" "] * len(mesh_to_idx)
	for mesh, idx in mesh_to_idx.items():
		mesh_vocab[idx] = mesh

	fi = os.listdir('../data/bioasq_dataset/train_batches/')
	train_fi = rd.sample(fi, 20)

	if os.path.isfile("../data/subsampled_train_fi_names.pkl"):
		train_fi = pickle.load(open("../data/subsampled_train_fi_names.pkl", 'rb'))
	else:
		pickle.dump(train_fi, open("../data/subsampled_train_fi_names.pkl", 'wb'))

	# # read source word embedding matrix
	# src_emb = read_embeddings('../data/embeddings/word.processed.embeddings', src_vocab)

	# # read mesh embedding matrix
	# tgt_emb = read_embeddings('../data/embeddings/mesh.processed.embeddings', mesh_vocab)

	# create the ontology tree with nodes as idx of the mesh terms
	ontology_idx_tree = create_idx_ontology_tree()
	root = [n for n,d in ontology_idx_tree.in_degree() if d==0] [0]
	print ("Ontolgy tree created with # Nodes: ", len(ontology_idx_tree.nodes()))

	# Define values for all the parameters
	d_word_vec = 512 
	d_model = 512
	d_inner = 1024
	n_layers = 2
	n_head = 4
	d_k = 64
	d_v = 64
	dropout = 0.1
	print ("Model constructed with following parameters: ")
	print ("d_word_vec=", d_word_vec, "d_model=", d_model, "d_inner=", d_inner, "n_layers=", n_layers,
		"n_heads=", n_head, "d_k=", d_k, "d_v=", d_v)
	
	# create the model, criterior, optimizer etc
	transformer = Transformer(len(src_vocab), len(mesh_vocab), src_max_seq_len, tgt_max_seq_len, d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=0.1)

	if torch.cuda.device_count() > 1:
		transformer = nn.DataParallel(transformer, output_device=out_device)

	transformer.to(in_device)

	# mseloss = nn.MSELoss(reduction="none")
	# loss_criterion= nn.BCELoss(reduction="none")
	loss_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
	optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9,0.999))
	# optimizer = torch.optim.SGD(transformer.parameters(), lr=learning_rate)
	# load the saved model
	if load_model and os.path.isfile('../saved_models/model.pt'):
		transformer.load_state_dict(torch.load('../saved_models/model.pt'))
		print ("Done loading the saved model .....")

	if train_model:
		# train the model 
		transformer = transformer.train()
		transformer = train(transformer, loss_criterion, optimizer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root, train_fi)
	
	# # save the learned model
	# if save_model:
	# 	torch.save(transformer.state_dict(), '../saved_models/model.pt')

	
	# validate the model
	validate_model(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root)

	# prepare the test submission
	path = '../data/bioasq_dataset/test-batches-task-5A/'
	test_data_files = os.listdir(path)
	transformer = transformer.eval()
	for file in test_data_files:
		test_data = json.load(open(path+file, 'r', encoding="utf8", errors='ignore'))
		pred_mesh_terms = predict(transformer, ontology_idx_tree, mesh_vocab, word_to_idx, mesh_to_idx, root, test_data)
		
		pred_mesh_indicators = []
		for mesh_idx_list in pred_mesh_terms:
			pred_mesh_indicators.append([mesh_vocab[idx] for idx in mesh_idx_list])

		pmid_list = [doc['pmid'] for doc in test_data['documents']]

		# create test file for submission
		out_file_name = file.split(".")[0] + "_output.json"	
		f_write = open(path+out_file_name, 'w')

		out_data = []
		for i in range(len(pmid_list)):
			out_data.append({'pmid':pmid[i], "labels": pred_mesh_indicators[i]})
		out_data = {"documents": out_data}

		json.dump(out_data, f_write)




if __name__ == '__main__':
	main()