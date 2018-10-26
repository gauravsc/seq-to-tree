import pickle

with open('../../data/mesh_to_idx.pkl', 'rb') as fread:
	mesh_to_idx = pickle.load(fread)

idx_to_mesh = [""] * len(mesh_to_idx)

for mesh, idx in mesh_to_idx.items():
	idx_to_mesh[idx] = mesh


with open('../../data/embeddings/mesh.embeddings', 'r') as fread:
	embeds = fread.readlines()
	embeds = embeds[1:]
	

mesh_to_embed = {}
for embed_str in embeds:
	embed_list = embed_str.split(" ")
	idx = int(embed_list[0])
	embed = list(map(float, embed_list[1:]))
	mesh_to_embed[idx_to_mesh[idx]] = embed
	embed_dim = len(embed)

fwrite = open('../../data/embeddings/mesh.processed.embeddings', 'w')
# write the count and dim of embeddings
fwrite.write(str(len(mesh_to_embed))+ " "+str(embed_dim) +"\n")
for mesh, embed in mesh_to_embed.items():
	obj_to_write = [mesh] + list(map(str, embed))
	fwrite.write(" ".join(obj_to_write)+"\n")
fwrite.close()
