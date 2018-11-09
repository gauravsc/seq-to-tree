

def mesh_name_to_id_mapping():
	with open('../data/bioasq_dataset/MeSH_name_id_mapping_2017.txt','r') as f:
		mappings = f.readlines()
		mappings = [mapping.strip().split("=") for mapping in mappings]

	mappings_dict = {}
	for mapping in mappings:
		mappings_dict[mapping[0]] = mapping[1]

	return mappings_dict

