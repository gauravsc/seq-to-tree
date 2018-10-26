
# Fixing the format of word embeddings for the biomedical text, and 
# making it identical to the mesh embedding format

with open('../../data/embeddings/vectors.txt', 'r') as f:
	vectors = f.readlines()
	vectors = [vector.strip() for vector in vectors]

with open('../../data/embeddings/types.txt', 'r') as f:
	words = f.readlines()
	words = [word.strip() for word in words]

fwrite = open('../../data/embeddings/word.processed.embeddings', 'w')
fwrite.write(str(len(words))+" "+str(len(vectors[0].strip().split(" ")))+"\n")
for word, vector in zip(words, vectors):
	fwrite.write(word+" "+vector+"\n")
fwrite.close()
