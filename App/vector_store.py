import faiss
import numpy as np

class vectorStore:
    def __init__(self,dim):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []
    
    def add(self,embeddings, chunks):
        self.index.add(np.array(embeddings).astype('float32'))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding).astype('float32')
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1) # Reshape to 2D for faiss

        distance, indices = self.index.search(query_embedding, k)
        return [self.text_chunks[i] for i in indices[0]]
