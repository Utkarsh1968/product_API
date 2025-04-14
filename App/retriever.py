def retrieve_chunks(question, embedder, vector_store, k=3):
    question_embedding = embedder.embed([question])[0]
    return vector_store.search(question_embedding, k)
