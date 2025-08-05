import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_store, stored_chunks):
        self.vector_store = vector_store
        self.stored_chunks = stored_chunks

    def retrieve(self, query, top_k=5):
        indices = self.vector_store.search(query, top_k)
        logger.info(f"Retrieved chunks indices: {indices}")
        relevant_chunks = [self.stored_chunks[i] for i in indices]
        return " ".join(relevant_chunks)
