import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_model_name: str, index_path: str = None):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

    def create_index(self, texts):
        logger.info(f"Creating FAISS index for {len(texts)} texts")
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype(np.float32))
        if self.index_path:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query], convert_to_tensor=False).astype(np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        logger.info(f"Vector search returned indices {indices} with distances {distances}")
        return indices[0]

    def add_texts(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=False).astype(np.float32)
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
