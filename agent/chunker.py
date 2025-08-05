try:
  from llama_index import Document
  from llama_index.text_splitter import SentenceSplitter
except ImportError:
  from llama_index.core import Document
  from llama_index.core.text_splitter import SentenceSplitter

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def semantic_splitter_chunk_documents(texts, chunk_size=512, chunk_overlap=50):
    """
    Use LlamaIndex's SemanticSplitterNodeParser which adaptively splits
    text based on embedding similarities between sentences.

    Args:
        texts (List[str]): List of raw document texts to chunk.
        chunk_size (int): Target chunk size (approx tokens or sentences).
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List[str]: List of semantically split chunks.
    """

    # Choose an open-source embedding model (must be compatible with LlamaIndex)
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = [Document(text=t) for t in texts]

    # Initialize the SemanticSplitterNodeParser with embedding model
    splitter = SemanticSplitterNodeParser(
        embed_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # Extract chunk texts from nodes
    chunks = [node.get_content() for node in nodes]
    return chunks
