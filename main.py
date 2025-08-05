import yaml
import logging
from agent.data_fetcher import ResearchPaperFetcher
from agent.chunker import semantic_splitter_chunk_documents
from agent.vector_store import VectorStore
from agent.retriever import Retriever
from agent.agent import ResearchAgent

logging.basicConfig(level=logging.INFO)

def main():
    # Load config file
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Fetch Papers
    fetcher = ResearchPaperFetcher(api_url=config["semantic_scholar_api"])
    papers = fetcher.fetch("machine learning software engineering", limit=10)
    abstracts = [paper.get("abstract", "") for paper in papers if paper.get("abstract")]

    # Chunk abstracts semantically using LlamaIndex + open source embeddings
    all_chunks = semantic_splitter_chunk_documents(
        abstracts,
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )

    # Vectorize and create FAISS index
    vector_store = VectorStore(config["embedding_model_name"], index_path=config["faiss_index_path"])
    if vector_store.index is None:
        vector_store.create_index(all_chunks)

    # Setup retriever and agent
    retriever = Retriever(vector_store, all_chunks)
    agent = ResearchAgent(retriever, ollama_base_url=config["ollama_server_url"])

    # Interactive query loop
    print("Enter your research query ('exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() == 'exit':
            break
        answer = agent.answer(query)
        print(f"Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
