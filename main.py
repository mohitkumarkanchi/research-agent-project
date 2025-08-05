import os
import logging
import yaml
import pandas as pd
from agent.data_fetcher import ResearchPaperFetcher
from agent.chunker import semantic_splitter_chunk_documents  # or your chosen chunker
from agent.vector_store import VectorStore
from agent.retriever import Retriever
from agent.agent import ResearchAgent
from data_downloader import ResearchPaperDownloader

logging.basicConfig(level=logging.INFO)

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    downloader = ResearchPaperDownloader(api_url=config["semantic_scholar_api"])
    cached_path = config.get("cached_papers_path", "data/papers.parquet")

    # Load cached papers if exist
    df = downloader.load(path=cached_path)

    if df is None:
        # If no cache, fetch and save
        logging.info("No cached data found. Downloading papers from API...")
        df = downloader.fetch("machine learning software engineering", limit=100)
        downloader.save(df, path=cached_path)

    # Extract abstracts from df, drop empty abstracts
    abstracts = df["abstract"].dropna().tolist()

    # Chunk the abstracts using your semantic chunker
    all_chunks = semantic_splitter_chunk_documents(abstracts, chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])

    # Build or load vector store index
    vector_store = VectorStore(config["embedding_model_name"], index_path=config["faiss_index_path"])

    if vector_store.index is None:
        vector_store.create_index(all_chunks)

    retriever = Retriever(vector_store, all_chunks)
    agent = ResearchAgent(retriever, ollama_base_url=config["ollama_server_url"])

    print("Enter your research query ('exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break
        answer = agent.answer(query)
        print(f"Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
